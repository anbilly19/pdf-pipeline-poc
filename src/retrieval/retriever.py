"""Hybrid retriever: FAISS -> BM25 -> optional cross-encoder rerank.

Pipeline:
    query
        -> FAISS top-(top_k * faiss_multiplier) candidates by cosine similarity
        -> BM25 rerank over those candidates
        -> page-range boost: chunks from early pages (Kurzfassung) get a
           score multiplier when the query contains contract-specific keywords
        -> OllamaReranker cross-encoder pass (if reranker is configured)
        -> top_k returned to LLM

BM25 fixes the core problem where nomic-embed-text ranks German legal
keywords poorly: BM25 is exact-term based and always surfaces chunks
that literally contain the query words.

Kurzfassung boost (field-test fix)
------------------------------------
The short contract form (pages 1-2) contains contract-specific clauses
(Verlängerungsoption, Verschwiegenheit, Leistungsumfang) that get buried
by the much longer AGB section. When the query contains any of a set of
Kurzfassung-specific keywords, chunks from pages 1-2 receive a 1.5x BM25
score multiplier before the final ranking.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from src.models import Chunk, Source
from src.retrieval.embedder import ChunkEmbedder
from src.retrieval.store import FAISSStore

if TYPE_CHECKING:
    from src.retrieval.reranker import OllamaReranker

logger = logging.getLogger(__name__)

_FAISS_MULTIPLIER = 4

# Keywords that signal the query is about contract-specific (Kurzfassung) content.
# When any of these appear in the query, pages 1-2 get a score boost.
_KURZFASSUNG_KEYWORDS = {
    "verlängerungsoption", "verlängerung", "verschwiegenheit", "vertragsnummer",
    "leistungsumfang", "honorarvereinbarung", "mitteilungsblatt", "seminarbeginn",
    "bundesakademie", "absage", "teilnehmerzahl", "sonderveranstaltung",
    "reisekosten", "reisezeiten", "zuschlagserteilung",
}
_KURZFASSUNG_MAX_PAGE = 2   # pages 1-2 are the Kurzfassung
_KURZFASSUNG_BOOST = 1.5    # score multiplier for Kurzfassung chunks


def _tokenize(text: str) -> list[str]:
    """Whitespace + punctuation tokenizer for BM25 with full German character support."""
    return re.findall(r"[\w\u00c0-\u024f\u00df]+", text.lower())


def _is_kurzfassung_query(query: str) -> bool:
    """Return True if the query contains any Kurzfassung-specific keywords."""
    tokens = set(_tokenize(query))
    return bool(tokens & _KURZFASSUNG_KEYWORDS)


class BBoxRetriever:
    """Hybrid FAISS+BM25+reranker retriever preserving full bbox metadata.

    Args:
        store: Initialised FAISSStore.
        embedder: Initialised ChunkEmbedder.
        top_k: Final number of results returned to the caller.
        faiss_multiplier: How many times top_k to fetch from FAISS before reranking.
        reranker: Optional OllamaReranker for cross-encoder second pass.
    """

    def __init__(
        self,
        store: FAISSStore,
        embedder: ChunkEmbedder,
        top_k: int = 15,
        faiss_multiplier: int = _FAISS_MULTIPLIER,
        reranker: OllamaReranker | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._top_k = top_k
        self._faiss_multiplier = faiss_multiplier
        self._reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_doc_id: str | None = None,
        filter_chunk_type: str | None = None,
    ) -> list[Chunk]:
        """Retrieve the most relevant chunks using hybrid search.

        Pipeline: FAISS candidates -> BM25 rerank -> Kurzfassung boost
        -> optional cross-encoder rerank.

        Args:
            query: Natural language question.
            top_k: Override default result count.
            filter_doc_id: Restrict search to a single document.
            filter_chunk_type: Restrict to 'text', 'table', or 'figure'.

        Returns:
            List of Chunk objects ordered by hybrid relevance (best first).
        """
        k = top_k or self._top_k
        candidates_k = min(k * self._faiss_multiplier, self._store.count() or 1)

        query_vec = self._embedder.embed_query(query)
        candidates = self._store.query(
            query_embedding=query_vec,
            n_results=candidates_k,
            filter_doc_id=filter_doc_id,
            filter_chunk_type=filter_chunk_type,
        )

        if not candidates:
            return []

        # Pass 1: BM25 keyword rerank with optional Kurzfassung boost
        boost_kurzfassung = _is_kurzfassung_query(query)
        bm25_ranked = self._bm25_rerank(
            query, candidates, top_k=k, boost_kurzfassung=boost_kurzfassung
        )

        # Pass 2: cross-encoder rerank (optional)
        if self._reranker is not None:
            final = self._reranker.rerank(query, bm25_ranked, top_k=k)
        else:
            final = bm25_ranked

        logger.info(
            "Retrieve: %d FAISS -> %d BM25%s -> %d final (reranker=%s, query: %.60s)",
            len(candidates),
            len(bm25_ranked),
            " +Kurzfassung boost" if boost_kurzfassung else "",
            len(final),
            "on" if self._reranker else "off",
            query,
        )
        return final

    def retrieve_as_sources(self, query: str, top_k: int | None = None) -> list[Source]:
        """Retrieve chunks and convert to Source objects."""
        chunks = self.retrieve(query, top_k=top_k)
        return [
            Source(
                text=c.text,
                page=c.page_number,
                bboxes=c.bboxes,
                image=c.image_path,
            )
            for c in chunks
        ]

    @staticmethod
    def _bm25_rerank(
        query: str,
        chunks: list[Chunk],
        top_k: int,
        boost_kurzfassung: bool = False,
    ) -> list[Chunk]:
        """Rerank chunks using BM25 keyword scoring.

        Falls back to original FAISS order if:
        - rank_bm25 is not installed, OR
        - all BM25 scores are zero (no keyword overlap).

        Args:
            query: The search query.
            chunks: Candidate chunks from FAISS.
            top_k: Number of chunks to return.
            boost_kurzfassung: If True, multiply scores for pages 1-2 by
                _KURZFASSUNG_BOOST to surface contract-specific clauses.

        Returns:
            Reranked list of up to top_k chunks.
        """
        try:
            from rank_bm25 import BM25Okapi  # noqa: PLC0415
        except ImportError:
            logger.warning("rank-bm25 not installed — falling back to FAISS order.")
            return chunks[:top_k]

        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        tokenized_query = _tokenize(query)

        bm25 = BM25Okapi(tokenized_corpus)
        scores = list(bm25.get_scores(tokenized_query))

        if max(scores, default=0.0) <= 0.0:
            logger.debug("BM25 all-zero for query %.60s — keeping FAISS order", query)
            return chunks[:top_k]

        # Apply Kurzfassung page boost
        if boost_kurzfassung:
            for i, chunk in enumerate(chunks):
                if chunk.page_number <= _KURZFASSUNG_MAX_PAGE:
                    scores[i] *= _KURZFASSUNG_BOOST
                    logger.debug(
                        "Kurzfassung boost applied to page %d (score %.3f)",
                        chunk.page_number, scores[i],
                    )

        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_k]]
