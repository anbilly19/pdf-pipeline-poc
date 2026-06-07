"""Hybrid retriever: FAISS -> BM25 -> optional cross-encoder rerank.

Pipeline:
    query
        -> FAISS top-(top_k * faiss_multiplier) candidates by cosine similarity
        -> BM25 rerank over those candidates
        -> OllamaReranker cross-encoder pass (if reranker is configured)
        -> top_k returned to LLM

BM25 fixes the core problem where nomic-embed-text ranks German legal
keywords poorly: BM25 is exact-term based and always surfaces chunks
that literally contain the query words (e.g. "Reaktionszeit", "Vertragsstrafe").

Cross-encoder rerank (roadmap #1)
-----------------------------------
Pass an OllamaReranker instance to BBoxRetriever.__init__ to enable the
third reranking pass.  Falls back to BM25 order silently if Ollama is
unavailable so the pipeline is never blocked.

Fallback chain:
  FAISS order -> BM25 order -> cross-encoder order (best available wins)

Dependencies: rank-bm25 (pure Python, no native libs)
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

_FAISS_MULTIPLIER = 4  # fetch 4x candidates from FAISS before BM25 rerank


def _tokenize(text: str) -> list[str]:
    """Whitespace + punctuation tokenizer for BM25 with full German character support.

    Includes:
    - Basic Latin extended (\u00c0-\u024f) for umlauts (\u00e4, \u00f6, \u00fc, \u00c4, \u00d6, \u00dc)
    - \u00df (\u00df) explicitly included
    - All matched tokens lowercased
    """
    return re.findall(r"[\w\u00c0-\u024f\u00df]+", text.lower())


class BBoxRetriever:
    """Hybrid FAISS+BM25+reranker retriever preserving full bbox metadata.

    Args:
        store: Initialised FAISSStore.
        embedder: Initialised ChunkEmbedder.
        top_k: Final number of results returned to the caller.
        faiss_multiplier: How many times top_k to fetch from FAISS before reranking.
        reranker: Optional OllamaReranker for cross-encoder second pass.
                  When None, pipeline stops after BM25.
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

        Pipeline: FAISS candidates -> BM25 rerank -> optional cross-encoder rerank.

        Args:
            query: Natural language question.
            top_k: Override default result count.
            filter_doc_id: Restrict search to a single document.
            filter_chunk_type: Restrict to 'text', 'table', or 'figure'.

        Returns:
            List of Chunk objects ordered by hybrid relevance (best first).
            Falls back to FAISS order when BM25 yields no keyword overlap.
            Falls back to BM25 order when cross-encoder is unavailable.
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

        # Pass 1: BM25 keyword rerank
        bm25_ranked = self._bm25_rerank(query, candidates, top_k=k)

        # Pass 2: cross-encoder rerank (optional)
        if self._reranker is not None:
            final = self._reranker.rerank(query, bm25_ranked, top_k=k)
        else:
            final = bm25_ranked

        logger.info(
            "Retrieve: %d FAISS -> %d BM25 -> %d final (reranker=%s, query: %.60s)",
            len(candidates),
            len(bm25_ranked),
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
    def _bm25_rerank(query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        """Rerank chunks using BM25 keyword scoring.

        Falls back to original FAISS order if:
        - rank_bm25 is not installed, OR
        - all BM25 scores are zero (no keyword overlap between query and candidates).
          In that case the semantic FAISS ranking is already the best signal.

        Args:
            query: The search query.
            chunks: Candidate chunks from FAISS.
            top_k: Number of chunks to return.

        Returns:
            Reranked list of up to top_k chunks.
        """
        try:
            from rank_bm25 import BM25Okapi  # noqa: PLC0415
        except ImportError:
            logger.warning("rank-bm25 not installed, falling back to FAISS order. Run: uv add rank-bm25")
            return chunks[:top_k]

        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        tokenized_query = _tokenize(query)

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        if max(scores, default=0.0) <= 0.0:
            logger.debug(
                "BM25 all-zero scores for query %.60s \u2014 keeping FAISS order", query
            )
            return chunks[:top_k]

        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_k]]
