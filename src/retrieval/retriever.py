"""Hybrid retriever: FAISS -> BM25 -> page-rank decay -> optional cross-encoder rerank.

Pipeline:
    query
        -> FAISS top-(top_k * faiss_multiplier) candidates by cosine similarity
        -> BM25 rerank over those candidates
        -> page-rank decay: score *= 1 / log2(page + 1)
           Generic boost — earlier pages always rank higher for equal BM25 scores.
           Works across all PDF types (contracts, policies, forms, reports).
        -> OllamaReranker cross-encoder pass (if reranker is configured)
        -> top_k returned to LLM

BM25 fixes the core problem where nomic-embed-text ranks German legal
keywords poorly: BM25 is exact-term based and always surfaces chunks
that literally contain the query words.

Page-rank decay (Option A)
---------------------------
Early pages of any PDF are almost always the most document-specific:
contracts -> Kurzfassung, insurance -> coverage summary, tax -> header.
Applying a smooth log decay rewards early pages without hard-coding
any domain keywords.
"""
from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING

from src.models import Chunk, Source
from src.retrieval.embedder import ChunkEmbedder
from src.retrieval.store import FAISSStore

if TYPE_CHECKING:
    from src.retrieval.reranker import OllamaReranker

logger = logging.getLogger(__name__)

_FAISS_MULTIPLIER = 4
_PAGE_DECAY_WEIGHT = 0.25  # blend: final = (1 - w) * bm25_norm + w * page_boost


def _tokenize(text: str) -> list[str]:
    """Whitespace + punctuation tokenizer for BM25 with full German character support."""
    return re.findall(r"[\w\u00c0-\u024f\u00df]+", text.lower())


def _page_boost(page_number: int) -> float:
    """Return a 0..1 score boost that decays with page number.

    Uses 1 / log2(page + 1) so:
      page 1  -> 1.000
      page 2  -> 0.631
      page 3  -> 0.500
      page 5  -> 0.387
      page 10 -> 0.289
      page 20 -> 0.231
    """
    return 1.0 / math.log2(max(page_number, 1) + 1)


class BBoxRetriever:
    """Hybrid FAISS+BM25+page-decay+reranker retriever preserving full bbox metadata.

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

        Pipeline: FAISS candidates -> BM25 rerank -> page-rank decay blend
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

        bm25_ranked = self._bm25_rerank(query, candidates, top_k=k)

        if self._reranker is not None:
            final = self._reranker.rerank(query, bm25_ranked, top_k=k)
        else:
            final = bm25_ranked

        logger.info(
            "Retrieve: %d FAISS -> %d BM25+decay -> %d final (reranker=%s, query: %.60s)",
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
        """Rerank chunks using BM25 + page-rank decay blend.

        Scoring:
            bm25_norm  = bm25_score / max_bm25_score   (0..1)
            page_score = 1 / log2(page + 1)            (0..1, decays with page)
            final      = (1 - w) * bm25_norm + w * page_score

        Falls back to FAISS order when BM25 yields no keyword overlap.

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
            logger.warning("rank-bm25 not installed — falling back to FAISS order.")
            return chunks[:top_k]

        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        tokenized_query = _tokenize(query)

        bm25 = BM25Okapi(tokenized_corpus)
        raw_scores = list(bm25.get_scores(tokenized_query))

        max_score = max(raw_scores, default=0.0)
        if max_score <= 0.0:
            logger.debug("BM25 all-zero for query %.60s — keeping FAISS order", query)
            return chunks[:top_k]

        w = _PAGE_DECAY_WEIGHT
        blended: list[tuple[float, Chunk]] = []
        for score, chunk in zip(raw_scores, chunks):
            bm25_norm  = score / max_score
            page_score = _page_boost(chunk.page_number)
            final      = (1 - w) * bm25_norm + w * page_score
            blended.append((final, chunk))
            logger.debug(
                "page=%d bm25_norm=%.3f page_boost=%.3f final=%.3f",
                chunk.page_number, bm25_norm, page_score, final,
            )

        ranked = sorted(blended, key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_k]]
