"""Hybrid retriever: FAISS -> BM25 -> page-rank decay -> optional cross-encoder rerank.

Pipeline:
    query
        -> FAISS top-(top_k * faiss_multiplier) candidates by cosine similarity
        -> BM25 rerank over those candidates
        -> page-rank decay: score *= 1 / log2(page + 1)
           Weight reduced to 0.10 (from 0.25) so late-page legal clauses
           (Vertragsstrafe, Servicezeiten etc.) are not unfairly demoted.
        -> OllamaReranker cross-encoder pass (if reranker is configured)
        -> top_k returned to LLM
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
_PAGE_DECAY_WEIGHT = 0.10  # reduced from 0.25 — late-page clauses were being demoted


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u00c0-\u024f\u00df]+", text.lower())


def _page_boost(page_number: int) -> float:
    return 1.0 / math.log2(max(page_number, 1) + 1)


class BBoxRetriever:
    """Hybrid FAISS+BM25+page-decay+reranker retriever preserving full bbox metadata."""

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
