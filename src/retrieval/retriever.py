"""High-level retriever: embeds a query and returns sourced Chunks."""
from __future__ import annotations

import logging

from src.models import Chunk, QAResponse, Source
from src.retrieval.embedder import ChunkEmbedder
from src.retrieval.store import ChromaStore

logger = logging.getLogger(__name__)


class BBoxRetriever:
    """Retrieves top-k chunks for a query, preserving full bbox metadata.

    Args:
        store: Initialised ChromaStore.
        embedder: Initialised ChunkEmbedder.
        top_k: Default number of results to retrieve.
    """

    def __init__(
        self,
        store: ChromaStore,
        embedder: ChunkEmbedder,
        top_k: int = 5,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_doc_id: str | None = None,
        filter_chunk_type: str | None = None,
    ) -> list[Chunk]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: Natural language question (German or English).
            top_k: Override default result count.
            filter_doc_id: Restrict search to a single document.
            filter_chunk_type: Restrict to 'text', 'table', or 'figure'.

        Returns:
            List of Chunk objects ordered by relevance.
        """
        k = top_k or self._top_k
        query_vec = self._embedder.embed_query(query)
        chunks = self._store.query(
            query_embedding=query_vec,
            n_results=k,
            filter_doc_id=filter_doc_id,
            filter_chunk_type=filter_chunk_type,
        )
        logger.info("Retrieved %d chunks for query: %.60s...", len(chunks), query)
        return chunks

    def retrieve_as_sources(self, query: str, top_k: int | None = None) -> list[Source]:
        """Retrieve chunks and convert to Source objects for QAResponse.

        Args:
            query: Natural language question.
            top_k: Override default result count.

        Returns:
            List of Source objects with page and bbox information.
        """
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
