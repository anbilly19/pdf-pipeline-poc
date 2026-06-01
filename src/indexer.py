"""Document indexer: runs the full pipeline and stores chunks in Chroma.

This is the write path. The read path (retrieval) goes through BBoxRetriever.
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.pipeline import PDFPipeline, PipelineConfig
from src.retrieval.embedder import ChunkEmbedder
from src.retrieval.store import ChromaStore

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Indexes a PDF: extracts, chunks, embeds, and stores.

    Args:
        pipeline_config: Configuration for extraction and chunking.
        embedder: Embedder instance.
        store: Chroma store instance.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig | None = None,
        embedder: ChunkEmbedder | None = None,
        store: ChromaStore | None = None,
    ) -> None:
        self._pipeline = PDFPipeline(config=pipeline_config)
        self._embedder = embedder or ChunkEmbedder()
        self._store = store or ChromaStore()

    def index(self, pdf_path: Path, doc_id: str | None = None) -> int:
        """Index a PDF document end-to-end.

        Args:
            pdf_path: Path to the PDF.
            doc_id: Identifier for this document (defaults to filename stem).

        Returns:
            Number of chunks indexed.
        """
        effective_id = doc_id or pdf_path.stem
        logger.info("Indexing document: %s (id=%s)", pdf_path, effective_id)

        chunks = self._pipeline.run(pdf_path)
        if not chunks:
            logger.warning("No chunks produced for %s", pdf_path)
            return 0

        embeddings = self._embedder.embed_chunks(chunks)
        self._store.add_chunks(chunks, embeddings, doc_id=effective_id)

        logger.info("Indexed %d chunks for '%s'", len(chunks), effective_id)
        return len(chunks)
