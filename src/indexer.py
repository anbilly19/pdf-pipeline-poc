"""Document indexer: runs the full pipeline and stores chunks in FAISS.

Write path. Read path goes through BBoxRetriever.

At index time:
  1. Extract + chunk via PDFPipeline
  2. Embed + store in FAISS
  3. Classify document type -> persist domain_config.json
  4. Build knowledge graph  -> persist graph.json
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.pipeline import PDFPipeline, PipelineConfig
from src.retrieval.embedder import ChunkEmbedder
from src.retrieval.store import FAISSStore
from src.agent.classifier import classify_document
from src.agent.domain_config import DocTypeConfig
from src.graph.builder import build_graph, save_graph

logger = logging.getLogger(__name__)

_GRAPH_FILENAME = "graph.json"


class DocumentIndexer:
    """Indexes a PDF: extracts, chunks, embeds, stores, classifies, and builds graph.

    Args:
        pipeline_config: Configuration for extraction and chunking.
        embedder: Embedder instance.
        store: FAISS store instance.
        llm_provider: Provider for LLM fallback in classifier.
        llm_model: Model for LLM fallback in classifier.
        graph_dir: Directory to persist graph.json alongside FAISS index.
                   Defaults to same directory as the FAISS store.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig | None = None,
        embedder: ChunkEmbedder | None = None,
        store: FAISSStore | None = None,
        llm_provider: str = "ollama",
        llm_model: str = "qwen2.5:3b",
        graph_dir: Path | None = None,
    ) -> None:
        self._pipeline = PDFPipeline(config=pipeline_config)
        self._embedder = embedder or ChunkEmbedder()
        self._store = store or FAISSStore()
        self._llm_provider = llm_provider
        self._llm_model = llm_model
        self._graph_dir = graph_dir
        self._last_doc_type: DocTypeConfig | None = None

    @property
    def last_doc_type(self) -> DocTypeConfig | None:
        """The DocTypeConfig detected during the last index() call."""
        return self._last_doc_type

    def _graph_path(self) -> Path:
        """Resolve path for graph.json."""
        if self._graph_dir:
            return self._graph_dir / _GRAPH_FILENAME
        # Co-locate with FAISS persist dir if available
        persist_dir = getattr(self._store, "_persist_dir", None)
        if persist_dir:
            return Path(persist_dir) / _GRAPH_FILENAME
        return Path("outputs/faiss_index") / _GRAPH_FILENAME

    def index(self, pdf_path: Path, doc_id: str | None = None) -> int:
        """Index a PDF document end-to-end.

        Runs: extraction -> chunking -> embedding -> FAISS -> classify -> graph.

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

        # Classify document type and persist domain_config.json
        self._last_doc_type = classify_document(
            chunks,
            provider=self._llm_provider,
            model=self._llm_model,
            save=True,
        )

        # Build knowledge graph and persist graph.json
        try:
            graph = build_graph(chunks)
            save_graph(graph, self._graph_path())
        except Exception as e:  # noqa: BLE001
            logger.warning("Graph build failed (non-fatal): %s", e)

        logger.info(
            "Indexed %d chunks for '%s' | doc_type=%s",
            len(chunks), effective_id, self._last_doc_type.doc_type,
        )
        return len(chunks)
