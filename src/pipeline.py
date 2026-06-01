"""Top-level pipeline orchestrator for Phase 1.

Wires together: extraction -> rendering -> chunking.
Phase 2 will add: embedding -> vector store -> agent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.chunking.chunker import ChunkerConfig, LayoutChunker
from src.extraction.router import ExtractionRouter, RoutedPage
from src.models import Chunk
from src.rendering.renderer import PageRenderer

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline.

    Args:
        confidence_threshold: Pages below this trigger fallback extraction.
        max_chunk_chars: Maximum chars per chunk.
        render_dpi: PNG render resolution.
        output_dir: Base directory for rendered pages and outputs.
    """

    confidence_threshold: float = 0.85
    max_chunk_chars: int = 1000
    render_dpi: int = 150
    output_dir: Path = Path("outputs")


class PDFPipeline:
    """Orchestrates extraction, rendering, and chunking for a single PDF.

    Args:
        config: Pipeline configuration.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._cfg = config or PipelineConfig()
        self._router = ExtractionRouter(threshold=self._cfg.confidence_threshold)
        self._renderer = PageRenderer(
            output_dir=self._cfg.output_dir / "pages",
            dpi=self._cfg.render_dpi,
        )
        self._chunker = LayoutChunker(
            config=ChunkerConfig(max_chars=self._cfg.max_chunk_chars)
        )

    def run(self, pdf_path: Path) -> list[Chunk]:
        """Run the full Phase 1 pipeline on a PDF.

        Args:
            pdf_path: Path to the input PDF.

        Returns:
            List of Chunks ready for embedding in Phase 2.
        """
        logger.info("Pipeline starting: %s", pdf_path)

        routed_pages: list[RoutedPage] = self._router.extract(pdf_path)
        pages = [rp.page for rp in routed_pages]

        fallback_count = sum(1 for rp in routed_pages if rp.used_fallback)
        if fallback_count:
            logger.info("%d page(s) used fallback extractor", fallback_count)

        pages = self._renderer.render(pdf_path, pages)
        chunks = self._chunker.chunk(pages)

        logger.info(
            "Pipeline complete: %d pages, %d chunks (%.0f%% fallback)",
            len(pages),
            len(chunks),
            100 * fallback_count / max(len(pages), 1),
        )
        return chunks
