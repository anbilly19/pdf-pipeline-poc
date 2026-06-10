"""Top-level pipeline orchestrator (extraction + rendering + chunking)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from src.chunking.chunker import ChunkerConfig, LayoutChunker
from src.extraction.router import ExtractionRouter, RoutedPage
from src.models import Chunk
from src.rendering.renderer import PageRenderer

logger = logging.getLogger(__name__)
_DBG = os.environ.get("DEBUG_PIPELINE", "0") == "1"


@dataclass
class PipelineConfig:
    confidence_threshold: float = 0.85
    max_chunk_chars: int = 1000
    render_dpi: int = 150
    output_dir: Path = field(default_factory=lambda: Path("outputs"))


class PDFPipeline:
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
        logger.info("Pipeline starting: %s", pdf_path)

        routed_pages: list[RoutedPage] = self._router.extract(pdf_path)
        pages = [rp.page for rp in routed_pages]

        fallback_count = sum(1 for rp in routed_pages if rp.used_fallback)
        if fallback_count:
            logger.info("%d page(s) used fallback extractor", fallback_count)

        # --- CHECKPOINT 1: after extraction, before rendering ---
        if _DBG and pages:
            p0 = pages[0]
            e0 = p0.elements[0] if p0.elements else None
            logger.info(
                "[DBG-CP1] page 1 image_path=%r  elements=%d  "
                "first_element_bbox=%s",
                p0.image_path, len(p0.elements),
                e0.bbox if e0 else "NO_ELEMENTS",
            )

        pages = self._renderer.render(pdf_path, pages)

        # --- CHECKPOINT 2: after rendering ---
        if _DBG and pages:
            p0 = pages[0]
            logger.info(
                "[DBG-CP2] page 1 image_path after render=%r",
                p0.image_path,
            )

        chunks = self._chunker.chunk(pages)

        # --- CHECKPOINT 3: after chunking ---
        if _DBG and chunks:
            c0 = chunks[0]
            logger.info(
                "[DBG-CP3] first chunk: page=%d  bboxes=%s  "
                "image_path=%r  text[:80]=%r",
                c0.page_number, c0.bboxes, c0.image_path, c0.text[:80],
            )
            zero_bbox_count = sum(
                1 for c in chunks
                if not c.bboxes or c.bboxes == [[0.0, 0.0, 0.0, 0.0]]
            )
            logger.info(
                "[DBG-CP3] chunks with zero/empty bboxes: %d / %d",
                zero_bbox_count, len(chunks),
            )
            no_image_count = sum(1 for c in chunks if not c.image_path)
            logger.info(
                "[DBG-CP3] chunks with empty image_path: %d / %d",
                no_image_count, len(chunks),
            )

        logger.info(
            "Pipeline complete: %d pages \u2192 %d chunks (%.0f%% via fallback)",
            len(pages), len(chunks),
            100 * fallback_count / max(len(pages), 1),
        )
        return chunks
