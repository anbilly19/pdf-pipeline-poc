"""pymupdf-layout based extractor — CPU GNN for complex multi-column layouts.

Used automatically by ExtractionRouter when page confidence is below threshold.
Requires: pymupdf-layout>=1.27.2.3
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.extraction.base import BaseExtractor, ExtractionError
from src.models import Element, Page

logger = logging.getLogger(__name__)

_CONFIDENCE = 0.88  # layout GNN is generally accurate but not perfect


class PyMuPDFLayoutExtractor(BaseExtractor):
    """Extracts text with layout-aware reading order via pymupdf-layout.

    Handles multi-column editorial spreads, tables in-line with text,
    and German compound noun-heavy bodies better than the base extractor.
    """

    def extract(self, pdf_path: Path) -> list[Page]:
        """Extract pages using pymupdf-layout GNN engine.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of Page objects with layout-aware element ordering.

        Raises:
            ExtractionError: If extraction fails.
        """
        if not pdf_path.exists():
            raise ExtractionError(f"PDF not found: {pdf_path}")

        try:
            import pymupdf_layout  # type: ignore[import-untyped]  # noqa: PLC0415
        except ImportError as exc:
            raise ExtractionError(
                "pymupdf-layout is not installed. Run: uv sync"
            ) from exc

        try:
            result = pymupdf_layout.extract(str(pdf_path))
        except Exception as exc:
            raise ExtractionError(f"pymupdf-layout failed on {pdf_path}") from exc

        pages: list[Page] = []
        for page_data in result.pages:
            elements: list[Element] = []
            for block in page_data.blocks:
                text = block.text.strip()
                if not text:
                    continue
                elements.append(
                    Element(
                        type="text",
                        text=text,
                        bbox=list(block.bbox),
                        confidence=_CONFIDENCE,
                    )
                )
            pages.append(
                Page(
                    page_number=page_data.page_number,
                    image_path="",
                    elements=elements,
                )
            )
            logger.debug(
                "Layout page %d: %d elements", page_data.page_number, len(elements)
            )

        return pages
