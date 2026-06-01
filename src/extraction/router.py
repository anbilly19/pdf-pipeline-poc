"""Extraction router: runs PyMuPDF, scores confidence, flags low-confidence pages.

Architecture (matches the research plan):
    confidence >= threshold  ->  pages passed downstream as-is
    confidence <  threshold  ->  pages flagged for re-extraction (pymupdf-layout)

pdfmux is called as a subprocess / library when available; falls back to
PyMuPDF if pdfmux is not installed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.extraction.base import BaseExtractor, ExtractionError
from src.extraction.pymupdf_extractor import PyMuPDFExtractor
from src.models import Page

logger = logging.getLogger(__name__)

_DEFAULT_CONFIDENCE_THRESHOLD = 0.85
_MIN_ELEMENTS_PER_PAGE = 1


@dataclass
class RoutedPage:
    """A page with its routing decision attached.

    Args:
        page: The extracted page.
        confidence: Aggregate confidence for this page.
        used_fallback: True if a fallback extractor was used.
    """

    page: Page
    confidence: float
    used_fallback: bool = False


class ExtractionRouter:
    """Routes each page to the best available extractor.

    Primary:  PyMuPDF (fast, always available)
    Fallback: pymupdf-layout (CPU GNN, for complex layouts)

    Pages with per-page confidence below `threshold` are automatically
    re-extracted using the fallback engine.

    Args:
        threshold: Confidence below which fallback is triggered.
    """

    def __init__(self, threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        self._threshold = threshold
        self._primary: BaseExtractor = PyMuPDFExtractor()
        self._fallback: BaseExtractor | None = self._load_fallback()

    def _load_fallback(self) -> BaseExtractor | None:
        """Attempt to load pymupdf-layout as fallback extractor.

        Returns:
            Fallback extractor instance, or None if unavailable.
        """
        try:
            from src.extraction.layout_extractor import PyMuPDFLayoutExtractor  # noqa: PLC0415
            logger.info("pymupdf-layout fallback available")
            return PyMuPDFLayoutExtractor()
        except ImportError:
            logger.warning("pymupdf-layout not installed; fallback disabled")
            return None

    def extract(self, pdf_path: Path) -> list[RoutedPage]:
        """Extract all pages with automatic fallback on low-confidence pages.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of RoutedPage objects with confidence scores attached.

        Raises:
            ExtractionError: If primary extraction fails completely.
        """
        primary_pages = self._primary.extract(pdf_path)
        routed: list[RoutedPage] = []

        for page in primary_pages:
            conf = self._page_confidence(page)

            if conf >= self._threshold or self._fallback is None:
                routed.append(RoutedPage(page=page, confidence=conf))
                continue

            logger.info(
                "Page %d confidence %.2f < %.2f — routing to fallback",
                page.page_number,
                conf,
                self._threshold,
            )
            fallback_pages = self._fallback.extract_safe(pdf_path)
            if fallback_pages and page.page_number <= len(fallback_pages):
                fb_page = fallback_pages[page.page_number - 1]
                fb_conf = self._page_confidence(fb_page)
                routed.append(RoutedPage(page=fb_page, confidence=fb_conf, used_fallback=True))
            else:
                # fallback also failed — keep primary result
                routed.append(RoutedPage(page=page, confidence=conf))

        return routed

    @staticmethod
    def _page_confidence(page: Page) -> float:
        """Compute aggregate confidence for a page.

        Uses the mean element confidence, penalised if the page has
        very few elements (likely scanned or image-heavy).

        Args:
            page: The page to score.

        Returns:
            Confidence in [0.0, 1.0].
        """
        if not page.elements:
            return 0.0
        mean = sum(e.confidence for e in page.elements) / len(page.elements)
        # penalise sparse pages (likely image-heavy ads)
        if len(page.elements) < _MIN_ELEMENTS_PER_PAGE:
            mean *= 0.5
        return round(mean, 4)
