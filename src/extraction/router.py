"""Extraction router: tries ODL first, falls back to PyMuPDF, then pymupdf-layout.

Architecture:
    Primary:   OpenDataLoaderExtractor  (ODL + Java 11, best table/bbox quality)
    Secondary: PyMuPDFExtractor         (always available, fast, digital-born PDFs)
    Tertiary:  PyMuPDFLayoutExtractor   (CPU GNN, complex multi-column layouts)

Fallback logic:
    confidence >= threshold  ->  pages passed downstream as-is
    confidence <  threshold  ->  page re-extracted with next extractor in chain

If ODL is not installed or Java is unavailable, the router silently
degrades to PyMuPDF-only mode, preserving full backwards compatibility.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
        extractor_name: Name of the extractor that produced this page.
    """

    page: Page
    confidence: float
    used_fallback: bool = False
    extractor_name: str = "unknown"


class ExtractionRouter:
    """Routes each page to the best available extractor.

    Primary:   OpenDataLoader (ODL) — full bbox + table support, Java required
    Secondary: PyMuPDF        — fast, always available, digital-born PDFs
    Tertiary:  PyMuPDF-layout — CPU GNN for complex multi-column layouts

    Pages with per-page confidence below `threshold` are automatically
    re-extracted using the next extractor in the chain.

    Args:
        threshold: Confidence below which fallback is triggered.
    """

    def __init__(self, threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        self._threshold = threshold
        self._primary: BaseExtractor
        self._primary_name: str
        self._primary, self._primary_name = self._load_primary()
        self._secondary: BaseExtractor = PyMuPDFExtractor()
        self._tertiary: BaseExtractor | None = self._load_tertiary()

    def _load_primary(self) -> tuple[BaseExtractor, str]:
        """Attempt to load OpenDataLoader as primary extractor.

        Falls back to PyMuPDF if ODL or Java is unavailable.

        Returns:
            Tuple of (extractor instance, extractor name).
        """
        try:
            from src.extraction.opendataloader_extractor import OpenDataLoaderExtractor  # noqa: PLC0415
            logger.info("OpenDataLoader primary extractor available")
            return OpenDataLoaderExtractor(), "OpenDataLoader"
        except (ImportError, Exception) as exc:
            logger.warning(
                "OpenDataLoader unavailable (%s); falling back to PyMuPDF as primary",
                exc,
            )
            return PyMuPDFExtractor(), "PyMuPDF"

    def _load_tertiary(self) -> BaseExtractor | None:
        """Attempt to load pymupdf-layout as tertiary extractor.

        Returns:
            Tertiary extractor instance, or None if unavailable.
        """
        try:
            from src.extraction.layout_extractor import PyMuPDFLayoutExtractor  # noqa: PLC0415
            logger.info("pymupdf-layout tertiary extractor available")
            return PyMuPDFLayoutExtractor()
        except ImportError:
            logger.warning("pymupdf-layout not installed; tertiary extractor disabled")
            return None

    def extract(self, pdf_path: Path) -> list[RoutedPage]:
        """Extract all pages with automatic fallback on low-confidence pages.

        Tries primary extractor first. Pages below confidence threshold are
        re-extracted with secondary, then tertiary if still below threshold.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of RoutedPage objects with confidence scores and extractor
            name attached.

        Raises:
            ExtractionError: If all extractors fail completely.
        """
        # Attempt primary extraction
        primary_pages: list[Page] | None = None
        try:
            primary_pages = self._primary.extract(pdf_path)
        except ExtractionError as exc:
            logger.warning(
                "Primary extractor (%s) failed: %s — falling back to PyMuPDF",
                self._primary_name,
                exc,
            )

        if primary_pages is None:
            primary_pages = self._secondary.extract(pdf_path)

        routed: list[RoutedPage] = []

        for page in primary_pages:
            conf = self._page_confidence(page)
            extractor_used = self._primary_name

            if conf < self._threshold:
                logger.info(
                    "Page %d confidence %.2f < %.2f — trying secondary extractor",
                    page.page_number,
                    conf,
                    self._threshold,
                )
                # Try secondary (PyMuPDF)
                secondary_pages = self._secondary.extract_safe(pdf_path)
                if secondary_pages and page.page_number <= len(secondary_pages):
                    sec_page = secondary_pages[page.page_number - 1]
                    sec_conf = self._page_confidence(sec_page)
                    if sec_conf > conf:
                        page, conf = sec_page, sec_conf
                        extractor_used = "PyMuPDF"

                # Try tertiary (pymupdf-layout) if still below threshold
                if conf < self._threshold and self._tertiary is not None:
                    logger.info(
                        "Page %d still low confidence %.2f — trying tertiary extractor",
                        page.page_number,
                        conf,
                    )
                    tertiary_pages = self._tertiary.extract_safe(pdf_path)
                    if tertiary_pages and page.page_number <= len(tertiary_pages):
                        ter_page = tertiary_pages[page.page_number - 1]
                        ter_conf = self._page_confidence(ter_page)
                        if ter_conf > conf:
                            page, conf = ter_page, ter_conf
                            extractor_used = "PyMuPDFLayout"

            routed.append(RoutedPage(
                page=page,
                confidence=conf,
                used_fallback=extractor_used != self._primary_name,
                extractor_name=extractor_used,
            ))

        logger.info(
            "Extraction complete: %d pages | extractors used: %s",
            len(routed),
            {r.extractor_name for r in routed},
        )
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
        if len(page.elements) < _MIN_ELEMENTS_PER_PAGE:
            mean *= 0.5
        return round(mean, 4)
