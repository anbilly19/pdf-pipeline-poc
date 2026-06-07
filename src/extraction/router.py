"""Extraction router: Kreuzberg -> ODL -> PyMuPDF -> PyMuPDF-layout.

Architecture (Roadmap #4: Kreuzberg promoted to primary)
---------------------------------------------------------
    Primary:    KreuzbergExtractor       (Rust-speed, precise bboxes, no Java)
    Secondary:  OpenDataLoaderExtractor  (Java 11 required, best table quality)
    Tertiary:   PyMuPDFExtractor         (always available, digital-born PDFs)
    Quaternary: PyMuPDFLayoutExtractor   (CPU GNN, complex multi-column layouts)

Fallback logic:
    confidence >= threshold  ->  pages passed downstream as-is
    confidence <  threshold  ->  page re-extracted with next extractor in chain

If Kreuzberg is not installed, the router silently degrades to ODL or
PyMuPDF, preserving full backwards compatibility.

Bbox contract: all extractors must return top-left-origin coordinates.
KreuzbergExtractor normalises internally; others already comply.
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
        extractor_name: Name of the extractor that produced this page.
    """

    page: Page
    confidence: float
    used_fallback: bool = False
    extractor_name: str = "unknown"


class ExtractionRouter:
    """Routes each page through the extractor chain, best-first.

    Primary:    Kreuzberg    — Rust, precise bboxes, no Java, no GPU
    Secondary:  ODL          — Java-backed, highest table accuracy
    Tertiary:   PyMuPDF      — always available, fast
    Quaternary: PyMuPDF-layout — CPU GNN for complex layouts

    Each page’s confidence is checked after extraction.  Pages below
    `threshold` are re-extracted by the next extractor in the chain;
    the highest-confidence result is kept.

    Args:
        threshold: Per-page confidence below which the next extractor is tried.
    """

    def __init__(self, threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        self._threshold = threshold
        self._chain: list[tuple[BaseExtractor, str]] = self._build_chain()

    # ------------------------------------------------------------------
    # Chain construction
    # ------------------------------------------------------------------

    def _build_chain(self) -> list[tuple[BaseExtractor, str]]:
        """Build the ordered extractor chain, skipping unavailable engines.

        Returns:
            Ordered list of (extractor, name) tuples.  PyMuPDF is always
            present as the guaranteed fallback.
        """
        chain: list[tuple[BaseExtractor, str]] = []

        # 1. Kreuzberg (primary)
        try:
            from src.extraction.kreuzberg_extractor import KreuzbergExtractor  # noqa: PLC0415
            chain.append((KreuzbergExtractor(), "Kreuzberg"))
            logger.info("Kreuzberg extractor registered as primary")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Kreuzberg unavailable (%s); skipping", exc)

        # 2. OpenDataLoader (secondary)
        try:
            from src.extraction.opendataloader_extractor import OpenDataLoaderExtractor  # noqa: PLC0415
            chain.append((OpenDataLoaderExtractor(), "OpenDataLoader"))
            logger.info("OpenDataLoader extractor registered as secondary")
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenDataLoader unavailable (%s); skipping", exc)

        # 3. PyMuPDF — always available
        chain.append((PyMuPDFExtractor(), "PyMuPDF"))

        # 4. PyMuPDF-layout (quaternary)
        try:
            from src.extraction.layout_extractor import PyMuPDFLayoutExtractor  # noqa: PLC0415
            chain.append((PyMuPDFLayoutExtractor(), "PyMuPDFLayout"))
            logger.info("PyMuPDF-layout extractor registered as quaternary")
        except ImportError:
            logger.warning("pymupdf-layout not installed; quaternary extractor disabled")

        primary_name = chain[0][1] if chain else "PyMuPDF"
        logger.info(
            "Extractor chain: %s",
            " -> ".join(name for _, name in chain),
        )
        return chain

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, pdf_path: Path) -> list[RoutedPage]:
        """Extract all pages with automatic per-page fallback.

        For each page, the primary extractor’s result is used if its
        confidence meets the threshold.  Otherwise each subsequent
        extractor in the chain is tried and the highest-confidence
        result is kept.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of RoutedPage objects.

        Raises:
            ExtractionError: If every extractor in the chain fails completely.
        """
        # Try extractors in order until one succeeds for the full document
        primary_pages: list[Page] | None = None
        primary_name: str = ""

        for extractor, name in self._chain:
            try:
                primary_pages = extractor.extract(pdf_path)
                primary_name = name
                break
            except ExtractionError as exc:
                logger.warning("%s failed: %s — trying next extractor", name, exc)

        if primary_pages is None:
            raise ExtractionError(
                f"All extractors failed for {pdf_path.name}. "
                "Ensure at least PyMuPDF is installed: uv add pymupdf"
            )

        routed: list[RoutedPage] = []

        for page in primary_pages:
            best_page = page
            best_conf = self._page_confidence(page)
            best_name = primary_name

            if best_conf < self._threshold:
                # Walk the rest of the chain looking for a better result
                for extractor, name in self._chain:
                    if name == primary_name:
                        continue  # already tried
                    fallback_pages = extractor.extract_safe(pdf_path)
                    if fallback_pages and page.page_number <= len(fallback_pages):
                        candidate = fallback_pages[page.page_number - 1]
                        cand_conf = self._page_confidence(candidate)
                        if cand_conf > best_conf:
                            best_page, best_conf, best_name = candidate, cand_conf, name
                            logger.info(
                                "Page %d: %s (%.2f) -> %s (%.2f)",
                                page.page_number,
                                primary_name, self._page_confidence(page),
                                best_name, best_conf,
                            )
                    if best_conf >= self._threshold:
                        break

            routed.append(RoutedPage(
                page=best_page,
                confidence=best_conf,
                used_fallback=best_name != primary_name,
                extractor_name=best_name,
            ))

        logger.info(
            "Extraction complete: %d pages | extractors used: %s",
            len(routed),
            sorted({r.extractor_name for r in routed}),
        )
        return routed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _page_confidence(page: Page) -> float:
        """Compute aggregate confidence for a page.

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
