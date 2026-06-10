"""Extraction router: Kreuzberg -> ODL -> PyMuPDF -> PyMuPDF-layout.

Architecture
------------
    Primary:    KreuzbergExtractor       (Rust-speed, best text quality, no bboxes)
    Secondary:  OpenDataLoaderExtractor  (Java 11, best table quality)
    Tertiary:   PyMuPDF                  (always available, digital-born PDFs)
    Quaternary: PyMuPDFLayoutExtractor   (CPU GNN, complex multi-column layouts)

Bbox enrichment
---------------
If the winning extractor produces pages with all-zero bboxes (e.g. Kreuzberg),
the router runs a silent PyMuPDF pass *only* for bbox coordinates and stitches
them onto the Kreuzberg elements. Text content is never replaced.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.extraction.base import BaseExtractor, ExtractionError
from src.extraction.pymupdf_extractor import PyMuPDFExtractor
from src.models import Element, Page

logger = logging.getLogger(__name__)

_DEFAULT_CONFIDENCE_THRESHOLD = 0.85
_MIN_ELEMENTS_PER_PAGE = 1


@dataclass
class RoutedPage:
    page: Page
    confidence: float
    used_fallback: bool = False
    extractor_name: str = "unknown"


def _all_zero_bboxes(page: Page) -> bool:
    """Return True if every element on this page has a zero/empty bbox."""
    if not page.elements:
        return False
    return all(
        not e.bbox or e.bbox == [0.0, 0.0, 0.0, 0.0]
        for e in page.elements
    )


def _enrich_bboxes(
    pages: list[Page],
    pdf_path: Path,
    pymupdf: PyMuPDFExtractor,
) -> list[Page]:
    """Overwrite zero bboxes on pages using a silent PyMuPDF pass.

    For each page where all elements have zero bboxes we extract the real
    blocks via PyMuPDF and assign them positionally. If PyMuPDF returns a
    different element count we spread block bboxes across elements as best
    we can; text is never changed.

    Args:
        pages:    Pages from the primary extractor (Kreuzberg text quality).
        pdf_path: Source PDF needed to open a fitz document.
        pymupdf:  Shared PyMuPDFExtractor instance (avoids re-importing fitz).

    Returns:
        The same page list with bboxes filled in where possible.
    """
    pages_needing_bboxes = [p for p in pages if _all_zero_bboxes(p)]
    if not pages_needing_bboxes:
        return pages

    try:
        mupdf_pages: list[Page] = pymupdf.extract(pdf_path)
    except ExtractionError as exc:
        logger.warning("BBox enrichment: PyMuPDF pass failed (%s) — bboxes stay zero", exc)
        return pages

    mupdf_by_num: dict[int, Page] = {p.page_number: p for p in mupdf_pages}
    enriched = 0

    for page in pages_needing_bboxes:
        mp = mupdf_by_num.get(page.page_number)
        if not mp or not mp.elements:
            continue

        mu_bboxes = [e.bbox for e in mp.elements]
        n_krz = len(page.elements)
        n_mu  = len(mu_bboxes)

        for i, element in enumerate(page.elements):
            # map element i -> nearest PyMuPDF block by proportional position
            mu_idx = min(round(i * n_mu / n_krz), n_mu - 1)
            element.bbox = mu_bboxes[mu_idx]

        enriched += 1
        logger.debug(
            "BBox enrichment: page %d — %d Kreuzberg elements <- %d PyMuPDF blocks",
            page.page_number, n_krz, n_mu,
        )

    if enriched:
        logger.info(
            "BBox enrichment: filled real bboxes on %d/%d pages via PyMuPDF",
            enriched, len(pages),
        )
    return pages


class ExtractionRouter:
    """Routes each page through the extractor chain, best-first."""

    def __init__(self, threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        self._threshold = threshold
        self._pymupdf = PyMuPDFExtractor()   # kept for bbox enrichment
        self._chain: list[tuple[BaseExtractor, str]] = self._build_chain()

    def _build_chain(self) -> list[tuple[BaseExtractor, str]]:
        chain: list[tuple[BaseExtractor, str]] = []

        try:
            from src.extraction.kreuzberg_extractor import KreuzbergExtractor  # noqa: PLC0415
            chain.append((KreuzbergExtractor(), "Kreuzberg"))
            logger.info("Kreuzberg extractor registered as primary")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Kreuzberg unavailable (%s); skipping", exc)

        try:
            from src.extraction.opendataloader_extractor import OpenDataLoaderExtractor  # noqa: PLC0415
            chain.append((OpenDataLoaderExtractor(), "OpenDataLoader"))
            logger.info("OpenDataLoader extractor registered as secondary")
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenDataLoader unavailable (%s); skipping", exc)

        chain.append((self._pymupdf, "PyMuPDF"))

        try:
            from src.extraction.layout_extractor import PyMuPDFLayoutExtractor  # noqa: PLC0415
            chain.append((PyMuPDFLayoutExtractor(), "PyMuPDFLayout"))
            logger.info("PyMuPDF-layout extractor registered as quaternary")
        except ImportError:
            logger.warning("pymupdf-layout not installed; quaternary extractor disabled")

        logger.info(
            "Extractor chain: %s",
            " -> ".join(name for _, name in chain),
        )
        return chain

    def extract(self, pdf_path: Path) -> list[RoutedPage]:
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

        # --- Bbox enrichment: fill zero bboxes using PyMuPDF block coords ---
        primary_pages = _enrich_bboxes(primary_pages, pdf_path, self._pymupdf)

        routed: list[RoutedPage] = []

        for page in primary_pages:
            best_page = page
            best_conf = self._page_confidence(page)
            best_name = primary_name

            if best_conf < self._threshold:
                for extractor, name in self._chain:
                    if name == primary_name:
                        continue
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

    @staticmethod
    def _page_confidence(page: Page) -> float:
        if not page.elements:
            return 0.0
        mean = sum(e.confidence for e in page.elements) / len(page.elements)
        if len(page.elements) < _MIN_ELEMENTS_PER_PAGE:
            mean *= 0.5
        return round(mean, 4)
