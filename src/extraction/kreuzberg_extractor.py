"""Kreuzberg-based PDF extractor — Rust-speed, precise bounding boxes.

Kreuzberg (https://github.com/Goldziher/kreuzberg) is a Rust-backed
Python library for PDF text extraction with cell-level bounding boxes.
It requires no Java, no GPU, and no HuggingFace API.

Install
-------
    uv add kreuzberg

Coordinate system
-----------------
Kreuzberg returns bounding boxes in PDF points (72 pts = 1 inch),
origin at the *bottom-left* corner of the page (standard PDF convention).
This extractor normalises to top-left origin by flipping the y-axis:

    y0_normalised = page_height - bbox.y1
    y1_normalised = page_height - bbox.y0

so that all downstream consumers see a consistent top-left coordinate
system regardless of which extractor produced the page.

Fallback
--------
If kreuzberg is not installed (ImportError) the extractor raises
ExtractionError with a clear install hint.  The router catches this
and demotes to PyMuPDF automatically.

If a page has a very low confidence score (e.g. scanned image page)
the router's per-page confidence check will trigger the PyMuPDF
fallback for that individual page.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from pathlib import Path

from src.extraction.base import BaseExtractor, ExtractionError
from src.models import Element, Page

logger = logging.getLogger(__name__)

# Minimum text length for an element to be considered real content
_MIN_TEXT_LEN = 2
# Minimum bounding box area in points² to be considered a real element
_MIN_BBOX_AREA = 10.0
# Default confidence for Kreuzberg elements (it does not expose a score)
_DEFAULT_CONFIDENCE = 0.92


class KreuzbergExtractor(BaseExtractor):
    """PDF extractor backed by the Kreuzberg Rust library.

    Produces precise bounding boxes at the text-block level.
    Coordinates are normalised to top-left origin before returning.

    Args:
        confidence: Confidence value to assign to all extracted elements.
                    Kreuzberg does not expose per-element confidence scores,
                    so a fixed high value is used.
    """

    def __init__(self, confidence: float = _DEFAULT_CONFIDENCE) -> None:
        self._confidence = confidence
        self._check_import()

    def extract(self, pdf_path: Path) -> list[Page]:
        """Extract all pages from a PDF using Kreuzberg.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of normalised Page objects with bounding boxes in
            top-left coordinate system.

        Raises:
            ExtractionError: If the file cannot be read or Kreuzberg fails.
        """
        if not pdf_path.exists():
            raise ExtractionError(f"PDF not found: {pdf_path}")

        try:
            import kreuzberg  # noqa: PLC0415
        except ImportError as exc:
            raise ExtractionError(
                "kreuzberg is not installed. Run: uv add kreuzberg"
            ) from exc

        try:
            coro = kreuzberg.extract_file(str(pdf_path))
            if asyncio.iscoroutine(coro):
                try:
                    asyncio.get_running_loop()
                    # Already inside a running event loop (e.g. Streamlit).
                    # Dispatch to a worker thread to avoid nested-loop error.
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        document = pool.submit(asyncio.run, coro).result()
                except RuntimeError:
                    # No running loop — safe to block directly.
                    document = asyncio.run(coro)
            else:
                # kreuzberg < 4.x returned a plain object (sync path)
                document = coro
        except ExtractionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExtractionError(
                f"Kreuzberg failed to parse {pdf_path.name}: {exc}"
            ) from exc

        pages: list[Page] = []
        for page_data in document.pages:
            page = self._convert_page(page_data)
            pages.append(page)

        logger.info(
            "Kreuzberg extracted %d pages from %s",
            len(pages), pdf_path.name,
        )
        return pages

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _convert_page(self, page_data: object) -> Page:
        """Convert a Kreuzberg page object to our normalised Page model.

        Args:
            page_data: A kreuzberg page object with .number, .width,
                       .height, and .blocks attributes.

        Returns:
            Normalised Page with top-left bounding boxes.
        """
        page_height: float = getattr(page_data, "height", 842.0)  # A4 default
        page_number: int = getattr(page_data, "number", 1)
        image_path: str = getattr(page_data, "image_path", "")

        elements: list[Element] = []
        for block in getattr(page_data, "blocks", []):
            element = self._convert_block(block, page_height)
            if element is not None:
                elements.append(element)

        return Page(
            page_number=page_number,
            image_path=image_path,
            elements=elements,
        )

    def _convert_block(self, block: object, page_height: float) -> Element | None:
        """Convert a Kreuzberg text block to an Element.

        Normalises the y-axis from PDF bottom-left to top-left origin.
        Filters out empty or tiny elements.

        Args:
            block: A kreuzberg block object with .text, .bbox, .type.
            page_height: Page height in points for y-flip.

        Returns:
            Element or None if the block should be filtered.
        """
        text: str = getattr(block, "text", "") or ""
        text = text.strip()
        if len(text) < _MIN_TEXT_LEN:
            return None

        raw_bbox = getattr(block, "bbox", None)
        if raw_bbox is None:
            # Block has no bbox — assign a zero placeholder
            bbox = [0.0, 0.0, 0.0, 0.0]
        else:
            # Kreuzberg bbox: (x0, y0, x1, y1) bottom-left origin
            # Normalise to top-left: flip y-axis
            x0 = float(getattr(raw_bbox, "x0", raw_bbox[0]))
            y0_raw = float(getattr(raw_bbox, "y0", raw_bbox[1]))
            x1 = float(getattr(raw_bbox, "x1", raw_bbox[2]))
            y1_raw = float(getattr(raw_bbox, "y1", raw_bbox[3]))

            # y-flip: top-left y = page_height - bottom-left y
            y0 = page_height - y1_raw
            y1 = page_height - y0_raw
            bbox = [x0, y0, x1, y1]

        # Filter zero-area bboxes
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < _MIN_BBOX_AREA and bbox != [0.0, 0.0, 0.0, 0.0]:
            return None

        # Determine element type from block type hint
        raw_type = str(getattr(block, "type", "text")).lower()
        if "table" in raw_type:
            el_type = "table"
        elif "image" in raw_type or "figure" in raw_type:
            el_type = "image"
        else:
            el_type = "text"

        return Element(
            type=el_type,
            text=text,
            bbox=bbox,
            confidence=self._confidence,
        )

    @staticmethod
    def _check_import() -> None:
        """Warn at construction time if kreuzberg is not installed.

        Does not raise — the error is deferred to extract() so the
        router can still instantiate the class and catch the error
        gracefully during the actual extraction call.
        """
        try:
            import kreuzberg  # noqa: F401, PLC0415
        except ImportError:
            logger.warning(
                "kreuzberg not installed — KreuzbergExtractor will raise ExtractionError "
                "on use. Run: uv add kreuzberg"
            )
