"""PyMuPDF-based PDF extractor — CPU, always available as primary fallback."""
from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

from src.extraction.base import BaseExtractor, ExtractionError
from src.models import Element, Page

logger = logging.getLogger(__name__)

_CONFIDENCE = 0.90  # PyMuPDF text positions are highly reliable
_MIN_TEXT_LEN = 3   # skip noise spans shorter than this


class PyMuPDFExtractor(BaseExtractor):
    """Extracts text blocks with bounding boxes using PyMuPDF.

    Suitable for digitally-born PDFs. Falls back gracefully on
    scanned pages (returns empty element list, not an error).
    """

    def extract(self, pdf_path: Path) -> list[Page]:
        """Extract pages from a PDF using PyMuPDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of Page objects with text Elements and bboxes.

        Raises:
            ExtractionError: If the file cannot be opened or parsed.
        """
        if not pdf_path.exists():
            raise ExtractionError(f"PDF not found: {pdf_path}")

        try:
            doc: fitz.Document = fitz.open(str(pdf_path))
        except Exception as exc:
            raise ExtractionError(f"Cannot open PDF: {pdf_path}") from exc

        pages: list[Page] = []
        try:
            for page_idx in range(len(doc)):
                fitz_page: fitz.Page = doc[page_idx]
                elements = self._extract_elements(fitz_page)
                pages.append(
                    Page(
                        page_number=page_idx + 1,
                        image_path="",
                        elements=elements,
                    )
                )
                logger.debug(
                    "Page %d: extracted %d elements", page_idx + 1, len(elements)
                )
        finally:
            doc.close()

        return pages

    def _extract_elements(self, page: fitz.Page) -> list[Element]:
        """Extract text block elements from a single fitz page.

        Args:
            page: The fitz page object.

        Returns:
            List of Element objects with bboxes in PDF points.
        """
        elements: list[Element] = []
        raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block in raw["blocks"]:
            if block["type"] != 0:  # skip image blocks
                continue

            text = " ".join(
                span["text"]
                for line in block["lines"]
                for span in line["spans"]
            ).strip()

            if len(text) < _MIN_TEXT_LEN:
                continue

            elements.append(
                Element(
                    type="text",
                    text=text,
                    bbox=list(block["bbox"]),
                    confidence=_CONFIDENCE,
                )
            )

        return elements
