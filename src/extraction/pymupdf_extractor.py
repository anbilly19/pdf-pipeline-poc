"""PyMuPDF-based PDF extractor (CPU, always available as fallback)."""
from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

from src.extraction.base import BaseExtractor, ExtractionError
from src.models import Element, Page

logger = logging.getLogger(__name__)

_CONFIDENCE_DEFAULT = 0.9  # PyMuPDF text positions are highly reliable


class PyMuPDFExtractor(BaseExtractor):
    """Extracts text blocks with bounding boxes using PyMuPDF."""

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

        pages: list[Page] = []
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            raise ExtractionError(f"Cannot open PDF: {pdf_path}") from exc

        for page_idx in range(len(doc)):
            fitz_page = doc[page_idx]
            image_path = ""  # rendered separately if needed
            elements: list[Element] = []

            blocks = fitz_page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] == 0:  # text block
                    full_text = " ".join(
                        span["text"]
                        for line in block["lines"]
                        for span in line["spans"]
                    ).strip()
                    if not full_text:
                        continue
                    bbox = list(block["bbox"])  # (x0, y0, x1, y1)
                    elements.append(
                        Element(
                            type="text",
                            text=full_text,
                            bbox=bbox,
                            confidence=_CONFIDENCE_DEFAULT,
                        )
                    )

            pages.append(
                Page(
                    page_number=page_idx + 1,
                    image_path=image_path,
                    elements=elements,
                )
            )
            logger.debug("Extracted page %d: %d elements", page_idx + 1, len(elements))

        doc.close()
        return pages
