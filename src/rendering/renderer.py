"""Renders PDF pages to PNG images for frontend overlay display.

Each page is rendered once and cached to disk. The image path is
written back to the Page object so downstream stages can reference it.
"""
from __future__ import annotations

import logging
from pathlib import Path

import fitz

from src.models import Page

logger = logging.getLogger(__name__)

_DEFAULT_DPI = 150


class PageRenderer:
    """Renders PDF pages to PNG and attaches image paths to Page objects.

    Args:
        output_dir: Directory where rendered PNGs are stored.
        dpi: Render resolution (150 dpi is a good balance of quality/size).
    """

    def __init__(self, output_dir: Path, dpi: int = _DEFAULT_DPI) -> None:
        self._output_dir = output_dir
        self._dpi = dpi
        output_dir.mkdir(parents=True, exist_ok=True)

    def render(self, pdf_path: Path, pages: list[Page]) -> list[Page]:
        """Render pages to PNG and update image_path on each Page.

        Args:
            pdf_path: Source PDF (used to open fitz document).
            pages: Pages to render (mutated in-place with image_path).

        Returns:
            The same pages list with image_path populated.

        Raises:
            OSError: If the output directory cannot be written to.
        """
        try:
            doc: fitz.Document = fitz.open(str(pdf_path))
        except Exception as exc:
            logger.error("Cannot open PDF for rendering: %s — %s", pdf_path, exc)
            return pages

        scale = self._dpi / 72.0
        mat = fitz.Matrix(scale, scale)

        try:
            for page in pages:
                idx = page.page_number - 1
                if idx < 0 or idx >= len(doc):
                    logger.warning("Page %d out of range for %s", page.page_number, pdf_path)
                    continue

                out_path = self._output_dir / f"page_{page.page_number:04d}.png"
                if not out_path.exists():
                    pix: fitz.Pixmap = doc[idx].get_pixmap(matrix=mat)
                    pix.save(str(out_path))
                    logger.debug("Rendered page %d -> %s", page.page_number, out_path)

                page.image_path = str(out_path)
        finally:
            doc.close()

        return pages
