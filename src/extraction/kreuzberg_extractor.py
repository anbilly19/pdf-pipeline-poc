"""Kreuzberg-based PDF extractor — Rust-speed text extraction.

Kreuzberg >= 4.x returns an ExtractionResult with:
    .content   - full extracted text (str)
    .metadata  - dict, includes 'page_count' and optionally 'chunks'
    .chunks    - list of Chunk objects (if chunking enabled)

It does NOT return a document-with-pages object. This extractor uses
extract_file_sync (synchronous variant) to stay compatible with the
existing sync pipeline, then reconstructs per-page Page objects from
either chunk metadata or by splitting on page boundaries in .content.

Fallback
--------
If kreuzberg is not installed the extractor raises ExtractionError
and the router demotes to PyMuPDF automatically.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from src.extraction.base import BaseExtractor, ExtractionError
from src.models import Element, Page

logger = logging.getLogger(__name__)

_DEFAULT_CONFIDENCE = 0.92


class KreuzbergExtractor(BaseExtractor):
    """PDF extractor backed by the Kreuzberg Rust library (v4.x API)."""

    def __init__(self, confidence: float = _DEFAULT_CONFIDENCE) -> None:
        self._confidence = confidence
        self._check_import()

    def extract(self, pdf_path: Path) -> list[Page]:
        """Extract all pages from a PDF using Kreuzberg.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of normalised Page objects.

        Raises:
            ExtractionError: If the file cannot be read or Kreuzberg fails.
        """
        if not pdf_path.exists():
            raise ExtractionError(f"PDF not found: {pdf_path}")

        try:
            from kreuzberg import extract_file_sync  # noqa: PLC0415
        except ImportError as exc:
            raise ExtractionError(
                "kreuzberg is not installed. Run: uv add kreuzberg"
            ) from exc

        try:
            result = extract_file_sync(str(pdf_path))
        except Exception as exc:  # noqa: BLE001
            raise ExtractionError(
                f"Kreuzberg failed to parse {pdf_path.name}: {exc}"
            ) from exc

        pages = self._result_to_pages(result)
        logger.info("Kreuzberg extracted %d pages from %s", len(pages), pdf_path.name)
        return pages

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _result_to_pages(self, result: object) -> list[Page]:
        """Convert a kreuzberg ExtractionResult to a list of Page objects.

        Strategy (in order of preference):
        1. Use result.chunks if available — each chunk has page_number.
        2. Split result.content on form-feed characters (\x0c) which
           kreuzberg inserts as page separators.
        3. Return a single page with the full content.
        """
        # Strategy 1: chunk-based (most accurate, preserves page numbers)
        chunks = getattr(result, "chunks", None) or []
        if chunks:
            return self._pages_from_chunks(chunks)

        # Strategy 2: form-feed split
        content: str = getattr(result, "content", "") or ""
        if "\x0c" in content:
            return self._pages_from_formfeed(content)

        # Strategy 3: single page fallback
        metadata = getattr(result, "metadata", {}) or {}
        page_count = int(metadata.get("page_count", 1))
        if page_count > 1 and content:
            # Try to split roughly evenly by line count
            lines = content.splitlines(keepends=True)
            per_page = max(1, len(lines) // page_count)
            pages = []
            for i in range(page_count):
                start = i * per_page
                end = start + per_page if i < page_count - 1 else len(lines)
                chunk_text = "".join(lines[start:end]).strip()
                if chunk_text:
                    pages.append(self._make_page(i + 1, chunk_text))
            return pages if pages else [self._make_page(1, content)]

        return [self._make_page(1, content)] if content.strip() else []

    def _pages_from_chunks(self, chunks: list) -> list[Page]:
        """Build Page objects from kreuzberg Chunk list."""
        page_texts: dict[int, list[str]] = {}
        for chunk in chunks:
            page_num = int(getattr(chunk, "page_number", 1) or 1)
            text = str(getattr(chunk, "text", "") or "").strip()
            if text:
                page_texts.setdefault(page_num, []).append(text)
        return [
            self._make_page(pnum, "\n".join(texts))
            for pnum, texts in sorted(page_texts.items())
        ]

    def _pages_from_formfeed(self, content: str) -> list[Page]:
        """Build Page objects by splitting on form-feed characters."""
        raw_pages = content.split("\x0c")
        pages = []
        for i, text in enumerate(raw_pages, start=1):
            text = text.strip()
            if text:
                pages.append(self._make_page(i, text))
        return pages

    def _make_page(self, page_number: int, text: str) -> Page:
        """Wrap a text block in a Page with a single text Element."""
        # Split into paragraph-level elements for finer granularity
        elements: list[Element] = []
        paragraphs = re.split(r"\n{2,}", text)
        for para in paragraphs:
            para = para.strip()
            if len(para) >= 2:
                elements.append(
                    Element(
                        type="text",
                        text=para,
                        bbox=[0.0, 0.0, 0.0, 0.0],  # no bbox from this API
                        confidence=self._confidence,
                    )
                )
        if not elements and text.strip():
            elements.append(
                Element(
                    type="text",
                    text=text.strip(),
                    bbox=[0.0, 0.0, 0.0, 0.0],
                    confidence=self._confidence,
                )
            )
        return Page(page_number=page_number, image_path="", elements=elements)

    @staticmethod
    def _check_import() -> None:
        """Warn at construction time if kreuzberg is not installed."""
        try:
            import kreuzberg  # noqa: F401, PLC0415
        except ImportError:
            logger.warning(
                "kreuzberg not installed — KreuzbergExtractor will raise ExtractionError "
                "on use. Run: uv add kreuzberg"
            )
