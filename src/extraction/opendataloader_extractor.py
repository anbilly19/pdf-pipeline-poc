"""OpenDataLoader-based PDF extractor — primary extractor with full bbox and table support.

Produces Element/Page objects from ODL's nested JSON output.
Requires: opendataloader-pdf>=1.12.0 and Java 11+.

Element type mapping:
    paragraph / heading / list  ->  type="text"
    table                       ->  type="table"  (markdown-rendered)

Confidence constants:
    heading    0.95  (high — structural signal)
    paragraph  0.92  (standard digital-born text)
    list       0.90  (slightly lower — list items can be noisy)
    table      0.85  (lower — cell merging heuristics may err)
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from src.extraction.base import BaseExtractor, ExtractionError
from src.models import Element, Page

logger = logging.getLogger(__name__)

# Confidence scores per ODL element type
_CONFIDENCE: dict[str, float] = {
    "heading": 0.95,
    "paragraph": 0.92,
    "list": 0.90,
    "table": 0.85,
}
_CONFIDENCE_DEFAULT = 0.88

_MIN_TEXT_LEN = 3


class OpenDataLoaderExtractor(BaseExtractor):
    """Extracts text and tables with bounding boxes using OpenDataLoader PDF.

    ODL writes a JSON file to disk; this extractor reads that file and
    maps the nested structure into Element/Page model objects.

    Suitable for digitally-born PDFs with complex tables and German text.
    Falls back gracefully if ODL or Java is unavailable.
    """

    def extract(self, pdf_path: Path) -> list[Page]:
        """Extract pages from a PDF using OpenDataLoader.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            List of Page objects with text and table Elements and bboxes.

        Raises:
            ExtractionError: If ODL is not installed, Java is missing,
                             or the PDF cannot be parsed.
        """
        if not pdf_path.exists():
            raise ExtractionError(f"PDF not found: {pdf_path}")

        try:
            from opendataloader_pdf import convert  # noqa: PLC0415
        except ImportError as exc:
            raise ExtractionError(
                "opendataloader-pdf is not installed. "
                "Run: pip install opendataloader-pdf"
            ) from exc

        # ODL writes JSON next to the source PDF by default.
        # We redirect output to a temp directory to avoid polluting
        # the source folder.
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            json_out = tmp_path / (pdf_path.stem + ".json")

            try:
                convert(str(pdf_path.resolve()), str(tmp_path))
            except Exception as exc:
                raise ExtractionError(
                    f"OpenDataLoader failed on {pdf_path}: {exc}"
                ) from exc

            if not json_out.exists():
                # ODL may write to cwd if output_dir arg is unsupported —
                # fall back to looking next to the PDF.
                fallback = pdf_path.with_suffix(".json")
                if fallback.exists():
                    json_out = fallback
                else:
                    raise ExtractionError(
                        f"ODL produced no JSON output for {pdf_path}"
                    )

            try:
                raw = json.loads(json_out.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ExtractionError(
                    f"Failed to read ODL JSON output: {exc}"
                ) from exc

        return self._parse(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse(self, raw: dict) -> list[Page]:
        """Convert ODL JSON dict into a list of Page objects.

        Args:
            raw: Parsed ODL JSON output dict with 'kids' list.

        Returns:
            List of Page objects grouped by page number.
        """
        kids: list[dict] = raw.get("kids", [])

        # Group elements by page number (preserve document order)
        pages_map: dict[int, list[Element]] = {}
        for kid in kids:
            element = self._map_element(kid)
            if element is None:
                continue
            page_num: int = kid.get("page number", 1)
            pages_map.setdefault(page_num, []).append(element)

        pages: list[Page] = []
        for page_num in sorted(pages_map):
            elements = pages_map[page_num]
            pages.append(
                Page(
                    page_number=page_num,
                    image_path="",
                    elements=elements,
                )
            )
            logger.debug(
                "ODL page %d: %d elements", page_num, len(elements)
            )

        logger.info(
            "ODL extracted %d pages, %d elements total",
            len(pages),
            sum(len(p.elements) for p in pages),
        )
        return pages

    def _map_element(self, kid: dict) -> Element | None:
        """Map a single ODL 'kid' dict to an Element.

        Args:
            kid: A single element dict from ODL JSON 'kids' list.

        Returns:
            Element instance, or None if the element should be skipped.
        """
        odl_type: str = kid.get("type", "paragraph").lower()
        bbox: list[float] = kid.get("bounding box", [])
        confidence: float = _CONFIDENCE.get(odl_type, _CONFIDENCE_DEFAULT)

        if len(bbox) != 4:
            logger.debug("Skipping element with invalid bbox: %s", kid.get("id"))
            return None

        if odl_type == "table":
            text = _render_table_markdown(kid)
            if len(text.strip()) < _MIN_TEXT_LEN:
                return None
            return Element(
                type="table",
                text=text,
                bbox=bbox,
                confidence=confidence,
            )

        # paragraph / heading / list -> text
        text: str = kid.get("content", "").strip()
        if len(text) < _MIN_TEXT_LEN:
            return None

        return Element(
            type="text",
            text=text,
            bbox=bbox,
            confidence=confidence,
        )


# ------------------------------------------------------------------
# Table rendering
# ------------------------------------------------------------------

def _render_table_markdown(table: dict) -> str:
    """Render an ODL table element as a pipe-delimited markdown table.

    Handles row/cell nesting. Empty cells are rendered as a single space
    to preserve column alignment.

    Args:
        table: ODL table dict with 'rows' list.

    Returns:
        Markdown string representation of the table.
    """
    rows: list[dict] = table.get("rows", [])
    if not rows:
        return ""

    lines: list[str] = []
    for row_idx, row in enumerate(rows):
        cells: list[dict] = row.get("cells", [])
        # Sort cells by column number to guarantee left-to-right order
        cells_sorted = sorted(cells, key=lambda c: c.get("column number", 0))
        cell_texts = [
            _cell_text(c).replace("|", "\\|").replace("\n", " ")
            for c in cells_sorted
        ]
        lines.append("| " + " | ".join(cell_texts) + " |")
        # Insert markdown header separator after the first row
        if row_idx == 0:
            lines.append("| " + " | ".join("---" for _ in cell_texts) + " |")

    return "\n".join(lines)


def _cell_text(cell: dict) -> str:
    """Extract plain text from a table cell dict.

    Args:
        cell: ODL table cell dict.

    Returns:
        Stripped cell text, or a single space if empty.
    """
    text = cell.get("content", cell.get("text", "")).strip()
    return text if text else " "
