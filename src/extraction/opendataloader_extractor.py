"""OpenDataLoader-based PDF extractor — primary extractor with full bbox and table support.

Produces Element/Page objects from ODL's nested JSON output.
Requires: opendataloader-pdf>=1.12.0 and Java 11+.

ODL JSON structure
------------------
Top-level `kids` contains page elements. Each element may have nested
`kids` of its own (list items, table rows/cells). Content is stored in
`content` for paragraphs/headings, and recursively in nested `kids` for
list and table elements. This extractor flattens all nested content.

Element type mapping:
    paragraph / heading  ->  type="text"
    list                 ->  type="text"  (flattened bullet text)
    table                ->  type="table" (markdown-rendered)

Page number normalisation
--------------------------
ODL reports LOGICAL page labels from PDF metadata. We build a
label→physical-index map using fitz to ensure chunk.page_number always
matches the physical page the renderer saved as page_NNNN.png.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from src.extraction.base import BaseExtractor, ExtractionError
from src.models import Element, Page

logger = logging.getLogger(__name__)

_CONFIDENCE: dict[str, float] = {
    "heading": 0.95,
    "paragraph": 0.92,
    "list": 0.90,
    "table": 0.85,
}
_CONFIDENCE_DEFAULT = 0.88
_MIN_TEXT_LEN = 3


class OpenDataLoaderExtractor(BaseExtractor):
    """Extracts text and tables with bounding boxes using OpenDataLoader PDF."""

    def extract(self, pdf_path: Path) -> list[Page]:
        if not pdf_path.exists():
            raise ExtractionError(f"PDF not found: {pdf_path}")

        try:
            from opendataloader_pdf import convert  # noqa: PLC0415
        except ImportError as exc:
            raise ExtractionError(
                "opendataloader-pdf is not installed. Run: pip install opendataloader-pdf"
            ) from exc

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            json_out = tmp_path / (pdf_path.stem + ".json")
            try:
                convert(str(pdf_path.resolve()), str(tmp_path))
            except Exception as exc:
                raise ExtractionError(f"OpenDataLoader failed on {pdf_path}: {exc}") from exc

            if not json_out.exists():
                raise ExtractionError(f"ODL produced no JSON output for {pdf_path}")

            try:
                raw = json.loads(json_out.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ExtractionError(f"Failed to read ODL JSON: {exc}") from exc

        label_to_physical = _build_label_to_physical_map(pdf_path)
        return self._parse(raw, label_to_physical)

    def _parse(self, raw: dict, label_to_physical: dict[int, int]) -> list[Page]:
        kids: list[dict] = raw.get("kids", [])
        pages_map: dict[int, list[Element]] = {}

        for kid in kids:
            odl_page: int = kid.get("page number", 1)
            physical_page = label_to_physical.get(odl_page, odl_page)

            elements = self._map_element(kid)
            for element in elements:
                pages_map.setdefault(physical_page, []).append(element)

        pages: list[Page] = []
        for page_num in sorted(pages_map):
            elements = pages_map[page_num]
            pages.append(Page(
                page_number=page_num,
                image_path="",
                elements=elements,
            ))
            logger.debug("ODL page %d: %d elements", page_num, len(elements))

        logger.info(
            "ODL extracted %d pages, %d elements total",
            len(pages),
            sum(len(p.elements) for p in pages),
        )
        return pages

    def _map_element(self, kid: dict) -> list[Element]:
        """Map a single ODL element to one or more Element objects.

        Recursively extracts nested list items and table content.
        Returns a list (usually one item, but lists expand to multiple).
        """
        odl_type: str = kid.get("type", "paragraph").lower()
        bbox: list[float] = kid.get("bounding box", [])
        confidence: float = _CONFIDENCE.get(odl_type, _CONFIDENCE_DEFAULT)

        if len(bbox) != 4:
            logger.debug("Skipping element with invalid bbox: type=%s", odl_type)
            return []

        if odl_type == "table":
            text = _render_table_markdown(kid)
            if len(text.strip()) >= _MIN_TEXT_LEN:
                return [Element(type="table", text=text, bbox=bbox, confidence=confidence)]
            return []

        if odl_type == "list":
            # Flatten nested list kids into a single text block
            text = _flatten_list(kid)
            if len(text.strip()) >= _MIN_TEXT_LEN:
                return [Element(type="text", text=text, bbox=bbox, confidence=confidence)]
            return []

        # paragraph / heading / unknown
        text: str = kid.get("content", "").strip()
        if len(text) < _MIN_TEXT_LEN:
            return []
        return [Element(type="text", text=text, bbox=bbox, confidence=confidence)]


# ---------------------------------------------------------------------------
# List flattening
# ---------------------------------------------------------------------------

def _flatten_list(node: dict, depth: int = 0) -> str:
    """Recursively extract text from a list element and its nested kids.

    ODL list structure:
        {type: list, kids: [
            {type: list-item, content: "text", kids: [...]},
            ...
        ]}
    """
    lines: list[str] = []

    # Direct content on this node
    content = node.get("content", "").strip()
    if content:
        lines.append(content)

    # Recurse into kids
    for child in node.get("kids", []):
        child_type = child.get("type", "").lower()
        child_content = child.get("content", "").strip()

        if child_content:
            lines.append(child_content)

        # Recurse further if the child has its own kids
        if child.get("kids"):
            nested = _flatten_list(child, depth + 1)
            if nested:
                lines.append(nested)

    return "\n".join(line for line in lines if line.strip())


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _render_table_markdown(table: dict) -> str:
    """Render an ODL table element as a pipe-delimited markdown table.

    Handles both flat rows list and nested kids structure.
    """
    rows: list[dict] = table.get("rows", []) or [
        k for k in table.get("kids", []) if k.get("type", "").lower() == "row"
    ]
    if not rows:
        # Try flattening as list as last resort
        text = _flatten_list(table)
        return text if len(text.strip()) >= _MIN_TEXT_LEN else ""

    lines: list[str] = []
    for row_idx, row in enumerate(rows):
        cells: list[dict] = row.get("cells", []) or [
            k for k in row.get("kids", []) if k.get("type", "").lower() in ("cell", "table-cell")
        ]
        cells_sorted = sorted(cells, key=lambda c: c.get("column number", 0))
        cell_texts = [
            _cell_text(c).replace("|", "\\|").replace("\n", " ")
            for c in cells_sorted
        ]
        if not cell_texts:
            continue
        lines.append("| " + " | ".join(cell_texts) + " |")
        if row_idx == 0:
            lines.append("| " + " | ".join("---" for _ in cell_texts) + " |")

    return "\n".join(lines)


def _cell_text(cell: dict) -> str:
    text = cell.get("content", cell.get("text", "")).strip()
    if not text and cell.get("kids"):
        text = _flatten_list(cell)
    return text if text else " "


# ---------------------------------------------------------------------------
# Page label map
# ---------------------------------------------------------------------------

def _build_label_to_physical_map(pdf_path: Path) -> dict[int, int]:
    """Map PDF logical page labels to physical 1-based indices via fitz."""
    try:
        import fitz  # noqa: PLC0415
        doc = fitz.open(str(pdf_path))
        mapping: dict[int, int] = {}
        seen: set[int] = set()
        for idx in range(len(doc)):
            physical = idx + 1
            try:
                label_int = int(doc[idx].get_label())
            except (ValueError, TypeError):
                mapping[physical] = physical
                continue
            if label_int not in seen:
                mapping[label_int] = physical
                seen.add(label_int)
            else:
                logger.warning(
                    "Duplicate page label %d at physical page %d", label_int, physical
                )
        doc.close()
        logger.info("Page label map: %s", mapping)
        return mapping
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not build page label map: %s — using identity", exc)
        return {}
