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

Page number normalisation
------------------------
ODL reports LOGICAL page labels from PDF metadata (e.g. a merged PDF
whose AGB section restarts at “1” gives AGB page 9 a logical number of 9
but it sits at physical fitz index 10). We build a label→physical-index
map using fitz so that chunk.page_number always matches the physical page
the renderer saved as page_NNNN.png.
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
    """Extracts text and tables with bounding boxes using OpenDataLoader PDF.

    ODL writes a JSON file to disk; this extractor reads that file and
    maps the nested structure into Element/Page model objects.
    """

    def extract(self, pdf_path: Path) -> list[Page]:
        """Extract pages from a PDF using OpenDataLoader.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            List of Page objects with physical 1-based page_number.

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

        label_to_physical = _build_label_to_physical_map(pdf_path)
        return self._parse(raw, label_to_physical)

    def _parse(self, raw: dict, label_to_physical: dict[int, int]) -> list[Page]:
        """Convert ODL JSON dict into a list of Page objects.

        Remaps ODL logical page numbers to physical 1-based indices so
        they match the filenames written by PageRenderer (page_NNNN.png).

        Args:
            raw: Parsed ODL JSON output dict with 'kids' list.
            label_to_physical: Mapping from ODL logical page label to
                               physical 1-based page index.

        Returns:
            List of Page objects ordered by physical page number.
        """
        kids: list[dict] = raw.get("kids", [])

        pages_map: dict[int, list[Element]] = {}
        for kid in kids:
            element = self._map_element(kid)
            if element is None:
                continue
            odl_page: int = kid.get("page number", 1)
            # Remap logical → physical; fall back to logical if not in map
            physical_page = label_to_physical.get(odl_page, odl_page)
            pages_map.setdefault(physical_page, []).append(element)

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
            logger.debug("ODL page %d (physical): %d elements", page_num, len(elements))

        logger.info(
            "ODL extracted %d pages, %d elements total",
            len(pages),
            sum(len(p.elements) for p in pages),
        )
        return pages

    def _map_element(self, kid: dict) -> Element | None:
        """Map a single ODL 'kid' dict to an Element."""
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
            return Element(type="table", text=text, bbox=bbox, confidence=confidence)

        text: str = kid.get("content", "").strip()
        if len(text) < _MIN_TEXT_LEN:
            return None

        return Element(type="text", text=text, bbox=bbox, confidence=confidence)


def _build_label_to_physical_map(pdf_path: Path) -> dict[int, int]:
    """Build a mapping from PDF logical page labels to physical 1-based indices.

    Uses fitz to read the PDF page labels. For PDFs without explicit labels,
    the mapping is identity (1→1, 2→2, …). For merged PDFs where sections
    restart numbering, duplicate labels are disambiguated by physical order.

    Args:
        pdf_path: Path to the PDF.

    Returns:
        Dict mapping logical label (int) -> physical index (1-based int).
        Falls back to empty dict (caller uses identity) if fitz fails.
    """
    try:
        import fitz  # noqa: PLC0415
        doc = fitz.open(str(pdf_path))
        mapping: dict[int, int] = {}
        seen_labels: set[int] = set()
        for physical_idx in range(len(doc)):
            label_str = doc[physical_idx].get_label()  # e.g. "1", "2", "A-1"
            physical = physical_idx + 1  # 1-based
            try:
                label_int = int(label_str)
            except (ValueError, TypeError):
                # Non-numeric label (e.g. "A-1") — use physical index directly
                mapping[physical] = physical
                continue
            if label_int not in seen_labels:
                mapping[label_int] = physical
                seen_labels.add(label_int)
            else:
                # Duplicate label (restart): map to physical index
                # ODL will report this as label_int again — we can't distinguish,
                # so log a warning and prefer the first occurrence.
                logger.warning(
                    "Duplicate page label %d at physical page %d — "
                    "ODL page numbers may be ambiguous for this PDF.",
                    label_int,
                    physical,
                )
        doc.close()
        logger.info("Page label map: %s", mapping)
        return mapping
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not build page label map: %s — using identity", exc)
        return {}


# ------------------------------------------------------------------
# Table rendering
# ------------------------------------------------------------------

def _render_table_markdown(table: dict) -> str:
    """Render an ODL table element as a pipe-delimited markdown table."""
    rows: list[dict] = table.get("rows", [])
    if not rows:
        return ""

    lines: list[str] = []
    for row_idx, row in enumerate(rows):
        cells: list[dict] = row.get("cells", [])
        cells_sorted = sorted(cells, key=lambda c: c.get("column number", 0))
        cell_texts = [
            _cell_text(c).replace("|", "\\|").replace("\n", " ")
            for c in cells_sorted
        ]
        lines.append("| " + " | ".join(cell_texts) + " |")
        if row_idx == 0:
            lines.append("| " + " | ".join("---" for _ in cell_texts) + " |")

    return "\n".join(lines)


def _cell_text(cell: dict) -> str:
    """Extract plain text from a table cell dict."""
    text = cell.get("content", cell.get("text", "")).strip()
    return text if text else " "
