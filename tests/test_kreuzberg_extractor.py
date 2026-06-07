"""TDD tests for KreuzbergExtractor — Roadmap #4.

Fully offline — the kreuzberg library is mocked throughout.
Verifies:
  - y-axis flip: bottom-left -> top-left coordinate normalisation
  - element type mapping (text / table / image)
  - filtering of too-short text and zero-area bboxes
  - ExtractionError raised when file not found
  - ExtractionError raised when kreuzberg import fails
  - ExtractionError raised when kreuzberg.extract_file raises
  - KreuzbergExtractor promoted to head of chain in ExtractionRouter
  - router degrades gracefully when Kreuzberg is unavailable
  - bbox contract: all coords are non-negative after y-flip
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.extraction.base import ExtractionError


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

def _make_bbox(x0: float, y0: float, x1: float, y1: float) -> MagicMock:
    bb = MagicMock()
    bb.x0 = x0
    bb.y0 = y0
    bb.x1 = x1
    bb.y1 = y1
    # Also support index access for robustness
    bb.__getitem__ = lambda s, i: [x0, y0, x1, y1][i]
    return bb


def _make_block(
    text: str,
    bbox: object,
    block_type: str = "text",
) -> MagicMock:
    b = MagicMock()
    b.text = text
    b.bbox = bbox
    b.type = block_type
    return b


def _make_page(
    blocks: list,
    page_number: int = 1,
    height: float = 842.0,
    image_path: str = "",
) -> MagicMock:
    p = MagicMock()
    p.number = page_number
    p.height = height
    p.image_path = image_path
    p.blocks = blocks
    return p


def _make_document(pages: list) -> MagicMock:
    doc = MagicMock()
    doc.pages = pages
    return doc


def _kreuzberg_module(document: MagicMock) -> types.ModuleType:
    """Build a fake kreuzberg module that returns the given document."""
    mod = types.ModuleType("kreuzberg")
    mod.extract_file = MagicMock(return_value=document)  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# y-axis normalisation
# ---------------------------------------------------------------------------

class TestYAxisFlip:
    """Kreuzberg uses bottom-left origin; we must flip to top-left."""

    def test_y_flip_basic(self, tmp_path: Path) -> None:
        """y0_norm = page_height - y1_raw; y1_norm = page_height - y0_raw."""
        page_height = 842.0
        # raw bbox in bottom-left coords: x0=10, y0=100, x1=200, y1=200
        # expected top-left: x0=10, y0=842-200=642, x1=200, y1=842-100=742
        block = _make_block("Hallo Welt", _make_bbox(10, 100, 200, 200))
        page = _make_page([block], height=page_height)
        doc = _make_document([page])

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        mod = _kreuzberg_module(doc)
        with patch.dict(sys.modules, {"kreuzberg": mod}):
            from src.extraction.kreuzberg_extractor import KreuzbergExtractor
            extractor = KreuzbergExtractor()
            pages = extractor.extract(pdf)

        assert len(pages) == 1
        el = pages[0].elements[0]
        assert el.bbox == pytest.approx([10.0, 642.0, 200.0, 742.0])

    def test_all_coords_non_negative(self, tmp_path: Path) -> None:
        """After y-flip all bbox coordinates must be >= 0."""
        page_height = 500.0
        # Edge case: block at the very top of the page in bottom-left coords
        block = _make_block("Top block", _make_bbox(0, 450, 300, 500))
        page = _make_page([block], height=page_height)
        doc = _make_document([page])
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        mod = _kreuzberg_module(doc)
        with patch.dict(sys.modules, {"kreuzberg": mod}):
            from src.extraction.kreuzberg_extractor import KreuzbergExtractor
            pages = KreuzbergExtractor().extract(pdf)
        el = pages[0].elements[0]
        assert all(v >= 0 for v in el.bbox)


# ---------------------------------------------------------------------------
# Element type mapping
# ---------------------------------------------------------------------------

class TestElementTypeMapping:
    def _extract_single(self, tmp_path: Path, block_type: str) -> str:
        block = _make_block("Content", _make_bbox(0, 0, 100, 50), block_type=block_type)
        doc = _make_document([_make_page([block])])
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        mod = _kreuzberg_module(doc)
        with patch.dict(sys.modules, {"kreuzberg": mod}):
            from src.extraction.kreuzberg_extractor import KreuzbergExtractor
            pages = KreuzbergExtractor().extract(pdf)
        return pages[0].elements[0].type

    def test_text_type_mapped(self, tmp_path: Path) -> None:
        assert self._extract_single(tmp_path, "text") == "text"

    def test_table_type_mapped(self, tmp_path: Path) -> None:
        assert self._extract_single(tmp_path, "table") == "table"

    def test_image_type_mapped(self, tmp_path: Path) -> None:
        assert self._extract_single(tmp_path, "figure") == "image"


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestFiltering:
    def test_short_text_filtered(self, tmp_path: Path) -> None:
        """Blocks with text shorter than _MIN_TEXT_LEN are discarded."""
        block = _make_block("x", _make_bbox(0, 0, 100, 50))  # len=1 < 2
        doc = _make_document([_make_page([block])])
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        mod = _kreuzberg_module(doc)
        with patch.dict(sys.modules, {"kreuzberg": mod}):
            from src.extraction.kreuzberg_extractor import KreuzbergExtractor
            pages = KreuzbergExtractor().extract(pdf)
        assert pages[0].elements == []

    def test_zero_area_bbox_filtered(self, tmp_path: Path) -> None:
        """Blocks with a degenerate bbox (area < threshold) are discarded."""
        block = _make_block("Some text", _make_bbox(10, 10, 10, 10))  # zero area
        doc = _make_document([_make_page([block])])
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        mod = _kreuzberg_module(doc)
        with patch.dict(sys.modules, {"kreuzberg": mod}):
            from src.extraction.kreuzberg_extractor import KreuzbergExtractor
            pages = KreuzbergExtractor().extract(pdf)
        assert pages[0].elements == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_file_not_found_raises(self) -> None:
        mod = types.ModuleType("kreuzberg")
        mod.extract_file = MagicMock()  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"kreuzberg": mod}):
            from src.extraction.kreuzberg_extractor import KreuzbergExtractor
            with pytest.raises(ExtractionError, match="not found"):
                KreuzbergExtractor().extract(Path("/nonexistent/file.pdf"))

    def test_kreuzberg_import_error_raises_extraction_error(self, tmp_path: Path) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        # Remove kreuzberg from sys.modules to simulate not installed
        with patch.dict(sys.modules, {"kreuzberg": None}):  # type: ignore[dict-item]
            from src.extraction.kreuzberg_extractor import KreuzbergExtractor
            with pytest.raises(ExtractionError, match="not installed"):
                KreuzbergExtractor().extract(pdf)

    def test_kreuzberg_parse_error_raises_extraction_error(self, tmp_path: Path) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        mod = types.ModuleType("kreuzberg")
        mod.extract_file = MagicMock(side_effect=RuntimeError("corrupt pdf"))  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"kreuzberg": mod}):
            from src.extraction.kreuzberg_extractor import KreuzbergExtractor
            with pytest.raises(ExtractionError, match="corrupt pdf"):
                KreuzbergExtractor().extract(pdf)


# ---------------------------------------------------------------------------
# Router integration
# ---------------------------------------------------------------------------

class TestRouterWithKreuzberg:
    def test_kreuzberg_at_head_of_chain(self) -> None:
        """When Kreuzberg is available it should be first in the chain."""
        # Provide a minimal mock kreuzberg that won’t fail on import
        mod = types.ModuleType("kreuzberg")
        mod.extract_file = MagicMock()  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"kreuzberg": mod}):
            # Force fresh import of router after patching
            import importlib
            import src.extraction.router as router_mod
            importlib.reload(router_mod)
            router = router_mod.ExtractionRouter()
            assert router._chain[0][1] == "Kreuzberg"

    def test_router_degrades_when_kreuzberg_missing(self) -> None:
        """When kreuzberg is absent the chain should still work (PyMuPDF present)."""
        with patch.dict(sys.modules, {"kreuzberg": None}):  # type: ignore[dict-item]
            import importlib
            import src.extraction.kreuzberg_extractor as kx_mod
            import src.extraction.router as router_mod
            importlib.reload(kx_mod)
            importlib.reload(router_mod)
            router = router_mod.ExtractionRouter()
            # PyMuPDF must always be present
            names = [n for _, n in router._chain]
            assert "PyMuPDF" in names
