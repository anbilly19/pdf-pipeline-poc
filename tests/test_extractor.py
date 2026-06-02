"""Tests for extraction layer."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.extraction.base import BaseExtractor, ExtractionError
from src.models import Element, Page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_page(page_number: int = 1, elements: list[Element] | None = None) -> Page:
    return Page(
        page_number=page_number,
        image_path="",
        elements=elements or [],
    )


def _make_element(
    text: str = "Hello",
    bbox: list[float] | None = None,
    confidence: float = 0.95,
    element_type: str = "text",
) -> Element:
    return Element(
        type=element_type,
        text=text,
        bbox=bbox or [10.0, 20.0, 200.0, 40.0],
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# BaseExtractor (abstract)
# ---------------------------------------------------------------------------

def test_base_extractor_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseExtractor()  # type: ignore[abstract]


def test_extract_safe_returns_none_on_error() -> None:
    """extract_safe must return None (not raise) when extraction fails."""
    class BrokenExtractor(BaseExtractor):
        def extract(self, pdf_path: object) -> list[Page]:  # type: ignore[override]
            raise ExtractionError("broken")

    extractor = BrokenExtractor()
    result = extractor.extract_safe(Path("nonexistent.pdf"))
    assert result is None


# ---------------------------------------------------------------------------
# Element validation
# ---------------------------------------------------------------------------

def test_element_invalid_bbox_length() -> None:
    with pytest.raises(ValueError):
        Element(
            type="text",
            text="x",
            bbox=[0.0, 0.0, 10.0],  # only 3 values
            confidence=0.9,
        )


def test_element_confidence_out_of_range() -> None:
    with pytest.raises(ValueError):
        Element(
            type="text",
            text="x",
            bbox=[0.0, 0.0, 10.0, 20.0],
            confidence=1.5,  # > 1.0
        )


def test_element_valid() -> None:
    el = _make_element()
    assert el.text == "Hello"
    assert len(el.bbox) == 4


def test_element_type_table() -> None:
    el = _make_element(element_type="table")
    assert el.type == "table"


# ---------------------------------------------------------------------------
# Page model
# ---------------------------------------------------------------------------

def test_page_empty_elements() -> None:
    page = _make_page()
    assert page.elements == []
    assert page.page_number == 1
    assert page.image_path == ""


def test_page_with_elements() -> None:
    elements = [_make_element(text=f"Element {i}") for i in range(3)]
    page = _make_page(elements=elements)
    assert len(page.elements) == 3


# ---------------------------------------------------------------------------
# PyMuPDFExtractor (mocked fitz)
# ---------------------------------------------------------------------------

def test_pymupdf_extractor_missing_file_raises() -> None:
    """Extracting a non-existent PDF must raise ExtractionError."""
    from src.extraction.pymupdf_extractor import PyMuPDFExtractor
    extractor = PyMuPDFExtractor()
    with pytest.raises(ExtractionError):
        extractor.extract(Path("does_not_exist.pdf"))


def test_pymupdf_extractor_skips_short_spans() -> None:
    """Text blocks shorter than 3 chars should be dropped."""
    from src.extraction.pymupdf_extractor import PyMuPDFExtractor

    with patch("src.extraction.pymupdf_extractor.fitz") as mock_fitz:
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.close = MagicMock()
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "bbox": [0, 0, 100, 30],
                    "lines": [
                        {
                            "spans": [
                                {"text": "Hi"},          # too short
                                {"text": "Hello world"},
                            ]
                        }
                    ],
                }
            ]
        }
        mock_fitz.open.return_value = mock_doc
        mock_fitz.TEXT_PRESERVE_WHITESPACE = 0

        # Patch exists() so the path check passes
        with patch.object(Path, "exists", return_value=True):
            extractor = PyMuPDFExtractor()
            pages = extractor.extract(Path("dummy.pdf"))

        texts = [el.text for p in pages for el in p.elements]
        assert "Hello world" in texts
