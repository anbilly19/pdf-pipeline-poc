"""Tests for extraction layer."""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

from src.extraction.base import BaseExtractor
from src.models import Element, ElementType, Page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_page(page_number: int = 1, elements: list[Element] | None = None) -> Page:
    return Page(
        page_number=page_number,
        width=595.0,
        height=842.0,
        elements=elements or [],
    )


def _make_element(
    text: str = "Hello",
    bbox: list[float] | None = None,
    confidence: float = 0.95,
) -> Element:
    return Element(
        text=text,
        bbox=bbox or [10.0, 20.0, 200.0, 40.0],
        element_type=ElementType.TEXT,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# BaseExtractor (abstract)
# ---------------------------------------------------------------------------

def test_base_extractor_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseExtractor()  # type: ignore[abstract]


def test_extract_safe_returns_empty_on_error() -> None:
    class BrokenExtractor(BaseExtractor):
        def extract(self, pdf_path: object) -> list[Page]:  # type: ignore[override]
            raise RuntimeError("broken")

    extractor = BrokenExtractor()
    result = extractor.extract_safe("nonexistent.pdf")  # type: ignore[arg-type]
    assert result == []


# ---------------------------------------------------------------------------
# Element validation
# ---------------------------------------------------------------------------

def test_element_invalid_bbox_length() -> None:
    with pytest.raises((ValueError, TypeError)):
        Element(
            text="x",
            bbox=[0.0, 0.0, 10.0],  # only 3 values
            element_type=ElementType.TEXT,
            confidence=0.9,
        )


def test_element_confidence_out_of_range() -> None:
    with pytest.raises((ValueError, TypeError)):
        Element(
            text="x",
            bbox=[0.0, 0.0, 10.0, 20.0],
            element_type=ElementType.TEXT,
            confidence=1.5,  # > 1.0
        )


def test_element_valid() -> None:
    el = _make_element()
    assert el.text == "Hello"
    assert len(el.bbox) == 4


# ---------------------------------------------------------------------------
# PyMuPDFExtractor (mocked fitz)
# ---------------------------------------------------------------------------

def test_pymupdf_extractor_skips_short_spans() -> None:
    """Spans shorter than 3 chars should be dropped."""
    with patch("src.extraction.pymupdf_extractor.fitz") as mock_fitz:
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_page.number = 0
        mock_page.rect.width = 595.0
        mock_page.rect.height = 842.0
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "spans": [
                                {"text": "Hi", "bbox": [0, 0, 20, 10], "flags": 0},   # too short
                                {"text": "Hello world", "bbox": [0, 20, 100, 30], "flags": 0},
                            ]
                        }
                    ],
                }
            ]
        }
        mock_fitz.open.return_value.__enter__ = lambda s: mock_doc
        mock_fitz.open.return_value.__exit__ = MagicMock(return_value=False)

        from src.extraction.pymupdf_extractor import PyMuPDFExtractor
        extractor = PyMuPDFExtractor()
        pages = extractor.extract("dummy.pdf")  # type: ignore[arg-type]

        texts = [el.text for p in pages for el in p.elements]
        assert "Hi" not in texts
        assert "Hello world" in texts
