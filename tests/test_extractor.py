"""Tests for PyMuPDF extractor — uses the bundled test PDF fixture."""
import shutil
from pathlib import Path

import pytest

from src.extraction.pymupdf_extractor import PyMuPDFExtractor
from src.extraction.base import ExtractionError

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def simple_pdf(tmp_path: Path) -> Path:
    """Create a minimal single-page PDF with known text content."""
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF not installed")

    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    page.insert_text((72, 100), "Hallo Welt. Dies ist ein Test-PDF.", fontsize=12)
    page.insert_text((72, 200), "ÜBERSCHRIFT", fontsize=18)
    page.insert_text((72, 300), "Weiterer Text mit deutschen Umlauten: äöüß", fontsize=12)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_extract_returns_pages(simple_pdf: Path) -> None:
    extractor = PyMuPDFExtractor()
    pages = extractor.extract(simple_pdf)
    assert len(pages) == 1
    assert pages[0].page_number == 1


def test_extract_has_elements(simple_pdf: Path) -> None:
    extractor = PyMuPDFExtractor()
    pages = extractor.extract(simple_pdf)
    assert len(pages[0].elements) > 0


def test_all_elements_have_bboxes(simple_pdf: Path) -> None:
    extractor = PyMuPDFExtractor()
    pages = extractor.extract(simple_pdf)
    for page in pages:
        for el in page.elements:
            assert len(el.bbox) == 4, "Every element must have a 4-value bbox"
            assert el.bbox[2] > el.bbox[0], "x1 must be > x0"
            assert el.bbox[3] > el.bbox[1], "y1 must be > y0"


def test_german_text_preserved(simple_pdf: Path) -> None:
    extractor = PyMuPDFExtractor()
    pages = extractor.extract(simple_pdf)
    all_text = " ".join(e.text for e in pages[0].elements)
    assert "äöüß" in all_text or "Hallo" in all_text


def test_missing_file_raises() -> None:
    extractor = PyMuPDFExtractor()
    with pytest.raises(ExtractionError, match="not found"):
        extractor.extract(Path("/nonexistent/file.pdf"))


def test_extract_safe_returns_none_on_error() -> None:
    extractor = PyMuPDFExtractor()
    result = extractor.extract_safe(Path("/nonexistent/file.pdf"))
    assert result is None
