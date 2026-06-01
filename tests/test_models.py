"""Tests for core data models."""
import pytest

from src.models import Chunk, Element, Page, QAResponse, Source


def test_element_valid() -> None:
    el = Element(type="text", text="Hallo Welt", bbox=[0.0, 0.0, 100.0, 20.0], confidence=0.95)
    assert el.bbox == [0.0, 0.0, 100.0, 20.0]
    assert el.type == "text"


def test_element_invalid_bbox() -> None:
    with pytest.raises(ValueError, match="bbox must have exactly 4"):
        Element(type="text", text="x", bbox=[0.0, 0.0], confidence=0.9)


def test_element_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence must be in"):
        Element(type="text", text="x", bbox=[0.0, 0.0, 1.0, 1.0], confidence=1.5)


def test_page_defaults() -> None:
    page = Page(page_number=1, image_path="")
    assert page.elements == []


def test_chunk_carries_bboxes() -> None:
    chunk = Chunk(
        text="Test",
        page_number=1,
        bboxes=[[0.0, 0.0, 100.0, 20.0], [0.0, 25.0, 100.0, 45.0]],
        chunk_type="text",
        confidence=0.9,
        image_path="",
    )
    assert len(chunk.bboxes) == 2
    assert chunk.bboxes[0] == [0.0, 0.0, 100.0, 20.0]


def test_qa_response_structure() -> None:
    source = Source(text="Antwort", page=1, bboxes=[[0, 0, 10, 10]], image="")
    response = QAResponse(answer="42", sources=[source])
    assert response.sources[0].page == 1
    assert response.answer == "42"
