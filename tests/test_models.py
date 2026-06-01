"""Tests for core data models."""
from src.models import Chunk, Element, Page, QAResponse, Source


def test_element_creation() -> None:
    el = Element(type="text", text="Hallo Welt", bbox=[0.0, 0.0, 100.0, 20.0], confidence=0.95)
    assert el.bbox == [0.0, 0.0, 100.0, 20.0]
    assert el.type == "text"


def test_page_creation() -> None:
    page = Page(page_number=1, image_path="/tmp/page1.png")
    assert page.elements == []


def test_chunk_carries_bboxes() -> None:
    chunk = Chunk(
        text="Test",
        page_number=1,
        bboxes=[[0.0, 0.0, 100.0, 20.0]],
        chunk_type="text",
        confidence=0.9,
        image_path="/tmp/page1.png",
    )
    assert len(chunk.bboxes) == 1


def test_qa_response_structure() -> None:
    source = Source(text="Test", page=1, bboxes=[[0, 0, 10, 10]], image="/tmp/p1.png")
    response = QAResponse(answer="Antwort", sources=[source])
    assert response.sources[0].page == 1
