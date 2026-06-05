"""Tests for the layout-aware chunker."""
from src.chunking.chunker import ChunkerConfig, LayoutChunker
from src.models import Element, Page

# min_chunk_chars set to 1 so short synthetic test strings are never filtered
_CFG = ChunkerConfig(min_chunk_chars=1)


def _make_page(elements: list[Element], page_number: int = 1) -> Page:
    return Page(page_number=page_number, image_path="", elements=elements)


def _text(t: str, bbox: list[float] | None = None) -> Element:
    return Element(
        type="text",
        text=t,
        bbox=bbox or [0.0, 0.0, 100.0, 20.0],
        confidence=0.9,
    )


def _table(t: str) -> Element:
    return Element(type="table", text=t, bbox=[0.0, 50.0, 200.0, 100.0], confidence=0.85)


def test_basic_chunking() -> None:
    chunker = LayoutChunker(config=_CFG)
    page = _make_page([_text("Erster Absatz."), _text("Zweiter Absatz.")])
    chunks = chunker.chunk([page])
    assert len(chunks) >= 1
    assert all(len(c.bboxes) > 0 for c in chunks)


def test_table_is_own_chunk() -> None:
    chunker = LayoutChunker(config=_CFG)
    page = _make_page([
        _text("Vor der Tabelle."),
        _table("| A | B |\n| 1 | 2 |"),
        _text("Nach der Tabelle."),
    ])
    chunks = chunker.chunk([page])
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0].text == "| A | B |\n| 1 | 2 |"


def test_bboxes_aggregated() -> None:
    chunker = LayoutChunker(config=_CFG)
    elems = [_text("A", [0.0, 0.0, 50.0, 10.0]), _text("B", [0.0, 15.0, 50.0, 25.0])]
    page = _make_page(elems)
    chunks = chunker.chunk([page])
    combined_bboxes = [bbox for c in chunks for bbox in c.bboxes]
    assert [0.0, 0.0, 50.0, 10.0] in combined_bboxes
    assert [0.0, 15.0, 50.0, 25.0] in combined_bboxes


def test_max_chars_split() -> None:
    chunker = LayoutChunker(config=ChunkerConfig(max_chars=20, min_chunk_chars=1))
    page = _make_page([
        _text("Kurzer Text."),
        _text("Noch ein langer Text der ueber das Limit geht."),
    ])
    chunks = chunker.chunk([page])
    assert len(chunks) >= 2


def test_heading_starts_new_chunk() -> None:
    """A numbered section heading element must flush the current buffer and start a new chunk."""
    chunker = LayoutChunker(config=_CFG)
    elems = [
        _text("Einleitung text hier."),
        _text("15 Laufzeit und Kuendigung"),  # numbered heading matched by _HEADING_RE
        _text("Inhalt der neuen Sektion."),
    ]
    page = _make_page(elems)
    chunks = chunker.chunk([page])
    assert len(chunks) >= 2


def test_confidence_is_minimum() -> None:
    chunker = LayoutChunker(config=_CFG)
    page = _make_page([
        Element(type="text", text="High conf", bbox=[0, 0, 50, 10], confidence=0.95),
        Element(type="text", text="Low conf", bbox=[0, 15, 50, 25], confidence=0.60),
    ])
    chunks = chunker.chunk([page])
    assert chunks[0].confidence == 0.60


def test_page_number_preserved() -> None:
    chunker = LayoutChunker(config=_CFG)
    pages = [
        _make_page([_text("Seite eins.")], page_number=1),
        _make_page([_text("Seite zwei.")], page_number=2),
    ]
    chunks = chunker.chunk(pages)
    page_numbers = {c.page_number for c in chunks}
    assert page_numbers == {1, 2}
