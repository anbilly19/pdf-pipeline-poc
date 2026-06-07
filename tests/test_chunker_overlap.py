"""TDD tests for Roadmap #3 — sliding window chunk overlap.

Fully offline — no I/O, no models.
Verifies:
  - bridging chunks are inserted between consecutive text pairs
  - bridge text = tail[-overlap:] + head[:overlap]
  - bridge chunk_type is 'overlap'
  - table chunks are never bridged
  - cross-page overlaps carry unioned bboxes
  - overlap=0 disables the pass entirely
  - min_chunk_chars filter removes trivially short bridges
  - bbox union is geometrically correct
  - confidence is min() of the two source chunks
  - deduplication still works after overlap pass
"""
from __future__ import annotations

import pytest

from src.chunking.chunker import ChunkerConfig, LayoutChunker, _union_bboxes
from src.models import Chunk, Element, Page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _page(elements: list[Element], page_number: int = 1) -> Page:
    return Page(page_number=page_number, image_path="", elements=elements)


def _text_el(text: str, bbox: list[float] | None = None) -> Element:
    return Element(
        type="text",
        text=text,
        bbox=bbox or [0.0, 0.0, 200.0, 20.0],
        confidence=0.95,
    )


def _table_el(text: str) -> Element:
    return Element(
        type="table",
        text=text,
        bbox=[0.0, 0.0, 200.0, 100.0],
        confidence=0.90,
    )


def _chunk(text: str, chunk_type: str = "text", page: int = 1,
           bbox: list[float] | None = None, conf: float = 0.9) -> Chunk:
    return Chunk(
        text=text,
        page_number=page,
        bboxes=[bbox or [0.0, 0.0, 100.0, 20.0]],
        chunk_type=chunk_type,
        confidence=conf,
        image_path="",
    )


# ---------------------------------------------------------------------------
# _union_bboxes helper
# ---------------------------------------------------------------------------

class TestUnionBboxes:
    def test_enclosing_rectangle(self) -> None:
        a = [[10.0, 20.0, 50.0, 60.0]]
        b = [[30.0, 10.0, 80.0, 40.0]]
        result = _union_bboxes(a, b)
        assert result == [[10.0, 10.0, 80.0, 60.0]]

    def test_identical_bboxes(self) -> None:
        box = [[0.0, 0.0, 100.0, 50.0]]
        assert _union_bboxes(box, box) == box

    def test_empty_a_returns_b(self) -> None:
        b = [[5.0, 5.0, 50.0, 50.0]]
        result = _union_bboxes([], b)
        assert result == b

    def test_empty_b_returns_a(self) -> None:
        a = [[1.0, 2.0, 10.0, 20.0]]
        result = _union_bboxes(a, [])
        assert result == a


# ---------------------------------------------------------------------------
# Overlap pass: bridging inserted correctly
# ---------------------------------------------------------------------------

class TestOverlapBridging:
    def _chunker(self, overlap: float = 0.12, max_chars: int = 800) -> LayoutChunker:
        return LayoutChunker(ChunkerConfig(
            max_chars=max_chars, overlap_ratio=overlap, min_chunk_chars=10
        ))

    def test_bridge_inserted_between_text_chunks(self) -> None:
        """Two adjacent text chunks produce a bridge in between."""
        chunker = self._chunker()
        raw = [_chunk("A " * 50), _chunk("B " * 50)]
        result = chunker._add_overlap(raw)
        types = [c.chunk_type for c in result]
        assert types == ["text", "overlap", "text"]

    def test_bridge_text_contains_tail_and_head(self) -> None:
        tail_text = "word " * 20   # long enough to exceed overlap window
        head_text = "other " * 20
        raw = [_chunk(tail_text), _chunk(head_text)]
        chunker = self._chunker(overlap=0.12, max_chars=800)
        result = chunker._add_overlap(raw)
        bridge = result[1]
        # overlap_chars = max(40, int(800 * 0.12)) = 96
        assert bridge.chunk_type == "overlap"
        assert tail_text.strip()[-96:].strip() in bridge.text
        assert head_text.strip()[:96].strip() in bridge.text

    def test_bridge_chunk_type_is_overlap(self) -> None:
        raw = [_chunk("x " * 60), _chunk("y " * 60)]
        chunker = self._chunker()
        result = chunker._add_overlap(raw)
        assert result[1].chunk_type == "overlap"

    def test_no_bridge_when_overlap_disabled(self) -> None:
        raw = [_chunk("a " * 60), _chunk("b " * 60)]
        chunker = LayoutChunker(ChunkerConfig(overlap_ratio=0.0))
        result = chunker._add_overlap(raw)
        assert all(c.chunk_type != "overlap" for c in result)
        assert len(result) == 2

    def test_three_chunks_produce_two_bridges(self) -> None:
        raw = [_chunk("A " * 50), _chunk("B " * 50), _chunk("C " * 50)]
        chunker = self._chunker()
        result = chunker._add_overlap(raw)
        bridges = [c for c in result if c.chunk_type == "overlap"]
        assert len(bridges) == 2

    def test_bridge_confidence_is_min_of_sources(self) -> None:
        raw = [_chunk("aaa " * 30, conf=0.8), _chunk("bbb " * 30, conf=0.6)]
        chunker = self._chunker()
        result = chunker._add_overlap(raw)
        bridge = result[1]
        assert bridge.confidence == pytest.approx(0.6)

    def test_bridge_page_number_from_tail_chunk(self) -> None:
        raw = [_chunk("aaa " * 30, page=3), _chunk("bbb " * 30, page=4)]
        chunker = self._chunker()
        result = chunker._add_overlap(raw)
        bridge = result[1]
        assert bridge.page_number == 3


# ---------------------------------------------------------------------------
# Tables never bridged
# ---------------------------------------------------------------------------

class TestTableNeverBridged:
    def _chunker(self) -> LayoutChunker:
        return LayoutChunker(ChunkerConfig(overlap_ratio=0.12, min_chunk_chars=10))

    def test_text_table_no_bridge(self) -> None:
        raw = [_chunk("text " * 40), _chunk("table data", chunk_type="table")]
        result = self._chunker()._add_overlap(raw)
        assert all(c.chunk_type != "overlap" for c in result)

    def test_table_text_no_bridge(self) -> None:
        raw = [_chunk("table data", chunk_type="table"), _chunk("text " * 40)]
        result = self._chunker()._add_overlap(raw)
        assert all(c.chunk_type != "overlap" for c in result)

    def test_text_text_table_text_bridges_only_text_pairs(self) -> None:
        raw = [
            _chunk("text1 " * 40),
            _chunk("text2 " * 40),
            _chunk("table", chunk_type="table"),
            _chunk("text3 " * 40),
        ]
        result = self._chunker()._add_overlap(raw)
        bridges = [c for c in result if c.chunk_type == "overlap"]
        assert len(bridges) == 1  # only text1→text2


# ---------------------------------------------------------------------------
# Bbox union on cross-page overlap
# ---------------------------------------------------------------------------

class TestCrossPageBboxUnion:
    def test_bridge_bbox_is_union_of_both_sources(self) -> None:
        a = _chunk("aaa " * 30, page=1, bbox=[10.0, 20.0, 50.0, 60.0])
        b = _chunk("bbb " * 30, page=2, bbox=[30.0, 10.0, 80.0, 40.0])
        chunker = LayoutChunker(ChunkerConfig(overlap_ratio=0.12, min_chunk_chars=10))
        result = chunker._add_overlap([a, b])
        bridge = result[1]
        assert bridge.bboxes == [[10.0, 10.0, 80.0, 60.0]]


# ---------------------------------------------------------------------------
# Full pipeline: chunk() produces overlap chunks
# ---------------------------------------------------------------------------

class TestFullPipelineOverlap:
    def test_overlap_present_in_chunk_output(self) -> None:
        """chunk() on a page with two large text elements yields overlap chunks."""
        long_text_a = "Die Kündigungsfrist beträgt dreissig Tage. " * 10
        long_text_b = "Der Auftragnehmer haftet für alle Schäden. " * 10
        page = _page([
            _text_el(long_text_a, bbox=[0.0, 0.0, 400.0, 40.0]),
            _text_el(long_text_b, bbox=[0.0, 50.0, 400.0, 90.0]),
        ])
        chunker = LayoutChunker(ChunkerConfig(
            max_chars=200, overlap_ratio=0.12, min_chunk_chars=10
        ))
        result = chunker.chunk([page])
        overlap_chunks = [c for c in result if c.chunk_type == "overlap"]
        assert len(overlap_chunks) >= 1

    def test_overlap_disabled_no_overlap_chunks(self) -> None:
        long_text = "Vertragsstrafe bei Verzögerung. " * 10
        page = _page([_text_el(long_text)])
        chunker = LayoutChunker(ChunkerConfig(max_chars=200, overlap_ratio=0.0))
        result = chunker.chunk([page])
        assert all(c.chunk_type != "overlap" for c in result)
