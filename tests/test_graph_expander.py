"""TDD tests for src/graph/expander.py.

All tests fully offline.
"""
from __future__ import annotations

import pytest

from src.models import Chunk
from src.graph.builder import build_graph
from src.graph.expander import expand_chunks

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    pytest.skip("networkx not installed", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(text: str, page: int = 1) -> Chunk:
    return Chunk(
        text=text,
        page_number=page,
        bboxes=[[0, 0, 100, 20]],
        chunk_type="text",
        confidence=1.0,
        image_path="",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExpandChunks:
    def test_empty_seeds_returns_empty(self) -> None:
        g = build_graph([])
        result = expand_chunks([], g, [])
        assert result == []

    def test_empty_graph_returns_seeds_unchanged(self) -> None:
        chunks = [_chunk("a"), _chunk("b")]
        empty_g = nx.DiGraph()
        result = expand_chunks([chunks[0]], empty_g, chunks)
        assert result == [chunks[0]]

    def test_seeds_always_come_first(self) -> None:
        chunks = [
            _chunk("§1 Titel\nText"),
            _chunk("Inhalt Abschnitt 1."),
            _chunk("§2 Titel\nText"),
        ]
        g = build_graph(chunks)
        result = expand_chunks([chunks[1]], g, chunks)
        assert result[0] == chunks[1]

    def test_sibling_chunks_added_via_chunk_of(self) -> None:
        """Two chunks under same section: seeding one should expand to the other."""
        chunks = [
            _chunk("§1 Einleitung\nText"),  # chunk_0 -> sec_1
            _chunk("Weitere Infos."),        # chunk_1 -> sec_1 (same section)
        ]
        g = build_graph(chunks)
        result = expand_chunks([chunks[0]], g, chunks)
        assert chunks[1] in result

    def test_cross_ref_expansion(self) -> None:
        """Chunk with §-ref should pull in chunks from referenced section."""
        chunks = [
            _chunk("§14 Kündigung\nRegelungen zur Kündigung."),  # chunk_0 sec_14
            _chunk("Gemäß §14 beträgt die Frist 30 Tage."),        # chunk_1 refs sec_14
        ]
        g = build_graph(chunks)
        result = expand_chunks([chunks[1]], g, chunks)
        assert chunks[0] in result

    def test_sequential_expansion(self) -> None:
        """Sequential neighbour on same page is included."""
        chunks = [_chunk("first", page=2), _chunk("second", page=2), _chunk("third", page=3)]
        g = build_graph(chunks)
        result = expand_chunks([chunks[0]], g, chunks)
        assert chunks[1] in result
        assert chunks[2] not in result  # different page, no sequential edge

    def test_no_duplicates_in_result(self) -> None:
        chunks = [
            _chunk("§1 Titel\nText"),
            _chunk("Text."),
        ]
        g = build_graph(chunks)
        result = expand_chunks([chunks[0], chunks[1]], g, chunks)
        assert len(result) == len(set(id(c) for c in result))

    def test_max_expanded_cap_respected(self) -> None:
        chunks = [_chunk(f"Text {i}", page=1) for i in range(30)]
        g = build_graph(chunks)
        result = expand_chunks(chunks[:2], g, chunks, max_expanded=5)
        assert len(result) <= 5

    def test_chunk_not_in_graph_skipped_gracefully(self) -> None:
        chunks = [_chunk("orphan chunk not in graph")]
        g = nx.DiGraph()  # empty graph
        result = expand_chunks(chunks, g, chunks)
        assert result == chunks
