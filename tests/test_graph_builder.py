"""TDD tests for src/graph/builder.py.

All tests fully offline — no PDF, no FAISS, no LLM.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.models import Chunk
from src.graph.builder import (
    build_graph,
    save_graph,
    load_graph,
    _extract_section_label,
    _extract_cross_refs,
    _section_level,
    _parent_label,
)


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
# Unit tests: pure helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_section_level_top(self) -> None:
        assert _section_level("10") == 1

    def test_section_level_sub(self) -> None:
        assert _section_level("10.1") == 2

    def test_section_level_deep(self) -> None:
        assert _section_level("10.1.2") == 3

    def test_parent_label_returns_parent(self) -> None:
        assert _parent_label("10.1") == "10"

    def test_parent_label_top_level_returns_none(self) -> None:
        assert _parent_label("10") is None

    def test_parent_label_deep(self) -> None:
        assert _parent_label("10.1.2") == "10.1"

    def test_extract_section_label_paragraph_sign(self) -> None:
        assert _extract_section_label("§10 Vertragsstrafe\n") == "10"

    def test_extract_section_label_numeric(self) -> None:
        assert _extract_section_label("10.1 Fristen\n") == "10.1"

    def test_extract_section_label_no_heading(self) -> None:
        assert _extract_section_label("Dieser Vertrag regelt...") is None

    def test_extract_cross_refs_single(self) -> None:
        assert _extract_cross_refs("gemäß §14.3 gilt") == ["14.3"]

    def test_extract_cross_refs_multiple(self) -> None:
        refs = _extract_cross_refs("§10 und §11.1 und §12")
        assert refs == ["10", "11.1", "12"]

    def test_extract_cross_refs_none(self) -> None:
        assert _extract_cross_refs("kein Verweis hier") == []


# ---------------------------------------------------------------------------
# build_graph: node structure
# ---------------------------------------------------------------------------

class TestBuildGraphNodes:
    def test_empty_chunks_returns_empty_graph(self) -> None:
        g = build_graph([])
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0

    def test_chunk_nodes_created(self) -> None:
        chunks = [_chunk("some text"), _chunk("more text")]
        g = build_graph(chunks)
        assert g.has_node("chunk_0")
        assert g.has_node("chunk_1")

    def test_chunk_node_attributes(self) -> None:
        chunks = [_chunk("hello world", page=3)]
        g = build_graph(chunks)
        data = g.nodes["chunk_0"]
        assert data["node_type"] == "chunk"
        assert data["page"] == 3
        assert data["chunk_index"] == 0

    def test_section_node_created_for_heading_chunk(self) -> None:
        chunks = [_chunk("§10 Vertragsstrafe\nDieser Abschnitt regelt...")]
        g = build_graph(chunks)
        assert g.has_node("sec_10")

    def test_section_node_attributes(self) -> None:
        chunks = [_chunk("§10.1 Fristen\nDie Fristen betragen...")]
        g = build_graph(chunks)
        data = g.nodes["sec_10.1"]
        assert data["node_type"] == "section"
        assert data["label"] == "10.1"
        assert data["level"] == 2

    def test_no_duplicate_section_node(self) -> None:
        chunks = [
            _chunk("§10 Titel\nText"),
            _chunk("§10 Titel\nMehr Text"),  # same section, second chunk
        ]
        g = build_graph(chunks)
        section_nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == "section"]
        assert len(section_nodes) == 1


# ---------------------------------------------------------------------------
# build_graph: edge structure
# ---------------------------------------------------------------------------

class TestBuildGraphEdges:
    def test_chunk_of_edge_added(self) -> None:
        chunks = [
            _chunk("§10 Vertragsstrafe\nText"),
            _chunk("Weitergehende Regelungen."),
        ]
        g = build_graph(chunks)
        assert g.has_edge("chunk_0", "sec_10")
        assert g.has_edge("chunk_1", "sec_10")
        assert g.edges["chunk_0", "sec_10"]["edge_type"] == "chunk_of"

    def test_subsection_of_edge_added(self) -> None:
        chunks = [
            _chunk("§10 Titel\nText"),
            _chunk("§10.1 Untertitel\nText"),
        ]
        g = build_graph(chunks)
        assert g.has_edge("sec_10.1", "sec_10")
        assert g.edges["sec_10.1", "sec_10"]["edge_type"] == "subsection_of"

    def test_sequential_edge_same_page(self) -> None:
        chunks = [_chunk("a", page=1), _chunk("b", page=1)]
        g = build_graph(chunks)
        assert g.has_edge("chunk_0", "chunk_1")
        assert g.edges["chunk_0", "chunk_1"]["edge_type"] == "sequential"

    def test_no_sequential_edge_across_pages(self) -> None:
        chunks = [_chunk("a", page=1), _chunk("b", page=2)]
        g = build_graph(chunks)
        assert not g.has_edge("chunk_0", "chunk_1")

    def test_references_edge_added(self) -> None:
        chunks = [
            _chunk("§14 Kündigung\nDer Vertrag ..."),
            _chunk("Gemäß §14 gilt eine Frist."),
        ]
        g = build_graph(chunks)
        # chunk_1 references sec_14
        assert g.has_edge("chunk_1", "sec_14")
        assert g.edges["chunk_1", "sec_14"]["edge_type"] == "references"

    def test_no_self_references_edge(self) -> None:
        """A chunk that IS section §10 should not get a references edge to sec_10."""
        chunks = [_chunk("§10 Titel\nSiehe §10 für Details.")]
        g = build_graph(chunks)
        # chunk_0 has chunk_of -> sec_10, should NOT also have references -> sec_10
        edges_to_sec10 = [
            (u, v, d) for u, v, d in g.edges(data=True)
            if v == "sec_10" and d.get("edge_type") == "references"
        ]
        assert len(edges_to_sec10) == 0


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_round_trip(self, tmp_path: Path) -> None:
        chunks = [
            _chunk("§1 Einleitung\nText"),
            _chunk("Gemäß §1 gilt."),
        ]
        g = build_graph(chunks)
        path = tmp_path / "graph.json"
        save_graph(g, path)
        g2 = load_graph(path)
        assert g2.number_of_nodes() == g.number_of_nodes()
        assert g2.number_of_edges() == g.number_of_edges()

    def test_load_missing_file_returns_empty(self, tmp_path: Path) -> None:
        g = load_graph(tmp_path / "nonexistent.json")
        assert g.number_of_nodes() == 0

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        chunks = [_chunk("text")]
        g = build_graph(chunks)
        deep_path = tmp_path / "a" / "b" / "graph.json"
        save_graph(g, deep_path)
        assert deep_path.exists()
