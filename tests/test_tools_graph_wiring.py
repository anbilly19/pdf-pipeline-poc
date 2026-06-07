"""TDD tests verifying that build_tools wires graph expansion correctly.

Fully offline — no FAISS, no LLM, no PDF.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.models import Chunk
from src.agent.tools import build_tools, NO_RESULTS
from src.graph.builder import build_graph

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


def _mock_retriever(chunks: list[Chunk]) -> MagicMock:
    retriever = MagicMock()
    retriever.retrieve.return_value = chunks
    return retriever


# ---------------------------------------------------------------------------
# Tests: build_tools without graph (no-op path)
# ---------------------------------------------------------------------------

class TestBuildToolsNoGraph:
    def test_search_term_returns_results_without_graph(self) -> None:
        chunk = _chunk("Vertragsstrafe bei Verzug.")
        retriever = _mock_retriever([chunk])
        tools = build_tools(retriever)
        search = next(t for t in tools if t.name == "search_term")
        result = search.invoke({"query": "Vertragsstrafe"})
        assert "Vertragsstrafe" in result

    def test_search_term_no_results(self) -> None:
        retriever = _mock_retriever([])
        tools = build_tools(retriever)
        search = next(t for t in tools if t.name == "search_term")
        result = search.invoke({"query": "unbekannt"})
        assert result == NO_RESULTS


# ---------------------------------------------------------------------------
# Tests: build_tools WITH graph (expansion path)
# ---------------------------------------------------------------------------

class TestBuildToolsWithGraph:
    def _setup(self) -> tuple:
        all_chunks = [
            _chunk("§1 Einleitung\nDieser Vertrag regelt..."),   # chunk_0 -> sec_1
            _chunk("Weitere Details zur Einleitung."),           # chunk_1 -> sec_1 (sibling)
            _chunk("§2 Vertragsstrafe\nBei Verzug gilt..."),     # chunk_2 -> sec_2
        ]
        graph = build_graph(all_chunks)
        return all_chunks, graph

    def test_search_term_expands_siblings(self) -> None:
        all_chunks, graph = self._setup()
        # retriever returns only chunk_0; expect chunk_1 (sibling in §1) via expansion
        retriever = _mock_retriever([all_chunks[0]])
        tools = build_tools(retriever, graph=graph, all_chunks=all_chunks)
        search = next(t for t in tools if t.name == "search_term")
        result = search.invoke({"query": "Einleitung"})
        assert "Weitere Details" in result

    def test_summarize_section_expands_siblings(self) -> None:
        all_chunks, graph = self._setup()
        retriever = _mock_retriever([all_chunks[0]])
        tools = build_tools(retriever, graph=graph, all_chunks=all_chunks)
        summarize = next(t for t in tools if t.name == "summarize_section")
        result = summarize.invoke({"title": "Einleitung"})
        assert "Weitere Details" in result

    def test_expansion_does_not_duplicate_chunks(self) -> None:
        all_chunks, graph = self._setup()
        # Feed both sibling chunks as seeds — should not appear twice
        retriever = _mock_retriever([all_chunks[0], all_chunks[1]])
        tools = build_tools(retriever, graph=graph, all_chunks=all_chunks)
        search = next(t for t in tools if t.name == "search_term")
        result = search.invoke({"query": "Einleitung"})
        # Count occurrences of the unique text
        assert result.count("§1 Einleitung") == 1

    def test_graph_failure_is_non_fatal(self) -> None:
        """A broken graph must not crash the tool — seeds returned as-is."""
        chunk = _chunk("Some text")
        retriever = _mock_retriever([chunk])
        # Pass a non-DiGraph object to force expansion failure
        tools = build_tools(retriever, graph="broken", all_chunks=[chunk])
        search = next(t for t in tools if t.name == "search_term")
        result = search.invoke({"query": "text"})
        assert "Some text" in result

    def test_extract_table_not_expanded(self) -> None:
        """extract_table_to_csv never expands — one clear best chunk only."""
        all_chunks, graph = self._setup()
        retriever = MagicMock()
        retriever.retrieve.return_value = [all_chunks[2]]  # table result
        tools = build_tools(retriever, graph=graph, all_chunks=all_chunks)
        table_tool = next(t for t in tools if t.name == "extract_table_to_csv")
        result = table_tool.invoke({"query": "Vertragsstrafe Tabelle"})
        # Should contain just the one chunk, not siblings
        assert "Vertragsstrafe" in result
