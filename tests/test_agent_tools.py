"""Tests for agent tools — all mocked, no LLM or vector store required."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent.tools import ToolResult, _markdown_table_to_csv, build_tools
from src.models import Chunk
from src.retrieval.retriever import BBoxRetriever


def _make_chunk(
    text: str = "Test",
    page: int = 1,
    chunk_type: str = "text",
    bboxes: list[list[float]] | None = None,
) -> Chunk:
    return Chunk(
        text=text,
        page_number=page,
        bboxes=bboxes or [[0.0, 0.0, 100.0, 20.0]],
        chunk_type=chunk_type,  # type: ignore[arg-type]
        confidence=0.9,
        image_path="/tmp/p1.png",
    )


@pytest.fixture()
def mock_retriever() -> MagicMock:
    return MagicMock(spec=BBoxRetriever)


@pytest.fixture()
def tools(mock_retriever: MagicMock) -> list[object]:
    return build_tools(mock_retriever)


def _get_tool(tools: list[object], name: str) -> object:
    for t in tools:
        if t.name == name:  # type: ignore[union-attr]
            return t
    raise KeyError(f"Tool '{name}' not found")


# ---------------------------------------------------------------------------
# search_term
# ---------------------------------------------------------------------------

def test_search_term_returns_results(tools: list[object], mock_retriever: MagicMock) -> None:
    mock_retriever.retrieve.return_value = [_make_chunk("Ergebnis Text", page=3)]
    t = _get_tool(tools, "search_term")
    result = t.invoke({"query": "Thema", "top_k": 3})  # type: ignore[union-attr]
    assert "Ergebnis Text" in result
    assert "page 3" in result


def test_search_term_no_results(tools: list[object], mock_retriever: MagicMock) -> None:
    mock_retriever.retrieve.return_value = []
    t = _get_tool(tools, "search_term")
    result = t.invoke({"query": "nichts", "top_k": 5})  # type: ignore[union-attr]
    assert "nicht gefunden" in result


# ---------------------------------------------------------------------------
# extract_table_to_csv
# ---------------------------------------------------------------------------

def test_extract_table_returns_csv(tools: list[object], mock_retriever: MagicMock) -> None:
    table_text = "| Name | Wert |\n| ---- | ---- |\n| A | 1 |\n| B | 2 |"
    mock_retriever.retrieve.return_value = [_make_chunk(table_text, chunk_type="table")]
    t = _get_tool(tools, "extract_table_to_csv")
    result = t.invoke({"query": "Tabelle"})  # type: ignore[union-attr]
    assert "Name" in result
    assert "Wert" in result


def test_extract_table_no_results(tools: list[object], mock_retriever: MagicMock) -> None:
    mock_retriever.retrieve.return_value = []
    t = _get_tool(tools, "extract_table_to_csv")
    result = t.invoke({"query": "keine Tabelle"})  # type: ignore[union-attr]
    assert "nicht gefunden" in result


# ---------------------------------------------------------------------------
# summarize_section
# ---------------------------------------------------------------------------

def test_summarize_section_combines_chunks(
    tools: list[object], mock_retriever: MagicMock
) -> None:
    chunks = [_make_chunk(f"Abschnitt {i}", page=i + 1) for i in range(3)]
    mock_retriever.retrieve.return_value = chunks
    t = _get_tool(tools, "summarize_section")
    result = t.invoke({"title": "Einleitung"})  # type: ignore[union-attr]
    assert "Abschnitt 0" in result
    assert "Abschnitt 2" in result


# ---------------------------------------------------------------------------
# highlight_section
# ---------------------------------------------------------------------------

def test_highlight_section_filters_by_page(
    tools: list[object], mock_retriever: MagicMock
) -> None:
    chunks = [
        _make_chunk("Auf Seite 2", page=2, bboxes=[[10.0, 20.0, 50.0, 40.0]]),
        _make_chunk("Auf Seite 3", page=3),
    ]
    mock_retriever.retrieve.return_value = chunks
    t = _get_tool(tools, "highlight_section")
    result = t.invoke({"page_number": 2, "query": "Seite 2"})  # type: ignore[union-attr]
    assert "page 2" in result
    assert "10.0" in result


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------

def test_tool_result_str_format() -> None:
    r = ToolResult(content="Antwort", bboxes=[[0, 0, 10, 10]], page_number=4, image_path="/p.png")
    s = str(r)
    assert "Antwort" in s
    assert "page 4" in s


# ---------------------------------------------------------------------------
# _markdown_table_to_csv
# ---------------------------------------------------------------------------

def test_markdown_table_to_csv_basic() -> None:
    md = "| A | B |\n| - | - |\n| 1 | 2 |"
    csv = _markdown_table_to_csv(md)
    assert "A" in csv
    assert "1" in csv


def test_markdown_table_to_csv_fallback() -> None:
    plain = "Kein Tabellenformat hier."
    result = _markdown_table_to_csv(plain)
    assert result == plain
