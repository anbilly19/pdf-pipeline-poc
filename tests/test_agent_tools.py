"""Tests for agent tools — all mocked, no LLM or vector store required."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent.tools import (
    NO_REGION,
    NO_RESULTS,
    NO_SECTION,
    NO_TABLE,
    ToolResult,
    _markdown_table_to_csv,
    build_tools,
)
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
    mock_retriever.retrieve.return_value = [_make_chunk("Result text", page=3)]
    t = _get_tool(tools, "search_term")
    result = t.invoke({"query": "topic", "top_k": 3})  # type: ignore[union-attr]
    assert "Result text" in result
    assert "page 3" in result


def test_search_term_no_results(tools: list[object], mock_retriever: MagicMock) -> None:
    mock_retriever.retrieve.return_value = []
    t = _get_tool(tools, "search_term")
    result = t.invoke({"query": "nothing", "top_k": 5})  # type: ignore[union-attr]
    assert result == NO_RESULTS


# ---------------------------------------------------------------------------
# extract_table_to_csv
# ---------------------------------------------------------------------------

def test_extract_table_returns_csv(tools: list[object], mock_retriever: MagicMock) -> None:
    table_text = "| Name | Value |\n| ---- | ----- |\n| A | 1 |\n| B | 2 |"
    mock_retriever.retrieve.return_value = [_make_chunk(table_text, chunk_type="table")]
    t = _get_tool(tools, "extract_table_to_csv")
    result = t.invoke({"query": "table"})  # type: ignore[union-attr]
    assert "Name" in result
    assert "Value" in result


def test_extract_table_no_results(tools: list[object], mock_retriever: MagicMock) -> None:
    mock_retriever.retrieve.return_value = []
    t = _get_tool(tools, "extract_table_to_csv")
    result = t.invoke({"query": "no table"})  # type: ignore[union-attr]
    assert result == NO_TABLE


# ---------------------------------------------------------------------------
# summarize_section
# ---------------------------------------------------------------------------

def test_summarize_section_combines_chunks(
    tools: list[object], mock_retriever: MagicMock
) -> None:
    chunks = [_make_chunk(f"Section {i}", page=i + 1) for i in range(3)]
    mock_retriever.retrieve.return_value = chunks
    t = _get_tool(tools, "summarize_section")
    result = t.invoke({"title": "Introduction"})  # type: ignore[union-attr]
    assert "Section 0" in result
    assert "Section 2" in result


def test_summarize_section_no_results(tools: list[object], mock_retriever: MagicMock) -> None:
    mock_retriever.retrieve.return_value = []
    t = _get_tool(tools, "summarize_section")
    result = t.invoke({"title": "missing"})  # type: ignore[union-attr]
    assert result == NO_SECTION


# ---------------------------------------------------------------------------
# highlight_section
# ---------------------------------------------------------------------------

def test_highlight_section_filters_by_page(
    tools: list[object], mock_retriever: MagicMock
) -> None:
    chunks = [
        _make_chunk("On page 2", page=2, bboxes=[[10.0, 20.0, 50.0, 40.0]]),
        _make_chunk("On page 3", page=3),
    ]
    mock_retriever.retrieve.return_value = chunks
    t = _get_tool(tools, "highlight_section")
    result = t.invoke({"page_number": 2, "query": "page 2"})  # type: ignore[union-attr]
    assert "page 2" in result
    assert "10.0" in result


def test_highlight_section_no_results(tools: list[object], mock_retriever: MagicMock) -> None:
    mock_retriever.retrieve.return_value = []
    t = _get_tool(tools, "highlight_section")
    result = t.invoke({"page_number": 1, "query": "nothing"})  # type: ignore[union-attr]
    assert result == NO_REGION


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------

def test_tool_result_str_format() -> None:
    r = ToolResult(content="Answer", bboxes=[[0, 0, 10, 10]], page_number=4, image_path="/p.png")
    s = str(r)
    assert "Answer" in s
    assert "page 4" in s


# ---------------------------------------------------------------------------
# _markdown_table_to_csv
# ---------------------------------------------------------------------------

def test_markdown_table_to_csv_basic() -> None:
    md = "| A | B |\n| - | - |\n| 1 | 2 |"
    csv_out = _markdown_table_to_csv(md)
    assert "A" in csv_out
    assert "1" in csv_out


def test_markdown_table_to_csv_fallback() -> None:
    plain = "No table format here."
    result = _markdown_table_to_csv(plain)
    assert result == plain
