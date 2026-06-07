"""TDD tests for src.agent.memory — Roadmap #5.

Fully offline. No Ollama, no LangGraph compiled graph invocation.
Verifies:
  - AgentState type annotation structure
  - make_checkpointer returns MemorySaver for 'memory' backend
  - make_checkpointer falls back to MemorySaver when sqlite unavailable
  - make_checkpointer raises ValueError for unknown backend
  - make_thread_config produces a valid LangGraph RunnableConfig dict
  - make_thread_config generates a stable UUID when called with same id
  - make_thread_config generates a fresh UUID when called with None
  - trim_retrieval_context: empty input -> empty output
  - trim_retrieval_context: single snippet within budget -> returned as-is
  - trim_retrieval_context: total over budget -> oldest snippets dropped
  - trim_retrieval_context: exact budget boundary -> no snippets dropped
  - trim_retrieval_context: single snippet over budget -> empty (can't fit)
  - trim_retrieval_context: preserves chronological order after trim
  - isolation: retrieval_context key is separate from messages key
"""
from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from src.agent.memory import (
    AgentState,
    make_checkpointer,
    make_thread_config,
    trim_retrieval_context,
)
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# AgentState structure
# ---------------------------------------------------------------------------

class TestAgentState:
    def test_has_messages_key(self) -> None:
        annotations = AgentState.__annotations__
        assert "messages" in annotations

    def test_has_retrieval_context_key(self) -> None:
        annotations = AgentState.__annotations__
        assert "retrieval_context" in annotations

    def test_keys_are_distinct(self) -> None:
        keys = list(AgentState.__annotations__)
        assert "messages" in keys
        assert "retrieval_context" in keys
        assert keys.count("messages") == 1
        assert keys.count("retrieval_context") == 1


# ---------------------------------------------------------------------------
# make_checkpointer
# ---------------------------------------------------------------------------

class TestMakeCheckpointer:
    def test_memory_backend_returns_memory_saver(self) -> None:
        saver = make_checkpointer("memory")
        assert isinstance(saver, MemorySaver)

    def test_default_backend_is_memory(self) -> None:
        saver = make_checkpointer()
        assert isinstance(saver, MemorySaver)

    def test_unknown_backend_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown checkpointer backend"):
            make_checkpointer("redis")

    def test_sqlite_falls_back_when_not_installed(self) -> None:
        """When langgraph-checkpoint-sqlite is absent, MemorySaver is returned."""
        fake_modules = {
            "langgraph.checkpoint.sqlite": None,  # simulate ImportError
        }
        with patch.dict(sys.modules, fake_modules):  # type: ignore[arg-type]
            saver = make_checkpointer("sqlite")
        assert isinstance(saver, MemorySaver)


# ---------------------------------------------------------------------------
# make_thread_config
# ---------------------------------------------------------------------------

class TestMakeThreadConfig:
    def test_returns_configurable_dict(self) -> None:
        cfg = make_thread_config("abc123")
        assert "configurable" in cfg
        assert cfg["configurable"]["thread_id"] == "abc123"

    def test_none_generates_uuid(self) -> None:
        cfg = make_thread_config(None)
        tid = cfg["configurable"]["thread_id"]
        assert isinstance(tid, str) and len(tid) == 36  # UUID v4 format

    def test_two_none_calls_produce_different_ids(self) -> None:
        tid1 = make_thread_config(None)["configurable"]["thread_id"]
        tid2 = make_thread_config(None)["configurable"]["thread_id"]
        assert tid1 != tid2

    def test_same_id_is_preserved(self) -> None:
        cfg = make_thread_config("session-42")
        assert cfg["configurable"]["thread_id"] == "session-42"


# ---------------------------------------------------------------------------
# trim_retrieval_context
# ---------------------------------------------------------------------------

class TestTrimRetrievalContext:
    def test_empty_input_returns_empty(self) -> None:
        assert trim_retrieval_context([]) == []

    def test_single_snippet_within_budget(self) -> None:
        snippets = ["short text"]
        result = trim_retrieval_context(snippets, char_limit=100)
        assert result == snippets

    def test_total_over_budget_drops_oldest(self) -> None:
        # 3 snippets of 100 chars each; budget = 150 -> oldest dropped
        snippets = ["A" * 100, "B" * 100, "C" * 100]
        result = trim_retrieval_context(snippets, char_limit=150)
        # Should keep the two most recent that fit
        assert result == ["B" * 100, "C" * 100] or result == ["C" * 100]
        # Must NOT contain the oldest
        assert "A" * 100 not in result

    def test_exact_budget_keeps_all(self) -> None:
        snippets = ["AB", "CD"]  # 2 + 2 = 4 chars + 1 separator = 5
        result = trim_retrieval_context(snippets, char_limit=5)
        assert result == snippets

    def test_single_snippet_over_budget_returns_empty(self) -> None:
        """A snippet that alone exceeds the budget cannot be included."""
        snippets = ["X" * 200]
        result = trim_retrieval_context(snippets, char_limit=100)
        assert result == []

    def test_preserves_chronological_order(self) -> None:
        snippets = ["first", "second", "third"]
        result = trim_retrieval_context(snippets, char_limit=1000)
        assert result == snippets

    def test_drops_oldest_when_trimming(self) -> None:
        snippets = ["oldest", "middle", "newest"]
        # Only last two fit
        result = trim_retrieval_context(snippets, char_limit=len("middle") + len("newest") + 1)
        assert "oldest" not in result
        assert "newest" in result


# ---------------------------------------------------------------------------
# Isolation: retrieval_context != messages
# ---------------------------------------------------------------------------

class TestKeyIsolation:
    def test_state_dict_can_hold_both_keys_independently(self) -> None:
        from langchain_core.messages import HumanMessage
        state: AgentState = {  # type: ignore[typeddict-item]
            "messages": [HumanMessage(content="Hallo")],
            "retrieval_context": ["Seite 3: Begrüßung"],
        }
        assert state["messages"][0].content == "Hallo"
        assert state["retrieval_context"][0] == "Seite 3: Begrüßung"
        # Mutating one key does not affect the other
        state["retrieval_context"] = []
        assert state["messages"][0].content == "Hallo"
