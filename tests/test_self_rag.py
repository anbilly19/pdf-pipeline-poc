"""TDD tests for src.agent.self_rag — Roadmap #6.

Fully offline. All Ollama HTTP calls are patched.
Verifies:
  - BM25 gate: high-score chunks skip the LLM call entirely
  - BM25 gate: low-score chunks trigger the LLM call
  - LLM call: returns relevant=True when JSON says relevant=true
  - LLM call: returns relevant=False when score < threshold
  - LLM call: fails open when Ollama is unreachable
  - LLM call: fails open when JSON is malformed
  - LLM call: handles markdown-fenced JSON from model
  - filter: removes irrelevant chunks, keeps relevant
  - filter: falls back to originals when all filtered
  - filter: disabled mode passes all chunks through (no HTTP call)
  - check_one: delegates correctly
  - make_self_rag_filter: factory returns SelfRAGFilter
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agent.self_rag import (
    ScoredChunk,
    SelfRAGFilter,
    SelfRAGResult,
    make_self_rag_filter,
)
from src.models import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(text: str = "test chunk", page: int = 1) -> Chunk:
    return Chunk(
        text=text,
        page_number=page,
        bboxes=[[0, 0, 100, 20]],
        chunk_type="text",
        confidence=0.9,
        image_path="",
    )


def _filter(enabled: bool = True, bm25_gate: float = 0.5, threshold: float = 0.35) -> SelfRAGFilter:
    return SelfRAGFilter(
        model="test-model",
        bm25_gate=bm25_gate,
        relevance_threshold=threshold,
        enabled=enabled,
    )


def _mock_response(relevant: bool, score: float, reason: str = "test") -> MagicMock:
    """Build a mock requests.post response returning the given JSON."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "response": json.dumps({"relevant": relevant, "score": score, "reason": reason})
    }
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ---------------------------------------------------------------------------
# BM25 gate
# ---------------------------------------------------------------------------

class TestBM25Gate:
    def test_high_score_skips_llm(self) -> None:
        """Chunks with bm25_score >= gate must NOT call Ollama."""
        f = _filter(bm25_gate=0.5)
        sc = ScoredChunk(chunk=_chunk(), bm25_score=0.8)
        with patch("requests.post") as mock_post:
            result = f._check_single("query", sc)
        mock_post.assert_not_called()
        assert result.is_relevant is True
        assert result.skipped is True

    def test_low_score_calls_llm(self) -> None:
        """Chunks with bm25_score < gate must call Ollama."""
        f = _filter(bm25_gate=0.5)
        sc = ScoredChunk(chunk=_chunk(), bm25_score=0.2)
        with patch("requests.post", return_value=_mock_response(True, 0.9)) as mock_post:
            f._check_single("query", sc)
        mock_post.assert_called_once()

    def test_exact_gate_boundary_skips(self) -> None:
        """Score exactly equal to the gate threshold should skip."""
        f = _filter(bm25_gate=0.5)
        sc = ScoredChunk(chunk=_chunk(), bm25_score=0.5)
        with patch("requests.post") as mock_post:
            result = f._check_single("query", sc)
        mock_post.assert_not_called()
        assert result.skipped is True


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

class TestLLMResponse:
    def test_relevant_true_when_llm_says_relevant(self) -> None:
        f = _filter()
        sc = ScoredChunk(chunk=_chunk(), bm25_score=0.0)
        with patch("requests.post", return_value=_mock_response(True, 0.9)):
            result = f._check_single("query", sc)
        assert result.is_relevant is True
        assert result.score == pytest.approx(0.9)

    def test_relevant_false_when_score_below_threshold(self) -> None:
        f = _filter(threshold=0.35)
        sc = ScoredChunk(chunk=_chunk(), bm25_score=0.0)
        with patch("requests.post", return_value=_mock_response(True, 0.2)):
            result = f._check_single("query", sc)
        assert result.is_relevant is False

    def test_relevant_false_when_llm_says_false(self) -> None:
        f = _filter()
        sc = ScoredChunk(chunk=_chunk(), bm25_score=0.0)
        with patch("requests.post", return_value=_mock_response(False, 0.9)):
            result = f._check_single("query", sc)
        assert result.is_relevant is False

    def test_fails_open_on_network_error(self) -> None:
        f = _filter()
        sc = ScoredChunk(chunk=_chunk(), bm25_score=0.0)
        with patch("requests.post", side_effect=ConnectionError("Ollama down")):
            result = f._check_single("query", sc)
        assert result.is_relevant is True  # fail open

    def test_fails_open_on_malformed_json(self) -> None:
        f = _filter()
        sc = ScoredChunk(chunk=_chunk(), bm25_score=0.0)
        bad_resp = MagicMock()
        bad_resp.json.return_value = {"response": "not json at all!!!"}
        bad_resp.raise_for_status = MagicMock()
        with patch("requests.post", return_value=bad_resp):
            result = f._check_single("query", sc)
        assert result.is_relevant is True  # fail open

    def test_handles_markdown_fenced_json(self) -> None:
        f = _filter()
        sc = ScoredChunk(chunk=_chunk(), bm25_score=0.0)
        fenced_resp = MagicMock()
        fenced_resp.json.return_value = {
            "response": '```json\n{"relevant": true, "score": 0.85, "reason": "matches"}\n```'
        }
        fenced_resp.raise_for_status = MagicMock()
        with patch("requests.post", return_value=fenced_resp):
            result = f._check_single("query", sc)
        assert result.is_relevant is True
        assert result.score == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# filter() method
# ---------------------------------------------------------------------------

class TestFilterMethod:
    def test_irrelevant_chunks_removed(self) -> None:
        f = _filter()
        chunks = [
            ScoredChunk(chunk=_chunk("relevant text"), bm25_score=0.0),
            ScoredChunk(chunk=_chunk("unrelated"), bm25_score=0.0),
        ]
        responses = [
            _mock_response(True, 0.9),
            _mock_response(False, 0.1),
        ]
        with patch("requests.post", side_effect=responses):
            results = f.filter("query", chunks)
        assert len(results) == 1
        assert results[0].chunk.text == "relevant text"

    def test_disabled_filter_passes_all(self) -> None:
        f = _filter(enabled=False)
        chunks = [ScoredChunk(chunk=_chunk(), bm25_score=0.0) for _ in range(5)]
        with patch("requests.post") as mock_post:
            results = f.filter("query", chunks)
        mock_post.assert_not_called()
        assert len(results) == 5
        assert all(r.skipped for r in results)

    def test_all_filtered_returns_empty_list(self) -> None:
        """filter() itself returns empty; _self_rag_filter in tools.py adds fallback."""
        f = _filter()
        chunks = [ScoredChunk(chunk=_chunk(), bm25_score=0.0)]
        with patch("requests.post", return_value=_mock_response(False, 0.05)):
            results = f.filter("query", chunks)
        assert results == []


# ---------------------------------------------------------------------------
# check_one
# ---------------------------------------------------------------------------

class TestCheckOne:
    def test_check_one_delegates_to_check_single(self) -> None:
        f = _filter()
        c = _chunk("specific text")
        with patch("requests.post", return_value=_mock_response(True, 0.88)) as mock_post:
            result = f.check_one("my query", c, bm25_score=0.1)
        mock_post.assert_called_once()
        assert result.is_relevant is True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_make_self_rag_filter_returns_instance(self) -> None:
        f = make_self_rag_filter(model="test", enabled=False)
        assert isinstance(f, SelfRAGFilter)
        assert f._enabled is False
