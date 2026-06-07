"""TDD tests for OllamaReranker.

Fully offline — Ollama network calls are mocked.
All tests verify the non-fatal contract: any failure returns BM25 order.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.models import Chunk
from src.retrieval.reranker import OllamaReranker


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


def _mock_response(results: list[dict]) -> MagicMock:
    """Build a mock urllib response returning the given results payload."""
    body = json.dumps({"results": results}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestOllamaRerankerHappyPath:
    def test_rerank_sorts_by_score_descending(self) -> None:
        chunks = [_chunk("low relevance"), _chunk("high relevance"), _chunk("medium")]
        results = [
            {"index": 0, "relevance_score": 0.1},
            {"index": 1, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.5},
        ]
        with patch("urllib.request.urlopen", return_value=_mock_response(results)):
            reranker = OllamaReranker(model="bge-reranker-v2-m3")
            ranked = reranker.rerank("query", chunks, top_k=3)

        assert ranked[0].text == "high relevance"
        assert ranked[1].text == "medium"
        assert ranked[2].text == "low relevance"

    def test_top_k_truncation(self) -> None:
        chunks = [_chunk(f"chunk {i}") for i in range(5)]
        results = [{"index": i, "relevance_score": float(i) / 5} for i in range(5)]
        with patch("urllib.request.urlopen", return_value=_mock_response(results)):
            reranker = OllamaReranker()
            ranked = reranker.rerank("q", chunks, top_k=2)
        assert len(ranked) == 2

    def test_empty_chunks_returns_empty(self) -> None:
        reranker = OllamaReranker()
        assert reranker.rerank("q", [], top_k=5) == []

    def test_doc_truncated_to_max_chars(self) -> None:
        """Verify long chunk text is truncated before being sent to Ollama."""
        long_text = "x" * 1000
        chunks = [_chunk(long_text)]
        results = [{"index": 0, "relevance_score": 0.8}]
        captured: list[bytes] = []

        def fake_urlopen(req, timeout):  # noqa: ANN001
            captured.append(req.data)
            return _mock_response(results)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            OllamaReranker(max_doc_chars=100).rerank("q", chunks, top_k=1)

        payload = json.loads(captured[0])
        assert len(payload["documents"][0]) == 100


# ---------------------------------------------------------------------------
# Fallback / non-fatal tests
# ---------------------------------------------------------------------------

class TestOllamaRerankerFallback:
    def test_connection_error_returns_bm25_order(self) -> None:
        import urllib.error
        chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            reranker = OllamaReranker()
            result = reranker.rerank("q", chunks, top_k=3)
        assert result == chunks  # original order preserved

    def test_timeout_returns_bm25_order(self) -> None:
        import socket
        chunks = [_chunk("a"), _chunk("b")]
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            reranker = OllamaReranker()
            result = reranker.rerank("q", chunks, top_k=2)
        assert result == chunks

    def test_malformed_response_returns_bm25_order(self) -> None:
        """Wrong number of results in response -> fallback."""
        chunks = [_chunk("a"), _chunk("b")]
        # Only 1 result returned for 2 chunks -> ValueError -> fallback
        bad_results = [{"index": 0, "relevance_score": 0.9}]
        with patch("urllib.request.urlopen", return_value=_mock_response(bad_results)):
            reranker = OllamaReranker()
            result = reranker.rerank("q", chunks, top_k=2)
        assert result == chunks

    def test_json_decode_error_returns_bm25_order(self) -> None:
        bad_resp = MagicMock()
        bad_resp.read.return_value = b"not json"
        bad_resp.__enter__ = lambda s: s
        bad_resp.__exit__ = MagicMock(return_value=False)
        chunks = [_chunk("a")]
        with patch("urllib.request.urlopen", return_value=bad_resp):
            reranker = OllamaReranker()
            result = reranker.rerank("q", chunks, top_k=1)
        assert result == chunks


# ---------------------------------------------------------------------------
# Integration with BBoxRetriever (mock store + embedder)
# ---------------------------------------------------------------------------

class TestRetrieverWithReranker:
    def test_retriever_calls_reranker_when_configured(self) -> None:
        from src.retrieval.retriever import BBoxRetriever

        chunks = [_chunk("relevant"), _chunk("less relevant")]

        store = MagicMock()
        store.count.return_value = 2
        store.query.return_value = chunks

        embedder = MagicMock()
        embedder.embed_query.return_value = [0.0] * 384

        reranker = MagicMock()
        reranker.rerank.return_value = [chunks[0]]

        retriever = BBoxRetriever(
            store=store, embedder=embedder, top_k=1, reranker=reranker
        )
        result = retriever.retrieve("test query")

        reranker.rerank.assert_called_once()
        assert result == [chunks[0]]

    def test_retriever_skips_reranker_when_none(self) -> None:
        from src.retrieval.retriever import BBoxRetriever

        chunks = [_chunk("a"), _chunk("b")]
        store = MagicMock()
        store.count.return_value = 2
        store.query.return_value = chunks
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.0] * 384

        retriever = BBoxRetriever(store=store, embedder=embedder, top_k=2, reranker=None)
        result = retriever.retrieve("query")
        assert len(result) == 2
