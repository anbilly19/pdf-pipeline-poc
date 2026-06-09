"""Self-RAG knowledge filter — Roadmap #6.

For each candidate chunk, this module runs a lightweight local Ollama call
that asks the model whether the chunk actually contains information relevant
to the query.  Only chunks that pass the relevance check are forwarded to
the generator.

Latency gate (CLAUDE.md constraint)
-------------------------------------
The filter is **latency-gated**: it is only invoked when a chunk's BM25
score falls *below* a configurable threshold (`bm25_gate`).  High-scoring
BM25 chunks are considered relevant without an extra LLM call, keeping the
common case fast.

Offline / Ollama-only guarantee
---------------------------------
All inference goes through Ollama.  If Ollama is unavailable (network
error, model not pulled), the filter fails open: every chunk is considered
relevant and the pipeline continues without error.

Public API
-----------
    SelfRAGFilter         — main class, call .filter(query, scored_chunks)
    SelfRAGResult         — dataclass (chunk, is_relevant, score, reason)
    make_self_rag_filter  — convenience factory used by build_tools()
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import NamedTuple

from src.models import Chunk

logger = logging.getLogger(__name__)

_RELEVANCE_PROMPT = """\
Du bist ein Relevanzfilter für ein Dokumentenabfrage-System.
Beurteile, ob der folgende TEXTABSCHNITT eine nützliche Antwort auf die ANFRAGE enthält.

ANFRAGE: {query}

TEXTABSCHNITT:
{chunk_text}

Antworte NUR mit einem JSON-Objekt (keine Erklärung, kein Markdown):
{{"relevant": true, "score": 0.9, "reason": "enthält direkte Antwort"}}
oder
{{"relevant": false, "score": 0.1, "reason": "thematisch nicht verwandt"}}

"score" ist deine Sicherheit (0.0 bis 1.0)."""

_MAX_CHUNK_CHARS = 800
_DEFAULT_BM25_GATE = 0.5
_DEFAULT_RELEVANCE_THRESHOLD = 0.35
_DEFAULT_TIMEOUT = 60  # seconds — gemma4:e2b can take 20-60s on first call


@dataclass
class SelfRAGResult:
    chunk: Chunk
    is_relevant: bool
    score: float
    reason: str = ""
    skipped: bool = False


class ScoredChunk(NamedTuple):
    chunk: Chunk
    bm25_score: float


class SelfRAGFilter:
    """Latency-gated Self-RAG relevance filter.

    Args:
        model: Ollama model name to use for relevance checks.
        bm25_gate: Normalised BM25 score above which the LLM call is skipped.
        relevance_threshold: Minimum score for a chunk to be kept.
        enabled: Master switch. When False, all chunks pass through.
        timeout: HTTP timeout in seconds for the Ollama API call.
                 Default 60s — gemma4:e2b can take 20-60s on cold/slow hardware.
    """

    def __init__(
        self,
        model: str = "gemma4:e2b",
        bm25_gate: float = _DEFAULT_BM25_GATE,
        relevance_threshold: float = _DEFAULT_RELEVANCE_THRESHOLD,
        enabled: bool = True,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self._model = model
        self._bm25_gate = bm25_gate
        self._relevance_threshold = relevance_threshold
        self._enabled = enabled
        self._timeout = timeout

    def filter(
        self,
        query: str,
        scored_chunks: list[ScoredChunk],
    ) -> list[SelfRAGResult]:
        if not self._enabled:
            return [
                SelfRAGResult(chunk=sc.chunk, is_relevant=True, score=1.0, skipped=True)
                for sc in scored_chunks
            ]

        results: list[SelfRAGResult] = []
        for sc in scored_chunks:
            result = self._check_single(query, sc)
            if result.is_relevant:
                results.append(result)
            else:
                logger.debug(
                    "Self-RAG dropped chunk (page=%d, score=%.2f, reason=%r)",
                    sc.chunk.page_number, result.score, result.reason,
                )
        kept = len(results)
        dropped = len(scored_chunks) - kept
        logger.info("Self-RAG filter: %d kept, %d dropped", kept, dropped)
        return results

    def check_one(
        self,
        query: str,
        chunk: Chunk,
        bm25_score: float = 0.0,
    ) -> SelfRAGResult:
        return self._check_single(query, ScoredChunk(chunk=chunk, bm25_score=bm25_score))

    def _check_single(self, query: str, sc: ScoredChunk) -> SelfRAGResult:
        if sc.bm25_score >= self._bm25_gate:
            return SelfRAGResult(
                chunk=sc.chunk,
                is_relevant=True,
                score=sc.bm25_score,
                reason="BM25 gate passed",
                skipped=True,
            )
        raw_response = self._call_ollama(query, sc.chunk.text)
        return self._parse_response(sc.chunk, raw_response)

    def _call_ollama(self, query: str, chunk_text: str) -> str:
        """Call Ollama synchronously and return the raw response string.

        Fails open: returns a permissive JSON string on any error.
        """
        prompt = _RELEVANCE_PROMPT.format(
            query=query,
            chunk_text=chunk_text[:_MAX_CHUNK_CHARS],
        )
        try:
            import requests  # noqa: PLC0415
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "{}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Self-RAG Ollama call failed (%s) — failing open", exc)
            return '{"relevant": true, "score": 0.5, "reason": "filter unavailable"}'

    def _parse_response(self, chunk: Chunk, raw: str) -> SelfRAGResult:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(
                ln for ln in lines if not ln.startswith("```")
            ).strip()

        try:
            data = json.loads(cleaned)
            relevant = bool(data.get("relevant", True))
            score = float(data.get("score", 0.5))
            reason = str(data.get("reason", ""))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Self-RAG JSON parse failed (%s) — failing open", exc)
            relevant, score, reason = True, 0.5, "parse error"

        is_relevant = relevant and score >= self._relevance_threshold
        return SelfRAGResult(
            chunk=chunk,
            is_relevant=is_relevant,
            score=score,
            reason=reason,
        )


def make_self_rag_filter(
    model: str = "gemma4:e2b",
    bm25_gate: float = _DEFAULT_BM25_GATE,
    relevance_threshold: float = _DEFAULT_RELEVANCE_THRESHOLD,
    enabled: bool = True,
    timeout: int = _DEFAULT_TIMEOUT,
) -> SelfRAGFilter:
    """Factory used by build_tools() to create a configured filter."""
    return SelfRAGFilter(
        model=model,
        bm25_gate=bm25_gate,
        relevance_threshold=relevance_threshold,
        enabled=enabled,
        timeout=timeout,
    )
