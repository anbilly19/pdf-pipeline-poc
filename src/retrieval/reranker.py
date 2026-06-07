"""Cross-encoder reranking via Ollama.

Pipeline position:
    query
        -> FAISS top-(k * faiss_multiplier) candidates
        -> BM25 rerank (keyword pass)
        -> OllamaReranker second pass (cross-encoder)
        -> top_k returned to LLM

Ollama cross-encoder support
-----------------------------
Ollama exposes reranking via POST /api/rerank (added in Ollama 0.5.x).
The payload is::

    {"model": "bge-reranker-v2-m3", "query": "...", "documents": ["...", ...]}

The response is::

    {"results": [{"index": 0, "relevance_score": 0.92}, ...]}

Fallback behaviour
------------------
- If Ollama is not reachable (connection refused, timeout)  -> BM25 order preserved.
- If the model is not pulled yet                           -> BM25 order preserved.
- If the response is malformed                             -> BM25 order preserved.

Every fallback is logged at WARNING level so ops can see it.

Recommended model
-----------------
    ollama pull bge-reranker-v2-m3

Alternatives that also work:
    bge-reranker-v2-m3   (best quality, ~570 MB)
    bge-reranker-base    (faster, ~280 MB)
    jina-reranker-v2-base-multilingual  (multilingual, good for German)
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models import Chunk

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_MODEL = "bge-reranker-v2-m3"
_TIMEOUT_SECONDS = 30
_MAX_DOC_CHARS = 512  # truncate very long chunks to keep reranker latency bounded


class OllamaReranker:
    """Cross-encoder reranker backed by an Ollama /api/rerank endpoint.

    Args:
        model: Ollama model name.  Must support the /api/rerank endpoint.
        host: Base URL of the Ollama server.
        timeout: HTTP timeout in seconds.
        max_doc_chars: Truncate each document to this many characters before
                       sending to the reranker to keep latency predictable.

    Non-fatal contract
    ------------------
    Every public method catches all exceptions and returns the original chunk
    order unchanged.  This ensures the reranker never breaks the pipeline.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host: str = _DEFAULT_HOST,
        timeout: int = _TIMEOUT_SECONDS,
        max_doc_chars: int = _MAX_DOC_CHARS,
    ) -> None:
        self._model = model
        self._url = f"{host.rstrip('/')}/api/rerank"
        self._timeout = timeout
        self._max_doc_chars = max_doc_chars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:  # type: ignore[name-defined]
        """Rerank *chunks* for *query* using the Ollama cross-encoder.

        Args:
            query: The search query.
            chunks: Candidate chunks already filtered/ranked by BM25.
            top_k: Maximum number of chunks to return.

        Returns:
            Chunks sorted by cross-encoder relevance score (best first),
            truncated to top_k.  Falls back to the original order on any error.
        """
        if not chunks:
            return chunks

        try:
            scores = self._call_ollama(query, chunks)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "OllamaReranker fallback (BM25 order kept) — %s: %s",
                type(exc).__name__, exc,
            )
            return chunks[:top_k]

        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )
        result = [c for _, c in ranked[:top_k]]
        logger.info(
            "Cross-encoder rerank: %d candidates -> %d (query: %.60s)",
            len(chunks), len(result), query,
        )
        return result

    def is_available(self) -> bool:
        """Return True if the Ollama rerank endpoint responds.

        Performs a lightweight HEAD-style ping by sending a minimal request.
        Returns False on any network or server error.
        """
        try:
            self._call_ollama("ping", [], validate=False)
            return True
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_ollama(
        self,
        query: str,
        chunks: list[Chunk],  # type: ignore[name-defined]
        *,
        validate: bool = True,
    ) -> list[float]:
        """POST to /api/rerank and return a score per chunk.

        Args:
            query: Search query.
            chunks: Candidate chunks.
            validate: When False, skip result parsing (used for availability ping).

        Returns:
            List of relevance scores in the same order as *chunks*.

        Raises:
            urllib.error.URLError: On connection failure.
            ValueError: On unexpected response shape.
        """
        documents = [
            c.text[: self._max_doc_chars] for c in chunks
        ] if chunks else [""]

        payload = json.dumps(
            {"model": self._model, "query": query, "documents": documents}
        ).encode()

        req = urllib.request.Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
            body = json.loads(resp.read().decode())

        if not validate:
            return []

        results = body.get("results")
        if not results or len(results) != len(chunks):
            raise ValueError(
                f"Unexpected /api/rerank response: expected {len(chunks)} results, "
                f"got {len(results) if results else 0}. Body: {body}"
            )

        # results may be ordered by score or by original index — normalise
        scores: list[float] = [0.0] * len(chunks)
        for entry in results:
            scores[entry["index"]] = float(entry["relevance_score"])
        return scores
