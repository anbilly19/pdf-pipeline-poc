"""Chunk embedding — Ollama primary, sentence-transformers fallback.

Primary:  nomic-embed-text via local Ollama daemon
          Pull once: ollama pull nomic-embed-text
Fallback: sentence-transformers (paraphrase-multilingual-mpnet-base-v2)
          Fully offline, good German support, no Ollama required.

Zero-vector detection: if Ollama returns a zero/constant vector
(silent failure mode), the embedder automatically switches to the
sentence-transformers fallback for that batch/query.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

from src.models import Chunk

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_MODEL = "nomic-embed-text"
_FALLBACK_ST_MODEL = "paraphrase-multilingual-mpnet-base-v2"
_ZERO_THRESHOLD = 1e-6

# Suppress noisy transformers __path__ alias warnings emitted during
# sentence-transformers / transformers import (harmless, fixed upstream).
warnings.filterwarnings("ignore", message=r"Accessing `__path__`", module="transformers")
warnings.filterwarnings("ignore", message=r"Accessing `__path__`")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


class ChunkEmbedder:
    """Embeds Chunk objects into dense vectors.

    Tries Ollama first; falls back to sentence-transformers if Ollama
    is unavailable or returns zero/constant vectors.

    Args:
        model: Ollama embedding model identifier.
        base_url: Ollama API base URL.
        fallback_model: sentence-transformers model name for fallback.
    """

    def __init__(
        self,
        model: str = _DEFAULT_OLLAMA_MODEL,
        base_url: str = "http://localhost:11434",
        fallback_model: str = _FALLBACK_ST_MODEL,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._fallback_model = fallback_model
        self._use_fallback = False
        self._st_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Embed a list of chunks into dense vectors."""
        if not chunks:
            return []
        texts = [c.text for c in chunks]
        vectors = self._embed_texts(texts)
        logger.info(
            "Embedded %d chunks via %s",
            len(chunks),
            "sentence-transformers" if self._use_fallback else f"Ollama/{self._model}",
        )
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self._embed_texts([query])[0]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self._use_fallback:
            try:
                vectors = self._ollama_embed(texts)
                if self._is_broken(vectors):
                    logger.warning(
                        "Ollama/%s returned zero/constant vectors — switching to "
                        "sentence-transformers fallback permanently for this session.",
                        self._model,
                    )
                    self._use_fallback = True
                else:
                    return vectors
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Ollama embedding failed (%s) — switching to sentence-transformers fallback.",
                    exc,
                )
                self._use_fallback = True
        return self._st_embed(texts)

    def _ollama_embed(self, texts: list[str]) -> list[list[float]]:
        from langchain_ollama import OllamaEmbeddings  # noqa: PLC0415
        client = OllamaEmbeddings(model=self._model, base_url=self._base_url)
        return client.embed_documents(texts)  # type: ignore[return-value]

    def _st_embed(self, texts: list[str]) -> list[list[float]]:
        """Embed using sentence-transformers (offline, multilingual)."""
        if self._st_model is None:
            try:
                # Import inside suppressed warning context
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=r"Accessing `__path__`")
                    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
                    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers\n"
                    "Also ensure Ollama is running: ollama serve"
                ) from exc
            logger.info("Loading sentence-transformers model: %s", self._fallback_model)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"Accessing `__path__`")
                warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
                self._st_model = SentenceTransformer(self._fallback_model)
            logger.info("sentence-transformers model loaded.")
        vecs = self._st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    @staticmethod
    def _is_broken(vectors: list[list[float]]) -> bool:
        if not vectors:
            return True
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1)
        if np.all(norms < _ZERO_THRESHOLD):
            return True
        if len(arr) > 1 and np.max(np.abs(arr - arr[0])) < _ZERO_THRESHOLD:
            return True
        return False
