"""Chunk embedding — Ollama primary, sentence-transformers fallback.

Primary:  intfloat/multilingual-e5-small via local Ollama daemon
          Pull once:  ollama pull qllama/multilingual-e5-small
          Significantly better German umlaut and compound-noun retrieval
          than the previous nomic-embed-text model.

Fallback: sentence-transformers (intfloat/multilingual-e5-small, local weights)
          Fully offline, no Ollama required.
          Downloads model weights on first use (~120 MB).

multilingual-e5 query prefix convention
-----------------------------------------
The e5 family expects a task prefix on every input:
  - Chunks (passages) are indexed as-is:
      "passage: <text>"
  - Queries are prefixed with:
      "query: <text>"
Omitting these prefixes degrades retrieval quality, especially for
short queries against long passages.

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

# Full registered name as shown by `ollama list`
_DEFAULT_OLLAMA_MODEL = "qllama/multilingual-e5-small:latest"
_FALLBACK_ST_MODEL = "intfloat/multilingual-e5-small"
_ZERO_THRESHOLD = 1e-6

# e5 models require task-specific prefixes for optimal retrieval quality.
_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "

# Models that require the e5 prefix convention.
_E5_MODEL_SUBSTRINGS = ("multilingual-e5", "e5-small", "e5-base", "e5-large", "e5-mistral")

warnings.filterwarnings("ignore", message=r"Accessing `__path__`", module="transformers")
warnings.filterwarnings("ignore", message=r"Accessing `__path__`")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def _needs_e5_prefix(model_name: str) -> bool:
    """Return True if this model requires the 'query:' / 'passage:' prefix convention."""
    lower = model_name.lower()
    return any(s in lower for s in _E5_MODEL_SUBSTRINGS)


class ChunkEmbedder:
    """Embeds Chunk objects into dense vectors.

    Tries Ollama first; falls back to sentence-transformers if Ollama
    is unavailable or returns zero/constant vectors.

    Args:
        model: Ollama embedding model identifier.
        base_url: Ollama API base URL.
        fallback_model: sentence-transformers model name for offline fallback.
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
        self._apply_e5_prefix = _needs_e5_prefix(model)
        if self._apply_e5_prefix:
            logger.info(
                "e5 prefix mode enabled for model '%s' — passages prefixed with 'passage:', "
                "queries prefixed with 'query:'",
                model,
            )

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Embed a list of chunks into dense vectors."""
        if not chunks:
            return []
        texts = self._apply_passage_prefix([c.text for c in chunks])
        vectors = self._embed_texts(texts)
        logger.info(
            "Embedded %d chunks via %s",
            len(chunks),
            "sentence-transformers" if self._use_fallback else f"Ollama/{self._model}",
        )
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        text = f"{_E5_QUERY_PREFIX}{query}" if self._apply_e5_prefix else query
        return self._embed_texts([text])[0]

    def _apply_passage_prefix(self, texts: list[str]) -> list[str]:
        if not self._apply_e5_prefix:
            return texts
        return [f"{_E5_PASSAGE_PREFIX}{t}" for t in texts]

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
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=r"Accessing `__path__`")
                    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
                    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Run: uv add sentence-transformers\n"
                    "Also ensure Ollama is running: ollama serve"
                ) from exc
            logger.info("Loading sentence-transformers model: %s", self._fallback_model)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"Accessing `__path__`")
                warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
                self._st_model = SentenceTransformer(self._fallback_model)
            logger.info("sentence-transformers model loaded.")
        vecs = self._st_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
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
