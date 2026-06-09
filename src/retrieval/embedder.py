"""Chunk embedding — Ollama primary, sentence-transformers fallback.

Primary:  intfloat/multilingual-e5-small via local Ollama daemon
          Pull once:  ollama pull qllama/multilingual-e5-small

Fallback: sentence-transformers (intfloat/multilingual-e5-small, local weights)
          Fully offline — local_files_only=True prevents any HuggingFace HTTP
          requests after the first download. If the model is not yet cached,
          set ALLOW_HF_DOWNLOAD=1 once to fetch it, then remove the var.

multilingual-e5 query prefix convention
-----------------------------------------
  Chunks (passages): "passage: <text>"
  Queries:           "query: <text>"
Omitting these degrades retrieval quality for German compound nouns.

Zero-vector detection: if Ollama returns a zero/constant vector the
embedder automatically switches to the sentence-transformers fallback.
"""
from __future__ import annotations

import logging
import os
import warnings

import numpy as np

from src.models import Chunk

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_MODEL = "qllama/multilingual-e5-small:latest"
_FALLBACK_ST_MODEL = "intfloat/multilingual-e5-small"
_ZERO_THRESHOLD = 1e-6

_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "
_E5_MODEL_SUBSTRINGS = ("multilingual-e5", "e5-small", "e5-base", "e5-large", "e5-mistral")

# Kill all HuggingFace Hub network traffic unless the user explicitly opts in.
# Set ALLOW_HF_DOWNLOAD=1 in the environment only when downloading a model
# for the first time.
_HF_OFFLINE = os.environ.get("ALLOW_HF_DOWNLOAD", "").strip() not in ("1", "true", "yes")
if _HF_OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

warnings.filterwarnings("ignore", message=r"Accessing `__path__`", module="transformers")
warnings.filterwarnings("ignore", message=r"Accessing `__path__`")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def _needs_e5_prefix(model_name: str) -> bool:
    lower = model_name.lower()
    return any(s in lower for s in _E5_MODEL_SUBSTRINGS)


class ChunkEmbedder:
    """Embeds Chunk objects into dense vectors.

    Tries Ollama first (via direct ollama SDK, not langchain_ollama which
    routes through an internal proxy on a random port in 0.3+); falls back
    to sentence-transformers if Ollama is unavailable or returns zero vectors.

    The sentence-transformers fallback loads from the local HuggingFace
    cache only (local_files_only=True). Run with ALLOW_HF_DOWNLOAD=1
    once if the model is not yet cached.

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
                "e5 prefix mode enabled for model '%s'",
                model,
            )

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
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
        """Embed via the ollama SDK directly.

        Uses ollama.Client instead of langchain_ollama.OllamaEmbeddings.
        langchain-ollama 0.3+ routes embed() through an internal HTTP proxy
        on a random ephemeral port, causing 400 errors and wsarecv resets
        on Windows. The ollama SDK hits localhost:11434 directly.
        """
        import ollama  # noqa: PLC0415
        client = ollama.Client(host=self._base_url)
        response = client.embed(model=self._model, input=texts)
        return response.embeddings  # type: ignore[return-value]

    def _st_embed(self, texts: list[str]) -> list[list[float]]:
        """Embed using sentence-transformers from local cache only.

        local_files_only=True means no HTTP requests to HuggingFace.
        If the model is missing from cache, raises an OSError with a
        clear message telling the user to run with ALLOW_HF_DOWNLOAD=1.
        """
        if self._st_model is None:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=r"Accessing `__path__`")
                    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
                    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Run: uv add sentence-transformers"
                ) from exc

            # local_files_only=True: load from HuggingFace cache, no network.
            # If not cached yet: ALLOW_HF_DOWNLOAD=1 python -m src.main
            local_only = _HF_OFFLINE
            logger.info(
                "Loading sentence-transformers model '%s' (local_files_only=%s)",
                self._fallback_model,
                local_only,
            )
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=r"Accessing `__path__`")
                    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
                    self._st_model = SentenceTransformer(
                        self._fallback_model,
                        local_files_only=local_only,
                    )
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Model '{self._fallback_model}' not found in local cache.\n"
                    "Run once with ALLOW_HF_DOWNLOAD=1 to download it:\n"
                    "  ALLOW_HF_DOWNLOAD=1 python -m src.main"
                ) from exc
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
