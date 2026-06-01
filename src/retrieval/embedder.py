"""Chunk embedding using sentence-transformers.

Model: intfloat/multilingual-e5-small (CLAUDE.md recommendation)
- Supports German natively
- Small footprint, runs on CPU
- Fallback: all-MiniLM-L6-v2
"""
from __future__ import annotations

import logging
from functools import cached_property

from src.models import Chunk

logger = logging.getLogger(__name__)

_PRIMARY_MODEL = "intfloat/multilingual-e5-small"
_FALLBACK_MODEL = "all-MiniLM-L6-v2"

# multilingual-e5 requires a task prefix for retrieval
_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "


class ChunkEmbedder:
    """Embeds Chunk objects into dense vectors for storage and retrieval.

    Automatically falls back to all-MiniLM-L6-v2 if the primary model
    cannot be loaded.

    Args:
        model_name: Sentence-transformer model identifier.
        device: Compute device ('cpu', 'cuda', 'mps').
    """

    def __init__(
        self,
        model_name: str = _PRIMARY_MODEL,
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device

    @cached_property
    def _model(self) -> object:
        """Lazy-load the sentence-transformer model."""
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        for name in (self._model_name, _FALLBACK_MODEL):
            try:
                logger.info("Loading embedding model: %s", name)
                return SentenceTransformer(name, device=self._device)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load %s: %s", name, exc)
        raise RuntimeError("No embedding model could be loaded")

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Embed a list of chunks into dense vectors.

        For multilingual-e5 models the passage prefix is prepended.

        Args:
            chunks: Chunks to embed.

        Returns:
            List of embedding vectors, same order as input chunks.
        """
        if not chunks:
            return []

        texts = [
            f"{_PASSAGE_PREFIX}{c.text}"
            if "e5" in self._model_name
            else c.text
            for c in chunks
        ]
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        model: SentenceTransformer = self._model  # type: ignore[assignment]
        vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        logger.debug("Embedded %d chunks", len(chunks))
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Args:
            query: Natural language query (German or English).

        Returns:
            Embedding vector.
        """
        text = f"{_QUERY_PREFIX}{query}" if "e5" in self._model_name else query
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        model: SentenceTransformer = self._model  # type: ignore[assignment]
        vector = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return vector[0].tolist()
