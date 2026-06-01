"""Chunk embedding using Ollama — fully offline, zero HuggingFace calls.

Default model: nomic-embed-text
- Runs entirely via local Ollama daemon
- Pull once: ollama pull nomic-embed-text
"""
from __future__ import annotations

import logging

from src.models import Chunk

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "nomic-embed-text"


class ChunkEmbedder:
    """Embeds Chunk objects into dense vectors via Ollama.

    Args:
        model: Ollama embedding model identifier.
        base_url: Ollama API base URL.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._base_url = base_url

    def _client(self) -> object:
        from langchain_ollama import OllamaEmbeddings  # noqa: PLC0415
        return OllamaEmbeddings(model=self._model, base_url=self._base_url)

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Embed a list of chunks into dense vectors.

        Args:
            chunks: Chunks to embed.

        Returns:
            List of embedding vectors, same order as input chunks.
        """
        if not chunks:
            return []

        texts = [c.text for c in chunks]
        client = self._client()
        vectors = client.embed_documents(texts)  # type: ignore[union-attr]
        logger.debug("Embedded %d chunks via Ollama (%s)", len(chunks), self._model)
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Args:
            query: Natural language query.

        Returns:
            Embedding vector.
        """
        client = self._client()
        vector = client.embed_query(query)  # type: ignore[union-attr]
        return vector
