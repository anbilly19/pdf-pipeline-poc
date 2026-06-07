"""TDD tests for Roadmap #2 — multilingual-e5-small embedding upgrade.

Fully offline — all Ollama and sentence-transformers calls are mocked.
Verifies:
- e5 prefix applied to queries and passages for e5 models
- prefix NOT applied for non-e5 models (e.g. nomic-embed-text)
- fallback path still works with prefixed texts
- zero-vector detection still triggers fallback
- _needs_e5_prefix helper covers all e5 variants
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models import Chunk
from src.retrieval.embedder import ChunkEmbedder, _needs_e5_prefix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(text: str) -> Chunk:
    return Chunk(
        text=text,
        page_number=1,
        bboxes=[[0, 0, 100, 20]],
        chunk_type="text",
        confidence=1.0,
        image_path="",
    )


def _unit_vec(dim: int = 4) -> list[float]:
    v = np.ones(dim, dtype=np.float32)
    return (v / np.linalg.norm(v)).tolist()


# ---------------------------------------------------------------------------
# _needs_e5_prefix helper
# ---------------------------------------------------------------------------

class TestNeedsE5Prefix:
    @pytest.mark.parametrize("model", [
        "multilingual-e5-small",
        "multilingual-e5-base",
        "multilingual-e5-large",
        "intfloat/multilingual-e5-small",
        "e5-small",
        "e5-base",
        "e5-large",
        "e5-mistral-7b-instruct",
        "MULTILINGUAL-E5-SMALL",  # case-insensitive
    ])
    def test_e5_models_need_prefix(self, model: str) -> None:
        assert _needs_e5_prefix(model) is True

    @pytest.mark.parametrize("model", [
        "nomic-embed-text",
        "bge-reranker-v2-m3",
        "llama3",
        "mxbai-embed-large",
    ])
    def test_non_e5_models_no_prefix(self, model: str) -> None:
        assert _needs_e5_prefix(model) is False


# ---------------------------------------------------------------------------
# e5 prefix applied correctly
# ---------------------------------------------------------------------------

class TestE5PrefixApplication:
    def _make_embedder(self) -> tuple[ChunkEmbedder, list[list[str]]]:
        """Return embedder + captured texts list."""
        captured: list[list[str]] = []

        def fake_ollama_embed(self_inner, texts):  # noqa: ANN001
            captured.append(texts)
            return [_unit_vec() for _ in texts]

        embedder = ChunkEmbedder(model="multilingual-e5-small")
        embedder._ollama_embed = lambda texts: (captured.append(texts), [_unit_vec() for _ in texts])[1]  # noqa: E731
        return embedder, captured

    def test_query_gets_query_prefix(self) -> None:
        embedder, captured = self._make_embedder()
        with patch.object(embedder, "_ollama_embed", side_effect=lambda t: (captured.append(t), [_unit_vec() for _ in t])[1]):
            embedder.embed_query("Was ist die Reaktionszeit?")
        assert captured[0][0].startswith("query: ")

    def test_passage_gets_passage_prefix(self) -> None:
        embedder, captured = self._make_embedder()
        chunks = [_chunk("Die Reaktionszeit beträgt 48 Stunden.")]
        with patch.object(embedder, "_ollama_embed", side_effect=lambda t: (captured.append(t), [_unit_vec() for _ in t])[1]):
            embedder.embed_chunks(chunks)
        assert captured[0][0].startswith("passage: ")

    def test_original_text_preserved_after_prefix(self) -> None:
        """The original chunk text must appear after the prefix, unmodified."""
        embedder, captured = self._make_embedder()
        raw = "Kündigungsfrist beträgt 30 Tage."
        chunks = [_chunk(raw)]
        with patch.object(embedder, "_ollama_embed", side_effect=lambda t: (captured.append(t), [_unit_vec() for _ in t])[1]):
            embedder.embed_chunks(chunks)
        assert captured[0][0] == f"passage: {raw}"


# ---------------------------------------------------------------------------
# Non-e5 models: no prefix
# ---------------------------------------------------------------------------

class TestNomicNoPrefixApplied:
    def test_nomic_query_no_prefix(self) -> None:
        embedder = ChunkEmbedder(model="nomic-embed-text")
        captured: list[list[str]] = []
        with patch.object(
            embedder, "_ollama_embed",
            side_effect=lambda t: (captured.append(t), [_unit_vec() for _ in t])[1],
        ):
            embedder.embed_query("Reaktionszeit")
        assert not captured[0][0].startswith("query: ")
        assert captured[0][0] == "Reaktionszeit"

    def test_nomic_passage_no_prefix(self) -> None:
        embedder = ChunkEmbedder(model="nomic-embed-text")
        captured: list[list[str]] = []
        chunks = [_chunk("Text ohne Präfix.")]
        with patch.object(
            embedder, "_ollama_embed",
            side_effect=lambda t: (captured.append(t), [_unit_vec() for _ in t])[1],
        ):
            embedder.embed_chunks(chunks)
        assert not captured[0][0].startswith("passage: ")


# ---------------------------------------------------------------------------
# Fallback path still receives prefixed texts
# ---------------------------------------------------------------------------

class TestFallbackWithPrefix:
    def test_st_fallback_receives_prefixed_texts(self) -> None:
        """When Ollama fails, sentence-transformers fallback gets the already-prefixed text."""
        import urllib.error

        embedder = ChunkEmbedder(model="multilingual-e5-small")
        captured_st: list[list[str]] = []

        mock_st = MagicMock()
        mock_st.encode.side_effect = lambda texts, **kw: (
            captured_st.append(texts),
            np.array([_unit_vec() for _ in texts], dtype=np.float32),
        )[1]
        embedder._st_model = mock_st

        with patch.object(embedder, "_ollama_embed", side_effect=ConnectionError("offline")):
            embedder.embed_query("Vertragsstrafe")

        assert captured_st[0][0] == "query: Vertragsstrafe"


# ---------------------------------------------------------------------------
# Zero-vector detection still switches to fallback
# ---------------------------------------------------------------------------

class TestZeroVectorFallbackStillWorks:
    def test_zero_vector_triggers_fallback(self) -> None:
        embedder = ChunkEmbedder(model="multilingual-e5-small")
        zero_vec = [0.0] * 4

        mock_st = MagicMock()
        mock_st.encode.return_value = np.array([_unit_vec()], dtype=np.float32)
        embedder._st_model = mock_st

        with patch.object(embedder, "_ollama_embed", return_value=[zero_vec]):
            result = embedder.embed_query("test")

        assert embedder._use_fallback is True
        assert len(result) == len(_unit_vec())
