"""Tests for embedding, FAISS store, and retriever."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.models import Chunk
from src.retrieval.store import FAISSStore
from src.retrieval.retriever import BBoxRetriever
from src.retrieval.embedder import ChunkEmbedder


def _make_chunk(
    text: str = "Test",
    page: int = 1,
    bboxes: list[list[float]] | None = None,
    chunk_type: str = "text",
    confidence: float = 0.9,
) -> Chunk:
    return Chunk(
        text=text,
        page_number=page,
        bboxes=bboxes or [[0.0, 0.0, 100.0, 20.0]],
        chunk_type=chunk_type,  # type: ignore[arg-type]
        confidence=confidence,
        image_path="/tmp/p1.png",
    )


@pytest.fixture()
def store(tmp_path: Path) -> FAISSStore:
    return FAISSStore(persist_dir=tmp_path / "faiss")


def test_store_add_and_count(store: FAISSStore) -> None:
    chunks = [_make_chunk(f"chunk {i}") for i in range(3)]
    embeddings = [[float(i) * 0.1] * 384 for i in range(3)]
    store.add_chunks(chunks, embeddings, doc_id="doc1")
    assert store.count() == 3


def test_store_bboxes_roundtrip(store: FAISSStore) -> None:
    original_bboxes = [[10.0, 20.0, 110.0, 40.0], [10.0, 50.0, 110.0, 70.0]]
    chunk = _make_chunk(bboxes=original_bboxes)
    store.add_chunks([chunk], [[0.1] * 384], doc_id="doc1")
    results = store.query([0.1] * 384, n_results=1)
    assert len(results) == 1
    assert results[0].bboxes == original_bboxes


def test_store_metadata_integrity(store: FAISSStore) -> None:
    chunk = _make_chunk(text="Table content", page=5, chunk_type="table", confidence=0.75)
    chunk.image_path = "/tmp/p5.png"
    store.add_chunks([chunk], [[0.5] * 384], doc_id="doc_meta")
    results = store.query([0.5] * 384, n_results=1)
    r = results[0]
    assert r.page_number == 5
    assert r.chunk_type == "table"
    assert r.confidence == 0.75
    assert r.image_path == "/tmp/p5.png"


def test_store_upsert_replaces_doc(store: FAISSStore) -> None:
    chunks_v1 = [_make_chunk("version 1")]
    chunks_v2 = [_make_chunk("version 2"), _make_chunk("version 2b")]
    store.add_chunks(chunks_v1, [[0.1] * 384], doc_id="doc1")
    store.add_chunks(chunks_v2, [[0.2] * 384, [0.3] * 384], doc_id="doc1")
    assert store.count() == 2


def test_store_length_mismatch_raises(store: FAISSStore) -> None:
    with pytest.raises(ValueError, match="must match"):
        store.add_chunks([_make_chunk()], [], doc_id="x")


def test_store_empty_query_returns_empty(store: FAISSStore) -> None:
    results = store.query([0.1] * 384, n_results=5)
    assert results == []


def test_retriever_calls_embedder_and_store() -> None:
    mock_store = MagicMock(spec=FAISSStore)
    mock_embedder = MagicMock(spec=ChunkEmbedder)
    mock_embedder.embed_query.return_value = [0.1] * 384
    mock_store.query.return_value = [_make_chunk("result")]
    retriever = BBoxRetriever(store=mock_store, embedder=mock_embedder, top_k=3)
    results = retriever.retrieve("What is the topic?")
    mock_embedder.embed_query.assert_called_once_with("What is the topic?")
    mock_store.query.assert_called_once()
    assert len(results) == 1


def test_retriever_as_sources_preserves_bboxes() -> None:
    bboxes = [[5.0, 10.0, 50.0, 30.0]]
    mock_store = MagicMock(spec=FAISSStore)
    mock_embedder = MagicMock(spec=ChunkEmbedder)
    mock_embedder.embed_query.return_value = [0.0] * 384
    mock_store.query.return_value = [_make_chunk(bboxes=bboxes, page=3)]
    retriever = BBoxRetriever(store=mock_store, embedder=mock_embedder)
    sources = retriever.retrieve_as_sources("Question")
    assert sources[0].bboxes == bboxes
    assert sources[0].page == 3
