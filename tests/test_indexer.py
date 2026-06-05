"""Tests for DocumentIndexer including classifier integration.

All external I/O (FAISS, embedder, pipeline, classifier) is mocked.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models import Chunk
from src.indexer import DocumentIndexer
from src.agent.domain_config import DocTypeConfig, DomainSpec


def _make_chunk(text: str = "Test chunk") -> Chunk:
    return Chunk(
        text=text,
        page_number=1,
        bboxes=[[0.0, 0.0, 100.0, 20.0]],
        chunk_type="text",
        confidence=0.9,
        image_path="",
    )


def _make_doc_type_config(doc_type: str = "contract") -> DocTypeConfig:
    general = DomainSpec(
        name="general", display_name="General",
        keywords_de=[], keywords_en=[], search_terms=[],
        model="qwen2.5:3b", system_prompt="test prompt",
    )
    return DocTypeConfig(
        doc_type=doc_type,
        display_name=doc_type.capitalize(),
        detection_hints=[],
        domains={"general": general},
    )


@pytest.fixture()
def mock_indexer() -> DocumentIndexer:
    """DocumentIndexer with all external deps mocked."""
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = [_make_chunk(f"chunk {i}") for i in range(5)]

    mock_embedder = MagicMock()
    mock_embedder.embed_chunks.return_value = [[0.1] * 384] * 5

    mock_store = MagicMock()
    mock_store.add_chunks.return_value = None

    indexer = DocumentIndexer(embedder=mock_embedder, store=mock_store)
    indexer._pipeline = mock_pipeline
    return indexer


class TestDocumentIndexer:
    def test_index_returns_chunk_count(self, mock_indexer: DocumentIndexer, tmp_path: Path) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.touch()
        with patch("src.indexer.classify_document", return_value=_make_doc_type_config()):
            n = mock_indexer.index(pdf)
        assert n == 5

    def test_index_calls_classifier(self, mock_indexer: DocumentIndexer, tmp_path: Path) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.touch()
        with patch("src.indexer.classify_document", return_value=_make_doc_type_config()) as mock_clf:
            mock_indexer.index(pdf)
        mock_clf.assert_called_once()
        # Classifier receives the chunks
        call_args = mock_clf.call_args
        assert len(call_args.args[0]) == 5

    def test_index_stores_last_doc_type(self, mock_indexer: DocumentIndexer, tmp_path: Path) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.touch()
        expected = _make_doc_type_config("magazine")
        with patch("src.indexer.classify_document", return_value=expected):
            mock_indexer.index(pdf)
        assert mock_indexer.last_doc_type is not None
        assert mock_indexer.last_doc_type.doc_type == "magazine"

    def test_index_empty_chunks_returns_zero(self, tmp_path: Path) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = []  # no chunks
        mock_embedder = MagicMock()
        mock_store = MagicMock()
        indexer = DocumentIndexer(embedder=mock_embedder, store=mock_store)
        indexer._pipeline = mock_pipeline
        pdf = tmp_path / "empty.pdf"
        pdf.touch()
        with patch("src.indexer.classify_document") as mock_clf:
            n = indexer.index(pdf)
        assert n == 0
        mock_clf.assert_not_called()  # classifier not called if no chunks

    def test_index_uses_filename_stem_as_doc_id(self, mock_indexer: DocumentIndexer, tmp_path: Path) -> None:
        pdf = tmp_path / "my_document.pdf"
        pdf.touch()
        with patch("src.indexer.classify_document", return_value=_make_doc_type_config()):
            mock_indexer.index(pdf)
        mock_indexer._store.add_chunks.assert_called_once()
        call_kwargs = mock_indexer._store.add_chunks.call_args
        assert call_kwargs.kwargs["doc_id"] == "my_document"

    def test_index_respects_custom_doc_id(self, mock_indexer: DocumentIndexer, tmp_path: Path) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.touch()
        with patch("src.indexer.classify_document", return_value=_make_doc_type_config()):
            mock_indexer.index(pdf, doc_id="custom_id")
        call_kwargs = mock_indexer._store.add_chunks.call_args
        assert call_kwargs.kwargs["doc_id"] == "custom_id"
