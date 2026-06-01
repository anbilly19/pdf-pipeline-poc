"""Chroma vector store with full bbox metadata preservation.

Every chunk stored here carries its bboxes, page number, chunk_type,
confidence, and image_path in metadata — the bbox chain is never broken.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.models import Chunk

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "pdf_chunks"


class ChromaStore:
    """Persisted Chroma vector store for PDF chunks.

    Args:
        persist_dir: Directory where Chroma persists its database.
        collection_name: Chroma collection identifier.
    """

    def __init__(
        self,
        persist_dir: Path = Path("outputs/chroma_db"),
        collection_name: str = _COLLECTION_NAME,
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._persist_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _collection(self) -> object:
        """Lazy-initialise Chroma client and collection."""
        import chromadb  # noqa: PLC0415

        client = chromadb.PersistentClient(path=str(self._persist_dir))
        return client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        doc_id: str,
    ) -> None:
        """Store chunks and their embeddings in Chroma.

        Bboxes are JSON-serialised into metadata since Chroma only
        accepts flat string/int/float metadata values.

        Args:
            chunks: Chunks to store.
            embeddings: Corresponding embedding vectors (same order).
            doc_id: Source document identifier (e.g. filename stem).

        Raises:
            ValueError: If chunks and embeddings lengths differ.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
            )

        ids = [f"{doc_id}__p{c.page_number}__c{i}" for i, c in enumerate(chunks)]
        metadatas = [
            {
                "doc_id": doc_id,
                "page_number": c.page_number,
                "chunk_type": c.chunk_type,
                "confidence": c.confidence,
                "image_path": c.image_path,
                "bboxes": json.dumps(c.bboxes),  # serialised — deserialise on retrieval
            }
            for c in chunks
        ]
        documents = [c.text for c in chunks]

        self._collection.upsert(  # type: ignore[union-attr]
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Stored %d chunks for doc '%s'", len(chunks), doc_id)

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        filter_doc_id: str | None = None,
        filter_chunk_type: str | None = None,
    ) -> list[Chunk]:
        """Retrieve the top-k most similar chunks.

        Bboxes are deserialised from JSON metadata back into list[list[float]].
        The bbox chain is preserved end-to-end.

        Args:
            query_embedding: Dense query vector.
            n_results: Number of results to return.
            filter_doc_id: Optional Chroma metadata filter by document.
            filter_chunk_type: Optional filter by chunk type ('text', 'table', 'figure').

        Returns:
            List of Chunk objects ordered by similarity (best first).
        """
        where: dict[str, object] = {}
        if filter_doc_id and filter_chunk_type:
            where = {"$and": [{"doc_id": filter_doc_id}, {"chunk_type": filter_chunk_type}]}
        elif filter_doc_id:
            where = {"doc_id": filter_doc_id}
        elif filter_chunk_type:
            where = {"chunk_type": filter_chunk_type}

        kwargs: dict[str, object] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)  # type: ignore[union-attr]

        chunks: list[Chunk] = []
        for text, meta in zip(
            results["documents"][0],
            results["metadatas"][0],
        ):
            chunks.append(
                Chunk(
                    text=text,
                    page_number=int(meta["page_number"]),
                    bboxes=json.loads(meta["bboxes"]),
                    chunk_type=meta["chunk_type"],  # type: ignore[arg-type]
                    confidence=float(meta["confidence"]),
                    image_path=str(meta["image_path"]),
                )
            )

        return chunks

    def count(self) -> int:
        """Return total number of stored chunks.

        Returns:
            Integer count.
        """
        return self._collection.count()  # type: ignore[union-attr, return-value]
