"""FAISS vector store with full bbox metadata preservation.

Persists index + metadata as two files:
    outputs/faiss_index/index.faiss  - FAISS flat cosine index
    outputs/faiss_index/metadata.json - chunk metadata (bboxes, page, etc.)

No chromadb, no transformers, no HuggingFace — fully offline.
"""
from __future__ import annotations

import json
import logging
import numpy as np
from pathlib import Path

from src.models import Chunk

logger = logging.getLogger(__name__)

_INDEX_FILE = "index.faiss"
_META_FILE = "metadata.json"


class FAISSStore:
    """Persisted FAISS vector store for PDF chunks.

    Args:
        persist_dir: Directory where index and metadata are saved.
    """

    def __init__(self, persist_dir: Path = Path("outputs/faiss_index")) -> None:
        self._persist_dir = persist_dir
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._persist_dir / _INDEX_FILE
        self._meta_path = self._persist_dir / _META_FILE
        self._index: object = None
        self._metadata: list[dict[str, object]] = []
        self._texts: list[str] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load existing index and metadata from disk if present."""
        import faiss  # noqa: PLC0415
        if self._index_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            data = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._metadata = data["metadata"]
            self._texts = data["texts"]
            logger.info("Loaded FAISS index with %d vectors", len(self._metadata))
        else:
            self._index = None

    def _save(self) -> None:
        """Persist index and metadata to disk."""
        import faiss  # noqa: PLC0415
        faiss.write_index(self._index, str(self._index_path))  # type: ignore[arg-type]
        self._meta_path.write_text(
            json.dumps({"metadata": self._metadata, "texts": self._texts}, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        doc_id: str,
    ) -> None:
        """Store chunks and their embeddings.

        Existing entries for the same doc_id are replaced (upsert by doc_id).

        Args:
            chunks: Chunks to store.
            embeddings: Corresponding embedding vectors (same order).
            doc_id: Source document identifier.

        Raises:
            ValueError: If chunks and embeddings lengths differ.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
            )

        import faiss  # noqa: PLC0415

        # remove existing entries for this doc_id
        keep = [(t, m) for t, m in zip(self._texts, self._metadata) if m["doc_id"] != doc_id]
        kept_texts = [k[0] for k in keep]
        kept_meta = [k[1] for k in keep]

        new_texts = [c.text for c in chunks]
        new_meta = [
            {
                "doc_id": doc_id,
                "page_number": c.page_number,
                "chunk_type": c.chunk_type,
                "confidence": c.confidence,
                "image_path": c.image_path,
                "bboxes": c.bboxes,
            }
            for c in chunks
        ]

        self._texts = kept_texts + new_texts
        self._metadata = kept_meta + new_meta

        # rebuild index from all vectors
        all_embeddings = self._rebuild_embeddings(kept_meta) + embeddings
        vectors = np.array(all_embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)  # cosine similarity via inner product

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)  # type: ignore[union-attr]
        self._index = index
        self._save()
        logger.info("Stored %d chunks for doc '%s' (%d total)", len(chunks), doc_id, len(self._texts))

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        filter_doc_id: str | None = None,
        filter_chunk_type: str | None = None,
    ) -> list[Chunk]:
        """Retrieve the top-k most similar chunks.

        Args:
            query_embedding: Dense query vector.
            n_results: Number of results to return.
            filter_doc_id: Optional filter by document.
            filter_chunk_type: Optional filter by chunk type.

        Returns:
            List of Chunk objects ordered by similarity (best first).
        """
        if self._index is None or len(self._metadata) == 0:
            return []

        import faiss  # noqa: PLC0415

        vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(vec)

        # fetch more candidates to allow post-filtering
        fetch_k = min(len(self._metadata), n_results * 10)
        distances, indices = self._index.search(vec, fetch_k)  # type: ignore[union-attr]

        chunks: list[Chunk] = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            if filter_doc_id and meta["doc_id"] != filter_doc_id:
                continue
            if filter_chunk_type and meta["chunk_type"] != filter_chunk_type:
                continue
            chunks.append(Chunk(
                text=self._texts[idx],
                page_number=int(meta["page_number"]),
                bboxes=meta["bboxes"],  # type: ignore[arg-type]
                chunk_type=meta["chunk_type"],  # type: ignore[arg-type]
                confidence=float(meta["confidence"]),
                image_path=str(meta["image_path"]),
            ))
            if len(chunks) >= n_results:
                break

        return chunks

    def count(self) -> int:
        """Return total number of stored chunks."""
        return len(self._metadata)

    def _rebuild_embeddings(self, metadata: list[dict[str, object]]) -> list[list[float]]:
        """Placeholder — kept entries don't need re-embedding; return empty list.

        In practice we store all vectors in the FAISS index and keep/drop
        by rebuilding the full index on each add_chunks call. For the PoC
        scale this is fine.
        """
        # For simplicity in PoC: if there are kept entries, reload from saved index
        if not metadata or self._index_path.exists() is False:
            return []
        import faiss  # noqa: PLC0415
        old_index = faiss.read_index(str(self._index_path))
        n_kept = len(metadata)
        if n_kept == 0:
            return []
        vecs = old_index.reconstruct_n(0, old_index.ntotal)  # type: ignore[union-attr]
        # only keep first n_kept rows (those not removed)
        return vecs[:n_kept].tolist()
