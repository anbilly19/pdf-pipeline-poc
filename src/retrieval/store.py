"""FAISS vector store with full bbox metadata preservation."""
from __future__ import annotations

import json
import logging
import os
import numpy as np
from pathlib import Path

from src.models import Chunk

logger = logging.getLogger(__name__)
_DBG = os.environ.get("DEBUG_PIPELINE", "0") == "1"

_INDEX_FILE = "index.faiss"
_META_FILE = "metadata.json"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_PERSIST_DIR = _REPO_ROOT / "outputs" / "faiss_index"


class FAISSStore:
    def __init__(self, persist_dir: Path = _DEFAULT_PERSIST_DIR) -> None:
        self._persist_dir = persist_dir.resolve()
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._persist_dir / _INDEX_FILE
        self._meta_path = self._persist_dir / _META_FILE
        self._index: object = None
        self._metadata: list[dict[str, object]] = []
        self._texts: list[str] = []
        self._load()

    def _load(self) -> None:
        import faiss  # noqa: PLC0415
        if self._index_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            data = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._metadata = data["metadata"]
            self._texts = data["texts"]
            logger.info("Loaded FAISS index with %d vectors from %s", len(self._metadata), self._persist_dir)
            if _DBG and self._metadata:
                m0 = self._metadata[0]
                logger.info(
                    "[DBG-STORE-LOAD] first entry: page=%s  bboxes=%s  image_path=%r",
                    m0.get("page_number"), m0.get("bboxes"), m0.get("image_path"),
                )
        else:
            self._index = None
            logger.info("No existing FAISS index at %s", self._persist_dir)

    def _save(self) -> None:
        import faiss  # noqa: PLC0415
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))  # type: ignore[arg-type]
        self._meta_path.write_text(
            json.dumps({"metadata": self._metadata, "texts": self._texts}, ensure_ascii=False),
            encoding="utf-8",
        )

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        doc_id: str,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match")

        import faiss  # noqa: PLC0415

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

        # --- CHECKPOINT 4: what goes into the store ---
        if _DBG and new_meta:
            m0 = new_meta[0]
            logger.info(
                "[DBG-CP4-ADD] first chunk being stored: page=%s  bboxes=%s  image_path=%r",
                m0["page_number"], m0["bboxes"], m0["image_path"],
            )
            zero_bbox = sum(
                1 for m in new_meta
                if not m["bboxes"] or m["bboxes"] == [[0.0, 0.0, 0.0, 0.0]]
            )
            no_img = sum(1 for m in new_meta if not m["image_path"])
            logger.info(
                "[DBG-CP4-ADD] zero/empty bboxes: %d/%d  empty image_path: %d/%d",
                zero_bbox, len(new_meta), no_img, len(new_meta),
            )

        self._texts = kept_texts + new_texts
        self._metadata = kept_meta + new_meta

        all_embeddings = self._rebuild_embeddings(kept_meta) + embeddings
        vectors = np.array(all_embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)

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
        if self._index is None or len(self._metadata) == 0:
            return []

        import faiss  # noqa: PLC0415

        vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(vec)

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

        # --- CHECKPOINT 5: what comes out of the store ---
        if _DBG and chunks:
            c0 = chunks[0]
            logger.info(
                "[DBG-CP5-QUERY] first retrieved chunk: page=%d  bboxes=%s  image_path=%r",
                c0.page_number, c0.bboxes, c0.image_path,
            )

        return chunks

    def get_all_chunks(self) -> list[Chunk]:
        return [
            Chunk(
                text=self._texts[i],
                page_number=int(meta["page_number"]),
                bboxes=meta["bboxes"],  # type: ignore[arg-type]
                chunk_type=meta["chunk_type"],  # type: ignore[arg-type]
                confidence=float(meta["confidence"]),
                image_path=str(meta["image_path"]),
            )
            for i, meta in enumerate(self._metadata)
        ]

    def count(self) -> int:
        return len(self._metadata)

    def _rebuild_embeddings(self, metadata: list[dict[str, object]]) -> list[list[float]]:
        if not metadata or not self._index_path.exists():
            return []
        import faiss  # noqa: PLC0415
        old_index = faiss.read_index(str(self._index_path))
        n_kept = len(metadata)
        if n_kept == 0:
            return []
        vecs = old_index.reconstruct_n(0, old_index.ntotal)  # type: ignore[union-attr]
        return vecs[:n_kept].tolist()
