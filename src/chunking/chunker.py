"""Layout-aware chunker that aggregates elements into retrieval-ready Chunks."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.models import Chunk, Element, Page

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CHARS = 1000
_MIN_CHUNK_CHARS = 60
_MIN_BBOX_AREA = 100.0

_NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"EVB-IT\s+Dienst", re.IGNORECASE),
    re.compile(r"^Seite\s+\d+\s+von\s+\d+", re.IGNORECASE),
    re.compile(r"Version\s+\d+\.\d+\s+vom", re.IGNORECASE),
    re.compile(r"Die mit \* gekennzeichneten", re.IGNORECASE),
    re.compile(r"^Vertragsnummer.{0,50}$"),
    re.compile(r"^\s*_+\s*$"),
    re.compile(r"\.{5,}"),
    re.compile(r"^\d{1,2}\s+von\s+\d{1,2}$"),
]

_HEADING_HEURISTICS = (
    lambda e: len(e.text) < 120 and e.text.isupper(),
    lambda e: len(e.text) < 120 and e.text.endswith(":"),
)


@dataclass
class ChunkerConfig:
    max_chars: int = _DEFAULT_MAX_CHARS
    overlap_chars: int = 100
    min_chunk_chars: int = _MIN_CHUNK_CHARS
    deduplicate: bool = True
    heading_lookahead: bool = True  # prepend heading to next chunk body


def _valid_bbox(bbox: list[float]) -> bool:
    if len(bbox) != 4:
        return False
    x0, y0, x1, y1 = bbox
    area = (x1 - x0) * (y1 - y0)
    return area >= _MIN_BBOX_AREA


class LayoutChunker:
    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self._cfg = config or ChunkerConfig()

    def chunk(self, pages: list[Page]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in pages:
            chunks.extend(self._chunk_page(page))
        before = len(chunks)
        chunks = self._filter(chunks)
        logger.info(
            "Chunked %d pages into %d chunks (%d filtered)",
            len(pages), len(chunks), before - len(chunks),
        )
        return chunks

    def _chunk_page(self, page: Page) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffer_texts: list[str] = []
        buffer_bboxes: list[list[float]] = []
        buffer_confidence: list[float] = []
        pending_heading: str | None = None  # heading waiting to be prepended

        def flush(force_heading: str | None = None) -> None:
            nonlocal pending_heading
            if not buffer_texts:
                # If we only have a heading and nothing followed it on this page,
                # emit it alone so it isn't silently dropped.
                if pending_heading:
                    chunks.append(Chunk(
                        text=pending_heading,
                        page_number=page.page_number,
                        bboxes=[b for b in buffer_bboxes if _valid_bbox(b)] or [],
                        chunk_type="text",
                        confidence=0.90,
                        image_path=page.image_path,
                    ))
                    pending_heading = None
                return

            # Prepend the pending heading so the chunk body is searchable by title
            prefix = (pending_heading + "\n") if pending_heading else ""
            text = prefix + " ".join(buffer_texts)
            valid_bboxes = [b for b in buffer_bboxes if _valid_bbox(b)]
            chunks.append(Chunk(
                text=text,
                page_number=page.page_number,
                bboxes=valid_bboxes,
                chunk_type="text",
                confidence=min(buffer_confidence),
                image_path=page.image_path,
            ))
            buffer_texts.clear()
            buffer_bboxes.clear()
            buffer_confidence.clear()
            pending_heading = force_heading  # carry next heading into next chunk

        for element in page.elements:
            if _is_noise(element.text):
                continue

            if element.type == "table":
                flush()
                bbox = element.bbox
                if _valid_bbox(bbox):
                    chunks.append(Chunk(
                        text=element.text,
                        page_number=page.page_number,
                        bboxes=[bbox],
                        chunk_type="table",
                        confidence=element.confidence,
                        image_path=page.image_path,
                    ))
                else:
                    logger.warning(
                        "Dropping table on page %d — invalid bbox %s",
                        page.page_number, bbox,
                    )
                continue

            if self._is_heading(element):
                flush(force_heading=element.text)
                # Don't add heading to buffer — it'll be prepended on next flush
                buffer_bboxes.append(element.bbox)  # keep bbox for coverage
                buffer_confidence.append(element.confidence)
                continue

            current_len = sum(len(t) for t in buffer_texts)
            if current_len + len(element.text) > self._cfg.max_chars and buffer_texts:
                flush()

            buffer_texts.append(element.text)
            buffer_bboxes.append(element.bbox)
            buffer_confidence.append(element.confidence)

        flush()
        return chunks

    def _filter(self, chunks: list[Chunk]) -> list[Chunk]:
        seen: set[str] = set()
        out: list[Chunk] = []
        for c in chunks:
            stripped = c.text.strip()
            if len(stripped) < self._cfg.min_chunk_chars:
                continue
            if _is_noise(stripped):
                continue
            if not c.bboxes:
                logger.debug("Dropping chunk with no valid bboxes: %.60s", stripped)
                continue
            if self._cfg.deduplicate:
                key = re.sub(r"\s+", " ", stripped.lower())
                if key in seen:
                    continue
                seen.add(key)
            out.append(c)
        return out

    @staticmethod
    def _is_heading(element: Element) -> bool:
        return any(h(element) for h in _HEADING_HEURISTICS)


def _is_noise(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    return any(p.search(t) for p in _NOISE_PATTERNS)
