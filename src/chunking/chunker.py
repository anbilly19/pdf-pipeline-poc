"""Layout-aware chunker that aggregates elements into retrieval-ready Chunks.

Splits on:
1. ODL/PyMuPDF element boundaries (primary)
2. Section headings (§ N, digits + word) detected via heuristic
3. max_chars overflow
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.models import Chunk, Element, Page

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CHARS = 800
_MIN_CHUNK_CHARS = 40
_MIN_BBOX_AREA = 50.0  # relaxed from 100 to avoid dropping real content

_NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"EVB-IT\s+Dienst", re.IGNORECASE),
    re.compile(r"^Seite\s+\d+\s+von\s+\d+", re.IGNORECASE),
    re.compile(r"Version\s+\d+\.\d+\s+vom", re.IGNORECASE),
    re.compile(r"Die mit \* gekennzeichneten", re.IGNORECASE),
    re.compile(r"^\s*_+\s*$"),
    re.compile(r"\.{5,}"),
    re.compile(r"^\d{1,2}\s+von\s+\d{1,2}$"),
]

# Matches section headings like:
#   "15 Laufzeit und Kündigung"
#   "§ 15 Laufzeit"
#   "11 Schlechtleistung"
_SECTION_RE = re.compile(
    r"^(?:§\s*)?\d{1,2}(?:\.\d)?\s+[A-ZÄÖÜ][\wäöüÄÖÜ\s\-,]{3,60}$"
)


@dataclass
class ChunkerConfig:
    max_chars: int = _DEFAULT_MAX_CHARS
    overlap_chars: int = 80
    min_chunk_chars: int = _MIN_CHUNK_CHARS
    deduplicate: bool = True
    heading_lookahead: bool = True


def _valid_bbox(bbox: list[float]) -> bool:
    if len(bbox) != 4:
        return False
    x0, y0, x1, y1 = bbox
    return (x1 - x0) * (y1 - y0) >= _MIN_BBOX_AREA


def _is_section_heading(text: str) -> bool:
    """True if text looks like a numbered section heading."""
    return bool(_SECTION_RE.match(text.strip()))


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
            "Chunked %d pages → %d chunks (%d filtered out)",
            len(pages), len(chunks), before - len(chunks),
        )
        return chunks

    def _chunk_page(self, page: Page) -> list[Chunk]:
        """Split a page into chunks, respecting section boundaries.

        Strategy:
        - Each element is either a heading, table, or body text.
        - A heading always flushes the current buffer and starts a new chunk
          (the heading text is prepended to the following body).
        - A body element that would push the buffer over max_chars flushes first.
        - Tables are always their own chunk.
        """
        chunks: list[Chunk] = []
        buf_texts: list[str] = []
        buf_bboxes: list[list[float]] = []
        buf_conf: list[float] = []
        pending_heading: str | None = None

        def flush(next_heading: str | None = None) -> None:
            nonlocal pending_heading
            if buf_texts:
                prefix = (pending_heading + "\n") if pending_heading else ""
                text = prefix + " ".join(buf_texts)
                valid_bboxes = [b for b in buf_bboxes if _valid_bbox(b)]
                chunks.append(Chunk(
                    text=text,
                    page_number=page.page_number,
                    bboxes=valid_bboxes or buf_bboxes[:1],
                    chunk_type="text",
                    confidence=min(buf_conf) if buf_conf else 0.8,
                    image_path=page.image_path,
                ))
                buf_texts.clear()
                buf_bboxes.clear()
                buf_conf.clear()
            elif pending_heading:
                # Heading with no body yet — emit it alone
                valid_bboxes = [b for b in buf_bboxes if _valid_bbox(b)]
                chunks.append(Chunk(
                    text=pending_heading,
                    page_number=page.page_number,
                    bboxes=valid_bboxes or [],
                    chunk_type="text",
                    confidence=0.90,
                    image_path=page.image_path,
                ))
                buf_bboxes.clear()
                buf_conf.clear()
            pending_heading = next_heading

        for element in page.elements:
            if _is_noise(element.text):
                continue

            if element.type == "table":
                flush()
                if _valid_bbox(element.bbox):
                    chunks.append(Chunk(
                        text=element.text,
                        page_number=page.page_number,
                        bboxes=[element.bbox],
                        chunk_type="table",
                        confidence=element.confidence,
                        image_path=page.image_path,
                    ))
                continue

            # Detect section heading in element text
            # PyMuPDF often merges multiple sections into one element;
            # split on internal section boundaries too.
            sub_parts = _split_on_sections(element.text)

            for part_text, is_heading in sub_parts:
                part_text = part_text.strip()
                if not part_text:
                    continue

                if is_heading or _is_section_heading(part_text):
                    flush(next_heading=part_text)
                    buf_bboxes.append(element.bbox)
                    buf_conf.append(element.confidence)
                    continue

                current_len = sum(len(t) for t in buf_texts)
                if current_len + len(part_text) > self._cfg.max_chars and buf_texts:
                    flush(next_heading=pending_heading)

                buf_texts.append(part_text)
                buf_bboxes.append(element.bbox)
                buf_conf.append(element.confidence)

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
            if self._cfg.deduplicate:
                key = re.sub(r"\s+", " ", stripped.lower()[:200])
                if key in seen:
                    continue
                seen.add(key)
            out.append(c)
        return out


def _split_on_sections(text: str) -> list[tuple[str, bool]]:
    """Split a multi-section blob into (text, is_heading) pairs.

    Handles PyMuPDF's tendency to return an entire page as one element
    with embedded section markers like '15 Laufzeit und Kündigung ...'.

    Returns list of (text_fragment, is_heading) tuples.
    """
    # Split on lines that look like section headings
    lines = text.split("\n")
    if len(lines) <= 1:
        return [(text, False)]

    parts: list[tuple[str, bool]] = []
    buf: list[str] = []

    for line in lines:
        stripped = line.strip()
        if _is_section_heading(stripped):
            if buf:
                parts.append((" ".join(buf), False))
                buf = []
            parts.append((stripped, True))
        else:
            if stripped:
                buf.append(stripped)

    if buf:
        parts.append((" ".join(buf), False))

    return parts if parts else [(text, False)]


def _is_noise(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    return any(p.search(t) for p in _NOISE_PATTERNS)
