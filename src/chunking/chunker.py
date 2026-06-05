"""Layout-aware chunker with section-boundary splitting.

Handles both:
- ODL output: elements are already individual paragraphs (split by element)
- PyMuPDF output: entire page or section blob in one element (split by section pattern)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.models import Chunk, Element, Page

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CHARS = 800
_MIN_CHUNK_CHARS = 40
_MIN_BBOX_AREA = 50.0

_NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"EVB-IT\s+Dienst", re.IGNORECASE),
    re.compile(r"^Seite\s+\d+\s+von\s+\d+", re.IGNORECASE),
    re.compile(r"Version\s+\d+\.\d+\s+vom", re.IGNORECASE),
    re.compile(r"Die mit \* gekennzeichneten", re.IGNORECASE),
    re.compile(r"^\s*_+\s*$"),
    re.compile(r"\.{5,}"),
    re.compile(r"^\d{1,2}\s+von\s+\d{1,2}$"),
]

# Matches inline section markers like:
#   "15 Laufzeit und Kündigung", "11 Schlechtleistung", "§ 7 Mitwirkung"
# Used to split a blob like "... end of §10. 11 Schlechtleistung Wird ..."
_INLINE_SECTION_RE = re.compile(
    r"(?<=[.!?\s])(?:§\s*)?(?P<num>\d{1,2}(?:\.\d)?)\s+"
    r"(?P<title>[A-Z\u00c4\u00d6\u00dc][\w\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc\s\-,]{3,60}?)\s+(?=[A-Z\u00c4\u00d6\u00dc\w])"
)

# Standalone heading (entire element is just a heading)
_HEADING_RE = re.compile(
    r"^(?:§\s*)?\d{1,2}(?:\.\d)?\s+[A-Z\u00c4\u00d6\u00dc][\w\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc\s\-,]{3,60}$"
)


@dataclass
class ChunkerConfig:
    max_chars: int = _DEFAULT_MAX_CHARS
    min_chunk_chars: int = _MIN_CHUNK_CHARS
    deduplicate: bool = True


def _valid_bbox(bbox: list[float]) -> bool:
    if len(bbox) != 4:
        return False
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) >= _MIN_BBOX_AREA


def _is_standalone_heading(text: str) -> bool:
    return bool(_HEADING_RE.match(text.strip()))


def _split_on_inline_sections(text: str) -> list[tuple[str, bool]]:
    """Split a text blob at embedded section boundaries.

    Returns list of (fragment, is_heading) tuples.
    E.g. "...end of ten. 11 Schlechtleistung Wird eine..." becomes:
        ('...end of ten.', False)
        ('11 Schlechtleistung', True)
        ('Wird eine...', False)
    """
    parts: list[tuple[str, bool]] = []
    last = 0
    for m in _INLINE_SECTION_RE.finditer(text):
        pre = text[last:m.start()].strip()
        if pre:
            parts.append((pre, False))
        heading = f"{m.group('num')} {m.group('title').strip()}"
        parts.append((heading, True))
        last = m.end()
    tail = text[last:].strip()
    if tail:
        parts.append((tail, False))
    # If no splits found, return as-is
    return parts if len(parts) > 1 else [(text, False)]


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
            "Chunked %d pages \u2192 %d chunks (%d filtered)",
            len(pages), len(chunks), before - len(chunks),
        )
        return chunks

    def _chunk_page(self, page: Page) -> list[Chunk]:
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
            elif pending_heading:
                chunks.append(Chunk(
                    text=pending_heading,
                    page_number=page.page_number,
                    bboxes=[b for b in buf_bboxes if _valid_bbox(b)],
                    chunk_type="text",
                    confidence=0.90,
                    image_path=page.image_path,
                ))
            buf_texts.clear()
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

            # Standalone heading element
            if _is_standalone_heading(element.text):
                flush(next_heading=element.text)
                buf_bboxes.append(element.bbox)
                buf_conf.append(element.confidence)
                continue

            # Try to split large blobs on inline section markers
            sub_parts = _split_on_inline_sections(element.text)

            for part_text, is_heading in sub_parts:
                part_text = part_text.strip()
                if not part_text:
                    continue
                if is_heading:
                    flush(next_heading=part_text)
                    buf_bboxes.append(element.bbox)
                    buf_conf.append(element.confidence)
                    continue
                cur_len = sum(len(t) for t in buf_texts)
                if cur_len + len(part_text) > self._cfg.max_chars and buf_texts:
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


def _is_noise(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    return any(p.search(t) for p in _NOISE_PATTERNS)
