"""Layout-aware chunker with section-boundary splitting and sliding window overlap.

Handles both:
- ODL output: elements are already individual paragraphs (split by element)
- PyMuPDF output: entire page or section blob in one element (split by section pattern)

Sliding window overlap (Roadmap #3)
------------------------------------
After all hard chunks are produced for a page, an overlap pass injects a
bridging chunk between each consecutive pair that straddles the boundary.
This prevents answers being cut in two when the relevant sentence falls
exactly at the end of one chunk and the start of the next.

Default overlap ratio: 12 % of max_chars (configurable via ChunkerConfig).
Only text chunks are overlapped; table chunks are never modified.

Bbox contract: the bridging chunk inherits the union of the tail chunk’s
and head chunk’s bounding boxes so citation coordinates remain valid.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.models import Chunk, Element, Page

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CHARS = 800
_MIN_CHUNK_CHARS = 40
_MIN_BBOX_AREA = 50.0
_DEFAULT_OVERLAP_RATIO = 0.12   # 12 % ≈ ~96 chars of an 800-char chunk
_MIN_OVERLAP_CHARS = 40          # never create an overlap shorter than this

_NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"EVB-IT\s+Dienst", re.IGNORECASE),
    re.compile(r"^Seite\s+\d+\s+von\s+\d+", re.IGNORECASE),
    re.compile(r"Version\s+\d+\.\d+\s+vom", re.IGNORECASE),
    re.compile(r"Die mit \* gekennzeichneten", re.IGNORECASE),
    re.compile(r"^\s*_+\s*$"),
    re.compile(r"\.{5,}"),
    re.compile(r"^\d{1,2}\s+von\s+\d{1,2}$"),
]

_INLINE_SECTION_RE = re.compile(
    r"(?<=[.!?\s])(?:§\s*)?(?P<num>\d{1,2}(?:\.\d)?)\s+"
    r"(?P<title>[A-Z\u00c4\u00d6\u00dc][\w\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc\s\-,]{3,60}?)\s+(?=[A-Z\u00c4\u00d6\u00dc\w])"
)

_HEADING_RE = re.compile(
    r"^(?:§\s*)?\d{1,2}(?:\.\d)?\s+[A-Z\u00c4\u00d6\u00dc][\w\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc\s\-,]{3,60}$"
)


@dataclass
class ChunkerConfig:
    """Configuration for LayoutChunker.

    Args:
        max_chars: Maximum character length of a single chunk.
        min_chunk_chars: Chunks shorter than this are discarded.
        deduplicate: Remove near-duplicate chunks.
        overlap_ratio: Fraction of max_chars used as overlap window (0 = disabled).
                       Recommended: 0.10–0.15 (10–15 %).
    """
    max_chars: int = _DEFAULT_MAX_CHARS
    min_chunk_chars: int = _MIN_CHUNK_CHARS
    deduplicate: bool = True
    overlap_ratio: float = _DEFAULT_OVERLAP_RATIO


def _valid_bbox(bbox: list[float]) -> bool:
    if len(bbox) != 4:
        return False
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) >= _MIN_BBOX_AREA


def _union_bboxes(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Return the union bbox list of two bbox collections.

    When both collections lie on the same page this produces the minimal
    enclosing rectangle.  When they span two pages (cross-page overlap)
    both bbox lists are concatenated so all source regions are preserved.
    """
    valid_a = [bb for bb in a if _valid_bbox(bb)]
    valid_b = [bb for bb in b if _valid_bbox(bb)]
    all_bboxes = valid_a + valid_b
    if not all_bboxes:
        return a or b
    x0 = min(bb[0] for bb in all_bboxes)
    y0 = min(bb[1] for bb in all_bboxes)
    x1 = max(bb[2] for bb in all_bboxes)
    y1 = max(bb[3] for bb in all_bboxes)
    return [[x0, y0, x1, y1]]


def _is_standalone_heading(text: str) -> bool:
    return bool(_HEADING_RE.match(text.strip()))


def _split_on_inline_sections(text: str) -> list[tuple[str, bool]]:
    """Split a text blob at embedded section boundaries.

    Returns list of (fragment, is_heading) tuples.
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
    return parts if len(parts) > 1 else [(text, False)]


class LayoutChunker:
    """Layout-aware chunker with optional sliding window overlap.

    Args:
        config: ChunkerConfig instance.  Defaults to 800-char chunks, 12% overlap.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self._cfg = config or ChunkerConfig()

    def chunk(self, pages: list[Page]) -> list[Chunk]:
        """Chunk all pages and apply overlap bridging.

        Args:
            pages: List of normalised Page objects.

        Returns:
            Interleaved list of primary + overlap-bridge chunks, filtered.
        """
        raw_chunks: list[Chunk] = []
        for page in pages:
            raw_chunks.extend(self._chunk_page(page))

        with_overlap = self._add_overlap(raw_chunks)
        filtered = self._filter(with_overlap)

        logger.info(
            "Chunked %d pages → %d raw → %d with overlap → %d after filter",
            len(pages), len(raw_chunks), len(with_overlap), len(filtered),
        )
        return filtered

    # ------------------------------------------------------------------
    # Overlap pass (Roadmap #3)
    # ------------------------------------------------------------------

    def _add_overlap(self, chunks: list[Chunk]) -> list[Chunk]:
        """Insert bridging chunks between consecutive text chunk pairs.

        A bridging chunk contains the tail of chunk[i] + the head of
        chunk[i+1].  Its length is capped at overlap_chars on each side.
        Table chunks are never used as overlap sources or targets.

        The bridging chunk carries:
        - ``chunk_type = "overlap"`` so downstream can distinguish it.
        - ``bboxes`` = union of the two source chunks’ bboxes.
        - ``confidence`` = min of the two source chunks’ confidences.
        - ``page_number`` from the tail chunk (the earlier one).

        Args:
            chunks: Raw chunks from the page-chunking pass.

        Returns:
            New list with bridge chunks interleaved after each eligible pair.
        """
        if self._cfg.overlap_ratio <= 0:
            return chunks

        overlap_chars = max(
            _MIN_OVERLAP_CHARS,
            int(self._cfg.max_chars * self._cfg.overlap_ratio),
        )

        result: list[Chunk] = []
        for i, chunk in enumerate(chunks):
            result.append(chunk)
            if i >= len(chunks) - 1:
                continue

            tail_chunk = chunk
            head_chunk = chunks[i + 1]

            # Only bridge text→text boundaries; skip if either side is a table
            if tail_chunk.chunk_type == "table" or head_chunk.chunk_type == "table":
                continue

            tail_text = tail_chunk.text[-overlap_chars:].strip()
            head_text = head_chunk.text[:overlap_chars].strip()

            if not tail_text or not head_text:
                continue

            bridge_text = f"{tail_text} {head_text}"

            # Skip if bridge is too short to be useful
            if len(bridge_text) < self._cfg.min_chunk_chars:
                continue

            bridge = Chunk(
                text=bridge_text,
                page_number=tail_chunk.page_number,
                bboxes=_union_bboxes(tail_chunk.bboxes, head_chunk.bboxes),
                chunk_type="overlap",
                confidence=min(tail_chunk.confidence, head_chunk.confidence),
                image_path=tail_chunk.image_path,
            )
            result.append(bridge)

        return result

    # ------------------------------------------------------------------
    # Page-level chunking (unchanged logic)
    # ------------------------------------------------------------------

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

            if _is_standalone_heading(element.text):
                flush(next_heading=element.text)
                buf_bboxes.append(element.bbox)
                buf_conf.append(element.confidence)
                continue

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

    # ------------------------------------------------------------------
    # Filter (unchanged logic)
    # ------------------------------------------------------------------

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
