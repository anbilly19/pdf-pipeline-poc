"""Layout-aware chunker that aggregates elements into retrieval-ready Chunks.

Design rules (from CLAUDE.md):
- Never discard bboxes — every Chunk carries ALL contributing bboxes.
- Preserve table elements as single chunks (no splitting mid-table).
- Respect heading boundaries — a heading always starts a new chunk.
- Chunks do not cross page boundaries.
- Filter out noise: short fragments, repeated page headers/footers.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.models import Chunk, Element, Page

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CHARS = 1000
_MIN_CHUNK_CHARS = 50  # discard chunks shorter than this

# Patterns that match typical PDF header/footer noise
_NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^Seite\s+\d+\s+von\s+\d+", re.IGNORECASE),
    re.compile(r"^Version\s+\d+\.\d+\s+vom", re.IGNORECASE),
    re.compile(r"^Die mit \* gekennzeichneten", re.IGNORECASE),
    re.compile(r"^Vertragsnummer.{0,40}$"),
    re.compile(r"^\s*_+\s*$"),  # lines of underscores
]

_HEADING_HEURISTICS = (
    lambda e: len(e.text) < 120 and e.text.isupper(),
    lambda e: len(e.text) < 120 and e.text.endswith(":"),
)


@dataclass
class ChunkerConfig:
    """Configuration for the layout chunker.

    Args:
        max_chars: Maximum character count per chunk before forcing a split.
        overlap_chars: Character overlap between consecutive text chunks.
        min_chunk_chars: Minimum characters for a chunk to be kept.
        deduplicate: Drop exact-duplicate chunks across the document.
    """

    max_chars: int = _DEFAULT_MAX_CHARS
    overlap_chars: int = 100
    min_chunk_chars: int = _MIN_CHUNK_CHARS
    deduplicate: bool = True


class LayoutChunker:
    """Converts a list of Pages into retrieval-ready Chunks.

    Chunks are formed by accumulating elements within a page until:
    - a heading element is encountered (boundary)
    - a table element is encountered (emitted as its own chunk)
    - accumulated text exceeds max_chars (hard split)

    Short, noisy, and duplicate chunks are filtered out.

    Args:
        config: Chunker configuration.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self._cfg = config or ChunkerConfig()

    def chunk(self, pages: list[Page]) -> list[Chunk]:
        """Convert pages into chunks.

        Args:
            pages: Normalised pages from any BaseExtractor.

        Returns:
            List of Chunk objects ready for embedding.
        """
        chunks: list[Chunk] = []
        for page in pages:
            chunks.extend(self._chunk_page(page))

        before = len(chunks)
        chunks = self._filter(chunks)
        logger.info(
            "Chunked %d pages into %d chunks (%d filtered out)",
            len(pages), len(chunks), before - len(chunks),
        )
        return chunks

    def _chunk_page(self, page: Page) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffer_texts: list[str] = []
        buffer_bboxes: list[list[float]] = []
        buffer_confidence: list[float] = []

        def flush() -> None:
            if not buffer_texts:
                return
            text = " ".join(buffer_texts)
            chunks.append(
                Chunk(
                    text=text,
                    page_number=page.page_number,
                    bboxes=list(buffer_bboxes),
                    chunk_type="text",
                    confidence=min(buffer_confidence),
                    image_path=page.image_path,
                )
            )
            buffer_texts.clear()
            buffer_bboxes.clear()
            buffer_confidence.clear()

        for element in page.elements:
            if _is_noise(element.text):
                continue

            if element.type == "table":
                flush()
                chunks.append(
                    Chunk(
                        text=element.text,
                        page_number=page.page_number,
                        bboxes=[element.bbox],
                        chunk_type="table",
                        confidence=element.confidence,
                        image_path=page.image_path,
                    )
                )
                continue

            if self._is_heading(element) and buffer_texts:
                flush()

            current_len = sum(len(t) for t in buffer_texts)
            if current_len + len(element.text) > self._cfg.max_chars and buffer_texts:
                flush()

            buffer_texts.append(element.text)
            buffer_bboxes.append(element.bbox)
            buffer_confidence.append(element.confidence)

        flush()
        return chunks

    def _filter(self, chunks: list[Chunk]) -> list[Chunk]:
        """Remove short and duplicate chunks."""
        seen: set[str] = set()
        out: list[Chunk] = []
        for c in chunks:
            stripped = c.text.strip()
            if len(stripped) < self._cfg.min_chunk_chars:
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
    """Return True if the text matches known header/footer noise patterns."""
    t = text.strip()
    if not t:
        return True
    return any(p.search(t) for p in _NOISE_PATTERNS)
