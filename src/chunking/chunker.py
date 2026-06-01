"""Layout-aware chunker that aggregates elements into retrieval-ready Chunks.

Design rules (from CLAUDE.md):
- Never discard bboxes — every Chunk carries ALL contributing bboxes.
- Preserve table elements as single chunks (no splitting mid-table).
- Respect heading boundaries — a heading always starts a new chunk.
- Chunks do not cross page boundaries.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.models import Chunk, Element, Page

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CHARS = 1000
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
    """

    max_chars: int = _DEFAULT_MAX_CHARS
    overlap_chars: int = 100


class LayoutChunker:
    """Converts a list of Pages into retrieval-ready Chunks.

    Chunks are formed by accumulating elements within a page until:
    - a heading element is encountered (boundary)
    - a table element is encountered (emitted as its own chunk)
    - accumulated text exceeds max_chars (hard split)

    Bounding boxes from all contributing elements are aggregated onto each chunk.

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
        logger.info("Chunked %d pages into %d chunks", len(pages), len(chunks))
        return chunks

    def _chunk_page(self, page: Page) -> list[Chunk]:
        """Chunk a single page's elements.

        Args:
            page: The page to chunk.

        Returns:
            List of Chunk objects for this page.
        """
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

    @staticmethod
    def _is_heading(element: Element) -> bool:
        """Heuristically detect heading elements.

        Args:
            element: The element to test.

        Returns:
            True if the element is likely a heading.
        """
        return any(h(element) for h in _HEADING_HEURISTICS)
