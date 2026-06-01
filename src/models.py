"""Core data models for the PDF pipeline.

Immutable contracts — never remove or rename fields.
All pipeline stages must pass bboxes through unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Element:
    """A single extracted element from a PDF page.

    Args:
        type: Content type of the element.
        text: Plain or markdown text content.
        bbox: Bounding box [x0, y0, x1, y1] in PDF points.
        confidence: Parser confidence score between 0.0 and 1.0.
    """

    type: Literal["text", "table", "image"]
    text: str
    bbox: list[float]  # [x0, y0, x1, y1] in PDF points
    confidence: float  # 0.0 – 1.0

    def __post_init__(self) -> None:
        if len(self.bbox) != 4:
            raise ValueError(f"bbox must have exactly 4 values, got {len(self.bbox)}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class Page:
    """Normalised representation of one PDF page.

    Args:
        page_number: 1-based page index.
        image_path: Path to the rendered PNG of this page (empty string if not rendered).
        elements: Extracted elements in document order.
    """

    page_number: int
    image_path: str
    elements: list[Element] = field(default_factory=list)


@dataclass
class Chunk:
    """Retrieval unit carrying aggregated bounding boxes from one or more elements.

    Args:
        text: Combined text content.
        page_number: Source page (1-based).
        bboxes: All bounding boxes contributing to this chunk.
        chunk_type: Dominant content type.
        confidence: Minimum confidence across contributing elements.
        image_path: Path to source page PNG.
    """

    text: str
    page_number: int
    bboxes: list[list[float]]
    chunk_type: Literal["text", "table", "figure"]
    confidence: float
    image_path: str


@dataclass
class Source:
    """A single cited source with precise page location.

    Args:
        text: Excerpt used as evidence.
        page: 1-based page number.
        bboxes: Bounding boxes on that page.
        image: Path to page PNG for overlay rendering.
    """

    text: str
    page: int
    bboxes: list[list[float]]
    image: str


@dataclass
class QAResponse:
    """Final output of the Q&A pipeline.

    Args:
        answer: Natural language answer.
        sources: Cited sources with page locations.
    """

    answer: str
    sources: list[Source]
