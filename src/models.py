"""Core data models for the PDF pipeline.

These are immutable contracts – never remove or rename fields.
All pipeline stages must pass bboxes through unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Element:
    """A single extracted element from a PDF page."""

    type: Literal["text", "table", "image"]
    text: str
    bbox: list[float]  # [x0, y0, x1, y1] in PDF points
    confidence: float  # 0.0 – 1.0


@dataclass
class Page:
    """Normalised representation of one PDF page."""

    page_number: int
    image_path: str  # path to rendered PNG
    elements: list[Element] = field(default_factory=list)


@dataclass
class Chunk:
    """Retrieval unit carrying aggregated bounding boxes."""

    text: str
    page_number: int
    bboxes: list[list[float]]
    chunk_type: Literal["text", "table", "figure"]
    confidence: float
    image_path: str


@dataclass
class Source:
    """A single cited source with page location."""

    text: str
    page: int
    bboxes: list[list[float]]
    image: str  # path to page image


@dataclass
class QAResponse:
    """Final output of the Q&A pipeline."""

    answer: str
    sources: list[Source]
