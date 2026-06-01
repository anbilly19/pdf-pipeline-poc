"""Abstract base class for PDF extraction engines."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from src.models import Page

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """Parser-agnostic interface for PDF extraction."""

    @abstractmethod
    def extract(self, pdf_path: Path) -> list[Page]:
        """Extract all pages from a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of normalised Page objects with bounding boxes.

        Raises:
            ExtractionError: If the PDF cannot be parsed.
        """
        ...


class ExtractionError(Exception):
    """Raised when PDF extraction fails unrecoverably."""
