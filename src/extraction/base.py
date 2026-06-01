"""Abstract base class for PDF extraction engines."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from src.models import Page

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Raised when PDF extraction fails unrecoverably."""


class BaseExtractor(ABC):
    """Parser-agnostic interface for PDF extraction.

    All implementations must preserve bounding boxes on every element.
    """

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

    def extract_safe(self, pdf_path: Path) -> list[Page] | None:
        """Extract with error suppression; returns None on failure.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of Pages or None if extraction failed.
        """
        try:
            return self.extract(pdf_path)
        except ExtractionError as exc:
            logger.warning("Extraction failed for %s: %s", pdf_path, exc)
            return None
