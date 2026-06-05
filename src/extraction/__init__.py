"""PDF extraction engines."""
# Only expose the abstract base and error at package level.
# Concrete extractors (PyMuPDFExtractor, ExtractionRouter) import heavy
# optional deps (fitz, pymupdf4llm) — import them explicitly where needed
# to avoid ModuleNotFoundError during tests that mock those deps.
from src.extraction.base import BaseExtractor, ExtractionError

__all__ = ["BaseExtractor", "ExtractionError"]
