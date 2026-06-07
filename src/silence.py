"""Import this module FIRST — before transformers, sentence-transformers, or faiss.

Sets env vars and logging levels that suppress the hundreds of
'[transformers] Accessing `__path__`' lines emitted at import time.
Safe to import multiple times (guards are idempotent).
"""
from __future__ import annotations

import logging
import os
import warnings

# Must be set before transformers is imported for the first time
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class _PathFilter(logging.Filter):
    """Drop '[transformers] Accessing `__path__`' log records."""
    def filter(self, record: logging.LogRecord) -> bool:
        return "Accessing `__path__`" not in record.getMessage()


_NOISY_LOGGERS = ("transformers", "sentence_transformers", "huggingface_hub")

for _name in _NOISY_LOGGERS:
    logging.getLogger(_name).setLevel(logging.ERROR)

logging.getLogger().addFilter(_PathFilter())

warnings.filterwarnings("ignore", message=r"Accessing `__path__`")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"transformers.*")
