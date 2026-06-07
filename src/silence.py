"""Import this module FIRST — before transformers, sentence-transformers, or faiss.

Sets env vars and logging levels that suppress the hundreds of
'[transformers] Accessing `__path__`' lines emitted at import time.
Safe to import multiple times (guards are idempotent).

Ollama CPU note
---------------
Setting OLLAMA_LLM_LIBRARY=cpu_avx2 here (before any Ollama imports) tells
the Ollama server to use the CPU AVX2 backend for ALL models.  This is the
most reliable way to prevent GPU offloading on machines where models like
phi4-mini-reasoning cannot be fully loaded into VRAM.  To override and use
the GPU, set OLLAMA_LLM_LIBRARY='' or remove the variable from your .env
before starting the app.
"""
from __future__ import annotations

import logging
import os
import warnings

# --------------------------------------------------------------------------
# Ollama: force CPU backend before any Ollama/LangChain imports happen.
# cpu_avx2 is the correct library string on modern x86 CPUs (2013+).
# Fall back to plain 'cpu' if the user explicitly opts out via OLLAMA_NUM_GPU.
# --------------------------------------------------------------------------
if os.environ.get("OLLAMA_NUM_GPU", "0") == "0":
    os.environ.setdefault("OLLAMA_LLM_LIBRARY", "cpu_avx2")

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
