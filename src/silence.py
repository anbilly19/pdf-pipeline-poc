"""Import this module FIRST.

Configures all logging (console / file split) and suppresses
high-volume third-party noise before any other import runs.
"""
from __future__ import annotations

import os
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# configure logging immediately on import
from src.logging_config import configure_logging
configure_logging()

import logging
warnings.filterwarnings("ignore", message=r"Accessing `__path__`")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"transformers.*")
