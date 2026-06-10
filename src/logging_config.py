"""Centralised logging configuration.

Call configure_logging() once at startup (app.py imports silence.py which
calls it automatically).

Outputs
-------
logs/debug.log   All DEBUG+ records from src.* — debug checkpoint lines live here.
logs/app.log     INFO / WARNING / ERROR from every logger — rotating 5 MB x 3 files.
Console          WARNING+ only, minimal single-line format — stays clean.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

_configured = False

_CONSOLE_FMT  = "%(levelname)s %(name)s: %(message)s"
_FILE_FMT     = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATE_FMT     = "%H:%M:%S"

_LOG_DIR      = Path("logs")
_DEBUG_LOG    = _LOG_DIR / "debug.log"
_APP_LOG      = _LOG_DIR / "app.log"
_MAX_BYTES    = 5 * 1024 * 1024   # 5 MB
_BACKUP_COUNT = 3

_DBG = os.environ.get("DEBUG_PIPELINE", "0") == "1"


class _SrcOnlyFilter(logging.Filter):
    """Allow only records from src.* loggers."""
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith("src.")


class _NoisyFilter(logging.Filter):
    """Drop known high-volume noise lines."""
    _PATTERNS = (
        "Accessing `__path__`",
        "huggingface/tokenizers",
    )
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in self._PATTERNS)


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    _LOG_DIR.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)          # capture everything; handlers filter

    # ------------------------------------------------------------------ #
    # 1. Console — WARNING+ only, clean format                           #
    # ------------------------------------------------------------------ #
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter(_CONSOLE_FMT))
    console.addFilter(_NoisyFilter())
    root.addHandler(console)

    # ------------------------------------------------------------------ #
    # 2. logs/app.log — INFO+ rotating, all loggers                      #
    # ------------------------------------------------------------------ #
    app_handler = logging.handlers.RotatingFileHandler(
        _APP_LOG, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
    app_handler.addFilter(_NoisyFilter())
    root.addHandler(app_handler)

    # ------------------------------------------------------------------ #
    # 3. logs/debug.log — DEBUG+ from src.* only (checkpoint lines)      #
    # ------------------------------------------------------------------ #
    debug_handler = logging.handlers.RotatingFileHandler(
        _DEBUG_LOG, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
    debug_handler.addFilter(_SrcOnlyFilter())
    debug_handler.addFilter(_NoisyFilter())
    root.addHandler(debug_handler)

    # silence third-party noise
    for name in ("transformers", "sentence_transformers", "huggingface_hub",
                 "httpx", "httpcore", "urllib3", "filelock"):
        logging.getLogger(name).setLevel(logging.ERROR)
