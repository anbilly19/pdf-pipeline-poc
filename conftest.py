"""Root pytest configuration — suppress noisy third-party log output."""
from __future__ import annotations

import logging
import warnings


def pytest_configure(config: object) -> None:  # noqa: ANN001
    """Silence transformers __path__ alias warnings before any test is collected."""
    # transformers emits these via its own logger, not warnings.warn
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Belt-and-suspenders: also suppress via warnings module in case
    # future transformers versions switch to warnings.warn
    warnings.filterwarnings(
        "ignore",
        message=r"Accessing `__path__`",
        module=r"transformers.*",
    )
