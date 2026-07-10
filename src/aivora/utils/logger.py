"""Centralised logging setup.

Every module should do::

    from aivora.utils.logger import get_logger
    log = get_logger(__name__)

The first call configures a rotating file handler under ``logs/``
plus a Rich-formatted console handler.  Subsequent calls are
cheap — they only return a named child logger.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

from .config import get_config

_CONFIGURED = False


def _configure_root() -> None:
    """Attach console + file handlers exactly once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    cfg = get_config()
    log_dir: Path = cfg.paths["logs_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Avoid duplicate handlers if user manually attached one.
    for h in list(root.handlers):
        root.removeHandler(h)

    # ---- console ----
    console = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(name)s — %(message)s"))
    root.addHandler(console)

    # ---- rotating file ----
    file_handler = RotatingFileHandler(
        log_dir / "aivora.log",
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.  Safe to call from anywhere."""
    _configure_root()
    return logging.getLogger(name)
