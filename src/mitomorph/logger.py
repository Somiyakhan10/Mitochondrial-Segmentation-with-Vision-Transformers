"""Shared logging configuration for the mitomorph pipeline (NFR-12)."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_CONFIGURED = False


def configure_logging(log_dir: str | Path | None = None, level: int = logging.INFO) -> None:
    """Configure the root mitomorph logger with a console handler and, if
    ``log_dir`` is given, a rotating file handler. Safe to call multiple
    times; only the first call takes effect.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger("mitomorph")
    root.setLevel(level)

    formatter = logging.Formatter(_DEFAULT_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "mitomorph.log", maxBytes=5_000_000, backupCount=3
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``mitomorph``, configuring defaults on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(f"mitomorph.{name}")
