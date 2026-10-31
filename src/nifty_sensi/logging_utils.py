"""Consistent logging and lightweight stage timers."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

LOGGER_NAME = "nifty_sensi"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure(verbose: bool = False, log_file: str | Path | None = None) -> logging.Logger:
    """Configure the package logger once and return it."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    logger.addHandler(stream)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


@contextmanager
def timer(label: str, logger: logging.Logger | None = None):
    logger = logger or get_logger("timer")
    started = time.perf_counter()
    logger.info("%s: start", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        logger.info("%s: done in %.2fs", label, elapsed)
