"""
app/logger.py — Structured logging for production.
Writes to both console and rotating file logs.

Disk usage is capped automatically: when cctv.log reaches LOG_MAX_BYTES it rotates
and the oldest backup is deleted (keeps LOG_BACKUP_COUNT files max).
"""
import logging
import logging.handlers
import os
import sys

from app.config import LOG_DIR, LOG_LEVEL, LOG_MAX_BYTES, LOG_BACKUP_COUNT

LOG_FILE = os.path.join(LOG_DIR, "cctv.log")

os.makedirs(LOG_DIR, exist_ok=True)

_FMT = "%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _build_handler_console() -> logging.StreamHandler:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(_FMT, _DATE_FMT))
    return h


def _build_handler_file() -> logging.handlers.RotatingFileHandler:
    h = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    h.setFormatter(logging.Formatter(_FMT, _DATE_FMT))
    return h


def get_logger(name: str) -> logging.Logger:
    """Return a named logger wired to console + rotating file."""
    logger = logging.getLogger(name)
    if logger.handlers:          # already configured in this process
        return logger
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.addHandler(_build_handler_console())
    logger.addHandler(_build_handler_file())
    logger.propagate = False
    return logger
