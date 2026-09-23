"""Logging: console + rotating UTF-8 file, and the countable event lines.

Step 6 counts gestures on the log file, so every engine event is one line on the `events` logger
whose message starts with a fixed word and has space-separated fields:

    TRIGGER <label> <action type>          a segment's first fire: count these for "fires"
    TRIGGER <label> <action type> repeat   a wanted repetition of a repeat_while_held binding
    IGNORED <label> <reason>               a segment that will never fire (cooldown, disarmed, ...)
    ARMED / DISARMED

Every fire is logged, repetitions included, so a double fire (two unmarked TRIGGER lines for one
gesture) stays distinguishable from a held 👍 (admin decision, 2026-09-23).

Never log images or landmarks: a filter replaces any numpy array passed as a log argument.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np

LOG_FILE_NAME = "gesture-remote.log"
MAX_BYTES = 2_000_000
BACKUP_COUNT = 5
FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

EVENTS_LOGGER = "gesture_remote.events"
events = logging.getLogger(EVENTS_LOGGER)

_HANDLER_TAG = "_gesture_remote_handler"


class ArrayRedactor(logging.Filter):
    """Replace numpy arrays in log arguments by their shape: pixels and landmarks never leak."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(_redact(arg) for arg in record.args)
        elif isinstance(record.args, dict):
            record.args = {key: _redact(value) for key, value in record.args.items()}
        record.msg = _redact(record.msg)
        return True


def _redact(value: object) -> object:
    if isinstance(value, np.ndarray):
        return f"<array {value.shape} elided>"
    return value


def setup_logging(log_dir: Path, *, verbose: bool = False, console: bool = True) -> Path:
    """Configure the root logger; return the log file path. Safe to call again (no duplicates)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_FILE_NAME
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, _HANDLER_TAG, False)]:
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(FORMAT)
    handlers: list[logging.Handler] = [
        RotatingFileHandler(
            log_file, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
        )
    ]
    if console:
        # The console code page (cp1252 here) cannot encode every Start-menu name: never crash.
        reconfigure = getattr(sys.stderr, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")
        handlers.append(logging.StreamHandler(sys.stderr))
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(ArrayRedactor())
        setattr(handler, _HANDLER_TAG, True)
        root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    return log_file


def log_trigger(label: str, action_type: str, *, repeat: bool = False) -> None:
    if repeat:
        events.info("TRIGGER %s %s repeat", label, action_type)
    else:
        events.info("TRIGGER %s %s", label, action_type)


def log_ignored(label: str, reason: str) -> None:
    events.info("IGNORED %s %s", label, reason)


def log_armed(armed: bool) -> None:
    events.info("ARMED" if armed else "DISARMED")
