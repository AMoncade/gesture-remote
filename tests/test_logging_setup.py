"""logging_setup: file + console, UTF-8, rotation, countable event lines, no arrays in logs."""

from __future__ import annotations

import io
import logging
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from gesture_remote import logging_setup
from gesture_remote.logging_setup import (
    LOG_FILE_NAME,
    log_armed,
    log_ignored,
    log_trigger,
    setup_logging,
)

# What step 6 will run on the log file to count fires per gesture.
TRIGGER_LINE = re.compile(r" gesture_remote\.events: TRIGGER (\S+) (\S+)$")


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()  # release the file, or Windows cannot delete tmp_path
    root.setLevel(level)


def read_lines(log_file: Path) -> list[str]:
    for handler in logging.getLogger().handlers:
        handler.flush()
    return log_file.read_text(encoding="utf-8").splitlines()


def test_writes_utf8_file_in_log_dir(tmp_path: Path) -> None:
    log_file = setup_logging(tmp_path / "logs", console=False)
    assert log_file == tmp_path / "logs" / LOG_FILE_NAME
    logging.getLogger("gesture_remote.test").info("app %s", "Intel® Graphics – é")
    (line,) = read_lines(log_file)
    assert line.endswith("INFO    gesture_remote.test: app Intel® Graphics – é")


def test_event_lines_are_countable(tmp_path: Path) -> None:
    log_file = setup_logging(tmp_path, console=False)
    log_trigger("thumb_up", "keys")
    log_trigger("thumb_up", "keys", repeat=True)
    log_trigger("thumb_up", "keys", repeat=True)
    log_trigger("victory", "url")
    log_ignored("closed_fist", "cooldown")
    log_armed(False)
    log_armed(True)
    lines = read_lines(log_file)
    fires = [match.groups() for line in lines if (match := TRIGGER_LINE.search(line))]
    assert fires == [("thumb_up", "keys"), ("victory", "url")]  # repeats are not onsets
    messages = [line.split(": ", 1)[1] for line in lines]
    assert messages == [
        "TRIGGER thumb_up keys",
        "TRIGGER thumb_up keys repeat",
        "TRIGGER thumb_up keys repeat",
        "TRIGGER victory url",
        "IGNORED closed_fist cooldown",
        "DISARMED",
        "ARMED",
    ]


def test_setup_twice_does_not_duplicate_lines(tmp_path: Path) -> None:
    setup_logging(tmp_path, console=False)
    log_file = setup_logging(tmp_path, console=False)
    log_armed(True)
    assert len(read_lines(log_file)) == 1


def test_arrays_never_reach_the_log(tmp_path: Path) -> None:
    log_file = setup_logging(tmp_path, console=False)
    landmarks = np.full((21, 3), 0.123456, dtype=np.float32)
    logger = logging.getLogger("gesture_remote.test")
    logger.info("hand %s", landmarks)
    logger.info("frame %(frame)s", {"frame": np.zeros((4, 4, 3), np.uint8)})
    logger.info(landmarks)
    text = "\n".join(read_lines(log_file))
    assert "0.123" not in text and "[[" not in text
    assert text.count("elided>") == 3
    assert "<array (21, 3) elided>" in text and "<array (4, 4, 3) elided>" in text


def test_verbose_enables_debug(tmp_path: Path) -> None:
    log_file = setup_logging(tmp_path, console=False)
    logging.getLogger("gesture_remote.test").debug("hidden")
    setup_logging(tmp_path, verbose=True, console=False)
    logging.getLogger("gesture_remote.test").debug("shown")
    lines = read_lines(log_file)
    assert len(lines) == 1 and lines[0].endswith("shown")


def test_file_rotates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(logging_setup, "MAX_BYTES", 500)
    log_file = setup_logging(tmp_path, console=False)
    for index in range(40):
        log_ignored(f"label_{index}", "cooldown")
    read_lines(log_file)
    assert (tmp_path / f"{LOG_FILE_NAME}.1").exists()
    assert log_file.stat().st_size <= 500


def test_console_in_cp1252_escapes_what_it_cannot_encode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = io.BytesIO()
    console = io.TextIOWrapper(raw, encoding="cp1252")  # this machine's console code page
    monkeypatch.setattr(sys, "stderr", console)
    setup_logging(tmp_path, console=True)
    logging.getLogger("gesture_remote.test").warning("app %s", "Intel® 🤟")
    console.flush()
    out = raw.getvalue().decode("cp1252")
    assert out.rstrip().endswith("app Intel® \\U0001f91f")
    assert "Logging error" not in out
