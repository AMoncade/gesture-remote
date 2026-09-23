"""Privacy guard: webcam frames must never be written to disk."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(r"\b(imwrite|VideoWriter|imencode)\b")
SCANNED_DIRS = ("src", "tools", "macros")


def find_frame_writes(text: str) -> list[str]:
    return FORBIDDEN.findall(text)


def test_detector_flags_frame_writes_and_ignores_reads() -> None:
    assert find_frame_writes("ok = cv2.imwrite('x.png', frame)") == ["imwrite"]
    assert find_frame_writes("writer = cv2.VideoWriter(path, fourcc, 30, size)") == ["VideoWriter"]
    assert find_frame_writes("image = cv2.imread('x.png')") == []


def test_no_frame_writes_in_shipped_code() -> None:
    scanned = [
        path for folder in SCANNED_DIRS for path in sorted((REPO_ROOT / folder).rglob("*.py"))
    ]
    offenders = [
        f"{path.relative_to(REPO_ROOT)}: {hits}"
        for path in scanned
        if (hits := find_frame_writes(path.read_text(encoding="utf-8")))
    ]
    assert scanned, "the guard scanned nothing: check SCANNED_DIRS"
    assert offenders == []
