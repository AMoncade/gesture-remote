"""Shared test fixtures.

Round-1 contract: frozen during the parallel round (owner: admin). Lots keep their own helpers and
fakes inside their own test files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import gesture_remote

REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_sessionstart(session: pytest.Session) -> None:
    """Refuse to test another checkout's code.

    The shared venv holds an editable install pointing at the main checkout. In a worktree, the
    pytest `pythonpath = ["src"]` setting must make this checkout's src/ win; if it ever stops
    doing so, every test would silently exercise someone else's code and pass.
    """
    imported = Path(gesture_remote.__file__).resolve()
    expected = REPO_ROOT / "src" / "gesture_remote"
    if expected not in imported.parents:
        raise pytest.UsageError(
            f"gesture_remote was imported from {imported}, expected a file under {expected}"
        )


class FakeClock:
    """Manually advanced monotonic clock, in seconds."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()
