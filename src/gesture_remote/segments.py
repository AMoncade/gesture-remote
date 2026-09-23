"""Votes -> gesture segments. Pure: no clock, no I/O, time is passed in.

Each processed frame votes for at most one label (the engine decides what votes). After
`stable_frames(label)` identical consecutive votes a *segment* starts for that label, unless one
is already alive. A segment stays alive while its label was voted within the last `release_s`
seconds, so a gesture that comes back in time continues its segment, even after another gesture
in between; meanwhile that other gesture can start its own segment.

Phase 3 (short on release, hold, combos) builds on the segments exposed here without changing
this module: keep it free of actions, cooldowns and arming.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

TIME_EPSILON = 1e-6
"""Slack for time comparisons: timestamps are floats, and a threshold meant to fall on a frame
(e.g. 0.8 s at 30 fps) must not be missed by a rounding error."""


@dataclass(frozen=True, slots=True)
class Segment:
    """One continuous presence of a gesture, from its onset until it is released."""

    label: str
    started_at: float
    """Time of the first vote of the stable run that created the segment."""
    onset: float
    """Time of the vote that completed `stable_frames`: the segment's birth."""
    last_seen: float
    """Time of the latest vote for `label`."""


@dataclass(frozen=True, slots=True)
class SegmentStep:
    """What one vote changed."""

    started: Segment | None
    """The segment born on this vote, if any (at most one: only the voted label can start)."""
    ended: tuple[Segment, ...]
    """Segments released on this vote: their label was not voted for `release_s`."""


class SegmentTracker:
    def __init__(self, *, stable_frames: Callable[[str], int], release_s: float) -> None:
        self._stable_frames = stable_frames
        self._release_s = release_s
        self._live: dict[str, Segment] = {}
        self._run_label: str | None = None
        self._run_count = 0
        self._run_started = 0.0

    @property
    def segments(self) -> tuple[Segment, ...]:
        """Live segments, oldest onset first."""
        return tuple(sorted(self._live.values(), key=lambda segment: segment.onset))

    def get(self, label: str) -> Segment | None:
        """The live segment of `label`, or None."""
        return self._live.get(label)

    def update(self, vote: str | None, now: float) -> SegmentStep:
        """Feed the vote of one frame (None = the frame votes for nothing)."""
        ended = tuple(
            segment
            for segment in self._live.values()
            if now - segment.last_seen >= self._release_s - TIME_EPSILON
        )
        for segment in ended:
            del self._live[segment.label]

        if vote is None:
            self._run_label = None
            self._run_count = 0
            return SegmentStep(started=None, ended=ended)

        if vote == self._run_label:
            self._run_count += 1
        else:
            self._run_label = vote
            self._run_count = 1
            self._run_started = now

        live = self._live.get(vote)
        if live is not None:
            self._live[vote] = replace(live, last_seen=now)
            return SegmentStep(started=None, ended=ended)
        if self._run_count < self._stable_frames(vote):
            return SegmentStep(started=None, ended=ended)
        segment = Segment(label=vote, started_at=self._run_started, onset=now, last_seen=now)
        self._live[vote] = segment
        return SegmentStep(started=segment, ended=ended)
