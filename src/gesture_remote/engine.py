"""Gesture engine (phase 1): segments -> fire / ignore / arm decisions. Pure, time injected.

Semantics, and why (see CLAUDE.md "Engine semantics"; do not "simplify"):
- Fire at a segment's onset only, if the label is mapped, the engine armed and not in cooldown.
  Otherwise the segment is ignored for its whole life: firing later, when the cooldown ends,
  would trigger the fist when it is laid on the desk right after another gesture.
- One segment per gesture (segments.py): a gesture returning within `release_s` continues its
  segment and does not fire again.
- The arm gesture bypasses the cooldown, otherwise it would be refused right after a repeated
  thumb up, exactly when the user wants to disarm. One toggle per hold; a toggle restarts the
  others' cooldown and stops any repetition in progress.
- `Ignored` is an event for the log and the overlay; it must not beep (a call while disarmed).
- A new engine (launch, config reload) starts in cooldown, the arm gesture included, so a
  gesture already held at that moment does nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from gesture_remote.config import ActionSpec, EngineSettings, KeysAction
from gesture_remote.observation import NONE_LABEL, HandObservation
from gesture_remote.segments import TIME_EPSILON, Segment, SegmentTracker

IgnoreReason = Literal["cooldown", "disarmed", "unmapped"]
Outcome = Literal["fired", "toggled", "cooldown", "disarmed", "unmapped"]


@dataclass(frozen=True, slots=True)
class Triggered:
    """Run `action` now."""

    label: str
    action: ActionSpec
    repeat: bool = False
    """True for the repetitions of a `repeat_while_held` binding (not its first fire)."""


@dataclass(frozen=True, slots=True)
class Ignored:
    """A segment was born but will never fire. Logged and shown, never sounded."""

    label: str
    reason: IgnoreReason


@dataclass(frozen=True, slots=True)
class ArmedChanged:
    armed: bool


EngineEvent = Triggered | Ignored | ArmedChanged


@dataclass(frozen=True, slots=True)
class SegmentStatus:
    segment: Segment
    outcome: Outcome
    """Decided at the onset, for the segment's whole life."""
    repeating: bool
    """A `repeat_while_held` segment that will fire again while visible."""


@dataclass(frozen=True, slots=True)
class EngineSnapshot:
    """State for the debug overlay, as of the last `update`."""

    now: float
    armed: bool
    cooldown_remaining_s: float
    segments: tuple[SegmentStatus, ...]


@dataclass(slots=True)
class _Fate:
    outcome: Outcome
    next_repeat_at: float | None = None


class GestureEngine:
    def __init__(
        self,
        settings: EngineSettings,
        bindings: Mapping[str, ActionSpec],
        *,
        now: float,
        armed: bool | None = None,
    ) -> None:
        """`now` is the (re)start time: the engine is in cooldown until now + cooldown_s.

        `armed` carries the state of the previous engine over a config reload; None (a fresh
        start) means `settings.start_armed`. Without it a user who disarmed before a call would be
        re-armed silently by any edit of config.yaml. It is ignored when the new settings have no
        arm gesture: nothing could re-arm the engine (the loader refuses start_armed: false then).
        """
        self._settings = settings
        self._bindings = dict(bindings)
        self._tracker = SegmentTracker(
            stable_frames=self._stable_frames, release_s=settings.release_s
        )
        self._fates: dict[str, _Fate] = {}
        if armed is None or settings.arm_gesture is None:
            armed = settings.start_armed
        self._armed = armed
        self._startup_until = now + settings.cooldown_s
        self._cooldown_until = self._startup_until
        self._now = now

    @property
    def armed(self) -> bool:
        return self._armed

    def update(self, hand: HandObservation | None, now: float) -> list[EngineEvent]:
        """Feed one processed frame (`hand` = its primary hand, or None) and get its events."""
        self._now = now
        vote = self._vote(hand)
        step = self._tracker.update(vote, now)
        for segment in step.ended:
            self._fates.pop(segment.label, None)
        if step.started is not None:
            return self._on_onset(step.started.label, now)
        if vote is not None:
            return self._maybe_repeat(vote, now)
        return []

    def snapshot(self) -> EngineSnapshot:
        return EngineSnapshot(
            now=self._now,
            armed=self._armed,
            cooldown_remaining_s=max(0.0, self._cooldown_until - self._now),
            segments=tuple(
                SegmentStatus(
                    segment=segment,
                    outcome=self._fates[segment.label].outcome,
                    repeating=self._fates[segment.label].next_repeat_at is not None,
                )
                for segment in self._tracker.segments
            ),
        )

    # --- decisions ------------------------------------------------------------------------

    def _on_onset(self, label: str, now: float) -> list[EngineEvent]:
        if label == self._settings.arm_gesture:
            if self._before(now, self._startup_until):
                return self._ignore(label, "cooldown")
            self._armed = not self._armed
            self._fates[label] = _Fate("toggled")
            self._cooldown_until = now + self._settings.cooldown_s
            for fate in self._fates.values():
                fate.next_repeat_at = None
            return [ArmedChanged(self._armed)]

        action = self._bindings.get(label)
        if action is None:
            return self._ignore(label, "unmapped")
        if not self._armed:
            return self._ignore(label, "disarmed")
        if self._before(now, self._cooldown_until):
            return self._ignore(label, "cooldown")

        fate = _Fate("fired")
        if isinstance(action, KeysAction) and action.repeat_while_held:
            fate.next_repeat_at = now + self._settings.repeat_delay_s
        self._fates[label] = fate
        self._cooldown_until = now + self._settings.cooldown_s
        return [Triggered(label, action)]

    def _maybe_repeat(self, label: str, now: float) -> list[EngineEvent]:
        """Repetition of a fired `repeat_while_held` segment, on frames where it is visible."""
        fate = self._fates.get(label)
        if fate is None or fate.next_repeat_at is None or not self._armed:
            return []
        if self._before(now, fate.next_repeat_at):
            return []
        fate.next_repeat_at = now + self._settings.repeat_interval_s
        self._cooldown_until = now + self._settings.cooldown_s
        return [Triggered(label, self._bindings[label], repeat=True)]

    def _ignore(self, label: str, reason: IgnoreReason) -> list[EngineEvent]:
        self._fates[label] = _Fate(reason)
        return [Ignored(label, reason)]

    # --- per-gesture settings -------------------------------------------------------------

    def _vote(self, hand: HandObservation | None) -> str | None:
        if hand is None or hand.label == NONE_LABEL:
            return None
        return hand.label if hand.score >= self._min_score(hand.label) else None

    def _min_score(self, label: str) -> float:
        override = self._settings.per_gesture.get(label)
        if override is not None and override.min_score is not None:
            return override.min_score
        return self._settings.min_score

    def _stable_frames(self, label: str) -> int:
        override = self._settings.per_gesture.get(label)
        if override is not None and override.stable_frames is not None:
            return override.stable_frames
        return self._settings.stable_frames

    @staticmethod
    def _before(now: float, deadline: float) -> bool:
        return now < deadline - TIME_EPSILON
