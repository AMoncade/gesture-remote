"""SegmentTracker: votes -> segments, scripted at 30 fps."""

from __future__ import annotations

import pytest

from conftest import FakeClock
from gesture_remote.segments import Segment, SegmentStep, SegmentTracker

DT = 1 / 30
RELEASE_S = 0.8


def make_tracker(stable: dict[str, int] | None = None) -> SegmentTracker:
    overrides = stable or {}
    return SegmentTracker(stable_frames=lambda label: overrides.get(label, 10), release_s=RELEASE_S)


def feed(
    tracker: SegmentTracker, clock: FakeClock, vote: str | None, frames: int
) -> list[SegmentStep]:
    return [tracker.update(vote, clock.advance(DT)) for _ in range(frames)]


def started(steps: list[SegmentStep]) -> list[str]:
    return [step.started.label for step in steps if step.started is not None]


def ended(steps: list[SegmentStep]) -> list[str]:
    return [segment.label for step in steps for segment in step.ended]


def test_segment_starts_on_the_10th_vote_not_the_9th(clock: FakeClock) -> None:
    tracker = make_tracker()
    assert started(feed(tracker, clock, "open_palm", 9)) == []
    assert tracker.segments == ()
    assert started(feed(tracker, clock, "open_palm", 1)) == ["open_palm"]


def test_segment_records_run_start_onset_and_last_seen(clock: FakeClock) -> None:
    tracker = make_tracker()
    first = clock.now + DT
    feed(tracker, clock, "open_palm", 10)
    onset = clock.now
    feed(tracker, clock, "open_palm", 5)
    assert tracker.get("open_palm") == Segment(
        label="open_palm", started_at=first, onset=onset, last_seen=clock.now
    )


def test_a_frame_without_a_vote_resets_the_count(clock: FakeClock) -> None:
    tracker = make_tracker()
    steps = feed(tracker, clock, "open_palm", 9)
    steps += feed(tracker, clock, None, 1)
    steps += feed(tracker, clock, "open_palm", 9)
    assert started(steps) == []
    assert started(feed(tracker, clock, "open_palm", 1)) == ["open_palm"]


def test_alternating_labels_never_start_a_segment(clock: FakeClock) -> None:
    tracker = make_tracker()
    steps = []
    for _ in range(150):
        steps += feed(tracker, clock, "open_palm", 1)
        steps += feed(tracker, clock, "closed_fist", 1)
    assert started(steps) == []


def test_holding_5_s_is_one_segment(clock: FakeClock) -> None:
    tracker = make_tracker()
    steps = feed(tracker, clock, "open_palm", 150)
    assert started(steps) == ["open_palm"]
    assert ended(steps) == []


def test_a_gap_shorter_than_release_continues_the_segment(clock: FakeClock) -> None:
    tracker = make_tracker()
    steps = feed(tracker, clock, "open_palm", 30)
    steps += feed(tracker, clock, None, 15)  # 0.5 s
    steps += feed(tracker, clock, "open_palm", 30)
    assert started(steps) == ["open_palm"]
    assert ended(steps) == []


def test_another_gesture_in_between_does_not_end_the_segment(clock: FakeClock) -> None:
    tracker = make_tracker()
    steps = feed(tracker, clock, "open_palm", 30)
    steps += feed(tracker, clock, "pointing_up", 12)  # 0.4 s: long enough for its own segment
    steps += feed(tracker, clock, "open_palm", 30)
    assert started(steps) == ["open_palm", "pointing_up"]
    # The palm continues; the pointing finger, no longer seen, is released after release_s.
    assert ended(steps) == ["pointing_up"]
    assert [segment.label for segment in tracker.segments] == ["open_palm"]


def test_absence_of_release_s_ends_the_segment_and_a_new_run_is_needed(
    clock: FakeClock,
) -> None:
    tracker = make_tracker()
    feed(tracker, clock, "open_palm", 30)
    steps = feed(tracker, clock, None, 24)  # 0.8 s
    assert ended(steps) == ["open_palm"]
    assert tracker.segments == ()
    assert started(feed(tracker, clock, "open_palm", 9)) == []
    assert started(feed(tracker, clock, "open_palm", 1)) == ["open_palm"]


def test_absence_just_under_release_s_keeps_the_segment(clock: FakeClock) -> None:
    tracker = make_tracker()
    feed(tracker, clock, "open_palm", 30)
    assert ended(feed(tracker, clock, None, 23)) == []
    assert tracker.get("open_palm") is not None


@pytest.mark.parametrize(("frames", "expected"), [(14, []), (15, ["i_love_you"])])
def test_stable_frames_is_per_label(clock: FakeClock, frames: int, expected: list[str]) -> None:
    tracker = make_tracker({"i_love_you": 15})
    assert started(feed(tracker, clock, "i_love_you", frames)) == expected
