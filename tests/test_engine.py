"""GestureEngine, scripted at 30 fps with the default settings of the plan."""

from __future__ import annotations

import numpy as np

from conftest import FakeClock
from gesture_remote.config import (
    ActionSpec,
    EngineSettings,
    GestureOverride,
    KeysAction,
    UrlAction,
)
from gesture_remote.engine import (
    ArmedChanged,
    EngineEvent,
    GestureEngine,
    Ignored,
    Triggered,
)
from gesture_remote.observation import Handedness, HandObservation

FPS = 30
DT = 1 / FPS

PLAY = KeysAction(type="keys", keys=["playpause"])
VOLUME_UP = KeysAction(type="keys", keys=["volumeup"], repeat_while_held=True)
MUTE = KeysAction(type="keys", keys=["volumemute"])
STUDIUM = UrlAction(type="url", url="https://studium.umontreal.ca")

BINDINGS: dict[str, ActionSpec] = {
    "open_palm": PLAY,
    "thumb_up": VOLUME_UP,
    "closed_fist": MUTE,
    "pointing_up": STUDIUM,
}
PLAN_SETTINGS = EngineSettings(per_gesture={"i_love_you": GestureOverride(stable_frames=15)})


def hand(label: str, score: float = 0.9) -> HandObservation:
    return HandObservation(
        label=label,
        score=score,
        handedness=Handedness.RIGHT,
        handedness_score=0.99,
        landmarks=np.zeros((21, 3), dtype=np.float32),
        world_landmarks=np.zeros((21, 3), dtype=np.float32),
    )


class Script:
    """Feeds frames to an engine at 30 fps on the shared FakeClock."""

    def __init__(self, engine: GestureEngine, clock: FakeClock) -> None:
        self.engine = engine
        self.clock = clock

    def frames(self, label: str | None, n: int, score: float = 0.9) -> list[EngineEvent]:
        events: list[EngineEvent] = []
        for _ in range(n):
            observation = None if label is None else hand(label, score)
            events += self.engine.update(observation, self.clock.advance(DT))
        return events

    def seconds(self, label: str | None, seconds: float, score: float = 0.9) -> list[EngineEvent]:
        return self.frames(label, round(seconds * FPS), score)


def ready(
    clock: FakeClock,
    settings: EngineSettings = PLAN_SETTINGS,
    bindings: dict[str, ActionSpec] | None = None,
) -> Script:
    """An engine whose startup cooldown is already over."""
    engine = GestureEngine(
        settings, BINDINGS if bindings is None else bindings, now=clock.now - 60.0
    )
    return Script(engine, clock)


def fired(events: list[EngineEvent]) -> list[str]:
    return [event.label for event in events if isinstance(event, Triggered)]


# --- onset ----------------------------------------------------------------------------------


def test_fires_on_the_10th_frame_not_the_9th(clock: FakeClock) -> None:
    run = ready(clock)
    assert run.frames("open_palm", 9) == []
    assert run.frames("open_palm", 1) == [Triggered("open_palm", PLAY)]


def test_a_frame_under_the_threshold_resets_the_count(clock: FakeClock) -> None:
    run = ready(clock)
    events = run.frames("open_palm", 9)
    events += run.frames("open_palm", 1, score=0.59)
    events += run.frames("open_palm", 9)
    assert events == []
    assert fired(run.frames("open_palm", 1)) == ["open_palm"]


def test_alternating_labels_never_fire(clock: FakeClock) -> None:
    run = ready(clock)
    events: list[EngineEvent] = []
    for _ in range(75):
        events += run.frames("open_palm", 1) + run.frames("closed_fist", 1)
    assert events == []


def test_holding_5_s_fires_exactly_once(clock: FakeClock) -> None:
    run = ready(clock)
    assert fired(run.seconds("open_palm", 5.0)) == ["open_palm"]


def test_a_half_second_gap_still_fires_once(clock: FakeClock) -> None:
    run = ready(clock)
    events = run.seconds("open_palm", 1.0) + run.seconds(None, 0.5) + run.seconds("open_palm", 1.0)
    assert fired(events) == ["open_palm"]


def test_palm_then_pointing_for_0_4_s_then_palm_fires_palm_once(clock: FakeClock) -> None:
    run = ready(clock)
    events = run.seconds("open_palm", 0.5)
    events += run.seconds("pointing_up", 0.4)
    events += run.seconds("open_palm", 1.0)
    assert fired(events) == ["open_palm"]
    # The pointing finger got its own segment, born in the palm's cooldown.
    assert Ignored("pointing_up", "cooldown") in events


def test_release_then_redo_fires_again_once_the_cooldown_is_over(clock: FakeClock) -> None:
    run = ready(clock)
    events = run.frames("open_palm", 10)  # fires on the 10th frame
    events += run.seconds(None, 0.8)  # released
    events += run.frames("open_palm", 10)  # onset 1.13 s after the first fire
    assert fired(events) == ["open_palm", "open_palm"]


def test_redo_during_the_cooldown_is_ignored_for_the_whole_segment(clock: FakeClock) -> None:
    run = ready(clock, EngineSettings(cooldown_s=3.0))
    events = run.frames("open_palm", 10)
    events += run.seconds(None, 0.8)
    assert fired(events) == ["open_palm"]

    # Redone 1.13 s after the fire: born in the cooldown, ignored, and still nothing when the
    # cooldown ends while the gesture is held.
    redo = run.seconds("open_palm", 4.0)
    assert redo == [Ignored("open_palm", "cooldown")]

    # Released and redone after the cooldown: fires.
    again = run.seconds(None, 0.8) + run.frames("open_palm", 10)
    assert again == [Triggered("open_palm", PLAY)]


def test_a_gesture_stabilised_in_the_cooldown_of_another_is_ignored_until_redone(
    clock: FakeClock,
) -> None:
    run = ready(clock)
    events = run.frames("open_palm", 10)
    fist = run.seconds("closed_fist", 3.0)  # born 0.33 s after the palm fired
    assert fired(events) == ["open_palm"]
    assert fist == [Ignored("closed_fist", "cooldown")]
    assert fired(run.seconds(None, 0.8) + run.frames("closed_fist", 10)) == ["closed_fist"]


def test_none_and_unmapped_gestures_never_fire(clock: FakeClock) -> None:
    run = ready(clock)
    assert run.seconds("none", 3.0, score=0.99) == []
    assert run.seconds("victory", 3.0) == [Ignored("victory", "unmapped")]


# --- repeat_while_held ----------------------------------------------------------------------


def test_repeat_fires_after_the_delay_then_every_interval_and_stops_on_release(
    clock: FakeClock,
) -> None:
    run = ready(clock)
    onset = run.frames("thumb_up", 10)
    held = run.seconds("thumb_up", 2.0)
    released = run.seconds(None, 2.0)
    assert onset == [Triggered("thumb_up", VOLUME_UP)]
    # Repeats at 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7 and 1.9 s after the onset.
    assert held == [Triggered("thumb_up", VOLUME_UP, repeat=True)] * 8
    assert released == []


def test_repeat_only_on_frames_where_the_gesture_is_visible(clock: FakeClock) -> None:
    run = ready(clock)
    run.frames("thumb_up", 10)
    assert len(run.seconds("thumb_up", 0.5)) == 1  # the first repeat, at 0.5 s
    assert run.seconds(None, 0.4) == []  # a repeat was due at 0.7 s: not visible, no fire
    back = [run.frames("thumb_up", 1) for _ in range(12)]
    # One repeat on return (overdue), then every 6 frames, never a burst to catch up.
    assert [len(events) for events in back] == [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]


def test_repeats_keep_other_gestures_in_cooldown(clock: FakeClock) -> None:
    run = ready(clock)
    run.seconds("thumb_up", 2.0)
    assert run.seconds("closed_fist", 0.5) == [Ignored("closed_fist", "cooldown")]


# --- arming ---------------------------------------------------------------------------------


def test_arm_gesture_disarms_then_rearms(clock: FakeClock) -> None:
    run = ready(clock)
    assert run.frames("i_love_you", 14) == []
    assert run.frames("i_love_you", 1) == [ArmedChanged(False)]
    assert run.engine.armed is False

    events = run.seconds(None, 2.0) + run.seconds("open_palm", 1.0)
    assert events == [Ignored("open_palm", "disarmed")]

    events = run.seconds(None, 1.0) + run.seconds("i_love_you", 1.0)
    assert events == [ArmedChanged(True)]
    events = run.seconds(None, 1.0) + run.seconds("open_palm", 1.0)
    assert events == [Triggered("open_palm", PLAY)]


def test_arm_gesture_disarms_right_after_a_repeated_thumb_up(clock: FakeClock) -> None:
    run = ready(clock)
    repeats = run.seconds("thumb_up", 1.5)
    assert len(fired(repeats)) >= 3  # onset + repeats: the cooldown is running
    # The hand goes straight to the arm gesture: 15 frames later it toggles, cooldown or not.
    assert run.frames("i_love_you", 15) == [ArmedChanged(False)]
    # Thumb up comes back while its segment is still alive: its repetition is over.
    assert run.seconds("thumb_up", 2.0) == []


def test_rearming_does_not_resume_a_repetition_from_before_the_disarm(clock: FakeClock) -> None:
    run = ready(clock)
    run.seconds("thumb_up", 1.0)
    assert run.frames("i_love_you", 15) == [ArmedChanged(False)]
    # Thumb up keeps its segment alive while the arm gesture's segment is released...
    assert run.seconds("thumb_up", 0.9) == []
    # ...then the arm gesture rearms while that thumb up segment is still alive.
    assert run.frames("i_love_you", 15) == [ArmedChanged(True)]
    assert run.seconds("thumb_up", 2.0) == []


def test_arm_gesture_toggles_once_per_hold(clock: FakeClock) -> None:
    run = ready(clock)
    assert run.seconds("i_love_you", 5.0) == [ArmedChanged(False)]


def test_a_toggle_restarts_the_cooldown_of_the_other_gestures(clock: FakeClock) -> None:
    run = ready(clock, EngineSettings(start_armed=False))
    assert run.frames("i_love_you", 10) == [ArmedChanged(True)]
    assert run.frames("open_palm", 10) == [Ignored("open_palm", "cooldown")]


def test_without_an_arm_gesture_every_label_is_a_plain_gesture(clock: FakeClock) -> None:
    run = ready(clock, EngineSettings(arm_gesture=None), {"i_love_you": PLAY})
    assert run.frames("i_love_you", 10) == [Triggered("i_love_you", PLAY)]


# --- per-gesture overrides ------------------------------------------------------------------


def test_per_gesture_stable_frames(clock: FakeClock) -> None:
    settings = EngineSettings(per_gesture={"open_palm": GestureOverride(stable_frames=20)})
    run = ready(clock, settings)
    assert run.frames("open_palm", 19) == []
    assert fired(run.frames("open_palm", 1)) == ["open_palm"]
    # The others keep the default.
    run.seconds(None, 2.0)
    assert fired(run.frames("closed_fist", 10)) == ["closed_fist"]


def test_per_gesture_min_score(clock: FakeClock) -> None:
    settings = EngineSettings(per_gesture={"open_palm": GestureOverride(min_score=0.9)})
    run = ready(clock, settings)
    assert run.seconds("open_palm", 2.0, score=0.85) == []
    assert fired(run.seconds("closed_fist", 1.0, score=0.85)) == ["closed_fist"]
    run.seconds(None, 2.0)
    assert fired(run.frames("open_palm", 10, score=0.9)) == ["open_palm"]


# --- (re)start ------------------------------------------------------------------------------


def test_no_fire_at_restart_when_the_gesture_is_already_held(clock: FakeClock) -> None:
    run = Script(GestureEngine(PLAN_SETTINGS, BINDINGS, now=clock.now), clock)
    assert run.seconds("open_palm", 5.0) == [Ignored("open_palm", "cooldown")]
    assert fired(run.seconds(None, 0.8) + run.frames("open_palm", 10)) == ["open_palm"]


def test_no_toggle_at_restart_when_the_arm_gesture_is_already_held(clock: FakeClock) -> None:
    run = Script(GestureEngine(PLAN_SETTINGS, BINDINGS, now=clock.now), clock)
    assert run.seconds("i_love_you", 5.0) == [Ignored("i_love_you", "cooldown")]
    assert run.engine.armed is True


# --- snapshot -------------------------------------------------------------------------------


def test_snapshot_shows_arming_cooldown_and_live_segments(clock: FakeClock) -> None:
    run = ready(clock)
    run.frames("open_palm", 10)
    run.frames("closed_fist", 10)
    snap = run.engine.snapshot()
    assert snap.armed is True
    assert snap.now == clock.now
    assert abs(snap.cooldown_remaining_s - (1.0 - 10 * DT)) < 1e-9
    assert [(s.segment.label, s.outcome) for s in snap.segments] == [
        ("open_palm", "fired"),
        ("closed_fist", "cooldown"),
    ]

    run.seconds(None, 2.0)
    snap = run.engine.snapshot()
    assert snap.segments == ()
    assert snap.cooldown_remaining_s == 0.0


def test_snapshot_marks_a_repeating_segment(clock: FakeClock) -> None:
    run = ready(clock)
    run.frames("thumb_up", 10)
    [status] = run.engine.snapshot().segments
    assert status.outcome == "fired"
    assert status.repeating is True
