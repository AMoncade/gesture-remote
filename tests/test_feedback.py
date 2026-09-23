"""Feedback: which tones for which event, silence rules, and a beeper that never blocks."""

from __future__ import annotations

import threading
from collections.abc import Sequence

from gesture_remote.feedback import Feedback, ThreadedBeeper, Tone

WAIT_S = 10.0


class FakeBeeper:
    def __init__(self) -> None:
        self.played: list[tuple[Tone, ...]] = []

    def __call__(self, tones: Sequence[Tone]) -> None:
        self.played.append(tuple(tones))


def frequencies(tones: Sequence[Tone]) -> list[int]:
    return [frequency for frequency, _ in tones]


def test_arming_rises_and_disarming_falls() -> None:
    beeper = FakeBeeper()
    feedback = Feedback(sound=True, beeper=beeper)
    feedback.armed_changed(True)
    feedback.armed_changed(False)
    rising, falling = (frequencies(tones) for tones in beeper.played)
    assert len(rising) >= 2 and rising == sorted(rising) and len(set(rising)) == len(rising)
    assert falling == sorted(falling, reverse=True) and len(set(falling)) == len(falling)


def test_one_beep_per_fire() -> None:
    beeper = FakeBeeper()
    feedback = Feedback(sound=True, beeper=beeper)
    feedback.triggered()
    feedback.triggered()
    assert [len(tones) for tones in beeper.played] == [1, 1]


def test_ignored_is_silent() -> None:
    beeper = FakeBeeper()
    Feedback(sound=True, beeper=beeper).ignored()
    assert beeper.played == []


def test_sound_off_is_silent_and_can_be_switched_on_by_a_reload() -> None:
    beeper = FakeBeeper()
    feedback = Feedback(sound=False, beeper=beeper)
    feedback.armed_changed(True)
    feedback.triggered()
    assert beeper.played == []
    feedback.sound = True
    feedback.triggered()
    assert len(beeper.played) == 1


def test_threaded_beeper_returns_before_the_tones_play_and_keeps_their_order() -> None:
    release = threading.Event()
    heard: list[tuple[int, int]] = []
    done = threading.Event()

    def blocking_beep(frequency: int, duration_ms: int) -> None:
        release.wait(WAIT_S)
        heard.append((frequency, duration_ms))
        if len(heard) == 3:
            done.set()

    beeper = ThreadedBeeper(blocking_beep)
    beeper([(660, 90), (990, 90)])
    beeper([(880, 60)])
    assert heard == []
    release.set()
    assert done.wait(WAIT_S)
    assert heard == [(660, 90), (990, 90), (880, 60)]


def test_threaded_beeper_survives_a_failing_beep() -> None:
    heard: list[int] = []
    done = threading.Event()

    def flaky_beep(frequency: int, duration_ms: int) -> None:
        if frequency == 1:
            raise RuntimeError("no sound device")
        heard.append(frequency)
        done.set()

    beeper = ThreadedBeeper(flaky_beep)
    beeper([(1, 10)])
    beeper([(440, 10)])
    assert done.wait(WAIT_S)
    assert heard == [440]
