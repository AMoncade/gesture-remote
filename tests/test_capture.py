"""Camera (with a fake VideoCapture: the real camera is never opened) and IdleThrottle."""

from __future__ import annotations

import logging

import cv2
import numpy as np
import pytest

from conftest import FakeClock
from gesture_remote.capture import Camera, IdleThrottle
from gesture_remote.config import CameraSettings


class FakeCapture:
    """Stands in for cv2.VideoCapture; `frames` is consumed by read(), None = failed read."""

    def __init__(self, index: int, backend: int, *, opens: bool, frames: list) -> None:
        self.index, self.backend, self.opens = index, backend, opens
        self.frames = frames
        self.props: dict[int, float] = {}
        self.released = False

    def isOpened(self) -> bool:
        return self.opens and not self.released

    def set(self, prop_id: int, value: float) -> bool:
        self.props[prop_id] = value
        return True

    def get(self, prop_id: int) -> float:
        return self.props.get(prop_id, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        frame = self.frames.pop(0) if self.frames else None
        return (frame is not None), frame

    def release(self) -> None:
        self.released = True

    def getBackendName(self) -> str:
        if self.released:
            raise cv2.error("released")  # like OpenCV 5: raises once released
        return {cv2.CAP_DSHOW: "DSHOW", cv2.CAP_MSMF: "MSMF"}[self.backend]


class FakeFactory:
    """Records every open; `opening` = set of backends that open, `frames` shared by captures."""

    def __init__(self, opening: set[int], frames: list | None = None) -> None:
        self.opening = opening
        self.frames = frames if frames is not None else []
        self.opened: list[FakeCapture] = []

    def __call__(self, index: int, backend: int) -> FakeCapture:
        capture = FakeCapture(index, backend, opens=backend in self.opening, frames=self.frames)
        self.opened.append(capture)
        return capture

    @property
    def backends(self) -> list[int]:
        return [capture.backend for capture in self.opened]


def asymmetric_frame() -> np.ndarray:
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[:, :10] = (255, 0, 0)  # a blue stripe on the left edge only
    return frame


# --- Camera ---------------------------------------------------------------------------------


def test_auto_prefers_dshow_and_sets_size(caplog: pytest.LogCaptureFixture) -> None:
    factory = FakeFactory(opening={cv2.CAP_DSHOW, cv2.CAP_MSMF})
    camera = Camera(CameraSettings(index=2), capture_factory=factory)
    with caplog.at_level(logging.INFO, logger="gesture_remote.capture"):
        assert camera.open()
    assert factory.backends == [cv2.CAP_DSHOW]
    capture = factory.opened[0]
    assert capture.index == 2
    assert capture.props[cv2.CAP_PROP_FRAME_WIDTH] == 640
    assert capture.props[cv2.CAP_PROP_FRAME_HEIGHT] == 480
    assert camera.backend_name == "DSHOW"
    assert "DSHOW" in caplog.text


def test_auto_falls_back_to_msmf_when_dshow_does_not_open() -> None:
    factory = FakeFactory(opening={cv2.CAP_MSMF})
    camera = Camera(CameraSettings(), capture_factory=factory)
    assert camera.open()
    assert factory.backends == [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    assert factory.opened[0].released  # the failed DirectShow handle is not leaked
    assert camera.backend_name == "MSMF"


def test_explicit_dshow_never_tries_msmf() -> None:
    factory = FakeFactory(opening={cv2.CAP_MSMF})
    camera = Camera(CameraSettings(backend="dshow"), capture_factory=factory)
    assert not camera.open()
    assert factory.backends == [cv2.CAP_DSHOW]
    assert not camera.is_open


def test_explicit_msmf_skips_dshow() -> None:
    factory = FakeFactory(opening={cv2.CAP_DSHOW, cv2.CAP_MSMF})
    camera = Camera(CameraSettings(backend="msmf"), capture_factory=factory)
    assert camera.open()
    assert factory.backends == [cv2.CAP_MSMF]


def test_mirror_flips_the_frame_before_it_leaves_the_camera() -> None:
    frame = asymmetric_frame()
    factory = FakeFactory(opening={cv2.CAP_DSHOW}, frames=[frame.copy()])
    camera = Camera(CameraSettings(mirror=True), capture_factory=factory)
    got = camera.read()
    assert got is not None
    np.testing.assert_array_equal(got, frame[:, ::-1])
    assert got[0, -1].tolist() == [255, 0, 0] and got[0, 0].tolist() == [0, 0, 0]
    assert got.flags["C_CONTIGUOUS"]


def test_no_mirror_leaves_the_frame_alone() -> None:
    frame = asymmetric_frame()
    factory = FakeFactory(opening={cv2.CAP_DSHOW}, frames=[frame.copy()])
    camera = Camera(CameraSettings(mirror=False), capture_factory=factory)
    got = camera.read()
    assert got is not None
    np.testing.assert_array_equal(got, frame)


def test_read_opens_lazily_and_returns_none_when_no_camera(clock: FakeClock) -> None:
    factory = FakeFactory(opening=set())
    camera = Camera(CameraSettings(), capture_factory=factory, clock=clock, reopen_interval_s=2.0)
    assert camera.read() is None
    assert len(factory.opened) == 2  # DSHOW then MSMF
    clock.advance(1.0)
    assert camera.read() is None
    assert len(factory.opened) == 2  # too early to retry
    clock.advance(1.0)
    assert camera.read() is None
    assert len(factory.opened) == 4  # retried both backends


def test_camera_lost_is_released_then_reopened(
    clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    frames: list = [asymmetric_frame(), None, None, None]
    factory = FakeFactory(opening={cv2.CAP_DSHOW}, frames=frames)
    camera = Camera(
        CameraSettings(),
        capture_factory=factory,
        clock=clock,
        reopen_interval_s=2.0,
        max_failed_reads=3,
    )
    with caplog.at_level(logging.WARNING, logger="gesture_remote.capture"):
        assert camera.read() is not None
        assert [camera.read() for _ in range(3)] == [None, None, None]
    first = factory.opened[0]
    assert first.released and not camera.is_open
    assert "releasing it" in caplog.text

    frames.append(asymmetric_frame())  # Teams let go of the camera
    clock.advance(1.0)
    assert camera.read() is None and len(factory.opened) == 1
    clock.advance(1.0)
    assert camera.read() is not None
    assert len(factory.opened) == 2 and camera.is_open


def test_failed_reads_below_the_limit_keep_the_camera(clock: FakeClock) -> None:
    frames: list = [None, None, asymmetric_frame(), None, None, asymmetric_frame()]
    factory = FakeFactory(opening={cv2.CAP_DSHOW}, frames=frames)
    camera = Camera(CameraSettings(), capture_factory=factory, clock=clock, max_failed_reads=3)
    results = [camera.read() is not None for _ in range(6)]
    assert results == [False, False, True, False, False, True]
    assert len(factory.opened) == 1 and camera.is_open


def test_close_releases_and_context_manager_closes() -> None:
    factory = FakeFactory(opening={cv2.CAP_DSHOW})
    with Camera(CameraSettings(), capture_factory=factory) as camera:
        assert camera.open()
    assert factory.opened[0].released
    assert camera.backend_name is None


# --- IdleThrottle ---------------------------------------------------------------------------

CAMERA_FPS = 30.0


def run_frames(
    throttle: IdleThrottle, clock: FakeClock, seconds: float, hand: bool = False
) -> list[bool]:
    """Feed `seconds` of frames at CAMERA_FPS; report `hand` for every inferred frame."""
    decisions = []
    for _ in range(round(seconds * CAMERA_FPS)):
        clock.advance(1.0 / CAMERA_FPS)
        decision = throttle.should_infer()
        if decision:
            throttle.report(hand)
        decisions.append(decision)
    return decisions


def test_idle_runs_at_about_fps(clock: FakeClock) -> None:
    throttle = IdleThrottle(fps=5.0, after_s=1.0, clock=clock)
    decisions = run_frames(throttle, clock, 10.0)
    assert sum(decisions) in (50, 51)
    # evenly spread: never two inferred frames in a row at 30 fps camera / 5 fps idle
    assert not any(a and b for a, b in zip(decisions, decisions[1:], strict=False))


def test_first_frame_is_always_inferred(clock: FakeClock) -> None:
    throttle = IdleThrottle(fps=1.0, after_s=1.0, clock=clock)
    assert throttle.should_infer()
    assert not throttle.should_infer()


def test_every_frame_passes_while_a_hand_is_seen(clock: FakeClock) -> None:
    throttle = IdleThrottle(fps=5.0, after_s=1.0, clock=clock)
    run_frames(throttle, clock, 1.0)  # idle: wait for the first sampled frame with a hand
    decisions = run_frames(throttle, clock, 3.0, hand=True)
    assert sum(decisions) >= len(decisions) - 6  # at most the wait for the first sample
    assert all(decisions[6:])
    assert throttle.active


def test_back_to_idle_after_after_s_without_hand(clock: FakeClock) -> None:
    throttle = IdleThrottle(fps=5.0, after_s=1.0, clock=clock)
    throttle.should_infer()
    throttle.report(True)
    within = run_frames(throttle, clock, 1.0)  # the hand left: still full rate for after_s
    assert all(within[:29])
    later = run_frames(throttle, clock, 2.0)
    assert not throttle.active
    assert sum(later) in (10, 11)


def test_after_s_zero_only_keeps_the_frame_where_the_hand_was_seen(clock: FakeClock) -> None:
    throttle = IdleThrottle(fps=5.0, after_s=0.0, clock=clock)
    decisions = run_frames(throttle, clock, 2.0, hand=True)
    assert sum(decisions) in (10, 11)


def test_long_gap_resyncs_instead_of_bursting(clock: FakeClock) -> None:
    throttle = IdleThrottle(fps=5.0, after_s=1.0, clock=clock)
    run_frames(throttle, clock, 1.0)
    clock.advance(60.0)  # the loop stalled (camera reopening)
    decisions = run_frames(throttle, clock, 1.0)
    assert sum(decisions) in (5, 6)


@pytest.mark.parametrize(("fps", "after_s"), [(0.0, 1.0), (-1.0, 1.0), (5.0, -0.1)])
def test_invalid_parameters_raise(fps: float, after_s: float, clock: FakeClock) -> None:
    with pytest.raises(ValueError):
        IdleThrottle(fps=fps, after_s=after_s, clock=clock)
