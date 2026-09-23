"""debug_view drawing on synthetic arrays (the window itself is never opened in tests)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from conftest import FakeClock
from gesture_remote.debug_view import (
    BONE_COLOR,
    HAND_CONNECTIONS,
    JOINT_COLOR,
    FrameStats,
    RateMeter,
    draw_overlay,
    landmark_pixels,
    overlay_lines,
)
from gesture_remote.observation import NUM_LANDMARKS, FrameObservation, Handedness, HandObservation

WIDTH, HEIGHT = 640, 480
STATS = FrameStats(inferred=True, inference_ms=12.34, camera_fps=15.0, inference_fps=14.8)


@dataclass(frozen=True)
class FakeSegment:
    label: str


@dataclass(frozen=True)
class FakeStatus:
    segment: FakeSegment
    outcome: str
    repeating: bool


@dataclass(frozen=True)
class FakeSnapshot:
    armed: bool
    cooldown_remaining_s: float
    segments: tuple[FakeStatus, ...]


def hand(label: str = "victory", score: float = 0.91) -> HandObservation:
    # A fan of points in the lower right quarter, far from the text in the top left corner.
    angles = np.linspace(0.2, 1.3, NUM_LANDMARKS)
    radius = np.linspace(0.0, 0.3, NUM_LANDMARKS)
    landmarks = np.zeros((NUM_LANDMARKS, 3), np.float32)
    landmarks[:, 0] = 0.55 + radius * np.cos(angles)
    landmarks[:, 1] = 0.55 + radius * np.sin(angles) * 0.8
    return HandObservation(
        label=label,
        score=score,
        handedness=Handedness.RIGHT,
        handedness_score=0.98,
        landmarks=landmarks,
        world_landmarks=np.zeros((NUM_LANDMARKS, 3), np.float32),
    )


def observation(*hands: HandObservation) -> FrameObservation:
    return FrameObservation(hands=hands, image_size=(WIDTH, HEIGHT))


def test_connection_table_is_a_connected_hand_over_21_points() -> None:
    assert len(HAND_CONNECTIONS) == len(set(HAND_CONNECTIONS)) == 21
    points = {point for edge in HAND_CONNECTIONS for point in edge}
    assert points == set(range(NUM_LANDMARKS))
    reached, frontier = {0}, [0]
    while frontier:
        current = frontier.pop()
        for a, b in HAND_CONNECTIONS:
            for other in (b if a == current else None, a if b == current else None):
                if other is not None and other not in reached:
                    reached.add(other)
                    frontier.append(other)
    assert reached == points
    fingertips = {4, 8, 12, 16, 20}
    degree = {p: sum(p in edge for edge in HAND_CONNECTIONS) for p in points}
    assert all(degree[tip] == 1 for tip in fingertips)


def test_landmark_pixels_scale_to_the_image() -> None:
    landmarks = np.zeros((NUM_LANDMARKS, 3), np.float32)
    landmarks[3] = (0.5, 0.25, -0.1)
    landmarks[4] = (1.0, 1.0, 0.0)
    pixels = landmark_pixels(landmarks, (WIDTH, HEIGHT))
    assert pixels.shape == (NUM_LANDMARKS, 2)
    assert pixels[3].tolist() == [320, 120]
    assert pixels[4].tolist() == [640, 480]


def test_overlay_draws_joints_and_bones_and_keeps_the_input() -> None:
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    the_hand = hand()
    image = draw_overlay(frame, observation(the_hand), [])
    assert not frame.any(), "the camera frame must not be modified"
    assert image.shape == frame.shape and image.dtype == np.uint8
    pixels = landmark_pixels(the_hand.landmarks, (WIDTH, HEIGHT))
    for x, y in pixels:
        assert image[y, x].tolist() == list(JOINT_COLOR)
    # middle of a long bone (wrist -> pinky base) is painted with the bone colour
    (x0, y0), (x1, y1) = pixels[0], pixels[17]
    mid = image[(y0 + y1) // 2, (x0 + x1) // 2]
    assert mid.tolist() == list(BONE_COLOR)


def test_overlay_without_hand_draws_only_the_text() -> None:
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    image = draw_overlay(frame, observation(), ["ARMED"])
    assert image[: HEIGHT // 3, : WIDTH // 3].any()  # text in the top left corner
    assert not image[HEIGHT // 3 :, :].any()  # nothing else


def test_lines_show_label_state_segments_and_timings() -> None:
    snapshot = FakeSnapshot(
        armed=True,
        cooldown_remaining_s=0.42,
        segments=(
            FakeStatus(FakeSegment("thumb_up"), "fired", repeating=True),
            FakeStatus(FakeSegment("closed_fist"), "cooldown", repeating=False),
        ),
    )
    lines = overlay_lines(observation(hand()), snapshot, STATS)
    assert lines == [
        "victory 0.91  Right (0.98)",
        "ARMED  cooldown 0.4 s",
        "segment thumb_up: fired, repeating",
        "segment closed_fist: cooldown",
        "inference 12.3 ms  camera 15.0 fps  inferred 14.8 fps",
    ]


def test_lines_when_disarmed_idle_and_restart_required() -> None:
    snapshot = FakeSnapshot(armed=False, cooldown_remaining_s=0.0, segments=())
    stats = FrameStats(inferred=False, inference_ms=None, camera_fps=15.0, inference_fps=5.0)
    lines = overlay_lines(None, snapshot, stats, restart_required=("camera",))
    assert lines == [
        "idle: frame not inferred",
        "DISARMED",
        "inference -  camera 15.0 fps  inferred 5.0 fps",
        "RESTART REQUIRED: camera changed",
    ]


def test_the_last_fired_gesture_is_shown_under_the_state() -> None:
    snapshot = FakeSnapshot(armed=True, cooldown_remaining_s=0.0, segments=())
    lines = overlay_lines(observation(hand()), snapshot, STATS, last_fired="victory -> url")
    assert lines[1:3] == ["ARMED", "FIRED victory -> url"]


def test_tray_icon_image() -> None:
    from gesture_remote.tray import ARMED_RGB, icon_image

    image = icon_image(ARMED_RGB)
    assert image.size == (64, 64) and image.mode == "RGBA"
    assert image.getpixel((8, 32))[:3] == ARMED_RGB  # inside the disc, left of the hand
    assert image.getpixel((0, 0))[3] == 0  # transparent corner


def test_lines_without_hand_or_engine() -> None:
    assert overlay_lines(observation(), None, STATS)[0] == "no hand"


def test_primary_hand_is_described() -> None:
    lines = overlay_lines(observation(hand("none", 0.4), hand("thumb_up", 0.8)), None, STATS)
    assert lines[0].startswith("thumb_up 0.80")


def test_rate_meter(clock: FakeClock) -> None:
    meter = RateMeter(clock, smoothing=0.5)
    assert meter.rate == 0.0
    meter.tick()
    assert meter.rate == 0.0
    for _ in range(20):
        clock.advance(1 / 15)
        meter.tick()
    assert meter.rate == pytest.approx(15.0)
    for _ in range(20):
        clock.advance(1 / 5)
        meter.tick()
    assert meter.rate == pytest.approx(5.0, rel=1e-3)
