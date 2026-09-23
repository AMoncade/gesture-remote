"""--debug overlay: the camera image, the hand skeleton, the label and the engine state.

Drawing (`overlay_lines`, `draw_overlay`) is pure and tested on synthetic arrays; `DebugWindow`
only shows the result. Nothing is ever written to disk from here.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from gesture_remote.observation import FrameObservation, HandObservation

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # middle
    (9, 10), (10, 11), (11, 12),
    # ring
    (13, 14), (14, 15), (15, 16),
    # pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # palm
    (5, 9), (9, 13), (13, 17),
)  # fmt: skip
"""Our own table: `mp.solutions` (and its HAND_CONNECTIONS) is gone from mediapipe >= 0.10.30."""

BONE_COLOR = (255, 255, 255)
JOINT_COLOR = (0, 200, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BACKGROUND = (0, 0, 0)
WARNING_COLOR = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
LINE_HEIGHT = 20


# --- what the overlay reads from the engine (structural: no import of engine.py) -------------


class SegmentLike(Protocol):
    @property
    def label(self) -> str: ...


class SegmentStatusLike(Protocol):
    @property
    def segment(self) -> SegmentLike: ...

    @property
    def outcome(self) -> str: ...

    @property
    def repeating(self) -> bool: ...


class EngineSnapshotLike(Protocol):
    @property
    def armed(self) -> bool: ...

    @property
    def cooldown_remaining_s(self) -> float: ...

    @property
    def segments(self) -> Sequence[SegmentStatusLike]: ...


@dataclass(frozen=True, slots=True)
class FrameStats:
    inferred: bool
    """False when the idle throttle skipped inference on this frame."""
    inference_ms: float | None
    """Duration of the latest inference, None before the first one."""
    camera_fps: float
    inference_fps: float


class RateMeter:
    """Events per second, smoothed (exponential moving average of the intervals)."""

    def __init__(self, clock: Callable[[], float], smoothing: float = 0.1) -> None:
        self._clock = clock
        self._smoothing = smoothing
        self._last: float | None = None
        self._interval: float | None = None

    def tick(self) -> None:
        now = self._clock()
        if self._last is not None and now > self._last:
            interval = now - self._last
            if self._interval is None:
                self._interval = interval
            else:
                self._interval += self._smoothing * (interval - self._interval)
        self._last = now

    @property
    def rate(self) -> float:
        return 0.0 if not self._interval else 1.0 / self._interval


# --- pure drawing ---------------------------------------------------------------------------


def describe_hand(hand: HandObservation | None) -> str:
    if hand is None:
        return "no hand"
    return (
        f"{hand.label} {hand.score:.2f}  "
        f"{hand.handedness.value.capitalize()} ({hand.handedness_score:.2f})"
    )


def overlay_lines(
    observation: FrameObservation | None,
    snapshot: EngineSnapshotLike | None,
    stats: FrameStats,
    restart_required: Sequence[str] = (),
) -> list[str]:
    """The overlay text, top to bottom."""
    lines = [describe_hand(observation.primary()) if observation else "idle: frame not inferred"]
    if snapshot is not None:
        state = "ARMED" if snapshot.armed else "DISARMED"
        if snapshot.cooldown_remaining_s > 0:
            state += f"  cooldown {snapshot.cooldown_remaining_s:.1f} s"
        lines.append(state)
        for status in snapshot.segments:
            repeating = ", repeating" if status.repeating else ""
            lines.append(f"segment {status.segment.label}: {status.outcome}{repeating}")
    inference = "-" if stats.inference_ms is None else f"{stats.inference_ms:.1f} ms"
    lines.append(
        f"inference {inference}  camera {stats.camera_fps:.1f} fps  "
        f"inferred {stats.inference_fps:.1f} fps"
    )
    if restart_required:
        lines.append(f"RESTART REQUIRED: {', '.join(restart_required)} changed")
    return lines


def landmark_pixels(landmarks: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """(21, 3) normalised landmarks -> (21, 2) integer pixel positions."""
    width, height = image_size
    xy = np.asarray(landmarks, dtype=np.float64)[:, :2] * (width, height)
    return np.round(xy).astype(np.int32)


def draw_hand(image: np.ndarray, hand: HandObservation) -> None:
    """Draw the 21-point skeleton of `hand` on `image`, in place."""
    height, width = image.shape[:2]
    points = landmark_pixels(hand.landmarks, (width, height))
    for start, end in HAND_CONNECTIONS:
        cv2.line(image, tuple(points[start]), tuple(points[end]), BONE_COLOR, 2, cv2.LINE_AA)
    for x, y in points:
        cv2.circle(image, (int(x), int(y)), 4, JOINT_COLOR, -1, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray, observation: FrameObservation | None, lines: Sequence[str]
) -> np.ndarray:
    """A copy of `frame` (BGR, as shown) with every hand and the text lines drawn on it."""
    image = frame.copy()
    if observation is not None:
        for hand in observation.hands:
            draw_hand(image, hand)
    for index, text in enumerate(lines):
        origin = (8, LINE_HEIGHT * (index + 1))
        (text_width, text_height), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, 1)
        top_left = (origin[0] - 3, origin[1] - text_height - 3)
        bottom_right = (origin[0] + text_width + 3, origin[1] + baseline)
        cv2.rectangle(image, top_left, bottom_right, TEXT_BACKGROUND, -1)
        color = WARNING_COLOR if text.startswith(("DISARMED", "RESTART")) else TEXT_COLOR
        cv2.putText(image, text, origin, FONT, FONT_SCALE, color, 1, cv2.LINE_AA)
    return image


# --- display (not called by the tests) ------------------------------------------------------


class DebugWindow:
    """An OpenCV window. Must be used from a single thread (the pipeline's)."""

    TITLE = "gesture-remote (debug)"

    def show(self, image: np.ndarray) -> bool:
        """Display `image`; False once the user pressed q or Esc or closed the window."""
        cv2.imshow(self.TITLE, image)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        return cv2.getWindowProperty(self.TITLE, cv2.WND_PROP_VISIBLE) >= 1

    def close(self) -> None:
        with contextlib.suppress(cv2.error):  # never shown, or already closed by the user
            cv2.destroyWindow(self.TITLE)
