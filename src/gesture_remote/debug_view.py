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

# BGR
BONE_COLOR = (235, 200, 80)
JOINT_COLOR = (255, 255, 255)
JOINT_RING_COLOR = (180, 120, 30)
TEXT_COLOR = (255, 255, 255)
DIM_TEXT_COLOR = (175, 175, 175)
PANEL_COLOR = (25, 20, 18)
PANEL_ALPHA = 0.6
ARMED_COLOR = (90, 170, 40)
DISARMED_COLOR = (60, 60, 210)
FIRED_COLOR = (40, 170, 240)
WARNING_COLOR = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
TITLE_SCALE = 0.65
FONT_SCALE = 0.5
SMALL_SCALE = 0.42
PADDING = 10
LINE_GAP = 9


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
    last_fired: str | None = None,
) -> list[str]:
    """The overlay text, top to bottom."""
    lines = [describe_hand(observation.primary()) if observation else "idle: frame not inferred"]
    if snapshot is not None:
        state = "ARMED" if snapshot.armed else "DISARMED"
        if snapshot.cooldown_remaining_s > 0:
            state += f"  cooldown {snapshot.cooldown_remaining_s:.1f} s"
        lines.append(state)
    if last_fired:
        lines.append(f"FIRED {last_fired}")
    if snapshot is not None:
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
        cv2.line(image, tuple(points[start]), tuple(points[end]), BONE_COLOR, 3, cv2.LINE_AA)
    # every ring first, then every centre: close joints must not paint over each other's centre
    for x, y in points:
        cv2.circle(image, (int(x), int(y)), 6, JOINT_RING_COLOR, -1, cv2.LINE_AA)
    for x, y in points:
        cv2.circle(image, (int(x), int(y)), 3, JOINT_COLOR, -1, cv2.LINE_AA)


def _line_style(index: int, text: str) -> tuple[float, int, tuple[int, int, int] | None]:
    """(font scale, thickness, pill colour or None) for one overlay line."""
    if index == 0:
        return TITLE_SCALE, 2, None
    if text.startswith("ARMED"):
        return FONT_SCALE, 1, ARMED_COLOR
    if text.startswith("DISARMED"):
        return FONT_SCALE, 1, DISARMED_COLOR
    if text.startswith("FIRED"):
        return FONT_SCALE, 1, FIRED_COLOR
    if text.startswith("RESTART"):
        return FONT_SCALE, 1, WARNING_COLOR
    return SMALL_SCALE, 1, None


def draw_overlay(
    frame: np.ndarray, observation: FrameObservation | None, lines: Sequence[str]
) -> np.ndarray:
    """A copy of `frame` (BGR, as shown) with every hand and a translucent text panel."""
    image = frame.copy()
    if observation is not None:
        for hand in observation.hands:
            draw_hand(image, hand)
    if not lines:
        return image

    styles = [_line_style(index, text) for index, text in enumerate(lines)]
    sizes = [
        cv2.getTextSize(text, FONT, scale, thick)
        for text, (scale, thick, _) in zip(lines, styles, strict=True)
    ]
    width = max(w for (w, _), _ in sizes) + 2 * PADDING + 12
    height = sum(h + b + LINE_GAP for (_, h), b in sizes) + PADDING
    left, top = 10, 10
    right, bottom = min(left + width, image.shape[1] - 1), min(top + height, image.shape[0] - 1)
    panel = image[top:bottom, left:right]
    panel[:] = (panel * (1 - PANEL_ALPHA) + np.array(PANEL_COLOR) * PANEL_ALPHA).astype(np.uint8)

    y = top + PADDING
    for text, (scale, thick, pill), ((text_width, text_height), baseline) in zip(
        lines, styles, sizes, strict=True
    ):
        y += text_height
        x = left + PADDING
        if pill is not None:
            cv2.rectangle(
                image,
                (x - 5, y - text_height - 5),
                (x + text_width + 5, y + baseline + 2),
                pill,
                -1,
            )
        color = DIM_TEXT_COLOR if pill is None and scale == SMALL_SCALE else TEXT_COLOR
        cv2.putText(image, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
        y += baseline + LINE_GAP
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
