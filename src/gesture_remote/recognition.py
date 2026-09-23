"""Gesture recognition: RGB frame -> FrameObservation.

`Recognizer` is the phase-2 seam: a custom model will replace MediaPipe behind the same protocol.
Never log landmarks or images here: labels, scores and timings only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from types import TracebackType
from typing import Any, Protocol, Self

import numpy as np

from gesture_remote.config import RecognitionSettings
from gesture_remote.observation import (
    NONE_LABEL,
    NUM_LANDMARKS,
    FrameObservation,
    Handedness,
    HandObservation,
)

logger = logging.getLogger(__name__)

CANNED_LABELS: Mapping[str, str] = {
    # The embedded labels.txt says "None"; the documentation says "Unknown". Accept both.
    "None": NONE_LABEL,
    "Unknown": NONE_LABEL,
    "Closed_Fist": "closed_fist",
    "Open_Palm": "open_palm",
    "Pointing_Up": "pointing_up",
    "Thumb_Down": "thumb_down",
    "Thumb_Up": "thumb_up",
    "Victory": "victory",
    "ILoveYou": "i_love_you",
}
"""Raw MediaPipe canned-gesture class name -> our snake_case label."""

CANNED_LABEL_SET: frozenset[str] = frozenset(CANNED_LABELS.values())
"""The 8 labels the canned recognizer can emit, `none` included."""

_HANDEDNESS = {"Left": Handedness.LEFT, "Right": Handedness.RIGHT}


class Recognizer(Protocol):
    """Anything that turns an RGB frame into a FrameObservation."""

    labels: frozenset[str]
    """Every label `recognize` can emit, `none` included."""

    def recognize(self, rgb: np.ndarray, timestamp_ms: int) -> FrameObservation: ...

    def close(self) -> None: ...


class StrictTimestamps:
    """Make millisecond timestamps strictly increasing (MediaPipe VIDEO mode raises otherwise).

    A clock that stalls or goes back yields last + 1 instead.
    """

    def __init__(self) -> None:
        self._last: int | None = None

    def __call__(self, timestamp_ms: int) -> int:
        value = int(timestamp_ms)
        if self._last is not None and value <= self._last:
            value = self._last + 1
        self._last = value
        return value


class LabelMapper:
    """Raw class name -> label through a fixed table; an unknown name warns once, then is none."""

    def __init__(self, table: Mapping[str, str] = CANNED_LABELS) -> None:
        self._table = dict(table)
        self._warned: set[str] = set()

    def __call__(self, raw_name: str) -> str:
        label = self._table.get(raw_name)
        if label is not None:
            return label
        if raw_name not in self._warned:
            self._warned.add(raw_name)
            logger.warning(
                "unknown gesture class %r from the model: treated as %r", raw_name, NONE_LABEL
            )
        return NONE_LABEL


def _points(landmarks: Sequence[Any]) -> np.ndarray:
    points = np.array([(point.x, point.y, point.z) for point in landmarks], dtype=np.float32)
    if points.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected {NUM_LANDMARKS} landmarks, got {len(landmarks)}")
    return points


def to_observation(
    result: Any, image_size: tuple[int, int], map_label: Callable[[str], str]
) -> FrameObservation:
    """Convert a MediaPipe GestureRecognizerResult (read by duck typing) to a FrameObservation.

    Reads `gestures`, `handedness`, `hand_landmarks`, `hand_world_landmarks`: one list per hand,
    each gesture/handedness list sorted by score with `category_name` and `score`. A hand whose
    gesture list is empty gets the `none` label with score 0.
    """
    hands = []
    for gestures, handedness, landmarks, world in zip(
        result.gestures,
        result.handedness,
        result.hand_landmarks,
        result.hand_world_landmarks,
        strict=True,
    ):
        if gestures:
            label, score = map_label(gestures[0].category_name), float(gestures[0].score)
        else:
            label, score = NONE_LABEL, 0.0
        side = handedness[0]
        hands.append(
            HandObservation(
                label=label,
                score=score,
                handedness=_HANDEDNESS[side.category_name],
                handedness_score=float(side.score),
                landmarks=_points(landmarks),
                world_landmarks=_points(world),
            )
        )
    return FrameObservation(hands=tuple(hands), image_size=image_size)


class MediaPipeGestureRecognizer:
    """MediaPipe canned gesture recognizer, VIDEO mode, synchronous.

    `model` holds the bytes of gesture_recognizer.task: the caller reads the file (its path is
    resolved against the config folder by the loader), so this class touches no file itself.
    """

    labels: frozenset[str] = CANNED_LABEL_SET

    def __init__(self, model: bytes, settings: RecognitionSettings) -> None:
        # Imported here so the pure helpers above stay usable without loading the native library.
        import mediapipe as mp
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import (
            GestureRecognizer,
            GestureRecognizerOptions,
            RunningMode,
        )

        self._mp = mp
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_buffer=model),
            running_mode=RunningMode.VIDEO,
            num_hands=settings.num_hands,
            min_hand_detection_confidence=settings.min_hand_detection_confidence,
            min_hand_presence_confidence=settings.min_hand_presence_confidence,
            min_tracking_confidence=settings.min_tracking_confidence,
        )
        self._recognizer = GestureRecognizer.create_from_options(options)
        self._timestamps = StrictTimestamps()
        self._map_label = LabelMapper()
        logger.info(
            "gesture recognizer ready: model %d bytes, VIDEO mode, num_hands=%d",
            len(model),
            settings.num_hands,
        )

    def recognize(self, rgb: np.ndarray, timestamp_ms: int) -> FrameObservation:
        """`rgb`: (height, width, 3) uint8 in RGB order. Any memory layout is accepted."""
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"expected an (h, w, 3) uint8 RGB image, got {rgb.shape} {rgb.dtype}")
        # mp.Image ignores numpy strides: a flipped or channel-reversed *view* is silently read as
        # the original buffer (measured with mediapipe 1.0.1). Always hand it contiguous memory.
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        result = self._recognizer.recognize_for_video(image, self._timestamps(timestamp_ms))
        height, width = rgb.shape[:2]
        return to_observation(result, (width, height), self._map_label)

    def close(self) -> None:
        self._recognizer.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
