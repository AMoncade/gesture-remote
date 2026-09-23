"""Webcam capture and idle throttling.

Frames stay in memory: nothing here writes an image anywhere, and nothing logs pixels.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from types import TracebackType
from typing import Protocol, Self

import cv2
import numpy as np

from gesture_remote.config import CameraSettings

logger = logging.getLogger(__name__)

Clock = Callable[[], float]

BACKENDS: dict[str, tuple[str, ...]] = {
    "auto": ("CAP_DSHOW", "CAP_MSMF"),
    "dshow": ("CAP_DSHOW",),
    "msmf": ("CAP_MSMF",),
}
"""Backend setting -> cv2 constants tried in order; the first that opens wins."""


class VideoCaptureLike(Protocol):
    """The part of cv2.VideoCapture that Camera uses (tests inject a fake)."""

    def isOpened(self) -> bool: ...

    def set(self, prop_id: int, value: float) -> bool: ...

    def get(self, prop_id: int) -> float: ...

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...

    def getBackendName(self) -> str: ...


CaptureFactory = Callable[[int, int], VideoCaptureLike]


class Camera:
    """One webcam, opened with the configured backend, frames mirrored before inference.

    `read()` never raises for camera trouble: it returns None, and after `max_failed_reads`
    consecutive failures (camera taken by Teams, unplugged) it releases the device and retries
    opening at most every `reopen_interval_s`. The caller must not spin when it gets None.
    """

    def __init__(
        self,
        settings: CameraSettings,
        *,
        capture_factory: CaptureFactory = cv2.VideoCapture,
        clock: Clock = time.monotonic,
        reopen_interval_s: float = 2.0,
        max_failed_reads: int = 30,
    ) -> None:
        self._settings = settings
        self._factory = capture_factory
        self._clock = clock
        self._reopen_interval_s = reopen_interval_s
        self._max_failed_reads = max_failed_reads
        self._capture: VideoCaptureLike | None = None
        self._failed_reads = 0
        self._last_open_attempt = -math.inf
        self.backend_name: str | None = None
        """Name of the backend in use (e.g. DSHOW), None while closed."""

    @property
    def is_open(self) -> bool:
        return self._capture is not None

    def open(self) -> bool:
        """Try each backend of the setting in order; True if one opened."""
        self.close()
        self._last_open_attempt = self._clock()
        index = self._settings.index
        for constant in BACKENDS[self._settings.backend]:
            backend = getattr(cv2, constant, None)
            if backend is None:
                logger.warning("camera backend %s is missing from this OpenCV build", constant)
                continue
            capture = self._factory(index, backend)
            if not capture.isOpened():
                capture.release()
                logger.warning("camera %d did not open with %s", index, constant)
                continue
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.height)
            try:
                name = capture.getBackendName()
            except cv2.error:
                name = constant
            self._capture, self.backend_name, self._failed_reads = capture, name, 0
            logger.info(
                "camera %d opened with backend %s (%s), %dx%d asked, %dx%d reported, mirror=%s",
                index,
                constant,
                name,
                self._settings.width,
                self._settings.height,
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                self._settings.mirror,
            )
            return True
        logger.error(
            "camera %d could not be opened (backend setting %r)", index, self._settings.backend
        )
        return False

    def read(self) -> np.ndarray | None:
        """Next BGR frame, mirrored if configured; None if no frame is available right now."""
        if self._capture is None:
            if self._clock() - self._last_open_attempt < self._reopen_interval_s:
                return None
            if not self.open():
                return None
        assert self._capture is not None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            self._failed_reads += 1
            if self._failed_reads >= self._max_failed_reads:
                logger.warning(
                    "camera returned no frame %d times in a row: releasing it, will reopen",
                    self._failed_reads,
                )
                self.close()
                self._last_open_attempt = self._clock()
            return None
        self._failed_reads = 0
        # Mirror here, before inference: MediaPipe's handedness assumes a selfie image.
        # Sample images bypass Camera, so they reach the recognizer unflipped.
        return cv2.flip(frame, 1) if self._settings.mirror else frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            logger.info("camera released")
        self._capture = None
        self.backend_name = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class IdleThrottle:
    """Decide whether *this* frame goes to inference.

    While a hand was seen within the last `after_s` seconds every frame passes; otherwise frames
    pass at about `fps` per second. The camera is still read at full rate by the caller so its
    buffer never ages: only inference is skipped.
    """

    def __init__(self, fps: float, after_s: float, clock: Clock = time.monotonic) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if after_s < 0:
            raise ValueError(f"after_s must not be negative, got {after_s}")
        self._period = 1.0 / fps
        self._after_s = after_s
        self._clock = clock
        self._last_hand = -math.inf
        self._next_due = -math.inf

    @property
    def active(self) -> bool:
        """True while a hand was seen within the last `after_s` seconds."""
        return self._clock() - self._last_hand <= self._after_s

    def should_infer(self) -> bool:
        now = self._clock()
        if now - self._last_hand <= self._after_s:
            return True
        if now < self._next_due:
            return False
        # Schedule from the previous due time, not from now, so the average rate is `fps` even
        # when camera frames do not land on the period; resync after a long gap.
        self._next_due += self._period
        if self._next_due <= now:
            self._next_due = now + self._period
        return True

    def report(self, hand_seen: bool) -> None:
        """Tell the throttle what the inference of the current frame found."""
        if hand_seen:
            self._last_hand = self._clock()
