"""Immutable observation types passed from recognition to the engine.

Round-1 contract: frozen during the parallel round (owner: admin). Landmark arrays are biometric-ish
data: never log them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

NONE_LABEL = "none"
"""Label for 'a hand is visible but no known gesture'. Never bindable."""

NUM_LANDMARKS = 21


class Handedness(Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True, slots=True, eq=False)
class HandObservation:
    """One detected hand in one processed frame."""

    label: str
    """snake_case gesture label; NONE_LABEL when no known gesture was recognised."""
    score: float
    """Confidence of `label`, 0..1."""
    handedness: Handedness
    handedness_score: float
    landmarks: np.ndarray
    """(21, 3): x, y normalised to the image (0..1), z relative to the wrist, same scale as x."""
    world_landmarks: np.ndarray
    """(21, 3): metres, origin at the hand's approximate geometric centre."""


@dataclass(frozen=True, slots=True, eq=False)
class FrameObservation:
    """Everything recognition produced for one processed frame."""

    hands: tuple[HandObservation, ...]
    image_size: tuple[int, int]
    """(width, height) in pixels of the image the landmarks refer to."""

    def primary(self) -> HandObservation | None:
        """The hand with the highest gesture score, or None when no hand was detected."""
        return max(self.hands, key=lambda hand: hand.score, default=None)
