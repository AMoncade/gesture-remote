"""Hand landmarks -> fixed-length feature vector for the phase-2 custom classifier.

Deliberately *not* rotation invariant: 👍 and 👎 differ only by a rotation. Never log the input or
the output of `normalize_landmarks`: both are landmark data.
"""

from __future__ import annotations

import numpy as np

from gesture_remote.observation import NUM_LANDMARKS, Handedness

FEATURES_VERSION = 1
"""Bump whenever `normalize_landmarks` changes: a trained model embeds it and refuses a mismatch."""

FEATURE_SIZE = NUM_LANDMARKS * 3

_SCALE_LANDMARK = 9
"""Middle-finger MCP: the wrist-to-p9 distance is the hand's unit length."""
_MIN_SCALE = 1e-6


def normalize_landmarks(
    landmarks: np.ndarray, image_size: tuple[int, int], handedness: Handedness
) -> np.ndarray:
    """Return a `float32[63]` vector invariant to translation, scale and left/right hand.

    `landmarks` is (21, 3) in MediaPipe image coordinates: x and y normalised to the image width
    and height, z on the same scale as x. `image_size` is (width, height) in pixels.
    Steps: aspect fix (x and z times w/h, so all axes share the height's unit), wrist origin,
    division by |p9| in 3D, x mirrored for left hands.
    Raises ValueError on a wrong shape, non-finite values or a degenerate scale.
    """
    points = np.array(landmarks, dtype=np.float64)
    if points.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected landmarks of shape ({NUM_LANDMARKS}, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("landmarks contain non-finite values")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {image_size}")

    aspect = width / height
    points[:, 0] *= aspect
    points[:, 2] *= aspect
    points -= points[0]
    scale = float(np.linalg.norm(points[_SCALE_LANDMARK]))
    if not scale > _MIN_SCALE:
        raise ValueError(f"degenerate hand scale (|p{_SCALE_LANDMARK}| = {scale:.3g})")
    points /= scale
    if handedness is Handedness.LEFT:
        points[:, 0] = -points[:, 0]
    return points.reshape(FEATURE_SIZE).astype(np.float32)
