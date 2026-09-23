"""normalize_landmarks: invariances, mirror, aspect fix, failure modes."""

from __future__ import annotations

import numpy as np
import pytest

from gesture_remote.features import FEATURE_SIZE, FEATURES_VERSION, normalize_landmarks
from gesture_remote.observation import NUM_LANDMARKS, Handedness

WIDTH, HEIGHT = 640, 480


def pixel_hand(seed: int = 7) -> np.ndarray:
    """A random but fixed 21-point hand in pixel units (x, y, z), wrist near the image centre."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(-80.0, 80.0, size=(NUM_LANDMARKS, 3))
    points[0] = 0.0
    points[:, :2] += (WIDTH / 2, HEIGHT / 2)
    return points


def to_image_coords(pixels: np.ndarray, size: tuple[int, int] = (WIDTH, HEIGHT)) -> np.ndarray:
    """MediaPipe convention: x / width, y / height, z on the same scale as x (so / width)."""
    width, height = size
    return pixels / np.array([width, height, width])


def pairwise_distances(vector: np.ndarray) -> np.ndarray:
    points = vector.reshape(NUM_LANDMARKS, 3).astype(np.float64)
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)


def test_shape_dtype_and_version() -> None:
    features = normalize_landmarks(to_image_coords(pixel_hand()), (WIDTH, HEIGHT), Handedness.RIGHT)
    assert features.shape == (FEATURE_SIZE,) == (63,)
    assert features.dtype == np.float32
    assert isinstance(FEATURES_VERSION, int)


def test_wrist_is_origin_and_p9_has_unit_length() -> None:
    features = normalize_landmarks(to_image_coords(pixel_hand()), (WIDTH, HEIGHT), Handedness.RIGHT)
    points = features.reshape(NUM_LANDMARKS, 3)
    np.testing.assert_allclose(points[0], 0.0, atol=1e-7)
    assert np.linalg.norm(points[9]) == pytest.approx(1.0, abs=1e-6)


def test_invariant_to_translation_and_scale() -> None:
    hand = pixel_hand()
    # z is relative to the wrist (wrist z = 0): it scales with the hand but never shifts.
    moved = (hand - hand[0]) * 1.7 + hand[0] + (40.0, -25.0, 0.0)
    a = normalize_landmarks(to_image_coords(hand), (WIDTH, HEIGHT), Handedness.RIGHT)
    b = normalize_landmarks(to_image_coords(moved), (WIDTH, HEIGHT), Handedness.RIGHT)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_left_hand_equals_mirrored_right_hand() -> None:
    right = to_image_coords(pixel_hand())
    left = right.copy()
    left[:, 0] = 1.0 - left[:, 0]  # the same hand seen in a mirror
    a = normalize_landmarks(right, (WIDTH, HEIGHT), Handedness.RIGHT)
    b = normalize_landmarks(left, (WIDTH, HEIGHT), Handedness.LEFT)
    np.testing.assert_allclose(a, b, atol=1e-6)
    # Without the handedness mirror the two would differ: the test above can fail.
    c = normalize_landmarks(left, (WIDTH, HEIGHT), Handedness.RIGHT)
    assert not np.allclose(a, c, atol=1e-3)


def test_not_rotation_invariant_thumb_up_differs_from_thumb_down() -> None:
    hand = pixel_hand()
    centre = hand[0, :2]
    upside_down = hand.copy()
    upside_down[:, :2] = centre - (hand[:, :2] - centre)  # 180° in the image plane
    a = normalize_landmarks(to_image_coords(hand), (WIDTH, HEIGHT), Handedness.RIGHT)
    b = normalize_landmarks(to_image_coords(upside_down), (WIDTH, HEIGHT), Handedness.RIGHT)
    assert not np.allclose(a, b, atol=1e-3)


def test_quarter_turn_in_the_image_plane_keeps_distances_in_4_3() -> None:
    """Fails without the x * w/h correction: a 4:3 image squeezes x relative to y."""
    hand = pixel_hand()
    centre = hand[0, :2]
    turned = hand.copy()
    dx, dy = (hand[:, :2] - centre).T
    turned[:, 0], turned[:, 1] = centre[0] - dy, centre[1] + dx
    a = normalize_landmarks(to_image_coords(hand), (WIDTH, HEIGHT), Handedness.RIGHT)
    b = normalize_landmarks(to_image_coords(turned), (WIDTH, HEIGHT), Handedness.RIGHT)
    np.testing.assert_allclose(pairwise_distances(a), pairwise_distances(b), atol=1e-5)


def test_quarter_turn_in_depth_keeps_distances_in_4_3() -> None:
    """Fails without the z * w/h correction: z shares x's scale, so it needs the same fix."""
    hand = pixel_hand()
    turned = hand.copy()
    dx = hand[:, 0] - hand[0, 0]
    dz = hand[:, 2] - hand[0, 2]
    turned[:, 0], turned[:, 2] = hand[0, 0] - dz, dx
    a = normalize_landmarks(to_image_coords(hand), (WIDTH, HEIGHT), Handedness.RIGHT)
    b = normalize_landmarks(to_image_coords(turned), (WIDTH, HEIGHT), Handedness.RIGHT)
    np.testing.assert_allclose(pairwise_distances(a), pairwise_distances(b), atol=1e-5)


def test_degenerate_scale_raises() -> None:
    collapsed = np.full((NUM_LANDMARKS, 3), 0.5)
    with pytest.raises(ValueError, match="degenerate"):
        normalize_landmarks(collapsed, (WIDTH, HEIGHT), Handedness.RIGHT)


@pytest.mark.parametrize(
    "landmarks",
    [np.zeros((20, 3)), np.zeros((21, 2)), np.full((21, 3), np.nan)],
    ids=["20 points", "2D", "nan"],
)
def test_malformed_input_raises(landmarks: np.ndarray) -> None:
    with pytest.raises(ValueError):
        normalize_landmarks(landmarks, (WIDTH, HEIGHT), Handedness.RIGHT)


def test_input_is_not_modified() -> None:
    landmarks = to_image_coords(pixel_hand())
    before = landmarks.copy()
    normalize_landmarks(landmarks, (WIDTH, HEIGHT), Handedness.LEFT)
    np.testing.assert_array_equal(landmarks, before)
