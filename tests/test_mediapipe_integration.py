"""Real MediaPipe on the 4 official sample images, in VIDEO mode (the mode the app uses).

Skipped only when the model or the images are missing (run tools/download_models.py). The
samples are fed unmirrored, as the app never mirrors anything but camera frames.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from gesture_remote.config import RecognitionSettings
from gesture_remote.features import normalize_landmarks
from gesture_remote.observation import NUM_LANDMARKS, FrameObservation, Handedness
from gesture_remote.recognition import CANNED_LABEL_SET, MediaPipeGestureRecognizer

MODELS = Path(__file__).resolve().parents[1] / "models"
MODEL = MODELS / "gesture_recognizer.task"
SAMPLES = {
    "thumbs_up.jpg": ("thumb_up", Handedness.RIGHT),
    "thumbs_down.jpg": ("thumb_down", Handedness.RIGHT),
    "victory.jpg": ("victory", Handedness.RIGHT),
    "pointing_up.jpg": ("pointing_up", Handedness.LEFT),
}
"""Expected label and raw handedness (unmirrored images; IMAGE mode gave the same, 2026-09-23)."""
FRAMES_PER_SAMPLE = 3

missing = [
    path for path in (MODEL, *(MODELS / "samples" / name for name in SAMPLES)) if not path.exists()
]
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(bool(missing), reason=f"run tools/download_models.py; missing: {missing}"),
]


def load_rgb(name: str) -> np.ndarray:
    bgr = cv2.imread(str(MODELS / "samples" / name))
    assert bgr is not None, name
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def recognize_still(rgb: np.ndarray, frames: int = FRAMES_PER_SAMPLE) -> list[FrameObservation]:
    """A fresh recognizer, the same image `frames` times at 30 fps, like a camera on a still hand.

    Fresh per image on purpose: in VIDEO mode MediaPipe tracks the hand from the previous frame,
    and a hard cut to another photo loses it for one frame (measured: 'no hand', then the label).
    """
    with MediaPipeGestureRecognizer(MODEL.read_bytes(), RecognitionSettings()) as recognizer:
        return [recognizer.recognize(rgb, 1000 + 33 * i) for i in range(frames)]


@pytest.mark.parametrize("name", list(SAMPLES))
def test_sample_image_gives_its_label_in_video_mode(name: str) -> None:
    label, handedness = SAMPLES[name]
    rgb = load_rgb(name)
    for observation in recognize_still(rgb):
        assert isinstance(observation.hands, tuple)
        assert observation.image_size == (rgb.shape[1], rgb.shape[0])
        hand = observation.primary()
        assert hand is not None, f"{name}: no hand"
        assert (hand.label, hand.handedness) == (label, handedness)
        assert hand.landmarks.shape == hand.world_landmarks.shape == (NUM_LANDMARKS, 3)
        features = normalize_landmarks(hand.landmarks, observation.image_size, hand.handedness)
        assert np.linalg.norm(features[27:30]) == pytest.approx(1.0, abs=1e-5)  # |p9| = 1


def test_mirrored_view_is_read_as_mirrored() -> None:
    """mp.Image ignores numpy strides: without ascontiguousarray a flipped *view* is read as the
    original image, so handedness would not flip. The recognizer must copy it to contiguous
    memory; this fails if it does not."""
    rgb = load_rgb("victory.jpg")
    view = rgb[:, ::-1]
    assert not view.flags["C_CONTIGUOUS"]
    from_view = recognize_still(view, frames=1)[0].primary()
    from_copy = recognize_still(cv2.flip(rgb, 1), frames=1)[0].primary()
    assert from_view is not None and from_copy is not None
    assert from_view.handedness is from_copy.handedness is Handedness.LEFT
    np.testing.assert_allclose(from_view.landmarks, from_copy.landmarks, atol=1e-6)


def test_blank_frame_has_no_hand_and_labels_are_the_canned_set() -> None:
    with MediaPipeGestureRecognizer(MODEL.read_bytes(), RecognitionSettings()) as recognizer:
        assert recognizer.labels == CANNED_LABEL_SET
        observation = recognizer.recognize(np.zeros((480, 640, 3), np.uint8), 0)
        # a stalled clock must not make VIDEO mode raise
        again = recognizer.recognize(np.zeros((480, 640, 3), np.uint8), 0)
    assert observation.hands == () and again.hands == ()
    assert observation.image_size == (640, 480)
