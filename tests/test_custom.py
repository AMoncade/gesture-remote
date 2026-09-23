"""Custom gestures: recordings -> features -> model -> relabelled observations."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from gesture_remote.custom import (
    CSV_HEADER,
    CustomGestureRecognizer,
    CustomModel,
    StaleModelError,
    hand_row,
    read_recordings,
)
from gesture_remote.features import FEATURES_VERSION, normalize_landmarks
from gesture_remote.observation import (
    NONE_LABEL,
    NUM_LANDMARKS,
    FrameObservation,
    Handedness,
    HandObservation,
)

SIZE = (640, 480)


def a_hand(label: str = NONE_LABEL, spread: float = 1.0, seed: int = 0) -> HandObservation:
    rng = np.random.default_rng(seed)
    base = np.linspace(0.3, 0.7, NUM_LANDMARKS * 3).reshape(NUM_LANDMARKS, 3)
    landmarks = (base * spread + rng.normal(0, 0.003, base.shape)).astype(np.float32)
    landmarks[0] = (0.5, 0.8, 0.0)
    return HandObservation(
        label=label,
        score=0.9,
        handedness=Handedness.RIGHT,
        handedness_score=0.97,
        landmarks=landmarks,
        world_landmarks=np.zeros((NUM_LANDMARKS, 3), np.float32),
    )


class Classifier:
    """predict_proba returns fixed probabilities; enough for the recognizer's logic."""

    def __init__(self, classes: list[str], probabilities: list[float]) -> None:
        self.classes_ = np.array(classes)
        self.probabilities = np.array([probabilities])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        assert features.shape == (1, NUM_LANDMARKS * 3)
        return self.probabilities


class Inner:
    labels = frozenset({NONE_LABEL, "thumb_up"})

    def __init__(self, *hands: HandObservation) -> None:
        self.hands = hands
        self.closed = False

    def recognize(self, rgb: np.ndarray, timestamp_ms: int) -> FrameObservation:
        return FrameObservation(hands=self.hands, image_size=SIZE)

    def close(self) -> None:
        self.closed = True


def recognize(inner: Inner, probabilities: list[float], min_score: float = 0.8):
    model = CustomModel(Classifier([NONE_LABEL, "rock"], probabilities))
    recognizer = CustomGestureRecognizer(inner, model, min_score)
    return recognizer, recognizer.recognize(np.zeros((480, 640, 3), np.uint8), 1)


def test_a_confident_custom_gesture_relabels_a_none_hand() -> None:
    recognizer, observation = recognize(Inner(a_hand()), [0.1, 0.9])
    assert recognizer.labels == {NONE_LABEL, "thumb_up", "rock"}
    assert (observation.hands[0].label, observation.hands[0].score) == ("rock", 0.9)


@pytest.mark.parametrize(
    ("hand", "probabilities"),
    [
        (a_hand(), [0.3, 0.7]),  # below min_score
        (a_hand(), [0.9, 0.1]),  # the model says none
        (a_hand("thumb_up"), [0.0, 1.0]),  # a built-in gesture is never overridden
    ],
    ids=["not-confident", "model-says-none", "built-in-gesture-kept"],
)
def test_the_hand_is_left_alone(hand: HandObservation, probabilities: list[float]) -> None:
    _, observation = recognize(Inner(hand), probabilities)
    assert observation.hands[0] is hand


def test_model_round_trip_and_stale_version(tmp_path: Path) -> None:
    from sklearn.ensemble import RandomForestClassifier

    features = np.vstack([np.zeros((5, 63)), np.ones((5, 63))])
    labels = np.array(["none"] * 5 + ["rock"] * 5)
    path = tmp_path / "m.joblib"
    CustomModel(RandomForestClassifier(n_estimators=5).fit(features, labels), samples=10).save(path)
    loaded = CustomModel.load(path)
    assert loaded.labels == {"none", "rock"} and loaded.samples == 10
    assert loaded.predict(np.ones(63))[0] == "rock"

    CustomModel(loaded.classifier, features_version=FEATURES_VERSION + 1).save(path)
    with pytest.raises(StaleModelError, match="train.py again"):
        CustomModel.load(path)


def test_recordings_round_trip_to_the_run_time_features(tmp_path: Path) -> None:
    hands = {"rock": a_hand(spread=1.0, seed=1), "call_me": a_hand(spread=0.6, seed=2)}
    paths = []
    for label, hand in hands.items():
        path = tmp_path / label / "one.csv"
        path.parent.mkdir()
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_HEADER)
            writer.writerow(hand_row(0.5, label, hand, SIZE))
        paths.append(path)

    features, labels, groups = read_recordings(paths)
    assert labels.tolist() == ["rock", "call_me"] and groups.tolist() == [0, 1]
    expected = normalize_landmarks(hands["rock"].landmarks, SIZE, Handedness.RIGHT)
    np.testing.assert_allclose(features[0], expected, atol=1e-3)  # CSV keeps 5 decimals
