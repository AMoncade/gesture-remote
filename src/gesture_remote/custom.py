"""Your own gestures (phase 2): a small scikit-learn classifier on top of MediaPipe's landmarks.

tools/record.py saves raw landmarks per gesture, tools/train.py fits the classifier and saves a
`CustomModel`; the app then wraps the canned recognizer in a `CustomGestureRecognizer`, which
relabels a hand when the classifier is confident enough. The canned gestures keep working.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gesture_remote.features import FEATURES_VERSION, normalize_landmarks
from gesture_remote.observation import (
    NONE_LABEL,
    NUM_LANDMARKS,
    FrameObservation,
    Handedness,
    HandObservation,
)
from gesture_remote.recognition import Recognizer

logger = logging.getLogger(__name__)

# --- recordings: raw landmarks, one CSV per recording (data/<label>/<time>.csv) ---------------

AXES = ("x", "y", "z")
CSV_HEADER = (
    "t",
    "label",
    "handedness",
    "handedness_score",
    "width",
    "height",
    *(f"{axis}{index}" for index in range(NUM_LANDMARKS) for axis in AXES),
    *(f"w{axis}{index}" for index in range(NUM_LANDMARKS) for axis in AXES),
)
"""Raw data, never features: a later feature version can be computed again from these."""


def hand_row(t: float, label: str, hand: HandObservation, image_size: tuple[int, int]) -> list:
    width, height = image_size
    return [
        f"{t:.3f}",
        label,
        hand.handedness.value,
        f"{hand.handedness_score:.3f}",
        width,
        height,
        *(f"{value:.5f}" for value in np.asarray(hand.landmarks).reshape(-1)),
        *(f"{value:.5f}" for value in np.asarray(hand.world_landmarks).reshape(-1)),
    ]


def read_recordings(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feature matrix, labels and recording index (one group per file) from CSV recordings."""
    import csv

    features, labels, groups = [], [], []
    for group, path in enumerate(paths):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                points = np.array(
                    [[float(row[f"{axis}{i}"]) for axis in AXES] for i in range(NUM_LANDMARKS)]
                )
                size = (int(row["width"]), int(row["height"]))
                try:
                    vector = normalize_landmarks(points, size, Handedness(row["handedness"]))
                except ValueError:
                    continue  # degenerate hand: skipped, as at run time
                features.append(vector)
                labels.append(row["label"])
                groups.append(group)
    return np.array(features), np.array(labels), np.array(groups)


class StaleModelError(ValueError):
    """The saved model was trained on another feature version: run tools/train.py again."""


@dataclass(frozen=True)
class CustomModel:
    classifier: Any
    """A fitted scikit-learn classifier with predict_proba and classes_."""
    features_version: int = FEATURES_VERSION
    samples: int = 0

    @property
    def labels(self) -> frozenset[str]:
        return frozenset(str(label) for label in self.classifier.classes_)

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> CustomModel:
        """Load a model written by tools/train.py (a pickle: only load your own files)."""
        import joblib

        model = joblib.load(path)
        if not isinstance(model, CustomModel):
            raise StaleModelError(f"{path} is not a gesture-remote custom model")
        if model.features_version != FEATURES_VERSION:
            raise StaleModelError(
                f"{path} was trained with features v{model.features_version}, this code computes "
                f"v{FEATURES_VERSION}: run tools/train.py again"
            )
        return model

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """Best label and its probability for one feature vector."""
        probabilities = self.classifier.predict_proba(features.reshape(1, -1))[0]
        best = int(np.argmax(probabilities))
        return str(self.classifier.classes_[best]), float(probabilities[best])


class CustomGestureRecognizer:
    """Wraps a recognizer; a hand showing no built-in gesture becomes a custom gesture when the
    model is confident enough."""

    def __init__(self, inner: Recognizer, model: CustomModel, min_score: float) -> None:
        self._inner = inner
        self._model = model
        self._min_score = min_score
        self.labels = inner.labels | model.labels
        logger.info(
            "custom gestures ready: %s (min score %.2f, %d training samples)",
            ", ".join(sorted(model.labels - {NONE_LABEL})),
            min_score,
            model.samples,
        )

    def recognize(self, rgb: np.ndarray, timestamp_ms: int) -> FrameObservation:
        observation = self._inner.recognize(rgb, timestamp_ms)
        if not observation.hands:
            return observation
        hands = tuple(self._relabel(hand, observation.image_size) for hand in observation.hands)
        return dataclasses.replace(observation, hands=hands)

    def _relabel(self, hand: HandObservation, image_size: tuple[int, int]) -> HandObservation:
        # Only hands the canned model calls `none`: a classifier trained on a few gestures of
        # yours would otherwise map a real thumb up onto the nearest one of them.
        if hand.label != NONE_LABEL:
            return hand
        try:
            features = normalize_landmarks(hand.landmarks, image_size, hand.handedness)
        except ValueError:
            return hand
        label, score = self._model.predict(features)
        if label == NONE_LABEL or score < self._min_score:
            return hand
        return dataclasses.replace(hand, label=label, score=score)

    def close(self) -> None:
        self._inner.close()
