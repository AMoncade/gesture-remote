"""Train the custom gesture model from data/<label>/*.csv (made by tools/record.py).

    train.py            fit, report accuracy, save models/custom_gestures.joblib
    train.py --check    report accuracy only, save nothing

Accuracy is measured on whole recordings held out (a model tested on frames of a recording it
was trained on would look far better than it is), so record each gesture at least twice for a
meaningful score. Restart gesture-remote afterwards: the model is loaded at startup.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut

from gesture_remote.custom import CustomModel, read_recordings
from gesture_remote.observation import NONE_LABEL

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MODEL_PATH = REPO_ROOT / "models" / "custom_gestures.joblib"


def make_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=0)


def held_out_accuracy(features, labels, groups) -> tuple[float, dict[str, float]] | None:
    """Accuracy when each recording is predicted by a model trained without it."""
    if len(set(groups)) < 2:
        return None
    predicted = np.empty_like(labels)
    for train, test in LeaveOneGroupOut().split(features, labels, groups):
        if len(set(labels[train])) < 2:
            return None
        predicted[test] = (
            make_classifier().fit(features[train], labels[train]).predict(features[test])
        )
    per_label = {
        label: accuracy_score(labels[labels == label], predicted[labels == label])
        for label in sorted(set(labels))
    }
    return accuracy_score(labels, predicted), per_label


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report only, save nothing")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=MODEL_PATH)
    options = parser.parse_args()

    paths = sorted(options.data.glob("*/*.csv"))
    features, labels, groups = read_recordings(paths)
    if len(set(labels)) < 2:
        found = ", ".join(sorted(str(label) for label in set(labels))) or "none"
        print(f"need at least 2 gestures in {options.data} (found: {found})")
        print("record some: .venv\\Scripts\\python tools\\record.py <name>")
        return 1

    recordings = Counter(Path(paths[g]).parent.name for g in set(groups))
    for label, count in sorted(Counter(labels).items()):
        print(f"{label:16} {count:5} samples  {recordings[label]} recording(s)")
    if NONE_LABEL not in labels:
        print(f"tip: record '{NONE_LABEL}' too (relaxed hand, typing...): fewer false triggers")

    scores = held_out_accuracy(features, labels, groups)
    if scores is None:
        print("accuracy: not measured (record each gesture at least twice)")
    else:
        total, per_label = scores
        print(f"accuracy on held-out recordings: {total:.0%}")
        for label, score in per_label.items():
            print(f"  {label:16} {score:.0%}")

    if options.check:
        return 0
    model = CustomModel(make_classifier().fit(features, labels), samples=len(labels))
    model.save(options.out)
    print(f"saved {options.out}")
    print("restart gesture-remote, then bind the new names in config.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
