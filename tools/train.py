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

from sklearn.ensemble import RandomForestClassifier

from gesture_remote.custom import CustomModel, read_recordings
from gesture_remote.observation import NONE_LABEL

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MODEL_PATH = REPO_ROOT / "models" / "custom_gestures.joblib"


def make_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=0)


def held_out_report(features, labels, groups) -> dict[str, tuple[float, str]]:
    """Per gesture: accuracy when each of its recordings is predicted by a model trained
    without that recording, and what the misses were mostly taken for.

    A gesture with a single recording is left out: held out, the model would never have seen
    it, so its score would be 0 % by construction, not a measurement.
    """
    report: dict[str, tuple[float, str]] = {}
    for label in sorted(set(labels)):
        own = sorted(set(groups[labels == label]))
        if len(own) < 2:
            continue
        hits, total, misses = 0, 0, Counter()
        for group in own:
            train, test = groups != group, groups == group
            predicted = (
                make_classifier().fit(features[train], labels[train]).predict(features[test])
            )
            hits += int((predicted == label).sum())
            total += len(predicted)
            misses.update(str(p) for p in predicted if p != label)
        taken_for = ", ".join(f"{name} {count}" for name, count in misses.most_common(2))
        report[str(label)] = (hits / total, taken_for)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report only, save nothing")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=MODEL_PATH)
    options = parser.parse_args()

    # data/_something/ holds recordings put aside (e.g. a wrong gesture): never used
    paths = sorted(p for p in options.data.glob("*/*.csv") if not p.parent.name.startswith("_"))
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

    report = held_out_report(features, labels, groups)
    print("accuracy on held-out recordings:")
    for label in sorted(set(str(label) for label in labels)):
        if label not in report:
            print(f"  {label:16} not measured: record it once more")
            continue
        score, taken_for = report[label]
        print(
            f"  {label:16} {score:.0%}" + (f"   misses taken for: {taken_for}" if taken_for else "")
        )

    if options.check:
        return 0
    model = CustomModel(make_classifier().fit(features, labels), samples=len(labels))
    model.save(options.out)
    print(f"saved {options.out}")
    print("restart gesture-remote, then bind the new names in config.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
