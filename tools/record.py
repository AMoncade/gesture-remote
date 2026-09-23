"""Record one of your own gestures: the hand's 21 landmarks go to data/<label>/<time>.csv.

    record.py rock                  3 s to get ready, then 10 s of recording
    record.py none                  record "no gesture" (relaxed hand, typing...): recommended
    record.py peace2 --seconds 15

While recording, move your hand a little (angle, distance, left/right of the frame): varied
examples make a better model. Only landmark numbers are saved, never an image. If the camera
does not open, quit the tray app first. Then run tools/train.py.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import re
import sys
import time
from pathlib import Path

import cv2

from gesture_remote.capture import Camera
from gesture_remote.config import CameraSettings, RecognitionSettings
from gesture_remote.custom import CSV_HEADER, hand_row
from gesture_remote.debug_view import draw_overlay
from gesture_remote.recognition import MediaPipeGestureRecognizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MODEL = REPO_ROOT / "models" / "gesture_recognizer.task"
LABEL = re.compile(r"^[a-z][a-z0-9_]*$")
TITLE = "gesture-remote: record"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("label", help="snake_case name, e.g. rock, call_me, none")
    parser.add_argument("--seconds", type=float, default=10.0, help="recording time (10)")
    parser.add_argument("--countdown", type=float, default=3.0, help="time to get ready (3)")
    options = parser.parse_args()
    if not LABEL.match(options.label):
        print(f"{options.label!r}: use lowercase letters, digits and _ (e.g. call_me)")
        return 2

    camera = Camera(CameraSettings())
    if not camera.open():
        print("the camera did not open: quit gesture-remote (tray icon) and try again")
        return 1
    recognizer = MediaPipeGestureRecognizer(MODEL.read_bytes(), RecognitionSettings())
    target = DATA_DIR / options.label / f"{time.strftime('%Y%m%d-%H%M%S')}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)

    samples, aborted = 0, False
    started = time.monotonic()
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        while True:
            frame = camera.read()
            if frame is None:
                continue
            now = time.monotonic()
            elapsed = now - started
            if elapsed >= options.countdown + options.seconds:
                break
            observation = recognizer.recognize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(now * 1000)
            )
            recording = elapsed >= options.countdown
            hand = observation.primary()
            if recording and hand is not None:
                writer.writerow(hand_row(elapsed, options.label, hand, observation.image_size))
                samples += 1
            if recording:
                left = options.countdown + options.seconds - elapsed
                lines = [
                    f"REC {options.label}  {left:.1f} s  {samples} samples",
                    "move your hand a little: angle, distance, position",
                ]
            else:
                lines = [f"get ready: {options.label} in {options.countdown - elapsed:.1f} s"]
            if hand is None:
                lines.append("no hand seen")
            cv2.imshow(TITLE, draw_overlay(frame, observation, lines))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                aborted = True
                break

    cv2.destroyAllWindows()
    camera.close()
    recognizer.close()
    if aborted or samples == 0:
        target.unlink()
        with contextlib.suppress(OSError):  # only succeeds when the label folder is now empty
            target.parent.rmdir()
        print("aborted: nothing saved" if aborted else "no hand seen: nothing saved")
        return 1
    print(f"saved {samples} samples of {options.label!r} to {target.relative_to(REPO_ROOT)}")
    print("record more gestures, then run: .venv\\Scripts\\python tools\\train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
