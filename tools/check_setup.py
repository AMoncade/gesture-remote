"""Check that this machine can run gesture-remote: models, camera, Start menu apps, keys.

Standalone on purpose: it does not import gesture_remote, so it still runs when the package is
broken. Nothing it reads from the camera ever leaves memory.

    check_setup.py                  models + sample images + class names, then the camera
    check_setup.py --no-camera      models only
    check_setup.py --apps TEXT      Start menu entries whose name contains TEXT (name + AppID)
    check_setup.py --press KEY      send one real key press through pyautogui (e.g. playpause)
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
GESTURE_MODEL = MODELS_DIR / "gesture_recognizer.task"
SAMPLES = {
    "thumbs_up.jpg": "Thumb_Up",
    "thumbs_down.jpg": "Thumb_Down",
    "victory.jpg": "Victory",
    "pointing_up.jpg": "Pointing_Up",
}
# Label files embedded in the bundle: a .task is a zip, and a .tflite with metadata carries its
# associated files as a zip appended at the end, which zipfile reads from the tail.
LABEL_FILES = {
    "gesture classes": (
        "hand_gesture_recognizer.task",
        "canned_gesture_classifier.tflite",
        "labels.txt",
    ),
    "handedness": ("hand_landmarker.task", "hand_landmarks_detector.tflite", "handedness.txt"),
}
CAMERA_BACKENDS = ("CAP_DSHOW", "CAP_MSMF")
CAMERA_SIZE = (640, 480)
CAMERA_FRAMES = 60
START_APPS_COMMAND = (
    "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-StartApps | ConvertTo-Json -Compress"
)


def read_nested(archive: Path, members: tuple[str, ...]) -> bytes:
    data = archive.read_bytes()
    for member in members:
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            if member not in bundle.namelist():
                raise KeyError(f"{member} not found; members: {bundle.namelist()}")
            data = bundle.read(member)
    return data


def check_models() -> bool:
    print("== models ==")
    if not GESTURE_MODEL.exists():
        print(f"FAIL  {GESTURE_MODEL} is missing: run tools/download_models.py")
        return False

    ok = True
    for title, members in LABEL_FILES.items():
        try:
            text = read_nested(GESTURE_MODEL, members).decode("utf-8")
        except (KeyError, zipfile.BadZipFile) as error:
            print(f"FAIL  {title}: {error}")
            ok = False
            continue
        names = [line.strip() for line in text.splitlines() if line.strip()]
        print(f"OK    {title} ({'/'.join(members)}): {names}")

    import mediapipe as mp
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions

    print(f"OK    mediapipe {mp.__version__}")
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_buffer=GESTURE_MODEL.read_bytes())
    )
    with GestureRecognizer.create_from_options(options) as recognizer:
        for filename, expected in SAMPLES.items():
            path = MODELS_DIR / "samples" / filename
            if not path.exists():
                print(f"FAIL  {filename} is missing: run tools/download_models.py")
                ok = False
                continue
            result = recognizer.recognize(mp.Image.create_from_file(str(path)))
            if not result.gestures:
                print(f"FAIL  {filename}: no hand detected")
                ok = False
                continue
            top = result.gestures[0][0]
            hand = result.handedness[0][0].category_name
            points = len(result.hand_landmarks[0])
            status = "OK  " if top.category_name == expected else "FAIL"
            ok = ok and top.category_name == expected
            print(
                f"{status}  {filename}: raw class {top.category_name!r} {top.score:.2f} "
                f"(expected {expected!r}), handedness {hand!r}, {points} landmarks"
            )
    return ok


def check_camera() -> bool:
    print("== camera ==")
    import cv2

    width, height = CAMERA_SIZE
    for backend_name in CAMERA_BACKENDS:
        backend = getattr(cv2, backend_name, None)
        if backend is None:
            print(f"SKIP  {backend_name}: not in this OpenCV build")
            continue
        opened_at = time.perf_counter()
        capture = cv2.VideoCapture(0, backend)
        try:
            if not capture.isOpened():
                print(f"FAIL  {backend_name}: camera 0 did not open")
                continue
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, frame = capture.read()
            open_s = time.perf_counter() - opened_at
            if not ok or frame is None:
                print(f"FAIL  {backend_name}: opened but returned no frame")
                continue
            api_name = capture.getBackendName()  # raises once the capture is released

            brightness = []
            started = time.perf_counter()
            for _ in range(CAMERA_FRAMES):
                ok, frame = capture.read()
                if not ok:
                    break
                brightness.append(float(frame.mean()))
            elapsed = time.perf_counter() - started
        finally:
            capture.release()

        if len(brightness) < CAMERA_FRAMES:
            print(f"FAIL  {backend_name}: only {len(brightness)}/{CAMERA_FRAMES} frames read")
            continue
        got_h, got_w = frame.shape[:2]
        mean = sum(brightness) / len(brightness)
        dark = "  (image almost black: lens covered or privacy shutter?)" if mean < 10 else ""
        print(
            f"OK    backend {backend_name} ({api_name}), {got_w}x{got_h} "
            f"(asked {width}x{height}), opened in {open_s:.2f} s, "
            f"{CAMERA_FRAMES / elapsed:.1f} fps measured, mean brightness {mean:.0f}/255{dark}"
        )
        return (got_w, got_h) == CAMERA_SIZE
    return False


def start_apps() -> list[dict[str, str]]:
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", START_APPS_COMMAND],
        capture_output=True,
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    parsed = json.loads(completed.stdout.decode("utf-8-sig"))
    # ConvertTo-Json emits a bare object, not a list, when there is a single entry.
    return [parsed] if isinstance(parsed, dict) else parsed


def check_apps(text: str) -> bool:
    print(f"== Start menu apps matching {text!r} ==")
    entries = start_apps()
    matches = [entry for entry in entries if text.casefold() in entry["Name"].casefold()]
    for entry in matches:
        print(f"{entry['Name']}  ->  {entry['AppID']}")
    print(f"{len(matches)} match(es) out of {len(entries)} entries")
    return bool(matches)


def press(key: str) -> bool:
    print(f"== press {key!r} ==")
    import pyautogui

    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0
    if not pyautogui.isValidKey(key):
        print(f"FAIL  {key!r} is not a pyautogui key name (press() would ignore it silently)")
        return False
    pyautogui.press(key)
    print(f"OK    sent {key!r}")
    return True


def main() -> int:
    # Start menu names can hold characters the console code page lacks: never crash on them.
    sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-camera", action="store_true", help="skip the camera check")
    parser.add_argument("--apps", metavar="TEXT", help="list Start menu apps containing TEXT")
    parser.add_argument("--press", metavar="KEY", help="send one real key press")
    options = parser.parse_args()

    if options.apps is not None:
        ok = check_apps(options.apps)
    elif options.press is not None:
        ok = press(options.press)
    else:
        ok = check_models()
        if not options.no_camera:
            ok = check_camera() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
