"""Download the MediaPipe model bundles and the official sample images into models/.

This is the only part of gesture-remote that uses the network. Each file is downloaded to a
temporary name, checked against its pinned SHA-256, then renamed: an interrupted or tampered
download never replaces a good file.

    download_models.py            fetch what is missing, verify what is present
    download_models.py --force    fetch everything again
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
MODELS_BASE = "https://storage.googleapis.com/mediapipe-models"
SAMPLES_BASE = "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer"
TIMEOUT_S = 60


@dataclass(frozen=True)
class Asset:
    url: str
    target: Path
    sha256: str | None
    """Pinned after the first verified download; None means trust on first use."""


ASSETS: tuple[Asset, ...] = (
    Asset(
        f"{MODELS_BASE}/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
        MODELS_DIR / "gesture_recognizer.task",
        "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482",
    ),
    Asset(
        f"{MODELS_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODELS_DIR / "hand_landmarker.task",
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    ),
    Asset(
        f"{SAMPLES_BASE}/thumbs_up.jpg",
        MODELS_DIR / "samples" / "thumbs_up.jpg",
        "2aee0e3a69ba5f0d3287597e61d265f4f3ac2a44ccec198dddd2639b0c8ef7ba",
    ),
    Asset(
        f"{SAMPLES_BASE}/thumbs_down.jpg",
        MODELS_DIR / "samples" / "thumbs_down.jpg",
        "080b589bf3b91ba10cc6c03645be3b5b491a8ca8c8f7d65b5f32c563ae266af9",
    ),
    Asset(
        f"{SAMPLES_BASE}/victory.jpg",
        MODELS_DIR / "samples" / "victory.jpg",
        "6ac265f3ace6a6c4ac4a9079b63fcce4ab6517272afb1e430857f55ef324fde6",
    ),
    Asset(
        f"{SAMPLES_BASE}/pointing_up.jpg",
        MODELS_DIR / "samples" / "pointing_up.jpg",
        "f4a701316b63dd8fa56e622f2b3042766369ccc189f0d89513f803cd985b993b",
    ),
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(asset: Asset) -> str:
    """Download `asset` next to its target, verify it, then atomically move it into place."""
    asset.target.parent.mkdir(parents=True, exist_ok=True)
    partial = asset.target.with_name(asset.target.name + ".part")
    request = urllib.request.Request(asset.url, headers={"User-Agent": "gesture-remote/0.1"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response, partial.open("wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    actual = sha256_of(partial)
    if asset.sha256 is not None and actual != asset.sha256:
        partial.unlink()
        raise RuntimeError(f"SHA-256 mismatch for {asset.url}: got {actual}, pinned {asset.sha256}")
    os.replace(partial, asset.target)
    return actual


def check_present(asset: Asset) -> str:
    actual = sha256_of(asset.target)
    if asset.sha256 is not None and actual != asset.sha256:
        raise RuntimeError(
            f"{asset.target} does not match its pinned SHA-256 (got {actual}); "
            "delete it or run with --force"
        )
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="download every file again")
    options = parser.parse_args()

    failures = 0
    for asset in ASSETS:
        name = asset.target.relative_to(REPO_ROOT).as_posix()
        try:
            if asset.target.exists() and not options.force:
                digest, action = check_present(asset), "present"
            else:
                digest, action = fetch(asset), "downloaded"
        except Exception as error:  # report every asset, then fail once
            failures += 1
            print(f"FAIL  {name}: {error}")
            continue
        pin = "pinned OK" if asset.sha256 else "NOT PINNED (trust on first use)"
        size = asset.target.stat().st_size
        print(f"OK    {name}  {action}  {size} bytes  sha256={digest}  [{pin}]")

    if failures:
        print(f"{failures} file(s) failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
