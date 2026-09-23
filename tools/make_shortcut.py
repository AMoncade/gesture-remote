"""Create a "gesture-remote" shortcut that starts the app in the background (tray icon, no console).

make_shortcut.py              on the Desktop
make_shortcut.py --startup    also in the Startup folder: starts with Windows
make_shortcut.py --remove     delete both shortcuts
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHONW = REPO_ROOT / ".venv" / "Scripts" / "pythonw.exe"
NAME = "gesture-remote.lnk"


def folder(special: str) -> Path:
    """A Windows special folder ("Desktop", "Startup"), as the shell knows it."""
    command = f"[Environment]::GetFolderPath('{special}')"
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return Path(completed.stdout.strip())


def create(target_dir: Path) -> Path:
    link = target_dir / NAME
    script = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:GR_LINK); "
        "$s.TargetPath = $env:GR_TARGET; $s.Arguments = '-m gesture_remote --tray'; "
        "$s.WorkingDirectory = $env:GR_DIR; $s.Description = 'gesture-remote (tray)'; $s.Save()"
    )
    env = {**os.environ, "GR_LINK": str(link), "GR_TARGET": str(PYTHONW), "GR_DIR": str(REPO_ROOT)}
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return link


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--startup", action="store_true", help="also start with Windows")
    parser.add_argument("--remove", action="store_true", help="delete the shortcuts")
    options = parser.parse_args()

    places = [folder("Desktop"), folder("Startup")]
    if options.remove:
        for place in places:
            (place / NAME).unlink(missing_ok=True)
            print(f"removed {place / NAME}")
        return 0
    if not PYTHONW.exists():
        print(f"{PYTHONW} is missing: create the venv first (see README)")
        return 1
    for place in places if options.startup else places[:1]:
        print(f"created {create(place)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
