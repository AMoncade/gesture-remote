"""Create "gesture-remote" shortcuts that start the app in the background (tray icon, no console).

make_shortcut.py              Desktop + Start menu (so Windows search finds it)
make_shortcut.py --startup    also in the Startup folder: starts with Windows
make_shortcut.py --remove     delete every shortcut
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from gesture_remote.tray import ARMED_RGB, icon_image

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHONW = REPO_ROOT / ".venv" / "Scripts" / "pythonw.exe"
ICON = REPO_ROOT / "assets" / "gesture-remote.ico"
NAME = "gesture-remote.lnk"
PLACES = ("Desktop", "Programs", "Startup")
"""Windows special folders; "Programs" is the Start menu, which Windows search indexes."""


def folder(special: str) -> Path:
    """A Windows special folder ("Desktop", "Programs", "Startup"), as the shell knows it."""
    command = f"[Environment]::GetFolderPath('{special}')"
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return Path(completed.stdout.strip())


def write_icon() -> Path:
    """The tray icon (green disc, white hand) as a multi-size .ico for the shortcuts."""
    ICON.parent.mkdir(exist_ok=True)
    image = icon_image(ARMED_RGB).resize((256, 256))
    image.save(ICON, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (256, 256)])
    return ICON


def create(target_dir: Path, icon: Path) -> Path:
    link = target_dir / NAME
    script = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:GR_LINK); "
        "$s.TargetPath = $env:GR_TARGET; $s.Arguments = '-m gesture_remote --tray'; "
        "$s.WorkingDirectory = $env:GR_DIR; $s.IconLocation = $env:GR_ICON; "
        "$s.Description = 'gesture-remote: webcam hand gestures'; $s.Save()"
    )
    env = {
        **os.environ,
        "GR_LINK": str(link),
        "GR_TARGET": str(PYTHONW),
        "GR_DIR": str(REPO_ROOT),
        "GR_ICON": str(icon),
    }
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
    parser.add_argument("--remove", action="store_true", help="delete every shortcut")
    options = parser.parse_args()

    if options.remove:
        for special in PLACES:
            link = folder(special) / NAME
            link.unlink(missing_ok=True)
            print(f"removed {link}")
        return 0
    if not PYTHONW.exists():
        print(f"{PYTHONW} is missing: create the venv first (see README)")
        return 1
    icon = write_icon()
    for special in PLACES if options.startup else PLACES[:2]:
        print(f"created {create(folder(special), icon)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
