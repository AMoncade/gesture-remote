"""Count gesture events in the app log, for the step-6 measurements.

    count_events.py                         everything in logs/gesture-remote.log*
    count_events.py --since "13:40"         only lines at or after this time today
    count_events.py --since "2026-09-23 13:40:00"

Reads the rotated files too (oldest first). Lines are split with splitlines(), so the CRLF
endings the log has on Windows do not matter. First fires and repeats are counted apart: a
repeat is a wanted re-fire of a held key gesture, never a double fire.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = REPO_ROOT / "logs" / "gesture-remote.log"
LINE = re.compile(
    r"^(?P<stamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ +\S+ +gesture_remote\.events: "
    r"(?P<event>TRIGGER|IGNORED|ARMED|DISARMED)(?: (?P<rest>.*?))?\s*$"
)
STAMP = "%Y-%m-%d %H:%M:%S"


def log_files(base: Path) -> list[Path]:
    rotated = sorted(
        base.parent.glob(base.name + ".*"),
        key=lambda path: int(path.suffix[1:]) if path.suffix[1:].isdigit() else 0,
        reverse=True,
    )
    return [*rotated, base] if base.exists() else rotated


def parse_since(text: str | None) -> datetime | None:
    if text is None:
        return None
    if len(text) <= 8:  # a time of day: today
        today = datetime.now().strftime("%Y-%m-%d")
        text = f"{today} {text}" + (":00" if text.count(":") == 1 else "")
    return datetime.strptime(text, STAMP)


def count(lines: list[str], since: datetime | None) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = {
        "first fires": Counter(),
        "repeats": Counter(),
        "ignored": Counter(),
        "arming": Counter(),
    }
    for line in lines:
        match = LINE.match(line)
        if match is None:
            continue
        if since is not None and datetime.strptime(match["stamp"], STAMP) < since:
            continue
        event, rest = match["event"], (match["rest"] or "").split()
        if event == "TRIGGER":
            bucket = "repeats" if rest[-1:] == ["repeat"] else "first fires"
            counts[bucket][rest[0]] += 1
        elif event == "IGNORED":
            counts["ignored"][f"{rest[0]} ({rest[1] if len(rest) > 1 else '?'})"] += 1
        else:
            counts["arming"][event] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--since", help='"HH:MM[:SS]" today, or "YYYY-MM-DD HH:MM:SS"')
    parser.add_argument("--log", type=Path, default=LOG_FILE, help="base log file")
    options = parser.parse_args()

    files = log_files(options.log)
    if not files:
        print(f"no log file at {options.log}")
        return 1
    lines = [line for path in files for line in path.read_text(encoding="utf-8").splitlines()]
    counts = count(lines, parse_since(options.since))
    for title, counter in counts.items():
        total = sum(counter.values())
        detail = ", ".join(f"{key} {value}" for key, value in sorted(counter.items()))
        print(f"{title:12} {total:4}  {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
