"""Start-menu app names -> AppUserModelIDs, through PowerShell `Get-StartApps`.

`Get-StartApps` takes 1-2 s, so the list is cached for the session and read again once when a
name is missing (the app may have just been installed).
"""

from __future__ import annotations

import difflib
import json
import logging
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass

from gesture_remote.config import AppResolutionError

logger = logging.getLogger(__name__)

START_APPS_COMMAND = (
    "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-StartApps | ConvertTo-Json -Compress"
)
MAX_SUGGESTIONS = 5

Runner = Callable[[], bytes]
"""Returns the raw stdout of `Get-StartApps | ConvertTo-Json`."""


@dataclass(frozen=True, slots=True)
class StartApp:
    name: str
    app_id: str


def run_get_start_apps() -> bytes:
    """Real runner: PowerShell without profile, prompts or window, UTF-8 output."""
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", START_APPS_COMMAND],
        capture_output=True,
        check=True,
        timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return completed.stdout


def parse_start_apps(raw: bytes) -> list[StartApp]:
    """Parse `ConvertTo-Json` output: a list, a single object (one entry) or nothing (none)."""
    text = raw.decode("utf-8-sig").strip()
    if not text:
        return []
    parsed = json.loads(text)
    items = [parsed] if isinstance(parsed, dict) else parsed
    apps = []
    for item in items:
        name, app_id = item.get("Name"), item.get("AppID")
        if isinstance(name, str) and isinstance(app_id, str) and name and app_id:
            apps.append(StartApp(name, app_id))
    return apps


class StartAppsIndex:
    """Session cache of the Start menu; thread-safe (loader and action worker both resolve)."""

    def __init__(self, runner: Runner = run_get_start_apps) -> None:
        self._runner = runner
        self._apps: list[StartApp] | None = None
        self._lock = threading.Lock()

    def resolve(self, name: str) -> str:
        """Exact name first, then case-insensitive; raises AppResolutionError otherwise."""
        with self._lock:
            if self._apps is None:
                self._apps = self._load()
            app_id = self._match(name)
            if app_id is None:
                logger.info("Start-menu app %r not in cache, reading Get-StartApps again", name)
                self._apps = self._load()
                app_id = self._match(name)
            if app_id is None:
                raise AppResolutionError(self._missing_message(name))
            return app_id

    def _load(self) -> list[StartApp]:
        apps = parse_start_apps(self._runner())
        logger.debug("Get-StartApps: %d entries", len(apps))
        return apps

    def _match(self, name: str) -> str | None:
        """AppID of the unique match, None if nothing matches; raises if ambiguous."""
        assert self._apps is not None
        for matches in (
            [app for app in self._apps if app.name == name],
            [app for app in self._apps if app.name.casefold() == name.casefold()],
        ):
            app_ids = sorted({app.app_id for app in matches})
            if len(app_ids) == 1:
                return app_ids[0]
            if len(app_ids) > 1:
                listed = "\n".join(f"  {app.name} -> {app.app_id}" for app in matches)
                raise AppResolutionError(
                    f"Start-menu app {name!r} is ambiguous; use app_id with one of:\n{listed}"
                )
        return None

    def _missing_message(self, name: str) -> str:
        assert self._apps is not None
        by_folded = {app.name.casefold(): app.name for app in self._apps}
        close = difflib.get_close_matches(name.casefold(), by_folded, n=MAX_SUGGESTIONS, cutoff=0.5)
        contains = [folded for folded in by_folded if name.casefold() in folded]
        ranked = close + [folded for folded in contains if folded not in close]
        suggestions = [by_folded[folded] for folded in ranked[:MAX_SUGGESTIONS]]
        hint = f"; did you mean: {', '.join(map(repr, suggestions))}?" if suggestions else ""
        return f"Start-menu app {name!r} not found among {len(self._apps)} entries{hint}"
