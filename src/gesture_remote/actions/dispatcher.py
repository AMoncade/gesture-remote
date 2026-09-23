"""ActionDispatcher: runs actions off the vision thread, one at a time, and never raises.

It lives for the whole session. A config reload replaces the gesture -> action mapping (held by
the app), not the dispatcher: the running-scripts registry and the Start-menu cache live in its
handlers, and recreating them would let a running script start twice.
"""

from __future__ import annotations

import logging
import os
import webbrowser
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, get_args

from gesture_remote.actions.keys import KeyPresser, KeysHandler, PyAutoGuiPresser
from gesture_remote.actions.launch import (
    LaunchHandler,
    ResolveApp,
    RunCommand,
    StartFile,
    run_launcher,
)
from gesture_remote.actions.script import ScriptRunner
from gesture_remote.actions.start_apps import StartAppsIndex
from gesture_remote.actions.url import OpenTab, UrlHandler
from gesture_remote.config import (
    ActionSpec,
    KeysAction,
    LaunchAction,
    QuitAction,
    ScriptAction,
    UrlAction,
)


class QuitHandler:
    """Runs `on_quit` (the app sets its stop event): the app then shuts down cleanly."""

    def __init__(self, on_quit: Callable[[], None]) -> None:
        self._on_quit = on_quit

    def __call__(self, action: QuitAction) -> None:
        logger.info("quit requested by a gesture")
        self._on_quit()


logger = logging.getLogger(__name__)

Handler = Callable[[Any], None]
ACTION_TYPES: tuple[type, ...] = get_args(get_args(ActionSpec)[0])
"""Every model of the ActionSpec union."""


class ActionDispatcher:
    def __init__(self, handlers: Mapping[type, Handler], *, dry_run: bool = False) -> None:
        missing = [model.__name__ for model in ACTION_TYPES if model not in handlers]
        if missing:
            raise ValueError(f"no handler for action type(s): {', '.join(missing)}")
        self._handlers = dict(handlers)
        self._dry_run = dry_run
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="actions")

    def dispatch(self, action: ActionSpec) -> None:
        """Queue `action` and return at once; failures are logged, never raised."""
        if self._dry_run:
            logger.info("dry-run: would run %r", action)
            return
        try:
            self._executor.submit(self._run, action)
        except RuntimeError:
            logger.warning("dispatcher closed: dropped %r", action)

    def close(self) -> None:
        """Drop queued actions and wait for the one in progress (running scripts go on)."""
        self._executor.shutdown(cancel_futures=True)

    def _run(self, action: ActionSpec) -> None:
        try:
            handler = self._handlers.get(type(action))
            if handler is None:
                logger.error("no handler for %s", type(action).__name__)
                return
            handler(action)
        except Exception:
            logger.exception("action %r failed", action)


def default_handlers(
    *,
    presser: KeyPresser,
    resolve_app: ResolveApp,
    scripts: ScriptRunner,
    open_tab: OpenTab = webbrowser.open_new_tab,
    run: RunCommand = run_launcher,
    startfile: StartFile = os.startfile,
    on_quit: Callable[[], None] = lambda: None,
) -> dict[type, Handler]:
    return {
        KeysAction: KeysHandler(presser),
        LaunchAction: LaunchHandler(resolve_app, run=run, startfile=startfile),
        UrlAction: UrlHandler(open_tab),
        ScriptAction: scripts,
        QuitAction: QuitHandler(on_quit),
    }


def build_dispatcher(
    *,
    log_dir: Path,
    start_apps: StartAppsIndex,
    dry_run: bool = False,
    on_quit: Callable[[], None] = lambda: None,
) -> ActionDispatcher:
    """Real dispatcher. Pass the same StartAppsIndex whose `resolve` the config loader uses.

    `log_dir` is the scripts log folder (logs/scripts); `on_quit` stops the app (quit action).
    """
    return ActionDispatcher(
        default_handlers(
            presser=PyAutoGuiPresser(),
            resolve_app=start_apps.resolve,
            scripts=ScriptRunner(log_dir),
            on_quit=on_quit,
        ),
        dry_run=dry_run,
    )
