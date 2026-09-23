"""`launch` actions: a Start-menu app (by name or AppID) or a file/program path."""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Callable, Sequence

from gesture_remote.config import LaunchAction

logger = logging.getLogger(__name__)

ResolveApp = Callable[[str], str]
RunCommand = Callable[[Sequence[str]], object]
StartFile = Callable[..., object]
"""os.startfile-compatible: (path, arguments=str)."""


def run_launcher(argv: Sequence[str]) -> None:
    """Run a short-lived launcher; its exit code is logged, never interpreted."""
    completed = subprocess.run(list(argv), check=False, timeout=30)
    # explorer.exe may return 1 even when the app opened fine.
    logger.debug("%s exited with %s", argv[0], completed.returncode)


def apps_folder_argv(app_id: str) -> list[str]:
    return ["explorer.exe", f"shell:AppsFolder\\{app_id}"]


class LaunchHandler:
    def __init__(
        self,
        resolve_app: ResolveApp,
        run: RunCommand = run_launcher,
        startfile: StartFile = os.startfile,
    ) -> None:
        self._resolve_app = resolve_app
        self._run = run
        self._startfile = startfile

    def __call__(self, action: LaunchAction) -> None:
        if action.path is not None:
            arguments = subprocess.list2cmdline(action.args)
            logger.info("launch path %s %s", action.path, arguments)
            self._startfile(str(action.path), arguments=arguments)
            return
        if action.app is not None:
            # The loader already validated the name; the index cache makes this cheap.
            app_id = self._resolve_app(action.app)
        else:
            assert action.app_id is not None  # guaranteed by LaunchAction's validator
            app_id = action.app_id
        logger.info("launch app %s", app_id)
        self._run(apps_folder_argv(app_id))
