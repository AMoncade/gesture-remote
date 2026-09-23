"""`script` actions: a Python macro in its own windowless process, one instance per script.

The running-scripts registry lives as long as the ScriptRunner, i.e. the whole session: a config
reload must not let a running script start a second time. A running script outlives the app on
purpose (the process is not tied to ours).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import IO, Protocol

from gesture_remote.config import ScriptAction

logger = logging.getLogger(__name__)


class Process(Protocol):
    def poll(self) -> int | None: ...

    def wait(self) -> int: ...


Popen = Callable[..., Process]
"""subprocess.Popen-compatible."""


class ScriptRunner:
    def __init__(
        self,
        log_dir: Path,
        popen: Popen = subprocess.Popen,
        python: str = sys.executable,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._log_dir = log_dir
        self._popen = popen
        self._python = python
        self._clock = clock
        self._running: dict[str, Process] = {}
        self._watchers: list[threading.Thread] = []
        self._lock = threading.Lock()

    def is_running(self, path: Path) -> bool:
        with self._lock:
            process = self._running.get(_instance_key(path))
            return process is not None and process.poll() is None

    def __call__(self, action: ScriptAction) -> None:
        script = Path(os.path.abspath(action.path))
        key = _instance_key(script)
        with self._lock:
            previous = self._running.get(key)
            if previous is not None and previous.poll() is None:
                logger.info("script %s still running: fire ignored", script.name)
                return
            self._log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self._log_dir / f"{script.stem}.log"
            log_file = log_path.open("ab")
            try:
                stamp = datetime.now().isoformat(timespec="seconds")
                log_file.write(f"\n--- {stamp} {script.name} {action.args}\n".encode())
                log_file.flush()
                process = self._popen(
                    [self._python, str(script), *action.args],
                    cwd=str(script.parent),
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"},
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            except BaseException:
                log_file.close()
                raise
            self._running[key] = process
            watcher = threading.Thread(
                target=self._watch,
                args=(script.name, process, log_file, self._clock()),
                name=f"script-{script.stem}",
                daemon=True,
            )
            self._watchers = [thread for thread in self._watchers if thread.is_alive()]
            self._watchers.append(watcher)
        logger.info("script %s started, output in %s", script.name, log_path)
        watcher.start()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait until every started script has ended and been logged; False on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            watchers = list(self._watchers)
        for watcher in watchers:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            watcher.join(remaining)
            if watcher.is_alive():
                return False
        return True

    def _watch(self, name: str, process: Process, log_file: IO[bytes], started: float) -> None:
        try:
            code = process.wait()
            logger.info(
                "script %s exited with %s after %.1f s", name, code, self._clock() - started
            )
        finally:
            log_file.close()


def _instance_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))
