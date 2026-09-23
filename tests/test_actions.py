"""Actions: handlers with fakes (no real key, app, URL or beep), one real script subprocess."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gesture_remote.actions import (
    ACTION_TYPES,
    ActionDispatcher,
    KeysHandler,
    LaunchHandler,
    PyAutoGuiPresser,
    ScriptRunner,
    UrlHandler,
    default_handlers,
)
from gesture_remote.config import (
    KeysAction,
    LaunchAction,
    QuitAction,
    ScriptAction,
    UrlAction,
)

APPLE_MUSIC_ID = "AppleInc.AppleMusicWin_nzyj5cx40ttqa!App"
WAIT_S = 10.0


# --- fakes ----------------------------------------------------------------------------------


class FakePresser:
    def __init__(self) -> None:
        self.chords: list[tuple[str, ...]] = []

    def hotkey(self, *keys: str) -> None:
        self.chords.append(keys)


class FakeProcess:
    """Runs until `finish()` is called."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self._done = threading.Event()

    def poll(self) -> int | None:
        return self.returncode

    def wait(self) -> int:
        self._done.wait()
        assert self.returncode is not None
        return self.returncode

    def finish(self, code: int = 0) -> None:
        self.returncode = code
        self._done.set()


class FakePopen:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.processes: list[FakeProcess] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> FakeProcess:
        self.calls.append((argv, kwargs))
        process = FakeProcess()
        self.processes.append(process)
        return process


class Recorder:
    """Callable that records its positional and keyword arguments."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


@pytest.fixture
def popen() -> FakePopen:
    return FakePopen()


@pytest.fixture
def runner(tmp_path: Path, popen: FakePopen) -> ScriptRunner:
    return ScriptRunner(tmp_path / "logs" / "scripts", popen=popen, python="py.exe")


def make_script(folder: Path, name: str = "macro.py", body: str = "") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(body, encoding="utf-8")
    return path


# --- keys -----------------------------------------------------------------------------------


def test_keys_presses_the_exact_chord() -> None:
    presser = FakePresser()
    handler = KeysHandler(presser)
    handler(KeysAction(type="keys", keys=["ctrl", "shift", "esc"]))
    handler(KeysAction(type="keys", keys=["playpause"], repeat_while_held=True))
    assert presser.chords == [("ctrl", "shift", "esc"), ("playpause",)]


def test_pyautogui_adapter_disables_failsafe_and_pause() -> None:
    calls: list[tuple[str, ...]] = []
    module = SimpleNamespace(FAILSAFE=True, PAUSE=0.1, hotkey=lambda *keys: calls.append(keys))
    presser = PyAutoGuiPresser(module)  # type: ignore[arg-type]
    assert module.FAILSAFE is False
    assert module.PAUSE == 0
    presser.hotkey("volumeup")
    assert calls == [("volumeup",)]


# --- url ------------------------------------------------------------------------------------


def test_url_opens_the_exact_url_in_a_new_tab() -> None:
    opened: list[str] = []
    UrlHandler(opened.append)(UrlAction(type="url", url="https://studium.umontreal.ca/a?b=1"))
    assert opened == ["https://studium.umontreal.ca/a?b=1"]


# --- launch ---------------------------------------------------------------------------------


def test_launch_app_resolves_then_opens_apps_folder() -> None:
    run = Recorder()
    resolved: list[str] = []

    def resolve(name: str) -> str:
        resolved.append(name)
        return APPLE_MUSIC_ID

    LaunchHandler(resolve, run=run, startfile=Recorder())(
        LaunchAction(type="launch", app="Apple Music")
    )
    assert resolved == ["Apple Music"]
    assert run.calls == [((["explorer.exe", "shell:AppsFolder\\" + APPLE_MUSIC_ID],), {})]


def test_launch_app_id_skips_resolution() -> None:
    run = Recorder()

    def resolve(name: str) -> str:
        raise AssertionError("app_id must not be resolved")

    LaunchHandler(resolve, run=run, startfile=Recorder())(
        LaunchAction(type="launch", app_id="calc!App")
    )
    assert run.calls == [((["explorer.exe", "shell:AppsFolder\\calc!App"],), {})]


def test_launch_path_calls_startfile_with_one_argument_string() -> None:
    startfile = Recorder()
    exe = Path("C:/Program Files/App/app.exe")
    LaunchHandler(lambda name: "", run=Recorder(), startfile=startfile)(
        LaunchAction(type="launch", path=exe, args=["--open", "My File.txt", ""])
    )
    assert startfile.calls == [((str(exe),), {"arguments": '--open "My File.txt" ""'})]


def test_launch_relative_path_without_args_does_not_crash() -> None:
    startfile = Recorder()
    LaunchHandler(lambda name: "", run=Recorder(), startfile=startfile)(
        LaunchAction(type="launch", path=Path("notes.txt"))
    )
    assert startfile.calls == [(("notes.txt",), {"arguments": ""})]


# --- script ---------------------------------------------------------------------------------


def test_script_argv_cwd_flags_and_environment(
    tmp_path: Path, runner: ScriptRunner, popen: FakePopen
) -> None:
    script = make_script(tmp_path / "macros")
    runner(ScriptAction(type="script", path=script, args=["a b", "é"]))
    [(argv, kwargs)] = popen.calls
    assert argv == ["py.exe", str(script), "a b", "é"]
    assert kwargs["cwd"] == str(script.parent)
    assert kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW
    assert kwargs["stderr"] == subprocess.STDOUT
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["env"]["PYTHONUTF8"] == "1"
    assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"
    assert kwargs["env"]["PYTHONIOENCODING"] == "utf-8"
    assert kwargs["env"]["PATH"] == os.environ["PATH"]
    assert Path(kwargs["stdout"].name) == tmp_path / "logs" / "scripts" / "macro.log"
    popen.processes[0].finish()
    assert runner.wait_idle(WAIT_S)


def test_script_relative_path_is_made_absolute(
    tmp_path: Path, runner: ScriptRunner, popen: FakePopen, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_script(tmp_path / "macros")
    monkeypatch.chdir(tmp_path)
    runner(ScriptAction(type="script", path=Path("macros/macro.py")))
    [(argv, kwargs)] = popen.calls
    assert argv[1] == str(tmp_path / "macros" / "macro.py")
    assert kwargs["cwd"] == str(tmp_path / "macros")
    popen.processes[0].finish()
    assert runner.wait_idle(WAIT_S)


def test_second_fire_while_running_is_ignored_then_accepted_after_the_end(
    tmp_path: Path, runner: ScriptRunner, popen: FakePopen, caplog: pytest.LogCaptureFixture
) -> None:
    action = ScriptAction(type="script", path=make_script(tmp_path))
    runner(action)
    with caplog.at_level(logging.INFO):
        runner(action)
    assert len(popen.calls) == 1
    assert "still running" in caplog.text

    popen.processes[0].finish(0)
    runner(action)
    assert len(popen.calls) == 2
    popen.processes[1].finish()
    assert runner.wait_idle(WAIT_S)


def test_other_scripts_are_not_blocked_by_a_running_one(
    tmp_path: Path, runner: ScriptRunner, popen: FakePopen
) -> None:
    runner(ScriptAction(type="script", path=make_script(tmp_path, "one.py")))
    runner(ScriptAction(type="script", path=make_script(tmp_path, "two.py")))
    assert len(popen.calls) == 2
    for process in popen.processes:
        process.finish()
    assert runner.wait_idle(WAIT_S)


def test_running_script_survives_a_config_reload(tmp_path: Path, popen: FakePopen) -> None:
    """A reload builds new, equal action models; the dispatcher (and its registry) is kept."""
    script = make_script(tmp_path / "macros")
    runner = ScriptRunner(tmp_path / "logs", popen=popen, python="py.exe")
    reloaded = ScriptAction.model_validate({"type": "script", "path": str(script).upper()})
    run_all(fake_handlers(scripts=runner), [ScriptAction(type="script", path=script), reloaded])
    assert len(popen.calls) == 1
    assert runner.is_running(script)
    popen.processes[0].finish()
    assert runner.wait_idle(WAIT_S)
    assert not runner.is_running(script)


def test_exit_code_and_duration_are_logged(
    tmp_path: Path, popen: FakePopen, caplog: pytest.LogCaptureFixture
) -> None:
    ticks = iter([100.0, 102.5])
    runner = ScriptRunner(
        tmp_path / "logs", popen=popen, python="py.exe", clock=lambda: next(ticks)
    )
    with caplog.at_level(logging.INFO):
        runner(ScriptAction(type="script", path=make_script(tmp_path)))
        popen.processes[0].finish(3)
        assert runner.wait_idle(WAIT_S)
    assert "macro.py exited with 3 after 2.5 s" in caplog.text


def test_popen_failure_closes_the_log_and_keeps_the_script_startable(tmp_path: Path) -> None:
    def failing_popen(argv: list[str], **kwargs: Any) -> Any:
        raise OSError("boom")

    runner = ScriptRunner(tmp_path / "logs", popen=failing_popen, python="py.exe")
    script = make_script(tmp_path)
    with pytest.raises(OSError):
        runner(ScriptAction(type="script", path=script))
    assert not runner.is_running(script)
    (tmp_path / "logs" / "macro.log").unlink()  # would fail on Windows if left open


def test_real_subprocess_writes_utf8_output_to_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The runner alone must provide UTF-8: a Claude Code shell exports PYTHONIOENCODING=utf-8,
    # which would make this test pass even without PYTHONUTF8 (seen 2026-09-23).
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    script = make_script(
        tmp_path / "macros",
        body="import os, sys\nprint('é', os.getcwd())\nsys.exit(0)\n",
    )
    runner = ScriptRunner(tmp_path / "logs" / "scripts")
    runner(ScriptAction(type="script", path=script))
    assert runner.wait_idle(WAIT_S), "the mini script did not end"
    log = (tmp_path / "logs" / "scripts" / "macro.log").read_bytes().decode("utf-8")
    assert f"é {script.parent}" in log


def test_real_subprocess_ignores_a_hostile_pythonioencoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # PYTHONIOENCODING wins over PYTHONUTF8 for stdout: the runner must override it too.
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    script = make_script(tmp_path / "macros", body="print('é')\n")
    runner = ScriptRunner(tmp_path / "logs" / "scripts")
    runner(ScriptAction(type="script", path=script))
    assert runner.wait_idle(WAIT_S), "the mini script did not end"
    log = (tmp_path / "logs" / "scripts" / "macro.log").read_bytes().decode("utf-8")
    assert "é" in log


# --- dispatcher -----------------------------------------------------------------------------


def fake_handlers(scripts: ScriptRunner | None = None) -> dict[type, Any]:
    return default_handlers(
        presser=FakePresser(),
        resolve_app=lambda name: "x!App",
        scripts=scripts or ScriptRunner(Path("unused"), popen=FakePopen()),
        open_tab=Recorder(),
        run=Recorder(),
        startfile=Recorder(),
    )


def run_all(handlers: dict[type, Any], actions: list[Any]) -> None:
    """Dispatch `actions`, wait until the worker has passed them all, then close.

    close() drops queued actions, so a flush sentinel (a keys action) marks the end.
    """
    flushed = threading.Event()
    dispatcher = ActionDispatcher({**handlers, KeysAction: lambda action: flushed.set()})
    try:
        for action in actions:
            dispatcher.dispatch(action)
        dispatcher.dispatch(KeysAction(type="keys", keys=["flush"]))
        assert flushed.wait(WAIT_S), "the worker never reached the flush sentinel"
    finally:
        dispatcher.close()


def test_every_action_type_has_a_handler() -> None:
    assert set(ACTION_TYPES) == {KeysAction, LaunchAction, UrlAction, ScriptAction, QuitAction}
    assert set(fake_handlers()) == set(ACTION_TYPES)


def test_quit_action_calls_on_quit() -> None:
    quits = Recorder()
    handlers = default_handlers(
        presser=FakePresser(),
        resolve_app=lambda name: "x!App",
        scripts=ScriptRunner(Path("unused"), popen=FakePopen()),
        on_quit=quits,
    )
    handlers[QuitAction](QuitAction(type="quit"))
    assert quits.calls == [((), {})]


def test_dispatcher_refuses_an_incomplete_registry() -> None:
    handlers = fake_handlers()
    del handlers[UrlAction]
    with pytest.raises(ValueError, match="UrlAction"):
        ActionDispatcher(handlers)


def test_dispatch_returns_before_the_action_runs() -> None:
    started, release = threading.Event(), threading.Event()
    done: list[str] = []

    def slow(action: UrlAction) -> None:
        started.set()
        release.wait(WAIT_S)
        done.append(action.url)

    handlers = fake_handlers()
    handlers[UrlAction] = slow
    dispatcher = ActionDispatcher(handlers)
    dispatcher.dispatch(UrlAction(type="url", url="https://a.example"))
    assert started.wait(WAIT_S)
    assert done == []
    release.set()
    dispatcher.close()
    assert done == ["https://a.example"]


def test_actions_run_one_at_a_time_in_order_off_the_caller_thread() -> None:
    seen: list[tuple[str, str]] = []
    handlers = fake_handlers()
    handlers[UrlAction] = lambda action: seen.append((action.url, threading.current_thread().name))
    run_all(handlers, [UrlAction(type="url", url=f"https://a.example/{n}") for n in range(5)])
    assert [url for url, _ in seen] == [f"https://a.example/{n}" for n in range(5)]
    threads = {thread for _, thread in seen}
    assert len(threads) == 1
    assert threading.current_thread().name not in threads


def test_exception_is_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    def broken(action: UrlAction) -> None:
        raise RuntimeError("handler exploded")

    handlers = fake_handlers()
    handlers[UrlAction] = broken
    with caplog.at_level(logging.ERROR):
        # run_all fails unless the action queued after the broken one still runs.
        run_all(handlers, [UrlAction(type="url", url="https://a.example")])
    assert "handler exploded" in caplog.text


def test_dry_run_executes_nothing(caplog: pytest.LogCaptureFixture) -> None:
    presser, opened, run, startfile = FakePresser(), Recorder(), Recorder(), Recorder()
    popen = FakePopen()
    dispatcher = ActionDispatcher(
        default_handlers(
            presser=presser,
            resolve_app=lambda name: "x!App",
            scripts=ScriptRunner(Path("unused"), popen=popen),
            open_tab=opened,
            run=run,
            startfile=startfile,
        ),
        dry_run=True,
    )
    with caplog.at_level(logging.INFO):
        dispatcher.dispatch(KeysAction(type="keys", keys=["a"]))
        dispatcher.dispatch(UrlAction(type="url", url="https://a.example"))
        dispatcher.dispatch(LaunchAction(type="launch", app="Apple Music"))
        dispatcher.dispatch(ScriptAction(type="script", path=Path("m.py")))
        dispatcher.close()
    assert (presser.chords, opened.calls, run.calls, startfile.calls, popen.calls) == (
        [],
        [],
        [],
        [],
        [],
    )
    assert caplog.text.count("dry-run") == 4


def test_close_drops_queued_actions_and_later_dispatch_is_harmless(
    caplog: pytest.LogCaptureFixture,
) -> None:
    started, release = threading.Event(), threading.Event()
    ran: list[str] = []

    def slow(action: UrlAction) -> None:
        started.set()
        release.wait(WAIT_S)
        ran.append(action.url)

    handlers = fake_handlers()
    handlers[UrlAction] = slow
    dispatcher = ActionDispatcher(handlers)
    dispatcher.dispatch(UrlAction(type="url", url="https://a.example/running"))
    assert started.wait(WAIT_S)
    dispatcher.dispatch(UrlAction(type="url", url="https://a.example/queued"))
    closer = threading.Thread(target=dispatcher.close)
    with caplog.at_level(logging.WARNING):
        closer.start()
        # A refused dispatch proves the shutdown flag is set, and the executor cancels the queue
        # under the same lock: nothing queued can run once the handler is released. Accepted
        # probes are queued too, so they are cancelled with the rest.
        for _ in range(int(WAIT_S / 0.01)):
            dispatcher.dispatch(UrlAction(type="url", url="https://a.example/late"))
            if "dispatcher closed" in caplog.text:
                break
            time.sleep(0.01)
    assert "dispatcher closed" in caplog.text
    release.set()
    closer.join(WAIT_S)
    assert not closer.is_alive()
    assert ran == ["https://a.example/running"]
