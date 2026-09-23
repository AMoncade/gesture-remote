"""Seams between lots A (recognition), B (config, engine) and C (actions).

Every test assembles the real modules on both sides of the seam under test. Fakes stand only at
the machine boundary: MediaPipe's result object (read by duck typing), PowerShell's stdout, the
key presser, the browser, explorer, os.startfile and Popen. No camera, key, app or URL is used.

A test that fails on a real defect is marked xfail(strict=True) with the defect and its owner in
the reason: the owner's fix turns it into an unexpected pass, which fails until the marker goes.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gesture_remote.actions import (
    ActionDispatcher,
    LaunchHandler,
    ScriptRunner,
    StartAppsIndex,
    default_handlers,
)
from gesture_remote.config import (
    Config,
    ConfigError,
    ConfigStore,
    EngineSettings,
    load_config,
)
from gesture_remote.engine import ArmedChanged, EngineEvent, GestureEngine, Ignored, Triggered
from gesture_remote.observation import NONE_LABEL, HandObservation
from gesture_remote.recognition import CANNED_LABEL_SET, CANNED_LABELS, LabelMapper, to_observation

REPO_ROOT = Path(__file__).resolve().parents[1]
APPLE_MUSIC_ID = "AppleInc.AppleMusicWin_nzyj5cx40ttqa!App"
FPS = 30.0
WAIT_S = 10.0
RAW_NAME = {label: raw for raw, label in CANNED_LABELS.items() if label != NONE_LABEL}
"""snake_case label -> MediaPipe class name."""


# --- machine-boundary fakes -----------------------------------------------------------------


def mp_result(*hands: tuple[str | None, float]) -> SimpleNamespace:
    """A MediaPipe GestureRecognizerResult: one (raw class name or None = no gesture, score)
    per hand."""
    point = SimpleNamespace(x=0.5, y=0.5, z=0.0)
    return SimpleNamespace(
        gestures=[
            [SimpleNamespace(category_name=name, score=score)] if name else []
            for name, score in hands
        ],
        handedness=[[SimpleNamespace(category_name="Right", score=0.95)] for _ in hands],
        hand_landmarks=[[point] * 21 for _ in hands],
        hand_world_landmarks=[[point] * 21 for _ in hands],
    )


def start_menu_json(*entries: tuple[str, str]) -> bytes:
    items = [{"Name": name, "AppID": app_id} for name, app_id in entries]
    return json.dumps(items[0] if len(items) == 1 else items).encode("utf-8")


class PowerShell:
    """Get-StartApps stdout, scripted; `fail` makes the next calls raise like check=True."""

    def __init__(self, *entries: tuple[str, str]) -> None:
        self.entries = list(entries)
        self.fail = False
        self.calls = 0

    def __call__(self) -> bytes:
        self.calls += 1
        if self.fail:
            raise subprocess.CalledProcessError(1, "powershell", stderr=b"Get-StartApps failed")
        return start_menu_json(*self.entries)


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


class FakePresser:
    def __init__(self) -> None:
        self.chords: list[tuple[str, ...]] = []

    def hotkey(self, *keys: str) -> None:
        self.chords.append(keys)


class FakeProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self._done = threading.Event()

    def poll(self) -> int | None:
        return self.returncode

    def wait(self) -> int:
        self._done.wait()
        return self.returncode or 0

    def finish(self) -> None:
        self.returncode = 0
        self._done.set()


class FakePopen:
    def __init__(self) -> None:
        self.argvs: list[list[str]] = []
        self.processes: list[FakeProcess] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> FakeProcess:
        self.argvs.append(argv)
        self.processes.append(FakeProcess())
        return self.processes[-1]


# --- real-module helpers --------------------------------------------------------------------


def valid_key(key: str) -> bool:
    import pyautogui  # isValidKey only reads a table: no key is sent

    return pyautogui.isValidKey(key)


def load(path: Path, powershell: PowerShell) -> tuple[Config, StartAppsIndex]:
    index = StartAppsIndex(powershell)
    return load_with(path, index), index


def load_with(path: Path, index: StartAppsIndex) -> Config:
    return load_config(
        path, labels=CANNED_LABEL_SET, is_valid_key=valid_key, resolve_app=index.resolve
    )


def write_config(folder: Path, bindings: str, engine: str = "") -> Path:
    path = folder / "config.yaml"
    text = f"settings:\n  engine: {{{engine}}}\nbindings:\n{bindings}"
    path.write_text(text, encoding="utf-8")
    return path


def primary(result: SimpleNamespace) -> HandObservation | None:
    return to_observation(result, (640, 480), LabelMapper()).primary()


class Feed:
    """Drives a real GestureEngine with frames converted by the real recognition code."""

    def __init__(self, engine: GestureEngine, start: float = 2.0) -> None:
        self.engine = engine
        self.now = start
        self.events: list[EngineEvent] = []

    def frames(self, result: SimpleNamespace, count: int) -> list[EngineEvent]:
        new: list[EngineEvent] = []
        for _ in range(count):
            new += self.engine.update(primary(result), self.now)
            self.now += 1 / FPS
        self.events += new
        return new

    def gesture(self, label: str, count: int, score: float = 0.9) -> list[EngineEvent]:
        return self.frames(mp_result((RAW_NAME[label], score)), count)

    def rest(self, seconds: float) -> list[EngineEvent]:
        return self.frames(mp_result(), round(seconds * FPS))


# --- 1. labels A <-> B ----------------------------------------------------------------------


def test_every_canned_label_but_none_and_the_arm_gesture_is_bindable(tmp_path: Path) -> None:
    bindable = sorted(CANNED_LABEL_SET - {NONE_LABEL, "i_love_you"})
    assert len(bindable) == 6
    lines = "".join(f"  {label}: {{type: keys, keys: [playpause]}}\n" for label in bindable)
    config, _ = load(write_config(tmp_path, lines), PowerShell())
    assert sorted(config.bindings) == bindable


@pytest.mark.parametrize("label", sorted(CANNED_LABEL_SET - {NONE_LABEL}))
def test_every_canned_label_but_none_can_arm_and_be_tuned(tmp_path: Path, label: str) -> None:
    engine = f"arm_gesture: {label}, per_gesture: {{{label}: {{stable_frames: 3}}}}"
    config, _ = load(write_config(tmp_path, "  {}\n", engine), PowerShell())
    assert config.settings.engine.arm_gesture == label


def test_none_is_refused_everywhere_by_the_loader(tmp_path: Path) -> None:
    engine = "arm_gesture: none, per_gesture: {none: {stable_frames: 3}}"
    path = write_config(tmp_path, "  none: {type: keys, keys: [playpause]}\n", engine)
    with pytest.raises(ConfigError) as error:
        load(path, PowerShell())
    assert [problem.split(":")[0] for problem in error.value.problems] == [
        "settings.engine.arm_gesture",
        "settings.engine.per_gesture.none",
        "bindings.none",
    ]


@pytest.mark.parametrize(
    "hand",
    [("None", 0.99), ("Unknown", 0.99), (None, 0.0), ("Thumbs_Sideways", 0.99)],
    ids=["None-class", "Unknown-class", "empty-gesture-list", "unknown-raw-class"],
)
def test_a_hand_without_known_gesture_never_votes(hand: tuple[str | None, float]) -> None:
    feed = Feed(GestureEngine(EngineSettings(stable_frames=3), {}, now=0.0))
    assert primary(mp_result(hand)).label == NONE_LABEL  # type: ignore[union-attr]
    assert feed.frames(mp_result(hand), 90) == []
    assert feed.engine.snapshot().segments == ()


# --- 2. HandObservation A -> engine B -------------------------------------------------------


def test_none_frames_neither_feed_nor_cut_a_live_segment(tmp_path: Path) -> None:
    """A hand without gesture counts like no hand: it does not refresh the segment, which dies
    release_s after the last real vote, and it does not end it earlier."""
    settings = EngineSettings(stable_frames=10, release_s=0.8, cooldown_s=0.0)
    palm = load_config(
        write_config(tmp_path, "  open_palm: {type: keys, keys: [playpause]}\n"),
        labels=CANNED_LABEL_SET,
        is_valid_key=valid_key,
        resolve_app=StartAppsIndex(PowerShell()).resolve,
    ).bindings
    feed = Feed(GestureEngine(settings, palm, now=0.0))
    none_hand = mp_result(("None", 0.95))

    assert [type(event) for event in feed.gesture("open_palm", 10)] == [Triggered]
    feed.frames(none_hand, 18)  # 0.6 s < release_s: the segment is still alive
    assert feed.gesture("open_palm", 12) == []  # same segment: no second fire

    feed.frames(none_hand, 24)  # 0.8 s = release_s: released, though a hand was visible
    assert feed.engine.snapshot().segments == ()
    assert [type(event) for event in feed.gesture("open_palm", 10)] == [Triggered]


def test_score_threshold_is_compared_to_the_recognizer_score() -> None:
    settings = EngineSettings(stable_frames=3, cooldown_s=0.0)
    feed = Feed(GestureEngine(settings, {}, now=0.0))
    assert feed.gesture("victory", 10, score=0.59) == []
    assert feed.gesture("victory", 3, score=0.6) == [Ignored("victory", "unmapped")]


def test_primary_prefers_a_gesture_over_a_confident_none_hand() -> None:
    hand = primary(mp_result(("Thumb_Up", 0.8), ("None", 0.95)))
    assert hand is not None
    assert hand.label == "thumb_up"


# --- 3. AppResolutionError C -> B, Start-menu cache -----------------------------------------


def test_launch_app_is_resolved_once_for_load_and_fire(tmp_path: Path) -> None:
    powershell = PowerShell(("Apple Music", APPLE_MUSIC_ID), ("Calculator", "calc!App"))
    config, index = load(
        write_config(tmp_path, '  pointing_up: {type: launch, app: "apple music"}\n'), powershell
    )
    action = config.bindings["pointing_up"]
    assert action.app == "apple music"  # type: ignore[union-attr]  # kept as written
    run = Recorder()
    LaunchHandler(index.resolve, run=run, startfile=Recorder())(action)  # type: ignore[arg-type]
    assert run.calls == [((["explorer.exe", "shell:AppsFolder\\" + APPLE_MUSIC_ID],), {})]
    assert powershell.calls == 1


def test_missing_or_ambiguous_app_is_a_config_problem(tmp_path: Path) -> None:
    powershell = PowerShell(("Calc", "one!App"), ("Calc", "two!App"), ("Paint", "p!App"))
    bindings = (
        '  pointing_up: {type: launch, app: Calc}\n  victory: {type: launch, app: "Apple Music"}\n'
    )
    with pytest.raises(ConfigError) as error:
        load(write_config(tmp_path, bindings), powershell)
    first, second = error.value.problems
    assert first.startswith("bindings.pointing_up") and "one!App" in first and "two!App" in first
    assert second.startswith("bindings.victory") and "not found" in second


def test_powershell_failure_at_startup_escapes_the_loader_as_is(tmp_path: Path) -> None:
    """Documents the current contract: only AppResolutionError becomes a config problem, so a
    failing Get-StartApps at startup is raised raw (not ConfigError) to the caller (app.py)."""
    powershell = PowerShell()
    powershell.fail = True
    path = write_config(tmp_path, '  pointing_up: {type: launch, app: "Apple Music"}\n')
    with pytest.raises(subprocess.CalledProcessError):
        load(path, powershell)


class Store:
    """A real ConfigStore on a real file, with the real loader and a fake clock."""

    def __init__(self, folder: Path, bindings: str, powershell: PowerShell) -> None:
        self.folder, self.powershell, self.clock = folder, powershell, 1000.0
        self.index = StartAppsIndex(powershell)
        self.version = 0
        self.edit(bindings)
        self.store = ConfigStore(
            folder / "config.yaml",
            lambda path: load_with(path, self.index),
            clock=lambda: self.clock,
        )

    def edit(self, bindings: str) -> None:
        path = write_config(self.folder, bindings)
        self.version += 1
        stamp = 1_700_000_000_000_000_000 + self.version * 10**9
        os.utime(path, ns=(stamp, stamp))

    def settle(self, polls: int = 2) -> list[Any]:
        """Poll once per second; a change is loaded on the second poll that sees it."""
        results = []
        for _ in range(polls):
            self.clock += 1.0
            results.append(self.store.poll())
        return results


def test_reload_reads_the_start_menu_once_per_file_change_not_per_poll(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    powershell = PowerShell(("Apple Music", APPLE_MUSIC_ID))
    store = Store(tmp_path, '  pointing_up: {type: launch, app: "Apple Music"}\n', powershell)
    assert powershell.calls == 1

    store.edit("  pointing_up: {type: launch, app: Spotify}\n")  # missing: one re-read
    with caplog.at_level(logging.ERROR):
        assert store.settle(5) == [None] * 5
    assert powershell.calls == 2
    assert caplog.text.count("not found") == 1
    assert store.store.config.bindings["pointing_up"].app == "Apple Music"  # type: ignore[union-attr]


def test_powershell_failure_during_a_reload_keeps_the_config_until_the_next_edit(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    powershell = PowerShell(("Apple Music", APPLE_MUSIC_ID))
    store = Store(tmp_path, '  pointing_up: {type: launch, app: "Apple Music"}\n', powershell)

    powershell.fail = True
    store.edit("  pointing_up: {type: launch, app: Spotify}\n")
    with caplog.at_level(logging.ERROR):
        assert store.settle(5) == [None] * 5
    assert "Reloading" in caplog.text and "CalledProcessError" in caplog.text
    assert powershell.calls == 2  # not retried on later polls: the file did not change
    assert store.store.config.bindings["pointing_up"].app == "Apple Music"  # type: ignore[union-attr]

    powershell.fail = False
    powershell.entries.append(("Spotify", "spotify!App"))
    store.edit(
        "  pointing_up: {type: launch, app: Spotify}\n  victory: {type: url, url: 'https://a.b'}\n"
    )
    reloads = [result for result in store.settle() if result is not None]
    assert len(reloads) == 1
    assert reloads[0].config.bindings["pointing_up"].app == "Spotify"


# --- 4. paths B -> C ------------------------------------------------------------------------


def test_script_is_one_instance_across_reloads_with_other_spellings(tmp_path: Path) -> None:
    (tmp_path / "macros" / "sub").mkdir(parents=True)
    (tmp_path / "macros" / "m.py").write_text("", encoding="utf-8")
    popen = FakePopen()
    runner = ScriptRunner(tmp_path / "logs", popen=popen, python="py.exe")
    spellings = [
        "macros/m.py",
        "./macros/sub/../m.py",
        "MACROS\\M.PY",
        str(tmp_path / "macros/m.py"),
    ]
    for spelling in spellings:
        config, _ = load(
            write_config(tmp_path, f"  thumb_down: {{type: script, path: '{spelling}'}}\n"),
            PowerShell(),
        )
        runner(config.bindings["thumb_down"])  # type: ignore[arg-type]
    assert len(popen.argvs) == 1
    assert popen.argvs[0][1] == str(tmp_path / "macros" / "m.py")
    popen.processes[0].finish()
    assert runner.wait_idle(WAIT_S)


@pytest.mark.skipif(shutil.which("notepad.exe") is None, reason="notepad.exe not on PATH")
def test_bare_launch_path_from_path_reaches_startfile_absolute(tmp_path: Path) -> None:
    config, index = load(
        write_config(tmp_path, "  open_palm: {type: launch, path: notepad.exe, args: [a b]}\n"),
        PowerShell(),
    )
    startfile = Recorder()
    LaunchHandler(index.resolve, run=Recorder(), startfile=startfile)(
        config.bindings["open_palm"]  # type: ignore[arg-type]
    )
    [((target,), kwargs)] = startfile.calls
    assert Path(target).is_absolute() and Path(target).is_file()
    assert os.path.samefile(target, shutil.which("notepad.exe"))  # type: ignore[arg-type]
    assert kwargs == {"arguments": '"a b"'}


# --- 5. union <-> handlers, whole chain on the shipped config --------------------------------


def test_shipped_config_fires_every_binding_through_the_real_dispatcher(tmp_path: Path) -> None:
    """recognition -> engine -> dispatcher -> handlers, with config.yaml as shipped."""
    powershell = PowerShell(("Apple Music", APPLE_MUSIC_ID), ("Calculator", "calc!App"))
    config, index = load(REPO_ROOT / "config.yaml", powershell)
    presser, open_tab, run, popen = FakePresser(), Recorder(), Recorder(), FakePopen()
    scripts = ScriptRunner(tmp_path / "logs", popen=popen, python="py.exe")
    handlers = default_handlers(
        presser=presser,
        resolve_app=index.resolve,
        scripts=scripts,
        open_tab=open_tab,
        run=run,
        startfile=Recorder(),
    )
    dispatcher = ActionDispatcher(handlers)

    feed = Feed(GestureEngine(config.settings.engine, config.bindings, now=0.0))
    for label in config.bindings:
        feed.gesture(label, 12)
        feed.rest(1.2)  # longer than release_s and cooldown_s
    fired = [event for event in feed.events if isinstance(event, Triggered)]
    assert [event.label for event in fired] == list(config.bindings)
    try:
        for event in fired:
            dispatcher.dispatch(event.action)
        # One worker, in order: once this last keys action is pressed, all the others ran.
        # (close() would drop whatever is still queued.)
        dispatcher.dispatch(config.bindings["open_palm"])
        deadline = time.monotonic() + WAIT_S
        while len(presser.chords) < 4 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        dispatcher.close()
    assert len(presser.chords) == 4, "the worker did not reach the last action"
    assert presser.chords == [("playpause",), ("volumeup",), ("volumemute",), ("playpause",)]
    assert open_tab.calls == [(("https://studium.umontreal.ca",), {})]
    assert run.calls == [((["explorer.exe", "shell:AppsFolder\\" + APPLE_MUSIC_ID],), {})]
    assert popen.argvs == [["py.exe", str(REPO_ROOT / "macros" / "example_hello.py")]]
    assert powershell.calls == 1
    popen.processes[0].finish()
    assert scripts.wait_idle(WAIT_S)


# --- 7. engine state across a config reload -------------------------------------------------


def test_disarmed_state_survives_a_config_reload() -> None:
    settings = EngineSettings(arm_gesture="i_love_you", start_armed=True, stable_frames=5)
    bindings = {"open_palm": {"type": "keys", "keys": ["playpause"]}}
    config = Config.model_validate(
        {"settings": {"engine": settings.model_dump()}, "bindings": bindings}
    )
    feed = Feed(GestureEngine(config.settings.engine, config.bindings, now=0.0))
    assert feed.gesture("i_love_you", 5) == [ArmedChanged(False)]
    feed.rest(2.0)

    reloaded = GestureEngine(  # type: ignore[call-arg]
        config.settings.engine, config.bindings, now=feed.now, armed=feed.engine.armed
    )
    after = Feed(reloaded, start=feed.now)
    after.rest(1.5)  # past the restart cooldown
    assert after.gesture("open_palm", 5) == [Ignored("open_palm", "disarmed")]


def test_reload_without_carrying_state_re_arms_silently() -> None:
    """The symptom behind the xfail above: an engine recreated without the old state fires
    right away, with no ArmedChanged to tell the user (and so no sound)."""
    settings = EngineSettings(arm_gesture="i_love_you", start_armed=True, stable_frames=5)
    config = Config.model_validate(
        {
            "settings": {"engine": settings.model_dump()},
            "bindings": {"open_palm": {"type": "keys", "keys": ["playpause"]}},
        }
    )
    feed = Feed(GestureEngine(config.settings.engine, config.bindings, now=0.0))
    assert feed.gesture("i_love_you", 5) == [ArmedChanged(False)]
    feed.rest(2.0)
    after = Feed(GestureEngine(config.settings.engine, config.bindings, now=feed.now), feed.now)
    after.rest(1.5)
    events = after.gesture("open_palm", 5)
    assert [type(event) for event in events] == [Triggered]
    assert not any(isinstance(event, ArmedChanged) for event in after.events)
