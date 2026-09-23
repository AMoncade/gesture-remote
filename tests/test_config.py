"""load_config (contextual validation, path resolution) and ConfigStore (hot reload)."""

from __future__ import annotations

import functools
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import pyautogui
import pytest

from conftest import REPO_ROOT, FakeClock
from gesture_remote import config as config_module
from gesture_remote.config import (
    AppResolutionError,
    Config,
    ConfigError,
    ConfigReload,
    ConfigStore,
    KeysAction,
    LaunchAction,
    ScriptAction,
    load_config,
)

LABELS = frozenset(
    {
        "none",
        "closed_fist",
        "open_palm",
        "pointing_up",
        "thumb_down",
        "thumb_up",
        "victory",
        "i_love_you",
    }
)
"""The label set agreed with lot A (Recognizer.labels), `none` included."""

APPLE_MUSIC_ID = "AppleInc.AppleMusicWin_nzyj5cx40ttqa!App"


class FakeStartMenu:
    """resolve_app double: records calls, raises AppResolutionError like lot C's index."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, name: str) -> str:
        self.calls.append(name)
        if name == "Apple Music":
            return APPLE_MUSIC_ID
        if name == "Twin":
            raise AppResolutionError("'Twin' is ambiguous: A.Twin!App, B.Twin!App")
        raise AppResolutionError(f"no Start-menu app named {name!r}; did you mean 'Apple Music'?")


def load(path: Path, start_menu: FakeStartMenu | None = None) -> Config:
    return load_config(
        path,
        labels=LABELS,
        is_valid_key=pyautogui.isValidKey,
        resolve_app=start_menu or FakeStartMenu(),
    )


@pytest.fixture
def cfg_dir(tmp_path: Path) -> Path:
    """A config folder with a macro, as next to the real config.yaml."""
    (tmp_path / "macros").mkdir()
    (tmp_path / "macros" / "hello.py").write_text("print('hi')\n", encoding="utf-8")
    return tmp_path


def write(folder: Path, text: str) -> Path:
    path = folder / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def problems(path: Path) -> tuple[str, ...]:
    with pytest.raises(ConfigError) as info:
        load(path)
    return info.value.problems


# --- the shipped config.yaml ----------------------------------------------------------------


def test_shipped_config_loads_with_the_real_key_check(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # paths must come from the config folder, not the cwd
    start_menu = FakeStartMenu()
    config = load(REPO_ROOT / "config.yaml", start_menu)

    assert set(config.bindings) == {
        "open_palm",
        "victory",
        "pointing_up",
        "thumb_down",
        "thumb_up",
        "closed_fist",
    }
    assert config.bindings["thumb_down"] == ScriptAction(
        type="script", path=REPO_ROOT / "macros" / "example_hello.py"
    )
    assert config.settings.recognition.model == REPO_ROOT / "models" / "gesture_recognizer.task"
    # app: is validated through resolve_app but kept as written (lot C resolves it again).
    assert config.bindings["pointing_up"] == LaunchAction(type="launch", app="Apple Music")
    assert start_menu.calls == ["Apple Music"]
    assert config.bindings["thumb_up"] == KeysAction(
        type="keys", keys=["volumeup"], repeat_while_held=True
    )
    assert config.settings.engine.per_gesture["i_love_you"].stable_frames == 15


# --- every refusal has its message ----------------------------------------------------------

KEY = "{type: keys, keys: [playpause]}"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            f"bindings: {{thumbs_up: {KEY}}}",
            "bindings.thumbs_up: unknown gesture label 'thumbs_up' (known: closed_fist, ",
            id="unknown-label-in-bindings",
        ),
        pytest.param(
            "settings: {engine: {per_gesture: {wave: {stable_frames: 3}}}}",
            "settings.engine.per_gesture.wave: unknown gesture label 'wave'",
            id="unknown-label-in-per_gesture",
        ),
        pytest.param(
            "settings: {engine: {arm_gesture: rock}}",
            "settings.engine.arm_gesture: unknown gesture label 'rock'",
            id="unknown-arm_gesture",
        ),
        pytest.param(
            f"bindings: {{none: {KEY}}}",
            "bindings.none: 'none' means 'no known gesture' and cannot be bound",
            id="none-bound",
        ),
        pytest.param(
            "settings: {engine: {per_gesture: {none: {min_score: 0.9}}}}",
            "settings.engine.per_gesture.none: 'none' never votes",
            id="none-in-per_gesture",
        ),
        pytest.param(
            "settings: {engine: {arm_gesture: none}}",
            "settings.engine.arm_gesture: 'none' means 'no known gesture' and cannot arm",
            id="none-as-arm_gesture",
        ),
        pytest.param(
            f"bindings: {{i_love_you: {KEY}}}",
            "bindings.i_love_you: 'i_love_you' is the arm gesture",
            id="arm-gesture-bound",
        ),
        pytest.param(
            "settings: {engine: {arm_gesture: null, start_armed: false}}",
            "start_armed: false without an arm_gesture would never arm",
            id="never-armed",
        ),
        pytest.param(
            "bindings: {open_palm: {type: keys, keys: [ctrl, nope]}}",
            "bindings.open_palm: unknown key 'nope'; pyautogui would silently ignore it",
            id="invalid-key",
        ),
        pytest.param(
            "bindings: {open_palm: {type: keys, keys: [Ctrl, w]}}",
            "unknown key 'Ctrl' (key names are lowercase: 'ctrl')",
            id="invalid-key-case",
        ),
        pytest.param(
            "bindings: {pointing_up: {type: launch, app: Nope}}",
            "bindings.pointing_up: launch app 'Nope': no Start-menu app named 'Nope'",
            id="launch-app-missing",
        ),
        pytest.param(
            "bindings: {pointing_up: {type: launch, app: Twin}}",
            "launch app 'Twin': 'Twin' is ambiguous: A.Twin!App, B.Twin!App",
            id="launch-app-ambiguous",
        ),
        pytest.param(
            "bindings: {pointing_up: {type: launch, path: sub/missing.exe}}",
            "bindings.pointing_up: launch path not found: ",
            id="launch-path-missing",
        ),
        pytest.param(
            "bindings: {pointing_up: {type: launch, path: '%GR_TEST_UNSET%/x.exe'}}",
            "environment variable %GR_TEST_UNSET% is not set",
            id="launch-path-undefined-variable",
        ),
        pytest.param(
            "bindings: {pointing_up: {type: launch}}",
            "launch needs exactly one of app, app_id, path",
            id="launch-no-target",
        ),
        pytest.param(
            "bindings: {thumb_down: {type: script, path: macros/missing.py}}",
            "bindings.thumb_down: script not found: ",
            id="script-missing",
        ),
        pytest.param(
            "bindings: {thumb_down: {type: script, path: macros/hello.txt}}",
            "script path must be a .py file",
            id="script-not-py",
        ),
        pytest.param(
            "bindings: {victory: {type: url, url: 'ftp://example.org'}}",
            "url must be an http(s) URL with a host",
            id="url-invalid",
        ),
        pytest.param(
            "bindings: {victory: {type: url, url: 'https://a.org', repeat_while_held: true}}",
            "bindings.victory: repeat_while_held is only supported with type: keys "
            "(this binding is type: url)",
            id="repeat-not-keys",
        ),
        pytest.param(
            f"bindings: {{thumb_up_hold: {KEY}}}",
            "bindings.thumb_up_hold: hold bindings (x_hold) arrive in phase 3",
            id="x_hold",
        ),
        pytest.param(
            f"bindings: {{'closed_fist>thumb_up': {KEY}}}",
            "bindings.closed_fist>thumb_up: combos (a>b) arrive in phase 3",
            id="a>b",
        ),
        pytest.param(
            "bindings: {open_palm: {type: cmd, command: dir}}",
            "bindings.open_palm: type: cmd arrives in phase 3",
            id="type-cmd",
        ),
        pytest.param(
            "bindings: {open_palm: {type: lock}}",
            "type: lock arrives in phase 3",
            id="type-lock",
        ),
        pytest.param(
            "bindings: {open_palm: {type: macro, steps: []}}",
            "type: macro arrives in phase 3",
            id="type-macro",
        ),
        pytest.param(
            "bindings: {open_palm: {type: confirm}}",
            "type: confirm arrives in phase 3",
            id="type-confirm",
        ),
        pytest.param(
            "bindings: {open_palm: {type: keys, keys: [playpause], confirm: victory}}",
            "bindings.open_palm: confirm: arrives in phase 3",
            id="confirm-field",
        ),
        pytest.param(
            "bindings: {open_palm: {type: keyz, keys: [playpause]}}",
            "does not match any of the expected tags",
            id="unknown-type",
        ),
        pytest.param(
            "settings: {engine: {cooldown: 2}}",
            "settings.engine.cooldown: Extra inputs are not permitted",
            id="unknown-field",
        ),
        pytest.param(
            f"bindings:\n  thumb_up: {KEY}\n  thumb_up: {KEY}\n",
            "found duplicate key 'thumb_up'",
            id="duplicate-key",
        ),
        pytest.param(
            "bindings: !!python/object/apply:os.getcwd []",
            "invalid YAML",
            id="unsafe-tag",
        ),
        pytest.param("bindings: {open_palm: [", "invalid YAML", id="yaml-syntax"),
        pytest.param("", "the file is empty", id="empty-file"),
        pytest.param("- open_palm\n", "the file is a list", id="not-a-mapping"),
    ],
)
def test_each_rule_refuses_with_a_clear_message(
    cfg_dir: Path, monkeypatch, text: str, expected: str
) -> None:
    monkeypatch.delenv("GR_TEST_UNSET", raising=False)
    (cfg_dir / "macros" / "hello.txt").write_text("", encoding="utf-8")
    found = problems(write(cfg_dir, text))
    assert any(expected in problem for problem in found), found


def test_a_phase_3_binding_is_reported_once_not_again_by_pydantic(cfg_dir: Path) -> None:
    found = problems(write(cfg_dir, "bindings: {open_palm: {type: cmd, command: dir}}"))
    assert found == ("bindings.open_palm: type: cmd arrives in phase 3",)


def test_every_problem_is_reported_not_just_the_first(cfg_dir: Path) -> None:
    text = (
        "bindings:\n"
        "  none: {type: keys, keys: [playpause]}\n"
        "  open_palm: {type: keys, keys: [nope]}\n"
        "  thumb_down: {type: script, path: macros/missing.py}\n"
    )
    assert len(problems(write(cfg_dir, text))) == 3


def test_error_message_names_the_file_and_lists_the_problems(cfg_dir: Path) -> None:
    path = write(cfg_dir, "bindings: {open_palm: {type: keys, keys: [nope]}}")
    with pytest.raises(ConfigError, match="config refused \\(1 problem\\(s\\)\\)") as info:
        load(path)
    assert str(path) in str(info.value)
    assert "  - bindings.open_palm: unknown key 'nope'" in str(info.value)


# --- paths ----------------------------------------------------------------------------------


def launch_path(config: Config) -> Path | None:
    action = config.bindings["pointing_up"]
    assert isinstance(action, LaunchAction)
    return action.path


def test_relative_paths_are_resolved_from_the_config_folder(
    cfg_dir: Path, tmp_path_factory, monkeypatch
) -> None:
    (cfg_dir / "apps").mkdir()
    (cfg_dir / "apps" / "tool.exe").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path_factory.mktemp("elsewhere"))
    config = load(
        write(
            cfg_dir,
            "settings: {recognition: {model: models/m.task}}\n"
            "bindings:\n"
            "  pointing_up: {type: launch, path: apps/tool.exe, args: [--x]}\n"
            "  thumb_down: {type: script, path: macros/hello.py, args: [a]}\n",
        )
    )
    assert launch_path(config) == cfg_dir / "apps" / "tool.exe"
    assert config.bindings["pointing_up"] == LaunchAction(
        type="launch", path=cfg_dir / "apps" / "tool.exe", args=["--x"]
    )
    assert config.bindings["thumb_down"] == ScriptAction(
        type="script", path=cfg_dir / "macros" / "hello.py", args=["a"]
    )
    assert config.settings.recognition.model == cfg_dir / "models" / "m.task"


def test_absolute_paths_are_kept(cfg_dir: Path, tmp_path_factory) -> None:
    other = tmp_path_factory.mktemp("apps") / "tool.exe"
    other.write_text("", encoding="utf-8")
    config = load(write(cfg_dir, f"bindings: {{pointing_up: {{type: launch, path: '{other}'}}}}"))
    assert launch_path(config) == other


def test_environment_variables_are_expanded_in_launch_path(
    cfg_dir: Path, tmp_path_factory, monkeypatch
) -> None:
    folder = tmp_path_factory.mktemp("programs")
    (folder / "tool.exe").write_text("", encoding="utf-8")
    monkeypatch.setenv("GR_TEST_PROGRAMS", str(folder))
    text = "bindings: {pointing_up: {type: launch, path: '%GR_TEST_PROGRAMS%/tool.exe'}}"
    assert launch_path(load(write(cfg_dir, text))) == folder / "tool.exe"


BARE = "bindings: {pointing_up: {type: launch, path: notepad.exe}}"


def test_bare_name_found_on_path(cfg_dir: Path, tmp_path_factory, monkeypatch) -> None:
    on_path = tmp_path_factory.mktemp("path") / "notepad.exe"
    on_path.write_text("", encoding="utf-8")
    seen: list[str] = []

    def fake_which(name: str) -> str | None:
        seen.append(name)
        return str(on_path)

    monkeypatch.setattr(config_module.shutil, "which", fake_which)
    assert launch_path(load(write(cfg_dir, BARE))) == on_path
    assert seen == ["notepad.exe"]


def test_bare_name_in_the_config_folder_wins_over_path(cfg_dir: Path, monkeypatch) -> None:
    (cfg_dir / "notepad.exe").write_text("", encoding="utf-8")
    monkeypatch.setattr(config_module.shutil, "which", lambda name: pytest.fail("PATH searched"))
    assert launch_path(load(write(cfg_dir, BARE))) == cfg_dir / "notepad.exe"


def test_bare_name_found_nowhere_is_refused(cfg_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(config_module.shutil, "which", lambda name: None)
    [problem] = problems(write(cfg_dir, BARE))
    assert problem.endswith("notepad.exe nor on PATH")


@pytest.mark.skipif(shutil.which("notepad.exe") is None, reason="notepad.exe is not on PATH")
def test_real_notepad_resolves_to_an_absolute_path(cfg_dir: Path) -> None:
    resolved = launch_path(load(write(cfg_dir, BARE)))
    assert resolved is not None
    assert resolved.is_absolute()
    assert resolved.name.lower() == "notepad.exe"


def test_loaded_models_stay_frozen(cfg_dir: Path) -> None:
    config = load(write(cfg_dir, "bindings: {thumb_down: {type: script, path: macros/hello.py}}"))
    action = config.bindings["thumb_down"]
    assert isinstance(action, ScriptAction)
    with pytest.raises(ValueError, match="frozen"):
        action.path = Path("other.py")  # type: ignore[misc]


# --- ConfigStore ----------------------------------------------------------------------------

PALM_PLAYS = "bindings: {open_palm: {type: keys, keys: [playpause]}}\n"
PALM_MUTES = "bindings: {open_palm: {type: keys, keys: [volumemute]}}\n"
PALM_TYPO = "bindings: {open_palm: {type: keys, keys: [playpuase]}}\n"


class ScriptedFile:
    """A real file whose (mtime_ns, size) signature the test controls, through `stat`."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.mtime_ns = 0
        self.stat_calls = 0

    def write(self, text: str) -> None:
        self.path.write_text(text, encoding="utf-8")
        self.mtime_ns += 1

    def delete(self) -> None:
        self.path.unlink()

    def stat(self, path: Path) -> SimpleNamespace:
        self.stat_calls += 1
        size = path.stat().st_size  # FileNotFoundError when deleted, like os.stat
        return SimpleNamespace(st_mtime_ns=self.mtime_ns, st_size=size)


class Loads:
    """The production loader, counted."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, path: Path) -> Config:
        self.count += 1
        return functools.partial(
            load_config,
            labels=LABELS,
            is_valid_key=pyautogui.isValidKey,
            resolve_app=FakeStartMenu(),
        )(path)


def palm_keys(config: Config) -> list[str]:
    action = config.bindings["open_palm"]
    assert isinstance(action, KeysAction)
    return action.keys


@pytest.fixture
def store_parts(tmp_path: Path, clock: FakeClock) -> tuple[ScriptedFile, Loads]:
    scripted = ScriptedFile(tmp_path / "config.yaml")
    scripted.write(PALM_PLAYS)
    return scripted, Loads()


def make_store(scripted: ScriptedFile, loads: Loads, clock: FakeClock) -> ConfigStore:
    return ConfigStore(scripted.path, loads, clock=clock, stat=scripted.stat)


def ticks(store: ConfigStore, clock: FakeClock, n: int) -> list[ConfigReload | None]:
    results = []
    for _ in range(n):
        clock.advance(1.0)
        results.append(store.poll())
    return results


def errors(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.levelno >= logging.ERROR]


def test_store_valid_then_invalid_keeps_the_old_config_and_logs_once_then_valid(
    store_parts, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    scripted, loads = store_parts
    store = make_store(scripted, loads, clock)
    assert palm_keys(store.config) == ["playpause"]

    scripted.write(PALM_TYPO)
    assert ticks(store, clock, 10) == [None] * 10
    assert palm_keys(store.config) == ["playpause"]
    assert len(errors(caplog)) == 1
    assert "unknown key 'playpuase'" in errors(caplog)[0].getMessage()

    scripted.write(PALM_MUTES)
    reloads = [result for result in ticks(store, clock, 3) if result is not None]
    assert len(reloads) == 1
    assert reloads[0].restart_required == ()
    assert palm_keys(reloads[0].config) == ["volumemute"]
    assert store.config is reloads[0].config
    assert len(errors(caplog)) == 1


def test_store_stats_at_most_once_per_interval(store_parts, clock: FakeClock) -> None:
    scripted, loads = store_parts
    store = make_store(scripted, loads, clock)
    for _ in range(100):  # 10 s of 10 Hz polls
        clock.advance(0.1)
        store.poll()
    assert scripted.stat_calls == 1 + 10
    assert loads.count == 1


def test_store_reads_a_two_step_write_once_complete_without_error(
    store_parts, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    scripted, loads = store_parts
    store = make_store(scripted, loads, clock)
    scripted.write("")  # the editor truncates...
    assert ticks(store, clock, 1) == [None]
    scripted.write(PALM_MUTES)  # ...then writes the content
    results = ticks(store, clock, 3)
    assert [result is not None for result in results] == [False, True, False]
    assert errors(caplog) == []
    assert loads.count == 2  # the initial load and the complete file, never the empty one


def test_store_logs_an_empty_file_that_stays_once(
    store_parts, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    scripted, loads = store_parts
    store = make_store(scripted, loads, clock)
    scripted.write("")
    assert ticks(store, clock, 10) == [None] * 10
    assert len(errors(caplog)) == 1
    assert palm_keys(store.config) == ["playpause"]


def test_store_keeps_the_config_when_the_file_disappears(
    store_parts, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    scripted, loads = store_parts
    store = make_store(scripted, loads, clock)
    scripted.delete()
    assert ticks(store, clock, 10) == [None] * 10
    assert len(errors(caplog)) == 1
    assert "missing" in errors(caplog)[0].getMessage()
    scripted.write(PALM_MUTES)
    assert sum(result is not None for result in ticks(store, clock, 3)) == 1


def test_store_flags_camera_and_recognition_changes_as_needing_a_restart(
    store_parts, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    scripted, loads = store_parts
    store = make_store(scripted, loads, clock)
    caplog.set_level(logging.INFO, logger="gesture_remote.config")
    scripted.write("settings: {camera: {index: 1}, recognition: {num_hands: 2}}\n" + PALM_PLAYS)
    [reload] = [result for result in ticks(store, clock, 3) if result is not None]
    assert reload.restart_required == ("camera", "recognition")
    assert any("restart required" in record.getMessage() for record in caplog.records)

    scripted.write("settings: {camera: {index: 1}, recognition: {num_hands: 2}}\n" + PALM_MUTES)
    [reload] = [result for result in ticks(store, clock, 3) if result is not None]
    assert reload.restart_required == ()


def test_store_refuses_to_start_on_an_invalid_file(store_parts, clock: FakeClock) -> None:
    scripted, loads = store_parts
    scripted.write(PALM_TYPO)
    with pytest.raises(ConfigError, match="playpuase"):
        make_store(scripted, loads, clock)
