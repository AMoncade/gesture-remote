"""app wiring with doubles for camera, recognizer, store, dispatcher and sound; the real engine.

No camera, key, app, URL or beep is ever touched here.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, get_args

import numpy as np
import pytest

from conftest import FakeClock
from gesture_remote import __main__ as entry
from gesture_remote.actions import StartAppsIndex
from gesture_remote.app import (
    App,
    ConfigWatcher,
    Pipeline,
    ReloadSlot,
    build_app,
    make_engine,
    route_event,
)
from gesture_remote.config import (
    CameraSettings,
    Config,
    ConfigError,
    ConfigReload,
    EngineSettings,
    FeedbackSettings,
    IdleSettings,
    KeysAction,
    Settings,
    UrlAction,
)
from gesture_remote.engine import ArmedChanged, EngineEvent, Ignored, Triggered
from gesture_remote.observation import NUM_LANDMARKS, FrameObservation, Handedness, HandObservation

FPS = 15.0
SIZE = (64, 48)
URL = UrlAction(type="url", url="https://studium.umontreal.ca")
PLAYPAUSE = KeysAction(type="keys", keys=["playpause"])
VOLUME_UP = KeysAction(type="keys", keys=["volumeup"], repeat_while_held=True)
APPLE_MUSIC_ID = "AppleInc.AppleMusicWin_nzyj5cx40ttqa!App"


# --- doubles --------------------------------------------------------------------------------


class FakeCamera:
    def __init__(self, frames: int | None = None) -> None:
        self.frame = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
        self.frame[:, :, 0] = 255  # pure blue in BGR
        self.remaining = frames
        self.reads = 0
        self.closed = False

    def read(self) -> np.ndarray | None:
        self.reads += 1
        if self.remaining is not None:
            if self.remaining <= 0:
                return None
            self.remaining -= 1
        return self.frame.copy()

    def close(self) -> None:
        self.closed = True


class FakeRecognizer:
    labels = frozenset({"none", "victory", "i_love_you", "thumb_up"})

    def __init__(self) -> None:
        self.current: str | None = None
        """Label of the hand in front of the camera, None = no hand."""
        self.calls: list[tuple[np.ndarray, int]] = []
        self.closed = False

    def recognize(self, rgb: np.ndarray, timestamp_ms: int) -> FrameObservation:
        self.calls.append((rgb, timestamp_ms))
        hands = () if self.current is None else (hand(self.current),)
        return FrameObservation(hands=hands, image_size=(rgb.shape[1], rgb.shape[0]))

    def close(self) -> None:
        self.closed = True


class FakeStore:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.next_reload: ConfigReload | None = None

    def queue(self, config: Config) -> None:
        self.config = config
        self.next_reload = ConfigReload(config=config, restart_required=())

    def poll(self) -> ConfigReload | None:
        reload, self.next_reload = self.next_reload, None
        return reload


class FakeDispatcher:
    def __init__(self) -> None:
        self.actions: list[Any] = []
        self.closed = False

    def dispatch(self, action: Any) -> None:
        assert not self.closed, "dispatch on a closed dispatcher"
        self.actions.append(action)

    def close(self) -> None:
        self.closed = True


class FakeFeedback:
    def __init__(self, sound: bool = True) -> None:
        self.sound = sound
        self.played: list[str] = []

    def triggered(self) -> None:
        self.played.append("fire")

    def armed_changed(self, armed: bool) -> None:
        self.played.append("armed" if armed else "disarmed")


class PassThrough:
    """Throttle double: every frame goes to inference."""

    def should_infer(self) -> bool:
        return True

    def report(self, hand_seen: bool) -> None:
        pass


class FakeView:
    def __init__(self, keep_open: int = 10**9) -> None:
        self.images: list[np.ndarray] = []
        self.keep_open = keep_open
        self.closed = False

    def show(self, image: np.ndarray) -> bool:
        self.images.append(image)
        return len(self.images) < self.keep_open

    def close(self) -> None:
        self.closed = True


def hand(label: str, score: float = 0.9) -> HandObservation:
    return HandObservation(
        label=label,
        score=score,
        handedness=Handedness.RIGHT,
        handedness_score=0.95,
        landmarks=np.full((NUM_LANDMARKS, 3), 0.5, np.float32),
        world_landmarks=np.zeros((NUM_LANDMARKS, 3), np.float32),
    )


def config(
    bindings: dict[str, Any] | None = None,
    *,
    sound: bool = True,
    camera: CameraSettings | None = None,
    idle: IdleSettings | None = None,
) -> Config:
    return Config(
        settings=Settings(
            engine=EngineSettings(),  # stable 10, release 0.8 s, cooldown 1 s, arm i_love_you
            feedback=FeedbackSettings(sound=sound),
            camera=camera or CameraSettings(),
            idle=idle or IdleSettings(),
        ),
        bindings={"victory": URL} if bindings is None else bindings,
    )


class Rig:
    """A pipeline on doubles, driven one camera frame at a time at FPS."""

    def __init__(self, clock: FakeClock, cfg: Config | None = None, **kwargs: Any) -> None:
        self.clock = clock
        self.camera = FakeCamera()
        self.recognizer = FakeRecognizer()
        self.config = cfg or config()
        self.store = FakeStore(self.config)
        self.reloads = ReloadSlot()
        self.dispatcher = FakeDispatcher()
        self.feedback = FakeFeedback()
        kwargs.setdefault("throttle_factory", lambda settings, clock: PassThrough())
        self.pipeline = Pipeline(
            camera=self.camera,
            recognizer=self.recognizer,
            config=self.config,
            reloads=self.reloads,
            dispatcher=self.dispatcher,
            feedback=self.feedback,
            clock=clock,
            **kwargs,
        )

    def reload(self, cfg: Config) -> None:
        self.reloads.put(ConfigReload(config=cfg, restart_required=()))

    def show(self, label: str | None, frames: int) -> None:
        self.recognizer.current = label
        for _ in range(frames):
            self.clock.advance(1 / FPS)
            assert self.pipeline.step() == "frame"


@pytest.fixture
def events_log(caplog: pytest.LogCaptureFixture) -> Iterator[Callable[[], list[str]]]:
    """The countable event lines so far, as step 6 will read them."""
    with caplog.at_level(logging.INFO, logger="gesture_remote.events"):
        yield lambda: [
            record.getMessage()
            for record in caplog.records
            if record.name == "gesture_remote.events"
        ]


# --- event routing --------------------------------------------------------------------------

SAMPLE_EVENTS: dict[type, EngineEvent] = {
    Triggered: Triggered("victory", URL),
    Ignored: Ignored("closed_fist", "cooldown"),
    ArmedChanged: ArmedChanged(False),
}


@pytest.mark.parametrize("event_type", get_args(EngineEvent), ids=lambda t: t.__name__)
def test_every_engine_event_type_is_routed(event_type: type) -> None:
    """A new EngineEvent type fails here (KeyError) until it has a sample and a route."""
    route_event(SAMPLE_EVENTS[event_type], FakeDispatcher(), FakeFeedback())


def test_an_unknown_event_is_never_swallowed() -> None:
    with pytest.raises(AssertionError):
        route_event(object(), FakeDispatcher(), FakeFeedback())  # type: ignore[arg-type]


def test_triggered_dispatches_logs_and_beeps(events_log: Callable[[], list[str]]) -> None:
    dispatcher, feedback = FakeDispatcher(), FakeFeedback()
    route_event(Triggered("victory", URL), dispatcher, feedback)
    assert dispatcher.actions == [URL]
    assert feedback.played == ["fire"]
    assert events_log() == ["TRIGGER victory url"]


def test_repeat_dispatches_and_logs_without_a_beep(events_log: Callable[[], list[str]]) -> None:
    dispatcher, feedback = FakeDispatcher(), FakeFeedback()
    route_event(Triggered("thumb_up", VOLUME_UP, repeat=True), dispatcher, feedback)
    assert dispatcher.actions == [VOLUME_UP]
    assert feedback.played == []
    assert events_log() == ["TRIGGER thumb_up keys repeat"]


def test_ignored_is_logged_only(events_log: Callable[[], list[str]]) -> None:
    dispatcher, feedback = FakeDispatcher(), FakeFeedback()
    route_event(Ignored("closed_fist", "disarmed"), dispatcher, feedback)
    assert dispatcher.actions == [] and feedback.played == []
    assert events_log() == ["IGNORED closed_fist disarmed"]


def test_armed_changed_is_logged_and_sounded(events_log: Callable[[], list[str]]) -> None:
    dispatcher, feedback = FakeDispatcher(), FakeFeedback()
    route_event(ArmedChanged(False), dispatcher, feedback)
    route_event(ArmedChanged(True), dispatcher, feedback)
    assert dispatcher.actions == []
    assert feedback.played == ["disarmed", "armed"]
    assert events_log() == ["DISARMED", "ARMED"]


# --- pipeline, with the real engine ---------------------------------------------------------


def test_start_in_cooldown_then_one_fire_per_hold(
    clock: FakeClock, events_log: Callable[[], list[str]]
) -> None:
    rig = Rig(clock)
    rig.show("victory", 12)  # onset at ~0.67 s, inside the 1 s startup cooldown
    rig.show(None, 15)  # 1 s without the gesture: > release_s, the segment ends
    rig.show("victory", 40)  # a new hold: one fire, however long it lasts
    assert events_log() == ["IGNORED victory cooldown", "TRIGGER victory url"]
    assert rig.dispatcher.actions == [URL]
    assert rig.feedback.played == ["fire"]


def test_recognizer_gets_contiguous_rgb_and_millisecond_timestamps(clock: FakeClock) -> None:
    rig = Rig(clock)
    rig.show(None, 3)
    rgb, first_ms = rig.recognizer.calls[0]
    assert rgb.flags["C_CONTIGUOUS"]
    assert rgb[0, 0].tolist() == [0, 0, 255]  # the blue BGR frame, converted to RGB
    stamps = [ms for _, ms in rig.recognizer.calls]
    assert first_ms == int((1000 + 1 / FPS) * 1000)
    assert stamps[1] - stamps[0] in (66, 67)


def test_no_frame_skips_everything(clock: FakeClock) -> None:
    rig = Rig(clock)
    rig.camera.remaining = 0
    assert rig.pipeline.step() == "no_frame"
    assert rig.recognizer.calls == []


def test_throttled_frame_reaches_neither_recognizer_nor_engine(clock: FakeClock) -> None:
    updates: list[Any] = []

    class CountingEngine:
        armed = True

        def update(self, hand: Any, now: float) -> list[Any]:
            updates.append(hand)
            return []

        def snapshot(self) -> Any:
            raise AssertionError("no view in this test")

    class Closed:
        def should_infer(self) -> bool:
            return False

        def report(self, hand_seen: bool) -> None:
            raise AssertionError("no inference, nothing to report")

    rig = Rig(
        clock,
        engine_factory=lambda cfg, now, armed: CountingEngine(),
        throttle_factory=lambda settings, clock: Closed(),
    )
    rig.show("victory", 5)
    assert rig.recognizer.calls == [] and updates == []


def test_real_throttle_samples_idle_frames(clock: FakeClock) -> None:
    from gesture_remote.app import make_throttle

    rig = Rig(
        clock, config(idle=IdleSettings(fps=5.0, after_s=1.0)), throttle_factory=make_throttle
    )
    rig.show(None, 30)  # 2 s without a hand at 15 fps: about 5 inferences per second
    assert len(rig.recognizer.calls) in (10, 11)


# --- hot reload -----------------------------------------------------------------------------


def test_reload_keeps_the_dispatcher_and_restarts_the_engine_in_cooldown(
    clock: FakeClock, events_log: Callable[[], list[str]]
) -> None:
    rig = Rig(clock)
    dispatcher = rig.dispatcher
    rig.show(None, 16)
    rig.show("victory", 12)
    assert rig.dispatcher.actions == [URL]

    rig.reload(config({"victory": PLAYPAUSE}, sound=False))
    rig.show("victory", 12)  # still held across the reload: the new engine is in cooldown
    rig.show(None, 15)
    rig.show("victory", 12)  # new bindings, same dispatcher

    assert events_log() == [
        "TRIGGER victory url",
        "IGNORED victory cooldown",
        "TRIGGER victory keys",
    ]
    assert dispatcher.actions == [URL, PLAYPAUSE]
    assert not dispatcher.closed
    assert rig.feedback.sound is False


def test_reload_keeps_the_arm_state(clock: FakeClock, events_log: Callable[[], list[str]]) -> None:
    rig = Rig(clock)
    rig.show(None, 16)  # past the startup cooldown (the arm gesture is in it too)
    rig.show("i_love_you", 11)
    assert rig.pipeline.engine.armed is False
    rig.reload(config({"victory": PLAYPAUSE}))
    rig.show(None, 1)
    assert rig.pipeline.engine.armed is False  # a reload never re-arms silently
    rig.show(None, 16)
    rig.show("victory", 12)
    assert events_log() == ["DISARMED", "IGNORED victory disarmed"]
    assert rig.dispatcher.actions == []


def test_restart_required_is_tracked_against_startup(clock: FakeClock) -> None:
    rig = Rig(clock)
    rig.reload(config(camera=CameraSettings(index=1)))
    rig.show(None, 1)
    assert rig.pipeline.restart_required == ("camera",)
    rig.reload(config({"victory": PLAYPAUSE}, camera=CameraSettings(index=1)))
    rig.show(None, 1)
    assert rig.pipeline.restart_required == ("camera",)  # still not the running camera
    rig.reload(config())
    rig.show(None, 1)
    assert rig.pipeline.restart_required == ()


def test_reload_rebuilds_the_throttle_only_when_idle_changes(clock: FakeClock) -> None:
    built: list[IdleSettings] = []

    def factory(settings: IdleSettings, clock: Any) -> PassThrough:
        built.append(settings)
        return PassThrough()

    rig = Rig(clock, throttle_factory=factory)
    rig.reload(config({"victory": PLAYPAUSE}))
    rig.show(None, 1)
    assert len(built) == 1
    rig.reload(config(idle=IdleSettings(fps=2.0)))
    rig.show(None, 1)
    assert [settings.fps for settings in built] == [5.0, 2.0]


# --- run loop and view ----------------------------------------------------------------------


def test_view_gets_the_overlay_and_closing_it_stops_the_loop(clock: FakeClock) -> None:
    view = FakeView(keep_open=3)
    rig = Rig(clock, view=view)
    rig.recognizer.current = "victory"
    stop = threading.Event()
    rig.pipeline.run(stop)
    assert stop.is_set() and view.closed
    assert len(view.images) == 3
    image = view.images[0]
    assert image.shape == rig.camera.frame.shape
    assert not np.array_equal(image, rig.camera.frame)  # something was drawn on it


def test_tray_mode_opens_and_hides_the_window_at_run_time(clock: FakeClock) -> None:
    views: list[FakeView] = []

    def factory() -> FakeView:
        views.append(FakeView(keep_open=3))
        return views[-1]

    rig = Rig(clock, view_factory=factory, close_quits=False)
    assert rig.pipeline.step() == "frame" and views == []  # hidden at start
    rig.pipeline.show_view = True
    rig.pipeline.step()
    assert len(views) == 1 and len(views[0].images) == 1
    rig.pipeline.show_view = False
    rig.pipeline.step()
    assert views[0].closed
    # closing the window by hand only hides it: the loop goes on
    rig.pipeline.show_view = True
    for _ in range(3):
        assert rig.pipeline.step() == "frame"
    assert views[1].closed and rig.pipeline.show_view is False


def test_no_frame_does_not_spin(clock: FakeClock) -> None:
    rig = Rig(clock, no_frame_wait_s=0.05)
    rig.camera.remaining = 0
    stop = threading.Event()
    worker = threading.Thread(target=rig.pipeline.run, args=(stop,))
    worker.start()
    time.sleep(0.3)
    stop.set()
    worker.join(2)
    assert not worker.is_alive()
    assert 2 <= rig.camera.reads <= 12  # ~6 at 0.05 s; a busy loop would read thousands


def test_an_error_stops_the_loop_and_is_reported(
    clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    rig = Rig(clock)

    def broken(rgb: np.ndarray, timestamp_ms: int) -> FrameObservation:
        raise RuntimeError("inference exploded")

    rig.recognizer.recognize = broken  # type: ignore[method-assign]
    stop = threading.Event()
    with caplog.at_level(logging.ERROR):
        rig.pipeline.run(stop)
    assert stop.is_set() and rig.pipeline.failed
    assert "inference exploded" in caplog.text


def test_app_run_closes_everything(clock: FakeClock) -> None:
    rig = Rig(clock, view=FakeView(keep_open=2))
    watcher = ConfigWatcher(rig.store, rig.reloads)
    app = App(rig.pipeline, watcher, rig.store, rig.dispatcher, rig.recognizer, rig.camera)  # type: ignore[arg-type]
    assert app.run() == 0
    assert rig.camera.closed and rig.recognizer.closed and rig.dispatcher.closed


def test_make_engine_can_override_the_arm_state() -> None:
    assert make_engine(config(), 0.0).armed is True
    assert make_engine(config(), 0.0, armed=False).armed is False


# --- build_app ------------------------------------------------------------------------------


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    (tmp_path / "fake.task").write_bytes(b"model bytes")
    path = tmp_path / "config.yaml"
    path.write_text(
        "settings:\n"
        "  recognition: { model: fake.task }\n"
        "bindings:\n"
        '  victory: { type: launch, app: "Apple Music" }\n'
        "  open_palm: { type: keys, keys: [playpause] }\n",
        encoding="utf-8",
    )
    return path


class Spy:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self.result


def build(config_file: Path, tmp_path: Path, **overrides: Any) -> tuple[App, dict[str, Any]]:
    runs: list[int] = []

    def runner() -> bytes:
        runs.append(1)
        return json.dumps([{"Name": "Apple Music", "AppID": APPLE_MUSIC_ID}]).encode()

    parts: dict[str, Any] = {
        "index": StartAppsIndex(runner),
        "runs": runs,
        "dispatcher": Spy(FakeDispatcher()),
        "recognizer": Spy(FakeRecognizer()),
        "camera": Spy(FakeCamera()),
    }
    kwargs: dict[str, Any] = {
        "log_dir": tmp_path / "logs",
        "dry_run": True,
        "start_apps": parts["index"],
        "is_valid_key": lambda key: key == "playpause",
        "dispatcher_factory": parts["dispatcher"],
        "recognizer_factory": parts["recognizer"],
        "camera_factory": parts["camera"],
        "feedback_factory": FakeFeedback,
    }
    kwargs.update(overrides)
    return build_app(config_file, **kwargs), parts


def test_a_custom_model_makes_its_gestures_bindable(config_file: Path, tmp_path: Path) -> None:
    from sklearn.ensemble import RandomForestClassifier

    from gesture_remote.custom import CustomGestureRecognizer, CustomModel

    features = np.vstack([np.zeros((4, 63)), np.ones((4, 63))])
    classifier = RandomForestClassifier(n_estimators=3).fit(features, ["none"] * 4 + ["rock"] * 4)
    CustomModel(classifier).save(tmp_path / "models" / "custom_gestures.joblib")
    config_file.write_text(
        config_file.read_text(encoding="utf-8") + "  rock: { type: keys, keys: [playpause] }\n",
        encoding="utf-8",
    )
    app, _ = build(config_file, tmp_path)
    assert "rock" in app.store.config.bindings
    assert isinstance(app.recognizer, CustomGestureRecognizer)
    assert "rock" in app.recognizer.labels and "victory" in app.recognizer.labels


def test_without_a_custom_model_its_bindings_are_skipped(
    config_file: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config_file.write_text(
        config_file.read_text(encoding="utf-8") + "  rock: { type: keys, keys: [playpause] }\n",
        encoding="utf-8",
    )
    app, _ = build(config_file, tmp_path)  # a fresh clone must still start
    assert "rock" not in app.store.config.bindings
    assert "victory" in app.store.config.bindings
    assert "bindings.rock: 'rock' is not a known gesture" in caplog.text


def test_with_a_custom_model_an_unknown_label_is_still_refused(
    config_file: Path, tmp_path: Path
) -> None:
    from sklearn.ensemble import RandomForestClassifier

    from gesture_remote.custom import CustomModel

    features = np.vstack([np.zeros((4, 63)), np.ones((4, 63))])
    classifier = RandomForestClassifier(n_estimators=3).fit(features, ["none"] * 4 + ["rock"] * 4)
    CustomModel(classifier).save(tmp_path / "models" / "custom_gestures.joblib")
    config_file.write_text(
        config_file.read_text(encoding="utf-8") + "  rokc: { type: keys, keys: [playpause] }\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="rokc"):  # a typo is not silently skipped
        build(config_file, tmp_path)


def test_the_quit_action_stops_the_app(config_file: Path, tmp_path: Path) -> None:
    app, parts = build(config_file, tmp_path)
    ((_, dispatcher_kwargs),) = parts["dispatcher"].calls
    assert not app.stop.is_set()
    dispatcher_kwargs["on_quit"]()  # what QuitHandler calls
    assert app.stop.is_set()


def test_build_app_shares_one_start_menu_index(config_file: Path, tmp_path: Path) -> None:
    app, parts = build(config_file, tmp_path)
    ((_, dispatcher_kwargs),) = parts["dispatcher"].calls
    assert dispatcher_kwargs["start_apps"] is parts["index"]
    assert parts["runs"] == [1]  # the loader resolved "Apple Music" through that same index
    assert dispatcher_kwargs["log_dir"] == tmp_path / "logs" / "scripts"
    assert dispatcher_kwargs["dry_run"] is True
    assert app.store.config.bindings["victory"].app == "Apple Music"


def test_build_app_passes_model_bytes_and_settings(config_file: Path, tmp_path: Path) -> None:
    app, parts = build(config_file, tmp_path)
    ((model, recognition), _) = parts["recognizer"].calls[0]
    assert model == b"model bytes"
    assert recognition == app.store.config.settings.recognition
    ((camera_settings,), _) = parts["camera"].calls[0]
    assert camera_settings == CameraSettings()


def test_build_app_refuses_a_missing_model(config_file: Path, tmp_path: Path) -> None:
    (tmp_path / "fake.task").unlink()
    with pytest.raises(ConfigError, match="settings.recognition.model"):
        build(config_file, tmp_path)


def test_build_app_refuses_an_invalid_config(config_file: Path, tmp_path: Path) -> None:
    config_file.write_text("bindings:\n  none: { type: keys, keys: [playpause] }\n", "utf-8")
    with pytest.raises(ConfigError, match="bindings.none"):
        build(config_file, tmp_path)


# --- __main__ -------------------------------------------------------------------------------


@pytest.fixture
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)


def test_cli_defaults_point_at_the_repo_config() -> None:
    args = entry.parse_args([])
    assert args.config == entry.DEFAULT_CONFIG
    assert entry.DEFAULT_CONFIG.exists(), "REPO_ROOT must be the checkout root"
    assert (args.debug, args.dry_run, args.verbose) == (False, False, False)
    args = entry.parse_args(["--config", "x.yaml", "--debug", "--dry-run", "-v"])
    assert (args.config, args.debug, args.dry_run, args.verbose) == (
        Path("x.yaml"),
        True,
        True,
        True,
    )


@pytest.mark.usefixtures("restore_root_logger")
def test_cli_bad_config_exits_2_with_the_reason_in_the_log(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    assert entry.main(["--config", str(missing)], log_dir=tmp_path / "logs") == 2
    for handler in logging.getLogger().handlers:
        handler.flush()
    text = (tmp_path / "logs" / "gesture-remote.log").read_text(encoding="utf-8")
    assert "cannot read the file" in text


# --- config watcher: hot reload off the vision thread -----------------------------------------


def test_reload_slot_hands_over_the_latest_once() -> None:
    slot = ReloadSlot()
    assert slot.take() is None
    first = ConfigReload(config=config(), restart_required=())
    second = ConfigReload(config=config({"victory": PLAYPAUSE}), restart_required=())
    slot.put(first)
    slot.put(second)
    assert slot.take() is second
    assert slot.take() is None


def test_watcher_puts_a_reload_in_the_slot_and_survives_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FlakyStore(FakeStore):
        def poll(self) -> ConfigReload | None:
            if self.next_reload is None:
                raise RuntimeError("PowerShell went away")
            return super().poll()

    store, slot = FlakyStore(config()), ReloadSlot()
    watcher = ConfigWatcher(store, slot)
    with caplog.at_level(logging.ERROR):
        watcher.poll_once()
    assert slot.take() is None and "PowerShell went away" in caplog.text
    store.queue(config({"victory": PLAYPAUSE}))
    watcher.poll_once()
    reload = slot.take()
    assert reload is not None and reload.config.bindings == {"victory": PLAYPAUSE}


def test_a_slow_store_never_stalls_the_vision_loop(clock: FakeClock) -> None:
    """ConfigStore.poll may sit in Get-StartApps for seconds: frames must keep flowing."""
    times: list[float] = []

    class SlowStore(FakeStore):
        def poll(self) -> ConfigReload | None:
            time.sleep(0.5)
            return None

    class TimedView(FakeView):
        def show(self, image: np.ndarray) -> bool:
            times.append(time.perf_counter())
            return super().show(image)

    rig = Rig(clock, view=TimedView(keep_open=30))
    watcher = ConfigWatcher(SlowStore(rig.config), rig.reloads, interval_s=0.0)
    app = App(rig.pipeline, watcher, rig.store, rig.dispatcher, rig.recognizer, rig.camera)  # type: ignore[arg-type]
    assert app.run() == 0
    assert len(times) == 30
    assert times[-1] - times[0] < 0.4  # while the watcher was blocked 0.5 s in each poll


def _powershell_fails() -> bytes:
    import subprocess

    raise subprocess.CalledProcessError(1, ["powershell", "Get-StartApps"])


def _powershell_prints_garbage() -> bytes:
    return b"Get-StartApps : le terme n'est pas reconnu"


@pytest.mark.usefixtures("restore_root_logger")
@pytest.mark.parametrize(
    ("runner", "error_name"),
    [(_powershell_fails, "CalledProcessError"), (_powershell_prints_garbage, "JSONDecodeError")],
    ids=["powershell fails", "not json"],
)
def test_cli_start_menu_failure_exits_3_with_a_clear_message(
    config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: Callable[[], bytes],
    error_name: str,
) -> None:
    from gesture_remote import app as app_module

    monkeypatch.setattr(app_module, "StartAppsIndex", lambda: StartAppsIndex(runner))
    assert entry.main(["--config", str(config_file)], log_dir=tmp_path / "logs") == 3
    for handler in logging.getLogger().handlers:
        handler.flush()
    text = (tmp_path / "logs" / "gesture-remote.log").read_text(encoding="utf-8")
    assert "startup failed while validating" in text and error_name in text
    assert "Traceback" not in text
