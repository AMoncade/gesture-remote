"""Wiring: Camera -> IdleThrottle -> Recognizer -> GestureEngine -> ActionDispatcher, hot reload.

Threads: `Pipeline.run(stop)` (vision) and `ConfigWatcher.run(stop)` (hot reload) each run on
their own worker thread, so the main thread stays free (phase 3: tray icon). The watcher exists
because `ConfigStore.poll()` may run PowerShell (Get-StartApps, 1-2 s, up to a 60 s timeout): it
must never stall the vision loop. It drops each new config in a `ReloadSlot`; the vision loop
takes it on its next frame and rebuilds the engine itself, on its own clock.

The dispatcher, the Start-menu index, the recognizer and the camera live for the whole session;
a config reload only replaces the engine (bindings included), the idle throttle and the sound
switch.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, assert_never

import cv2
import numpy as np

from gesture_remote.actions import StartAppsIndex, build_dispatcher
from gesture_remote.capture import Camera, IdleThrottle
from gesture_remote.config import (
    RESTART_SECTIONS,
    ActionSpec,
    Config,
    ConfigError,
    ConfigReload,
    ConfigStore,
    IdleSettings,
    custom_model_path,
    load_config,
)
from gesture_remote.custom import CustomGestureRecognizer, CustomModel, StaleModelError
from gesture_remote.debug_view import (
    DebugWindow,
    EngineSnapshotLike,
    FrameStats,
    RateMeter,
    draw_overlay,
    overlay_lines,
)
from gesture_remote.engine import ArmedChanged, EngineEvent, GestureEngine, Ignored, Triggered
from gesture_remote.feedback import Feedback
from gesture_remote.logging_setup import log_armed, log_ignored, log_trigger
from gesture_remote.observation import FrameObservation, HandObservation
from gesture_remote.recognition import MediaPipeGestureRecognizer, Recognizer

logger = logging.getLogger(__name__)

Clock = Callable[[], float]
StepResult = Literal["frame", "no_frame", "quit"]

NO_FRAME_WAIT_S = 0.05
FIRED_SHOWN_S = 2.0
"""How long the overlay shows the last fired gesture."""
"""Pause when the camera has no frame: Camera.read() never sleeps, so this avoids a busy loop."""


# --- collaborators, structurally typed so tests can pass doubles -----------------------------


class Engine(Protocol):
    @property
    def armed(self) -> bool: ...

    def update(self, hand: HandObservation | None, now: float) -> list[EngineEvent]: ...

    def snapshot(self) -> EngineSnapshotLike: ...


class Dispatcher(Protocol):
    def dispatch(self, action: ActionSpec) -> None: ...

    def close(self) -> None: ...


class FeedbackLike(Protocol):
    sound: bool

    def triggered(self) -> None: ...

    def armed_changed(self, armed: bool) -> None: ...


class FrameSource(Protocol):
    def read(self) -> np.ndarray | None: ...


class Throttle(Protocol):
    def should_infer(self) -> bool: ...

    def report(self, hand_seen: bool) -> None: ...


class Store(Protocol):
    @property
    def config(self) -> Config: ...

    def poll(self) -> ConfigReload | None: ...


class View(Protocol):
    def show(self, image: np.ndarray) -> bool: ...

    def close(self) -> None: ...


EngineFactory = Callable[[Config, float, bool | None], Engine]
ThrottleFactory = Callable[[IdleSettings, Clock], Throttle]


def make_engine(config: Config, now: float, armed: bool | None = None) -> GestureEngine:
    """A fresh engine, in cooldown until now + cooldown_s; `armed` carries the previous state."""
    return GestureEngine(config.settings.engine, config.bindings, now=now, armed=armed)


def make_throttle(settings: IdleSettings, clock: Clock) -> IdleThrottle:
    return IdleThrottle(settings.fps, settings.after_s, clock)


# --- event routing --------------------------------------------------------------------------


def route_event(event: EngineEvent, dispatcher: Dispatcher, feedback: FeedbackLike) -> None:
    """Log the countable line, run the action, give the sound. Exhaustive over EngineEvent."""
    match event:
        case Triggered(label=label, action=action, repeat=repeat):
            log_trigger(label, action.type, repeat=repeat)
            dispatcher.dispatch(action)
            if not repeat:  # a held 👍 would beep every repeat_interval_s otherwise
                feedback.triggered()
        case Ignored(label=label, reason=reason):
            log_ignored(label, reason)  # never a sound: e.g. every gesture during a call
        case ArmedChanged(armed=armed):
            log_armed(armed)
            feedback.armed_changed(armed)
        case _:
            # A new EngineEvent type must get its own case: never swallow it silently.
            assert_never(event)


# --- hot reload, off the vision thread ------------------------------------------------------

WATCH_INTERVAL_S = 0.25
"""How often the watcher asks the store; the store itself stats the file at most once a second."""


class ReloadSlot:
    """Hands the latest reloaded config from the watcher thread to the vision thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: ConfigReload | None = None

    def put(self, reload: ConfigReload) -> None:
        """Latest wins: a reload not taken yet is replaced by a newer one."""
        with self._lock:
            self._pending = reload

    def take(self) -> ConfigReload | None:
        with self._lock:
            reload, self._pending = self._pending, None
            return reload


class ConfigWatcher:
    def __init__(
        self, store: Store, slot: ReloadSlot, interval_s: float = WATCH_INTERVAL_S
    ) -> None:
        self._store = store
        self._slot = slot
        self._interval_s = interval_s

    def poll_once(self) -> None:
        try:
            reload = self._store.poll()
        except Exception:  # ConfigStore logs its own errors; this keeps the watcher alive anyway
            logger.exception("config reload check failed; keeping the current config")
            return
        if reload is not None:
            self._slot.put(reload)

    def run(self, stop: threading.Event) -> None:
        while not stop.wait(self._interval_s):
            self.poll_once()


# --- pipeline -------------------------------------------------------------------------------


class Pipeline:
    def __init__(
        self,
        *,
        camera: FrameSource,
        recognizer: Recognizer,
        config: Config,
        reloads: ReloadSlot,
        dispatcher: Dispatcher,
        feedback: FeedbackLike,
        clock: Clock = time.monotonic,
        engine_factory: EngineFactory = make_engine,
        throttle_factory: ThrottleFactory = make_throttle,
        view: View | None = None,
        view_factory: Callable[[], View] | None = None,
        close_quits: bool = True,
        no_frame_wait_s: float = NO_FRAME_WAIT_S,
    ) -> None:
        """`view` is shown from the start. With `view_factory`, another thread may flip
        `show_view` to open or close the window at run time (it is always created, shown and
        destroyed on the pipeline thread). `close_quits=False` (tray mode): closing the window
        only hides it."""
        self._camera = camera
        self._recognizer = recognizer
        self._reloads = reloads
        self._dispatcher = dispatcher
        self._feedback = feedback
        self._clock = clock
        self._engine_factory = engine_factory
        self._throttle_factory = throttle_factory
        self._view = view
        self._view_factory = view_factory
        self.show_view = view is not None
        """Written by the tray thread, read by the pipeline thread (a plain bool is enough)."""
        self._close_quits = close_quits
        self._no_frame_wait_s = no_frame_wait_s
        self._last_fired: tuple[str, float] | None = None

        self._startup_config = config
        self._config = config
        self._engine = engine_factory(self._config, clock(), None)
        self._throttle = throttle_factory(self._config.settings.idle, clock)
        feedback.sound = self._config.settings.feedback.sound
        self.restart_required: tuple[str, ...] = ()
        """Sections changed since startup that only a restart applies (shown in the overlay)."""
        self.failed = False
        """True once run() stopped on an unexpected exception."""

        self._camera_rate = RateMeter(clock)
        self._inference_rate = RateMeter(clock)
        self._inference_ms: float | None = None

    @property
    def engine(self) -> Engine:
        return self._engine

    def run(self, stop: threading.Event) -> None:
        """Process frames until `stop` is set (or the debug window is closed)."""
        logger.info("pipeline started")
        try:
            while not stop.is_set():
                result = self.step()
                if result == "quit":
                    logger.info("debug window closed: stopping")
                    stop.set()
                elif result == "no_frame":
                    stop.wait(self._no_frame_wait_s)
        except Exception:
            self.failed = True
            logger.exception("pipeline stopped on an unexpected error")
            stop.set()
        finally:
            if self._view is not None:
                self._view.close()
            logger.info("pipeline stopped")

    def step(self) -> StepResult:
        """Handle one camera frame (reload check, inference if due, events, overlay)."""
        self._apply_reload(self._reloads.take())
        self._sync_view()
        frame = self._camera.read()
        if frame is None:
            return "no_frame"
        self._camera_rate.tick()
        now = self._clock()

        observation: FrameObservation | None = None
        if self._throttle.should_infer():
            observation = self._infer(frame, now)
            hand = observation.primary()
            self._throttle.report(hand is not None)
            for event in self._engine.update(hand, now):
                route_event(event, self._dispatcher, self._feedback)
                if isinstance(event, Triggered) and not event.repeat:
                    self._last_fired = (f"{event.label} -> {event.action.type}", now)

        if self._view is not None and not self._view.show(self._overlay(frame, observation, now)):
            if self._close_quits:
                return "quit"
            self.show_view = False
            self._sync_view()
        return "frame"

    def _sync_view(self) -> None:
        if self._view_factory is None:
            return
        if self.show_view and self._view is None:
            self._view = self._view_factory()
        elif not self.show_view and self._view is not None:
            self._view.close()
            self._view = None

    def _infer(self, frame: np.ndarray, now: float) -> FrameObservation:
        # cvtColor returns a new contiguous array; never a frame[..., ::-1] view: mp.Image
        # ignores numpy strides and would read BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        started = time.perf_counter()
        observation = self._recognizer.recognize(rgb, int(now * 1000))
        self._inference_ms = (time.perf_counter() - started) * 1000
        self._inference_rate.tick()
        return observation

    def _overlay(
        self, frame: np.ndarray, observation: FrameObservation | None, now: float
    ) -> np.ndarray:
        stats = FrameStats(
            inferred=observation is not None,
            inference_ms=self._inference_ms,
            camera_fps=self._camera_rate.rate,
            inference_fps=self._inference_rate.rate,
        )
        fired = self._last_fired
        recent = fired[0] if fired and now - fired[1] < FIRED_SHOWN_S else None
        lines = overlay_lines(
            observation, self._engine.snapshot(), stats, self.restart_required, recent
        )
        return draw_overlay(frame, observation, lines)

    def _apply_reload(self, reload: ConfigReload | None) -> None:
        if reload is None:
            return
        new, old = reload.config, self._config
        self._config = new
        # New bindings and settings, in cooldown; the arm state is kept: a reload must never
        # re-arm a remote the user disarmed (e.g. before a call).
        self._engine = self._engine_factory(new, self._clock(), self._engine.armed)
        if new.settings.idle != old.settings.idle:
            self._throttle = self._throttle_factory(new.settings.idle, self._clock)
        self._feedback.sound = new.settings.feedback.sound
        restart = tuple(
            section
            for section in RESTART_SECTIONS
            if getattr(new.settings, section) != getattr(self._startup_config.settings, section)
        )
        if restart != self.restart_required:
            if restart:
                logger.warning("restart required: %s changed since startup", ", ".join(restart))
            else:
                logger.info("camera and recognition settings are back to their startup values")
        self.restart_required = restart
        logger.info(
            "engine recreated (in cooldown, %s), bindings: %s",
            "armed" if self._engine.armed else "disarmed",
            ", ".join(sorted(new.bindings)) or "none",
        )


# --- the application ------------------------------------------------------------------------


@dataclass
class App:
    """Everything that lives for the session; `run()` blocks until Ctrl+C or the window closes."""

    pipeline: Pipeline
    watcher: ConfigWatcher
    store: Store
    dispatcher: Dispatcher
    recognizer: Recognizer
    camera: Camera

    stop: threading.Event = field(default_factory=threading.Event)
    """Set it from any thread (e.g. the tray's Quit) to end run()."""

    WATCHER_JOIN_S = 2.0
    """The watcher may sit in Get-StartApps (up to 60 s): it is a daemon, never waited for long."""

    def run(self) -> int:
        stop = self.stop
        worker = threading.Thread(
            target=self.pipeline.run, args=(stop,), name="pipeline", daemon=True
        )
        watcher = threading.Thread(
            target=self.watcher.run, args=(stop,), name="config-watcher", daemon=True
        )
        worker.start()
        watcher.start()
        try:
            while worker.is_alive():
                worker.join(0.5)  # a bare join() would not let Ctrl+C through on Windows
        except KeyboardInterrupt:
            logger.info("Ctrl+C: stopping")
        finally:
            stop.set()
            worker.join()
            watcher.join(self.WATCHER_JOIN_S)
            self.close()
        return 1 if self.pipeline.failed else 0

    def close(self) -> None:
        self.camera.close()
        self.recognizer.close()
        self.dispatcher.close()


def build_app(
    config_path: Path,
    *,
    log_dir: Path,
    debug: bool = False,
    dry_run: bool = False,
    tray: bool = False,
    start_apps: StartAppsIndex | None = None,
    is_valid_key: Callable[[str], bool] | None = None,
    dispatcher_factory: Callable[..., Dispatcher] = build_dispatcher,
    recognizer_factory: Callable[..., Recognizer] = MediaPipeGestureRecognizer,
    camera_factory: Callable[..., Camera] = Camera,
    feedback_factory: Callable[[bool], FeedbackLike] = Feedback,
) -> App:
    """Load the config and build every session object. Raises ConfigError on a bad config.

    One StartAppsIndex serves both the loader (validation) and the dispatcher (launch), so the
    Start menu is read once and cached for the session.
    """
    index = start_apps if start_apps is not None else StartAppsIndex()
    if is_valid_key is None:
        import pyautogui

        is_valid_key = pyautogui.isValidKey
    # Custom gestures first: their labels must be known to validate the bindings.
    custom_path = custom_model_path(config_path)
    custom: CustomModel | None = None
    if custom_path.exists():
        try:
            custom = CustomModel.load(custom_path)
        except (StaleModelError, OSError, EOFError) as error:
            raise ConfigError(
                Path(config_path), [f"settings.recognition.custom_model: {error}"]
            ) from None
    labels = MediaPipeGestureRecognizer.labels | (custom.labels if custom else frozenset())
    load = functools.partial(
        load_config,
        labels=labels,
        is_valid_key=is_valid_key,
        resolve_app=index.resolve,
        skip_unknown_bindings=custom is None,  # fresh clone / not trained yet: still start
    )
    store = ConfigStore(config_path, load)
    settings = store.config.settings
    try:
        model = settings.recognition.model.read_bytes()  # absolute: resolved by the loader
    except OSError as error:
        raise ConfigError(
            store.path,
            [f"settings.recognition.model: cannot read {settings.recognition.model}: {error}"],
        ) from None

    dispatcher = dispatcher_factory(log_dir=log_dir / "scripts", start_apps=index, dry_run=dry_run)
    recognizer = recognizer_factory(model, settings.recognition)
    if custom is not None:
        recognizer = CustomGestureRecognizer(
            recognizer, custom, settings.recognition.custom_min_score
        )
    camera = camera_factory(settings.camera)
    reloads = ReloadSlot()
    pipeline = Pipeline(
        camera=camera,
        recognizer=recognizer,
        config=store.config,
        reloads=reloads,
        dispatcher=dispatcher,
        feedback=feedback_factory(settings.feedback.sound),
        view=DebugWindow() if debug else None,
        view_factory=DebugWindow if tray else None,
        close_quits=not tray,
    )
    logger.info(
        "config %s: %d binding(s), dry_run=%s, debug=%s",
        store.path,
        len(store.config.bindings),
        dry_run,
        debug,
    )
    return App(pipeline, ConfigWatcher(store, reloads), store, dispatcher, recognizer, camera)
