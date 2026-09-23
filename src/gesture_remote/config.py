"""Configuration schema: pydantic models for config.yaml.

Round-1 contract (written by the admin in step 0):
- the public names, fields, defaults and model validators below are frozen for the parallel round;
- lot B owns this file afterwards and adds loading, validation and ConfigStore;
- context-dependent checks (known labels, key names, Start-menu apps, file existence, phase-3
  forms) belong in the loader, never in model validators: other lots build these models directly
  in their tests.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, Self, assert_never
from urllib.parse import urlsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from gesture_remote.observation import NONE_LABEL

logger = logging.getLogger(__name__)


class _Strict(BaseModel):
    """Unknown fields are errors (typo protection); models are immutable once loaded."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --- settings -------------------------------------------------------------------------------


class CameraSettings(_Strict):
    index: int = Field(0, ge=0)
    width: int = Field(640, gt=0)
    height: int = Field(480, gt=0)
    mirror: bool = True
    """Flip before inference: MediaPipe handedness assumes a selfie (mirrored) image."""
    backend: Literal["auto", "dshow", "msmf"] = "auto"
    """auto = DirectShow first, then Media Foundation."""


class RecognitionSettings(_Strict):
    model: Path = Path("models/gesture_recognizer.task")
    """Relative paths are resolved from the folder of the config file."""
    num_hands: int = Field(1, ge=1)
    min_hand_detection_confidence: float = Field(0.5, ge=0, le=1)
    min_hand_presence_confidence: float = Field(0.5, ge=0, le=1)
    min_tracking_confidence: float = Field(0.5, ge=0, le=1)
    custom_model: Path = Path("models/custom_gestures.joblib")
    """Your own gestures, trained by tools/train.py. Used only when the file exists."""
    custom_min_score: float = Field(0.8, gt=0, le=1)
    """A custom gesture replaces the built-in label only at or above this confidence."""


class GestureOverride(_Strict):
    min_score: float | None = Field(None, gt=0, le=1)
    stable_frames: int | None = Field(None, ge=1)


class EngineSettings(_Strict):
    min_score: float = Field(0.6, gt=0, le=1)
    """Minimum confidence for a frame to vote for its label."""
    stable_frames: int = Field(10, ge=1)
    """Consecutive identical votes before a gesture segment starts (its onset)."""
    release_s: float = Field(0.8, gt=0)
    """A segment stays alive while its label was seen within this many seconds."""
    cooldown_s: float = Field(1.0, ge=0)
    """Global pause after every fire (the arm gesture is exempt)."""
    repeat_delay_s: float = Field(0.5, gt=0)
    repeat_interval_s: float = Field(0.2, gt=0)
    arm_gesture: str | None = "i_love_you"
    start_armed: bool = True
    per_gesture: dict[str, GestureOverride] = Field(default_factory=dict)


class IdleSettings(_Strict):
    fps: float = Field(5.0, gt=0)
    """Inference rate while no hand has been seen for `after_s` seconds."""
    after_s: float = Field(1.0, ge=0)


class FeedbackSettings(_Strict):
    sound: bool = True


class Settings(_Strict):
    camera: CameraSettings = Field(default_factory=CameraSettings)
    recognition: RecognitionSettings = Field(default_factory=RecognitionSettings)
    engine: EngineSettings = Field(default_factory=EngineSettings)
    idle: IdleSettings = Field(default_factory=IdleSettings)
    feedback: FeedbackSettings = Field(default_factory=FeedbackSettings)


# --- actions --------------------------------------------------------------------------------


class KeysAction(_Strict):
    """Press `keys` together as one chord (pyautogui.hotkey), e.g. [playpause] or [ctrl, w]."""

    type: Literal["keys"]
    keys: list[str] = Field(min_length=1)
    repeat_while_held: bool = False

    @field_validator("keys")
    @classmethod
    def _no_blank_key(cls, keys: list[str]) -> list[str]:
        if any(not key.strip() for key in keys):
            raise ValueError("key names must not be blank")
        return keys


class LaunchAction(_Strict):
    """Open an app or file. Exactly one of `app`, `app_id` or `path`."""

    type: Literal["launch"]
    app: str | None = None
    """Start-menu display name, resolved to an AppID when the config is loaded."""
    app_id: str | None = None
    """AppUserModelID, e.g. AppleInc.AppleMusicWin_nzyj5cx40ttqa!App."""
    path: Path | None = None
    """.exe, .lnk or any file; %VAR% expanded when the config is loaded."""
    args: list[str] = Field(default_factory=list)
    """Only meaningful with `path`."""

    @model_validator(mode="after")
    def _exactly_one_target(self) -> Self:
        given = [name for name in ("app", "app_id", "path") if getattr(self, name) is not None]
        if len(given) != 1:
            raise ValueError(
                f"launch needs exactly one of app, app_id, path (got {given or 'none'})"
            )
        if self.args and self.path is None:
            raise ValueError("launch args are only supported with path")
        return self


class UrlAction(_Strict):
    """Open `url` in the default browser."""

    type: Literal["url"]
    url: str

    @field_validator("url")
    @classmethod
    def _http_url(cls, url: str) -> str:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError("url must be an http(s) URL with a host")
        return url


class ScriptAction(_Strict):
    """Run a Python macro with the project's interpreter, in its own process."""

    type: Literal["script"]
    path: Path
    """Relative paths are resolved from the folder of the config file."""
    args: list[str] = Field(default_factory=list)

    @field_validator("path")
    @classmethod
    def _python_file(cls, path: Path) -> Path:
        if path.suffix.lower() != ".py":
            raise ValueError("script path must be a .py file")
        return path


ActionSpec = Annotated[
    KeysAction | LaunchAction | UrlAction | ScriptAction,
    Field(discriminator="type"),
]
"""Every action type. Adding one = model here + handler in actions/ + exhaustiveness test."""


# --- root -----------------------------------------------------------------------------------


class Config(_Strict):
    settings: Settings = Field(default_factory=Settings)
    bindings: dict[str, ActionSpec] = Field(default_factory=dict)
    """Gesture label -> action."""


class AppResolutionError(LookupError):
    """A Start-menu app name did not resolve to exactly one AppID (missing or ambiguous)."""


# --- loading and contextual validation ------------------------------------------------------

PHASE_3_TYPES = frozenset({"cmd", "lock", "macro", "confirm"})
"""Action types of the full spec that are not implemented yet: refused with a clear message."""

RESTART_SECTIONS = ("camera", "recognition")
"""Settings sections read once at startup: a hot reload that changes them needs a restart."""

_UNEXPANDED_VAR = re.compile(r"%[^%\\/]+%")


class ConfigError(ValueError):
    """The config file was refused. `problems` holds every reason found, one line each."""

    def __init__(self, path: Path, problems: Sequence[str]) -> None:
        self.path = path
        self.problems = tuple(problems)
        listing = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"{path}: config refused ({len(self.problems)} problem(s)):\n{listing}")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """yaml.SafeLoader, except that a key repeated in one mapping is an error.

    Plain safe_load keeps the last duplicate silently: a copy-pasted binding would replace the
    first one without any message.
    """


def _construct_unique_mapping(loader: _UniqueKeySafeLoader, node: yaml.MappingNode) -> Any:
    seen: set[object] = set()
    for key_node, _ in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=True)
        try:
            duplicate = key in seen
            seen.add(key)
        except TypeError:  # unhashable key: construct_mapping reports it
            continue
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
    return loader.construct_mapping(node, deep=True)


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def load_config(
    path: str | os.PathLike[str],
    *,
    labels: frozenset[str],
    is_valid_key: Callable[[str], bool],
    resolve_app: Callable[[str], str],
) -> Config:
    """Read, validate and resolve a config file; raise ConfigError listing every problem.

    - `labels`: every label the recognizer can emit (`Recognizer.labels`), `none` included.
    - `is_valid_key`: `pyautogui.isValidKey` in production (pyautogui ignores unknown names).
    - `resolve_app`: Start-menu name -> AppID, raising AppResolutionError. Called to validate
      only: `launch.app` is returned as written and the launch handler resolves it again.

    The returned Config has absolute paths (`recognition.model`, `script.path`, `launch.path`),
    resolved from the folder of the config file, with %VAR% expanded in `launch.path`.
    """
    config_path = Path(path).absolute()
    raw = _read_yaml(config_path)
    if not isinstance(raw, dict):
        kind = "empty" if raw is None else f"a {type(raw).__name__}"
        raise ConfigError(
            config_path, [f"the file is {kind}; expected a mapping with settings: and bindings:"]
        )

    problems: list[str] = []
    screened = dict(raw)
    if "bindings" in raw:
        screened["bindings"], early = _screen_raw_bindings(raw["bindings"])
        problems.extend(early)
    try:
        config = Config.model_validate(screened)
    except ValidationError as exc:
        raise ConfigError(config_path, problems + _describe_validation_error(exc)) from None

    resolved, late = _check_in_context(
        config,
        base_dir=config_path.parent,
        labels=labels,
        is_valid_key=is_valid_key,
        resolve_app=resolve_app,
    )
    problems.extend(late)
    if problems:
        raise ConfigError(config_path, problems)
    return resolved


def _read_yaml(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ConfigError(path, [f"the file is not UTF-8 ({exc.reason})"]) from None
    except OSError as exc:
        raise ConfigError(path, [f"cannot read the file: {exc.strerror or exc}"]) from None
    try:
        return yaml.load(text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(path, [f"invalid YAML: {exc}"]) from None


def _screen_raw_bindings(raw_bindings: object) -> tuple[object, list[str]]:
    """Refuse phase-3 forms before pydantic, whose messages for them are generic or opaque.

    An unknown `type` only yields "tag does not match", and `repeat_while_held` on another type
    only "extra inputs are not permitted". Refused bindings are dropped so that pydantic does not
    report them a second time; everything else still goes through the models.
    """
    if not isinstance(raw_bindings, dict):
        return raw_bindings, []  # the model reports the wrong shape
    kept: dict[object, object] = {}
    problems: list[str] = []
    for label, spec in raw_bindings.items():
        issue = _raw_binding_issue(label, spec)
        if issue is None:
            kept[label] = spec
        else:
            problems.append(f"bindings.{label}: {issue}")
    return kept, problems


def _raw_binding_issue(label: object, spec: object) -> str | None:
    if isinstance(label, str):
        if ">" in label:
            return "combos (a>b) arrive in phase 3"
        if label.endswith("_hold"):
            return "hold bindings (x_hold) arrive in phase 3"
    if not isinstance(spec, dict):
        return None
    kind = spec.get("type")
    if isinstance(kind, str) and kind in PHASE_3_TYPES:
        return f"type: {kind} arrives in phase 3"
    if "confirm" in spec:
        return "confirm: arrives in phase 3"
    if "repeat_while_held" in spec and kind != "keys":
        return f"repeat_while_held is only supported with type: keys (this binding is type: {kind})"
    return None


def _describe_validation_error(exc: ValidationError) -> list[str]:
    lines = []
    for error in exc.errors(include_url=False):
        where = ".".join(str(part) for part in error["loc"])
        lines.append(f"{where}: {error['msg']}" if where else error["msg"])
    return lines


def _check_in_context(
    config: Config,
    *,
    base_dir: Path,
    labels: frozenset[str],
    is_valid_key: Callable[[str], bool],
    resolve_app: Callable[[str], str],
) -> tuple[Config, list[str]]:
    """Checks that need the recognizer, pyautogui, the Start menu or the file system.

    Returns the config with absolute paths, and the problems found (all of them, not the first).
    """
    problems: list[str] = []
    engine = config.settings.engine
    arm = engine.arm_gesture

    if arm is None:
        if not engine.start_armed:
            problems.append(
                "settings.engine: start_armed: false without an arm_gesture would never arm"
            )
    elif arm == NONE_LABEL:
        problems.append(
            f"settings.engine.arm_gesture: {NONE_LABEL!r} means 'no known gesture' and cannot arm"
        )
    elif arm not in labels:
        problems.append(f"settings.engine.arm_gesture: {_unknown_label(arm, labels)}")

    for label in engine.per_gesture:
        where = f"settings.engine.per_gesture.{label}"
        if label == NONE_LABEL:
            problems.append(f"{where}: {NONE_LABEL!r} never votes, an override has no effect")
        elif label not in labels:
            problems.append(f"{where}: {_unknown_label(label, labels)}")

    bindings: dict[str, ActionSpec] = {}
    for label, action in config.bindings.items():
        where = f"bindings.{label}"
        if label == NONE_LABEL:
            problems.append(f"{where}: {NONE_LABEL!r} means 'no known gesture' and cannot be bound")
        elif label not in labels:
            problems.append(f"{where}: {_unknown_label(label, labels)}")
        elif label == arm:
            problems.append(
                f"{where}: {label!r} is the arm gesture (settings.engine.arm_gesture) "
                "and cannot also be bound"
            )
        resolved, issues = _check_action(action, base_dir, is_valid_key, resolve_app)
        bindings[label] = resolved
        problems.extend(f"{where}: {issue}" for issue in issues)

    recognition = config.settings.recognition
    settings = config.settings.model_copy(
        update={
            "recognition": recognition.model_copy(
                update={
                    "model": _absolute(recognition.model, base_dir),
                    "custom_model": _absolute(recognition.custom_model, base_dir),
                }
            )
        }
    )
    # model_copy skips validation (checked: pydantic 2.13); only paths are replaced here, with
    # values the validators already accepted in their relative form.
    return config.model_copy(update={"settings": settings, "bindings": bindings}), problems


def _check_action(
    action: ActionSpec,
    base_dir: Path,
    is_valid_key: Callable[[str], bool],
    resolve_app: Callable[[str], str],
) -> tuple[ActionSpec, list[str]]:
    match action:
        case KeysAction():
            return action, [
                _key_issue(key, is_valid_key) for key in action.keys if not is_valid_key(key)
            ]
        case LaunchAction(app=str() as app):
            try:
                resolve_app(app)
            except AppResolutionError as exc:
                named = app in str(exc)  # lot C's messages already name the app
                return action, [f"launch app: {exc}" if named else f"launch app {app!r}: {exc}"]
            return action, []
        case LaunchAction(path=Path() as raw_path):
            target, issue = _resolve_launch_path(raw_path, base_dir)
            if issue is not None:
                return action, [issue]
            return action.model_copy(update={"path": target}), []
        case LaunchAction():
            return action, []  # app_id: nothing to check without starting the app
        case UrlAction():
            return action, []  # scheme and host checked by the model
        case ScriptAction():
            script = _absolute(action.path, base_dir)
            if not script.is_file():
                return action, [f"script not found: {script}"]
            return action.model_copy(update={"path": script}), []
        case _:
            assert_never(action)


def _key_issue(key: str, is_valid_key: Callable[[str], bool]) -> str:
    hint = ""
    if key != key.lower() and is_valid_key(key.lower()):
        hint = f" (key names are lowercase: {key.lower()!r})"
    return f"unknown key {key!r}{hint}; pyautogui would silently ignore it"


def _unknown_label(label: str, labels: Iterable[str]) -> str:
    known = ", ".join(sorted(set(labels) - {NONE_LABEL}))
    return f"unknown gesture label {label!r} (known: {known})"


def custom_model_path(config_path: str | os.PathLike[str]) -> Path:
    """Where `settings.recognition.custom_model` points, before the config is fully loaded.

    The custom model's labels must be known to validate the bindings, so the app reads this
    first. Tolerant on purpose: on any problem it returns the default location, and the real
    load reports the problem.
    """
    config_path = Path(config_path).absolute()
    default = RecognitionSettings.model_fields["custom_model"].default
    try:
        raw = _read_yaml(config_path)
        value = raw["settings"]["recognition"]["custom_model"]  # type: ignore[index]
        path = Path(value) if isinstance(value, str) else default
    except (ConfigError, KeyError, TypeError):
        path = default
    return _absolute(path, config_path.parent)


def _absolute(path: Path, base_dir: Path) -> Path:
    return Path(os.path.normpath(path if path.is_absolute() else base_dir / path))


def _resolve_launch_path(raw: Path, base_dir: Path) -> tuple[Path | None, str | None]:
    """Expand %VAR%, then resolve like the other paths; a bare name may also come from PATH.

    Decision for bare names such as `notepad.exe` (no folder part): the config folder is tried
    first, like every relative path, then the absolute PATH entries (with PATHEXT), never the
    current directory (see `_which_on_path`). The found path is stored absolute, so the launch
    handler gets the same absolute contract for every `path`. Programs registered only under
    "App Paths" (found by the Run box but not on PATH) are refused here: use `app:` (Start-menu
    name) or an absolute path for them.
    """
    text = os.path.expandvars(str(raw))
    undefined = _UNEXPANDED_VAR.search(text)
    if undefined is not None:
        return (
            None,
            f"launch path {str(raw)!r}: environment variable {undefined.group()} is not set",
        )
    candidate = Path(text)
    target = _absolute(candidate, base_dir)
    bare = not candidate.is_absolute() and len(candidate.parts) == 1
    if not target.exists() and bare:
        found = _which_on_path(text)
        if found is not None:
            target = found
    if not target.exists():
        where = " nor on PATH" if bare else ""
        return None, f"launch path not found: {target}{where}"
    return target, None


def _which_on_path(name: str) -> Path | None:
    """Look `name` up in the absolute PATH entries only, honouring PATHEXT.

    Plain `shutil.which(name)` on Windows searches the current directory first (even with an
    explicit `path=`), so a notepad.exe lying in the folder the app was started from would win
    over the one in system32. Giving `which` a directory part turns that search off; relative
    PATH entries (".", "bin") are skipped for the same reason.
    """
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        folder = Path(entry.strip('"'))
        if not entry or not folder.is_absolute():
            continue
        found = shutil.which(str(folder / name))
        if found is not None:
            return Path(found)
    return None


# --- hot reload -----------------------------------------------------------------------------


class _StatResult(Protocol):
    @property
    def st_mtime_ns(self) -> int: ...

    @property
    def st_size(self) -> int: ...


_Signature = tuple[int, int] | None
"""(mtime_ns, size) of the file, or None when it cannot be stat'ed (missing, being replaced)."""


@dataclass(frozen=True, slots=True)
class ConfigReload:
    """A new config replaced the previous one. The caller recreates the engine (in cooldown)."""

    config: Config
    restart_required: tuple[str, ...]
    """Sections among RESTART_SECTIONS that differ from the config loaded at startup: the camera
    and the model in use are the startup ones, whatever the reloads in between."""


class ConfigStore:
    """Holds the current config and reloads it when the file changes.

    `poll()` is cheap and meant to be called every frame: it stats the file at most once per
    `interval_s`. A change is loaded once the signature `(mtime_ns, size)` is the same on two
    consecutive checks, so an editor that writes in two steps (empty file, then content) is read
    once, complete, instead of producing a transient error or a half-empty config. The price is
    one to two seconds of reload latency.

    An invalid file keeps the previous config and is logged once per file state, not once per
    check. The first load happens in the constructor and raises ConfigError: without a previous
    config there is nothing to keep.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        load: Callable[[Path], Config],
        *,
        clock: Callable[[], float] = time.monotonic,
        stat: Callable[[Path], _StatResult] = os.stat,
        interval_s: float = 1.0,
    ) -> None:
        self._path = Path(path).absolute()
        self._load = load
        self._clock = clock
        self._stat = stat
        self._interval_s = interval_s
        signature = self._signature()
        self._config = load(self._path)
        self._startup = self._config
        self._last_seen: _Signature = signature
        self._last_attempt: _Signature = signature
        self._next_check = clock() + interval_s

    @property
    def path(self) -> Path:
        return self._path

    @property
    def config(self) -> Config:
        return self._config

    def poll(self) -> ConfigReload | None:
        """Return a ConfigReload when a changed, valid file replaced the config, else None."""
        now = self._clock()
        if now < self._next_check:
            return None
        self._next_check = now + self._interval_s

        signature = self._signature()
        settled = signature == self._last_seen
        self._last_seen = signature
        if not settled or signature == self._last_attempt:
            return None
        self._last_attempt = signature

        if signature is None:
            logger.error("Config file %s is missing; keeping the previous config.", self._path)
            return None
        try:
            new = self._load(self._path)
        except ConfigError as exc:
            logger.error("%s\nKeeping the previous config.", exc)
            return None
        except Exception:
            logger.exception("Reloading %s failed; keeping the previous config.", self._path)
            return None

        self._config = new
        restart = tuple(
            section
            for section in RESTART_SECTIONS
            if getattr(new.settings, section) != getattr(self._startup.settings, section)
        )
        if restart:
            logger.warning(
                "Config reloaded from %s; %s changed: restart required for those settings.",
                self._path,
                " and ".join(restart),
            )
        else:
            logger.info("Config reloaded from %s.", self._path)
        return ConfigReload(config=new, restart_required=restart)

    def _signature(self) -> _Signature:
        try:
            result = self._stat(self._path)
        except OSError:
            return None
        return (result.st_mtime_ns, result.st_size)
