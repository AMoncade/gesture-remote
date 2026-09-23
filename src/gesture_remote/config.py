"""Configuration schema: pydantic models for config.yaml.

Round-1 contract (written by the admin in step 0):
- the public names, fields, defaults and model validators below are frozen for the parallel round;
- lot B owns this file afterwards and adds loading, validation and ConfigStore;
- context-dependent checks (known labels, key names, Start-menu apps, file existence, phase-3
  forms) belong in the loader, never in model validators: other lots build these models directly
  in their tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
