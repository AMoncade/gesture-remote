"""The gestures popup's rows come from the config: symbols, labels, made-up descriptions."""

from __future__ import annotations

from pathlib import Path

import pytest

from gesture_remote.config import (
    Config,
    KeysAction,
    LaunchAction,
    QuitAction,
    ScriptAction,
    UrlAction,
)
from gesture_remote.help_popup import describe_action, help_rows

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("action", "text"),
    [
        (KeysAction(type="keys", keys=["ctrl", "w"]), "Touche ctrl + w"),
        (LaunchAction(type="launch", app="Apple Music"), "Ouvre Apple Music"),
        (LaunchAction(type="launch", path=Path("C:/x/notepad.exe")), "Ouvre notepad.exe"),
        (LaunchAction(type="launch", app_id="A!B"), "Ouvre une app"),
        (UrlAction(type="url", url="https://www.youtube.com/x"), "Ouvre youtube.com"),
        (ScriptAction(type="script", path=Path("macros/hello.py")), "Macro hello"),
        (QuitAction(type="quit"), "Ferme gesture-remote"),
        (QuitAction(type="quit", label="Bye"), "Bye"),
    ],
)
def test_every_action_type_is_described(action, text: str) -> None:
    assert describe_action(action) == text


def test_rows_start_with_the_arm_gesture_then_follow_the_file() -> None:
    config = Config.model_validate(
        {
            "bindings": {
                "open_palm": {"type": "url", "url": "https://studium.umontreal.ca"},
                "my_new_sign": {"type": "quit"},
            }
        }
    )
    assert help_rows(config) == [
        ("🤟", "Armer / désarmer"),
        ("✋", "Ouvre studium.umontreal.ca"),
        ("my_new_sign", "Ferme gesture-remote"),  # no symbol known: its name
    ]


def test_the_shipped_config_is_described_with_its_labels() -> None:
    import yaml

    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    rows = dict(help_rows(Config.model_validate(raw)))
    assert rows["✋"] == "Ouvre StudiUM"
    assert rows["🤙"] == "4 terminaux Claude + app Claude"
