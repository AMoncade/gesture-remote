"""StartAppsIndex: Get-StartApps parsing, resolution rules, session cache."""

from __future__ import annotations

import json
import shutil
import sys

import pytest

from gesture_remote.actions.start_apps import (
    StartApp,
    StartAppsIndex,
    parse_start_apps,
    run_get_start_apps,
)
from gesture_remote.config import AppResolutionError

APPLE_MUSIC_ID = "AppleInc.AppleMusicWin_nzyj5cx40ttqa!App"


def as_json(entries: list[tuple[str, str]], bom: bool = False) -> bytes:
    """Mimic `ConvertTo-Json -Compress`: a bare object when there is a single entry."""
    items = [{"Name": name, "AppID": app_id} for name, app_id in entries]
    payload = items[0] if len(items) == 1 else items
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return b"\xef\xbb\xbf" + raw if bom else raw


class ScriptedRunner:
    """Returns the queued outputs in order (the last one forever) and counts calls."""

    def __init__(self, *outputs: bytes) -> None:
        self.outputs = list(outputs)
        self.calls = 0

    def __call__(self) -> bytes:
        self.calls += 1
        return self.outputs[min(self.calls, len(self.outputs)) - 1]


# --- parsing --------------------------------------------------------------------------------


def test_parses_a_list() -> None:
    raw = as_json([("Apple Music", APPLE_MUSIC_ID), ("Calculator", "calc!App")])
    assert parse_start_apps(raw) == [
        StartApp("Apple Music", APPLE_MUSIC_ID),
        StartApp("Calculator", "calc!App"),
    ]


def test_parses_a_single_object() -> None:
    raw = as_json([("Apple Music", APPLE_MUSIC_ID)])
    assert raw.startswith(b"{")
    assert parse_start_apps(raw) == [StartApp("Apple Music", APPLE_MUSIC_ID)]


def test_parses_output_preceded_by_a_bom() -> None:
    raw = as_json([("Apple Music", APPLE_MUSIC_ID), ("Calculator", "calc!App")], bom=True)
    assert len(parse_start_apps(raw)) == 2


def test_keeps_accented_names() -> None:
    raw = as_json([("Paramètres", "windows.immersivecontrolpanel"), ("Intel® Graphics", "i!App")])
    names = [app.name for app in parse_start_apps(raw)]
    assert names == ["Paramètres", "Intel® Graphics"]


def test_empty_output_means_no_apps() -> None:
    assert parse_start_apps(b"") == []
    assert parse_start_apps(b"\xef\xbb\xbf\r\n") == []


def test_skips_entries_without_name_or_app_id() -> None:
    raw = json.dumps([{"Name": "x", "AppID": None}, {"Name": "ok", "AppID": "ok!App"}]).encode()
    assert parse_start_apps(raw) == [StartApp("ok", "ok!App")]


# --- resolution -----------------------------------------------------------------------------


def test_exact_name() -> None:
    index = StartAppsIndex(ScriptedRunner(as_json([("Apple Music", APPLE_MUSIC_ID)])))
    assert index.resolve("Apple Music") == APPLE_MUSIC_ID


def test_case_insensitive_name() -> None:
    index = StartAppsIndex(ScriptedRunner(as_json([("Apple Music", APPLE_MUSIC_ID), ("B", "b")])))
    assert index.resolve("apple MUSIC") == APPLE_MUSIC_ID


def test_accented_name_resolves() -> None:
    index = StartAppsIndex(ScriptedRunner(as_json([("Paramètres", "settings!App"), ("B", "b")])))
    assert index.resolve("paramÈtres") == "settings!App"


def test_exact_match_wins_over_case_insensitive_ones() -> None:
    index = StartAppsIndex(
        ScriptedRunner(as_json([("Notes", "upper!App"), ("notes", "lower!App")]))
    )
    assert index.resolve("notes") == "lower!App"
    with pytest.raises(AppResolutionError, match="ambiguous"):
        index.resolve("NOTES")


def test_ambiguous_name_lists_the_app_ids() -> None:
    index = StartAppsIndex(
        ScriptedRunner(as_json([("Calculator", "one!App"), ("Calculator", "two!App")]))
    )
    with pytest.raises(AppResolutionError) as error:
        index.resolve("Calculator")
    assert "one!App" in str(error.value)
    assert "two!App" in str(error.value)


def test_duplicate_entry_with_the_same_app_id_is_not_ambiguous() -> None:
    index = StartAppsIndex(ScriptedRunner(as_json([("Calculator", "one!App")] * 2)))
    assert index.resolve("Calculator") == "one!App"


def test_missing_name_suggests_close_names() -> None:
    entries = [("Apple Music", APPLE_MUSIC_ID), ("Calculator", "calc!App"), ("Paint", "p!App")]
    index = StartAppsIndex(ScriptedRunner(as_json(entries)))
    with pytest.raises(AppResolutionError) as error:
        index.resolve("Aple Music")
    message = str(error.value)
    assert "'Apple Music'" in message
    assert "Paint" not in message


def test_missing_name_suggests_names_that_contain_it() -> None:
    entries = [("Apple Music Preview", "a!App"), ("Calculator", "calc!App")]
    index = StartAppsIndex(ScriptedRunner(as_json(entries)))
    with pytest.raises(AppResolutionError, match="Apple Music Preview"):
        index.resolve("music")


# --- cache ----------------------------------------------------------------------------------


def test_list_is_cached_for_the_session() -> None:
    runner = ScriptedRunner(as_json([("Apple Music", APPLE_MUSIC_ID), ("Calculator", "c!App")]))
    index = StartAppsIndex(runner)
    index.resolve("Apple Music")
    index.resolve("Calculator")
    index.resolve("apple music")
    assert runner.calls == 1


def test_missing_name_reads_the_list_again_once() -> None:
    before = as_json([("Calculator", "c!App"), ("Paint", "p!App")])
    after = as_json([("Calculator", "c!App"), ("Apple Music", APPLE_MUSIC_ID)])
    runner = ScriptedRunner(before, after)
    index = StartAppsIndex(runner)
    assert index.resolve("Calculator") == "c!App"
    assert index.resolve("Apple Music") == APPLE_MUSIC_ID
    assert runner.calls == 2


def test_still_missing_after_one_reload_raises_without_more_reads() -> None:
    runner = ScriptedRunner(as_json([("Calculator", "c!App"), ("Paint", "p!App")]))
    index = StartAppsIndex(runner)
    with pytest.raises(AppResolutionError, match="not found"):
        index.resolve("Apple Music")
    assert runner.calls == 2


# --- real Get-StartApps (read-only) ---------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    sys.platform != "win32" or shutil.which("powershell") is None,
    reason="needs Windows PowerShell",
)
def test_real_get_start_apps_parses() -> None:
    apps = parse_start_apps(run_get_start_apps())
    assert len(apps) > 1
    assert all(app.name and app.app_id for app in apps)
    # Encoding sanity: mojibake of a UTF-8 multi-byte char decoded as cp1252 starts with 'Ã'/'Â'.
    assert not [app.name for app in apps if "Ã" in app.name or "Â" in app.name]
