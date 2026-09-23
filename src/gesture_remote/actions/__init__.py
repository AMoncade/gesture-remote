"""Actions run on gesture fires: keys, launch (+ Start-menu index), url, script."""

from gesture_remote.actions.dispatcher import (
    ACTION_TYPES,
    ActionDispatcher,
    build_dispatcher,
    default_handlers,
)
from gesture_remote.actions.keys import KeysHandler, PyAutoGuiPresser
from gesture_remote.actions.launch import LaunchHandler
from gesture_remote.actions.script import ScriptRunner
from gesture_remote.actions.start_apps import StartAppsIndex
from gesture_remote.actions.url import UrlHandler

__all__ = [
    "ACTION_TYPES",
    "ActionDispatcher",
    "KeysHandler",
    "LaunchHandler",
    "PyAutoGuiPresser",
    "ScriptRunner",
    "StartAppsIndex",
    "UrlHandler",
    "build_dispatcher",
    "default_handlers",
]
