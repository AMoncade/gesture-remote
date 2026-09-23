"""`keys` actions: one chord per fire (repeat_while_held is the engine's job)."""

from __future__ import annotations

import logging
from types import ModuleType
from typing import Protocol

from gesture_remote.config import KeysAction

logger = logging.getLogger(__name__)


class KeyPresser(Protocol):
    def hotkey(self, *keys: str) -> None: ...


class PyAutoGuiPresser:
    """Real presser. Key names are validated by the config loader (`isValidKey`), not here."""

    def __init__(self, module: ModuleType | None = None) -> None:
        if module is None:
            import pyautogui as module
        # FAILSAFE would raise whenever the mouse rests in a screen corner; PAUSE adds 0.1 s
        # after every call.
        module.FAILSAFE = False
        module.PAUSE = 0
        self._module = module

    def hotkey(self, *keys: str) -> None:
        self._module.hotkey(*keys)


class KeysHandler:
    def __init__(self, presser: KeyPresser) -> None:
        self._presser = presser

    def __call__(self, action: KeysAction) -> None:
        logger.info("keys %s", "+".join(action.keys))
        self._presser.hotkey(*action.keys)
