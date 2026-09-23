"""`url` actions: a new tab in the default browser."""

from __future__ import annotations

import logging
import webbrowser
from collections.abc import Callable

from gesture_remote.config import UrlAction

logger = logging.getLogger(__name__)

OpenTab = Callable[[str], object]


class UrlHandler:
    def __init__(self, open_tab: OpenTab = webbrowser.open_new_tab) -> None:
        self._open_tab = open_tab

    def __call__(self, action: UrlAction) -> None:
        logger.info("url %s", action.url)
        if self._open_tab(action.url) is False:
            logger.warning("no browser accepted %s", action.url)
