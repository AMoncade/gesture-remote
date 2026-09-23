"""Audible feedback: rising tones when armed, falling when disarmed, one beep per fire.

Ignored segments stay silent on purpose: no beeping during a call while disarmed.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

Tone = tuple[int, int]
"""(frequency Hz, duration ms)."""
Beeper = Callable[[Sequence[Tone]], None]
"""Plays tones in order; must return without waiting for them."""

ARMED_TONES: tuple[Tone, ...] = ((660, 90), (990, 90))
DISARMED_TONES: tuple[Tone, ...] = ((990, 90), (660, 90))
FIRE_TONES: tuple[Tone, ...] = ((880, 60),)


class ThreadedBeeper:
    """Plays tone sequences one after another on a daemon thread (winsound.Beep blocks)."""

    def __init__(self, beep: Callable[[int, int], None] | None = None) -> None:
        if beep is None:
            import winsound

            beep = winsound.Beep
        self._beep = beep
        self._queue: queue.SimpleQueue[Sequence[Tone]] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def __call__(self, tones: Sequence[Tone]) -> None:
        self._queue.put(tuple(tones))
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._play, name="beeper", daemon=True)
                self._thread.start()

    def _play(self) -> None:
        while True:
            for frequency, duration_ms in self._queue.get():
                try:
                    self._beep(frequency, duration_ms)
                except RuntimeError:
                    logger.warning("beep failed (no sound device?)", exc_info=True)


class Feedback:
    def __init__(self, sound: bool, beeper: Beeper | None = None) -> None:
        self.sound = sound
        """Mutable: a config reload may switch sound on or off."""
        self._beeper = beeper
        self._lock = threading.Lock()

    def armed_changed(self, armed: bool) -> None:
        self._play(ARMED_TONES if armed else DISARMED_TONES)

    def triggered(self) -> None:
        self._play(FIRE_TONES)

    def ignored(self) -> None:
        """Deliberately silent (disarmed or cooldown)."""

    def _play(self, tones: Sequence[Tone]) -> None:
        if not self.sound:
            return
        with self._lock:
            if self._beeper is None:
                self._beeper = ThreadedBeeper()
            beeper = self._beeper
        beeper(tones)
