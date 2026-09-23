"""System-tray mode: the app runs in the background, a coloured icon near the clock shows its state.

    green = armed, red = disarmed, grey = starting or stopped

Menu: show/hide the camera window (also a double-click on the icon), open config.yaml, open the
log, quit. pystray owns the main thread; the app runs on its own threads as usual.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from gesture_remote.app import App

logger = logging.getLogger(__name__)

ICON_SIZE = 64
ARMED_RGB = (60, 180, 90)
DISARMED_RGB = (215, 60, 60)
IDLE_RGB = (130, 130, 130)
REFRESH_S = 0.5


def icon_image(rgb: tuple[int, int, int]):  # -> PIL.Image.Image
    """A filled circle with a white ring and a small open-hand mark."""
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((2, 2, ICON_SIZE - 3, ICON_SIZE - 3), fill=rgb, outline="white", width=4)
    # palm and four fingers
    draw.rounded_rectangle((20, 30, 44, 50), radius=6, fill="white")
    for index in range(4):
        x = 20 + index * 6
        draw.rounded_rectangle((x, 14 + abs(1.5 - index) * 3, x + 4, 34), radius=2, fill="white")
    return image


def run_with_tray(app: App, config_path: Path, log_file: Path) -> int:
    import pystray

    result: list[int] = []
    worker = threading.Thread(target=lambda: result.append(app.run()), name="app", daemon=True)
    images = {state: icon_image(rgb) for state, rgb in
              (("armed", ARMED_RGB), ("disarmed", DISARMED_RGB), ("idle", IDLE_RGB))}  # fmt: skip

    def toggle_window(icon, item) -> None:
        app.pipeline.show_view = not app.pipeline.show_view

    def quit_app(icon, item) -> None:
        logger.info("tray: quit")
        app.stop.set()
        icon.stop()

    icon = pystray.Icon(
        "gesture-remote",
        images["idle"],
        "gesture-remote",
        menu=pystray.Menu(
            pystray.MenuItem(
                "Afficher la caméra",
                toggle_window,
                checked=lambda item: app.pipeline.show_view,
                default=True,
            ),
            pystray.MenuItem(
                "Modifier les gestes (config.yaml)", lambda: os.startfile(config_path)
            ),
            pystray.MenuItem("Ouvrir le journal", lambda: os.startfile(log_file)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quitter", quit_app),
        ),
    )

    def refresh(icon) -> None:
        icon.visible = True
        shown = None
        while worker.is_alive() and not app.stop.is_set():
            state = "armed" if app.pipeline.engine.armed else "disarmed"
            if state != shown:
                icon.icon = images[state]
                icon.title = f"gesture-remote : {'armé' if state == 'armed' else 'désarmé'}"
                shown = state
            app.stop.wait(REFRESH_S)
        icon.stop()  # the app ended on its own (error, Ctrl+C): take the icon down too

    worker.start()
    icon.run(setup=refresh)
    app.stop.set()
    worker.join()
    return result[0] if result else 1
