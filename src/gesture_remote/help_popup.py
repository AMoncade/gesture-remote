"""A small always-on-top card listing the gestures you can use, closed with its × button.

Shown when the tray app starts and from the tray menu ("Gestes disponibles"). The rows come from
the live config (`help_rows`, pure and tested); the window itself is plain tkinter on its own
thread, one card at a time.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import Protocol, assert_never
from urllib.parse import urlsplit

import numpy as np

from gesture_remote.config import (
    ActionSpec,
    Config,
    KeysAction,
    LaunchAction,
    QuitAction,
    ScriptAction,
    UrlAction,
)

logger = logging.getLogger(__name__)

SYMBOLS: dict[str, str] = {
    "open_palm": "✋",
    "victory": "✌️",
    "pointing_up": "☝️",
    "closed_fist": "✊",
    "thumb_up": "👍",
    "thumb_down": "👎",
    "i_love_you": "🤟",
    "call_me": "🤙",
    "ok": "👌",
    "trois": "3️⃣",
}
"""Label -> symbol; an unknown (new custom) label shows as its name."""

BACKGROUND = "#1e1f24"
ROW_BACKGROUND = "#2a2c33"
TEXT = "#f2f2f2"
DIM_TEXT = "#a0a3ab"
ACCENT = "#3cb45a"
WIDTH = 330

_open = threading.Lock()
"""Held while a card is on screen: a second request is ignored instead of stacking cards."""


def describe_action(action: ActionSpec) -> str:
    if action.label:
        return action.label
    match action:
        case KeysAction():
            return "Touche " + " + ".join(action.keys)
        case LaunchAction(app=str() as app):
            return f"Ouvre {app}"
        case LaunchAction(path=path) if path is not None:
            return f"Ouvre {path.name}"
        case LaunchAction():
            return "Ouvre une app"
        case UrlAction():
            host = urlsplit(action.url).netloc.removeprefix("www.")
            return f"Ouvre {host}"
        case ScriptAction():
            return f"Macro {action.path.stem}"
        case QuitAction():
            return "Ferme gesture-remote"
        case _:
            assert_never(action)


def help_rows(config: Config) -> list[tuple[str, str]]:
    """(symbol, what it does), the arm gesture first, then the bindings in file order."""
    rows = []
    arm = config.settings.engine.arm_gesture
    if arm is not None:
        rows.append((SYMBOLS.get(arm, arm), "Armer / désarmer"))
    for label, action in config.bindings.items():
        rows.append((SYMBOLS.get(label, label), describe_action(action)))
    return rows


class PreviewSource(Protocol):
    """What the popup needs from the pipeline to show the camera (app.Pipeline has both)."""

    preview_wanted: bool
    latest_preview: np.ndarray | None


def show_help(config: Config, preview: PreviewSource | None = None) -> None:
    """Open the card on its own thread; does nothing if one is already open.

    With `preview`, the card has a "Voir la caméra" button showing the live debug image.
    """
    if not _open.acquire(blocking=False):
        return
    rows = help_rows(config)

    def run() -> None:
        try:
            _window(rows, preview)
        except Exception:  # a broken popup must never take the app down
            logger.exception("gestures popup failed")
        finally:
            if preview is not None:
                preview.preview_wanted = False  # stop drawing frames nobody looks at
            _open.release()

    threading.Thread(target=run, name="help-popup", daemon=True).start()


def _window(rows: list[tuple[str, str]], preview: PreviewSource | None) -> None:
    import ctypes
    import tkinter as tk

    # Without this Tk draws at 96 DPI and Windows stretches it: blurry on a scaled screen.
    # Per thread, so the camera window and the tray icon are left as they are.
    with contextlib.suppress(AttributeError, OSError):
        ctypes.windll.user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))  # per-monitor v2

    root = tk.Tk()
    # Tk prints callback errors to stderr, which pythonw does not have: send them to the log.
    root.report_callback_exception = lambda *exc: logger.error(
        "gestures popup callback failed", exc_info=exc
    )
    scale = root.winfo_fpixels("1i") / 96  # fonts are in points; pixel sizes must follow
    root.overrideredirect(True)  # no title bar: our own × closes it
    root.attributes("-topmost", True)
    root.configure(bg=BACKGROUND, highlightthickness=1, highlightbackground=ACCENT)

    header = tk.Frame(root, bg=BACKGROUND)
    header.pack(fill="x", padx=12, pady=(10, 4))
    tk.Label(
        header, text="Gestes disponibles", bg=BACKGROUND, fg=TEXT, font=("Segoe UI", 12, "bold")
    ).pack(side="left")
    close = tk.Label(
        header, text="✕", bg=BACKGROUND, fg=DIM_TEXT, font=("Segoe UI", 12), cursor="hand2"
    )
    close.pack(side="right")
    close.bind("<Button-1>", lambda _event: root.destroy())
    close.bind("<Enter>", lambda _event: close.configure(fg=TEXT))
    close.bind("<Leave>", lambda _event: close.configure(fg=DIM_TEXT))

    for symbol, text in rows:
        row = tk.Frame(root, bg=ROW_BACKGROUND)
        row.pack(fill="x", padx=12, pady=2)
        tk.Label(
            row, text=symbol, bg=ROW_BACKGROUND, fg=TEXT, font=("Segoe UI Emoji", 16), width=3
        ).pack(side="left", padx=(4, 6), pady=3)
        tk.Label(
            row, text=text, bg=ROW_BACKGROUND, fg=TEXT, font=("Segoe UI", 10), anchor="w"
        ).pack(side="left", fill="x", expand=True)

    tk.Label(
        root,
        text="Tiens le geste ~1 s (fermer et armer : un peu plus).\n"
        "Icône : près de l'horloge, flèche ^.",
        justify="left",
        bg=BACKGROUND,
        fg=DIM_TEXT,
        font=("Segoe UI", 9),
    ).pack(padx=12, pady=(6, 10), anchor="w")

    root.update_idletasks()
    width = max(round(WIDTH * scale), root.winfo_reqwidth())

    def place() -> None:
        """Bottom-right corner, above the taskbar; called again when the height changes."""
        root.update_idletasks()
        height = root.winfo_reqheight()
        x = root.winfo_screenwidth() - width - round(16 * scale)
        y = root.winfo_screenheight() - height - round(64 * scale)
        root.geometry(f"{width}x{height}+{x}+{y}")

    if preview is not None:
        _camera_preview(root, preview, width - round(24 * scale), place)
    place()
    root.mainloop()


def _camera_preview(root, preview: PreviewSource, image_width: int, relayout) -> None:
    """A button that shows or hides the live debug image under the list (~15 fps)."""
    import tkinter as tk

    from PIL import Image, ImageTk

    image_height = image_width * 3 // 4
    # A black image of the final size from the start: a Label without an image measures
    # width/height in characters, which made the card thousands of pixels high (off screen)
    # until the first frame arrived.
    placeholder = ImageTk.PhotoImage(Image.new("RGB", (image_width, image_height), "black"))
    screen = tk.Label(root, bg="black", bd=0, image=placeholder)
    shown: list = [placeholder]  # keeps the PhotoImage alive: Tk only holds a weak reference

    def refresh() -> None:
        if not preview.preview_wanted:
            return
        frame = preview.latest_preview
        if frame is not None:
            rgb = np.ascontiguousarray(frame[..., ::-1])  # BGR -> RGB, memory only
            picture = Image.fromarray(rgb).resize((image_width, image_height))
            shown[0] = ImageTk.PhotoImage(picture)
            screen.configure(image=shown[0])
        root.after(66, refresh)

    def toggle() -> None:
        preview.preview_wanted = not preview.preview_wanted
        if preview.preview_wanted:
            button.configure(text="📷  Cacher la caméra")
            shown[0] = placeholder
            screen.configure(image=placeholder)
            screen.pack(padx=12, pady=(0, 10), before=button)
            refresh()
        else:
            button.configure(text="📷  Voir la caméra")
            screen.pack_forget()
            preview.latest_preview = None
        relayout()

    button = tk.Label(
        root,
        text="📷  Voir la caméra",
        bg=ROW_BACKGROUND,
        fg=TEXT,
        font=("Segoe UI", 10),
        cursor="hand2",
        pady=5,
    )
    button.pack(fill="x", padx=12, pady=(0, 10))
    button.bind("<Button-1>", lambda _event: toggle())
    button.bind("<Enter>", lambda _event: button.configure(bg=ACCENT))
    button.bind("<Leave>", lambda _event: button.configure(bg=ROW_BACKGROUND))
