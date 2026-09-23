"""A small always-on-top card listing the gestures you can use, closed with its × button.

Shown when the tray app starts and from the tray menu ("Gestes disponibles"). The rows come from
the live config (`help_rows`, pure and tested); the window itself is plain tkinter on its own
thread, one card at a time.
"""

from __future__ import annotations

import logging
import threading
from typing import assert_never
from urllib.parse import urlsplit

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


def show_help(config: Config) -> None:
    """Open the card on its own thread; does nothing if one is already open."""
    if not _open.acquire(blocking=False):
        return
    rows = help_rows(config)

    def run() -> None:
        try:
            _window(rows)
        except Exception:  # a broken popup must never take the app down
            logger.exception("gestures popup failed")
        finally:
            _open.release()

    threading.Thread(target=run, name="help-popup", daemon=True).start()


def _window(rows: list[tuple[str, str]]) -> None:
    import ctypes
    import tkinter as tk

    # Without this Tk draws at 96 DPI and Windows stretches it: blurry on a scaled screen.
    # Per thread, so the camera window and the tray icon are left as they are.
    try:
        ctypes.windll.user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))  # per-monitor v2
    except (AttributeError, OSError):
        pass

    root = tk.Tk()
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
        text="Tiens le geste ~1 s (✌️ et 🤟 un peu plus).\nIcône : près de l'horloge, flèche ^.",
        justify="left",
        bg=BACKGROUND,
        fg=DIM_TEXT,
        font=("Segoe UI", 9),
    ).pack(padx=12, pady=(6, 10), anchor="w")

    root.update_idletasks()
    width = max(round(WIDTH * scale), root.winfo_reqwidth())
    height = root.winfo_reqheight()
    x = root.winfo_screenwidth() - width - round(16 * scale)
    y = root.winfo_screenheight() - height - round(64 * scale)  # above the taskbar
    root.geometry(f"{width}x{height}+{x}+{y}")
    root.mainloop()
