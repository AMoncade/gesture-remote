"""Macro: play an animated GIF full screen. A click, any key, or 10 s closes it.

    fullscreen_gif.py                  macros/media/cat_dance.gif
    fullscreen_gif.py other.gif        a path relative to this folder, or absolute

Bound to thumb_up in config.yaml. Output goes to logs/scripts/fullscreen_gif.log.
"""

import contextlib
import ctypes
import sys
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageSequence, ImageTk

HERE = Path(__file__).resolve().parent
DEFAULT_GIF = HERE / "media" / "cat_dance.gif"
CLOSE_AFTER_MS = 10_000
BACKGROUND = "black"  # until the GIF is loaded; then the colour of its top-left pixel


def load_frames(path: Path, screen: tuple[int, int]) -> tuple[list[Image.Image], list[int]]:
    """Every frame scaled to fit the screen, and each frame's duration in ms."""
    width, height = screen
    frames, durations = [], []
    with Image.open(path) as gif:
        ratio = min(width / gif.width, height / gif.height)
        size = (round(gif.width * ratio), round(gif.height * ratio))
        for frame in ImageSequence.Iterator(gif):
            frames.append(frame.convert("RGBA").resize(size, Image.Resampling.LANCZOS))
            durations.append(max(int(frame.info.get("duration", 50)), 20))
    return frames, durations


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GIF
    path = path if path.is_absolute() else HERE / path
    if not path.is_file():
        print(f"fullscreen_gif: {path} not found")
        return 1

    with contextlib.suppress(AttributeError, OSError):  # real pixels on a scaled screen
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    root = tk.Tk()
    root.configure(bg=BACKGROUND, cursor="none")
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    screen = (root.winfo_screenwidth(), root.winfo_screenheight())
    frames, durations = load_frames(path, screen)
    red, green, blue, _ = frames[0].getpixel((0, 0))  # fill the screen with the GIF's own edge
    background = f"#{red:02x}{green:02x}{blue:02x}"
    root.configure(bg=background)
    images = [ImageTk.PhotoImage(frame) for frame in frames]
    label = tk.Label(root, bg=background, bd=0)
    label.pack(expand=True)

    def play(index: int = 0) -> None:
        label.configure(image=images[index])
        root.after(durations[index], play, (index + 1) % len(images))

    for event in ("<Button>", "<Key>"):
        root.bind(event, lambda _event: root.destroy())
    root.after(CLOSE_AFTER_MS, root.destroy)
    root.focus_force()
    print(f"fullscreen_gif: playing {path.name} ({len(images)} frames) on {screen}")
    play()
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
