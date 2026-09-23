# gesture-remote

A Windows background app that watches the webcam and turns hand gestures into actions:
open an app, open a web page, run one of your Python macros, press keys (media keys included).
100 % local and free: no cloud, no API key. Webcam frames stay in memory and are never saved or logged.

Personal learning project (computer vision + ML). Architecture, conventions and progress: see `CLAUDE.md`.

## Setup (Windows, Python 3.13)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python tools\download_models.py
.\.venv\Scripts\python tools\check_setup.py
```

Always use `.venv\Scripts\python`: on this machine, a bare `python` is the Microsoft Store alias.
To reproduce the exact tested versions: `.\.venv\Scripts\python -m pip install -r requirements.lock`.

## Run

```powershell
.\.venv\Scripts\python -m gesture_remote --debug --dry-run   # debug window, actions only logged
.\.venv\Scripts\python -m gesture_remote                     # for real
```

Gestures are mapped in `config.yaml`, which is reloaded automatically when saved.
🤟 (`i_love_you`) arms or disarms the remote.

### In the background (tray icon, no console)

```powershell
.\.venv\Scripts\python tools\make_shortcut.py             # "gesture-remote" shortcut on the Desktop
.\.venv\Scripts\python tools\make_shortcut.py --startup   # ...and start with Windows
.\.venv\Scripts\python tools\make_shortcut.py --remove    # delete both shortcuts
```

The shortcut runs `pythonw -m gesture_remote --tray`. The icon near the clock is green when armed,
red when disarmed. Its menu: show the camera window (or double-click the icon), edit
`config.yaml`, open the log, quit. A startup error is shown in a message box.

## Macros

A macro is a Python script in `macros/`, bound with `{ type: script, path: macros/my_macro.py }`.
It runs with the project's Python in its own process, without a window; its `print()` output goes to
`logs/scripts/<name>.log`. Only one instance of a given script runs at a time.

To find the Start-menu name or AppID of an app for `{ type: launch, app: "..." }`:
`.\.venv\Scripts\python tools\check_setup.py --apps music`.

## Tests

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m ruff check .
```
