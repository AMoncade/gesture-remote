"""python -m gesture_remote [--config PATH] [--debug] [--dry-run] [-v]"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"
LOG_DIR = REPO_ROOT / "logs"

EXIT_OK, EXIT_FAILED, EXIT_BAD_CONFIG, EXIT_ENVIRONMENT = 0, 1, 2, 3

logger = logging.getLogger("gesture_remote")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m gesture_remote",
        description="Webcam hand gestures -> actions, 100 % local. Ctrl+C to quit.",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help=f"default: {DEFAULT_CONFIG}"
    )
    parser.add_argument(
        "--debug", action="store_true", help="show the camera overlay (q or Esc closes it)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="log the actions instead of running them"
    )
    parser.add_argument(
        "--tray",
        action="store_true",
        help="run in the background with an icon near the clock (use pythonw for no console)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")
    return parser.parse_args(argv)


def show_error(message: str) -> None:
    """Without a console (pythonw), a startup error would vanish: show it in a message box."""
    if sys.stderr is not None:
        return
    import ctypes

    MB_ICONERROR = 0x10
    ctypes.windll.user32.MessageBoxW(None, message, "gesture-remote", MB_ICONERROR)


def main(argv: Sequence[str] | None = None, *, log_dir: Path = LOG_DIR) -> int:
    args = parse_args(argv)
    # Imported after argument parsing: --help stays instant and works without mediapipe.
    from gesture_remote.app import build_app
    from gesture_remote.config import ConfigError
    from gesture_remote.logging_setup import setup_logging

    # pythonw has no console: sys.stderr is None there, so log to the file only.
    log_file = setup_logging(log_dir, verbose=args.verbose, console=sys.stderr is not None)
    logger.info("gesture-remote starting; log file %s", log_file)
    try:
        app = build_app(
            args.config,
            log_dir=log_dir,
            debug=args.debug,
            dry_run=args.dry_run,
            tray=args.tray,
        )
    except ConfigError as error:  # first: ConfigError is a ValueError, like JSONDecodeError
        logger.error("%s", error)  # lists every problem, one per line
        show_error(f"config.yaml refusé :\n\n{error}")
        return EXIT_BAD_CONFIG
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as error:
        # Not a ConfigError: resolving a Start-menu app runs PowerShell (Get-StartApps), which
        # can fail, time out or print something that is not JSON while the config is validated.
        logger.error(
            "startup failed while validating %s (Get-StartApps or file access): %s: %s",
            args.config,
            type(error).__name__,
            error,
        )
        show_error(f"Démarrage impossible : {type(error).__name__}: {error}")
        return EXIT_ENVIRONMENT
    if args.tray:
        from gesture_remote.tray import run_with_tray

        return run_with_tray(app, args.config, log_file)
    return app.run()


MUTEX_NAME = "Local\\gesture-remote"
ERROR_ALREADY_EXISTS = 183


def already_running() -> bool:
    """True when another gesture-remote holds the per-user mutex.

    Two copies fight over the webcam: the second one only gets black frames. The mutex is held
    until this process exits (the handle is deliberately never closed).
    """
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW(None, False, MUTEX_NAME)
    return ctypes.get_last_error() == ERROR_ALREADY_EXISTS


if __name__ == "__main__":
    if already_running():
        message = "gesture-remote is already running: look for its icon near the clock."
        show_error("gesture-remote tourne déjà : son icône est près de l'horloge (flèche ^).")
        if sys.stderr is not None:
            print(message, file=sys.stderr)
        sys.exit(EXIT_ENVIRONMENT)
    sys.exit(main())
