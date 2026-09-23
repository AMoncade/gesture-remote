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
    parser.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, log_dir: Path = LOG_DIR) -> int:
    args = parse_args(argv)
    # Imported after argument parsing: --help stays instant and works without mediapipe.
    from gesture_remote.app import build_app
    from gesture_remote.config import ConfigError
    from gesture_remote.logging_setup import setup_logging

    log_file = setup_logging(log_dir, verbose=args.verbose)
    logger.info("gesture-remote starting; log file %s", log_file)
    try:
        app = build_app(args.config, log_dir=log_dir, debug=args.debug, dry_run=args.dry_run)
    except ConfigError as error:  # first: ConfigError is a ValueError, like JSONDecodeError
        logger.error("%s", error)  # lists every problem, one per line
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
        return EXIT_ENVIRONMENT
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
