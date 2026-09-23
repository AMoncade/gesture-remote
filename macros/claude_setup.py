"""Macro: a new Windows Terminal window with 4 tabs running `claude`, then the Claude app.

Bound to call_me (thumb + pinky) in config.yaml. Output goes to logs/scripts/claude_setup.log.
"""

import os
import subprocess
import sys

TABS = 4
FOLDER = os.path.expanduser("~")
CLAUDE_APP_ID = "Claude_pzs8sxrjxfjjc!Claude"  # tools/check_setup.py --apps claude


# `claude` is an npm shim (claude.cmd): only cmd resolves it through PATHEXT, wt alone does not.
SESSION = ["cmd", "/k", "claude"]


def terminal_command() -> list[str]:
    """wt -w new -d FOLDER cmd /k claude ; new-tab -d FOLDER cmd /k claude ; ... (one per tab)."""
    command = ["wt", "-w", "new", "-d", FOLDER, *SESSION]
    for _ in range(TABS - 1):
        command += [";", "new-tab", "-d", FOLDER, *SESSION]
    return command


def main() -> int:
    print("claude_setup: opening", TABS, "terminal tabs with claude")
    subprocess.Popen(terminal_command())
    print("claude_setup: opening the Claude app")
    subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{CLAUDE_APP_ID}"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
