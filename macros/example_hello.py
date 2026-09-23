"""Example macro: shows a Windows message box, to prove the gesture -> script chain works.

Macros run with the project's Python, in their own process and without a console window.
Anything printed goes to logs/scripts/example_hello.log.
"""

import ctypes
import sys

MB_ICONINFORMATION = 0x00000040
MB_SETFOREGROUND = 0x00010000
MB_TOPMOST = 0x00040000


def main() -> int:
    print("example_hello: showing the message box")
    ctypes.windll.user32.MessageBoxW(
        None,
        "Macro OK: the gesture ran macros/example_hello.py",
        "gesture-remote",
        MB_ICONINFORMATION | MB_SETFOREGROUND | MB_TOPMOST,
    )
    print("example_hello: closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
