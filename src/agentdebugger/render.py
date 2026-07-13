"""Terminal rendering for the CLI.

Colour is opt-out: disabled when stdout is not a TTY, when ``NO_COLOR`` is set,
or when ``TERM=dumb``. Nothing here is load-bearing — it is presentation only.
"""

from __future__ import annotations

import os
import sys
import textwrap

_RESET = "\033[0m"
_STYLES = {
    "dim": "\033[2m",
    "bold": "\033[1m",
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
}


def colour_enabled() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return sys.stdout.isatty()


def style(text: str, *names: str) -> str:
    """Wrap ``text`` in ANSI styles, or return it unchanged if colour is off."""
    if not colour_enabled() or not names:
        return text
    codes = "".join(_STYLES[name] for name in names)
    return f"{codes}{text}{_RESET}"


def heading(text: str) -> str:
    return style(text, "bold", "cyan")


def field(label: str, value: str, width: int = 12, indent: int = 5) -> str:
    """A label/value line, with the value wrapped and aligned under itself."""
    pad = " " * indent
    wrapped = textwrap.wrap(value, width=max(30, _terminal_width() - indent - width - 1)) or [""]
    lines = [f"{pad}{style(label.ljust(width), 'dim')}{wrapped[0]}"]
    lines.extend(f"{pad}{' ' * width}{line}" for line in wrapped[1:])
    return "\n".join(lines)


def verdict(ok: bool, yes: str = "yes", no: str = "no") -> str:
    return style(yes, "green") if ok else style(no, "red")


def signed(value: float) -> str:
    text = f"{value:+.2f}"
    if value > 0:
        return style(text, "green")
    if value < 0:
        return style(text, "red")
    return style(text, "dim")


def bar(passed: int, total: int, width: int = 10) -> str:
    """A pass/fail bar: `██████░░░░ 6/8`."""
    if total <= 0:
        return style("n/a", "dim")
    filled = round(width * passed / total)
    colour = "green" if passed == total else "yellow" if passed else "red"
    return f"{style('█' * filled, colour)}{style('░' * (width - filled), 'dim')} {passed}/{total}"


def _terminal_width() -> int:
    try:
        return min(os.get_terminal_size().columns, 100)
    except OSError:
        return 80
