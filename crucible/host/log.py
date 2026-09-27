from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from ..platform.paths import LOG_ROLL_BYTES


_ASCII_SPELLING = str.maketrans(
    {
        "—": "-",
        "–": "-",
        "…": "...",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "→": "->",
        "←": "<-",
        "×": "x",
        "≥": ">=",
        "≤": "<=",
        " ": " ",
        "\x00": "",
    }
)


def plain(text: str) -> str:
    return text.translate(_ASCII_SPELLING).encode("ascii", "replace").decode("ascii")


def timestamp(now: float | None = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))


class HostLog:
    def __init__(
        self,
        path: Path,
        previous: Path,
        *,
        roll_bytes: int = LOG_ROLL_BYTES,
        clock: Callable[[], float] = time.time,
        echo: Callable[[str], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self.previous = Path(previous)
        self.roll_bytes = roll_bytes
        self._clock = clock
        self._echo = echo

    def write(self, message: str) -> str:
        line = plain(f"{timestamp(self._clock())} {message}")
        self._roll_if_needed()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="ascii", newline="\n") as handle:
            handle.write(line + "\n")
        if self._echo is not None:
            self._echo(line)
        return line

    def _roll_if_needed(self) -> None:
        if not self.path.exists():
            return
        if self.path.stat().st_size < self.roll_bytes:
            return
        if self.previous.exists():
            self.previous.unlink()
        self.path.replace(self.previous)
