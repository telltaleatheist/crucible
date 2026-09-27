"""`%LOCALAPPDATA%\\Crucible\\host.log` — PHASE15-HOST.md 4.1.

The tray has no console (4.1: the Startup shortcut points at `pythonw.exe`), so
the log is the ONLY place a failure is written down, and "engine did not start —
open the log" is a menu item that has to lead somewhere useful.

Rolled at 2 MiB and never deleted beyond one previous file. Both halves matter:
a login item that runs every day for a year with a 15 s watch writes a line
every time the engine's state changes, and an unbounded file eventually is the
problem; while a log that keeps only the current file loses exactly the thing a
person wants, which is what happened BEFORE the restart they are asking about.

`logging` with a `RotatingFileHandler` would do this. It is not used, for one
reason: this module is also imported by `crucible/cli.py` on a machine where the
directory may not exist yet, and a handler that opens its file at construction
turns "the host could not write its log" into an exception from an import. Here
the file is opened per write, which costs nothing at one line per fifteen
seconds and cannot fail at a moment nobody is ready for.

ASCII, AND ONLY ASCII (#12, fresh install on kylies-pc, 2026-09-26). The file
used to be UTF-8, and Windows PowerShell 5.1's `Get-Content` reads a BOM-less
file as the ANSI code page, so every em dash came back as mojibake to the one
person who had found the log. "Read it with `-Encoding utf8`" is an instruction
nobody follows, so each line is made ASCII before it is written: the
punctuation this code base writes has an ASCII spelling, and anything else
becomes `?` rather than bytes a reader cannot see.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from .paths import LOG_ROLL_BYTES


#: The non-ASCII punctuation this code base writes, spelled in ASCII. Anything
#: not in here becomes `?` in `plain`.
_ASCII_SPELLING = str.maketrans(
    {
        "—": "-",  # em dash
        "–": "-",  # en dash
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
        # wsl.exe answers in UTF-16, and text read from it as bytes carries a
        # NUL between every character.
        "\x00": "",
    }
)


def plain(text: str) -> str:
    """`text` in ASCII: what `host.log` and the installer's console print (#12)."""
    return text.translate(_ASCII_SPELLING).encode("ascii", "replace").decode("ascii")


def timestamp(now: float | None = None) -> str:
    """`2026-09-14 18:40:03` in LOCAL time, because the reader is at the machine."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))


class HostLog:
    """Append lines to one file, rolling it once at `roll_bytes`.

    Takes its paths rather than deriving them, so a test writes into tmp_path
    and the Windows rule lives in `paths.py` where it can be read.
    """

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
        # `crucible orchestrator --install` runs the sequence in the foreground (4.3),
        # and a person watching it should see what the log says WITHOUT opening
        # it. The tray passes nothing and the file is the only reader.
        self._echo = echo

    def write(self, message: str) -> str:
        """One line, timestamped. Returns the line, which is what tests read."""
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
        # Replace, not append: two previous files is a policy nobody asked for
        # and this one is bounded at 2 x roll_bytes, which is a number that can
        # be stated.
        if self.previous.exists():
            self.previous.unlink()
        self.path.replace(self.previous)
