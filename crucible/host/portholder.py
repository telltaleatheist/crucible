from __future__ import annotations

import csv
import io
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Sequence

from .paths import DOOR_PORT

NETSTAT_ARGV: tuple[str, ...] = ("netstat", "-ano", "-p", "tcp")

LOOKUP_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class Holder:
    pid: int
    name: str | None

    def __str__(self) -> str:
        return f"{self.name or 'an unnamed process'} (pid {self.pid})"


def tasklist_argv(pid: int) -> tuple[str, ...]:
    return ("tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH")


def listening_pid(netstat_text: str, port: int) -> int | None:
    for raw in netstat_text.splitlines():
        parts = raw.split()
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue
        local, foreign, pid = parts[1], parts[2], parts[4]
        listening = foreign.rsplit(":", 1)[-1] == "0"
        if listening and pid.isdigit() and int(pid) != 0 and local.rsplit(":", 1)[-1] == str(port):
            return int(pid)
    return None


def image_name(tasklist_csv: str) -> str | None:
    for row in csv.reader(io.StringIO(tasklist_csv)):
        if row and row[0] and not row[0].startswith("INFO:"):
            return row[0]
    return None


def _run(argv: Sequence[str]) -> str:
    done = subprocess.run(
        list(argv), capture_output=True, text=True, timeout=LOOKUP_TIMEOUT_SECONDS,
        stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return done.stdout


def holder(port: int = DOOR_PORT, run: Callable[[Sequence[str]], str] = _run) -> Holder | None:
    if sys.platform != "win32":
        return None
    try:
        pid = listening_pid(run(NETSTAT_ARGV), port)
        if pid is None:
            return None
        return Holder(pid, image_name(run(tasklist_argv(pid))))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def held_sentence(port: int = DOOR_PORT, run: Callable[[Sequence[str]], str] = _run) -> str:
    found = holder(port, run)
    if found is None:
        return (
            f"port {port} is held by a process this build could not name; find it "
            f"with `netstat -ano | findstr :{port}` and stop it, or run "
            "`crucible local shutdown` if it is an older Crucible"
        )
    return f"port {port} is held by {found}; stop it or run `crucible local shutdown`"
