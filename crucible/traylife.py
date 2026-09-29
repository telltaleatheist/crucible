from __future__ import annotations

import time
from pathlib import Path

from . import processlock
from .controller_client import LocalError

PID_NAME = "tray.pid"
CLOSE_NAME = "tray.close"
LOCK_NAME = "tray.lock"

CLOSE_SECONDS = 15.0
POLL_SECONDS = 0.1


def pid_path(home: Path) -> Path:
    return Path(home) / PID_NAME


def close_request_path(home: Path) -> Path:
    return Path(home) / CLOSE_NAME


def running_pid(home: Path) -> int | None:
    pid_file = pid_path(home)
    if not pid_file.exists():
        return None
    raw = pid_file.read_text().strip()
    if not raw.isdigit() or not processlock.alive(int(raw)):
        pid_file.unlink(missing_ok=True)
        return None
    return int(raw)


def ask_to_close(home: Path) -> int | None:
    pid = running_pid(home)
    if pid is not None:
        close_request_path(home).write_text("close\n")
    return pid


def close_tray(home: Path) -> None:
    pid = ask_to_close(home)
    if pid is None:
        return
    deadline = time.monotonic() + CLOSE_SECONDS
    while True:
        if not processlock.alive(pid):
            pid_path(home).unlink(missing_ok=True)
            return
        if time.monotonic() >= deadline:
            raise LocalError(
                f"tray_close_failed: the tray (pid {pid}) was asked to close through "
                f"{close_request_path(home)} and was still running {CLOSE_SECONDS:.0f} s "
                "later; installation is unchanged. Close it from its icon by the clock "
                "(Quit), then run this again"
            )
        time.sleep(POLL_SECONDS)
