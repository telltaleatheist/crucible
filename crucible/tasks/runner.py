from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .. import interpreter, tasks
from ..errors import ApiError
from .states import Task

PROGRESS_INTERVAL_SECONDS = 0.5

TERMINATE_GRACE_SECONDS = 10.0

REASON_PREFIX = "crucible: "

Emit = Callable[[str, dict[str, Any]], None]


def install_command() -> str:
    sibling = Path(sys.executable).resolve().parent / "crucible"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    found = tasks.which("crucible")
    if found is not None:
        return found
    raise ApiError(
        503,
        "install_command_missing",
        f"this server cannot install anything: there is no `crucible` console "
        f"script at {sibling} and none on PATH {tasks.searched_note()}. It is the "
        "script `pip install crucible` writes beside the interpreter this "
        "server runs on",
    )


def install_argv(job_type: str, narrator_engine: str | None) -> list[str]:
    argv = [tasks.install_command(), "install", job_type, "--verbose"]
    if narrator_engine is not None:
        argv += ["--narrator-engine", narrator_engine]
    return argv


class Throttle:

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._last = 0.0

    def due(self) -> bool:
        now = time.monotonic()
        if now - self._last < self._interval:
            return False
        self._last = now
        return True


def relay_install_line(task: Task, line: str, throttle: Throttle, emit: Emit) -> None:
    if line.startswith(REASON_PREFIX):
        task.reason = line[len(REASON_PREFIX):].strip() or None
    measured = interpreter.parse_progress_line(line)
    if measured is None:
        emit("progress", {"line": line})
    elif throttle.due():
        emit("progress", measured)


def run_install_process(task: Task, argv: list[str], home: Path, emit: Emit) -> int:
    environment = dict(os.environ)
    environment["CRUCIBLE_HOME"] = str(home)
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    task.process = process
    assert process.stdout is not None
    throttle = Throttle(PROGRESS_INTERVAL_SECONDS)
    try:
        for line in process.stdout:
            if task.cancel_requested and process.poll() is None:
                process.terminate()
            relay_install_line(task, line.rstrip("\n"), throttle, emit)
        code = process.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        code = process.wait()
    finally:
        task.process = None
    return code
