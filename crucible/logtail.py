from __future__ import annotations

import os
import re
from pathlib import Path

RUN_HEADER_PREFIX = "=== crucible "

TAIL_WINDOW_BYTES = 64 * 1024

TAIL_MAX_BYTES = 4 * 1024 * 1024

# One process's prefix in a log several write to: vLLM's "(EngineCore pid=4242) ".
_PROCESS_PREFIX = re.compile(r"^\((?:[\w.-]+ )?pid=\d+\) ")
# A logged line's level and stamp, as vLLM writes them: "ERROR 10-09 17:35:04 [core.py:1100] ".
_LOGGED = re.compile(
    r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL) \d{2}-\d{2} \d{2}:\d{2}:\d{2} \[[^\]]*\] "
)
# A traceback a program logged and carried on from (vLLM logs a deep_gemm import it could
# not make as a WARNING) is not why it stopped.
_CARRIED_ON = frozenset({"DEBUG", "INFO", "WARNING"})
_TRACEBACK = "Traceback (most recent call last):"
_EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception)(?::|$)")


def tail_of_last_run(log_path: Path, lines: int) -> str:
    return "\n".join(_last_run(log_path)[-lines:])


def first_error_of_last_run(log_path: Path) -> str | None:
    """The exception line of the first traceback the last run printed as an error.

    A failed start prints its own reason first and the wrappers after it: Python prints
    a chain root first, and vLLM's API server ends the log with "Engine core
    initialization failed. See root cause above", below the engine core's traceback
    that names it (``ValueError: ... KV cache ...``), which a 40-line tail cuts off.
    Each process's lines are read apart, since another process may write between the
    frames of one traceback. With no traceback, the last exception line the run
    printed; None when it printed neither."""
    open_tracebacks: dict[str, str | None] = {}
    last_bare: str | None = None
    for raw in _last_run(log_path):
        found = _PROCESS_PREFIX.match(raw)
        process = found.group(0) if found else ""
        text = raw[found.end():] if found else raw
        logged = _LOGGED.match(text)
        level = logged.group(1) if logged else None
        body = text[logged.end():] if logged else text
        if body.startswith(_TRACEBACK):
            open_tracebacks[process] = level
            continue
        if process in open_tracebacks:
            if not body.strip() or body[0] in " \t":
                continue
            if open_tracebacks.pop(process) not in _CARRIED_ON:
                return body.strip()
            continue
        if level not in _CARRIED_ON and _EXCEPTION_LINE.match(body):
            last_bare = body.strip()
    return last_bare


def led_by_first_error(log_path: Path, report: str) -> str:
    """``report`` (a log's tail, as a refusal quotes it), led by the error that tail may
    have cut off."""
    first = first_error_of_last_run(log_path)
    if first is None:
        return report
    return f"First error in its log: {first}\n{report}"


def _last_run(log_path: Path) -> list[str]:
    path = Path(log_path)
    if not path.is_file():
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            window = TAIL_WINDOW_BYTES
            while True:
                start = max(0, end - window)
                handle.seek(start)
                text = handle.read(end - start).decode("utf-8", errors="replace")
                if start > 0:
                    text = text.split("\n", 1)[-1]
                found = text.splitlines()
                header = _last_header(found)
                if header is not None:
                    return found[header:]
                if start == 0 or window >= TAIL_MAX_BYTES:
                    return found
                window *= 2
    except OSError:
        return []


def _last_header(found: list[str]) -> int | None:
    for index in range(len(found) - 1, -1, -1):
        if found[index].startswith(RUN_HEADER_PREFIX):
            return index
    return None


__all__ = [
    "RUN_HEADER_PREFIX",
    "TAIL_MAX_BYTES",
    "TAIL_WINDOW_BYTES",
    "first_error_of_last_run",
    "led_by_first_error",
    "tail_of_last_run",
]
