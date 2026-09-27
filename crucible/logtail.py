from __future__ import annotations

import os
from pathlib import Path

RUN_HEADER_PREFIX = "=== crucible "

TAIL_WINDOW_BYTES = 64 * 1024

TAIL_MAX_BYTES = 4 * 1024 * 1024


def tail_of_last_run(log_path: Path, lines: int) -> str:
    path = Path(log_path)
    if not path.is_file():
        return ""
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
                    return "\n".join(found[header:][-lines:])
                if start == 0 or window >= TAIL_MAX_BYTES:
                    return "\n".join(found[-lines:])
                window *= 2
    except OSError:
        return ""


def _last_header(found: list[str]) -> int | None:
    for index in range(len(found) - 1, -1, -1):
        if found[index].startswith(RUN_HEADER_PREFIX):
            return index
    return None


__all__ = [
    "RUN_HEADER_PREFIX",
    "TAIL_MAX_BYTES",
    "TAIL_WINDOW_BYTES",
    "tail_of_last_run",
]
