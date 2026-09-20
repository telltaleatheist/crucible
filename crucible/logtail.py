"""Reading the tail of a log that holds MANY runs.

Engine and worker logs append (see `SubprocessEngine.start()` and
`workers._open_log`, both changed on 2026-09-20 so that investigating a hang
stops erasing it). That makes two of the old habits wrong at once:

1. **Slurping the file to keep the last forty lines** was free while every start
   truncated it. It is not free now.
2. **Reading "the last N lines" full stop** would cross a run boundary — and one
   caller acts on what it finds. `llama_server._fatal_in_log()` refuses to start
   an engine when it sees a fatal line in the tail, so a dead run's "out of
   memory", left behind by the append, would refuse the NEXT start of an engine
   that was going to come up. The reload-after-a-hang case walks straight into
   it: hang, OOM in the log, reload, refused from then on.

So a tail is read BACKWARDS from the end, and it stops at the last run header.
Both writers open their run with a line beginning `=== crucible `, which is what
`RUN_HEADER_PREFIX` names; a tail therefore never reports a line that belongs to
an earlier run, and the header it does include says which run the lines are from.
"""

from __future__ import annotations

import os
from pathlib import Path

#: What both log writers put at the top of every run. `engines/base.py` writes
#: `=== crucible <name> engine, <date>`; `workers.py` writes
#: `=== crucible worker <script>, <date>`.
RUN_HEADER_PREFIX = "=== crucible "

#: How much of the end of a log is read at a time. 64 KiB holds far more than a
#: tail's worth of anything these processes print; the walk doubles it when a run
#: header has not been reached yet.
TAIL_WINDOW_BYTES = 64 * 1024

#: Where the backwards walk gives up looking for a run header. A run that has
#: printed this much has printed far more than `lines` lines of its own, so
#: everything in the last window belongs to it and there is nothing to cut.
TAIL_MAX_BYTES = 4 * 1024 * 1024


def tail_of_last_run(log_path: Path, lines: int) -> str:
    """The last `lines` lines of the MOST RECENT run in `log_path`.

    Empty string when the file is missing or unreadable — this is only ever
    called to decorate a message, and a failure to read a log must never be the
    error a caller reports.
    """
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
                    # The window opened mid-line; that partial line is dropped
                    # rather than reported as a line of its own.
                    text = text.split("\n", 1)[-1]
                found = text.splitlines()
                header = _last_header(found)
                if header is not None:
                    # This run begins here. Everything above it is an older run
                    # and is not this tail's to report.
                    return "\n".join(found[header:][-lines:])
                if start == 0 or window >= TAIL_MAX_BYTES:
                    return "\n".join(found[-lines:])
                window *= 2
    except OSError:
        return ""


def _last_header(found: list[str]) -> int | None:
    """Index of the last run header in `found`, or None if there is none."""
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
