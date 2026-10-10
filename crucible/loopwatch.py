"""Says, in the server's own log, what the event loop was doing when it stopped answering.

Every route, every SSE stream and the chat proxy's reads share one asyncio loop, so
anything that blocks it for seconds is felt by every client at once. On the PC
(2026-10-02) the loop was blocked ~3 s out of every 4 during a long book render: the
desktop app's 3 s status ping missed half its reads and flipped the window to "not
running", and a chat proxied in the same seconds came back ReadError twice and ended a
book. Nothing in the log said why, and the guest's server cannot be sampled from outside
without root.

A daemon thread posts a beat onto the loop every BEAT_S. When no beat has run for
BLOCKED_S it prints the loop thread's stack once for that stall, and again if the same
stall reaches REPEAT_S, with how long the loop has been blocked. Reading another
thread's frame is `sys._current_frames()`: no signal, no ptrace, nothing stops.

The loop's stack alone cannot say WHY it is where it is. On Victoria's laptop
(2026-10-10) it showed the loop in `logging.getLogger`'s `_lock.acquire()` (uvicorn
calls getLogger for every new connection) right as the kernel OOM-killed a YuE2 worker:
a lock some other thread held, or a loop the kernel had stalled on a page fault while
the guest ran out of memory, and nothing in the report told the two apart. So each
report also says where every other thread is (a holder of the lock the loop waits on
is among them) and the machine's memory state: the kernel's memory pressure (the share
of time tasks stalled waiting for memory) and what is left of RAM and swap.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, TextIO

PRESSURE = Path("/proc/pressure/memory")
MEMINFO = Path("/proc/meminfo")

BEAT_S = 0.25
BLOCKED_S = 1.0
REPEAT_S = 5.0


class LoopWatch:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        out: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._loop = loop
        self._out = out
        self._clock = clock
        self._loop_thread: int | None = None
        self._last_beat = clock()
        self._said_at = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Called on the loop's own thread, which is the one it watches."""
        self._loop_thread = threading.get_ident()
        self._last_beat = self._clock()
        self._thread = threading.Thread(target=self._watch, name="crucible-loopwatch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _beat(self) -> None:
        self._last_beat = self._clock()
        self._said_at = 0.0

    def _watch(self) -> None:
        while not self._stop.wait(BEAT_S):
            try:
                self._loop.call_soon_threadsafe(self._beat)
            except RuntimeError:
                return
            self.check()

    def check(self) -> None:
        """One look: say so if the loop has been blocked past a threshold not yet said."""
        blocked = self._clock() - self._last_beat
        due = BLOCKED_S if self._said_at == 0.0 else REPEAT_S
        if blocked < due or self._said_at >= due:
            return
        self._said_at = due
        self._say(blocked)

    def _say(self, blocked: float) -> None:
        frames = sys._current_frames()
        frame = frames.get(self._loop_thread or -1)
        stack = "".join(traceback.format_stack(frame)) if frame is not None else "(no frame)\n"
        others = _other_threads(frames, {self._loop_thread, threading.get_ident()})
        out = self._out if self._out is not None else sys.stderr
        print(
            f"crucible: the event loop has not run for {blocked:.1f} s; every route waits "
            f"behind it. What it is doing:\n{stack}"
            f"The other threads, innermost frame first:\n{others}"
            f"Memory: {memory_state()}",
            file=out,
            flush=True,
        )


def _other_threads(frames: dict[int, Any], skip: set[int | None]) -> str:
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    lines = []
    for ident, frame in frames.items():
        if ident in skip:
            continue
        where = traceback.extract_stack(frame)[-3:]
        trail = " <- ".join(
            f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in reversed(where)
        )
        lines.append(f"  {names.get(ident, ident)}: {trail}\n")
    return "".join(sorted(lines)) or "  (none)\n"


def _pressure(text: str) -> str:
    said = []
    for line in text.splitlines():
        kind, _, rest = line.partition(" ")
        fields = dict(part.split("=", 1) for part in rest.split() if "=" in part)
        if "avg10" in fields:
            said.append(f"{kind} {fields['avg10']}%")
    return ", ".join(said)


def memory_state(pressure: Path = PRESSURE, meminfo: Path = MEMINFO) -> str:
    """One line: the kernel's memory pressure over the last 10 s (`some`: the share of
    time at least one task stalled waiting for memory; `full`: all of them did), and the
    RAM and swap left. Off Linux it says it cannot be read here."""
    parts = []
    try:
        parts.append("pressure over 10 s " + _pressure(pressure.read_text(encoding="utf-8")))
    except OSError:
        pass
    try:
        kib: dict[str, int] = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition(":")
            if value.strip().endswith("kB"):
                kib[name] = int(value.split()[0])
    except (OSError, ValueError):
        kib = {}

    def gib(name: str) -> str:
        return f"{kib[name] / 1048576:.1f} GiB"

    if "MemAvailable" in kib and "MemTotal" in kib:
        parts.append(f"available {gib('MemAvailable')} of {gib('MemTotal')}")
    if kib.get("SwapTotal") and "SwapFree" in kib:
        parts.append(f"swap free {gib('SwapFree')} of {gib('SwapTotal')}")
    return ("; ".join(parts) if parts else "not readable on this host") + "\n"


__all__ = ["BEAT_S", "BLOCKED_S", "REPEAT_S", "LoopWatch", "memory_state"]
