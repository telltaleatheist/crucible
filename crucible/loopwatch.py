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
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
import traceback
from typing import Callable, TextIO

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
        frame = sys._current_frames().get(self._loop_thread or -1)
        stack = "".join(traceback.format_stack(frame)) if frame is not None else "(no frame)\n"
        out = self._out if self._out is not None else sys.stderr
        print(
            f"crucible: the event loop has not run for {blocked:.1f} s; every route waits "
            f"behind it. What it is doing:\n{stack}",
            file=out,
            flush=True,
        )


__all__ = ["BEAT_S", "BLOCKED_S", "REPEAT_S", "LoopWatch"]
