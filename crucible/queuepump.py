"""Walks the waiting line and admits its front whenever the lane is free.

Every second (and at once whenever the lane goes idle or the line changes) the pump
expires what has waited too long or been abandoned, re-announces positions (the lease
holder's jobs move to the front while its lease is open), and, when the lane is free,
offers the front of the line to admission — the same checks a fresh submit meets. A
refusal that only says "busy" (``server_busy``, ``leased``, ``engine_in_use``) leaves
the job waiting in its place; any other refusal ends the job ``failed`` with that
refusal as its error. A job refused ``leased`` does not block the jobs behind it that
the lease does not refuse.

A queued chat or decision (a *call*, crucible/callqueue.py) is offered its model
instead of the lane: a free slot on the resident model admits it, and a model that is
not resident is loaded for it when the lane is free. A queued job that would change
what is on the card waits while any chat is in flight, so the pump never takes a model
out from under a completion it let in.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Callable

from .admission import (
    KEEPS_WAITING,
    AdmissionContext,
    admit_waiting,
    chats_hold_the_card,
)
from .callqueue import IN, LEASED, WAIT, admit_call
from .inflight import InFlight
from .jobs.line import SERVER_RESTART, WaitingLine

TICK_SECONDS = 1.0


class QueuePump:
    def __init__(
        self,
        line: WaitingLine,
        admission: Callable[[], AdmissionContext],
        inflight: InFlight | None = None,
    ) -> None:
        self._line = line
        self._admission = admission
        self._inflight = InFlight() if inflight is None else inflight
        self._wake: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None

    def wake(self) -> None:
        wake, loop = self._wake, self._loop
        if wake is None or loop is None:
            return
        try:
            here = asyncio.get_running_loop()
        except RuntimeError:
            here = None
        if here is loop:
            wake.set()
        elif not loop.is_closed():
            loop.call_soon_threadsafe(wake.set)

    def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("the queue pump is already running")
        self._wake = asyncio.Event()
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._run(), name="crucible-queue-pump")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                _say(f"the queue pump had already died: {type(exc).__name__}: {exc}")
        ended = self._line.drain(
            SERVER_RESTART,
            "the server stopped while this job waited in its queue; submit it "
            "again once the server is back",
        )
        if ended:
            _say(f"ended {ended} queued job(s) because the server is stopping")

    async def _run(self) -> None:
        assert self._wake is not None
        while True:
            self._wake.clear()
            try:
                await self.step()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _say(f"the queue pump failed a step: {type(exc).__name__}: {exc}")
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=TICK_SECONDS)
            except asyncio.TimeoutError:
                pass

    async def step(self) -> None:
        line = self._line
        line.expire()
        line.reorder()
        if len(line) == 0:
            return
        ctx = self._admission()
        for waiting in line.ordered():
            if waiting.is_call:
                verdict: Any = await admit_call(waiting, ctx, self._inflight)
                if verdict in (IN, LEASED):
                    continue
                if verdict == WAIT:
                    return
                if not waiting.gone:
                    line.fail(waiting, verdict)
                continue
            if not ctx.store.lane_free:
                return
            if chats_hold_the_card(waiting.job.type, len(self._inflight)):
                return
            refusal = await admit_waiting(waiting, ctx)
            if refusal is None:
                return
            if refusal.code == "leased":
                continue
            if refusal.code in KEEPS_WAITING:
                return
            if not waiting.gone:
                line.fail(waiting, refusal)



def _say(line: str) -> None:
    print(f"crucible: {line}", file=sys.stderr)
