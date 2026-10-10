"""Walks the waiting line and admits its front whenever the lane is free.

Every second (and at once whenever the lane goes idle or the line changes) the pump
expires what has waited too long or been abandoned, closes the open queue session when
it has gone idle or held the server for its maximum, re-announces positions (the open
session's items move to the front), and, when the lane is free, offers the front of the
line to admission — the same checks a fresh submit meets. A refusal that only says
"busy" (``server_busy``, ``engine_in_use``) leaves the job waiting in its place; so does
``accelerator_busy`` (memory on the card held by a process Crucible does not own), which
is recorded on the item with the guard's sentence and checked again only every
``CARD_RECHECK_S`` (crucible/jobs/line.py). Any other refusal ends the job ``failed``
with that refusal as its error.

A queued chat or decision (a *call*, crucible/callqueue.py) is offered its model
instead of the lane: a free slot on the resident model admits it, and a model that is
not resident is loaded for it when the lane is free. A queued job that would change
what is on the card waits while any chat is in flight, so the pump never takes a model
out from under a completion it let in.

While a resident model that keeps its calls together has queued calls, they go ahead of
what would take it off the card; the item they go ahead of takes its turn, fixing which
of them do, the first time the lane is free when it would be next (crucible/keeptogether.py).

A queue session (crucible/queuesessions.py) at the front is opened (crucible/sessionqueue.py).
While a session is open, only its own items are offered anything: nothing from anyone
else runs until it closes.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Callable

from . import clock
from .admission import (
    KEEPS_WAITING,
    AdmissionContext,
    admit_waiting,
    chats_hold_the_card,
)
from .callqueue import IN, WAIT, admit_call
from .inflight import InFlight
from .jobs.line import SERVER_RESTART, WaitingLine
from .queuesessions import SERVER_RESTART as SESSION_SERVER_RESTART
from .sessionqueue import SessionCloser, admit_session

TICK_SECONDS = 1.0


class QueuePump:
    def __init__(
        self,
        line: WaitingLine,
        admission: Callable[[], AdmissionContext],
        inflight: InFlight | None = None,
        closer: SessionCloser | None = None,
    ) -> None:
        self._line = line
        self._admission = admission
        self._inflight = InFlight() if inflight is None else inflight
        self._closer = closer
        self._wake: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._stepping = asyncio.Lock()

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
        held = self._line.sessions.current()
        if held is not None:
            self._line.sessions.close(
                held, SESSION_SERVER_RESTART,
                "the server stopped while the session was open; open a new session once "
                "the server is back",
            )
            _say(f"closed session {held.id} because the server is stopping")
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

    async def close_due_session(self) -> None:
        line = self._line
        due = line.sessions.due()
        if due is None:
            return
        session, reason, message = due
        assert self._closer is not None, "a pump that closes sessions needs a SessionCloser"
        await self._closer.end(session, reason, message)
        _say(f"closed session {session.id} ({reason}): {message}")

    async def step(self) -> None:
        """One walk of the line. Steps never overlap: a route that wants its new item
        offered at once (POST /v1/queue/sessions) takes a step of its own."""
        async with self._stepping:
            await self._step()

    async def _step(self) -> None:
        line = self._line
        line.expire()
        await self.close_due_session()
        line.reorder()
        if len(line) == 0:
            return
        ctx = self._admission()
        if ctx.store.lane_free and line.take_kept_turn():
            line.reorder()
        held = line.sessions.current()
        for waiting in line.ordered():
            if held is not None and waiting.session != held.id:
                return
            if waiting.is_session:
                verdict: Any = await admit_session(waiting, ctx, self._inflight)
                if verdict == IN:
                    return
                if verdict == WAIT:
                    return
                if not waiting.gone:
                    line.fail(waiting, verdict)
                continue
            if waiting.is_call:
                verdict = await admit_call(waiting, ctx, self._inflight)
                if verdict == IN:
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
            if not waiting.card_due(clock.now()):
                return
            refusal = await admit_waiting(waiting, ctx)
            if refusal is None:
                return
            if refusal.code in KEEPS_WAITING:
                if not waiting.gone:
                    line.not_yet(waiting, refusal)
                return
            if not waiting.gone:
                line.fail(waiting, refusal)



def _say(line: str) -> None:
    print(f"crucible: {line}", file=sys.stderr)
