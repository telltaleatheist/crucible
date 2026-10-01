"""Opening a queue session when it reaches the front of the line.

A session (crucible/queuesessions.py) opens only on a free lane, with no TTS stream session open and,
when it must load its model, with no chat in flight. A session that names a ``model`` that
is not resident has it loaded first by an ordinary ``load-model`` job attributed to the
session (its client, ``client_ref: "opening session ses-…"``); the session reports ``open``
only once that load is done, and a load that fails ends it ``load_failed``. Closing a
session, for any reason, goes through ``SessionCloser.end`` so its waiting items leave the line
and the card is settled the same way every time.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .admission import KEEPS_WAITING, AdmissionContext, JobRequest, Refusal, admit
from .callqueue import IN, WAIT, load_ended
from .errors import ApiError
from .inflight import InFlight
from .jobs.base import DONE
from .jobs.line import WaitingLine, WaitingSession
from .queuesessions import QueueSession, QueueSessions

MAX_LOADS = 2


def _load_failed(session: QueueSession, job: Any) -> ApiError:
    failure = job.failure
    why = "" if failure is None else f": {failure.message}"
    return ApiError(
        502,
        "session_load_failed",
        f"session {session.id} was opening with {session.model!r}, and the load-model job "
        f"{job.id} that was loading it ended {job.status}{why}. The session never "
        "opened; open a new one once the model can load",
        {"session_id": session.id, "model": session.model, "load_job": job.id,
         "status": job.status},
    )


async def _resident(ctx: AdmissionContext, session: QueueSession) -> bool:
    async with ctx.residency.settled_for(f"opening session {session.id}"):
        resident = ctx.residency.resident_model
        return resident is not None and resident.model_id == session.model


async def _load_for(
    waiting: WaitingSession, ctx: AdmissionContext, inflight: InFlight
) -> str | ApiError:
    session: QueueSession = waiting.job  # type: ignore[assignment]
    if waiting.loads >= MAX_LOADS:
        return ApiError(
            409,
            "model_not_resident",
            f"session {session.id} was loading {session.model!r} to open, {waiting.loads} "
            "time(s), and something else took the card each time before it opened",
            {"session_id": session.id, "model": session.model},
        )
    if len(inflight) > 0:
        return WAIT
    outcome = await admit(
        JobRequest(
            type="load-model",
            model=session.model,
            client=session.client,
            client_ref=f"opening session {session.id}",
            from_the_line=True,
            session=session.id,
        ),
        ctx,
    )
    if isinstance(outcome, Refusal):
        if outcome.error.code in KEEPS_WAITING:
            return WAIT
        return outcome.error
    waiting.load_job = session.load_job = outcome.job.id
    waiting.loads += 1
    return WAIT


async def admit_session(
    waiting: WaitingSession, ctx: AdmissionContext, inflight: InFlight
) -> str | ApiError:
    """Open the session at the front of the line. IN: it is open. WAIT: it keeps its
    place and nothing behind it goes. An ApiError ends it with that error."""
    session: QueueSession = waiting.job  # type: ignore[assignment]
    line = ctx.store.line
    if waiting.load_job is not None:
        ended = load_ended(ctx, waiting.load_job)
        if ended is None:
            return WAIT
        waiting.load_job = None
        if ended.status != DONE:
            return _load_failed(session, ended)
    if not ctx.store.lane_free or ctx.streaming():
        return WAIT
    if session.model is not None and not await _resident(ctx, session):
        return await _load_for(waiting, ctx, inflight)
    if waiting.gone or line is None:
        return IN
    line.started(waiting)
    return IN


@dataclass(frozen=True)
class SessionCloser:
    """The one way an open queue session ends, whatever ended it."""

    sessions: QueueSessions
    line: WaitingLine
    settlement: Any
    streams: Any

    async def end(self, session: QueueSession, reason: str, message: str) -> bool:
        """Close an open session: its items still waiting leave the line
        (``session_closed``), a TTS stream session open inside it closes (its `closed`
        frame says `session_closed` and why), and the card is settled, unloaded unless
        something else holds it. False when the session was not open."""
        if not self.sessions.close(session, reason, message):
            return False
        self.line.remove_items_of(
            session.id,
            f"its session {session.id} closed ({reason}) before it ran: {message}",
        )
        stream = self.streams.session
        if stream is not None and stream.id in session.stream_sessions:
            await asyncio.to_thread(
                self.streams.close, stream,
                f"its queue session {session.id} closed ({reason}): {message}",
                {"code": "session_closed", "queue_session_id": session.id,
                 "session_reason": reason},
            )
        await asyncio.to_thread(
            self.settlement.settle_quietly, f"session {session.id} closed ({reason})"
        )
        return True


__all__ = ["SessionCloser", "admit_session"]
