"""A TTS stream session's turn on the server: it runs inside a queue session.

A stream opens only inside a queue session held by its client: the one its
session header names, or the open session its client already holds. A client
that holds none gets one opened for the stream (``act: "tts"``, the stream-open's
``idle_s``), which waits in the line behind other sessions like any other and ends when
the stream closes. There is no priority: while another client's session holds the
server, the stream waits its turn (or, sent with ``"queue": false``, is refused
``session_open``).

The open is a held-open request, like a queued chat: it answers once the session is open
and the voice is resident, because only then does a stream have a sample rate and a
fingerprint to answer with. A voice that is not resident is loaded by a ``load-voice``
job inside the session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request, Response

from ..admission import JobRequest, Refusal, admit
from ..errors import ApiError, JobError
from ..jobs.base import DONE, TERMINAL_STATES
from ..jobs.line import MAX_MAX_WAIT_S
from ..queuesessions import CLIENT, OPEN, QUEUED, QueueSession, QueueSessions
from .caller import queue_session
from .context import AppContext
from .deps import error_response
from .proxy import unless_the_caller_leaves

STREAM_ACT = "tts"


@dataclass(frozen=True)
class StreamTurn:
    session: QueueSession
    opened_for_it: bool


def _caller_left(what: str) -> Response:
    return error_response(
        ApiError(
            499,
            "client_disconnected",
            f"the caller closed the connection while its TTS stream waited for {what}; "
            "no stream was opened",
        )
    )


def _ended_before_it_opened(session: QueueSession) -> ApiError:
    return ApiError(
        409,
        "session_closed",
        f"the queue session {session.id} this stream waited in ended before it opened "
        f"({session.reason}): {session.message}. No stream was opened",
        {"queue_session_id": session.id, "reason": session.reason, "opened": False},
    )


async def _until_open(sessions: QueueSessions, session: QueueSession) -> bool:
    """True once the session has left the line, opened or not."""
    waiter = sessions.subscribe(session)
    try:
        while session.status == QUEUED:
            await waiter.wait()
            waiter.clear()
    finally:
        sessions.unsubscribe(session, waiter)
    return True


async def _until_ended(store: Any, job: Any) -> bool:
    """True once the job has ended, however it ended."""
    waiter = store.subscribe(job)
    try:
        while job.status not in TERMINAL_STATES:
            await waiter.wait()
            waiter.clear()
    finally:
        store.unsubscribe(job, waiter)
    return True


def _refuse_without_waiting(ctx: AppContext) -> None:
    ctx.sessions.refuse_call_if_held(None, "a TTS stream session", queueable=False)
    ctx.store.refuse_if_busy()
    raise ApiError(
        409,
        "server_busy",
        f"this server has {len(ctx.line)} item(s) waiting in its queue ahead of a session "
        "for this stream, and the stream was opened with \"queue\": false. Leave "
        "`queue` out to wait in the line",
        {"queue_depth": len(ctx.line)},
    )


async def take_the_server(
    ctx: AppContext, request: Request, *, client: str | None, idle_s: int,
    queue: Any,
) -> StreamTurn | Response:
    """The queue session this stream opens in, opened for it when its client holds
    none; a Response when the caller left while it waited."""
    held = queue_session(request, ctx.sessions)
    if held is not None:
        return StreamTurn(held, opened_for_it=False)
    line, sessions = ctx.line, ctx.sessions
    line.refuse_if_full(client)
    session = sessions.create(
        act=STREAM_ACT, client=client, model=None, idle_s=idle_s,
        max_wait_s=queue.max_wait_s if queue is not False else MAX_MAX_WAIT_S,
    )
    line.join_session(session)
    await ctx.queue_pump.step()
    if session.status == OPEN:
        return StreamTurn(session, opened_for_it=True)
    if queue is False:
        if session.status == QUEUED:
            line.remove(
                session.id, CLIENT,
                "the stream was opened with \"queue\": false and the server was not free",
            )
        _refuse_without_waiting(ctx)
    waited = await unless_the_caller_leaves(_until_open(sessions, session), request)
    if waited is None:
        await let_go(ctx, session, "the caller closed the connection while it waited")
        return _caller_left("a queue session")
    if session.status != OPEN:
        raise _ended_before_it_opened(session)
    return StreamTurn(session, opened_for_it=True)


async def let_go(ctx: AppContext, session: QueueSession, message: str) -> None:
    """Give back a session opened for a stream that never opened."""
    if session.status == QUEUED:
        ctx.line.remove(session.id, CLIENT, message)
    elif session.status == OPEN:
        await ctx.session_closer.end(session, CLIENT, message)


async def voice_ready(
    ctx: AppContext, request: Request, turn: StreamTurn, voice: str
) -> Response | None:
    """The voice resident for the stream, loaded by a load-voice job inside the session
    when it is not; a Response when the caller left while it loaded."""
    residency = ctx.residency
    async with residency.settled_for(f"streaming {voice!r}"):
        try:
            residency.refuse_if_stopping(f"stream {voice!r}")
        except JobError as exc:
            raise ApiError(409, exc.code, exc.message) from None
        resident = residency.resident_voice
        if resident is not None and resident.voice_id == voice:
            return None
    session = turn.session
    outcome = await admit(
        JobRequest(
            type="load-voice",
            model=voice,
            client=session.client,
            client_ref=f"for a TTS stream in queue session {session.id}",
            queue=MAX_MAX_WAIT_S,
            session=session.id,
        ),
        ctx.admission(),
    )
    if isinstance(outcome, Refusal):
        raise outcome.error
    ctx.sessions.item_arrived(session)
    job = outcome.job
    ended = await unless_the_caller_leaves(_until_ended(ctx.store, job), request)
    if ended is None:
        return _caller_left(f"its voice to load (job {job.id})")
    if job.status != DONE:
        failure = job.failure
        why = "" if failure is None else f": {failure.message}"
        raise ApiError(
            502,
            "stream_voice_load_failed",
            f"the load-voice job {job.id} that was loading {voice!r} for this stream "
            f"ended {job.status}{why}. No stream was opened",
            {"voice": voice, "load_job": job.id, "status": job.status,
             "queue_session_id": session.id},
        )
    return None


__all__ = ["StreamTurn", "let_go", "take_the_server", "voice_ready"]
