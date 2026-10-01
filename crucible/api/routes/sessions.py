"""Queue sessions (crucible/queuesessions.py): ask for the server for a run of requests,
follow it, keep it, end it. Not TTS stream sessions (routes/tts_stream.py)."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import StreamingResponse

from ...admission import refuse_an_upstream_model
from ...errors import ApiError
from ...inflight import require_act_name
from ...protocol import CLIENT_HEADER, USER_AGENT_HEADER
from ...queuesessions import CLIENT, CLOSED, OPEN, QUEUED, QueueSession, who
from .. import sse
from ..caller import client_agent
from ..context import AppContext, Routers
from ..responses import NOT_FOUND, SessionState, SessionTicket
from ..schemas import SessionOpen


def _refuse_an_unknown_model(model: str) -> None:
    from ...manifests import load_manifest

    refuse_an_upstream_model(model)
    try:
        load_manifest(model)
    except Exception as exc:
        raise ApiError(
            404,
            "unknown_model",
            f"{model!r} is not a model this server has a manifest for ({exc}), so a "
            "session cannot open with it resident. GET /v1/models lists the models; "
            "leave `model` out to open on whatever is resident",
            {"model": model},
        ) from None


def _refuse_if_not_theirs(session: QueueSession, request: Request) -> None:
    caller = client_agent(request)
    if session.client == caller:
        return
    raise ApiError(
        409,
        "session_not_yours",
        f"session {session.id} belongs to {who(session.client)}, and this request "
        f"comes from {who(caller)} ({CLIENT_HEADER}, else {USER_AGENT_HEADER}). "
        "An operator ends another client's session with DELETE /v1/queue/{id}",
        {"session_id": session.id, "client": session.client, "caller": caller},
    )


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private

    @private.post("/queue/sessions", status_code=202, response_model=SessionTicket)
    async def open_session(request: Request, body: SessionOpen) -> dict[str, Any]:
        """Ask for the server for a run of requests you cannot know in advance. It waits
        in the line like any queued item and answers at once with a ticket; follow
        `GET /v1/queue/sessions/{id}/events` for `opened`. While it is open, nothing
        from any other client runs.
        """
        act = require_act_name(body.act.strip(), "a session's `act`")
        if body.model is not None:
            _refuse_an_unknown_model(body.model)
        client = client_agent(request)
        line, sessions = ctx.line, ctx.sessions
        line.refuse_if_full(client)
        session = sessions.create(
            act=act, client=client, model=body.model, idle_s=body.idle_s,
            max_wait_s=body.max_wait_s,
        )
        line.join_session(session)
        await ctx.queue_pump.step()
        return {"session_id": session.id, "status": session.status,
                "position": session.position}

    @private.get(
        "/queue/sessions/{session_id}", response_model=SessionState, responses=NOT_FOUND
    )
    async def session_state(session_id: str) -> dict[str, Any]:
        """Where the session stands: queued (and where), open (what it has run and has
        in flight, when it would go idle), or closed and why."""
        session = ctx.sessions.get(session_id)
        if session.status == QUEUED:
            ctx.line.touch(job_id=session.id)
        return ctx.sessions.state(session)

    @private.get("/queue/sessions/{session_id}/events", responses=NOT_FOUND)
    async def session_events(request: Request, session_id: str) -> StreamingResponse:
        """The session's own stream: `queued {position, of}` and `moved`, then
        `opened`, then `closed {reason}`; `removed {reason}` in place of `closed` when
        it never opened."""
        session = ctx.sessions.get(session_id)
        if session.status == QUEUED:
            ctx.line.touch(job_id=session.id)
        return sse.queue_session_events(
            request, ctx.sessions, session, sse.last_event_id(request)
        )

    @private.post("/queue/sessions/{session_id}/touch", responses=NOT_FOUND)
    async def touch_session(request: Request, session_id: str) -> dict[str, Any]:
        """"Still here", for a long gap on the client's side with nothing in flight.
        Cheap: a timestamp in memory."""
        session = ctx.sessions.get(session_id)
        _refuse_if_not_theirs(session, request)
        if session.status == CLOSED:
            raise ApiError(
                409,
                "session_closed",
                f"session {session.id} closed ({session.reason}): {session.message}. "
                "Open a new one with POST /v1/queue/sessions",
                {"session_id": session.id, "reason": session.reason},
            )
        if session.status == OPEN:
            ctx.sessions.touch(session)
        else:
            ctx.line.touch(job_id=session.id)
        return {"session_id": session.id, "status": session.status}

    @private.delete(
        "/queue/sessions/{session_id}", response_model=SessionState, responses=NOT_FOUND
    )
    async def close_session(request: Request, session_id: str) -> dict[str, Any]:
        """End your session (reason `client`): an open one closes and the card is
        settled before this answers; a queued one leaves the line. A session already
        closed answers as it is."""
        sessions = ctx.sessions
        session = sessions.get(session_id)
        _refuse_if_not_theirs(session, request)
        if session.status == QUEUED:
            ctx.line.remove(
                session.id, CLIENT,
                "the client that asked for it removed it (DELETE /v1/queue/sessions/{id})",
            )
        elif session.status == OPEN:
            await ctx.session_closer.end(
                session, CLIENT,
                "the client closed it (DELETE /v1/queue/sessions/{id})",
            )
        return sessions.state(session)
