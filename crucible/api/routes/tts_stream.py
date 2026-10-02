from __future__ import annotations

import asyncio
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from ...jobs.tts.common import known_voice
from ...queuesessions import CLIENT
from ...ttsstream.validate import require_sayable, require_streamable
from .. import sse
from ..caller import client_agent
from ..context import AppContext, Routers
from ..deps import tts_enabled
from ..schemas import StreamOp, StreamOpen
from ..streamturn import let_go, take_the_server, voice_ready


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    config, residency = ctx.config, ctx.residency

    def _streaming_voice(voice: str) -> Any:
        manifest = known_voice(voice)
        require_streamable(manifest, config.backend_kind)
        return manifest

    def _touch_its_queue_session(stream_session_id: str) -> None:
        owner = ctx.sessions.of_stream_session(stream_session_id)
        if owner is not None:
            ctx.sessions.touch(owner)

    async def _close(session: Any) -> dict[str, Any]:
        owner = ctx.sessions.of_stream_session(session.id)
        closed = await asyncio.to_thread(
            ctx.streams.close, session, "the client closed the session"
        )
        if closed and owner is not None and owner.opened_for_stream == session.id:
            await ctx.session_closer.end(
                owner, CLIENT,
                f"the TTS stream session {session.id} it was opened for was closed by "
                "its client",
            )
        return {"session_id": session.id, "closed": closed}

    @private.post(
        "/tts/stream", status_code=201, dependencies=[tts_enabled(config)],
        response_model=None,
    )
    async def open_stream(request: Request, body: StreamOpen) -> dict[str, Any] | Response:
        """Open the one streaming session this server will hold at a time, inside a
        queue session: the client's own (the session header, or the open one it holds),
        else one opened for the stream, which waits in the line like any session and
        closes with the stream. Answers once that session is open and the voice is
        resident (loaded in the session when it is not).

        Sent with the queue-ticket header set to `1`, an open whose session has to
        wait answers `202 {queue_session_id, status, position}` at once instead: follow
        `GET /v1/queue/sessions/{id}/events` and, after `opened`, open again with the
        session header naming it, which claims that session for the stream (it closes
        with it). Unclaimed 60 s after opening, it closes. docs/QUEUE.md has the wire.
        """
        streams = ctx.streams
        manifest = _streaming_voice(body.voice)
        client = client_agent(request)
        turn = await take_the_server(
            ctx, request, client=client, idle_s=body.idle_s, queue=body.queue
        )
        if isinstance(turn, Response):
            return turn
        try:
            left = await voice_ready(ctx, request, turn, body.voice)
            if left is not None:
                if turn.opened_for_it:
                    await let_go(ctx, turn.session, "the caller left while its voice loaded")
                return left
            async with residency.settled_for(f"streaming {body.voice!r}"):
                session = streams.open(
                    voice=body.voice,
                    language=body.language,
                    manifest=manifest,
                    client=client,
                    loop=asyncio.get_running_loop(),
                )
        except BaseException as exc:
            if turn.opened_for_it:
                await let_go(
                    ctx, turn.session,
                    f"the stream it was opened for could not open: {exc}",
                )
            raise
        ctx.sessions.adopt_stream_session(
            turn.session, session.id, opened_for_it=turn.opened_for_it
        )
        ctx.sessions.item_arrived(turn.session)
        return JSONResponse(status_code=201, content={
            "session_id": session.id,
            "voice": session.voice,
            "fingerprint": session.fingerprint,
            "sample_rate": session.sample_rate,
            "backend": session.backend,
            "queue_session_id": turn.session.id,
            "queue_session_opened_for_stream": turn.opened_for_it,
        })

    @private.get("/tts/stream/{session_id}/events")
    async def stream_events(request: Request, session_id: str) -> StreamingResponse:
        """The session's SSE stream, audio included. Reattach with `Last-Event-ID`
        within the grace window to replay what was missed.
        """
        session = ctx.streams.get(session_id)
        delivered = sse.last_event_id(request)
        session.check_replayable(delivered)
        _touch_its_queue_session(session.id)
        return sse.session_events(request, session, delivered)

    @private.post("/tts/stream/{session_id}", status_code=202)
    async def stream_op(session_id: str, body: StreamOp) -> dict[str, Any]:
        """One op: `say`, `cancel`, `cancel_all` or `close`. `say` answers with the row
        id; the audio arrives on the stream. Every op is activity of the queue session
        the stream runs in.
        """
        session = ctx.streams.get(session_id)
        _touch_its_queue_session(session.id)
        if body.op == "say":
            manifest = known_voice(session.voice)
            sampling = require_sayable(manifest, body.take)
            return {"id": session.say(body.id, body.text, body.take, sampling)}
        if body.op == "cancel":
            return {"id": body.id, "outcome": session.cancel(body.id)}
        if body.op == "cancel_all":
            return {"cancelled": session.cancel_all()}
        return await _close(session)

    @private.delete("/tts/stream/{session_id}")
    async def close_stream(session_id: str) -> dict[str, Any]:
        """The same as `{"op": "close"}`, for a client that only has verbs. A queue
        session opened for the stream closes with it; one the client opened itself
        stays open."""
        return await _close(ctx.streams.get(session_id))
