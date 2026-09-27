from __future__ import annotations

import asyncio
from typing import Any

from fastapi import Request
from fastapi.responses import StreamingResponse

from ...jobs import disabled_error
from ...jobs.tts.common import known_voice
from ...ttsstream import StreamManager, require_sayable, require_streamable
from ..caller import client_agent
from ..context import AppContext, Routers
from ..schemas import StreamOp, StreamOpen
from ..sse import _last_event_id, _session_event_stream


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    config, residency = ctx.config, ctx.residency

    def _streaming_voice(voice: str) -> Any:
        if not config.enable_tts:
            raise disabled_error("tts", config)
        manifest = known_voice(voice)
        require_streamable(manifest, config.backend_kind)
        return manifest

    @private.post("/tts/stream", status_code=201)
    async def open_stream(request: Request, body: StreamOpen) -> dict[str, Any]:
        """Open the one streaming session this server will hold at a time."""
        streams: StreamManager = request.app.state.streams
        manifest = _streaming_voice(body.voice)
        async with residency.settled_for(f"streaming {body.voice!r}"):
            session = streams.open(
                voice=body.voice,
                language=body.language,
                manifest=manifest,
                client=client_agent(request),
                loop=asyncio.get_running_loop(),
            )
        return {
            "session_id": session.id,
            "voice": session.voice,
            "fingerprint": session.fingerprint,
            "sample_rate": session.sample_rate,
            "backend": session.backend,
        }

    @private.get("/tts/stream/{session_id}/events")
    async def stream_events(request: Request, session_id: str) -> StreamingResponse:
        """The session's SSE stream, audio included. Reattach with `Last-Event-ID`
        within the grace window to replay what was missed.
        """
        streams: StreamManager = request.app.state.streams
        session = streams.get(session_id)
        delivered = _last_event_id(request)
        session.check_replayable(delivered)
        return StreamingResponse(
            _session_event_stream(request, session, delivered),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @private.post("/tts/stream/{session_id}", status_code=202)
    async def stream_op(
        request: Request, session_id: str, body: StreamOp
    ) -> dict[str, Any]:
        """One op: `say`, `cancel`, `cancel_all` or `close`. `say` answers with the row
        id; the audio arrives on the stream.
        """
        streams: StreamManager = request.app.state.streams
        session = streams.get(session_id)
        if body.op == "say":
            manifest = known_voice(session.voice)
            sampling = require_sayable(manifest, body.take)
            return {"id": session.say(body.id, body.text, body.take, sampling)}
        if body.op == "cancel":
            return {"id": body.id, "outcome": session.cancel(body.id)}
        if body.op == "cancel_all":
            return {"cancelled": session.cancel_all()}
        closed = await asyncio.to_thread(
            streams.close, session, "the client closed the session"
        )
        return {"session_id": session.id, "closed": closed}

    @private.delete("/tts/stream/{session_id}")
    async def close_stream(request: Request, session_id: str) -> dict[str, Any]:
        """The same as `{"op": "close"}`, for a client that only has verbs."""
        streams: StreamManager = request.app.state.streams
        session = streams.get(session_id)
        closed = await asyncio.to_thread(
            streams.close, session, "the client closed the session"
        )
        return {"session_id": session.id, "closed": closed}
