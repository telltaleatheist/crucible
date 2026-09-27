from __future__ import annotations

import asyncio
from typing import Any

from fastapi import Request
from fastapi.responses import StreamingResponse

from ...jobs import disabled_error
from ...jobs.tts.common import known_voice
from ...ttsstream import StreamManager, require_sayable, require_streamable
from ..caller import client_agent
from ..schemas import StreamOp, StreamOpen
from ..sse import _last_event_id, _session_event_stream
from ..context import AppContext, Routers


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    config, residency = ctx.config, ctx.residency

    # -------------------------------------------------------- tts streaming
    #
    # PHASE3-TTS.md section 7. Four routes and no socket: the Listen path, the
    # in-app Play button and the browser extension, built out of the two things
    # this server already does well. The session's own machinery — the rows, the
    # replay buffer, the grace window, the per-row cancel — is
    # `crucible/ttsstream.py`; what is here is the wire.

    def _streaming_voice(voice: str) -> Any:
        """The manifest for a voice this server may be asked to stream."""
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
        # WAITED OUT, NOT REFUSED (2026-09-24, Briefcase). A session opening
        # while the settlement clears the card used to be refused
        # `engine_in_use` by `claim()`, naming the settlement. It now waits and
        # opens against the settled card — or says `voice_not_resident`, which
        # is then true. `streams.open` is synchronous and takes its claim
        # inside, under the lock the settlement's check-and-claim takes, so the
        # two cannot interleave.
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
        """The session's SSE stream — everything it has to say, audio included.

        `Last-Event-ID` is the reattach: a connection that dropped in a tunnel
        comes back here inside the grace window, is replayed what it missed and
        follows live from there. It is the one behaviour a WebSocket could not
        have given for free, which is why this door is not one.
        """
        streams: StreamManager = request.app.state.streams
        session = streams.get(session_id)
        delivered = _last_event_id(request)
        # Asked before the response is built, so an unreplayable resume is a 409
        # with a body rather than a 200 that ends at once.
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
        """One op: `say`, `cancel`, `cancel_all` or `close`.

        **`say` answers with the row's id and not the audio.** A client that
        wants the audio reads the stream; a client that never opened one is
        refused by name rather than generating into nothing.
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
        # `close`, which the validator has already proved is the only one left.
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
