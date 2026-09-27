from __future__ import annotations

import json
import time
from typing import Any

from fastapi import Request

from ... import peer as peer_module
from ...errors import ApiError
from ..context import AppContext, Routers


def register(routers: Routers, ctx: AppContext) -> None:
    peer_router = routers.peer

    # ------------------------------------------------------------------ peer

    @peer_router.get("")
    async def read_peer(request: Request) -> dict[str, Any]:
        """`GET /v1/peer` — who manages this engine, and how old this process is.

        PHASE17 2.4: health flows ONE way. The orchestrator polls this and
        `/v1/ping`; the engine calls nothing back. An engine that phoned home
        would need to know its orchestrator's address, keep it fresh across
        restarts, and behave when it is wrong — three facts to own for a push
        a 15-second poll already delivers.
        """
        state: peer_module.PeerState = request.app.state.peer
        return state.document(_uptime_s(request))

    @peer_router.post("/claim")
    async def claim_peer(request: Request) -> dict[str, Any]:
        """`POST /v1/peer/claim` — an orchestrator says it manages this engine.

        A STATEMENT OF FACT, not a grant of permission: nothing on this server
        consults `managed_by` before doing anything, because there is nothing
        an orchestrator asks an engine to do that an app may not also ask
        (`crucible/peer.py`'s preamble). What it buys is that `/v1/info` can
        answer "who manages this".
        """
        body = await _peer_body(request)
        state: peer_module.PeerState = request.app.state.peer
        orchestrator = peer_module.Orchestrator.from_body(body.get("orchestrator"))
        force = body.get("force")
        if force is not None and not isinstance(force, bool):
            raise ApiError(
                400,
                "invalid_request",
                "`force` is a boolean. It takes an engine away from another "
                "orchestrator and is a person's act through the page, never "
                "an orchestrator's own (PHASE17-ORCHESTRATOR.md 2.1)",
            )
        claim = state.claim(orchestrator, force=bool(force))
        return {
            "role": peer_module.ROLE_ENGINE,
            "managed_by": claim.managed_by(),
            "claimed": claim.claimed,
        }

    @peer_router.delete("/claim")
    async def release_peer(request: Request) -> dict[str, Any]:
        """`DELETE /v1/peer/claim` — the orchestrator's Quit (PHASE17 2.2).

        Nothing claimed is NOT a refusal: "there is no claim" is the state the
        caller asked for. Somebody else's claim is refused, because releasing
        one by accident is how an engine ends up unmanaged with a tray still
        watching it.
        """
        body = await _peer_body(request)
        state: peer_module.PeerState = request.app.state.peer
        raw = body.get("orchestrator")
        who = None if raw is None else peer_module.Orchestrator.from_body(raw)
        state.release(who, force=bool(body.get("force")))
        return {"role": peer_module.ROLE_ENGINE, "managed_by": None}

    async def _peer_body(request: Request) -> dict[str, Any]:
        """The claim body, or `{}`. A DELETE with no body is the common one."""
        raw = await request.body()
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(
                400, "invalid_request", f"the peer body is not JSON: {exc}"
            ) from None
        if not isinstance(parsed, dict):
            raise ApiError(
                400,
                "invalid_request",
                f"the peer body is an object; a bare {type(parsed).__name__} "
                "says nothing about who is claiming",
            )
        return parsed

    def _uptime_s(request: Request) -> float:
        """MONOTONIC seconds since this process started serving.

        It is what tells an orchestrator that an engine answering again is a
        NEW process rather than the one it claimed — the signal that a
        re-claim is owed. A wall clock would make that signal lie across an
        NTP correction, which is the reason `started_at` is monotonic in the
        first place.
        """
        return time.monotonic() - request.app.state.started_at
