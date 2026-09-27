from __future__ import annotations

import json
import time
from typing import Any

from fastapi import Request

from ... import peer as peer_module
from ...errors import ApiError
from ..context import AppContext, Routers


async def _peer_body(request: Request) -> dict[str, Any]:
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
    return time.monotonic() - request.app.state.started_at


def register(routers: Routers, ctx: AppContext) -> None:
    peer_router = routers.peer

    @peer_router.get("")
    async def read_peer(request: Request) -> dict[str, Any]:
        """Who manages this engine, and this process's uptime."""
        state: peer_module.PeerState = request.app.state.peer
        return state.document(_uptime_s(request))

    @peer_router.post("/claim")
    async def claim_peer(request: Request) -> dict[str, Any]:
        """An orchestrator states that it manages this engine; `force` takes it from
        another orchestrator.
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
        """The orchestrator releases its claim; releasing when nothing is claimed
        succeeds.
        """
        body = await _peer_body(request)
        state: peer_module.PeerState = request.app.state.peer
        raw = body.get("orchestrator")
        who = None if raw is None else peer_module.Orchestrator.from_body(raw)
        state.release(who, force=bool(body.get("force")))
        return {"role": peer_module.ROLE_ENGINE, "managed_by": None}
