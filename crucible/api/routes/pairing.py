from __future__ import annotations

from typing import Any

from fastapi import Depends, Request, Response

from ... import API_VERSION
from ...connect import DecidePairing, PollPairing, StartPairing
from ..deps import require_api_version
from ..context import AppContext, Routers


def register(routers: Routers, ctx: AppContext) -> None:
    public, private = routers.public, routers.private
    app, config = ctx.app, ctx.config

    # ------------------------------------------------------------------ ping

    @public.get("/ping")
    async def ping() -> dict[str, Any]:
        """Unauthenticated. Lets a client tell "wrong token" from "not a Crucible"."""
        return {"crucible": True, "name": config.name, "api_version": API_VERSION,
                "pairing_version": 1}

    @public.post("/pairing/start", dependencies=[Depends(require_api_version)])
    async def start_pairing(body: StartPairing, request: Request, response: Response) -> dict:
        response.headers["Cache-Control"] = "no-store"
        address = request.client.host if request.client is not None else "unknown"
        return {"name": config.name, **app.state.pairing_requests.start(body.client_name, address)}

    @public.post("/pairing/poll", dependencies=[Depends(require_api_version)])
    async def poll_pairing(body: PollPairing, response: Response) -> dict:
        response.headers["Cache-Control"] = "no-store"
        status = app.state.pairing_requests.poll(body.id, body.device_code)
        result = {"status": status}
        if status == "approved":
            result.update(name=config.name, token=config.token)
        return result

    @private.get("/pairing/requests")
    async def pairing_requests(response: Response) -> dict:
        response.headers["Cache-Control"] = "no-store"
        return {"requests": app.state.pairing_requests.pending()}

    @private.post("/pairing/decision")
    async def decide_pairing(body: DecidePairing, response: Response) -> dict:
        response.headers["Cache-Control"] = "no-store"
        return app.state.pairing_requests.decide(body.id, body.user_code, body.allow)
