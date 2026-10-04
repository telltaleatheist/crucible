from __future__ import annotations

from typing import Any

from fastapi import Depends, Request, Response

from ...connect import DecidePairing, PollPairing, StartPairing
from ...protocol import API_VERSION
from ..context import AppContext, Routers
from ..deps import require_api_version
from ..responses import Ping


def register(routers: Routers, ctx: AppContext) -> None:
    public, private = routers.public, routers.private
    config = ctx.config

    @public.get("/ping", response_model=Ping, response_model_exclude_unset=True)
    async def ping() -> dict[str, Any]:
        """Unauthenticated. Lets a client tell "wrong token" from "not a Crucible"."""
        return {"crucible": True, "name": config.name, "api_version": API_VERSION,
                "pairing_version": 1}

    @public.post("/pairing/start", dependencies=[Depends(require_api_version)])
    async def start_pairing(body: StartPairing, request: Request, response: Response) -> dict:
        """Ask this server for a token. Answers an `id`, a `device_code` to poll with and a
        short `user_code` the operator compares before approving it (on the operator page,
        or `POST /v1/pairing/decision`). No token is needed to ask; one is needed to approve.
        """
        response.headers["Cache-Control"] = "no-store"
        address = request.client.host if request.client is not None else "unknown"
        return {"name": config.name, **ctx.pairing_requests.start(body.client_name, address)}

    @public.post("/pairing/poll", dependencies=[Depends(require_api_version)])
    async def poll_pairing(body: PollPairing, response: Response) -> dict:
        """How a pairing request stands: `pending`, `approved` (the answer then carries the
        server's `name` and its `token`), `denied` or `expired`. Poll with the `id` and
        `device_code` from `POST /v1/pairing/start`.
        """
        response.headers["Cache-Control"] = "no-store"
        status = ctx.pairing_requests.poll(body.id, body.device_code)
        result = {"status": status}
        if status == "approved":
            result.update(name=config.name, token=config.token)
        return result

    @private.get("/pairing/requests")
    async def pairing_requests(response: Response) -> dict:
        """The pairing requests waiting for an answer: who asked, from where, and the
        `user_code` to compare."""
        response.headers["Cache-Control"] = "no-store"
        return {"requests": ctx.pairing_requests.pending()}

    @private.post("/pairing/decision")
    async def decide_pairing(body: DecidePairing, response: Response) -> dict:
        """Approve (`allow: true`) or deny a pairing request, naming its `id` and the
        `user_code` the asking client shows."""
        response.headers["Cache-Control"] = "no-store"
        return ctx.pairing_requests.decide(body.id, body.user_code, body.allow)
