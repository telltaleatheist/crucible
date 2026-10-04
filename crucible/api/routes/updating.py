"""`POST|DELETE /v1/server/updating`: the hold a deploy takes before it restarts this server.

See crucible/updating.py for why the hold comes before the question "is anything
working?" rather than after it.
"""
from __future__ import annotations

from typing import Any

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field

from ...errors import ApiError
from ...updating import DEFAULT_HOLD_S, MAX_HOLD_S
from ..caller import client_agent
from ..context import AppContext, Routers

SERVER_WORKING = "server_working"


class HoldRequest(BaseModel):
    """`POST /v1/server/updating`."""

    model_config = ConfigDict(extra="forbid")

    seconds: int = Field(
        default=DEFAULT_HOLD_S,
        ge=1,
        le=MAX_HOLD_S,
        description="How long the hold stands if nothing restarts the server or lets it go.",
    )
    release: str | None = Field(
        default=None, description="The release about to be installed, for the refusals to name."
    )


def working(ctx: AppContext) -> list[str]:
    """Everything a restart would cut short, one line each; empty when nothing would be."""
    store, residency = ctx.store, ctx.residency
    found: list[str] = []
    running = store.running
    if running is not None:
        found.append(
            f"job {running.id} ({running.type}) {round(running.progress * 100)}% done"
            + ("" if running.client is None else f" for {running.client}")
        )
    queued = store.queued(calls=True)
    if queued:
        found.append(f"{len(queued)} waiting in the queue")
    session = ctx.sessions.current()
    if session is not None:
        found.append(f"a queue session held by {session.client or 'an unnamed client'}")
    chats = len(ctx.inflight)
    if chats:
        found.append(f"{chats} chat completion(s) in flight")
    if ctx.streams.session is not None:
        found.append("a TTS stream")
    task = ctx.tasks.running
    if task is not None:
        found.append(f"task {task.id} ({task.type})")
    if residency.warming is not None:
        found.append(f"a model loading: {residency.warming}")
    if residency.claimed_by is not None:
        found.append(f"the card claimed by {residency.claimed_by}")
    return found


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private

    @private.post("/server/updating")
    async def hold_for_an_update(request: Request, body: HoldRequest) -> dict[str, Any]:
        """Stop admitting work so a deploy can restart this server, if nothing is working.
        The hold is taken first and the server read second, so nothing slips in between:
        idle, it answers `holding: true` and every door that creates work refuses
        `503 server_updating` until the restart, a `DELETE`, or `seconds` pass; working,
        it lets the hold go at once and refuses `409 server_working`, naming the work.
        """
        holding = ctx.updating.begin(
            seconds=body.seconds, release=body.release, by=client_agent(request)
        )
        busy = working(ctx)
        if busy:
            ctx.updating.end()
            raise ApiError(
                409,
                SERVER_WORKING,
                "this server is working, so it was not held for an update and goes on "
                "admitting work: " + "; ".join(busy) + ". A restart now would cut that "
                "short; ask again when it is done",
                {"working": busy},
            )
        return {"holding": True, **holding.to_dict()}

    @private.delete("/server/updating")
    async def release_the_hold() -> dict[str, Any]:
        """Let go of an update hold (a deploy whose install failed): work is admitted again."""
        was = ctx.updating.end()
        return {"holding": False, "released": None if was is None else was.to_dict()}
