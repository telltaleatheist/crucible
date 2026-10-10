"""`POST|DELETE /v1/server/updating`: the hold a deploy takes before it restarts this server.

See crucible/updating.py for why the hold comes before the question "is anything
working?" rather than after it.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field

from ... import selfrestart, service
from ...errors import ApiError
from ...inflight import read_act
from ...tasks import hostdoor
from ...updating import DEFAULT_HOLD_S, FOR_SETTINGS, MAX_HOLD_S
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


def _working_refusal(busy: list[str], what: str) -> ApiError:
    return ApiError(
        409,
        SERVER_WORKING,
        f"this server is working, so it was not {what} and goes on admitting work: "
        + "; ".join(busy)
        + ". A restart now would cut that short; ask again when it is done",
        {"working": busy},
    )


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
            raise _working_refusal(busy, "held for an update")
        return {"holding": True, **holding.to_dict()}

    @private.post("/server/restart", status_code=202)
    async def restart_the_server(request: Request) -> dict[str, Any]:
        """Restart this server so it takes up `host` and `port` (and anything else a
        start reads), if nothing is working. Refused `409 server_working` naming the
        work, as `POST /v1/server/updating` is. A server a Windows host started is
        restarted by that host (an `engine-restart` task, `task_id`); one systemd or
        launchd runs stops cleanly and its service manager starts it again (`by`).
        Refused `restart_not_supervised` when nothing would start it again (started
        from a shell), and `restart_needs_service_install` when the service definition
        starts it on another address than the config: `details.command` rewrites it
        and restarts the server. Answers `202` before it stops; poll `GET /v1/ping`
        at `url` for the server that comes back.
        """
        config, client = ctx.config, client_agent(request)
        url = f"http://{config.host}:{config.port}"
        if hostdoor.door_from_environment() != "":
            busy = working(ctx)
            if busy:
                raise _working_refusal(busy, "restarted")
            task = ctx.tasks.submit({"type": "engine-restart"})
            _record_restart(request, "orchestrator")
            return {"restarting": True, "by": "orchestrator", "task_id": task.id, "url": url}
        mechanism = await asyncio.to_thread(
            lambda: selfrestart.supervisor(
                config, ctx.backend, runner=service.subprocess_runner,
                home=service.user_home(), pid=selfrestart.this_pid(),
            )
        )
        ctx.updating.begin(seconds=DEFAULT_HOLD_S, release=None, by=client, reason=FOR_SETTINGS)
        busy = working(ctx)
        if busy:
            ctx.updating.end()
            raise _working_refusal(busy, "restarted")
        try:
            selfrestart.exit_for_restart(ctx.app)
        except ApiError:
            ctx.updating.end()
            raise
        _record_restart(request, mechanism)
        return {"restarting": True, "by": mechanism, "pid": os.getpid(), "url": url}

    def _record_restart(request: Request, by: str) -> None:
        ctx.settings_history.record(
            act=read_act(request.headers),
            client=client_agent(request),
            changed=[f"server restart asked ({by} starts it again)"],
        )

    @private.delete("/server/updating")
    async def release_the_hold() -> dict[str, Any]:
        """Let go of an update hold (a deploy whose install failed): work is admitted again."""
        was = ctx.updating.end()
        return {"holding": False, "released": None if was is None else was.to_dict()}
