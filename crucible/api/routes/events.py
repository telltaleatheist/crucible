from __future__ import annotations

from typing import Any

from fastapi import Query, Request
from fastapi.responses import StreamingResponse

from ...events import parse_topics
from .. import sse
from ..context import AppContext, Routers
from .activity import activity_body


def snapshot(ctx: AppContext) -> dict[str, Any]:
    """What a dashboard draws before the first change: GET /v1/activity's body, the
    waiting line as GET /v1/queue lists it, and the recent tasks as GET /v1/tasks does."""
    line = ctx.line
    return {
        "activity": activity_body(ctx),
        "queue": {"items": line.rows(), "depth": len(line)},
        "tasks": [task.to_dict() for task in ctx.tasks.recent()],
    }


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private

    @private.get("/events")
    async def server_events(
        request: Request,
        topics: str | None = Query(
            None,
            description="Comma-separated topics to receive (job, queue, session, card, chat, "
            "task, settings, server). Leave it out for all; `server` is always sent.",
        ),
    ) -> StreamingResponse:
        """Every change on this server as one SSE stream, so an app need not poll: a
        `snapshot` first, then one event per change (jobs, the queue, queue sessions, the
        card, chats in flight, tasks, settings, the server stopping). Reconnect with Last-Event-ID to
        resume; the event names and payloads are in docs/EVENTS.md.
        """
        wanted = parse_topics(topics)
        return sse.server_events(request, ctx.events, wanted, lambda: snapshot(ctx))
