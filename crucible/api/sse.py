from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

from fastapi import Request
from fastapi.responses import StreamingResponse

from ..errors import ApiError
from ..events import ENDS, OVERFLOW, SNAPSHOT, STOPPING, EventHub
from ..jobs.base import Job
from ..jobs.line import WaitingLine
from ..jobs.queue import JobStore
from ..tasks import Task, TaskStore
from ..ttsstream import StreamSession
from .context import hub_of

TERMINAL_EVENTS = frozenset({"done", "failed", "cancelled", "removed"})
SESSION_END = frozenset({"closed"})
KEEPALIVE_SECONDS = 15.0
SSE_HEADERS = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}


def last_event_id(request: Request) -> int:
    raw = request.headers.get("last-event-id")
    if raw is None:
        return 0
    try:
        value = int(raw.strip())
    except ValueError:
        raise ApiError(
            400,
            "invalid_last_event_id",
            f"Last-Event-ID must be an integer, got {raw!r}",
        ) from None
    if value < 0:
        raise ApiError(
            400, "invalid_last_event_id", f"Last-Event-ID must not be negative: {value}"
        )
    return value


def format_event(event: dict[str, Any]) -> str:
    return (
        f"id: {event['id']}\n"
        f"event: {event['event']}\n"
        f"data: {json.dumps(event['data'], separators=(',', ':'))}\n\n"
    )


@dataclass(frozen=True)
class Feed:
    after: Callable[[int], list[tuple[int, dict[str, Any]]]]
    waiter: asyncio.Event
    ends: frozenset[str]
    moved: Callable[[int], None]
    close: Callable[[], None]


def stopping_frame(hub: EventHub) -> str:
    """The last thing every stream says when the server stops. It carries no `id`, so a
    client's Last-Event-ID stays the last real event and its reconnect resumes there."""
    data = {"reason": hub.stop_reason}
    return f"event: {STOPPING}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


async def events_after(
    request: Request, open_feed: Callable[[], Feed], cursor: int
) -> AsyncIterator[str]:
    """Every SSE stream this server serves runs here, and ends when the server stops:
    uvicorn waits for open responses before it shuts down, and a stream left to itself
    never ends (crucible/api/serving.py)."""
    hub: EventHub = hub_of(request.app)
    feed = open_feed()
    watched = False
    try:
        watched = hub.watch_stop(feed.waiter)
        while True:
            for position, event in feed.after(cursor):
                cursor = position
                feed.moved(cursor)
                yield format_event(event)
                if event["event"] in feed.ends:
                    return
            if hub.stopped:
                yield stopping_frame(hub)
                return
            feed.waiter.clear()
            if feed.after(cursor) or hub.stopped:
                continue
            try:
                await asyncio.wait_for(feed.waiter.wait(), timeout=KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    return
                yield ": keepalive\n\n"
    finally:
        if watched:
            hub.unwatch_stop(feed.waiter)
        feed.close()


def _job_feed(store: JobStore | TaskStore, job: Job | Task) -> Feed:
    waiter = store.subscribe(job)

    def after(index: int) -> list[tuple[int, dict[str, Any]]]:
        return [
            (position, event)
            for position, event in enumerate(job.events[index:], start=index + 1)
        ]

    return Feed(
        after=after,
        waiter=waiter,
        ends=TERMINAL_EVENTS,
        moved=lambda _: None,
        close=lambda: store.unsubscribe(job, waiter),
    )


def _session_feed(session: StreamSession, delivered: int) -> Feed:
    reader = session.attach(delivered)

    def after(cursor: int) -> list[tuple[int, dict[str, Any]]]:
        return [
            (frame.id, {"id": frame.id, "event": frame.event, "data": frame.data})
            for frame in session.frames_after(cursor)
        ]

    def moved(cursor: int) -> None:
        reader.delivered = cursor

    return Feed(
        after=after,
        waiter=reader.waiter,
        ends=SESSION_END,
        moved=moved,
        close=lambda: session.detach(reader),
    )


def _queue_feed(line: WaitingLine) -> Feed:
    waiter = line.subscribe()
    opened = line.last_event_id
    snapshot: list[tuple[int, dict[str, Any]]] = [
        (opened, {"id": opened, "event": "snapshot",
                  "data": {"items": line.rows(), "depth": len(line)}})
    ]

    def after(cursor: int) -> list[tuple[int, dict[str, Any]]]:
        first = snapshot if cursor < opened else []
        return first + [
            (event["id"], event) for event in line.events_after(max(cursor, opened))
        ]

    return Feed(
        after=after,
        waiter=waiter,
        ends=frozenset(),
        moved=lambda _: None,
        close=lambda: line.unsubscribe(waiter),
    )


def queue_events(request: Request, line: WaitingLine) -> StreamingResponse:
    return event_response(events_after(request, lambda: _queue_feed(line), 0))


def resume_from(request: Request) -> int | None:
    """The Last-Event-ID a reconnecting client sent, or None from a first connect."""
    if request.headers.get("last-event-id") is None:
        return None
    return last_event_id(request)


def _hub_feed(
    hub: EventHub,
    topics: frozenset[str],
    after: int | None,
    snapshot: Callable[[], dict[str, Any]],
) -> Feed:
    opening = hub.subscribe(topics, after)
    subscriber = opening.subscriber
    first: list[tuple[int, dict[str, Any]]] = []
    if opening.snapshot:
        first.append((opening.id, {
            "id": opening.id,
            "event": SNAPSHOT,
            "data": {"gap": opening.gap, "topics": sorted(subscriber.topics), **snapshot()},
        }))

    def after_cursor(cursor: int) -> list[tuple[int, dict[str, Any]]]:
        if subscriber.overflowed:
            return [(cursor, {"id": cursor, "event": OVERFLOW, "data": {
                "last_event_id": cursor,
                "limit": subscriber.limit,
                "message": (
                    f"this stream fell {subscriber.limit} events behind and was dropped "
                    "so it could not hold the server up. Reconnect with Last-Event-ID: "
                    f"{cursor} to pick up where it stopped"
                ),
            }})]
        return first + [(event["id"], event) for event in hub.pending(subscriber, cursor)]

    def moved(cursor: int) -> None:
        first.clear()
        hub.delivered(subscriber, cursor)

    return Feed(
        after=after_cursor,
        waiter=subscriber.waiter,
        ends=ENDS,
        moved=moved,
        close=lambda: hub.unsubscribe(subscriber),
    )


def server_events(
    request: Request,
    hub: EventHub,
    topics: frozenset[str],
    snapshot: Callable[[], dict[str, Any]],
) -> StreamingResponse:
    after = resume_from(request)
    return event_response(
        events_after(
            request,
            lambda: _hub_feed(hub, topics, after, snapshot),
            0 if after is None else after,
        )
    )


def event_response(stream: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(stream, media_type="text/event-stream", headers=SSE_HEADERS)


def job_events(
    request: Request, store: JobStore | TaskStore, job: Job | Task, delivered: int
) -> StreamingResponse:
    return event_response(
        events_after(request, lambda: _job_feed(store, job), delivered)
    )


def session_events(
    request: Request, session: StreamSession, delivered: int
) -> StreamingResponse:
    return event_response(
        events_after(request, lambda: _session_feed(session, delivered), delivered)
    )
