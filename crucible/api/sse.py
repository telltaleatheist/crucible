from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from fastapi import Request

from ..errors import ApiError
from ..jobs.base import Job
from ..jobs.queue import JobStore
from ..tasks import Task, TaskStore
from ..ttsstream import StreamSession

TERMINAL_EVENTS = frozenset({"done", "failed", "cancelled"})
KEEPALIVE_SECONDS = 15.0


def _last_event_id(request: Request) -> int:
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


def _format_event(event: dict[str, Any]) -> str:
    return (
        f"id: {event['id']}\n"
        f"event: {event['event']}\n"
        f"data: {json.dumps(event['data'], separators=(',', ':'))}\n\n"
    )


async def _session_event_stream(
    request: Request, session: StreamSession, last_event_id: int
) -> AsyncIterator[str]:
    reader = session.attach(last_event_id)
    try:
        while True:
            for event in session.frames_after(reader.delivered):
                reader.delivered = event.id
                yield _format_event(
                    {"id": event.id, "event": event.event, "data": event.data}
                )
                if event.event == "closed":
                    return
            reader.waiter.clear()
            if session.frames_after(reader.delivered):
                continue
            try:
                await asyncio.wait_for(reader.waiter.wait(), timeout=KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    return
                yield ": keepalive\n\n"
    finally:
        session.detach(reader)


async def _event_stream(
    request: Request,
    store: JobStore | TaskStore,
    job: Job | Task,
    last_event_id: int,
) -> AsyncIterator[str]:
    waiter = store.subscribe(job)
    index = last_event_id
    try:
        while True:
            while index < len(job.events):
                event = job.events[index]
                index += 1
                yield _format_event(event)
                if event["event"] in TERMINAL_EVENTS:
                    return
            waiter.clear()
            if index < len(job.events):
                continue
            try:
                await asyncio.wait_for(waiter.wait(), timeout=KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    return
                yield ": keepalive\n\n"
    finally:
        store.unsubscribe(job, waiter)
