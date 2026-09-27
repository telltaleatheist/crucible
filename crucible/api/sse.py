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
    """The job stream's shape, over a session's log instead of a job's events.

    Deliberately a second function rather than a parameterised one. The two look
    alike and are not the same: a job's events end at a terminal status and its
    log lives as long as the job does, while a session's end at `closed` and its
    log is pruned behind the readers (`StreamSession._prune`), so the cursor here
    has to be written back onto the reader rather than kept local. Folding them
    together would mean one of the two behaviours becoming a flag.
    """
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
                # The job stream's shape, byte for byte, and measured on
                # 2026-09-13 to be enough rather than assumed to be. A real
                # socket close cancels this generator through starlette's
                # disconnect listener in about 0.17 s, so this poll is for the
                # OTHER kind of departure: a tunnel that died without closing
                # anything, where nothing but a write that fails can discover
                # it. A shorter `is_disconnected()` tick was written, measured
                # to change neither case, and taken back out.
                if await request.is_disconnected():
                    return
                yield ": keepalive\n\n"
    finally:
        # Detaching is what starts the grace window. A dropped stream does not
        # cancel immediately — that is the whole reason this door is SSE — so
        # this marks the session unattended and the watchdog does the rest.
        session.detach(reader)


async def _task_event_stream(
    request: Request, tasks: TaskStore, task: Task, last_event_id: int
) -> AsyncIterator[str]:
    """A task's events, in the job stream's shape and with its own terminal set.

    A third copy of this loop rather than a parameterised one, which is the
    call `_session_event_stream` already made and for the same kind of reason:
    the three logs have three lifetimes. A job's log lives as long as its
    directory, a session's is pruned behind its readers, and a TASK's can be
    dropped whole when it ages past `HISTORY` — so this one has to survive its
    subject disappearing between two iterations, which the others never do.
    """
    waiter = tasks.subscribe(task)
    index = last_event_id
    try:
        while True:
            while index < len(task.events):
                event = task.events[index]
                index += 1
                yield _format_event(event)
                if event["event"] in TERMINAL_EVENTS:
                    return
            waiter.clear()
            if index < len(task.events):
                continue
            try:
                await asyncio.wait_for(waiter.wait(), timeout=KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    return
                yield ": keepalive\n\n"
    finally:
        tasks.unsubscribe(task, waiter)


async def _event_stream(
    request: Request, store: JobStore, job: Job, last_event_id: int
) -> AsyncIterator[str]:
    """Replay everything after `last_event_id`, then follow live until terminal."""
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
