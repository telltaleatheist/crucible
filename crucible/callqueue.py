"""Queued chats and decisions: a held-open request that waits in the job line.

A chat or decision that opts in (``"queue": {"max_wait_s": N}`` in its body) is not
refused while its model is not resident or every slot on its engine is taken. It takes
a place in the same line as queued jobs (crucible/jobs/line.py) and its HTTP request
stays open. When it reaches the front:

- its model is resident with a free slot: the pump opens the in-flight slot for it and
  the request goes on to the engine exactly as an unqueued one would;
- its model is not resident and the lane is free: the pump submits a ``load-model`` job
  for it through normal admission, and the call (with every call behind it for the same
  model) is admitted into the engine's slots once the load is done.

The call never takes the lane itself, and the pump never loads a model over chats that
are still in flight. A caller who closes the connection while waiting leaves the line
(reason ``client``); an operator's DELETE /v1/queue/{id} or the wait running out ends
the request ``409 removed_from_queue``.

A chat or decision that is an item of the open queue session (crucible/queuesessions.py) waits,
when it must, ahead of everything else in the line, and its load is attributed to the
session.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from fastapi import Request, Response
from pydantic import ValidationError

from .admission import KEEPS_WAITING, AdmissionContext, JobRequest, Refusal, admit
from .engines import chat_admission
from .errors import ApiError
from .inflight import Entry, InFlight
from .jobs.base import DONE, TERMINAL_STATES
from .jobs.line import ADMITTED, CLIENT, Call, WaitingCall, WaitingLine
from .queuerequest import QueueRequest
from .queuesessions import QueueSession

IN = "in"
WAIT = "wait"
MAX_LOADS = 2


def queue_of(value: Any) -> int | None:
    """``max_wait_s`` from a chat body's ``queue`` member, or None when it has none."""
    if value is None:
        return None
    try:
        return QueueRequest.model_validate(value).max_wait_s
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'queue'}: {error['msg']}"
            for error in exc.errors()
        )
        raise ApiError(
            400,
            "invalid_request",
            f'"queue" must be an object like {{"max_wait_s": 600}} or {{}}: {problems}',
        ) from None


def _slot_free(resident: Any, inflight: InFlight) -> bool:
    limit, _ = chat_admission(resident.engine, resident.engine_args)
    return limit is None or len(inflight) < limit


def load_ended(ctx: AdmissionContext, job_id: str) -> Any | None:
    """The load job once it has ended, else None."""
    try:
        job = ctx.store.get(job_id)
    except ApiError:
        return _Reaped()
    return job if job.status in TERMINAL_STATES else None


class _Reaped:
    id = None
    status = DONE
    failure = None


def _load_failed(call: Call, job: Any) -> ApiError:
    failure = job.failure
    why = "" if failure is None else f": {failure.message}"
    return ApiError(
        502,
        "queued_load_failed",
        f"this {call.type} waited in the queue for {call.model!r}, and the load-model "
        f"job {job.id} that was loading it ended {job.status}{why}. "
        "Nothing was sent to the engine",
        {"call_id": call.id, "model": call.model, "load_job": job.id,
         "status": job.status},
    )


async def admit_call(
    waiting: WaitingCall, ctx: AdmissionContext, inflight: InFlight
) -> str | ApiError:
    """Offer one waiting call its model. IN: it has a slot. WAIT: it keeps its place and
    nothing behind it goes. An ApiError ends it with that error."""
    call = waiting.job
    if waiting.load_job is not None:
        ended = load_ended(ctx, waiting.load_job)
        if ended is None:
            return WAIT
        waiting.load_job = None
        if ended.status != DONE:
            return _load_failed(call, ended)
    try:
        granted = await _open_slot(waiting, ctx, inflight)
    except ApiError as busy:
        if busy.code in KEEPS_WAITING:
            return WAIT
        raise
    if granted is not None:
        return granted
    if not ctx.store.lane_free or len(inflight) > 0:
        return WAIT
    return await _load_for(waiting, ctx)


async def _open_slot(
    waiting: WaitingCall, ctx: AdmissionContext, inflight: InFlight
) -> str | None:
    call = waiting.job
    line = ctx.store.line
    async with ctx.residency.settled_for(f"a queued {call.type}"):
        resident = ctx.residency.resident_model
        if resident is None or resident.model_id != call.model:
            return None
        if not _slot_free(resident, inflight):
            return WAIT
        entry = inflight.open(
            act=call.act, model=call.model, client=call.client, session=call.session
        )
        if waiting.gone or line is None:
            inflight.close(entry)
            return IN
        line.started(waiting, entry)
        return IN


async def _load_for(waiting: WaitingCall, ctx: AdmissionContext) -> str | ApiError:
    call = waiting.job
    if waiting.loads >= MAX_LOADS:
        return ApiError(
            409,
            "model_not_resident",
            f"this {call.type} waited in the queue for {call.model!r}; it was loaded "
            f"{waiting.loads} time(s) for it and something else took the card each "
            "time before it was answered. Nothing was sent to the engine",
            {"call_id": call.id, "model": call.model},
        )
    outcome = await admit(
        JobRequest(
            type="load-model",
            model=call.model,
            client=call.client,
            client_ref=f"for the queued {call.type} {call.id}",
            from_the_line=True,
            session=call.session,
        ),
        ctx,
    )
    if isinstance(outcome, Refusal):
        if outcome.error.code in KEEPS_WAITING:
            return WAIT
        return outcome.error
    waiting.load_job = outcome.job.id
    waiting.loads += 1
    return WAIT


def _caller_left(call: Call) -> Response:
    from .api.deps import error_response

    return error_response(
        ApiError(
            499,
            "client_disconnected",
            f"the caller closed the connection while its {call.type} for "
            f"{call.model!r} waited in the queue; it left the queue and nothing was "
            "sent to the engine",
            {"call_id": call.id},
        )
    )


async def wait_in_line(
    request: Request,
    line: WaitingLine,
    call: Call,
    max_wait_s: int,
    undo: Callable[[Any], Awaitable[None]],
) -> Any:
    """Hold this request in the line until the pump admits it: the in-flight slot it was
    admitted with, or a Response when the caller left while it waited. ``undo`` gives
    back what was admitted when the caller left at the same moment."""
    from .api.proxy import unless_the_caller_leaves

    item = line.join_call(call, max_wait_s)
    assert item.outcome is not None
    try:
        outcome = await unless_the_caller_leaves(asyncio.shield(item.outcome), request)
    finally:
        if not item.gone:
            line.remove(
                item.job.id, CLIENT,
                "the caller closed the connection while it waited",
            )
    if outcome is None:
        if item.outcome.done():
            status, value = item.outcome.result()
            if status == ADMITTED and value is not None:
                await undo(value)
        return _caller_left(item.job)
    status, value = outcome
    if status == ADMITTED:
        return value
    raise value


async def take_a_turn(
    request: Request,
    *,
    line: WaitingLine,
    residency: Any,
    inflight: InFlight,
    settle: Any,
    kind: str,
    model: str,
    act: str | None,
    client: str | None,
    max_wait_s: int,
    session: QueueSession | None = None,
) -> Entry | Response:
    """An open in-flight slot on ``model`` for this request, waiting in the line for it
    when it must; a Response when the caller left while it waited. Raises the line's
    refusal (``queue_full``) or the call's own ending (``removed_from_queue``, a failed
    load, a refusal of the load). An item of the open ``session`` waits only behind the
    session's own items; anyone else waits behind the whole line, and behind the open
    session whatever its line holds."""
    session_id = None if session is None else session.id
    if session is not None:
        nothing_ahead = not line.items_of(session.id)
    else:
        nothing_ahead = len(line) == 0 and line.sessions.current() is None
    if nothing_ahead:
        async with residency.settled_for(f"a queued {kind}"):
            resident = residency.resident_model
            if (
                resident is not None
                and resident.model_id == model
                and _slot_free(resident, inflight)
            ):
                return inflight.open(act=act, model=model, client=client, session=session_id)

    async def give_back(entry: Entry) -> None:
        inflight.close(entry)
        await settle()

    return await wait_in_line(
        request, line,
        Call(type=kind, model=model, client=client, act=act, session=session_id),
        max_wait_s, give_back,
    )


__all__ = [
    "IN", "WAIT", "admit_call", "load_ended", "queue_of", "take_a_turn", "wait_in_line",
]
