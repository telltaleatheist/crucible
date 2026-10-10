"""One call to the resident model on behalf of a verb's door (embed, rerank), taken through
the server's line exactly as a decision is (api/routes/decide.py): it waits for its turn
and its model is loaded for it, or with `"queue": false` it is refused at once; it holds
an in-flight slot on the engine while it runs; it is cancelled when its caller leaves; and
it says how long it waited (`timing_ms.queued`) apart from how long it ran."""

from __future__ import annotations

import sys
import time
from typing import Any, Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask

from .. import enginespec
from ..callqueue import take_a_turn
from ..engines import chat_admission
from ..inflight import read_act
from ..queuerequest import max_wait_of
from ..residency import serves_model
from .caller import client_agent, queue_session
from .context import AppContext
from .proxy import (
    caller_gone,
    chat_queue_full,
    model_not_resident,
    refuse_an_exited_engine,
    settle_after_chat,
    unless_the_caller_leaves,
)

Prepare = Callable[[Any, int], Awaitable[Any]]
"""Given the resident model and the engine's admission, the call's work. A plain function
returning the awaitable, so the refusals it can only make with the model resident are
raised as it is called, before the slot is taken."""


async def serve_verb(
    request: Request,
    ctx: AppContext,
    *,
    kind: str,
    what: str,
    model: str,
    form: str | None,
    queue: Any,
    prepare: Prepare,
    render: Callable[[Any], dict[str, Any]],
) -> Response:
    """`what` names the call in refusals ("an embedding"); `render` turns the answer
    (a model with `timing_ms.queued`) into the response's JSON."""
    arrived = time.monotonic()
    residency, inflight = ctx.residency, ctx.inflight
    chat_over = settle_after_chat(ctx.settlement)
    session = queue_session(request, ctx.sessions)
    session_id = None if session is None else session.id
    act = read_act(request.headers)
    turn: Any = None
    max_wait_s = max_wait_of(queue)
    if max_wait_s is not None:
        turn = await take_a_turn(
            request, line=ctx.line, residency=residency, inflight=inflight,
            settle=chat_over, kind=kind, model=model, act=act,
            client=client_agent(request), max_wait_s=max_wait_s,
            session=session, form=form,
        )
        if isinstance(turn, Response):
            return turn
    else:
        ctx.sessions.refuse_call_if_held(session_id, what)
    if session is not None:
        ctx.sessions.item_arrived(session)
    try:
        async with residency.settled_for(what):
            resident = residency.resident_model
            if not serves_model(resident, model, form):
                raise model_not_resident(model, resident, what, form)
            refuse_an_exited_engine(residency, resident)
            limit, basis = chat_admission(resident.engine, resident.engine_args)
            if turn is None and limit is not None and len(inflight) >= limit:
                return chat_queue_full(
                    resident=resident, limit=limit, basis=basis, wait=inflight.retry_after(),
                )
            concurrency = limit if limit is not None else enginespec.UNSTATED_ENGINE_CONCURRENCY
            work = prepare(resident, concurrency)
            entry = turn if turn is not None else inflight.open(
                act=act, model=resident.model_id, client=client_agent(request),
                session=session_id,
            )
    except BaseException:
        if turn is not None:
            inflight.close(turn)
            await chat_over()
        raise
    try:
        started = time.monotonic()
        answered = await unless_the_caller_leaves(work, request)
        if answered is None:
            response: Response = caller_gone(resident)
        else:
            answered.timing_ms.queued = round((started - arrived) * 1000.0, 1)
            response = JSONResponse(content=render(answered))
            print(
                f"crucible: {kind} on {model!r} for "
                f"{client_agent(request) or 'an unnamed client'}: waited "
                f"{(started - arrived) * 1000:.0f} ms, answered in "
                f"{(time.monotonic() - started) * 1000:.0f} ms",
                file=sys.stderr,
            )
        inflight.close(entry)
        response.background = BackgroundTask(chat_over)
        return response
    except BaseException:
        inflight.close(entry)
        await chat_over()
        raise


__all__ = ["Prepare", "serve_verb"]
