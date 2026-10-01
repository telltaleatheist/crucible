from __future__ import annotations

from datetime import datetime
from typing import Any

import json

import httpx
from fastapi import Request, Response
from starlette.background import BackgroundTask

from ... import upstreamrecord
from ...callqueue import queue_of, take_a_turn
from ...engines import chat_admission
from ...errors import ApiError
from ...inflight import read_act
from ...sampling import SAMPLING_HEADER, apply_defaults
from ..caller import client_agent
from ..context import AppContext, Routers
from ..proxy import (
    JSON_HEADERS,
    after_the_stream,
    caller_gone,
    chat_body,
    chat_queue_full,
    engine_unreachable,
    forward_body,
    model_not_resident,
    post_unless_the_caller_leaves,
    proxy_stream,
    refuse_an_exited_engine,
    restore_model_id,
    sent_across_the_wire,
    settle_after_chat,
)
from ..upstream import forward_to_upstream, routed_upstream_rows


def register(routers: Routers, ctx: AppContext) -> None:
    private, openai = routers.private, routers.openai
    config, residency = ctx.config, ctx.residency

    @private.get("/openai/models")
    @openai.get("/models")
    async def openai_models() -> dict[str, Any]:
        """The resident model in OpenAI's list shape, plus every upstream model a route
        names.
        """
        resident = residency.resident_model
        data: list[dict[str, Any]] = routed_upstream_rows(config)
        if resident is None:
            return {"object": "list", "data": data}
        return {
            "object": "list",
            "data": [
                {
                    "id": resident.model_id,
                    "object": "model",
                    "created": int(
                        datetime.fromisoformat(resident.loaded_at).timestamp()
                    ),
                    "owned_by": config.name,
                    "engine_model_name": resident.engine_model_name,
                    "revision": resident.revision,
                    "fingerprint": resident.fingerprint,
                    "max_model_len": resident.max_model_len,
                    "defaults": resident.defaults.to_dict(),
                },
                *data,
            ],
        }

    @private.post("/openai/chat/completions")
    @openai.post("/chat/completions")
    async def openai_chat_completions(request: Request) -> Response:
        """An OpenAI chat completion, proxied to the resident engine or, for a
        `<upstream>/<id>` model, forwarded to that upstream.
        """
        raw = await request.body()
        body = chat_body(raw)
        requested = body.get("model")
        if not isinstance(requested, str) or requested == "":
            raise ApiError(
                400,
                "model_required",
                "a chat request must name a model; this server proxies only to the "
                "model that is resident",
            )
        queued: int | None = None
        if "queue" in body:
            queued = queue_of(body.pop("queue"))
            raw = json.dumps(body).encode("utf-8")
        if upstreamrecord.split_model(requested) is not None:
            return await forward_to_upstream(
                ctx, request, requested, body, client_agent=client_agent(request)
            )
        inflight = ctx.inflight
        act = read_act(request.headers)
        chat_over = settle_after_chat(ctx.settlement)
        turn: Any = None
        if queued is not None:
            turn = await take_a_turn(
                request, line=ctx.line, residency=residency, inflight=inflight,
                settle=chat_over, kind="chat", model=requested, act=act,
                client=client_agent(request), max_wait_s=queued,
            )
            if isinstance(turn, Response):
                return turn
        try:
            async with residency.settled_for("a chat request"):
                resident = residency.resident_model
                if resident is None or resident.model_id != requested:
                    raise model_not_resident(requested, resident, "a chat request")
                refuse_an_exited_engine(residency, resident)

                applied = apply_defaults(body, resident.defaults)
                sampling_headers = {SAMPLING_HEADER: applied.header()}
                forwarded = forward_body(raw, applied, resident)
                url = f"{resident.base_url}/v1/chat/completions"
                client = ctx.http

                if turn is not None:
                    entry = turn
                else:
                    limit, limit_basis = chat_admission(
                        resident.engine, resident.engine_args
                    )
                    if limit is not None and len(inflight) >= limit:
                        wait = inflight.retry_after()
                        return chat_queue_full(
                            resident=resident, limit=limit, basis=limit_basis, wait=wait
                        )
                    entry = inflight.open(
                        act=act, model=resident.model_id, client=client_agent(request)
                    )
        except BaseException:
            if turn is not None:
                inflight.close(turn)
                await chat_over()
            raise
        try:
            if body.get("stream") is True:
                return await proxy_stream(
                    client,
                    url,
                    forwarded,
                    resident,
                    sampling_headers,
                    when_relayed=after_the_stream(inflight, entry, chat_over),
                )
            try:
                upstream = await sent_across_the_wire(
                    lambda: post_unless_the_caller_leaves(
                        client, url, forwarded, request, JSON_HEADERS
                    ),
                    where=f"the engine serving {resident.model_id!r}",
                )
            except httpx.HTTPError as exc:
                raise engine_unreachable(resident, exc) from None
            if upstream is None:
                response = caller_gone(resident)
            else:
                content = upstream.content
                if upstream.status_code == 200:
                    content = restore_model_id(content, resident)
                response = Response(
                    content=content,
                    status_code=upstream.status_code,
                    media_type=upstream.headers.get(
                        "content-type", "application/json"
                    ),
                    headers=sampling_headers,
                )
            inflight.close(entry)
            response.background = BackgroundTask(chat_over)
            return response
        except BaseException:
            inflight.close(entry)
            await chat_over()
            raise
