from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
from fastapi import Request, Response
from starlette.background import BackgroundTask

from ... import upstreams
from ...config import Config
from ...engines import chat_admission
from ...errors import ApiError
from ...inflight import InFlight, read_act
from ...sampling import SAMPLING_HEADER, apply_defaults
from ...settle import Settlement
from ..caller import client_agent
from ..context import AppContext, Routers
from ..proxy import (
    JSON_HEADERS,
    _after_the_stream,
    _caller_gone,
    _chat_body,
    _chat_over,
    _chat_queue_full,
    _engine_unreachable,
    _forward_body,
    _model_not_resident,
    _post_unless_the_caller_leaves,
    _proxy_stream,
    _refuse_an_exited_engine,
    _restore_model_id,
    _sent_across_the_wire,
)
from ..upstream import _forward_to_upstream, _routed_upstream_rows


def register(routers: Routers, ctx: AppContext) -> None:
    private, openai = routers.private, routers.openai
    config, residency = ctx.config, ctx.residency

    @private.get("/openai/models")
    @openai.get("/models")
    async def openai_models(request: Request) -> dict[str, Any]:
        """The resident model in OpenAI's list shape, plus every upstream model a route
        names.
        """
        live: Config = request.app.state.config
        resident = residency.resident_model
        data: list[dict[str, Any]] = _routed_upstream_rows(live)
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
        body = _chat_body(raw)
        requested = body.get("model")
        if not isinstance(requested, str) or requested == "":
            raise ApiError(
                400,
                "model_required",
                "a chat request must name a model; this server proxies only to the "
                "model that is resident",
            )
        if upstreams.split_model(requested) is not None:
            return await _forward_to_upstream(
                request, requested, body, client_agent=client_agent(request)
            )
        async with residency.settled_for("a chat request"):
            resident = residency.resident_model
            if resident is None or resident.model_id != requested:
                raise _model_not_resident(requested, resident, "a chat request")
            _refuse_an_exited_engine(residency, resident)

            applied = apply_defaults(body, resident.defaults)
            sampling_headers = {SAMPLING_HEADER: applied.header()}
            forwarded = _forward_body(raw, applied, resident)
            url = f"{resident.base_url}/v1/chat/completions"
            client: httpx.AsyncClient = request.app.state.http
            inflight: InFlight = request.app.state.inflight
            act = read_act(request.headers)

            settlement: Settlement = request.app.state.settlement
            chat_over = _chat_over(settlement)

            limit, limit_basis = chat_admission(resident.engine, resident.engine_args)
            if limit is not None and len(inflight) >= limit:
                wait = inflight.retry_after()
                return _chat_queue_full(
                    resident=resident, limit=limit, basis=limit_basis, wait=wait
                )

            entry = inflight.open(
                act=act, model=resident.model_id, client=client_agent(request)
            )
        try:
            if body.get("stream") is True:
                return await _proxy_stream(
                    client,
                    url,
                    forwarded,
                    resident,
                    sampling_headers,
                    when_relayed=_after_the_stream(inflight, entry, chat_over),
                )
            try:
                upstream = await _sent_across_the_wire(
                    lambda: _post_unless_the_caller_leaves(
                        client, url, forwarded, request, JSON_HEADERS
                    ),
                    where=f"the engine serving {resident.model_id!r}",
                )
            except httpx.HTTPError as exc:
                raise _engine_unreachable(resident, exc) from None
            if upstream is None:
                response = _caller_gone(resident)
            else:
                content = upstream.content
                if upstream.status_code == 200:
                    content = _restore_model_id(content, resident)
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
