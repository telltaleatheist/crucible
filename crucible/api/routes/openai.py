from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from typing import Any

import httpx
from fastapi import Request, Response
from starlette.background import BackgroundTask

from ... import upstreamrecord
from ...callqueue import take_a_turn
from ...engines import chat_admission, chat_prefill_reading
from ...errors import ApiError
from ...formrequest import refuse_unknown_form, refuse_upstream_form, take_form
from ...inflight import read_act
from ...manifests import ManifestError, load_manifest
from ...prefill import (
    refuse_unkeepable_prefill,
    refuse_upstream_prefill,
    take_prefill,
    with_prefill,
)
from ...queuerequest import queue_of
from ...residency import serves_model
from ...sampling import SAMPLING_HEADER, apply_defaults
from ..caller import client_agent, queue_session
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


def _refuse_a_prefill_before_waiting(
    body: dict[str, Any], model: str, backend_kind: str
) -> None:
    """What a queued chat's prefill would be refused for once its model is resident
    is refused now, before it waits or a model is loaded for it. A model with no
    manifest, or no block here, is left to the door's own refusal."""
    try:
        manifest = load_manifest(model)
        spec = manifest.spec(backend_kind)
    except ManifestError:
        return
    reading = chat_prefill_reading(spec.engine)
    refuse_unkeepable_prefill(
        engine=spec.engine,
        model_id=model,
        served=reading.served,
        basis=reading.basis,
        resolved_body=apply_defaults(body, manifest.defaults).body,
    )


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
                    "form": resident.form,
                },
                *data,
            ],
        }

    @private.post("/openai/chat/completions")
    @openai.post("/chat/completions")
    async def openai_chat_completions(request: Request) -> Response:
        """An OpenAI chat completion, proxied to the resident engine or, for a
        `<upstream>/<id>` model, forwarded to that upstream. A chat whose model is not
        resident, or whose engine has every slot taken, waits in the server's line
        (up to an hour, or `queue.max_wait_s`) and its model is loaded for it; with
        `"queue": false` it is refused at once instead. An upstream chat never waits.
        A `"prefill": "<text>"` member starts the answer with that text and the model
        writes on from it; the reply's content is what it wrote after the prefill
        (vLLM and llama-server; thinking stated off; no response_format or other
        grammar; refused by name otherwise: docs/internals/engines-and-capability.md,
        "Prefill"). A `"form": "<name>"` member names which form of a model that comes in
        more than one serves the chat (GET /v1/models, the row's `forms`); without it the
        resident form answers, and a load made for the chat loads the form this card
        takes. Another form on the card is a reload; an unknown name is refused
        `unknown_form` (docs/FITS-AND-THE-CARD.md section 8). It is never forwarded.
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
        sent_queue = "queue" in body
        queued = queue_of(body)
        prefill = take_prefill(body)
        form = take_form(body)
        if sent_queue or prefill is not None or form is not None:
            raw = json.dumps(body).encode("utf-8")
        if upstreamrecord.split_model(requested) is not None:
            if form is not None:
                refuse_upstream_form(requested)
            if prefill is not None:
                refuse_upstream_prefill(requested)
            return await forward_to_upstream(
                ctx, request, requested, body, client_agent=client_agent(request)
            )
        refuse_unknown_form(requested, form, ctx.backend.kind)
        if prefill is not None:
            _refuse_a_prefill_before_waiting(body, requested, ctx.backend.kind)
        inflight = ctx.inflight
        act = read_act(request.headers)
        chat_over = settle_after_chat(ctx.settlement)
        session = queue_session(request, ctx.sessions)
        session_id = None if session is None else session.id
        turn: Any = None
        if queued is not None:
            turn = await take_a_turn(
                request, line=ctx.line, residency=residency, inflight=inflight,
                settle=chat_over, kind="chat", model=requested, act=act,
                client=client_agent(request), max_wait_s=queued, session=session,
                form=form,
            )
            if isinstance(turn, Response):
                return turn
        else:
            ctx.sessions.refuse_call_if_held(session_id, "a chat")
        if session is not None:
            ctx.sessions.item_arrived(session)
        try:
            async with residency.settled_for("a chat request"):
                resident = residency.resident_model
                if not serves_model(resident, requested, form):
                    raise model_not_resident(requested, resident, "a chat request", form)
                refuse_an_exited_engine(residency, resident)

                applied = apply_defaults(body, resident.defaults)
                if prefill is not None:
                    reading = chat_prefill_reading(resident.engine)
                    refuse_unkeepable_prefill(
                        engine=resident.engine,
                        model_id=resident.model_id,
                        served=reading.served,
                        basis=reading.basis,
                        resolved_body=applied.body,
                    )
                    applied = replace(
                        applied, body=with_prefill(applied.body, prefill), changed=True
                    )
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
                        act=act, model=resident.model_id, client=client_agent(request),
                        session=session_id,
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
