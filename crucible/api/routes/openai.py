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
from ...engines import (
    chat_admission,
    chat_prefill_reading,
    structured_output_reading,
)
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
from ...structured import (
    COMPACT,
    constrained_fields,
    refuse_llguidance_grammar_with_thinking,
    refuse_unbuilt_llguidance_grammar,
    refuse_unenforced_constraint,
    refuse_unkept_json_whitespace,
    refuse_upstream_json_whitespace,
    take_json_whitespace,
    with_compact_json,
    with_llguidance_grammar,
)
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


def _refuse_before_waiting(
    body: dict[str, Any],
    model: str,
    backend_kind: str,
    prefill: str | None,
    json_whitespace: str | None,
) -> None:
    """What a chat would be refused for once its model is resident is refused now,
    before it waits or a model is loaded for it: a prefill the engine cannot keep, a
    constraint it does not enforce, compact JSON it cannot keep. A model with no
    manifest, or no block here, is left to the door's own refusal."""
    if prefill is None and not constrained_fields(body):
        return
    try:
        manifest = load_manifest(model)
        spec = manifest.spec(backend_kind)
    except ManifestError:
        return
    resolved = apply_defaults(body, manifest.defaults).body
    refuse_an_unenforced_constraint(
        spec.engine, backend_kind, model, resolved, json_whitespace
    )
    if prefill is None:
        return
    reading = chat_prefill_reading(spec.engine)
    refuse_unkeepable_prefill(
        engine=spec.engine,
        model_id=model,
        served=reading.served,
        basis=reading.basis,
        resolved_body=resolved,
    )


def refuse_an_unenforced_constraint(
    engine: str,
    backend_kind: str,
    model_id: str,
    resolved_body: dict[str, Any],
    json_whitespace: str | None,
) -> bool:
    """Refuse what the engine, as built for this backend, would not keep. True when the
    chat's JSON constraint goes to it as an llguidance grammar (see `as_sent`)."""
    reading = structured_output_reading(engine, backend_kind)
    refuse_unenforced_constraint(
        engine=engine,
        model_id=model_id,
        formats=reading.formats,
        fields=reading.fields,
        basis=reading.basis,
        body=resolved_body,
    )
    refuse_unkept_json_whitespace(
        engine=engine,
        model_id=model_id,
        compact=reading.compact_json,
        basis=reading.compact_json_basis,
        mode=json_whitespace,
    )
    refuse_unbuilt_llguidance_grammar(
        engine=engine, model_id=model_id, built=reading.llguidance_grammar, body=resolved_body
    )
    if reading.llguidance_grammar:
        refuse_llguidance_grammar_with_thinking(
            engine=engine, model_id=model_id, resolved_body=resolved_body
        )
        # What it would be sent, built now so a body it cannot be written from (a
        # schema stated twice, a grammar beside it) is refused before anything waits.
        with_llguidance_grammar(resolved_body)
    return reading.llguidance_grammar


def as_sent(
    body: dict[str, Any], json_whitespace: str | None, llguidance_grammar: bool
) -> dict[str, Any]:
    """The constraint as the engine is sent it: compact JSON written into the schema,
    then, on a build that takes llguidance grammars, the schema as one."""
    if json_whitespace == COMPACT:
        body = with_compact_json(body)
    if llguidance_grammar:
        body = with_llguidance_grammar(body)
    return body


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
        "Prefill"). A `response_format` or other grammar is enforced by the engine or
        refused `structured_output_not_served` before anything is sent: vLLM,
        llama-server and mlx-lm enforce a JSON schema, mlx-vlm enforces none ("Structured
        output"). A `"json_whitespace": "compact"` member, beside a JSON schema or
        json_object, keeps the answer's JSON free of whitespace between tokens (inside
        strings only); `"flexible"` is the default. vLLM, mlx-lm and llama-server on
        cuda-linux keep it (there the schema is sent as an llguidance grammar, and a JSON
        constraint needs thinking stated off: `structured_output_with_thinking`);
        llama-server on llama-windows and mlx-vlm are refused
        `json_whitespace_not_served`, and without a JSON
        constraint it is `json_whitespace_without_json`. A `"form": "<name>"` member
        names which form of a model that comes in more than one serves the chat (GET
        /v1/models, the row's `forms`); without it the resident form answers, and a load
        made for the chat loads the form this card takes. Another form on the card is a reload; an unknown name is refused
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
        json_whitespace = take_json_whitespace(body)
        if sent_queue or prefill is not None or form is not None or json_whitespace is not None:
            raw = json.dumps(body).encode("utf-8")
        if upstreamrecord.split_model(requested) is not None:
            if form is not None:
                refuse_upstream_form(requested)
            if prefill is not None:
                refuse_upstream_prefill(requested)
            if json_whitespace is not None:
                refuse_upstream_json_whitespace(requested)
            return await forward_to_upstream(
                ctx, request, requested, body, client_agent=client_agent(request)
            )
        refuse_unknown_form(requested, form, ctx.backend.kind)
        _refuse_before_waiting(body, requested, ctx.backend.kind, prefill, json_whitespace)
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
                llguidance_grammar = refuse_an_unenforced_constraint(
                    resident.engine,
                    ctx.backend.kind,
                    resident.model_id,
                    applied.body,
                    json_whitespace,
                )
                sent = as_sent(applied.body, json_whitespace, llguidance_grammar)
                if sent is not applied.body:
                    applied = replace(applied, body=sent, changed=True)
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
