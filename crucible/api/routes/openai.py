from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import datetime
from typing import Any

import httpx
from fastapi import Request, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from ... import upstreamrecord, verbmodel
from ...callqueue import take_a_turn
from ...classnames import DEFAULT_CHAT_VERB
from ...embed import refuse_vectors_only
from ...engines import (
    chat_admission,
    chat_prefill_reading,
    structured_output_reading,
)
from ...errors import ApiError
from ...formrequest import refuse_unknown_form, refuse_upstream_form, take_form
from ...inflight import read_act
from ...manifests import ManifestError, load_all_manifests, load_manifest
from ...prefill import (
    refuse_unkeepable_prefill,
    refuse_upstream_prefill,
    take_prefill,
    with_prefill,
)
from ...queuerequest import QueueChoice, queue_field, queue_of
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
from ..chatlog import ChatLog
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
from ..upstream import UpstreamLists, forward_to_upstream, upstream_model_rows


class ChatCompletionBody(BaseModel):
    """The chat door's body as the API reference shows it. The door reads the raw JSON
    (every member OpenAI's API has is forwarded to the engine as sent), so this model is
    never validated against a request: it documents the members the door itself reads,
    and the ones Crucible adds."""

    model_config = ConfigDict(extra="allow", use_attribute_docstrings=True)

    model: str | None = None
    """A Crucible model id, or `<upstream>/<id>` for a routed upstream. Left out (or
    null), the chat is served by the model this server registered for `verb`, capped by
    `max_params_b`, and the reply's `model` names the one chosen."""
    messages: list[dict[str, Any]]
    """OpenAI's messages, forwarded as sent."""
    verb: str | None = None
    """Crucible's: which verb's model serves a chat that names no model, one of
    `generate` (the default: open-ended chat), `clean`, `translate`, `simplify`,
    `analysis`. Beside a `model`, that model is held to the verb's rules (it must be in
    the verb's lineup here, `model_not_for_verb`, and fit the card unless it is the verb's
    Settings choice, `model_does_not_fit`). Another verb is refused `verb_not_for_chat`,
    an unknown name `unknown_verb`."""
    max_params_b: float | None = Field(default=None, gt=0)
    """Crucible's: a ceiling, in billions of parameters, on the verb's model: this
    server's own pick (its Settings choice, else the biggest that fits up to the verb's
    goal) is kept when it is at or below the ceiling, and a pick above it is replaced by
    the biggest model that fits this card at or below the ceiling, the most parameters
    first, then the highest precision. Never raises the pick. Nothing at or below it:
    `nothing_fits_ceiling`; beside a `model`: `model_with_ceiling`."""
    stream: bool = False
    """OpenAI's: answer as server-sent events."""
    tools: list[dict[str, Any]] | None = None
    """OpenAI's: the functions the model may call, `{"type": "function", "function":
    {name, description, parameters}}`. The app defines and runs them; Crucible keeps
    none. A local model writes them in the format its manifest declares (`[tools]`) and
    the reply carries its calls as `tool_calls`; a model with no format is refused
    `tools_unsupported`, and `tools` beside a `response_format` is refused
    `tools_with_response_format` (docs/internals/engines-and-capability.md,
    "Tool calling")."""
    tool_choice: str | dict[str, Any] | None = None
    """OpenAI's: `auto` (the default with tools), `none`, `required`, or
    `{"type": "function", "function": {"name": ...}}`."""
    response_format: dict[str, Any] | None = None
    """OpenAI's: a `json_schema` or `json_object` constraint, enforced by the engine or
    refused `structured_output_not_served` before anything is sent."""
    reasoning_effort: str | None = None
    """OpenAI's: how much the model reasons. For an `anthropic/<id>` chat it is sent as
    the Messages API's `output_config.effort`, one of `low`, `medium`, `high`, `xhigh`,
    `max`, and a level the model does not take is refused `upstream_param_unsupported`
    (docs/API-CLI.md, "The anthropic upstream"). An `openai/<id>` chat forwards it as
    sent; a local engine is sent it as any other member."""
    json_whitespace: str | None = None
    """Crucible's: `compact` or `flexible` (the default), beside a JSON
    `response_format`."""
    prefill: str | None = None
    """Crucible's: the start of the answer; the reply's content is what the model wrote
    after it."""
    form: str | None = None
    """Crucible's: which form of the named `model` serves the chat. Without a `model` it
    is refused `form_without_model`."""
    queue: QueueChoice = queue_field(
        "Crucible's: absent, a chat whose model is not resident or whose engine is full "
        "waits in the server's line (up to an hour); `{\"max_wait_s\": N}` changes the "
        "wait; `false` refuses at once."
    )


def _documented_body() -> dict[str, Any]:
    """ChatCompletionBody's schema for the route's OpenAPI entry, its nested models
    referred to the document's own components (QueueRequest is one already)."""
    schema = ChatCompletionBody.model_json_schema(ref_template="#/components/schemas/{model}")
    schema.pop("$defs", None)
    return {"requestBody": {"required": True, "content": {"application/json": {"schema": schema}}}}


def refuse_tools_this_model_cannot_take(
    body: dict[str, Any], model: str, backend_kind: str
) -> None:
    """A local chat that sends `tools` is answered by a model whose manifest declares the
    tool-call format its chat template is trained on ([tools]), so the engine writes the
    tools in that form and reads the model's calls back as OpenAI `tool_calls`. Without
    one, Crucible would be guessing a format: refused by name (a misconfiguration, not
    weather). `tools` beside `response_format` / a structured output is refused too: on
    the local engines the constraint covers the whole answer, so a tool call could never
    be written. A model with no manifest is left to the door's own refusal."""
    tools = body.get("tools")
    if tools is None or tools == []:
        return
    if not isinstance(tools, list):
        raise ApiError(
            400, "invalid_request",
            f"`tools` is a list of OpenAI function tools; got {type(tools).__name__}",
        )
    try:
        manifest = load_manifest(model)
    except ManifestError:
        return
    if manifest.tools is None:
        takers = sorted(
            found.id for found in load_all_manifests().values()
            if found.tools is not None and found.supports(backend_kind)
        )
        raise ApiError(
            400,
            "tools_unsupported",
            f"{model!r} declares no tool-call format (its manifest has no [tools]), so "
            "Crucible cannot write `tools` in the form it was trained on. Send the chat "
            f"without `tools`, or name a model that declares one: {takers}",
            {"model": model, "models_with_tools": takers},
        )
    constrained = constrained_fields(body)
    if constrained:
        raise ApiError(
            400,
            "tools_with_response_format",
            f"a chat to {model!r} sent `tools` beside {constrained}: on this engine a "
            "constraint covers the whole answer, so the model could never write a tool "
            "call. Send the tools without the constraint, and constrain the final round "
            "(the one after the tool results) instead",
            {"model": model, "constrained": constrained},
        )


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
    refuse_tools_this_model_cannot_take(body, model, backend_kind)
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


def refuse_unservable(requested: str, form: str | None, backend_kind: str) -> None:
    """What the chat door refuses about the model a chat is served by, before anything
    waits: a form it does not have, a model that only writes vectors. The door and the
    preview (GET /v1/resolve) both call this."""
    refuse_unknown_form(requested, form, backend_kind)
    refuse_vectors_only(requested, "chat")


def register(routers: Routers, ctx: AppContext) -> None:
    private, openai = routers.private, routers.openai
    config, residency = ctx.config, ctx.residency
    lists = UpstreamLists()

    @private.get("/openai/models")
    @openai.get("/models")
    async def openai_models() -> dict[str, Any]:
        """The resident model in OpenAI's list shape, then every model each configured
        upstream serves, as `<upstream>/<id>` (the id a chat sends to reach it). An
        upstream row carries `upstream`, `routed_for` (the classes a route sends there),
        `listed` (the vendor's own list names it; false only for a model a route names
        that the vendor did not list) and `facts`: the `tool_choice` values Crucible
        sends on, and for an `anthropic/<id>` model its rule row (`reasoning_efforts`,
        `sampling`, `structured_output`), or null where Crucible has no rule and
        forwards the chat as sent. `upstreams` says, per configured upstream, when its
        list was read (`listed_at`), whether that list is `stale`, and the `error` that
        kept it from being read again: each vendor's list is kept for 10 minutes, a
        vendor that does not answer is served its last list, and a key or URL change
        reads it again at once.
        """
        resident = residency.resident_model
        data, status = await upstream_model_rows(config, ctx.http, lists)
        if resident is None:
            return {"object": "list", "data": data, "upstreams": status}
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
            "upstreams": status,
        }

    @private.post("/openai/chat/completions", openapi_extra=_documented_body())
    @openai.post("/chat/completions", openapi_extra=_documented_body())
    async def openai_chat_completions(request: Request) -> Response:
        """An OpenAI chat completion, proxied to the resident engine or, for a
        `<upstream>/<id>` model, forwarded to that upstream. A chat whose model is not
        resident, or whose engine has every slot taken, waits in the server's line
        (up to an hour, or `queue.max_wait_s`) and its model is loaded for it; with
        `"queue": false` it is refused at once instead. An upstream chat never waits.
        An `anthropic/<id>` chat is translated to the Messages API member by member, and
        a member with no faithful form there, or one the named model does not take, is
        refused by name before anything is sent (`upstream_field_unsupported`,
        `upstream_param_unsupported`; docs/API-CLI.md, "The anthropic upstream").
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
        A chat may leave `model` out and be served by a verb's model instead: `verb`
        (default `generate`) and `max_params_b`, a ceiling on this server's own pick for
        the verb that never raises it (docs/VERB-SIZING.md rule 8); the reply's `model`
        names the model chosen. The same choice opens a queue session
        (`max_params_b` beside `act`) and loads a model (`load-model`'s `params.verb`,
        `params.max_params_b`). `tools` and `tool_choice` are OpenAI's: the reply's
        message carries the model's calls as `tool_calls` (finish_reason `tool_calls`),
        never as `<tool_call>` text, and the app runs them and sends each result back
        as a `{"role": "tool", "tool_call_id", "content"}` message beside the assistant
        message that made the call. Crucible stores no tools and keeps no state between
        rounds. A local model is started to write and read the format its manifest
        declares (Qwen 3.5 and 3.8: the Qwen3-Coder XML form); a model with none is
        refused `tools_unsupported`, and tools beside a `response_format` are refused
        `tools_with_response_format` ("Tool calling"). An `anthropic/<id>` or
        `ollama/<id>` chat is translated both ways.
        """
        arrived = time.monotonic()
        raw = await request.body()
        body = chat_body(raw)
        named = body.get("model")
        if named is not None and (not isinstance(named, str) or named == ""):
            raise ApiError(
                400,
                "model_required",
                f"a chat's `model` is a model id, or left out (or null) to be served by "
                f"the model this server registered for a verb (`verb`, default "
                f"{DEFAULT_CHAT_VERB!r}, capped by `max_params_b`); got {named!r}",
                {"model": named},
            )
        sent_choice = "verb" in body or verbmodel.CEILING in body or named is None
        verb = verbmodel.take_verb(body)
        ceiling = verbmodel.take_ceiling(body)
        sent_queue = "queue" in body
        queued = queue_of(body)
        prefill = take_prefill(body)
        form = take_form(body)
        json_whitespace = take_json_whitespace(body)
        resident_now = residency.resident_model
        chosen = verbmodel.chat_model(
            ctx.config, ctx.backend, model=named, verb=verb, form=form,
            max_params_b=ceiling,
            resident=None if resident_now is None else resident_now.model_id,
        )
        requested = chosen.model
        body["model"] = requested
        log = ChatLog.of(
            body, requested,
            verbmodel.source_of(chosen, DEFAULT_CHAT_VERB if verb is None else verb, ctx.config),
            client_agent(request), arrived,
        )
        if (
            sent_choice
            or sent_queue
            or prefill is not None
            or form is not None
            or json_whitespace is not None
        ):
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
        refuse_unservable(requested, form, ctx.backend.kind)
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
            log.sent()
            if body.get("stream") is True:
                relayed = after_the_stream(inflight, entry, chat_over)

                async def when_relayed() -> None:
                    try:
                        log.report()
                    finally:
                        await relayed()

                return await proxy_stream(
                    client,
                    url,
                    forwarded,
                    resident,
                    sampling_headers,
                    when_relayed=when_relayed,
                    observe=log.chunk,
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
                log.report("the caller left before the answer")
                response = caller_gone(resident)
            else:
                log.completion(upstream.status_code, upstream.content)
                log.report()
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
