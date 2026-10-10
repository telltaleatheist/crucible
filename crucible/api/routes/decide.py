from __future__ import annotations

import json
import sys
import time
from typing import Any

import httpx
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask

from ... import decide as decide_core
from ... import decide_items, decide_likelihood, enginespec, upstreamrecord
from ...callqueue import take_a_turn
from ...capabilityclasses import BY_NAME
from ...config import Config
from ...decide import DecideItemsResponse, DecideRequest, DecideResponse
from ...engines import (
    chat_admission,
    decide_items_reading,
    decide_reading,
    likelihood_reading,
)
from ...engines.items_forward import ITEMS_PATH
from ...errors import ApiError
from ...formrequest import refuse_unknown_form
from ...inflight import read_act
from ...manifests import load_manifest
from ...queuerequest import max_wait_of
from ...residency import serves_model
from ..caller import client_agent, queue_session
from ..context import AppContext, Routers
from ..proxy import (
    JSON_HEADERS,
    caller_gone,
    chat_queue_full,
    engine_unreachable,
    model_not_resident,
    refuse_an_exited_engine,
    sent_across_the_wire,
    settle_after_chat,
    unless_the_caller_leaves,
)


def _decide_engine_refused(resident: Any, response: httpx.Response) -> ApiError:
    text = response.text
    return ApiError(
        502,
        "engine_error",
        f"the {resident.engine} engine serving {resident.model_id!r} answered a "
        f"decision's request with {response.status_code}: {text[:500]}. Its log is "
        f"{resident.log_path}",
        {"engine": resident.engine, "status": response.status_code,
         "body": text[:2000]},
    )


def _items_route_missing(resident: Any) -> ApiError:
    load = json.dumps({"type": "load-model", "model": resident.model_id})
    return decide_core.decide_not_served(
        resident,
        f"the engine process answers no {ITEMS_PATH}: it was started before its "
        f"env carried the items route. Load the model again (POST /v1/jobs {load}); "
        "the load applies the route",
        {"form": "items"},
    )


def _items_route_older(resident: Any, payload: Any) -> ApiError | None:
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict) or error.get("code") != "unknown_field":
        return None
    load = json.dumps({"type": "load-model", "model": resident.model_id})
    return decide_core.decide_not_served(
        resident,
        f"the engine process's {ITEMS_PATH} is older than this server's and refused a "
        f"field it does not read ({error.get('message')}). Load the model again (POST "
        f"/v1/jobs {load}); the load applies the current route",
        {"form": "items"},
    )


def _refused(resident: Any, path: str, response: httpx.Response) -> ApiError:
    if path == ITEMS_PATH and response.status_code == 404:
        return _items_route_missing(resident)
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if path == ITEMS_PATH and response.status_code == 400:
        older = _items_route_older(resident, payload)
        if older is not None:
            return older
    named = decide_items.engine_refusal(response.status_code, payload)
    return named if named is not None else _decide_engine_refused(resident, response)


def _engine_call(client: httpx.AsyncClient, resident: Any) -> decide_items.EngineCall:
    async def call(path: str, body: dict[str, Any]) -> Any:
        url = f"{resident.base_url}{path}"
        payload = json.dumps(body).encode("utf-8")
        try:
            response = await sent_across_the_wire(
                lambda: client.post(url, content=payload, headers=JSON_HEADERS),
                where=f"the engine serving {resident.model_id!r}",
            )
        except httpx.HTTPError as exc:
            raise engine_unreachable(resident, exc) from None
        if response.status_code != 200:
            raise _refused(resident, path, response)
        try:
            return response.json()
        except ValueError as exc:
            raise ApiError(
                502,
                "engine_error",
                f"the {resident.engine} engine serving {resident.model_id!r} answered 200 "
                f"with a body that is not JSON: {exc}. Its log is {resident.log_path}",
                {"engine": resident.engine},
            ) from None

    return call


def _engine_post(client: httpx.AsyncClient, resident: Any) -> decide_core.EnginePost:
    call = _engine_call(client, resident)

    async def post(body: dict[str, Any]) -> Any:
        return await call("/v1/chat/completions", body)

    return post


def _image_models(backend_kind: str) -> list[str]:
    return [
        candidate.id
        for candidate in BY_NAME["decide"].candidates(backend_kind)
        if "image" in load_manifest(candidate.id).serves(backend_kind)
    ]


def _registered_model(config: Config, n_images: int) -> str:
    """The model a decision that names none is served by: what this server registered
    for `decide` (its capability record's row), or for a decision with images, the model
    that row names for one (docs/VERB-SIZING.md section 8). A model that is not resident
    is loaded for it in the line, as for a named one; a switch between the text and the
    vision form is a reload of the same download."""
    record = config.capability
    row = None if record is None else record.row("decide")
    if row is None:
        raise ApiError(
            503,
            "capability_undecided",
            "this decision names no model, and this server has registered nothing for "
            "decide: its capability record "
            + ("does not exist" if record is None else "predates the decide class")
            + ". Run `crucible capability --write`, or name a `model`",
            {"capability": "decide"},
        )
    if not row.enabled:
        raise ApiError(
            409,
            "capability_disabled",
            f"this decision names no model, and this server cannot decide: "
            f"{row.reason}",
            {"capability": "decide", "shortfall_bytes": row.shortfall_bytes},
        )
    if not n_images:
        return row.selected
    if row.with_images is None:
        raise ApiError(
            503,
            "capability_undecided",
            "this decision carries images and names no model, and this server's "
            "capability record was written before it registered a model for a "
            "decision with images. Run `crucible capability --write`, or name a "
            "`model` that serves images",
            {"capability": "decide", "images": n_images},
        )
    if not row.with_images:
        raise ApiError(
            409,
            "no_image_model_fits",
            f"this decision carries {n_images} image(s) and names no model, and no "
            f"model that reads images fits this server: {row.with_images_reason}",
            {"capability": "decide", "images": n_images, "text_model": row.selected},
        )
    return row.with_images


def _refuse_an_upstream(model: str) -> None:
    if upstreamrecord.split_model(model) is None:
        return
    raise ApiError(
        400,
        "decide_needs_logprobs",
        f"{model!r} names an upstream; a decision reads the next-token "
        "distribution at the resident model, and no upstream returns one. "
        "Name the resident Crucible model id",
        {"requested": model},
    )


def _refuse_a_malformed_decision(
    body: DecideRequest, model: str, backend_kind: str
) -> None:
    """What a queued decision would be refused for once its model is resident is
    refused now, before it waits."""
    n_images = decide_core.check_image_count(body.images)
    if n_images:
        try:
            manifest = load_manifest(model)
        except Exception:
            manifest = None
        if manifest is not None:
            decide_core.refuse_images_not_served(
                model, manifest, backend_kind, n_images,
                lambda: _image_models(backend_kind),
            )
    if body.items is not None:
        decide_items.check_item_count(body.items)
        decide_items.item_plans(body)
        return
    plans = decide_core.plan_all(body)
    if not decide_core.likelihood_plans(plans):
        return
    try:
        engine = load_manifest(model).spec(backend_kind).engine
    except Exception:
        return
    decide_likelihood.refuse_unscorable(
        model, engine, likelihood_reading(engine), n_images
    )


def _with_likelihood(
    client: httpx.AsyncClient,
    resident: Any,
    body: DecideRequest,
    plans: list[decide_core.Plan],
    route: str | None,
    *,
    questions_on_items: bool,
    max_logprobs: int | None,
    concurrency: int,
) -> Any:
    """A questions-form decision with likelihood questions in it: the label
    questions by the path they always take, the likelihood questions by the
    engine's route, merged in the request's order."""
    call = _engine_call(client, resident)
    labels = decide_core.label_plans(plans)
    scored = decide_core.likelihood_plans(plans)

    def labelled() -> Any:
        if questions_on_items:
            return decide_items.decide_questions_on_items(
                call, resident, body, labels, max_logprobs=max_logprobs,
            )
        return decide_core.decide_on_engine(
            _engine_post(client, resident), resident, body, labels,
            max_logprobs=max_logprobs, concurrency=concurrency,
        )

    def likely() -> Any:
        if route == "items":
            return decide_likelihood.score_on_items(call, resident, body, scored)
        if route == "forced-tokens":
            return decide_likelihood.score_on_forced_tokens(call, resident, body, scored)
        assert route == "prompt-logprobs", route
        return decide_likelihood.score_on_prompt_logprobs(
            call, resident, body, scored, concurrency=concurrency
        )

    return decide_likelihood.decide_with_likelihood(
        plans, labelled if labels else None, likely, resident, len(body.images or [])
    )


def _log_timing(
    body: DecideRequest, model: str, client: str | None, arrived: float, started: float
) -> None:
    finished = time.monotonic()
    asked = (
        f"{len(body.items)} item(s)" if body.items is not None
        else f"{len(body.questions or {})} question(s)"
    )
    print(
        f"crucible: decide on {model!r} for {client or 'an unnamed client'}: "
        f"{asked}, waited {(started - arrived) * 1000:.0f} ms, "
        f"answered in {(finished - started) * 1000:.0f} ms",
        file=sys.stderr,
    )


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    backend, residency = ctx.backend, ctx.residency

    @private.post(
        "/decide",
        response_model=None,
        responses={200: {"model": DecideResponse | DecideItemsResponse}},
    )
    async def decide(request: Request, body: DecideRequest) -> Response:
        """One answer distribution per question, read off the resident model's
        next-token logprobs; with `items`, one choice answer per item in one request.
        Every refusal a caller can cause is made before anything is decided. A decision
        whose model is not resident, or whose engine has every slot taken, waits in the
        server's line and its model is loaded for it; with `"queue": false` it is
        refused at once instead. With no `model`, the decision is served by the model
        this server registered for decide, or with `images`, the one it registered for
        a decision with images (`GET /v1/capability`, the decide row's `with_images`).
        """
        arrived = time.monotonic()
        act = read_act(request.headers)
        model = (
            body.model
            if body.model is not None
            else _registered_model(ctx.config, decide_core.check_image_count(body.images))
        )
        _refuse_an_upstream(model)
        refuse_unknown_form(model, body.form, backend.kind)
        inflight = ctx.inflight
        chat_over = settle_after_chat(ctx.settlement)
        session = queue_session(request, ctx.sessions)
        session_id = None if session is None else session.id
        turn: Any = None
        max_wait_s = max_wait_of(body.queue)
        if max_wait_s is not None:
            _refuse_a_malformed_decision(body, model, backend.kind)
            turn = await take_a_turn(
                request, line=ctx.line, residency=residency, inflight=inflight,
                settle=chat_over, kind="decide", model=model, act=act,
                client=client_agent(request), max_wait_s=max_wait_s,
                session=session, form=body.form,
            )
            if isinstance(turn, Response):
                return turn
        else:
            ctx.sessions.refuse_call_if_held(session_id, "a decision")
        if session is not None:
            ctx.sessions.item_arrived(session)
        try:
            async with residency.settled_for("a decision"):
                resident = residency.resident_model
                if not serves_model(resident, model, body.form):
                    raise model_not_resident(model, resident, "a decision", body.form)
                refuse_an_exited_engine(residency, resident)

                n_images = decide_core.check_image_count(body.images)
                if n_images:
                    decide_core.refuse_images_not_served(
                        resident.model_id,
                        load_manifest(resident.model_id),
                        backend.kind,
                        n_images,
                        lambda: _image_models(backend.kind),
                    )
                if body.items is not None:
                    decide_items.check_item_count(body.items)
                    plans = decide_items.item_plans(body)
                else:
                    plans = decide_core.plan_all(body)
                reading = decide_reading(resident.engine)
                decide_core.refuse_unreadable_labels(resident, reading, plans)
                scored_plans = (
                    [] if body.items is not None else decide_core.likelihood_plans(plans)
                )
                if scored_plans:
                    scoring = likelihood_reading(resident.engine)
                    decide_likelihood.refuse_unscorable(
                        resident.model_id, resident.engine, scoring, n_images
                    )

                limit, limit_basis = chat_admission(
                    resident.engine, resident.engine_args
                )
                if turn is None and limit is not None and len(inflight) >= limit:
                    return chat_queue_full(
                        resident=resident, limit=limit, basis=limit_basis,
                        wait=inflight.retry_after(),
                    )
                concurrency = (
                    limit if limit is not None
                    else enginespec.UNSTATED_ENGINE_CONCURRENCY
                )
                items_reading = decide_items_reading(resident.engine)
                if body.items is not None:
                    work = decide_items.decide_items_on_engine(
                        _engine_call(ctx.http, resident),
                        _engine_post(ctx.http, resident),
                        resident, body, plans,
                        batched=items_reading.batched,
                        max_logprobs=reading.max_logprobs, concurrency=concurrency,
                    )
                elif scored_plans:
                    work = _with_likelihood(
                        ctx.http, resident, body, plans, scoring.route,
                        questions_on_items=items_reading.questions,
                        max_logprobs=reading.max_logprobs, concurrency=concurrency,
                    )
                elif items_reading.questions:
                    work = decide_items.decide_questions_on_items(
                        _engine_call(ctx.http, resident), resident, body, plans,
                        max_logprobs=reading.max_logprobs,
                    )
                else:
                    work = decide_core.decide_on_engine(
                        _engine_post(ctx.http, resident), resident, body, plans,
                        max_logprobs=reading.max_logprobs, concurrency=concurrency,
                    )
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
                response = JSONResponse(content=answered.model_dump(mode="json"))
                _log_timing(body, model, client_agent(request), arrived, started)
            inflight.close(entry)
            response.background = BackgroundTask(chat_over)
            return response
        except BaseException:
            inflight.close(entry)
            await chat_over()
            raise
