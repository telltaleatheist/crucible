from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask

from ... import decide as decide_core
from ... import upstreams
from ...decide import DecideRequest, DecideResponse
from ...engines import chat_admission, decide_reading
from ...errors import ApiError
from ...inflight import read_act
from ...manifests import load_manifest
from ..caller import client_agent
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


def _engine_post(client: httpx.AsyncClient, resident: Any) -> decide_core.EnginePost:
    url = f"{resident.base_url}/v1/chat/completions"

    async def post(body: dict[str, Any]) -> Any:
        payload = json.dumps(body).encode("utf-8")
        try:
            response = await sent_across_the_wire(
                lambda: client.post(url, content=payload, headers=JSON_HEADERS),
                where=f"the engine serving {resident.model_id!r}",
            )
        except httpx.HTTPError as exc:
            raise engine_unreachable(resident, exc) from None
        if response.status_code != 200:
            raise _decide_engine_refused(resident, response)
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

    return post


def _refuse_an_upstream(model: str) -> None:
    if upstreams.split_model(model) is None:
        return
    raise ApiError(
        400,
        "decide_needs_logprobs",
        f"{model!r} names an upstream; a decision reads the next-token "
        "distribution at the resident model, and no upstream returns one. "
        "Name the resident Crucible model id",
        {"requested": model},
    )


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    backend, residency = ctx.backend, ctx.residency

    @private.post(
        "/decide",
        response_model=None,
        responses={200: {"model": DecideResponse}},
    )
    async def decide(request: Request, body: DecideRequest) -> Response:
        """One answer distribution per question, read off the resident model's
        next-token logprobs. Every refusal a caller can cause is made before anything is
        sent to the engine.
        """
        act = read_act(request.headers)
        _refuse_an_upstream(body.model)
        async with residency.settled_for("a decision"):
            resident = residency.resident_model
            if resident is None or resident.model_id != body.model:
                raise model_not_resident(body.model, resident, "a decision")
            refuse_an_exited_engine(residency, resident)

            n_images = decide_core.check_image_count(body.images)
            if n_images:
                decide_core.refuse_images_not_served(
                    resident.model_id, load_manifest(resident.model_id), backend.kind, n_images
                )
            plans = decide_core.plan_all(body)
            reading = decide_reading(resident.engine)
            decide_core.refuse_unreadable_labels(resident, reading, plans)

            inflight = ctx.inflight
            limit, limit_basis = chat_admission(resident.engine, resident.engine_args)
            if limit is not None and len(inflight) >= limit:
                return chat_queue_full(
                    resident=resident, limit=limit, basis=limit_basis,
                    wait=inflight.retry_after(),
                )
            concurrency = (
                limit if limit is not None else decide_core.UNSTATED_ENGINE_CONCURRENCY
            )
            chat_over = settle_after_chat(ctx.settlement)
            post = _engine_post(ctx.http, resident)
            entry = inflight.open(
                act=act, model=resident.model_id, client=client_agent(request)
            )
        try:
            answered = await unless_the_caller_leaves(
                decide_core.decide_on_engine(
                    post,
                    resident,
                    body,
                    plans,
                    max_logprobs=reading.max_logprobs,
                    concurrency=concurrency,
                ),
                request,
            )
            if answered is None:
                response: Response = caller_gone(resident)
            else:
                response = JSONResponse(content=answered.model_dump(mode="json"))
            inflight.close(entry)
            response.background = BackgroundTask(chat_over)
            return response
        except BaseException:
            inflight.close(entry)
            await chat_over()
            raise
