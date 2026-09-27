from __future__ import annotations

import asyncio
import json
import time
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
from ...inflight import InFlight, read_act
from ...manifests import load_manifest
from ...settle import Settlement
from ..caller import client_agent
from ..proxy import (
    JSON_HEADERS,
    _caller_gone,
    _chat_over,
    _chat_queue_full,
    _engine_unreachable,
    _model_not_resident,
    _refuse_an_exited_engine,
    _sent_across_the_wire,
    _unless_the_caller_leaves,
)
from ..context import AppContext, Routers


def _decide_not_served(
    resident: Any, reason: str, extra: dict[str, Any] | None
) -> ApiError:
    """503: the resident engine cannot return what a decision reads."""
    return ApiError(
        503,
        "decide_not_served",
        f"the {resident.engine} engine serving {resident.model_id!r} cannot serve "
        f"this decision: {reason}. Nothing was sent to it",
        {"model": resident.model_id, "engine": resident.engine, "reason": reason,
         **(extra or {})},
    )


def _decide_engine_refused(resident: Any, response: httpx.Response) -> ApiError:
    """502: the engine answered a decision's request with something other than 200.

    Named `engine_error` and not relayed as it stood, unlike the chat door: a
    decision is several requests and one answer, so there is no single engine
    body to hand back — the one that failed is quoted instead.
    """
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


async def _decide_on_engine(
    client: httpx.AsyncClient,
    resident: Any,
    body: DecideRequest,
    plans: list[decide_core.Plan],
    *,
    max_logprobs: int | None,
    concurrency: int,
) -> DecideResponse:
    """The prime, then every question, on the resident engine's chat route.

    PHASE22 section 2.5. With more than one question the shared prefix goes
    first and ALONE, so its KV is cached (vLLM) or its context checkpoint laid
    down (llama-server, hybrid Qwen3.5) before the questions ask for it; then
    the questions go out together, at most `concurrency` at once. Answers come
    back in the request's question order, and when questions fail the one
    reported is the first in THAT order, so a retry with the same body meets
    the same refusal first.
    """
    started = time.perf_counter()
    url = f"{resident.base_url}/v1/chat/completions"
    state_text = decide_core.render_state(body.state)
    images = list(body.images or [])
    engine = resident.engine

    async def forward(
        msgs: list[dict[str, Any]], k: int | None
    ) -> tuple[decide_core.Reading, float]:
        payload = json.dumps(
            decide_core.request_body(resident.engine_model_name, msgs, k)
        ).encode("utf-8")
        sent = time.perf_counter()
        try:
            response = await _sent_across_the_wire(
                lambda: client.post(url, content=payload, headers=JSON_HEADERS),
                where=f"the engine serving {resident.model_id!r}",
            )
        except httpx.HTTPError as exc:
            raise _engine_unreachable(resident, exc) from None
        wall_ms = (time.perf_counter() - sent) * 1000.0
        if response.status_code != 200:
            raise _decide_engine_refused(resident, response)
        try:
            data = response.json()
        except ValueError as exc:
            raise ApiError(
                502,
                "engine_error",
                f"the {engine} engine serving {resident.model_id!r} answered 200 "
                f"with a body that is not JSON: {exc}. Its log is {resident.log_path}",
                {"engine": engine},
            ) from None
        return decide_core.read_reply(data, engine, want_probs=k is not None), wall_ms

    def timing(reading: decide_core.Reading, wall_ms: float) -> decide_core.ForwardTiming:
        return decide_core.ForwardTiming(
            wall_ms=round(wall_ms, 1),
            prompt_tokens=reading.prompt_tokens,
            cached_tokens=reading.cached_tokens,
        )

    prime: decide_core.ForwardTiming | None = None
    if len(plans) > 1:
        # A PRIME IS NOT AN ANSWER (snap `3509bc5`): it asks for no logprobs
        # and its token is never read, so a reply without them is no fault.
        reading, wall_ms = await forward(
            decide_core.prime_messages(state_text, images), None
        )
        prime = timing(reading, wall_ms)

    gate = asyncio.Semaphore(concurrency)

    async def ask(
        item: decide_core.Plan,
    ) -> tuple[Any, decide_core.ForwardTiming, int]:
        k = decide_core.top_k(len(item.labels), max_logprobs)
        async with gate:
            reading, wall_ms = await forward(
                decide_core.question_messages(state_text, images, item), k
            )
        assert reading.top is not None  # want_probs=True always reads them
        dist = decide_core.label_distribution(
            reading.top, item, engine, missing=body.missing
        )
        return (
            decide_core.answer(item, dist, body.missing),
            timing(reading, wall_ms),
            reading.prompt_tokens,
        )

    tasks = [asyncio.create_task(ask(item)) for item in plans]
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for task in tasks:
            if task.cancelled():
                continue
            failure = task.exception()
            if failure is not None:
                raise failure from None
        raise

    answers = {}
    per_question = {}
    tokens = {}
    for item, (answered, timed, prompt_tokens) in zip(plans, results):
        answers[item.name] = answered
        per_question[item.name] = timed
        tokens[item.name] = prompt_tokens
    return DecideResponse(
        model=decide_core.ModelProvenance(
            id=resident.model_id,
            revision=resident.revision,
            fingerprint=resident.fingerprint,
        ),
        engine=engine,
        answers=answers,
        timing_ms=decide_core.DecideTiming(
            total=round((time.perf_counter() - started) * 1000.0, 1),
            per_question=per_question,
            prime=prime,
        ),
        tokens=decide_core.DecideTokens(per_question=tokens, images=len(images)),
    )


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    backend, residency = ctx.backend, ctx.residency

    # --------------------------------------------------------------- decide

    @private.post(
        "/decide",
        response_model=None,
        responses={200: {"model": DecideResponse}},
    )
    async def decide(request: Request, body: DecideRequest) -> Response:
        """One distribution per question, read off the resident model.

        PHASE22-DECIDE.md is the contract. A decision is the chat door's
        sibling and walks through the chat door's machinery — the act header,
        the resident check, `chat_admission`, the `InFlight` record,
        `_chat_over` — with a different body in and out. What is its own is the
        reading (`crucible/decide.py`): the frame, the letters, the parser.

        EVERY REFUSAL A CALLER CAN CAUSE IS MADE BEFORE ANYTHING IS SENT: the
        act, an upstream id, a model that is not resident, too many images,
        images on a text model, too many options, an engine that returns no
        top logprobs or too few of them, a full door. A decision that spent the
        card and then failed on a question the server could have read first
        would be a decision the client paid for twice.
        """
        # Read BEFORE the work starts, as on chat: an unknown act is a 400
        # rather than a decision reported under a name nobody knows.
        act = read_act(request.headers)
        if upstreams.split_model(body.model) is not None:
            raise ApiError(
                400,
                "decide_needs_logprobs",
                f"{body.model!r} names an upstream; a decision reads the next-token "
                "distribution at the resident model, and no upstream returns one. "
                "Name the resident Crucible model id",
                {"requested": body.model},
            )
        # WAITED OUT AND ATOMIC, exactly as on the chat door (2026-09-24,
        # Briefcase): a decision arriving mid-clearance is answered from the
        # settled card, and the one it proxies is an `InFlight` row the
        # settlement sees. Everything in this block is synchronous, which is
        # what `settled_for` requires of it.
        async with residency.settled_for("a decision"):
            resident = residency.resident_model
            if resident is None or resident.model_id != body.model:
                raise _model_not_resident(body.model, resident, "a decision")
            _refuse_an_exited_engine(residency, resident)

            n_images = decide_core.check_image_count(body.images)
            if n_images:
                # Read at decision time and only with images in hand: the record
                # carries no modalities, and a manifest whose modalities changed
                # also changed its engine line (`--language-model-only`, an
                # `mmproj`), which is a reload either way.
                #
                # WHAT THIS BACKEND SERVES, not what the weights accept (PHASE22
                # section 2.9). `qwen3.5-4b` accepts images everywhere and is
                # served them on cuda-linux and llama-windows only: on the Mac its
                # engine is mlx-lm, which is never handed a picture. Reading the
                # model-wide list here would pass a page to an engine that drops it
                # and answer from the text alone.
                manifest = load_manifest(resident.model_id)
                served = manifest.serves(backend.kind)
                if "image" not in served:
                    raise ApiError(
                        400,
                        "model_text_only",
                        f"{resident.model_id!r} is served {list(served)} on "
                        f"{backend.kind} (its weights accept "
                        f"{list(manifest.modalities)}) and this decision carries "
                        f"{n_images} image(s). Whether a model answers images HERE is "
                        "its manifest's backend block (`serves`, PHASE22 section 2.9)",
                        {"model": resident.model_id, "backend": backend.kind,
                         "serves": list(served),
                         "modalities": list(manifest.modalities),
                         "images": n_images},
                    )
            plans = decide_core.plan_all(body)

            reading = decide_reading(resident.engine)
            if not reading.served:
                raise _decide_not_served(resident, reading.basis, None)
            widest = max(plans, key=lambda item: len(item.labels))
            if reading.max_logprobs is not None and len(widest.labels) > reading.max_logprobs:
                raise _decide_not_served(
                    resident,
                    f"{resident.engine} returns at most {reading.max_logprobs} top "
                    f"logprobs and question {widest.name!r} has {len(widest.labels)} "
                    f"options ({reading.basis})",
                    {"question": widest.name, "options": len(widest.labels),
                     "max_logprobs": reading.max_logprobs},
                )

            inflight: InFlight = request.app.state.inflight
            limit, limit_basis = chat_admission(resident.engine, resident.engine_args)
            if limit is not None and len(inflight) >= limit:
                wait = inflight.retry_after()
                return _chat_queue_full(
                    resident=resident, limit=limit, basis=limit_basis, wait=wait
                )
            concurrency = (
                limit if limit is not None else decide_core.UNSTATED_ENGINE_CONCURRENCY
            )

            settlement: Settlement = request.app.state.settlement
            chat_over = _chat_over(settlement)
            client: httpx.AsyncClient = request.app.state.http
            entry = inflight.open(
                act=act, model=resident.model_id, client=client_agent(request)
            )
        try:
            answered = await _unless_the_caller_leaves(
                _decide_on_engine(
                    client,
                    resident,
                    body,
                    plans,
                    max_logprobs=reading.max_logprobs,
                    concurrency=concurrency,
                ),
                request,
            )
            if answered is None:
                response: Response = _caller_gone(resident)
            else:
                response = JSONResponse(content=answered.model_dump(mode="json"))
            inflight.close(entry)
            # After the answer is written, as on chat: the card is cleared
            # behind the decision, never in front of it.
            response.background = BackgroundTask(chat_over)
            return response
        except BaseException:
            inflight.close(entry)
            await chat_over()
            raise
