from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

from .decide import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecideItemsResponse,
    DecideRequest,
    DecideResponse,
    DecideTiming,
    DecideTokens,
    EnginePost,
    ForwardTiming,
    ItemsTiming,
    ItemsTokens,
    ModelProvenance,
    Plan,
    answer,
    decide_on_engine,
    image_part,
    label_distribution,
    plan,
    question_block,
    render_state,
    system_content,
    top_k,
)
from .engines.items_forward import ITEM_TOO_LONG, ITEMS_PATH, PROMPT_TOO_LONG, ItemsRefusal
from .errors import ApiError

MAX_ITEMS = 512

MAX_ITEM_TOKENS = 1024

MAX_PROMPT_TOKENS = 32768

BATCHED = "crucible_items"

EngineCall = Callable[[str, dict[str, Any]], Awaitable[Any]]


def check_item_count(items: list[Any]) -> int:
    count = len(items)
    if count > MAX_ITEMS:
        raise ApiError(
            400,
            "too_many_items",
            f"the request carries {count} items; one decision reads at most "
            f"{MAX_ITEMS}. Split the list into {math.ceil(count / MAX_ITEMS)} "
            "requests of at most that many",
            {"items": count, "max_items": MAX_ITEMS},
        )
    return count


def item_question(text: str, instructions: str | None) -> str:
    return text if instructions is None else f"{text}\n{instructions}"


def item_plans(body: DecideRequest) -> list[Plan]:
    shared = body.options or {}
    return [
        plan(
            f"items[{index}]",
            ChoiceQuestion(
                type="choice",
                instructions=item_question(item.text, body.instructions),
                options=item.options or shared,
            ),
        )
        for index, item in enumerate(body.items or [])
    ]


def blocks(plans: list[Plan]) -> list[str]:
    return [
        question_block(item.question.type, item.question.instructions, item.legend)
        for item in plans
    ]


def open_messages(state_text: str, images: list[str]) -> list[dict[str, Any]]:
    content: Any = [image_part(encoded) for encoded in images] if images else ""
    return [
        {"role": "system", "content": system_content(state_text, bool(images))},
        {"role": "user", "content": content},
    ]


def prompt_cap(max_model_len: int) -> int:
    return min(MAX_PROMPT_TOKENS, max_model_len - 1)


def items_body(
    engine_model_name: str, msgs: list[dict[str, Any]], questions: list[str], k: int, cap: int,
    item_cap: int = MAX_ITEM_TOKENS,
) -> dict[str, Any]:
    return {
        "model": engine_model_name,
        "messages": msgs,
        "questions": questions,
        "chat_template_kwargs": {"enable_thinking": False},
        "top_logprobs": k,
        "max_prompt_tokens": cap,
        "max_item_tokens": item_cap,
    }


@dataclass(frozen=True)
class ItemsReading:
    shared_tokens: int
    item_tokens: tuple[int, ...]
    tops: tuple[tuple[tuple[str, float], ...], ...]
    cached_tokens: int | None = None


def _engine_error(engine: str, detail: str) -> ApiError:
    return ApiError(
        502,
        "engine_error",
        f"the {engine} engine's reply cannot be read as an items decision: {detail}",
        {"engine": engine},
    )


def _field(container: Any, key: str, kind: Any, where: str, engine: str) -> Any:
    if not isinstance(container, dict) or key not in container:
        raise _engine_error(engine, f"{where}: {key!r} is missing")
    value = container[key]
    if isinstance(value, bool) or not isinstance(value, kind):
        raise _engine_error(engine, f"{where}.{key} is {type(value).__name__}, expected {kind}")
    return value


NEXT_STEP = {
    ITEM_TOO_LONG: "Shorten that item (Briefcase clips a unit at 300 characters)",
    PROMPT_TOO_LONG: (
        "Send a shorter state, or load the model with a longer context "
        "(POST /v1/jobs {\"type\": \"load-model\", \"model\": ..., \"context\": ...})"
    ),
}


def refusal_error(refusal: ItemsRefusal) -> ApiError:
    step = NEXT_STEP.get(refusal.code, "Fix the request as the message says and send it again")
    return ApiError(refusal.status, refusal.code, f"{refusal}. {step}", refusal.details)


def engine_refusal(status: int, payload: Any) -> ApiError | None:
    error = payload.get("error") if isinstance(payload, dict) else None
    if status != 400 or not isinstance(error, dict) or error.get("code") not in NEXT_STEP:
        return None
    details = error.get("details") if isinstance(error.get("details"), dict) else {}
    return refusal_error(ItemsRefusal(400, error["code"], str(error.get("message")), details))


def read_items_reply(data: Any, engine: str, n_items: int) -> ItemsReading:
    shared = _field(data, "shared_tokens", int, "reply", engine)
    counts = _field(data, "item_tokens", list, "reply", engine)
    slots = _field(data, "slots", list, "reply", engine)
    if len(slots) != n_items or len(counts) != n_items:
        raise _engine_error(
            engine, f"{len(slots)} slots and {len(counts)} item counts for {n_items} items"
        )
    tops = []
    for index, slot in enumerate(slots):
        entries = _field(slot, "top_logprobs", list, f"slots[{index}]", engine)
        row = []
        for n, entry in enumerate(entries):
            where = f"slots[{index}].top_logprobs[{n}]"
            token = _field(entry, "token", str, where, engine)
            logprob = _field(entry, "logprob", (int, float), where, engine)
            row.append((token, math.exp(logprob)))
        tops.append(tuple(row))
    cached = data.get("cached_tokens")
    if cached is not None and (isinstance(cached, bool) or not isinstance(cached, int)):
        raise _engine_error(engine, f"reply.cached_tokens is {type(cached).__name__}, expected int")
    return ItemsReading(
        shared_tokens=shared, item_tokens=tuple(counts), tops=tuple(tops), cached_tokens=cached
    )


def _choice_answers(
    plans: list[Plan], tops: tuple[tuple[tuple[str, float], ...], ...], engine: str,
    missing: Literal["refuse", "report"],
) -> list[ChoiceAnswer]:
    answered = []
    for item, top in zip(plans, tops):
        result = answer(item, label_distribution(top, item, engine, missing=missing), missing)
        assert isinstance(result, ChoiceAnswer)
        answered.append(result)
    return answered


def _provenance(resident: Any) -> ModelProvenance:
    return ModelProvenance(
        id=resident.model_id, revision=resident.revision, fingerprint=resident.fingerprint
    )


async def _batched(
    call: EngineCall, resident: Any, body: DecideRequest, plans: list[Plan],
    max_logprobs: int | None, started: float,
) -> DecideItemsResponse:
    images = list(body.images or [])
    k = top_k(max(len(item.labels) for item in plans), max_logprobs)
    wire = items_body(
        resident.engine_model_name, open_messages(render_state(body.state), images),
        blocks(plans), k, prompt_cap(resident.max_model_len),
    )
    reading = read_items_reply(await call(ITEMS_PATH, wire), resident.engine, len(plans))
    return DecideItemsResponse(
        model=_provenance(resident),
        engine=resident.engine,
        answers=_choice_answers(plans, reading.tops, resident.engine, body.missing),
        timing_ms=ItemsTiming(
            total=round((time.perf_counter() - started) * 1000.0, 1), engine_requests=1
        ),
        tokens=ItemsTokens(
            shared=reading.shared_tokens,
            per_item=[reading.shared_tokens + n for n in reading.item_tokens],
            images=len(images),
        ),
    )


async def decide_questions_on_items(
    call: EngineCall,
    resident: Any,
    body: DecideRequest,
    plans: list[Plan],
    *,
    max_logprobs: int | None,
) -> DecideResponse:
    """The question form read through the items route: the state once (or not
    at all, when the engine still holds it), every question one row of a batched
    forward, one request. The prompts are the ones `decide_on_engine` sends, so
    the answers are the same distributions; there is no prime."""
    started = time.perf_counter()
    images = list(body.images or [])
    k = top_k(max(len(item.labels) for item in plans), max_logprobs)
    cap = prompt_cap(resident.max_model_len)
    wire = items_body(
        resident.engine_model_name, open_messages(render_state(body.state), images),
        blocks(plans), k, cap, item_cap=cap,
    )
    sent = time.perf_counter()
    reading = read_items_reply(await call(ITEMS_PATH, wire), resident.engine, len(plans))
    wall_ms = round((time.perf_counter() - sent) * 1000.0, 1)
    answers: dict[str, Answer] = {}
    per_question: dict[str, ForwardTiming] = {}
    tokens: dict[str, int] = {}
    for item, top, own in zip(plans, reading.tops, reading.item_tokens):
        dist = label_distribution(top, item, resident.engine, missing=body.missing)
        answers[item.name] = answer(item, dist, body.missing)
        tokens[item.name] = reading.shared_tokens + own
        per_question[item.name] = ForwardTiming(
            wall_ms=wall_ms, prompt_tokens=tokens[item.name], cached_tokens=reading.cached_tokens
        )
    return DecideResponse(
        model=_provenance(resident),
        engine=resident.engine,
        answers=answers,
        timing_ms=DecideTiming(
            total=round((time.perf_counter() - started) * 1000.0, 1),
            per_question=per_question,
            prime=None,
        ),
        tokens=DecideTokens(per_question=tokens, images=len(images)),
    )


async def _one_per_item(
    post: EnginePost, resident: Any, body: DecideRequest, plans: list[Plan],
    max_logprobs: int | None, concurrency: int, started: float,
) -> DecideItemsResponse:
    decided = await decide_on_engine(
        post, resident, body, plans, max_logprobs=max_logprobs, concurrency=concurrency
    )
    answers = [decided.answers[item.name] for item in plans]
    assert all(isinstance(each, ChoiceAnswer) for each in answers)
    return DecideItemsResponse(
        model=decided.model,
        engine=decided.engine,
        answers=answers,
        timing_ms=ItemsTiming(
            total=round((time.perf_counter() - started) * 1000.0, 1),
            engine_requests=len(plans) + (decided.timing_ms.prime is not None),
        ),
        tokens=ItemsTokens(
            shared=None,
            per_item=[decided.tokens.per_question[item.name] for item in plans],
            images=decided.tokens.images,
        ),
    )


async def decide_items_on_engine(
    call: EngineCall,
    post: EnginePost,
    resident: Any,
    body: DecideRequest,
    plans: list[Plan],
    *,
    batched: bool,
    max_logprobs: int | None,
    concurrency: int,
) -> DecideItemsResponse:
    started = time.perf_counter()
    if batched:
        return await _batched(call, resident, body, plans, max_logprobs, started)
    return await _one_per_item(post, resident, body, plans, max_logprobs, concurrency, started)


__all__ = [
    "BATCHED",
    "EngineCall",
    "ItemsReading",
    "MAX_ITEMS",
    "MAX_ITEM_TOKENS",
    "MAX_PROMPT_TOKENS",
    "blocks",
    "check_item_count",
    "decide_items_on_engine",
    "decide_questions_on_items",
    "engine_refusal",
    "item_plans",
    "item_question",
    "items_body",
    "open_messages",
    "prompt_cap",
    "read_items_reply",
    "refusal_error",
]
