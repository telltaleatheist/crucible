from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Awaitable, Callable

from .decide import (
    LIKELIHOOD_SYSTEM_PROMPT,
    MAX_CANDIDATE_TOKENS,
    Answer,
    CandidateScore,
    DecideRequest,
    DecideResponse,
    DecideTiming,
    DecideTokens,
    ForwardTiming,
    LikelihoodAnswer,
    LikelihoodQuestion,
    ModelProvenance,
    Plan,
    image_part,
    render_state,
    system_content,
)
from .decide_items import EngineCall, prompt_cap, refusal_error
from .engines.items_forward import (
    CANDIDATE_NOT_A_REPLY,
    CANDIDATE_TOO_LONG,
    ITEM_TOO_LONG,
    ITEMS_PATH,
    PROMPT_TOO_LONG,
    ItemsRefusal,
    likelihood_split,
)
from .errors import ApiError

TOKENIZE_PATH = "/tokenize"

CHAT_PATH = "/v1/chat/completions"

VLLM_CLAMPED_LOGPROB = -9999.0
"""What vLLM writes for a prompt token whose log-probability is -inf
(`clamp_prompt_logprobs`, vllm/entrypoints/generate/base/serving.py L376-388):
the model gives that token no probability at all, which is no number."""

TEMPLATE_KWARGS = {"enable_thinking": False}

GROUP_CODES = frozenset({CANDIDATE_TOO_LONG, CANDIDATE_NOT_A_REPLY, PROMPT_TOO_LONG, ITEM_TOO_LONG})


def _question(item: Plan) -> LikelihoodQuestion:
    question = item.question
    assert isinstance(question, LikelihoodQuestion)
    return question


def likelihood_messages(state_text: str, images: list[str], instructions: str) -> list[dict[str, Any]]:
    """A likelihood question's context: the state in the system turn as every
    decision has it, under a system prompt that asks for a reply instead of a
    letter, and the request as the user turn."""
    user: Any = instructions
    if images:
        user = [*(image_part(encoded) for encoded in images), {"type": "text", "text": instructions}]
    return [
        {"role": "system", "content": system_content(state_text, bool(images), LIKELIHOOD_SYSTEM_PROMPT)},
        {"role": "user", "content": user},
    ]


def open_messages(state_text: str, images: list[str]) -> list[dict[str, Any]]:
    content: Any = [image_part(encoded) for encoded in images] if images else ""
    return [
        {"role": "system", "content": system_content(state_text, bool(images), LIKELIHOOD_SYSTEM_PROMPT)},
        {"role": "user", "content": content},
    ]


def reply(messages: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    return [*messages, {"role": "assistant", "content": text}]


def softmax(values: list[float]) -> list[float]:
    top = max(values)
    weights = [math.exp(value - top) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def likelihood_answer(
    item: Plan, rows: list[list[float]], context_tokens: int, boundary: int
) -> LikelihoodAnswer:
    """Each candidate's total, count and mean, a softmax over the totals, and the
    winner by the question's `rank_by` (the first in request order on a tie)."""
    question = _question(item)
    names = list(question.candidates)
    assert len(rows) == len(names)
    totals = [math.fsum(row) for row in rows]
    probabilities = softmax(totals)
    scores = {
        name: CandidateScore(
            logprob=total,
            tokens=len(row),
            mean_logprob=total / len(row),
            probability=probability,
        )
        for name, row, total, probability in zip(names, rows, totals, probabilities)
    }
    measure = (
        (lambda name: scores[name].logprob)
        if question.rank_by == "total"
        else (lambda name: scores[name].mean_logprob)
    )
    winner = names[0]
    for name in names[1:]:
        if measure(name) > measure(winner):
            winner = name
    return LikelihoodAnswer(
        winner=winner,
        rank_by=question.rank_by,
        candidates=scores,
        context_tokens=context_tokens,
        boundary_tokens=context_tokens - boundary,
    )


def named(error: ApiError, plans: list[Plan]) -> ApiError:
    """A refusal the engine made about a group and a candidate by index, said
    with the question's and the candidate's names."""
    details = dict(error.details or {})
    group = details.get("group")
    if error.code not in GROUP_CODES or not isinstance(group, int) or not 0 <= group < len(plans):
        return error
    item = plans[group]
    details["question"] = item.name
    names = list(_question(item).candidates)
    index = details.get("candidate")
    where = f"question {item.name!r}"
    if isinstance(index, int) and 0 <= index < len(names):
        details["candidate_name"] = names[index]
        where += f", candidate {names[index]!r}"
    return ApiError(error.status_code, error.code, f"{where}: {error.message}", details)


def _engine_error(engine: str, detail: str) -> ApiError:
    return ApiError(
        502,
        "engine_error",
        f"the {engine} engine's reply cannot be read as a likelihood: {detail}",
        {"engine": engine},
    )


def _int_list(container: Any, key: str, where: str, engine: str) -> list[int]:
    value = container.get(key) if isinstance(container, dict) else None
    if not isinstance(value, list) or not all(
        isinstance(n, int) and not isinstance(n, bool) for n in value
    ):
        raise _engine_error(engine, f"{where}.{key} is not a list of token ids")
    return value


def _int(container: Any, key: str, where: str, engine: str) -> int:
    value = container.get(key) if isinstance(container, dict) else None
    if not isinstance(value, int) or isinstance(value, bool):
        raise _engine_error(engine, f"{where}.{key} is not an integer")
    return value


# --- one batched items request (mlx-lm, mlx-vlm) -----------------------------------


def likelihood_body(
    engine_model_name: str, msgs: list[dict[str, Any]], plans: list[Plan], cap: int
) -> dict[str, Any]:
    return {
        "model": engine_model_name,
        "messages": msgs,
        "candidates": [
            {"question": _question(item).instructions,
             "texts": list(_question(item).candidates.values())}
            for item in plans
        ],
        "chat_template_kwargs": dict(TEMPLATE_KWARGS),
        "max_prompt_tokens": cap,
        "max_item_tokens": cap,
        "max_candidate_tokens": MAX_CANDIDATE_TOKENS,
    }


def read_likelihood_reply(
    data: Any, engine: str, plans: list[Plan]
) -> tuple[list[tuple[list[list[float]], int, int]], int | None]:
    """Each group's candidate rows, context tokens and boundary, and the state
    tokens the engine reused (null when it did not say)."""
    groups = data.get("groups") if isinstance(data, dict) else None
    if not isinstance(groups, list) or len(groups) != len(plans):
        raise _engine_error(engine, f"reply.groups is not a list of {len(plans)} groups")
    read = []
    for index, (group, item) in enumerate(zip(groups, plans)):
        where = f"groups[{index}]"
        context = _int(group, "context_tokens", where, engine)
        boundary = _int(group, "boundary", where, engine)
        rows = group.get("candidates") if isinstance(group, dict) else None
        expected = len(_question(item).candidates)
        if not isinstance(rows, list) or len(rows) != expected:
            raise _engine_error(engine, f"{where}.candidates is not a list of {expected}")
        values = []
        for n, row in enumerate(rows):
            logprobs = row.get("logprobs") if isinstance(row, dict) else None
            if (
                not isinstance(logprobs, list)
                or not logprobs
                or not all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in logprobs
                )
            ):
                raise _engine_error(
                    engine, f"{where}.candidates[{n}].logprobs is not a non-empty list of numbers"
                )
            values.append([float(v) for v in logprobs])
        read.append((values, context, boundary))
    cached = data.get("cached_tokens")
    if cached is not None and (isinstance(cached, bool) or not isinstance(cached, int)):
        raise _engine_error(engine, f"reply.cached_tokens is {type(cached).__name__}, expected int")
    return read, cached


async def score_on_items(
    call: EngineCall, resident: Any, body: DecideRequest, plans: list[Plan]
) -> dict[str, tuple[Answer, ForwardTiming, int]]:
    """Every likelihood question of the request in ONE items request: the state
    once (or not at all, when the engine still holds it), every candidate a row."""
    images = list(body.images or [])
    wire = likelihood_body(
        resident.engine_model_name, open_messages(render_state(body.state), images), plans,
        prompt_cap(resident.max_model_len),
    )
    sent = time.perf_counter()
    try:
        data = await call(ITEMS_PATH, wire)
    except ApiError as error:
        raise named(error, plans) from None
    wall_ms = round((time.perf_counter() - sent) * 1000.0, 1)
    groups, cached = read_likelihood_reply(data, resident.engine, plans)
    scored: dict[str, tuple[Answer, ForwardTiming, int]] = {}
    for item, (rows, context, boundary) in zip(plans, groups):
        prompt_tokens = sum(boundary + len(row) for row in rows)
        scored[item.name] = (
            likelihood_answer(item, rows, context, boundary),
            ForwardTiming(wall_ms=wall_ms, prompt_tokens=prompt_tokens, cached_tokens=cached),
            prompt_tokens,
        )
    return scored


# --- one request per candidate, prompt logprobs (vLLM) -----------------------------


def tokenize_body(
    engine_model_name: str, msgs: list[dict[str, Any]], *, reply_open: bool
) -> dict[str, Any]:
    return {
        "model": engine_model_name,
        "messages": msgs,
        "add_generation_prompt": not reply_open,
        "continue_final_message": reply_open,
        "chat_template_kwargs": dict(TEMPLATE_KWARGS),
    }


def prompt_logprobs_body(engine_model_name: str, msgs: list[dict[str, Any]]) -> dict[str, Any]:
    """The candidate's prompt with every prompt token's log-probability asked for
    (0 alternatives: the prompt token's own), and the prompt's ids, so the read
    can be checked against what /tokenize said. One token is decoded because a
    chat completion must decode one; nobody reads it."""
    return {
        "model": engine_model_name,
        "messages": msgs,
        "add_generation_prompt": False,
        "continue_final_message": True,
        "chat_template_kwargs": dict(TEMPLATE_KWARGS),
        "max_tokens": 1,
        "temperature": 0,
        "prompt_logprobs": 0,
        "return_token_ids": True,
        "stream": False,
    }


def read_prompt_logprobs(
    data: Any, engine: str, expected: list[int], boundary: int
) -> tuple[list[float], int, int | None]:
    """The log-probabilities of `expected[boundary:]`, read off a prompt-logprobs
    reply whose prompt must be exactly `expected`; and its usage."""
    ids = _int_list(data, "prompt_token_ids", "reply", engine)
    if ids != expected:
        raise _engine_error(
            engine,
            f"the completion's prompt ({len(ids)} tokens) is not the prompt /tokenize "
            f"rendered for it ({len(expected)} tokens); the two disagree about the template",
        )
    entries = data.get("prompt_logprobs") if isinstance(data, dict) else None
    if not isinstance(entries, list) or len(entries) != len(ids):
        raise _engine_error(engine, f"reply.prompt_logprobs is not a list of {len(ids)} positions")
    read: list[float] = []
    for position in range(boundary, len(ids)):
        entry = entries[position]
        token = str(ids[position])
        value = entry.get(token) if isinstance(entry, dict) else None
        logprob = value.get("logprob") if isinstance(value, dict) else None
        if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
            raise _engine_error(
                engine, f"prompt_logprobs[{position}] carries no logprob for token {token}"
            )
        if logprob <= VLLM_CLAMPED_LOGPROB or not math.isfinite(logprob):
            raise _engine_error(
                engine,
                f"prompt_logprobs[{position}] is {logprob}: the model gives token {token} no "
                "probability at all",
            )
        read.append(float(logprob))
    usage = data.get("usage") if isinstance(data, dict) else None
    prompt_tokens = _int(usage, "prompt_tokens", "usage", engine)
    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    if cached is not None and (isinstance(cached, bool) or not isinstance(cached, int)):
        raise _engine_error(engine, "usage.prompt_tokens_details.cached_tokens is not an integer")
    return read, prompt_tokens, cached


async def score_on_prompt_logprobs(
    call: EngineCall,
    resident: Any,
    body: DecideRequest,
    plans: list[Plan],
    *,
    concurrency: int,
) -> dict[str, tuple[Answer, ForwardTiming, int]]:
    """Each question's context and candidates tokenized through the engine's own
    template first (no forward pass), so every refusal is made before anything
    is scored; then one prompt-logprobs request per candidate under the
    engine's admission."""
    state_text = render_state(body.state)
    images = list(body.images or [])
    engine, name = resident.engine, resident.engine_model_name
    cap = prompt_cap(resident.max_model_len)

    async def tokens(msgs: list[dict[str, Any]], reply_open: bool) -> list[int]:
        data = await call(TOKENIZE_PATH, tokenize_body(name, msgs, reply_open=reply_open))
        return _int_list(data, "tokens", "tokenize reply", engine)

    contexts: list[list[int]] = []
    prompts: list[list[list[int]]] = []
    for item in plans:
        question = _question(item)
        msgs = likelihood_messages(state_text, images, question.instructions)
        contexts.append(await tokens(msgs, False))
        prompts.append(
            list(await asyncio.gather(*(tokens(reply(msgs, text), True)
                                        for text in question.candidates.values())))
        )
    try:
        split = likelihood_split(contexts, prompts, cap, cap, MAX_CANDIDATE_TOKENS)
    except ItemsRefusal as refusal:
        raise named(refusal_error(refusal), plans) from None

    gate = asyncio.Semaphore(concurrency)

    async def score(msgs: list[dict[str, Any]], expected: list[int], boundary: int) -> Any:
        async with gate:
            data = await call(CHAT_PATH, prompt_logprobs_body(name, msgs))
        return read_prompt_logprobs(data, engine, expected, boundary)

    scored: dict[str, tuple[Answer, ForwardTiming, int]] = {}
    for group, item in enumerate(plans):
        question = _question(item)
        msgs = likelihood_messages(state_text, images, question.instructions)
        boundary = split.boundaries[group]
        sent = time.perf_counter()
        results = await asyncio.gather(*(
            score(reply(msgs, text), expected, boundary)
            for text, expected in zip(question.candidates.values(), prompts[group])
        ))
        wall_ms = round((time.perf_counter() - sent) * 1000.0, 1)
        rows = [row for row, _, _ in results]
        prompt_tokens = sum(n for _, n, _ in results)
        reported = [cached for _, _, cached in results]
        cached = None if any(c is None for c in reported) else sum(c for c in reported if c is not None)
        scored[item.name] = (
            likelihood_answer(item, rows, split.context_tokens[group], boundary),
            ForwardTiming(wall_ms=wall_ms, prompt_tokens=prompt_tokens, cached_tokens=cached),
            prompt_tokens,
        )
    return scored


# --- the questions form with likelihood questions in it ----------------------------


async def decide_with_likelihood(
    plans: list[Plan],
    labelled: Callable[[], Awaitable[DecideResponse]] | None,
    scored: Callable[[], Awaitable[dict[str, tuple[Answer, ForwardTiming, int]]]],
    resident: Any,
    n_images: int,
) -> DecideResponse:
    """The label questions answered as they always are, then the likelihood
    questions, one after the other so the two never hold more of the engine than
    its admission; the answers in the request's question order."""
    started = time.perf_counter()
    labels = await labelled() if labelled is not None else None
    likely = await scored()
    answers: dict[str, Answer] = {}
    per_question: dict[str, ForwardTiming] = {}
    tokens: dict[str, int] = {}
    for item in plans:
        if item.name in likely:
            answers[item.name], per_question[item.name], tokens[item.name] = likely[item.name]
            continue
        assert labels is not None
        answers[item.name] = labels.answers[item.name]
        per_question[item.name] = labels.timing_ms.per_question[item.name]
        tokens[item.name] = labels.tokens.per_question[item.name]
    return DecideResponse(
        model=ModelProvenance(
            id=resident.model_id, revision=resident.revision, fingerprint=resident.fingerprint
        ),
        engine=resident.engine,
        answers=answers,
        timing_ms=DecideTiming(
            total=round((time.perf_counter() - started) * 1000.0, 1),
            per_question=per_question,
            prime=None if labels is None else labels.timing_ms.prime,
        ),
        tokens=DecideTokens(per_question=tokens, images=n_images),
    )


def refuse_unscorable(model: str, engine: str, reading: Any, n_images: int) -> None:
    """A likelihood question on an engine that cannot score candidates, or with
    images on one that scores them from text only, is refused by name before
    anything is sent (or waited for)."""
    if reading.route is None:
        raise ApiError(
            400,
            "likelihood_unsupported_on_engine",
            f"{model!r} runs on {engine}, which cannot score a likelihood question: "
            f"{reading.basis}. Send it to a model served by vLLM, mlx-lm or mlx-vlm",
            {"model": model, "engine": engine},
        )
    if n_images and not reading.images:
        raise ApiError(
            400,
            "likelihood_images_unsupported_on_engine",
            f"{model!r} runs on {engine}, which scores likelihood questions from text "
            f"only, and this decision carries {n_images} image(s): {reading.basis}. "
            "Send the likelihood questions without `images`",
            {"model": model, "engine": engine, "images": n_images},
        )


__all__ = [
    "CHAT_PATH",
    "TOKENIZE_PATH",
    "decide_with_likelihood",
    "likelihood_answer",
    "likelihood_body",
    "likelihood_messages",
    "named",
    "open_messages",
    "prompt_logprobs_body",
    "read_likelihood_reply",
    "read_prompt_logprobs",
    "refuse_unscorable",
    "score_on_items",
    "score_on_prompt_logprobs",
    "softmax",
    "tokenize_body",
]
