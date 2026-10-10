from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
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
    item_messages,
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
    """Each candidate's total, count and mean, its probability by the question's
    `normalize` (a softmax over the totals, or each total's own exp), and the
    winner by its `rank_by` (the first in request order on a tie)."""
    question = _question(item)
    names = list(question.candidates)
    assert len(rows) == len(names)
    totals = [math.fsum(row) for row in rows]
    probabilities = (
        softmax(totals) if question.normalize == "softmax"
        else [math.exp(total) for total in totals]
    )
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
        normalize=question.normalize,
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


# --- what is scored, whoever asks ----------------------------------------------------


@dataclass(frozen=True)
class Scoring:
    """Groups of candidate replies to score, each group a context its candidates
    continue: the decision door's likelihood questions, and the rerank door's
    documents (each a group of two, "yes" and "no").

    The chat form (`messages`): the open turns, ending in the user turn every group's
    `question` fills, rendered by the model's chat template with the candidate as the
    open assistant reply. The prompt form (`prompt`): text Crucible rendered from the
    model's manifest, each group's `question` and then each candidate appended to it
    and tokenized as they are. Exactly one of the two is set."""

    messages: list[dict[str, Any]] | None
    prompt: str | None
    questions: list[str]
    candidates: list[list[str]]

    def __post_init__(self) -> None:
        assert (self.messages is None) != (self.prompt is None), "one form, chat or prompt"
        assert len(self.questions) == len(self.candidates) and self.questions

    def context(self, group: int) -> list[dict[str, Any]]:
        assert self.messages is not None
        return item_messages(self.messages, self.questions[group])

    def text(self, group: int) -> str:
        assert self.prompt is not None
        return self.prompt + self.questions[group]


@dataclass(frozen=True)
class GroupScore:
    """One group's candidates scored: each candidate's log-probability at each of its
    tokens from the group's boundary, and what that took."""

    rows: list[list[float]]
    context_tokens: int
    boundary: int
    timing: ForwardTiming
    prompt_tokens: int


Refusal = Callable[[ApiError], ApiError]


def _as_is(error: ApiError) -> ApiError:
    return error


# --- one batched items request (mlx-lm, mlx-vlm) -----------------------------------


def scoring_body(engine_model_name: str, scoring: Scoring, cap: int) -> dict[str, Any]:
    form: dict[str, Any] = (
        {"messages": scoring.messages, "chat_template_kwargs": dict(TEMPLATE_KWARGS)}
        if scoring.messages is not None
        else {"prompt": scoring.prompt}
    )
    return {
        "model": engine_model_name,
        **form,
        "candidates": [
            {"question": question, "texts": list(texts)}
            for question, texts in zip(scoring.questions, scoring.candidates)
        ],
        "max_prompt_tokens": cap,
        "max_item_tokens": cap,
        "max_candidate_tokens": MAX_CANDIDATE_TOKENS,
    }


def _decide_scoring(body: DecideRequest, plans: list[Plan]) -> Scoring:
    return Scoring(
        messages=open_messages(render_state(body.state), list(body.images or [])),
        prompt=None,
        questions=[_question(item).instructions for item in plans],
        candidates=[list(_question(item).candidates.values()) for item in plans],
    )


def likelihood_body(
    engine_model_name: str, msgs: list[dict[str, Any]], plans: list[Plan], cap: int
) -> dict[str, Any]:
    return scoring_body(
        engine_model_name,
        Scoring(
            messages=msgs,
            prompt=None,
            questions=[_question(item).instructions for item in plans],
            candidates=[list(_question(item).candidates.values()) for item in plans],
        ),
        cap,
    )


def read_groups_reply(
    data: Any, engine: str, counts: list[int]
) -> tuple[list[tuple[list[list[float]], int, int, int]], int | None]:
    """Each group's candidate rows, context tokens, boundary and the prompt tokens the
    engine read for it (`read_tokens`), and the state tokens it took from a held cache
    (null when it did not say)."""
    groups = data.get("groups") if isinstance(data, dict) else None
    if not isinstance(groups, list) or len(groups) != len(counts):
        raise _engine_error(engine, f"reply.groups is not a list of {len(counts)} groups")
    read = []
    for index, (group, expected) in enumerate(zip(groups, counts)):
        where = f"groups[{index}]"
        context = _int(group, "context_tokens", where, engine)
        boundary = _int(group, "boundary", where, engine)
        group_read = _int(group, "read_tokens", where, engine)
        rows = group.get("candidates") if isinstance(group, dict) else None
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
        if group_read > boundary * len(values):
            raise _engine_error(
                engine,
                f"{where}.read_tokens is {group_read}, more than its {len(values)} "
                f"candidate(s)' prompts of {boundary} tokens",
            )
        read.append((values, context, boundary, group_read))
    cached = data.get("cached_tokens")
    if cached is not None and (isinstance(cached, bool) or not isinstance(cached, int)):
        raise _engine_error(engine, f"reply.cached_tokens is {type(cached).__name__}, expected int")
    return read, cached


def read_likelihood_reply(
    data: Any, engine: str, plans: list[Plan]
) -> tuple[list[tuple[list[list[float]], int, int, int]], int | None]:
    return read_groups_reply(data, engine, [len(_question(item).candidates) for item in plans])


async def score_groups_on_items(
    call: EngineCall, resident: Any, scoring: Scoring, refusal: Refusal = _as_is
) -> list[GroupScore]:
    """Every group in ONE items request: the state once (or not at all, when the
    engine still holds it), each group's context once, every candidate a row."""
    wire = scoring_body(resident.engine_model_name, scoring, prompt_cap(resident.max_model_len))
    sent = time.perf_counter()
    try:
        data = await call(ITEMS_PATH, wire)
    except ApiError as error:
        raise refusal(error) from None
    wall_ms = round((time.perf_counter() - sent) * 1000.0, 1)
    groups, _held = read_groups_reply(
        data, resident.engine, [len(texts) for texts in scoring.candidates]
    )
    scored: list[GroupScore] = []
    for rows, context, boundary, group_read in groups:
        # Each candidate's prompt is the group's context to its boundary, counted as
        # llama-server is sent it; whatever of that the engine did not read for this
        # group (the state read once, a held cache) is cached.
        prompt_tokens = boundary * len(rows)
        cached = prompt_tokens - group_read
        scored.append(GroupScore(
            rows=rows,
            context_tokens=context,
            boundary=boundary,
            timing=ForwardTiming(wall_ms=wall_ms, prompt_tokens=prompt_tokens, cached_tokens=cached),
            prompt_tokens=prompt_tokens,
        ))
    return scored


def _answers(
    plans: list[Plan], scored: list[GroupScore]
) -> dict[str, tuple[Answer, ForwardTiming, int]]:
    return {
        item.name: (
            likelihood_answer(item, group.rows, group.context_tokens, group.boundary),
            group.timing,
            group.prompt_tokens,
        )
        for item, group in zip(plans, scored)
    }


async def score_on_items(
    call: EngineCall, resident: Any, body: DecideRequest, plans: list[Plan]
) -> dict[str, tuple[Answer, ForwardTiming, int]]:
    """Every likelihood question of the request in ONE items request."""
    scored = await score_groups_on_items(
        call, resident, _decide_scoring(body, plans), lambda error: named(error, plans)
    )
    return _answers(plans, scored)


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


async def score_groups_on_prompt_logprobs(
    call: EngineCall,
    resident: Any,
    scoring: Scoring,
    *,
    concurrency: int,
    refusal: Refusal = _as_is,
) -> list[GroupScore]:
    """Each group's context and candidates tokenized through the engine's own
    template first (no forward pass), so every refusal is made before anything
    is scored; then one prompt-logprobs request per candidate under the
    engine's admission. The chat form only (the engine's reading says so)."""
    assert scoring.messages is not None, "vLLM's route scores the chat form only"
    engine, name = resident.engine, resident.engine_model_name
    cap = prompt_cap(resident.max_model_len)

    async def tokens(msgs: list[dict[str, Any]], reply_open: bool) -> list[int]:
        data = await call(TOKENIZE_PATH, tokenize_body(name, msgs, reply_open=reply_open))
        return _int_list(data, "tokens", "tokenize reply", engine)

    contexts: list[list[int]] = []
    prompts: list[list[list[int]]] = []
    for group, texts in enumerate(scoring.candidates):
        msgs = scoring.context(group)
        contexts.append(await tokens(msgs, False))
        prompts.append(
            list(await asyncio.gather(*(tokens(reply(msgs, text), True) for text in texts)))
        )
    try:
        split = likelihood_split(contexts, prompts, cap, cap, MAX_CANDIDATE_TOKENS)
    except ItemsRefusal as refused:
        raise refusal(refusal_error(refused)) from None

    gate = asyncio.Semaphore(concurrency)

    async def score(msgs: list[dict[str, Any]], expected: list[int], boundary: int) -> Any:
        async with gate:
            data = await call(CHAT_PATH, prompt_logprobs_body(name, msgs))
        return read_prompt_logprobs(data, engine, expected, boundary)

    scored: list[GroupScore] = []
    for group, texts in enumerate(scoring.candidates):
        msgs = scoring.context(group)
        boundary = split.boundaries[group]
        sent = time.perf_counter()
        results = await asyncio.gather(*(
            score(reply(msgs, text), expected, boundary)
            for text, expected in zip(texts, prompts[group])
        ))
        wall_ms = round((time.perf_counter() - sent) * 1000.0, 1)
        prompt_tokens = sum(n for _, n, _ in results)
        reported = [cached for _, _, cached in results]
        cached = None if any(c is None for c in reported) else sum(c for c in reported if c is not None)
        scored.append(GroupScore(
            rows=[row for row, _, _ in results],
            context_tokens=split.context_tokens[group],
            boundary=boundary,
            timing=ForwardTiming(wall_ms=wall_ms, prompt_tokens=prompt_tokens, cached_tokens=cached),
            prompt_tokens=prompt_tokens,
        ))
    return scored


async def score_on_prompt_logprobs(
    call: EngineCall,
    resident: Any,
    body: DecideRequest,
    plans: list[Plan],
    *,
    concurrency: int,
) -> dict[str, tuple[Answer, ForwardTiming, int]]:
    scored = await score_groups_on_prompt_logprobs(
        call, resident, _decide_scoring(body, plans),
        concurrency=concurrency, refusal=lambda error: named(error, plans),
    )
    return _answers(plans, scored)


# --- one forced continuation per candidate (llama-server) --------------------------


APPLY_TEMPLATE_PATH = "/apply-template"

COMPLETION_PATH = "/completion"

LLAMA_NO_PROBABILITY = -1e30
"""llama-server writes a probability of 0 as the float's lowest value,
-3.4e38 (`completion_token_output::logarithm`, tools/server/server-task.cpp
L304-307, b10970): the model gives that token no probability at all, which is
no number. Anything at or below this is read as that."""


def template_body(msgs: list[dict[str, Any]], *, reply_open: bool) -> dict[str, Any]:
    """llama-server's /apply-template renders through the parser its chat
    completions use (oaicompat_chat_params_parse), so the context and each
    candidate's open reply are the prompts its own chat path would send."""
    return {
        "messages": msgs,
        "add_generation_prompt": not reply_open,
        "continue_final_message": reply_open,
        "chat_template_kwargs": dict(TEMPLATE_KWARGS),
    }


def llama_tokenize_body(prompt: str, *, add_special: bool = True) -> dict[str, Any]:
    """The rendered prompt tokenized as the chat path tokenizes it: special
    tokens added where the model's vocabulary says to, and the template's
    special-token text read as those tokens. A Crucible-rendered prompt (the
    prompt form) adds none: everything the model reads is in the text."""
    return {"content": prompt, "add_special": add_special, "parse_special": True}


def forced_grammar(targets: list[int]) -> str:
    """A GBNF grammar that admits exactly this token sequence, by token id
    (`<[id]>`, src/llama-grammar.cpp parse_token L186-230, b10970), so the
    candidate is continued as the very tokens the boundary scored, not as
    whatever other tokenization of its text the model likes better."""
    return "root ::= " + " ".join(f"<[{token}]>" for token in targets)


def forced_body(prompt: list[int], targets: list[int]) -> dict[str, Any]:
    """The context up to the boundary as token ids, continued by exactly the
    candidate's tokens, each generated token's probability read from the raw
    logits (`post_sampling_probs: false`). `cache_prompt` lets every candidate
    of a question continue from the same cached context."""
    return {
        "prompt": prompt,
        "n_predict": len(targets),
        "grammar": forced_grammar(targets),
        "n_probs": 1,
        "post_sampling_probs": False,
        "temperature": 0,
        "cache_prompt": True,
        "return_tokens": True,
        "stream": False,
    }


def read_forced(data: Any, engine: str, targets: list[int]) -> tuple[list[float], int]:
    """The log-probability of each forced token, read off a /completion reply
    that must have generated exactly `targets`; and the prompt tokens the
    engine read from its cache."""
    tokens = _int_list(data, "tokens", "reply", engine)
    if tokens != targets:
        raise _engine_error(
            engine,
            f"the forced continuation generated {len(tokens)} tokens {tokens[:8]} and "
            f"the candidate is {len(targets)} tokens {targets[:8]}; the grammar did not "
            "hold the reply to the candidate",
        )
    entries = data.get("completion_probabilities") if isinstance(data, dict) else None
    if not isinstance(entries, list) or len(entries) != len(targets):
        raise _engine_error(
            engine, f"reply.completion_probabilities is not a list of {len(targets)} tokens"
        )
    read: list[float] = []
    for position, (entry, token) in enumerate(zip(entries, targets)):
        where = f"completion_probabilities[{position}]"
        if _int(entry, "id", where, engine) != token:
            raise _engine_error(engine, f"{where}.id is not the candidate's token {token}")
        logprob = entry.get("logprob")
        if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
            raise _engine_error(engine, f"{where} carries no logprob")
        if logprob <= LLAMA_NO_PROBABILITY or not math.isfinite(logprob):
            raise _engine_error(
                engine,
                f"{where}.logprob is {logprob}: the model gives token {token} no "
                "probability at all",
            )
        read.append(float(logprob))
    timings = data.get("timings") if isinstance(data, dict) else None
    return read, _int(timings, "cache_n", "reply.timings", engine)


async def score_groups_on_forced_tokens(
    call: EngineCall, resident: Any, scoring: Scoring, refusal: Refusal = _as_is
) -> list[GroupScore]:
    """Each group's context and candidates rendered and tokenized through the
    engine first (the chat form through its own template, the prompt form as it
    is), so every refusal is made before anything is scored; then each candidate
    as a forced continuation of its context, one request after another
    (llama-server has one slot). The context is the same prompt for every
    candidate of a group, and groups that share a prefix (a query) follow one
    another, so the engine reuses its cache and reads only what differs (a
    recurrent layer is restored from the checkpoint it took near the prompt's end)."""
    engine = resident.engine
    cap = prompt_cap(resident.max_model_len)

    async def chat_tokens(msgs: list[dict[str, Any]], reply_open: bool) -> list[int]:
        rendered = await call(APPLY_TEMPLATE_PATH, template_body(msgs, reply_open=reply_open))
        prompt = rendered.get("prompt") if isinstance(rendered, dict) else None
        if not isinstance(prompt, str):
            raise _engine_error(engine, "the /apply-template reply carries no prompt")
        data = await call(TOKENIZE_PATH, llama_tokenize_body(prompt))
        return _int_list(data, "tokens", "tokenize reply", engine)

    async def text_tokens(text: str) -> list[int]:
        data = await call(TOKENIZE_PATH, llama_tokenize_body(text, add_special=False))
        return _int_list(data, "tokens", "tokenize reply", engine)

    # One render and one tokenize at a time: llama-server answers them on its
    # HTTP threads, and 256 candidates sent at once took twice as long (4.4 s
    # against 2.0 s, b10970 on owens-pc, 2026-10-10).
    contexts: list[list[int]] = []
    prompts: list[list[list[int]]] = []
    for group, texts in enumerate(scoring.candidates):
        if scoring.messages is not None:
            msgs = scoring.context(group)
            contexts.append(await chat_tokens(msgs, False))
            prompts.append([await chat_tokens(reply(msgs, text), True) for text in texts])
        else:
            context = scoring.text(group)
            contexts.append(await text_tokens(context))
            prompts.append([await text_tokens(context + text) for text in texts])
    try:
        split = likelihood_split(contexts, prompts, cap, cap, MAX_CANDIDATE_TOKENS)
    except ItemsRefusal as refused:
        raise refusal(refusal_error(refused)) from None

    scored: list[GroupScore] = []
    for group in range(len(scoring.candidates)):
        boundary = split.boundaries[group]
        sent = time.perf_counter()
        rows: list[list[float]] = []
        cached = 0
        for expected in prompts[group]:
            targets = expected[boundary:]
            data = await call(COMPLETION_PATH, forced_body(expected[:boundary], targets))
            row, reused = read_forced(data, engine, targets)
            rows.append(row)
            cached += reused
        wall_ms = round((time.perf_counter() - sent) * 1000.0, 1)
        prompt_tokens = boundary * len(rows)
        scored.append(GroupScore(
            rows=rows,
            context_tokens=split.context_tokens[group],
            boundary=boundary,
            timing=ForwardTiming(wall_ms=wall_ms, prompt_tokens=prompt_tokens, cached_tokens=cached),
            prompt_tokens=prompt_tokens,
        ))
    return scored


async def score_on_forced_tokens(
    call: EngineCall,
    resident: Any,
    body: DecideRequest,
    plans: list[Plan],
) -> dict[str, tuple[Answer, ForwardTiming, int]]:
    scored = await score_groups_on_forced_tokens(
        call, resident, _decide_scoring(body, plans), lambda error: named(error, plans)
    )
    return _answers(plans, scored)


async def score_groups(
    call: EngineCall,
    resident: Any,
    scoring: Scoring,
    route: str | None,
    *,
    concurrency: int,
    refusal: Refusal = _as_is,
) -> list[GroupScore]:
    """`scoring` on the engine's likelihood route (engines.likelihood_reading)."""
    if route == "items":
        return await score_groups_on_items(call, resident, scoring, refusal)
    if route == "forced-tokens":
        return await score_groups_on_forced_tokens(call, resident, scoring, refusal)
    assert route == "prompt-logprobs", route
    return await score_groups_on_prompt_logprobs(
        call, resident, scoring, concurrency=concurrency, refusal=refusal
    )


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
            f"{reading.basis}. Send it to a model served by vLLM, llama-server, mlx-lm "
            "or mlx-vlm",
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
    "APPLY_TEMPLATE_PATH",
    "CHAT_PATH",
    "COMPLETION_PATH",
    "TOKENIZE_PATH",
    "GroupScore",
    "Scoring",
    "decide_with_likelihood",
    "forced_body",
    "read_groups_reply",
    "score_groups",
    "score_groups_on_forced_tokens",
    "score_groups_on_items",
    "score_groups_on_prompt_logprobs",
    "scoring_body",
    "forced_grammar",
    "likelihood_answer",
    "likelihood_body",
    "likelihood_messages",
    "named",
    "open_messages",
    "prompt_logprobs_body",
    "read_likelihood_reply",
    "read_forced",
    "read_prompt_logprobs",
    "refuse_unscorable",
    "score_on_items",
    "score_on_forced_tokens",
    "score_on_prompt_logprobs",
    "softmax",
    "template_body",
    "tokenize_body",
]
