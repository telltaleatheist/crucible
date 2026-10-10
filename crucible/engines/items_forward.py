from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

ITEMS_VERSION = 7

ITEMS_PATH = "/v1/crucible/items"

MAX_TOP_LOGPROBS = 40

CHUNK_TOKENS = 2048
"""Tokens one forward reads where the engine states no step of its own (the
mlx-vlm reader). On mlx-lm every forward here reads at most the engine's
--prefill-step-size tokens (rows x positions), which Crucible derives from the
model's size so one evaluation holds the GPU about 1.6 s (engines/mlx_lm.py,
EVAL_BUDGET_FLOPS): a longer one can freeze the desktop for all of it."""

ROW_BYTES = 512 << 20
"""Bytes of copied shared-state cache one batched item forward may hold. Every
row of the batch is the shared state's cache repeated, so this bounds the rows
per forward: a 300-token state on qwen3.5-9b is ~50 MB a row (24 linear-
attention layers of float32 recurrent state, 2 MiB each, plus 8 layers of KV),
so ~10 rows; a 32k-token state is ~1 GB a row, so one."""

MAX_ROWS = 32

STATE_ENTRIES = 4

STATE_BYTES = 1 << 30
"""The engine keeps the caches of the last STATE_ENTRIES shared states it read,
at most this many bytes together, so the next decision about the same state
skips its prefill. A state larger than this is read and not kept."""

ITEMS_FIELDS = frozenset(
    {
        "model",
        "messages",
        "questions",
        "chat_template_kwargs",
        "top_logprobs",
        "max_prompt_tokens",
        "max_item_tokens",
    }
)

LIKELIHOOD_FIELDS = frozenset(
    {
        "model",
        "messages",
        "prompt",
        "candidates",
        "chat_template_kwargs",
        "max_prompt_tokens",
        "max_item_tokens",
        "max_candidate_tokens",
    }
)

EMBED_FIELDS = frozenset({"model", "inputs", "max_input_tokens"})

EMBED_INPUT_TOO_LONG = "embed_input_too_long"

SCORE_CHUNK = 64
"""Positions one head application reads when a candidate's tokens are scored:
64 rows of float32 logits over a 248k vocabulary is ~63 MB."""

TEMPLATE_KWARGS = frozenset({"enable_thinking"})

ITEM_TOO_LONG = "item_too_long"

PROMPT_TOO_LONG = "item_prompt_too_long"

CANDIDATE_TOO_LONG = "candidate_too_long"

CANDIDATE_NOT_A_REPLY = "candidate_not_a_reply"

BOUNDARY_SLACK = 1
"""Context tokens a candidate may re-tokenize. Its first characters can merge
with the end of the open assistant turn into one token, and that end is one
pre-token (the blank line after Qwen's empty think block), so at most its last token
is read again with the candidate. More than that means the chat template did
not render the candidate as the continuation of the reply it opens."""


class ItemsRefusal(Exception):
    def __init__(
        self, status: int, code: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True)
class ItemsAsk:
    model: str
    messages: list[dict[str, Any]]
    questions: list[str]
    template_kwargs: dict[str, Any]
    top_logprobs: int
    max_prompt_tokens: int
    max_item_tokens: int


@dataclass(frozen=True)
class Split:
    shared: list[int]
    suffixes: list[list[int]]


def _positive_int(body: dict[str, Any], key: str, ceiling: int | None = None) -> int:
    value = body.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ItemsRefusal(400, "bad_" + key, f"{key} must be a positive int, not {value!r}")
    if ceiling is not None and value > ceiling:
        raise ItemsRefusal(
            400,
            "too_many_top_logprobs",
            f"{key} is {value} and this server returns at most {ceiling}; ask for "
            f"{ceiling} or fewer",
        )
    return value


def _template_kwargs(body: dict[str, Any]) -> dict[str, Any]:
    kwargs = body.get("chat_template_kwargs") or {}
    if not isinstance(kwargs, dict) or set(kwargs) - TEMPLATE_KWARGS:
        raise ItemsRefusal(
            400,
            "unknown_template_kwarg",
            f"chat_template_kwargs must be an object of {sorted(TEMPLATE_KWARGS)}, "
            f"not {kwargs!r}",
        )
    return dict(kwargs)


def _questions(body: dict[str, Any]) -> list[str]:
    questions = body.get("questions")
    if (
        not isinstance(questions, list)
        or not questions
        or not all(isinstance(text, str) and text for text in questions)
    ):
        raise ItemsRefusal(
            400, "bad_questions", "questions must be a non-empty list of non-empty strings"
        )
    return questions


def _messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ItemsRefusal(400, "no_messages", "messages must be a non-empty list of turns")
    last = messages[-1]
    content = last.get("content") if isinstance(last, dict) else None
    if (
        not isinstance(last, dict)
        or last.get("role") != "user"
        or not (content == "" or isinstance(content, list))
    ):
        raise ItemsRefusal(
            400,
            "open_user_turn",
            "the last message must be the user turn every question completes: role "
            "'user' with content \"\" or a list of image parts",
        )
    return messages


def parse_items(body: Any, served: Sequence[str]) -> ItemsAsk:
    if not isinstance(body, dict):
        raise ItemsRefusal(400, "bad_json", "the body is not a JSON object")
    unknown = sorted(set(body) - ITEMS_FIELDS)
    if unknown:
        raise ItemsRefusal(
            400,
            "unknown_field",
            f"an items request does not carry {unknown}; it reads {sorted(ITEMS_FIELDS)}",
        )
    if body.get("model") not in served:
        raise ItemsRefusal(
            404, "model_not_found", f"{body.get('model')!r} is not loaded; {served[0]!r} is"
        )
    return ItemsAsk(
        model=body["model"],
        messages=_messages(body),
        questions=_questions(body),
        template_kwargs=_template_kwargs(body),
        top_logprobs=_positive_int(body, "top_logprobs", MAX_TOP_LOGPROBS),
        max_prompt_tokens=_positive_int(body, "max_prompt_tokens"),
        max_item_tokens=_positive_int(body, "max_item_tokens"),
    )


@dataclass(frozen=True)
class CandidateGroup:
    question: str
    texts: list[str]


@dataclass(frozen=True)
class LikelihoodAsk:
    model: str
    messages: list[dict[str, Any]] | None
    """The chat form: the turns every question completes, rendered by the model's chat
    template; None in the prompt form."""
    groups: list[CandidateGroup]
    template_kwargs: dict[str, Any]
    max_prompt_tokens: int
    max_item_tokens: int
    max_candidate_tokens: int
    prompt: str | None = None
    """The prompt form: text every question continues, as it is (Crucible rendered it
    from the model's manifest); each question's text, then each candidate's, appended to
    it and tokenized whole with no special tokens added. None in the chat form."""


def _groups(body: dict[str, Any]) -> list[CandidateGroup]:
    groups = body.get("candidates")
    shape = (
        "candidates must be a non-empty list of {question, texts}: a non-empty question "
        "and at least one non-empty candidate text"
    )
    if not isinstance(groups, list) or not groups:
        raise ItemsRefusal(400, "bad_candidates", shape)
    read: list[CandidateGroup] = []
    for group in groups:
        if not isinstance(group, dict) or set(group) != {"question", "texts"}:
            raise ItemsRefusal(400, "bad_candidates", shape)
        question, texts = group["question"], group["texts"]
        if (
            not isinstance(question, str)
            or not question
            or not isinstance(texts, list)
            or not texts
            or not all(isinstance(text, str) and text for text in texts)
        ):
            raise ItemsRefusal(400, "bad_candidates", shape)
        read.append(CandidateGroup(question=question, texts=list(texts)))
    return read


def _prompt(body: dict[str, Any]) -> str:
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ItemsRefusal(400, "bad_prompt", "prompt must be a non-empty string")
    if "messages" in body or "chat_template_kwargs" in body:
        raise ItemsRefusal(
            400,
            "prompt_and_messages",
            "a likelihood request is the chat form (messages, chat_template_kwargs) or the "
            "prompt form (prompt), never both",
        )
    return prompt


def parse_likelihood(body: dict[str, Any], served: Sequence[str]) -> LikelihoodAsk:
    unknown = sorted(set(body) - LIKELIHOOD_FIELDS)
    if unknown:
        raise ItemsRefusal(
            400,
            "unknown_field",
            f"a likelihood request does not carry {unknown}; it reads "
            f"{sorted(LIKELIHOOD_FIELDS)}",
        )
    if body.get("model") not in served:
        raise ItemsRefusal(
            404, "model_not_found", f"{body.get('model')!r} is not loaded; {served[0]!r} is"
        )
    raw = "prompt" in body
    return LikelihoodAsk(
        model=body["model"],
        messages=None if raw else _messages(body),
        groups=_groups(body),
        template_kwargs={} if raw else _template_kwargs(body),
        max_prompt_tokens=_positive_int(body, "max_prompt_tokens"),
        max_item_tokens=_positive_int(body, "max_item_tokens"),
        max_candidate_tokens=_positive_int(body, "max_candidate_tokens"),
        prompt=_prompt(body) if raw else None,
    )


@dataclass(frozen=True)
class EmbedAsk:
    """Texts to vectors: each input as it is (Crucible rendered it from the model's
    manifest, its end-of-text marker included), tokenized with no special tokens added;
    its vector is the model's last hidden state at its last token, unnormalised."""

    model: str
    inputs: list[str]
    max_input_tokens: int


def parse_embed(body: dict[str, Any], served: Sequence[str]) -> EmbedAsk:
    unknown = sorted(set(body) - EMBED_FIELDS)
    if unknown:
        raise ItemsRefusal(
            400,
            "unknown_field",
            f"an embed request does not carry {unknown}; it reads {sorted(EMBED_FIELDS)}",
        )
    if body.get("model") not in served:
        raise ItemsRefusal(
            404, "model_not_found", f"{body.get('model')!r} is not loaded; {served[0]!r} is"
        )
    inputs = body.get("inputs")
    if (
        not isinstance(inputs, list)
        or not inputs
        or not all(isinstance(text, str) and text for text in inputs)
    ):
        raise ItemsRefusal(
            400, "bad_inputs", "inputs must be a non-empty list of non-empty strings"
        )
    return EmbedAsk(
        model=body["model"],
        inputs=list(inputs),
        max_input_tokens=_positive_int(body, "max_input_tokens"),
    )


def parse_request(body: Any, served: Sequence[str]) -> ItemsAsk | LikelihoodAsk | EmbedAsk:
    """The items route reads three asks: questions, whose answer is the top tokens
    at each prompt's end; candidates, whose answer is the log-probability of
    every token of each candidate reply; and inputs, whose answer is each one's
    vector."""
    if isinstance(body, dict) and "inputs" in body:
        return parse_embed(body, served)
    if isinstance(body, dict) and "candidates" in body:
        return parse_likelihood(body, served)
    return parse_items(body, served)


def item_messages(messages: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    last = messages[-1]
    content = last["content"]
    if isinstance(content, list):
        filled: Any = [*content, {"type": "text", "text": question}]
    else:
        filled = question
    return [*messages[:-1], {**last, "content": filled}]


def reply_messages(
    messages: list[dict[str, Any]], question: str, reply: str
) -> list[dict[str, Any]]:
    """The question's turns with the candidate as the open assistant reply: the
    chat template renders it with `continue_final_message`, so the prompt ends
    on the candidate's last character."""
    return [*item_messages(messages, question), {"role": "assistant", "content": reply}]


@dataclass(frozen=True)
class Scored:
    group: int
    start: int
    """Where in the row's suffix the hidden state that predicts the first
    scored token is: the token before the boundary."""
    targets: list[int]
    """The candidate's tokens, from the boundary to the end of its prompt."""


@dataclass(frozen=True)
class LikelihoodSplit:
    shared: list[int]
    suffixes: list[list[int]]
    scored: list[Scored]
    context_tokens: list[int]
    boundaries: list[int]


def likelihood_split(
    contexts: Sequence[Sequence[int]],
    candidates: Sequence[Sequence[Sequence[int]]],
    max_prompt_tokens: int,
    max_item_tokens: int,
    max_candidate_tokens: int,
) -> LikelihoodSplit:
    """Where each group's context ends in its candidates' prompts, and the rows
    one forward reads. A group's boundary is the token prefix its context (the
    question with the reply opened) shares with every candidate's prompt, each
    tokenized whole, so a candidate is scored as it tokenizes after the context:
    a first token that merged across the boundary is scored, and it costs the
    context at most BOUNDARY_SLACK tokens. Every candidate of a group is scored
    from the same boundary, so their totals compare the same thing. The shared
    part every row continues from stops before the earliest boundary: a row has
    to read the hidden state that predicts its first scored token."""
    boundaries: list[int] = []
    for group, (context, prompts) in enumerate(zip(contexts, candidates)):
        common = common_prefix([context, *prompts])
        if common < max(1, len(context) - BOUNDARY_SLACK):
            raise ItemsRefusal(
                400,
                CANDIDATE_NOT_A_REPLY,
                f"group {group}'s candidates share {common} tokens with its context of "
                f"{len(context)}; a candidate may re-read at most {BOUNDARY_SLACK} "
                "context token, so the chat template did not render the candidates as "
                "the reply the context opens",
                {"group": group, "context_tokens": len(context), "common_tokens": common},
            )
        for index, prompt in enumerate(prompts):
            scored = len(prompt) - common
            if scored < 1:
                raise ItemsRefusal(
                    400,
                    CANDIDATE_NOT_A_REPLY,
                    f"group {group}'s candidate {index} adds no token to its context",
                    {"group": group, "candidate": index, "tokens": scored},
                )
            if scored > max_candidate_tokens:
                raise ItemsRefusal(
                    400,
                    CANDIDATE_TOO_LONG,
                    f"group {group}'s candidate {index} is {scored} tokens; one candidate "
                    f"may be at most {max_candidate_tokens}",
                    {"group": group, "candidate": index, "tokens": scored,
                     "max_tokens": max_candidate_tokens},
                )
            if len(prompt) > max_prompt_tokens:
                raise ItemsRefusal(
                    400,
                    PROMPT_TOO_LONG,
                    f"group {group}'s candidate {index}'s prompt is {len(prompt)} tokens; "
                    f"one prompt may be at most {max_prompt_tokens}",
                    {"group": group, "candidate": index, "tokens": len(prompt),
                     "max_tokens": max_prompt_tokens},
                )
        boundaries.append(common)
    rows = [(group, list(prompt)) for group, prompts in enumerate(candidates) for prompt in prompts]
    shared = min(common_prefix([prompt for _, prompt in rows]), min(boundaries) - 1)
    suffixes: list[list[int]] = []
    scored_rows: list[Scored] = []
    for index, (group, prompt) in enumerate(rows):
        suffix = prompt[shared:]
        if len(suffix) > max_item_tokens:
            raise ItemsRefusal(
                400,
                ITEM_TOO_LONG,
                f"row {index} (group {group}) is {len(suffix)} tokens past the shared "
                f"state; one row may be at most {max_item_tokens}",
                {"group": group, "item": index, "tokens": len(suffix),
                 "max_tokens": max_item_tokens},
            )
        suffixes.append(suffix)
        boundary = boundaries[group]
        scored_rows.append(
            Scored(group=group, start=boundary - 1 - shared, targets=prompt[boundary:])
        )
    return LikelihoodSplit(
        shared=rows[0][1][:shared],
        suffixes=suffixes,
        scored=scored_rows,
        context_tokens=[len(context) for context in contexts],
        boundaries=boundaries,
    )


def likelihood_prompts(
    ask: LikelihoodAsk,
    tokenize_context: Callable[[list[dict[str, Any]]], Sequence[int]],
    tokenize_reply: Callable[[list[dict[str, Any]]], Sequence[int]],
) -> LikelihoodSplit:
    """The chat form's split: each question's context and each candidate's prompt
    rendered by the chat template."""
    assert ask.messages is not None, "the prompt form is split by prompt_likelihood_split"
    messages = ask.messages
    contexts = [tokenize_context(item_messages(messages, group.question)) for group in ask.groups]
    candidates = [
        [tokenize_reply(reply_messages(messages, group.question, text)) for text in group.texts]
        for group in ask.groups
    ]
    return likelihood_split(
        contexts, candidates, ask.max_prompt_tokens, ask.max_item_tokens,
        ask.max_candidate_tokens,
    )


def prompt_likelihood_split(
    ask: LikelihoodAsk, tokenize: Callable[[str], Sequence[int]]
) -> LikelihoodSplit:
    """The prompt form's split: each question's context is the prompt and the question
    as one text, each candidate's prompt that text and the candidate, every one
    tokenized whole."""
    assert ask.prompt is not None, "the chat form is split by likelihood_prompts"
    contexts = [tokenize(ask.prompt + group.question) for group in ask.groups]
    candidates = [
        [tokenize(ask.prompt + group.question + text) for text in group.texts]
        for group in ask.groups
    ]
    return likelihood_split(
        contexts, candidates, ask.max_prompt_tokens, ask.max_item_tokens,
        ask.max_candidate_tokens,
    )


def token_logprobs(head: Callable[[Any], Any], hidden: Any, targets: Sequence[int]) -> list[float]:
    """ln P(target i | everything before it) for each position of `hidden`
    (one row, [positions, width]), in float32, SCORE_CHUNK positions per head
    application."""
    import mlx.core as mx

    read: list[float] = []
    for start in range(0, len(targets), SCORE_CHUNK):
        stop = min(len(targets), start + SCORE_CHUNK)
        logits = head(hidden[start:stop]).astype(mx.float32)
        picked = mx.take_along_axis(
            logits, mx.array(list(targets[start:stop]))[:, None], axis=-1
        )[:, 0]
        values = picked - mx.logsumexp(logits, axis=-1)
        mx.eval(values)
        read.extend(float(value) for value in values.tolist())
    return read


ScoreFn = Callable[[Any, Sequence[int]], list[float]]

Span = tuple[int, int, Sequence[int]]
"""One candidate in a batched forward's hidden states: its row, the position
that predicts its first token, and its tokens."""

SpansFn = Callable[[Any, Sequence[Span]], list[list[float]]]


def spans_logprobs(
    head: Callable[[Any], Any], hidden: Any, spans: Sequence[Span]
) -> list[list[float]]:
    """`token_logprobs` for every candidate of one batched forward at once: their
    positions gathered into one [positions, width] matrix and scored SCORE_CHUNK
    at a time, so the head's weights (the whole vocabulary, 1.3 GB on a 4B) are
    read once per chunk instead of once per candidate."""
    import mlx.core as mx

    gathered = mx.concatenate(
        [hidden[row, start:start + len(targets)] for row, start, targets in spans], axis=0
    )
    flat = token_logprobs(head, gathered, [token for _, _, targets in spans for token in targets])
    read: list[list[float]] = []
    at = 0
    for _, _, targets in spans:
        read.append(flat[at:at + len(targets)])
        at += len(targets)
    return read


def next_logprobs(head: Callable[[Any], Any], hidden: Any, targets: Sequence[int]) -> list[float]:
    """ln P(target | the context) for many targets at ONE position (`hidden`,
    [1, width]): the head applied once, in float32, every target gathered from
    the same distribution."""
    import mlx.core as mx

    logits = head(hidden).astype(mx.float32)[0]
    values = mx.take(logits, mx.array(list(targets))) - mx.logsumexp(logits)
    mx.eval(values)
    return [float(value) for value in values.tolist()]


def question_tail(split: LikelihoodSplit, member: int, read: int) -> list[int]:
    """What a question's candidates share past the `read` tokens the shared pass
    read: its context up to its boundary (the question, the opened reply), read
    once per question."""
    scored = split.scored[member]
    whole = split.shared + split.suffixes[member]
    assert read <= len(split.shared)
    return whole[read: len(split.shared) + scored.start + 1]


def read_likelihood_rows(
    split: LikelihoodSplit,
    read_by_shared: int,
    shared_pass: Callable[[], list[Any]],
    question_pass: Callable[[list[Any], list[int]], tuple[list[Any], Any]],
    rows_pass: Callable[[list[Any], list[list[int]]], Any],
    score: SpansFn,
    first: ScoreFn,
    per_row_bytes: Callable[[list[Any]], int],
    step: int = CHUNK_TOKENS,
) -> list[list[float]]:
    """Every candidate scored at every one of its tokens over the shared state
    read once (`shared_pass` reads the first `read_by_shared` tokens of
    `split.shared`). Per question: the rest of its context is read once over a
    copy of the state's cache (`question_pass`, which also returns the hidden
    state at its last token), every candidate's FIRST token is read off that
    one position (`first`), and a candidate of k tokens is one row of its first
    k-1 tokens in a batched forward over the question's cache: the row's
    position i predicts token i+1, and its last token is never an input. A
    one-token candidate costs no forward at all. Rows are grouped as
    `read_rows` groups items, bounded by the bytes of the repeated cache, and
    every row of a forward is scored together."""
    import mlx.core as mx

    cache = shared_pass()
    read: list[list[float] | None] = [None] * len(split.suffixes)
    members: dict[int, list[int]] = {}
    for index, scored in enumerate(split.scored):
        members.setdefault(scored.group, []).append(index)
    for indexes in members.values():
        own, last = question_pass(cache, question_tail(split, indexes[0], read_by_shared))
        firsts = first(last, [split.scored[index].targets[0] for index in indexes])
        for index, value in zip(indexes, firsts):
            read[index] = [value]
        longer = [index for index in indexes if len(split.scored[index].targets) > 1]
        if not longer:
            continue
        inputs = [list(split.scored[index].targets[:-1]) for index in longer]
        row_bytes = per_row_bytes(own)
        for batch in row_groups(inputs, lambda longest: rows_per_pass(row_bytes, longest, step)):
            rows, _ = padded([inputs[at] for at in batch])
            hidden = rows_pass(own, rows)
            spans = [(row, 0, split.scored[longer[at]].targets[1:]) for row, at in enumerate(batch)]
            for at, values in zip(batch, score(hidden, spans)):
                head_value = read[longer[at]]
                assert head_value is not None and len(head_value) == 1
                read[longer[at]] = head_value + values
        mx.clear_cache()
    done = [row for row in read if row is not None]
    assert len(done) == len(read)
    return done


def read_likelihood(
    split: LikelihoodSplit,
    shared_pass: Callable[[], list[Any]],
    item_pass: Callable[[list[Any], list[int]], Any],
    score: ScoreFn,
) -> list[list[float]]:
    """The candidates one forward each over a copy of the shared cache: the
    mlx-vlm reader's way, whose item pass places image positions itself."""
    import mlx.core as mx

    cache = shared_pass()
    read: list[list[float]] = []
    for suffix, scored in zip(split.suffixes, split.scored):
        hidden = item_pass(cache, suffix)
        stop = scored.start + len(scored.targets)
        read.append(score(hidden[0, scored.start:stop], scored.targets))
    mx.clear_cache()
    return read


def question_read_tokens(split: LikelihoodSplit, state_end: int, reused: int) -> list[int]:
    """The prompt tokens each question's pass actually read on the per-question reader
    (read_likelihood_rows): its tail from the state's end to its boundary, once per
    question, and the first question also the state past what a held cache supplied.
    A candidate's own tokens are scored, not counted: they are the reply, as on
    llama-server, whose forced tokens are not prompt."""
    read = [boundary - state_end for boundary in split.boundaries]
    if read:
        read[min(scored.group for scored in split.scored)] += state_end - reused
    return read


def row_read_tokens(split: LikelihoodSplit) -> list[int]:
    """The same for the per-row reader (read_likelihood, mlx-vlm): the whole shared part
    once, then every candidate's own row up to its group's boundary."""
    shared = len(split.shared)
    read = [0] * len(split.boundaries)
    for scored in split.scored:
        read[scored.group] += split.boundaries[scored.group] - shared
    if read:
        read[min(scored.group for scored in split.scored)] += shared
    return read


def likelihood_document(
    split: LikelihoodSplit,
    logprobs: Sequence[Sequence[float]],
    read_tokens: Sequence[int],
    cached_tokens: int | None = None,
) -> dict[str, Any]:
    """`read_tokens`, per question, are the prompt tokens this request actually read
    for it (question_read_tokens or row_read_tokens): Crucible reports each question's
    prompt as its boundary once per candidate, as llama-server is sent it, and what
    was not read as cached, so a shared state read once reads as cached for every
    other question (version 6 sent no such count, and a rerank on the Mac reported
    its query once per document and nothing cached, 2026-10-10)."""
    groups: list[dict[str, Any]] = [
        {"context_tokens": context, "boundary": boundary, "read_tokens": read, "candidates": []}
        for context, boundary, read in zip(split.context_tokens, split.boundaries, read_tokens)
    ]
    for index, (scored, row) in enumerate(zip(split.scored, logprobs)):
        if len(row) != len(scored.targets):
            raise ItemsRefusal(
                500, "engine_error",
                f"row {index} read {len(row)} log-probabilities for {len(scored.targets)} tokens",
            )
        if not all(math.isfinite(value) for value in row):
            raise ItemsRefusal(
                500, "engine_error",
                f"row {index} (group {scored.group}) has a token whose log-probability is "
                "not finite; the model gives it no probability at all",
                {"group": scored.group},
            )
        groups[scored.group]["candidates"].append({"logprobs": [float(v) for v in row]})
    document: dict[str, Any] = {
        "object": "crucible.likelihood",
        "shared_tokens": len(split.shared),
        "groups": groups,
    }
    if cached_tokens is not None:
        document["cached_tokens"] = cached_tokens
    return document


def split_shared(
    prompts: Sequence[Sequence[int]], max_prompt_tokens: int, max_item_tokens: int
) -> Split:
    shortest = min(len(prompt) for prompt in prompts)
    common = 0
    while common < shortest - 1 and len({prompt[common] for prompt in prompts}) == 1:
        common += 1
    suffixes = [list(prompt[common:]) for prompt in prompts]
    for index, suffix in enumerate(suffixes):
        if len(suffix) > max_item_tokens:
            raise ItemsRefusal(
                400,
                ITEM_TOO_LONG,
                f"item {index} is {len(suffix)} tokens past the shared state; one item "
                f"may be at most {max_item_tokens}",
                {"item": index, "tokens": len(suffix), "max_tokens": max_item_tokens},
            )
        if common + len(suffix) > max_prompt_tokens:
            raise ItemsRefusal(
                400,
                PROMPT_TOO_LONG,
                f"item {index}'s prompt is {common + len(suffix)} tokens (the shared "
                f"state is {common}); one prompt may be at most {max_prompt_tokens}",
                {"item": index, "tokens": common + len(suffix),
                 "shared_tokens": common, "max_tokens": max_prompt_tokens},
            )
    return Split(shared=list(prompts[0][:common]), suffixes=suffixes)


def top_of(head: Callable[[Any], Any], hidden_rows: Any, k: int) -> list[list[tuple[int, float]]]:
    import mlx.core as mx

    logits = head(hidden_rows).astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    order = mx.argsort(-logprobs, axis=-1)[:, :k]
    values = mx.take_along_axis(logprobs, order, axis=-1)
    mx.eval(order, values)
    return [list(zip(tokens, logps)) for tokens, logps in zip(order.tolist(), values.tolist())]


def copied(fresh: list[Any], shared: list[Any]) -> list[Any]:
    import mlx.core as mx

    for mine, theirs in zip(fresh, shared):
        mine.state = [mx.array(array) for array in theirs.state]
    return fresh


def read_items(
    split: Split,
    shared_pass: Callable[[], list[Any]],
    item_pass: Callable[[list[Any], list[int]], Any],
    head: Callable[[Any], Any],
    k: int,
) -> list[list[tuple[int, float]]]:
    import mlx.core as mx

    cache = shared_pass()
    tops: list[list[tuple[int, float]]] = []
    for suffix in split.suffixes:
        hidden = item_pass(cache, suffix)
        tops.extend(top_of(head, hidden[0, -1:, :], k))
    mx.clear_cache()
    return tops


def common_prefix(prompts: Sequence[Sequence[int]]) -> int:
    shortest = min(len(prompt) for prompt in prompts)
    common = 0
    while common < shortest and len({prompt[common] for prompt in prompts}) == 1:
        common += 1
    return common


def rows_per_pass(per_row_bytes: int, longest: int, step: int = CHUNK_TOKENS) -> int:
    by_bytes = ROW_BYTES // max(1, per_row_bytes)
    by_tokens = step // max(1, longest)
    return max(1, min(MAX_ROWS, by_bytes, by_tokens))


def row_groups(
    suffixes: Sequence[Sequence[int]], rows: Callable[[int], int]
) -> list[list[int]]:
    """Item indexes in the groups one forward reads, longest first so a group
    pads little; `rows(longest)` is how many rows a group that long may hold."""
    order = sorted(range(len(suffixes)), key=lambda index: -len(suffixes[index]))
    groups: list[list[int]] = []
    at = 0
    while at < len(order):
        size = rows(len(suffixes[order[at]]))
        groups.append(order[at:at + size])
        at += size
    return groups


def padded(suffixes: Sequence[Sequence[int]]) -> tuple[list[list[int]], list[int]]:
    """Right-padded rows and each row's last real position. The model is causal,
    so a real position never reads the padding after it."""
    longest = max(len(suffix) for suffix in suffixes)
    rows = [list(suffix) + [suffix[-1]] * (longest - len(suffix)) for suffix in suffixes]
    return rows, [len(suffix) - 1 for suffix in suffixes]


def read_rows(
    split: Split,
    shared_pass: Callable[[], list[Any]],
    rows_pass: Callable[[list[Any], list[list[int]], list[int]], Any],
    head: Callable[[Any], Any],
    k: int,
    per_row_bytes: Callable[[list[Any]], int],
    step: int = CHUNK_TOKENS,
) -> list[list[tuple[int, float]]]:
    """Every item read as one row of a batched forward over the shared state's
    cache: one forward per group of items, not one per item. On Apple silicon a
    forward of 2 to 64 tokens costs the same ~115 ms on a 9B (MLX's bf16 matmul
    leaves its matrix-vector kernel past one row), so items in one forward are
    nearly free next to items in turn."""
    import mlx.core as mx

    cache = shared_pass()
    row_bytes = per_row_bytes(cache)
    tops: list[list[tuple[int, float]] | None] = [None] * len(split.suffixes)
    for group in row_groups(split.suffixes, lambda longest: rows_per_pass(row_bytes, longest, step)):
        rows, lasts = padded([split.suffixes[index] for index in group])
        for index, top in zip(group, top_of(head, rows_pass(cache, rows, lasts), k)):
            tops[index] = top
    mx.clear_cache()
    read = [top for top in tops if top is not None]
    assert len(read) == len(tops)
    return read


@dataclass
class HeldState:
    model: Any
    tokens: tuple[int, ...]
    cache: list[Any]
    nbytes: int


class StateCache:
    """The caches of the shared states the engine read last, so the next
    decision about the same state continues from it instead of reading it
    again. A held cache is only ever reused as a PREFIX: a recurrent layer's
    state cannot be trimmed back, so it is used when its tokens open the new
    state and never otherwise. It lives and dies with the engine process."""

    def __init__(self, entries: int, max_bytes: int) -> None:
        self._entries = entries
        self._max_bytes = max_bytes
        self._held: list[HeldState] = []

    def __len__(self) -> int:
        return len(self._held)

    @property
    def nbytes(self) -> int:
        return sum(held.nbytes for held in self._held)

    def nearest(self, model: Any, tokens: Sequence[int]) -> HeldState | None:
        best: HeldState | None = None
        for held in self._held:
            n = len(held.tokens)
            if held.model is not model or n > len(tokens) or tuple(tokens[:n]) != held.tokens:
                continue
            if best is None or n > len(best.tokens):
                best = held
        if best is not None:
            self._held.remove(best)
            self._held.append(best)
        return best

    def keep(self, model: Any, tokens: Sequence[int], cache: list[Any], nbytes: int) -> None:
        key = tuple(tokens)
        self._held = [
            held for held in self._held if held.model is model and held.tokens != key
        ]
        if not key or nbytes > self._max_bytes:
            return
        self._held.append(HeldState(model=model, tokens=key, cache=cache, nbytes=nbytes))
        while len(self._held) > self._entries or self.nbytes > self._max_bytes:
            self._held.pop(0)


STATES = StateCache(STATE_ENTRIES, STATE_BYTES)


def text_parts(model: Any) -> tuple[Any, Callable[[Any], Any]]:
    language = getattr(model, "language_model", model)
    inner = language.model
    head = getattr(language, "lm_head", None)
    if head is None:
        head = inner.embed_tokens.as_linear
    return inner, head


def items_document(
    split: Split,
    tops: Sequence[Sequence[tuple[int, float]]],
    decode: Callable[[int], str],
    cached_tokens: int | None = None,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "object": "crucible.items",
        "shared_tokens": len(split.shared),
        "item_tokens": [len(suffix) for suffix in split.suffixes],
        "slots": [
            {
                "top_logprobs": [
                    {"token": decode(token), "logprob": float(logprob)}
                    for token, logprob in row
                    if math.isfinite(logprob)
                ],
            }
            for row in tops
        ],
    }
    if cached_tokens is not None:
        document["cached_tokens"] = cached_tokens
    return document


def answer_items(
    ask: ItemsAsk,
    tokenize: Callable[[list[dict[str, Any]]], Sequence[int]],
    read: Callable[[Split], Sequence[Sequence[tuple[int, float]]]],
    decode: Callable[[int], str],
) -> dict[str, Any]:
    prompts = [tokenize(item_messages(ask.messages, question)) for question in ask.questions]
    split = split_shared(prompts, ask.max_prompt_tokens, ask.max_item_tokens)
    return items_document(split, read(split), decode)


def refusal_document(refusal: ItemsRefusal) -> dict[str, Any]:
    return {
        "error": {
            "code": refusal.code,
            "message": str(refusal),
            "type": "invalid_request_error",
            "details": refusal.details,
        }
    }


@dataclass
class MlxLmItemsJob:
    ask: ItemsAsk | LikelihoodAsk | EmbedAsk


def state_bytes(cache: list[Any]) -> int:
    return sum(array.nbytes for entry in cache for array in entry.state if array is not None)


class MlxLmShared:
    """The shared part of one items or likelihood pass on mlx-lm: the state
    taken from a held cache where one opens it, read and kept where none does,
    then whatever else every row shares. No forward reads more than `step`
    tokens (rows x positions): the engine's --prefill-step-size."""

    def __init__(
        self, model: Any, inner: Any, shared: list[int], state_end: int, step: int
    ) -> None:
        self.model = model
        self.inner = inner
        self.shared = shared
        self.state_end = state_end
        self.step = step
        self.reused = 0

    def _prefill(self, cache: list[Any], tokens: list[int]) -> None:
        import mlx.core as mx

        ids = mx.array(tokens)
        for start in range(0, len(tokens), self.step):
            self.inner(ids[None, start:start + self.step], cache=cache)
            mx.eval([entry.state for entry in cache])

    def _rows_forward(self, own: list[Any], rows: list[list[int]]) -> Any:
        """Equal-length rows read over `own` in pieces of at most `step` tokens
        (rows x positions), each evaluated before the next is sent, and their
        hidden states joined. The model is causal and each piece continues the
        cache the one before it filled, so the result is the one forward's."""
        import mlx.core as mx

        ids = mx.array(rows)
        width = max(1, self.step // len(rows))
        if len(rows[0]) <= width:
            return self.inner(ids, cache=own)
        pieces = []
        for start in range(0, len(rows[0]), width):
            hidden = self.inner(ids[:, start:start + width], cache=own)
            mx.eval(hidden, [entry.state for entry in own])
            pieces.append(hidden)
        return mx.concatenate(pieces, axis=1)

    def shared_pass(self) -> list[Any]:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        model, shared, state_end = self.model, self.shared, self.state_end
        cache = make_prompt_cache(model)
        held = STATES.nearest(model, shared[:state_end])
        if held is not None:
            copied(cache, held.cache)
            self.reused = len(held.tokens)
        if state_end > self.reused:
            self._prefill(cache, shared[self.reused:state_end])
            kept = copied(make_prompt_cache(model), cache)
            mx.eval([entry.state for entry in kept])
            STATES.keep(model, shared[:state_end], kept, state_bytes(kept))
        if len(shared) > state_end:
            self._prefill(cache, shared[state_end:])
        return cache

    def rows_hidden(self, cache: list[Any], rows: list[list[int]]) -> Any:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        own = make_prompt_cache(self.model)
        if self.shared:
            for mine, theirs in zip(own, cache):
                mine.state = [mx.repeat(array, len(rows), axis=0) for array in theirs.state]
        return self._rows_forward(own, rows)

    def question_pass(self, cache: list[Any], tail: list[int]) -> tuple[list[Any], Any]:
        """A question's context tail read over a copy of the state's cache: that
        copy, and the hidden state at the tail's last token ([1, width])."""
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        own = make_prompt_cache(self.model)
        if self.shared:
            copied(own, cache)
        ids = mx.array(tail)
        hidden: Any = None
        for start in range(0, len(tail), self.step):
            hidden = self.inner(ids[None, start:start + self.step], cache=own)
            mx.eval([entry.state for entry in own])
        steps = len(tail) - 1 - (len(tail) - 1) // self.step * self.step
        return own, hidden[0, steps:steps + 1]

    def question_rows(self, cache: list[Any], rows: list[list[int]]) -> Any:
        """Rows over a question's cache, which always holds its tail."""
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        own = make_prompt_cache(self.model)
        for mine, theirs in zip(own, cache):
            mine.state = [mx.repeat(array, len(rows), axis=0) for array in theirs.state]
        return self._rows_forward(own, rows)

    def per_row_bytes(self, cache: list[Any]) -> int:
        return state_bytes(cache) if self.shared else 0


def mlx_lm_answer(provider: Any, job: MlxLmItemsJob) -> dict[str, Any]:
    import mlx.core as mx

    model, tokenizer = provider.load("default_model", None, "default_model")
    inner, head = text_parts(model)
    ask = job.ask
    # The engine's own prefill step, which Crucible derives from the model's size:
    # an items pass holds the GPU no longer per evaluation than a chat's prompt.
    step = provider.cli_args.prefill_step_size

    if isinstance(ask, EmbedAsk):
        return mlx_lm_embed(model, inner, tokenizer, ask, step)

    def tokenize(messages: list[dict[str, Any]], reply: bool = False) -> list[int]:
        return list(tokenizer.apply_chat_template(
            messages, add_generation_prompt=not reply, tokenize=True,
            **({"continue_final_message": True} if reply else {}), **ask.template_kwargs,
        ))

    def encode(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False))

    if isinstance(ask, LikelihoodAsk):
        if ask.prompt is None:
            assert ask.messages is not None
            likely = likelihood_prompts(ask, tokenize, lambda messages: tokenize(messages, True))
            open_turn = tokenize(item_messages(ask.messages, ""))
        else:
            likely = prompt_likelihood_split(ask, encode)
            open_turn = encode(ask.prompt)
        # The state is kept where it ends, as for items: what every row shares
        # with the open turn left empty (in the prompt form, the prompt). Only the
        # state is read as shared: what follows it up to each question's boundary
        # is that question's own pass, which also yields the hidden state its
        # candidates' first tokens are read from.
        state_end = min(len(likely.shared), common_prefix([*likely_rows(likely), open_turn]))
        held = MlxLmShared(model, inner, likely.shared[:state_end], state_end, step)
        logprobs = read_likelihood_rows(
            likely, state_end, held.shared_pass, held.question_pass, held.question_rows,
            lambda hidden, spans: spans_logprobs(head, hidden, spans),
            lambda hidden, targets: next_logprobs(head, hidden, targets), state_bytes, step,
        )
        return likelihood_document(
            likely, logprobs, question_read_tokens(likely, state_end, held.reused), held.reused
        )

    open_turn = tokenize(item_messages(ask.messages, ""))
    prompts = [tokenize(item_messages(ask.messages, question)) for question in ask.questions]
    split = split_shared(prompts, ask.max_prompt_tokens, ask.max_item_tokens)
    # Where the state ends: what every item shares with the open turn left
    # empty. The state's cache is kept at that point, so a later decision about
    # the same state, with other questions, continues from it.
    state_end = min(len(split.shared), common_prefix([*prompts, open_turn]))
    if len(split.suffixes) == 1 and state_end < len(split.shared):
        # One item: what lies past the state is read in its row's forward. A
        # forward of its own would cost a whole forward (~115 ms on a 9B at any
        # length up to 64) for nothing another row could share.
        split = Split(
            shared=split.shared[:state_end],
            suffixes=[split.shared[state_end:] + split.suffixes[0]],
        )
    held = MlxLmShared(model, inner, split.shared, state_end, step)

    def rows_pass(cache: list[Any], rows: list[list[int]], lasts: list[int]) -> Any:
        hidden = held.rows_hidden(cache, rows)
        return hidden[mx.arange(len(rows)), mx.array(lasts)]

    tops = read_rows(
        split, held.shared_pass, rows_pass, head, ask.top_logprobs, held.per_row_bytes, step
    )
    return items_document(
        split, tops, lambda token: tokenizer.convert_ids_to_tokens([token])[0], held.reused
    )


def embed_ids(
    tokenizer: Any, ask: EmbedAsk
) -> list[list[int]]:
    """Each input's tokens as the model reads it, every refusal made before a forward."""
    ids = [list(tokenizer.encode(text, add_special_tokens=False)) for text in ask.inputs]
    for index, row in enumerate(ids):
        if len(row) > ask.max_input_tokens:
            raise ItemsRefusal(
                400,
                EMBED_INPUT_TOO_LONG,
                f"input {index} is {len(row)} tokens; one input may be at most "
                f"{ask.max_input_tokens}",
                {"input": index, "tokens": len(row), "max_tokens": ask.max_input_tokens},
            )
    return ids


def embed_document(ids: Sequence[Sequence[int]], vectors: Sequence[Sequence[float]]) -> dict[str, Any]:
    widths = {len(vector) for vector in vectors}
    if len(vectors) != len(ids) or len(widths) != 1:
        raise ItemsRefusal(
            500, "engine_error",
            f"{len(vectors)} vectors of widths {sorted(widths)} for {len(ids)} inputs",
        )
    for index, vector in enumerate(vectors):
        if not all(math.isfinite(value) for value in vector):
            raise ItemsRefusal(
                500, "engine_error", f"input {index}'s vector has a value that is not finite",
                {"input": index},
            )
    return {
        "object": "crucible.embeddings",
        "dimensions": widths.pop(),
        "data": [
            {"embedding": [float(v) for v in vector], "tokens": len(row)}
            for row, vector in zip(ids, vectors)
        ],
    }


def mlx_lm_embed(model: Any, inner: Any, tokenizer: Any, ask: EmbedAsk, step: int) -> dict[str, Any]:
    """Every input's vector: the inner model's last hidden state (after its final norm,
    what transformers' AutoModel calls last_hidden_state) at the input's last token,
    unnormalised. Inputs are rows of one forward where they fit the step (right-padded:
    the model is causal, so a real position never reads the padding after it), and a
    long one is read in pieces of the step over its own cache."""
    import mlx.core as mx

    ids = embed_ids(tokenizer, ask)
    reader = MlxLmShared(model, inner, [], 0, step)
    vectors: list[list[float] | None] = [None] * len(ids)
    for group in row_groups(ids, lambda longest: rows_per_pass(0, longest, step)):
        rows, lasts = padded([ids[index] for index in group])
        hidden = reader.rows_hidden([], rows)
        picked = hidden[mx.arange(len(rows)), mx.array(lasts)].astype(mx.float32)
        mx.eval(picked)
        for index, vector in zip(group, picked.tolist()):
            vectors[index] = vector
    mx.clear_cache()
    done = [vector for vector in vectors if vector is not None]
    assert len(done) == len(ids)
    return embed_document(ids, done)


def likely_rows(split: LikelihoodSplit) -> list[list[int]]:
    return [split.shared + suffix for suffix in split.suffixes]


def mlx_lm_run_job(provider: Any, job: MlxLmItemsJob, rqueue: Any) -> None:
    try:
        rqueue.put(mlx_lm_answer(provider, job))
    except ItemsRefusal as refusal:
        rqueue.put(refusal)
    except Exception as exc:
        rqueue.put(ItemsRefusal(500, "engine_error", f"the items pass failed: {exc!r}"))


def _write(handler: Any, status: int, document: dict[str, Any]) -> None:
    payload = json.dumps(document).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def mlx_lm_names(model: str) -> tuple[str, ...]:
    import os

    return (model, os.path.realpath(model))


def mlx_lm_serve_http(handler: Any) -> None:
    from queue import Queue

    served = mlx_lm_names(handler.response_generator.cli_args.model)
    try:
        length = int(handler.headers.get("Content-Length") or 0)
        ask = parse_request(json.loads(handler.rfile.read(length).decode("utf-8")), served)
    except (ValueError, UnicodeDecodeError) as exc:
        refusal = ItemsRefusal(400, "bad_json", f"the body is not JSON: {exc}")
        _write(handler, 400, refusal_document(refusal))
        return
    except ItemsRefusal as refusal:
        _write(handler, refusal.status, refusal_document(refusal))
        return
    rqueue: Queue = Queue()
    handler.response_generator.requests.put((rqueue, MlxLmItemsJob(ask), None))
    result = rqueue.get()
    if isinstance(result, ItemsRefusal):
        _write(handler, result.status, refusal_document(result))
        return
    _write(handler, 200, result)
