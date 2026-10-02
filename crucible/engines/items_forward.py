from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

ITEMS_VERSION = 2

ITEMS_PATH = "/v1/crucible/items"

MAX_TOP_LOGPROBS = 40

CHUNK_TOKENS = 2048

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

TEMPLATE_KWARGS = frozenset({"enable_thinking"})

ITEM_TOO_LONG = "item_too_long"

PROMPT_TOO_LONG = "item_prompt_too_long"


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


def item_messages(messages: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    last = messages[-1]
    content = last["content"]
    if isinstance(content, list):
        filled: Any = [*content, {"type": "text", "text": question}]
    else:
        filled = question
    return [*messages[:-1], {**last, "content": filled}]


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


def rows_per_pass(per_row_bytes: int, longest: int) -> int:
    by_bytes = ROW_BYTES // max(1, per_row_bytes)
    by_tokens = CHUNK_TOKENS // max(1, longest)
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
    for group in row_groups(split.suffixes, lambda longest: rows_per_pass(row_bytes, longest)):
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
    ask: ItemsAsk


def state_bytes(cache: list[Any]) -> int:
    return sum(array.nbytes for entry in cache for array in entry.state if array is not None)


def mlx_lm_answer(provider: Any, job: MlxLmItemsJob) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer = provider.load("default_model", None, "default_model")
    inner, head = text_parts(model)

    def tokenize(messages: list[dict[str, Any]]) -> list[int]:
        return list(tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, **job.ask.template_kwargs
        ))

    ask = job.ask
    prompts = [tokenize(item_messages(ask.messages, question)) for question in ask.questions]
    split = split_shared(prompts, ask.max_prompt_tokens, ask.max_item_tokens)
    # Where the state ends: what every item shares with the open turn left
    # empty. The state's cache is kept at that point, so a later decision about
    # the same state, with other questions, continues from it.
    state_end = min(
        len(split.shared), common_prefix([*prompts, tokenize(item_messages(ask.messages, ""))])
    )
    if len(split.suffixes) == 1 and state_end < len(split.shared):
        # One item: what lies past the state is read in its row's forward. A
        # forward of its own would cost a whole forward (~115 ms on a 9B at any
        # length up to 64) for nothing another row could share.
        split = Split(
            shared=split.shared[:state_end],
            suffixes=[split.shared[state_end:] + split.suffixes[0]],
        )
    reused = 0

    def prefill(cache: list[Any], tokens: list[int]) -> None:
        ids = mx.array(tokens)
        for start in range(0, len(tokens), CHUNK_TOKENS):
            inner(ids[None, start:start + CHUNK_TOKENS], cache=cache)
            mx.eval([entry.state for entry in cache])

    def shared_pass() -> list[Any]:
        nonlocal reused
        cache = make_prompt_cache(model)
        held = STATES.nearest(model, split.shared[:state_end])
        if held is not None:
            copied(cache, held.cache)
            reused = len(held.tokens)
        if state_end > reused:
            prefill(cache, split.shared[reused:state_end])
            kept = copied(make_prompt_cache(model), cache)
            mx.eval([entry.state for entry in kept])
            STATES.keep(model, split.shared[:state_end], kept, state_bytes(kept))
        if len(split.shared) > state_end:
            prefill(cache, split.shared[state_end:])
        return cache

    def rows_pass(cache: list[Any], rows: list[list[int]], lasts: list[int]) -> Any:
        own = make_prompt_cache(model)
        if split.shared:
            for mine, theirs in zip(own, cache):
                mine.state = [mx.repeat(array, len(rows), axis=0) for array in theirs.state]
        hidden = inner(mx.array(rows), cache=own)
        return hidden[mx.arange(len(rows)), mx.array(lasts)]

    tops = read_rows(
        split, shared_pass, rows_pass, head, ask.top_logprobs,
        lambda cache: state_bytes(cache) if split.shared else 0,
    )
    return items_document(
        split, tops, lambda token: tokenizer.convert_ids_to_tokens([token])[0], reused
    )


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
        ask = parse_items(json.loads(handler.rfile.read(length).decode("utf-8")), served)
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
