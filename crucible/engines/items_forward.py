from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

ITEMS_VERSION = 1

ITEMS_PATH = "/v1/crucible/items"

MAX_TOP_LOGPROBS = 40

CHUNK_TOKENS = 2048

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
) -> dict[str, Any]:
    return {
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


def mlx_lm_answer(provider: Any, job: MlxLmItemsJob) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer = provider.load("default_model", None, "default_model")
    inner, head = text_parts(model)

    def tokenize(messages: list[dict[str, Any]]) -> list[int]:
        return list(tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, **job.ask.template_kwargs
        ))

    def read(split: Split) -> list[list[tuple[int, float]]]:
        def shared_pass() -> list[Any]:
            cache = make_prompt_cache(model)
            ids = mx.array(split.shared)
            for start in range(0, len(split.shared), CHUNK_TOKENS):
                inner(ids[None, start:start + CHUNK_TOKENS], cache=cache)
                mx.eval([entry.state for entry in cache])
            return cache

        def item_pass(cache: list[Any], suffix: list[int]) -> Any:
            own = copied(make_prompt_cache(model), cache)
            return inner(mx.array(suffix)[None], cache=own)

        return read_items(split, shared_pass, item_pass, head, job.ask.top_logprobs)

    return answer_items(
        job.ask, tokenize, read, lambda token: tokenizer.convert_ids_to_tokens([token])[0]
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
