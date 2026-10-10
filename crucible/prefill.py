"""The start of the answer, written by the client: a chat's `prefill`.

A chat body may carry `"prefill": "<text>"`. The model's answer then begins with
that text and the model writes on from it: Crucible sends the engine the messages
followed by an assistant message holding the prefill, with
`continue_final_message: true` and `add_generation_prompt: false`, which vLLM
0.29.0 and llama-server b10970 both read (engines/vllm.py, engines/llama_server.py
`chat_prefill_basis`). The reply's content is what the model wrote AFTER the
prefill; the client joins the two.

It is a field of its own, not "a trailing assistant message", because the engines
read a trailing assistant message three different ways: llama-server continues it,
vLLM closes it and opens a new answer unless told otherwise, and mlx-lm 0.31.3
always closes it. One field, one meaning, refused by name where it cannot be kept:

- `prefill_not_served`: the engine has no way to continue a message (mlx-lm,
  mlx-vlm), or the model is an upstream's.
- `prefill_with_grammar`: a `response_format` or other grammar constrains the
  answer from its FIRST generated token, on every engine; it does not start
  where the prefill ends, so the two together write a second object over the
  first.
- `prefill_with_thinking`: the prefill is written after the model's thinking
  block, which the template closes empty (Qwen3.5 renders `<think>\\n\\n</think>`
  before a continued message whether thinking is on or off), so a prefilled
  answer never thinks. Thinking must be stated off, by the request or by the
  manifest's `[defaults]`.
- `prefill_conflict`: the body also ends in an assistant message, or states
  `continue_final_message` / `add_generation_prompt` itself.
- `invalid_request`: not a non-empty string, or it begins or ends with
  whitespace. Qwen3.5's template trims an assistant message, so the engine would
  continue from text without it (rendered 2026-10-10: `{"answer": ` became
  `{"answer":`).
"""

from __future__ import annotations

from typing import Any

from .errors import ApiError
from .sampling import TEMPLATE_KWARGS, THINKING_KEY
from .structured import GRAMMAR_FIELDS, constrained_fields

PREFILL_KEY = "prefill"

CONTINUATION_FIELDS: tuple[str, ...] = ("continue_final_message", "add_generation_prompt")



def take_prefill(body: dict[str, Any]) -> str | None:
    """Take `prefill` out of a chat body and refuse what no engine could keep.

    The body is forwarded without the member; `with_prefill` puts the prefill
    back as the engine reads it. What the resident engine and the resolved
    thinking decide is `refuse_unkeepable_prefill`'s, once they are known.
    """
    if PREFILL_KEY not in body:
        return None
    value = body.pop(PREFILL_KEY)
    if not isinstance(value, str) or value == "":
        raise ApiError(
            400,
            "invalid_request",
            f"{PREFILL_KEY} must be a non-empty string, the start of the answer the "
            f"model writes on from; got {type(value).__name__}"
            + (" ''" if value == "" else ""),
        )
    if value != value.strip():
        raise ApiError(
            400,
            "invalid_request",
            f"{PREFILL_KEY} begins or ends with whitespace ({value!r}). The chat "
            "template trims an assistant message, so the model would write on from "
            f"{value.strip()!r} instead; send the prefill without it and let the "
            "model write the space",
        )
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ApiError(
            400,
            "invalid_request",
            f"a chat with a {PREFILL_KEY} needs a non-empty messages list for the "
            "prefill to answer",
        )
    last = messages[-1]
    if isinstance(last, dict) and last.get("role") == "assistant":
        raise ApiError(
            400,
            "prefill_conflict",
            f"this chat sends a {PREFILL_KEY} AND ends in an assistant message. The "
            f"start of the answer goes in {PREFILL_KEY}, or nowhere: engines read a "
            "final assistant message differently (llama-server continues it, vLLM "
            "and mlx-lm answer after it)",
        )
    stated = [key for key in CONTINUATION_FIELDS if key in body]
    if stated:
        raise ApiError(
            400,
            "prefill_conflict",
            f"this chat sends a {PREFILL_KEY} and states {', '.join(stated)} itself. "
            f"With a {PREFILL_KEY} the server sets both (continue_final_message "
            "true, add_generation_prompt false); drop them",
            {"fields": stated},
        )
    grammar = constrained_fields(body)
    if grammar:
        raise ApiError(
            400,
            "prefill_with_grammar",
            f"this chat sends a {PREFILL_KEY} and constrains its answer with "
            f"{', '.join(grammar)}. The grammar starts at the first token the model "
            "generates, not where the prefill ends, so the answer would be a whole "
            "new document written after the prefill. Send one: the prefill, or the "
            "grammar",
            {"fields": grammar},
        )
    return value


def refuse_unkeepable_prefill(
    *,
    engine: str,
    model_id: str,
    served: bool,
    basis: str,
    resolved_body: dict[str, Any],
) -> None:
    """What the engine and the thinking the request resolved to decide.

    `resolved_body` is the body after the manifest's `[defaults]` were applied
    (sampling.apply_defaults), so a manifest's `thinking = false` counts.
    """
    if not served:
        raise ApiError(
            400,
            "prefill_not_served",
            f"the {engine} engine serving {model_id!r} cannot continue a prefilled "
            f"answer: {basis}. Nothing was sent to it",
            {"model": model_id, "engine": engine},
        )
    kwargs = resolved_body.get(TEMPLATE_KWARGS)
    thinking = kwargs.get(THINKING_KEY) if isinstance(kwargs, dict) else None
    if thinking is not False:
        stated = "unstated" if thinking is None else repr(thinking)
        raise ApiError(
            400,
            "prefill_with_thinking",
            f"this chat sends a {PREFILL_KEY} with thinking {stated} on {model_id!r}. "
            "The prefill is written after the model's thinking block, which the "
            "chat template closes empty, so a prefilled answer never thinks. State "
            f'thinking off ("{TEMPLATE_KWARGS}": {{"{THINKING_KEY}": false}}), or '
            "drop the prefill",
            {"model": model_id, "thinking": thinking},
        )


def refuse_upstream_prefill(model: str) -> None:
    raise ApiError(
        400,
        "prefill_not_served",
        f"{model!r} is an upstream's model, and Crucible continues a prefilled "
        "answer only on an engine it runs. Nothing was sent",
        {"model": model},
    )


def with_prefill(body: dict[str, Any], prefill: str) -> dict[str, Any]:
    """The body the engine is sent: the prefill as an open assistant message."""
    return {
        **body,
        "messages": [*body["messages"], {"role": "assistant", "content": prefill}],
        "continue_final_message": True,
        "add_generation_prompt": False,
    }


__all__ = [
    "CONTINUATION_FIELDS",
    "GRAMMAR_FIELDS",
    "PREFILL_KEY",
    "refuse_unkeepable_prefill",
    "refuse_upstream_prefill",
    "take_prefill",
    "with_prefill",
]
