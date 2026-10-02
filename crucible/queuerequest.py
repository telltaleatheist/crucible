"""The `queue` member of a request that can wait: a job, a chat, a decision, a TTS stream.

Waiting is the default. A request without `queue` that finds the server busy takes a
place in the line (crucible/jobs/line.py) and waits up to ``DEFAULT_MAX_WAIT_S``
(a job that is an item of the open queue session: up to ``MAX_MAX_WAIT_S``).
``{"max_wait_s": N}`` changes the wait. ``false`` opts out: the request is refused at
once (`409 server_busy`, `session_open`, `model_not_resident`, `503 chat_queue_full`)
instead of waiting. Nothing else is a `queue`: `{}`, `true` and `null` are refused by
name, so no request means "wait" by one spelling and "refuse" by another.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from .errors import ApiError
from .jobs.line import DEFAULT_MAX_WAIT_S, MAX_MAX_WAIT_S, MIN_MAX_WAIT_S

QUEUE_SHAPE = (
    f'"queue" is false (refuse at once when the server is busy) or {{"max_wait_s": N}} '
    f"(N from {MIN_MAX_WAIT_S} to {MAX_MAX_WAIT_S} seconds); leave it out to wait in "
    "the line, up to an hour"
)


class QueueRequest(BaseModel):
    """How long this request may wait in the server's line."""

    model_config = ConfigDict(extra="forbid")

    max_wait_s: int = Field(
        ge=MIN_MAX_WAIT_S,
        le=MAX_MAX_WAIT_S,
        description="How long the request may wait for its turn before it is removed "
        "`expired`.",
    )


def _queue_shape(value: Any) -> Any:
    if value is False:
        return value
    if isinstance(value, dict) and "max_wait_s" in value:
        return value
    if value == {}:
        raise ValueError(
            f"{QUEUE_SHAPE}. {{}} is not one: waiting is the default, so leave the "
            "member out"
        )
    raise ValueError(f"{QUEUE_SHAPE}, not {json.dumps(value, default=repr)}")


QueueChoice = Annotated[
    Union[QueueRequest, Literal[False], None], BeforeValidator(_queue_shape)
]
"""A request model's `queue` field: `{"max_wait_s": N}` or `false`. Declare it with
``queue_field()``: the default (None) is never validated, so None means the member was
left out (wait), while an explicit `null` meets the validator and is refused."""


def _no_null(schema: dict[str, Any]) -> None:
    """The schema says what may be sent: an explicit `null` is refused, so it is not
    offered (the member is optional because leaving it out is the default)."""
    shapes = schema.get("anyOf")
    if shapes is not None:
        schema["anyOf"] = [shape for shape in shapes if shape.get("type") != "null"]


def queue_field(description: str | None = None) -> Any:
    """A request model's `queue` field: optional, None when left out."""
    return Field(default=None, description=description, json_schema_extra=_no_null)


def max_wait_of(
    choice: QueueRequest | Literal[False] | None, default: int = DEFAULT_MAX_WAIT_S
) -> int | None:
    """Seconds this request may wait in the line: ``default`` when it sent no `queue`,
    its own `max_wait_s`, or None for `false` (refuse at once rather than wait)."""
    if choice is None:
        return default
    if choice is False:
        return None
    return choice.max_wait_s


def queue_of(body: dict[str, Any]) -> int | None:
    """Take the `queue` member out of a raw request body (a chat is forwarded to its
    engine as sent, less this member) and answer ``max_wait_of`` it."""
    if "queue" not in body:
        return max_wait_of(None)
    value = body.pop("queue")
    try:
        _queue_shape(value)
    except ValueError as exc:
        raise ApiError(400, "invalid_request", str(exc)) from None
    if value is False:
        return None
    try:
        return QueueRequest.model_validate(value).max_wait_s
    except ValidationError as exc:
        problem = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'queue'}: {error['msg']}"
            for error in exc.errors()
        )
        raise ApiError(400, "invalid_request", f"{QUEUE_SHAPE}: {problem}") from None


__all__ = [
    "DEFAULT_MAX_WAIT_S", "MAX_MAX_WAIT_S", "QUEUE_SHAPE", "QueueChoice", "QueueRequest",
    "max_wait_of", "queue_field", "queue_of",
]
