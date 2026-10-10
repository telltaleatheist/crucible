"""What a chat asks to constrain its answer with, and whether the engine serving it
enforces that: a grammar is kept, or refused by name, never passed to be ignored.

A chat body can constrain its answer in several ways, and each engine Crucible runs
reads a different subset of them (docs/internals/engines-and-capability.md,
"Structured output"). An engine that does not read a field drops it without a word:
stock mlx-lm 0.31.3 read no `response_format` at all, so B-Sides' strict schemas and
BookForge's pronunciation guide were answered unconstrained on the Mac while the PC
enforced them. Every engine therefore states, with where it was read, which
`response_format` types and which other fields it enforces
(`structured_output_formats`, `structured_output_fields`,
`structured_output_basis`), and the chat door refuses `structured_output_not_served`
for anything the resident engine does not state, before a byte is sent to it.

What the door does not judge is the constraint itself (a schema the grammar engine
will not compile, a regex that does not parse): the engine refuses that by name with
its own 400, which the door relays.
"""

from __future__ import annotations

from typing import Any

from .errors import ApiError

RESPONSE_FORMAT = "response_format"

# Every body field other than response_format that constrains the answer with a
# grammar on an engine Crucible runs: vLLM's structured_outputs and its older guided_*
# fields (gone from vLLM 0.29.0's ChatCompletionRequest, which drops what it does not
# know), and llama-server's grammar and json_schema.
GRAMMAR_FIELDS: tuple[str, ...] = (
    "structured_outputs",
    "guided_json",
    "guided_regex",
    "guided_choice",
    "guided_grammar",
    "grammar",
    "json_schema",
)

UNCONSTRAINED_FORMATS: tuple[Any, ...] = (None, "text")


def constraints_of(body: dict[str, Any]) -> list[tuple[str, str | None]]:
    """What `body` constrains its answer with: (field, response_format type or None),
    GRAMMAR_FIELDS in their order, then response_format. A `response_format` of type
    text (or none) is no constraint; one that is not an object is reported with type
    None and left for the engine to refuse."""
    found: list[tuple[str, str | None]] = [
        (field, None) for field in GRAMMAR_FIELDS if body.get(field) is not None
    ]
    response_format = body.get(RESPONSE_FORMAT)
    if response_format is not None:
        kind = response_format.get("type") if isinstance(response_format, dict) else None
        if kind not in UNCONSTRAINED_FORMATS or not isinstance(response_format, dict):
            found.append((RESPONSE_FORMAT, kind if isinstance(kind, str) else None))
    return found


def constrained_fields(body: dict[str, Any]) -> list[str]:
    return [field for field, _ in constraints_of(body)]


def refuse_unenforced_constraint(
    *,
    engine: str,
    model_id: str,
    formats: frozenset[str],
    fields: frozenset[str],
    basis: str,
    body: dict[str, Any],
) -> None:
    """Refuse a constraint the engine serving `model_id` does not enforce."""
    unserved: list[str] = []
    for field, kind in constraints_of(body):
        if field == RESPONSE_FORMAT:
            if kind is None or kind in formats:
                continue
            unserved.append(f"{RESPONSE_FORMAT} type {kind!r}")
        elif field not in fields:
            unserved.append(field)
    if not unserved:
        return
    enforced = [f"{RESPONSE_FORMAT} {kind}" for kind in sorted(formats)] + sorted(fields)
    raise ApiError(
        400,
        "structured_output_not_served",
        f"the {engine} engine serving {model_id!r} does not enforce "
        f"{', '.join(unserved)}, and an engine that does not read a constraint "
        f"answers without it: {basis}. It enforces "
        f"{', '.join(enforced) if enforced else 'no structured output'}. Nothing was "
        "sent to it",
        {
            "model": model_id,
            "engine": engine,
            "fields": unserved,
            "enforced": enforced,
        },
    )


__all__ = [
    "GRAMMAR_FIELDS",
    "RESPONSE_FORMAT",
    "constrained_fields",
    "constraints_of",
    "refuse_unenforced_constraint",
]
