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

**Whitespace between JSON tokens.** Every engine compiles a JSON schema with flexible
whitespace (a space, or newlines and indentation, wherever JSON allows it), as OpenAI
and vLLM do, and that stays the default (Owen, 2026-10-10). A client whose model is
trained on compact JSON states `"json_whitespace": "compact"`: no whitespace between
tokens, whitespace only inside strings. It is Crucible's member, taken off the body
before the body is forwarded, and it is valid only beside a JSON constraint
(`response_format` json_schema / json_object, `structured_outputs` json / json_object).
Each engine states whether it keeps it (`json_whitespace_compact`,
`json_whitespace_basis`); one that does not is refused `json_whitespace_not_served`
before the chat waits or loads anything. On the engines that keep it (vLLM and mlx-lm,
both of which compile JSON with llguidance) the door writes it into the schema as
llguidance's own option, `"x-guidance": {"whitespace_flexible": false}`, which
llguidance takes over the `whitespace_flexible: true` default the engine passes; a
json_object goes as the schema `{"type": "object"}`, which is what both engines compile
a json_object to.
"""

from __future__ import annotations

import copy
import json
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

# structured_outputs members that are options, not constraints, and that no engine
# Crucible runs reads per request: vLLM 0.29.0's StructuredOutputsParams declares them
# (vllm/sampling_params.py L96-98) but its guidance backend reads only the server-wide
# --structured-outputs-config ones (v1/structured_output/backend_guidance.py L91-95);
# mlx-lm's patch refuses them; llama-server reads no structured_outputs. Set to anything
# but their default they are refused, never dropped: whitespace is `json_whitespace`.
UNREAD_STRUCTURED_OPTIONS: dict[str, Any] = {
    "disable_any_whitespace": False,
    "disable_additional_properties": False,
    "whitespace_pattern": None,
}

JSON_WHITESPACE = "json_whitespace"
COMPACT = "compact"
FLEXIBLE = "flexible"
JSON_WHITESPACE_MODES: tuple[str, ...] = (FLEXIBLE, COMPACT)

# llguidance's per-schema options: its JSON schema compiler reads them from the root of
# the schema, over the defaults the engine passes (1.7.6 on the PC, 1.8.0 on the Mac,
# both measured 2026-10-10).
GUIDANCE_KEY = "x-guidance"
GUIDANCE_WHITESPACE_KEYS: tuple[str, ...] = ("whitespace_flexible", "whitespace_pattern")
COMPACT_GUIDANCE: dict[str, Any] = {"whitespace_flexible": False}
ANY_OBJECT: dict[str, Any] = {"type": "object"}
JSON_OBJECT_SCHEMA_NAME = "json_object"


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
        refuse_unread_structured_options(body)
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


def refuse_unread_structured_options(body: dict[str, Any]) -> None:
    """Refuse a structured_outputs option that no engine reads per request."""
    options = body.get("structured_outputs")
    if not isinstance(options, dict):
        return
    stated = sorted(
        key
        for key, default in UNREAD_STRUCTURED_OPTIONS.items()
        if key in options and options[key] != default
    )
    if not stated:
        return
    raise ApiError(
        400,
        "structured_output_not_served",
        f"structured_outputs states {', '.join(stated)}, which no engine here reads "
        "per request: vLLM 0.29.0 takes them only from its server-wide config "
        "(v1/structured_output/backend_guidance.py L91-95) and drops them from a "
        "request without a word, and mlx-lm's patch does not act on them. Whitespace "
        f'between JSON tokens is stated with "{JSON_WHITESPACE}": "{COMPACT}". Nothing '
        "was sent",
        {"fields": [f"structured_outputs.{key}" for key in stated]},
    )


def _json_targets(body: dict[str, Any]) -> list[str]:
    """Where `body` asks for JSON: `response_format` (json_schema, json_object), then
    `structured_outputs` (json, json_object)."""
    found: list[str] = []
    response_format = body.get(RESPONSE_FORMAT)
    if isinstance(response_format, dict) and response_format.get("type") in (
        "json_schema",
        "json_object",
    ):
        found.append(RESPONSE_FORMAT)
    outputs = body.get("structured_outputs")
    if isinstance(outputs, dict) and any(
        outputs.get(kind) not in (None, False) for kind in ("json", "json_object")
    ):
        found.append("structured_outputs")
    return found


def _schema_text(value: Any) -> Any:
    """A schema as given: structured_outputs.json may be the schema's JSON text."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def _schemas_of(body: dict[str, Any]) -> list[Any]:
    """The JSON schemas `body` states (a json_object states none)."""
    schemas: list[Any] = []
    response_format = body.get(RESPONSE_FORMAT)
    if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
        wrapper = response_format.get("json_schema")
        schemas.append(wrapper.get("schema") if isinstance(wrapper, dict) else None)
    outputs = body.get("structured_outputs")
    if isinstance(outputs, dict) and outputs.get("json") not in (None, False):
        schemas.append(_schema_text(outputs["json"]))
    return schemas


def take_json_whitespace(body: dict[str, Any]) -> str | None:
    """Take `json_whitespace` off a chat body and refuse what no engine could keep: a
    value other than compact or flexible, no JSON constraint beside it, a schema that
    is not an object or that states llguidance's whitespace itself. None when the member
    is absent. Whether the engine keeps it is `refuse_unkept_json_whitespace`'s, once
    the engine is known."""
    if JSON_WHITESPACE not in body:
        return None
    mode = body.pop(JSON_WHITESPACE)
    if mode not in JSON_WHITESPACE_MODES:
        raise ApiError(
            400,
            "invalid_request",
            f'"{JSON_WHITESPACE}" is "{COMPACT}" (no whitespace between JSON tokens, '
            f'whitespace only inside strings) or "{FLEXIBLE}" (the default: whitespace '
            f"wherever JSON allows it); got {mode!r}",
            {JSON_WHITESPACE: mode},
        )
    if not _json_targets(body):
        raise ApiError(
            400,
            "json_whitespace_without_json",
            f'"{JSON_WHITESPACE}" shapes the whitespace of a JSON answer, and this chat '
            "asks for none: it goes with response_format json_schema or json_object, "
            "or structured_outputs json or json_object. Send the schema, or drop "
            f'"{JSON_WHITESPACE}"',
            {"constrained": constrained_fields(body)},
        )
    for schema in _schemas_of(body):
        if not isinstance(schema, dict):
            raise ApiError(
                400,
                "invalid_request",
                f'"{JSON_WHITESPACE}" is written into the JSON schema, and this chat '
                f"states a schema that is not a JSON object ({type(schema).__name__})",
            )
        guidance = schema.get(GUIDANCE_KEY)
        stated = (
            sorted(key for key in GUIDANCE_WHITESPACE_KEYS if key in guidance)
            if isinstance(guidance, dict)
            else []
        )
        if stated:
            raise ApiError(
                400,
                "json_whitespace_conflict",
                f'this chat states "{JSON_WHITESPACE}" and its schema states '
                f"{GUIDANCE_KEY} {', '.join(stated)}: the same thing twice. State it "
                f'once, as "{JSON_WHITESPACE}"',
                {"fields": [f"{GUIDANCE_KEY}.{key}" for key in stated]},
            )
    return mode


def refuse_unkept_json_whitespace(
    *, engine: str, model_id: str, compact: bool, basis: str, mode: str | None
) -> None:
    """Refuse compact JSON on an engine that does not keep it. Flexible is what every
    engine that enforces a JSON schema does already, so it is never refused here."""
    if mode != COMPACT or compact:
        return
    raise ApiError(
        400,
        "json_whitespace_not_served",
        f'the {engine} engine serving {model_id!r} cannot keep "{JSON_WHITESPACE}": '
        f'"{COMPACT}": {basis}. Nothing was sent to it',
        {"model": model_id, "engine": engine},
    )


def refuse_upstream_json_whitespace(model: str) -> None:
    raise ApiError(
        400,
        "json_whitespace_not_served",
        f"{model!r} is an upstream's model, and \"{JSON_WHITESPACE}\" is kept only on "
        "an engine Crucible runs. Nothing was sent",
        {"model": model},
    )


def _compact_schema(schema: Any) -> dict[str, Any]:
    compact = copy.deepcopy(_schema_text(schema))
    guidance = compact.get(GUIDANCE_KEY)
    compact[GUIDANCE_KEY] = {
        **(guidance if isinstance(guidance, dict) else {}),
        **COMPACT_GUIDANCE,
    }
    return compact


def with_compact_json(body: dict[str, Any]) -> dict[str, Any]:
    """The body an llguidance engine is sent for compact JSON: every JSON schema it
    states carries `x-guidance.whitespace_flexible: false`, and a json_object goes as
    the schema `{"type": "object"}` carrying it. `take_json_whitespace` has already
    refused a body this cannot be written into."""
    sent = dict(body)
    response_format = sent.get(RESPONSE_FORMAT)
    if isinstance(response_format, dict):
        kind = response_format.get("type")
        if kind == "json_schema":
            wrapper = dict(response_format["json_schema"])
            wrapper["schema"] = _compact_schema(wrapper["schema"])
            sent[RESPONSE_FORMAT] = {**response_format, "json_schema": wrapper}
        elif kind == "json_object":
            sent[RESPONSE_FORMAT] = {
                "type": "json_schema",
                "json_schema": {
                    "name": JSON_OBJECT_SCHEMA_NAME,
                    "schema": _compact_schema(ANY_OBJECT),
                },
            }
    outputs = sent.get("structured_outputs")
    if isinstance(outputs, dict):
        if outputs.get("json") not in (None, False):
            sent["structured_outputs"] = {**outputs, "json": _compact_schema(outputs["json"])}
        elif outputs.get("json_object") not in (None, False):
            rest = {key: value for key, value in outputs.items() if key != "json_object"}
            sent["structured_outputs"] = {**rest, "json": _compact_schema(ANY_OBJECT)}
    return sent


__all__ = [
    "COMPACT",
    "FLEXIBLE",
    "GRAMMAR_FIELDS",
    "GUIDANCE_KEY",
    "JSON_WHITESPACE",
    "RESPONSE_FORMAT",
    "UNREAD_STRUCTURED_OPTIONS",
    "constrained_fields",
    "constraints_of",
    "refuse_unenforced_constraint",
    "refuse_unkept_json_whitespace",
    "refuse_unread_structured_options",
    "refuse_upstream_json_whitespace",
    "take_json_whitespace",
    "with_compact_json",
]
