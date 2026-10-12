from __future__ import annotations

import asyncio
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterator

import httpx

from .errors import ApiError
from .upstreamrecord import UpstreamRecord

ANTHROPIC_BASE = "https://api.anthropic.com"
OPENAI_BASE = "https://api.openai.com"

ANTHROPIC_VERSION = "2023-06-01"

ANTHROPIC_MAX_TOKENS_DEFAULT = 4096

TEST_TIMEOUT_SECONDS = 20.0


def _models_url(record: UpstreamRecord) -> str:
    if record.name == "anthropic":
        return f"{ANTHROPIC_BASE}/v1/models"
    if record.name == "openai":
        return f"{OPENAI_BASE}/v1/models"
    return f"{record.url}/api/tags"


def auth_headers(record: UpstreamRecord) -> dict[str, str]:
    if record.name == "anthropic":
        return {
            "x-api-key": record.key or "",
            "anthropic-version": ANTHROPIC_VERSION,
        }
    if record.name == "openai":
        return {"Authorization": f"Bearer {record.key}"}
    return {}


def _read_model_ids(record: UpstreamRecord, payload: Any) -> list[str]:
    if record.name == "ollama":
        rows = payload.get("models") if isinstance(payload, dict) else None
        field = "name"
    else:
        rows = payload.get("data") if isinstance(payload, dict) else None
        field = "id"
    if not isinstance(rows, list):
        raise ApiError(
            502,
            "upstream_unreachable",
            f"{record.name} answered {_models_url(record)} with a body this "
            f"server cannot read: expected an object carrying a list, got "
            f"{type(payload).__name__}. Something is at that address and it is "
            f"not {record.name}",
            {"upstream": record.name, "url": _models_url(record)},
        )
    found: list[str] = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get(field), str):
            found.append(row[field])
    return found


async def list_models(client: httpx.AsyncClient, record: UpstreamRecord) -> list[str]:
    try:
        response = await client.get(
            _models_url(record),
            headers=auth_headers(record),
            timeout=TEST_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        raise ApiError(
            502,
            "upstream_unreachable",
            f"nothing answered at {_models_url(record)}, which is where this "
            f"server reaches {record.name}: {type(exc).__name__}: {exc}",
            {"upstream": record.name, "url": _models_url(record)},
        ) from None
    if response.status_code != 200:
        raise ApiError(
            502,
            "upstream_rejected",
            f"{record.name} rejected the credential this server sent it, with "
            f"HTTP {response.status_code}. {record.name} said: "
            f"{upstream_message(response.content)}. This is the upstream's "
            f"answer about the key, not Crucible's about your token",
            {
                "upstream": record.name,
                "upstream_status": response.status_code,
                "url": _models_url(record),
            },
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise ApiError(
            502,
            "upstream_unreachable",
            f"{record.name} answered {_models_url(record)} with 200 and a body "
            f"that is not JSON: {exc}",
            {"upstream": record.name, "url": _models_url(record)},
        ) from None
    return _read_model_ids(record, payload)


def upstream_message(body: bytes) -> str:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body.decode("utf-8", "replace")[:500]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        if isinstance(error, str):
            return error
    return json.dumps(payload)[:500]


def chat_url(record: UpstreamRecord) -> str:
    if record.name == "anthropic":
        return f"{ANTHROPIC_BASE}/v1/messages"
    if record.name == "openai":
        return f"{OPENAI_BASE}/v1/chat/completions"
    return f"{record.url}/api/chat"


def chat_headers(record: UpstreamRecord) -> dict[str, str]:
    return {"Content-Type": "application/json", **auth_headers(record)}


DIALECT_OPENAI = "openai"
DIALECT_ANTHROPIC = "anthropic"
DIALECT_OLLAMA = "ollama"

CONTEXT_FIELD = "context_tokens"

CONTEXT_HEADER = "X-Crucible-Context"

CONTEXT_SOURCE_REQUEST = "request"
CONTEXT_SOURCE_MODELFILE = "modelfile"
CONTEXT_SOURCE_MODEL = "model"
CONTEXT_SOURCE_DROPPED = "dropped"
CONTEXT_SOURCE_UPSTREAM = "upstream"


@dataclass(frozen=True)
class OllamaContext:

    num_ctx: int
    source: str


@dataclass(frozen=True)
class Forwarded:

    body: bytes
    sources: dict[str, str]
    dialect: str
    context: dict[str, Any]
    include_usage: bool = False


SOURCE_DROPPED = "dropped"

SOURCE_UPSTREAM_DEFAULT = f"upstream default {ANTHROPIC_MAX_TOKENS_DEFAULT}"


def _sources(
    body: dict[str, Any], *, filled_max_tokens: bool, forwards_thinking: bool = False
) -> dict[str, str]:
    from .manifests import DEFAULTS_KEYS, DEFAULTS_WIRE_KEYS
    from .sampling import SOURCE_ENGINE, SOURCE_REQUEST, TEMPLATE_KWARGS, THINKING_KEY

    sources: dict[str, str] = {}
    for key in DEFAULTS_WIRE_KEYS:
        if key in body:
            sources[key] = SOURCE_REQUEST
        else:
            sources[key] = SOURCE_ENGINE
    if "max_completion_tokens" in body:
        sources["max_tokens"] = SOURCE_REQUEST
    elif filled_max_tokens and "max_tokens" not in body:
        sources["max_tokens"] = SOURCE_UPSTREAM_DEFAULT
    kwargs = body.get(TEMPLATE_KWARGS)
    if isinstance(kwargs, dict) and THINKING_KEY in kwargs:
        sources["thinking"] = SOURCE_REQUEST if forwards_thinking else SOURCE_DROPPED
    else:
        sources["thinking"] = SOURCE_ENGINE
    return {key: sources[key] for key in DEFAULTS_KEYS}


def stated_context(body: dict[str, Any]) -> int | None:
    if CONTEXT_FIELD not in body:
        return None
    value = body[CONTEXT_FIELD]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ApiError(
            400,
            "invalid_request",
            f"{CONTEXT_FIELD} must be a positive integer — the tokens of prompt "
            f"plus answer this chat needs — got {value!r}",
            {"field": CONTEXT_FIELD},
        )
    return value


def _hosted_context(body: dict[str, Any]) -> dict[str, Any]:
    stated = stated_context(body)
    return {
        "num_ctx": None,
        "source": CONTEXT_SOURCE_UPSTREAM if stated is None else CONTEXT_SOURCE_DROPPED,
    }


def forward_body(name: str, model_id: str, body: dict[str, Any]) -> Forwarded:
    from .sampling import TEMPLATE_KWARGS

    if name == "ollama":
        raise ValueError("ollama is forwarded by forward_ollama, which resolves num_ctx")
    context = _hosted_context(body)
    if name != "anthropic":
        document = {
            k: v for k, v in body.items() if k not in (TEMPLATE_KWARGS, CONTEXT_FIELD)
        }
        document["model"] = model_id
        return Forwarded(
            body=json.dumps(document).encode("utf-8"),
            sources=_sources(body, filled_max_tokens=False),
            dialect=DIALECT_OPENAI,
            context=context,
        )

    document, include_usage = _anthropic_document(model_id, body)
    return Forwarded(
        body=json.dumps(document).encode("utf-8"),
        sources=_sources(body, filled_max_tokens=True),
        dialect=DIALECT_ANTHROPIC,
        context=context,
        include_usage=include_usage,
    )


# The Messages API's effort levels (output_config.effort), and what each current model
# takes, from Anthropic's own documentation as read on 2026-10-12:
#   https://platform.claude.com/docs/en/build-with-claude/effort.md
#   https://platform.claude.com/docs/en/build-with-claude/structured-outputs.md
#   https://platform.claude.com/docs/en/models/opus-5-5/migration-guide.md
#   https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide.md
#   https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide.md
#   https://platform.claude.com/docs/en/models/fable-5-1/migration-guide.md
# A new model is one row. A model no row names is forwarded as asked and Anthropic
# answers for it; a refusal it sends is relayed as `502 upstream_rejected` in its words.
ANTHROPIC_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

_EVERY_EFFORT = ANTHROPIC_EFFORTS
_NO_XHIGH: tuple[str, ...] = ("low", "medium", "high", "max")


@dataclass(frozen=True)
class AnthropicModelRule:

    sampling: bool
    structured_output: bool
    efforts: tuple[str, ...]


ANTHROPIC_MODEL_RULES: dict[str, AnthropicModelRule] = {
    "claude-fable-5-1": AnthropicModelRule(False, True, _EVERY_EFFORT),
    "claude-mythos-5-1": AnthropicModelRule(False, True, _EVERY_EFFORT),
    "claude-fable-5": AnthropicModelRule(False, True, _EVERY_EFFORT),
    "claude-mythos-5": AnthropicModelRule(False, True, _EVERY_EFFORT),
    "claude-opus-5-5": AnthropicModelRule(False, True, _EVERY_EFFORT),
    "claude-opus-5": AnthropicModelRule(False, True, _EVERY_EFFORT),
    "claude-opus-4-8": AnthropicModelRule(False, True, _EVERY_EFFORT),
    "claude-opus-4-7": AnthropicModelRule(False, True, _EVERY_EFFORT),
    "claude-opus-4-6": AnthropicModelRule(True, True, _NO_XHIGH),
    "claude-opus-4-5": AnthropicModelRule(True, True, ("low", "medium", "high")),
    "claude-sonnet-5-5": AnthropicModelRule(False, True, _EVERY_EFFORT),
    "claude-sonnet-5": AnthropicModelRule(False, True, _EVERY_EFFORT),
    "claude-sonnet-4-6": AnthropicModelRule(True, True, _NO_XHIGH),
    "claude-sonnet-4-5": AnthropicModelRule(True, True, ()),
    "claude-haiku-5-5": AnthropicModelRule(False, True, _EVERY_EFFORT),
    "claude-haiku-4-5": AnthropicModelRule(True, True, ()),
}

ANTHROPIC_SAMPLING: tuple[str, ...] = ("temperature", "top_p", "top_k")

# The one sampling value a model that takes none still accepts: Anthropic's default
# temperature. Anything else, and any top_p or top_k, is a 400 there.
ANTHROPIC_DEFAULT_TEMPERATURE = 1

ANTHROPIC_EFFORT_FIELD = "reasoning_effort"

_ANTHROPIC_READS: frozenset[str] = frozenset(
    {
        "model",
        "messages",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "response_format",
        "chat_template_kwargs",
        CONTEXT_FIELD,
        "user",
        "n",
        ANTHROPIC_EFFORT_FIELD,
        *ANTHROPIC_SAMPLING,
    }
)

_DATA_IMAGE_TYPED = re.compile(
    r"^data:(?P<media>image/[A-Za-z0-9.+-]+);base64,(?P<data>.+)$", re.S
)


def anthropic_rule(model_id: str) -> tuple[str, AnthropicModelRule] | None:
    best: str | None = None
    for prefix in ANTHROPIC_MODEL_RULES:
        if model_id == prefix or model_id.startswith(prefix + "-"):
            if best is None or len(prefix) > len(best):
                best = prefix
    return None if best is None else (best, ANTHROPIC_MODEL_RULES[best])


def _field_unsupported(
    upstream: str, fields: list[str], why: str, carried: frozenset[str]
) -> ApiError:
    return ApiError(
        400,
        "upstream_field_unsupported",
        f"the {upstream} upstream does not carry {fields}: {why}",
        {"upstream": upstream, "fields": fields, "carried": sorted(carried)},
    )


def _anthropic_unsupported(fields: list[str], why: str) -> ApiError:
    return _field_unsupported("anthropic", fields, why, _ANTHROPIC_READS)


def _param_unsupported(
    param: str, model_id: str, rule: str | None, why: str, **details: Any
) -> ApiError:
    return ApiError(
        400,
        "upstream_param_unsupported",
        f"anthropic/{model_id} does not take {param}: {why}",
        {"upstream": "anthropic", "param": param, "model": model_id, "rule": rule,
         **details},
    )


def _system_text(where: str, content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ApiError(400, "invalid_request",
                       f"{where}.content must be a string or a list of parts",
                       {"field": f"{where}.content"})
    texts: list[str] = []
    for number, part in enumerate(content):
        at = f"{where}.content[{number}]"
        if (not isinstance(part, dict) or part.get("type") != "text"
                or not isinstance(part.get("text"), str) or set(part) != {"type", "text"}):
            raise _anthropic_unsupported(
                [at], "a system message crosses as text only, into the top-level system"
            )
        texts.append(part["text"])
    return "\n".join(texts)


def _anthropic_image(at: str, part: dict[str, Any]) -> dict[str, Any]:
    extra = sorted(set(part) - {"type", "image_url"})
    if extra:
        raise _anthropic_unsupported([f"{at}.{key}" for key in extra],
                                     "an image part crosses as its image_url alone")
    image = part.get("image_url")
    if isinstance(image, dict):
        extra = sorted(set(image) - {"url", "detail"})
        if extra:
            raise _anthropic_unsupported([f"{at}.image_url.{key}" for key in extra],
                                         "an image crosses as its url alone")
        detail = image.get("detail")
        if detail is not None and detail != "auto":
            raise _anthropic_unsupported(
                [f"{at}.image_url.detail"],
                f"the Messages API has no per-image detail setting, so {detail!r} "
                "could not be honoured; leave it out (or `auto`)",
            )
        url = image.get("url")
    else:
        url = image
    if not isinstance(url, str):
        raise ApiError(400, "invalid_request", f"{at}.image_url.url must be a string",
                       {"field": f"{at}.image_url.url"})
    match = _DATA_IMAGE_TYPED.match(url)
    if match is not None:
        return {"type": "image", "source": {"type": "base64",
                                            "media_type": match.group("media"),
                                            "data": match.group("data")}}
    if url.startswith(("https://", "http://")):
        return {"type": "image", "source": {"type": "url", "url": url}}
    raise ApiError(
        400, "invalid_request",
        f"{at} must carry its image as a `data:image/...;base64,` URL or an http(s) "
        "address Anthropic fetches itself",
        {"field": at},
    )


def _anthropic_message(index: int, message: Any) -> dict[str, Any]:
    where = f"messages[{index}]"
    if not isinstance(message, dict):
        raise ApiError(400, "invalid_request", f"{where} must be an object",
                       {"field": where})
    role = message.get("role")
    if role in ("system", "developer"):
        raise _anthropic_unsupported(
            [f"{where}.role"],
            f"a {role} message after the conversation began has no faithful place "
            "here: only the leading system messages cross, as the top-level system. "
            "Move it to the front, or fold it into a user turn yourself",
        )
    if role not in ("user", "assistant"):
        raise _anthropic_unsupported(
            [f"{where}.role"],
            f"a {role!r} message has no translation here: tool calls and their "
            "results are not carried",
        )
    extra = sorted(set(message) - {"role", "content"})
    if extra:
        raise _anthropic_unsupported(
            [f"{where}.{key}" for key in extra],
            "a message crosses as its role and its content; tool calls and names "
            "have no translation here",
        )
    content = message.get("content")
    if isinstance(content, str):
        return {"role": role, "content": content}
    if not isinstance(content, list):
        raise ApiError(400, "invalid_request",
                       f"{where}.content must be a string or a list of parts",
                       {"field": f"{where}.content"})
    blocks: list[dict[str, Any]] = []
    for number, part in enumerate(content):
        at = f"{where}.content[{number}]"
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text" and isinstance(part.get("text"), str):
            extra = sorted(set(part) - {"type", "text"})
            if extra:
                raise _anthropic_unsupported([f"{at}.{key}" for key in extra],
                                             "a text part crosses as its text alone")
            blocks.append({"type": "text", "text": part["text"]})
            continue
        if kind == "image_url":
            blocks.append(_anthropic_image(at, part))
            continue
        raise _anthropic_unsupported(
            [at], f"a content part of type {kind!r} has no Messages API form here"
        )
    return {"role": role, "content": blocks}


def _anthropic_conversation(messages: Any) -> tuple[list[dict[str, Any]], str | None]:
    if not isinstance(messages, list) or not messages:
        raise ApiError(400, "invalid_request",
                       "messages must be a non-empty list", {"field": "messages"})
    system_parts: list[str] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") not in ("system", "developer"):
            break
        extra = sorted(set(message) - {"role", "content"})
        if extra:
            raise _anthropic_unsupported(
                [f"messages[{index}].{key}" for key in extra],
                "a system message crosses as its text alone",
            )
        system_parts.append(_system_text(f"messages[{index}]", message.get("content")))
        index += 1
    rest = [_anthropic_message(i, m) for i, m in enumerate(messages) if i >= index]
    if not rest:
        raise ApiError(
            400, "invalid_request",
            "messages holds only system messages; the Messages API needs at least one "
            "user turn",
            {"field": "messages"},
        )
    return rest, ("\n\n".join(system_parts) if system_parts else None)


def _anthropic_max_tokens(body: dict[str, Any]) -> int:
    if "max_tokens" in body and "max_completion_tokens" in body:
        raise ApiError(400, "invalid_request",
                       "state max_tokens or max_completion_tokens, not both",
                       {"field": "max_completion_tokens"})
    for field in ("max_tokens", "max_completion_tokens"):
        if field not in body or body[field] is None:
            continue
        value = body[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ApiError(400, "invalid_request",
                           f"{field} must be a positive integer, got {value!r}",
                           {"field": field})
        return value
    return ANTHROPIC_MAX_TOKENS_DEFAULT


def _anthropic_format(
    response_format: Any, model_id: str, rule: tuple[str, AnthropicModelRule] | None
) -> dict[str, Any] | None:
    if response_format is None:
        return None
    kind = response_format.get("type") if isinstance(response_format, dict) else None
    if kind == "text":
        return None
    if kind == "json_object":
        raise _param_unsupported(
            "response_format.type json_object", model_id,
            None if rule is None else rule[0],
            "the Messages API's structured output (output_config.format) takes a JSON "
            "schema and has no schema-less JSON mode; send response_format json_schema "
            "with the schema you want, or ask for JSON in the prompt",
        )
    if kind != "json_schema":
        raise ApiError(400, "invalid_request",
                       f"response_format.type must be text, json_object or json_schema, "
                       f"got {kind!r}", {"field": "response_format.type"})
    spec = response_format.get("json_schema")
    schema = spec.get("schema") if isinstance(spec, dict) else None
    if not isinstance(schema, dict):
        raise ApiError(400, "invalid_request",
                       "response_format.json_schema.schema must be an object",
                       {"field": "response_format.json_schema.schema"})
    extra = sorted(set(spec) - {"name", "schema", "strict"})
    if extra:
        raise _anthropic_unsupported(
            [f"response_format.json_schema.{key}" for key in extra],
            "output_config.format carries the schema alone; a description would be "
            "dropped, so put it in the prompt",
        )
    if rule is not None and not rule[1].structured_output:
        raise _param_unsupported(
            "response_format json_schema", model_id, rule[0],
            "this model has no structured output (output_config.format), and this "
            "server does not fake one with a forced tool",
        )
    return {"type": "json_schema", "schema": schema}


def _anthropic_effort(
    body: dict[str, Any], model_id: str, rule: tuple[str, AnthropicModelRule] | None
) -> str | None:
    if ANTHROPIC_EFFORT_FIELD not in body or body[ANTHROPIC_EFFORT_FIELD] is None:
        return None
    effort = body[ANTHROPIC_EFFORT_FIELD]
    if effort in ("none", "minimal"):
        raise _param_unsupported(
            f"{ANTHROPIC_EFFORT_FIELD} {effort!r}", model_id,
            None if rule is None else rule[0],
            "the Messages API's lowest effort is `low`, and thinking is not turned off "
            "through effort; send `low`",
            takes=list(ANTHROPIC_EFFORTS if rule is None else rule[1].efforts),
        )
    if effort not in ANTHROPIC_EFFORTS:
        raise ApiError(
            400, "invalid_request",
            f"{ANTHROPIC_EFFORT_FIELD} must be one of {list(ANTHROPIC_EFFORTS)}, got "
            f"{effort!r}",
            {"field": ANTHROPIC_EFFORT_FIELD},
        )
    if rule is not None and effort not in rule[1].efforts:
        takes = rule[1].efforts
        raise _param_unsupported(
            f"{ANTHROPIC_EFFORT_FIELD} {effort!r}", model_id, rule[0],
            (f"it takes {list(takes)}" if takes
             else "it has no effort setting (output_config.effort)"),
            takes=list(takes),
        )
    return effort


def _anthropic_document(model_id: str, body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    from .sampling import TEMPLATE_KWARGS, THINKING_KEY

    unread = sorted(set(body) - _ANTHROPIC_READS)
    if unread:
        raise _anthropic_unsupported(
            unread,
            "this server would otherwise drop them silently, and a dropped field is a "
            "setting the caller believes was applied",
        )
    rule = anthropic_rule(model_id)
    messages, system = _anthropic_conversation(body.get("messages"))
    document: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": _anthropic_max_tokens(body),
    }
    if system is not None:
        document["system"] = system

    for key in ANTHROPIC_SAMPLING:
        if key not in body:
            continue
        value = body[key]
        default = (
            key == "temperature"
            and type(value) in (int, float)
            and value == ANTHROPIC_DEFAULT_TEMPERATURE
        )
        if rule is not None and not rule[1].sampling and not default:
            raise _param_unsupported(
                key, model_id, rule[0],
                "this model takes no sampling parameters (only temperature at its "
                f"default, {ANTHROPIC_DEFAULT_TEMPERATURE}); leave it out and steer "
                "with the prompt",
                value=value,
            )
        document[key] = value

    stop = body.get("stop")
    if isinstance(stop, str):
        document["stop_sequences"] = [stop]
    elif isinstance(stop, list) and all(isinstance(s, str) for s in stop):
        document["stop_sequences"] = stop
    elif stop is not None:
        raise ApiError(400, "invalid_request",
                       "stop must be a string or a list of strings", {"field": "stop"})

    n = body.get("n")
    if n is not None and n != 1:
        raise _anthropic_unsupported(
            ["n"], f"the Messages API writes one answer per request, and n is {n!r}"
        )

    user = body.get("user")
    if user is not None:
        if not isinstance(user, str):
            raise ApiError(400, "invalid_request", "user must be a string",
                           {"field": "user"})
        document["metadata"] = {"user_id": user}

    kwargs = body.get(TEMPLATE_KWARGS)
    if kwargs is not None:
        if not isinstance(kwargs, dict):
            raise ApiError(400, "invalid_request",
                           f"{TEMPLATE_KWARGS} must be an object, got "
                           f"{type(kwargs).__name__}", {"field": TEMPLATE_KWARGS})
        extra = sorted(set(kwargs) - {THINKING_KEY})
        if extra:
            raise _anthropic_unsupported(
                [f"{TEMPLATE_KWARGS}.{key}" for key in extra],
                f"the Messages API has no chat-template arguments; {THINKING_KEY} is "
                "read (and dropped, as X-Crucible-Sampling says)",
            )

    output_config: dict[str, Any] = {}
    effort = _anthropic_effort(body, model_id, rule)
    if effort is not None:
        output_config["effort"] = effort
    response_format = _anthropic_format(body.get("response_format"), model_id, rule)
    if response_format is not None:
        output_config["format"] = response_format
    if output_config:
        document["output_config"] = output_config

    stream = body.get("stream")
    if stream is not None and not isinstance(stream, bool):
        raise ApiError(400, "invalid_request", "stream must be a boolean",
                       {"field": "stream"})
    if stream is True:
        document["stream"] = True
    include_usage = False
    stream_options = body.get("stream_options")
    if stream_options is not None:
        if not isinstance(stream_options, dict):
            raise ApiError(400, "invalid_request", "stream_options must be an object",
                           {"field": "stream_options"})
        extra = sorted(set(stream_options) - {"include_usage"})
        if extra:
            raise _anthropic_unsupported(
                [f"stream_options.{key}" for key in extra],
                "include_usage alone is read",
            )
        include_usage = stream_options.get("include_usage") is True
    return document, include_usage


# Anthropic's stop reasons in OpenAI's words; a reason not here is relayed as Anthropic
# said it rather than guessed at.
_STOP_REASON: dict[str, str] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "refusal": "content_filter",
}


def _finish_reason(reason: Any) -> str | None:
    if not isinstance(reason, str):
        return None
    return _STOP_REASON.get(reason, reason)


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _count(usage: dict[str, Any], key: str) -> int | None:
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def anthropic_usage(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict):
        usage = {}
    fresh = _count(usage, "input_tokens")
    written = _count(usage, "cache_creation_input_tokens")
    read = _count(usage, "cache_read_input_tokens")
    completion = _count(usage, "output_tokens")
    # OpenAI's prompt_tokens counts every prompt token; Anthropic's input_tokens leaves
    # out the ones written to and read from its cache, which it counts apart.
    prompt = None if fresh is None else fresh + (written or 0) + (read or 0)
    counted: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": (
            None if prompt is None or completion is None else prompt + completion
        ),
    }
    if read is not None:
        counted["prompt_tokens_details"] = {"cached_tokens": read}
    return counted


def anthropic_to_openai(payload: dict[str, Any], model: str) -> dict[str, Any]:
    text_parts: list[str] = []
    for block in payload.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(
            block.get("text"), str
        ):
            text_parts.append(block["text"])
    return {
        "id": payload.get("id") or _completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(text_parts)},
                "finish_reason": _finish_reason(payload.get("stop_reason")),
            }
        ],
        "usage": anthropic_usage(payload.get("usage")),
    }


class AnthropicStreamTranslator:

    def __init__(self, model: str, *, include_usage: bool = False) -> None:
        self.model = model
        self.include_usage = include_usage
        self._id = _completion_id()
        self._buffer = b""
        self._usage: dict[str, Any] = {}
        self._done = False
        self._failed = False

    def _frame(self, payload: dict[str, Any]) -> bytes:
        return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"

    def _chunk(self, delta: dict[str, Any], finish_reason: str | None) -> bytes:
        return self._frame(
            {
                "id": self._id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.model,
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish_reason}
                ],
            }
        )

    def _translate(self, event: dict[str, Any]) -> Iterator[bytes]:
        if self._done or self._failed:
            return
        kind = event.get("type")
        if kind == "message_start":
            message = event.get("message")
            if isinstance(message, dict):
                if isinstance(message.get("id"), str):
                    self._id = message["id"]
                if isinstance(message.get("usage"), dict):
                    self._usage.update(message["usage"])
            yield self._chunk({"role": "assistant"}, None)
        elif kind == "content_block_delta":
            delta = event.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "text_delta":
                text = delta.get("text")
                if isinstance(text, str):
                    yield self._chunk({"content": text}, None)
        elif kind == "message_delta":
            # Its usage is cumulative, and may restate the input counts.
            if isinstance(event.get("usage"), dict):
                self._usage.update(event["usage"])
            delta = event.get("delta")
            reason = delta.get("stop_reason") if isinstance(delta, dict) else None
            if isinstance(reason, str):
                yield self._chunk({}, _finish_reason(reason))
        elif kind == "message_stop":
            self._done = True
            if self.include_usage:
                yield self._frame(
                    {
                        "id": self._id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": self.model,
                        "choices": [],
                        "usage": anthropic_usage(self._usage),
                    }
                )
            yield b"data: [DONE]\n\n"
        elif kind == "error":
            self._failed = True
            error = event.get("error")
            said = error if isinstance(error, dict) else {"message": json.dumps(error)}
            yield self._frame({"error": {**said, "upstream": "anthropic"}})

    def feed(self, chunk: bytes) -> Iterator[bytes]:
        self._buffer += chunk
        while b"\n\n" in self._buffer:
            frame, self._buffer = self._buffer.split(b"\n\n", 1)
            yield from self._parse(frame)

    def _parse(self, frame: bytes) -> Iterator[bytes]:
        for line in frame.split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:"):].strip()
            if payload == b"":
                continue
            try:
                event = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if not self._failed and not self._done:
                    self._failed = True
                    yield self._frame(
                        {"error": {"message": "anthropic sent a stream event that is "
                                   "not JSON: " + payload[:200].decode("utf-8", "replace"),
                                   "upstream": "anthropic"}}
                    )
                continue
            if isinstance(event, dict):
                yield from self._translate(event)

    def finish(self) -> Iterator[bytes]:
        if self._buffer.strip():
            frame, self._buffer = self._buffer, b""
            yield from self._parse(frame)
        if not self._done and not self._failed:
            yield self._frame(
                {"error": {"message": "anthropic's stream ended before message_stop; "
                           "the answer is truncated", "upstream": "anthropic"}}
            )


OLLAMA_LOOKUP_ATTEMPTS = 3
OLLAMA_LOOKUP_BACKOFF_SECONDS: tuple[float, ...] = (0.5, 2.0)
OLLAMA_LOOKUP_TIMEOUT_SECONDS = 10.0


class OllamaContexts:

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], tuple[str, OllamaContext]] = {}

    def get(self, url: str, tag: str, digest: str) -> OllamaContext | None:
        entry = self._entries.get((url, tag))
        if entry is None or entry[0] != digest:
            return None
        return entry[1]

    def put(self, url: str, tag: str, digest: str, context: OllamaContext) -> None:
        self._entries[(url, tag)] = (digest, context)


def _ollama_tag(model_id: str) -> str:
    last = model_id.rsplit("/", 1)[-1]
    return model_id if ":" in last else f"{model_id}:latest"


def _context_unknown(record: UpstreamRecord, model_id: str, why: str) -> ApiError:
    return ApiError(
        502,
        "upstream_context_unknown",
        f"this server could not learn the context window of {model_id!r} from "
        f"ollama at {record.url}: {why}. It will not send the chat without one — "
        "an Ollama chat that states no num_ctx runs at Ollama's default (4096) "
        "and a longer prompt is silently cut from the front. State it yourself "
        f"with `{CONTEXT_FIELD}` or bring Ollama back",
        {"upstream": record.name, "model": model_id, "url": record.url},
    )


async def _ollama_read(
    client: httpx.AsyncClient,
    record: UpstreamRecord,
    model_id: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None,
) -> Any:
    url = f"{record.url}{path}"
    last = ""
    answered = False
    for attempt in range(1, OLLAMA_LOOKUP_ATTEMPTS + 1):
        try:
            response = await client.request(
                method, url, json=payload, timeout=OLLAMA_LOOKUP_TIMEOUT_SECONDS
            )
        except httpx.HTTPError as exc:
            answered = False
            last = f"{type(exc).__name__}: {exc}"
        else:
            answered = True
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    raise _context_unknown(
                        record, model_id, f"{path} answered 200 with a body that is "
                        f"not JSON ({exc}); something at that address is not Ollama"
                    ) from None
            if response.status_code < 500:
                raise ApiError(
                    502,
                    "upstream_rejected",
                    f"ollama refused {path} for {model_id!r} with "
                    f"{response.status_code}: {upstream_message(response.content)}",
                    {"upstream": record.name, "upstream_status": response.status_code,
                     "model": model_id},
                )
            last = f"HTTP {response.status_code}: {upstream_message(response.content)}"
        if attempt < OLLAMA_LOOKUP_ATTEMPTS:
            wait = OLLAMA_LOOKUP_BACKOFF_SECONDS[attempt - 1]
            print(
                f"crucible: ollama {path} for {model_id!r} at {url} did not answer "
                f"({last}), attempt {attempt} of {OLLAMA_LOOKUP_ATTEMPTS}; asking "
                f"again in {wait}s",
                file=sys.stderr,
            )
            await asyncio.sleep(wait)
    print(
        f"crucible: ollama {path} for {model_id!r} at {url} did not answer "
        f"({last}), attempt {OLLAMA_LOOKUP_ATTEMPTS} of {OLLAMA_LOOKUP_ATTEMPTS}; "
        "giving up",
        file=sys.stderr,
    )
    if not answered:
        raise ApiError(
            502,
            "upstream_unreachable",
            f"ollama did not answer at {url} on {OLLAMA_LOOKUP_ATTEMPTS} attempts "
            f"({last}); the chat was not sent",
            {"upstream": record.name, "url": url, "attempts": OLLAMA_LOOKUP_ATTEMPTS},
        )
    raise _context_unknown(
        record, model_id,
        f"{path} did not answer on {OLLAMA_LOOKUP_ATTEMPTS} attempts ({last})",
    )


def _context_from_show(
    record: UpstreamRecord, model_id: str, payload: Any
) -> OllamaContext:
    if not isinstance(payload, dict):
        raise _context_unknown(record, model_id, "/api/show answered with no object")
    parameters = payload.get("parameters")
    if isinstance(parameters, str):
        for line in parameters.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "num_ctx":
                if parts[1].isdigit() and int(parts[1]) > 0:
                    return OllamaContext(int(parts[1]), CONTEXT_SOURCE_MODELFILE)
                raise _context_unknown(
                    record, model_id,
                    f"the tag's num_ctx parameter is {parts[1]!r}, not a count",
                )
    info = payload.get("model_info")
    if not isinstance(info, dict):
        raise _context_unknown(
            record, model_id, "/api/show carried no model_info to read the "
            "trained context from, and the tag states no num_ctx"
        )
    arch = info.get("general.architecture")
    key = f"{arch}.context_length"
    length = info.get(key)
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise _context_unknown(
            record, model_id, f"/api/show's model_info has no positive {key!r} "
            f"(general.architecture is {arch!r}), and the tag states no num_ctx"
        )
    return OllamaContext(length, CONTEXT_SOURCE_MODEL)


async def resolve_ollama_context(
    client: httpx.AsyncClient,
    record: UpstreamRecord,
    model_id: str,
    body: dict[str, Any],
    contexts: OllamaContexts,
) -> OllamaContext:
    stated = stated_context(body)
    if stated is not None:
        return OllamaContext(stated, CONTEXT_SOURCE_REQUEST)
    tag = _ollama_tag(model_id)
    listing = await _ollama_read(client, record, model_id, "GET", "/api/tags", None)
    digest: str | None = None
    rows = listing.get("models") if isinstance(listing, dict) else None
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        if tag in (row.get("name"), row.get("model")) and isinstance(row.get("digest"), str):
            digest = row["digest"]
            break
    if digest is not None:
        known = contexts.get(record.url or "", tag, digest)
        if known is not None:
            return known
    shown = await _ollama_read(
        client, record, model_id, "POST", "/api/show", {"model": model_id}
    )
    context = _context_from_show(record, model_id, shown)
    if digest is not None:
        contexts.put(record.url or "", tag, digest, context)
    return context


_OLLAMA_OPTIONS: dict[str, str] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "seed": "seed",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "repetition_penalty": "repeat_penalty",
}

_OLLAMA_READS: frozenset[str] = frozenset(
    {
        "model",
        "messages",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "response_format",
        "chat_template_kwargs",
        CONTEXT_FIELD,
        "user",
        *_OLLAMA_OPTIONS,
    }
)

_DATA_IMAGE = re.compile(r"^data:image/[A-Za-z0-9.+-]+;base64,(?P<data>.+)$", re.S)


def _unsupported(fields: list[str], why: str) -> ApiError:
    return _field_unsupported("ollama", fields, why, _OLLAMA_READS)


def _ollama_message(index: int, message: Any) -> dict[str, Any]:
    where = f"messages[{index}]"
    if not isinstance(message, dict):
        raise ApiError(400, "invalid_request", f"{where} must be an object",
                       {"field": where})
    extra = sorted(set(message) - {"role", "content"})
    if extra:
        raise _unsupported(
            [f"{where}.{key}" for key in extra],
            "a message crosses as its role and its content; tool calls and names "
            "have no translation here",
        )
    role = message.get("role")
    if role not in ("system", "user", "assistant"):
        raise ApiError(
            400, "invalid_request",
            f"{where}.role must be system, user or assistant, got {role!r}",
            {"field": f"{where}.role"},
        )
    content = message.get("content")
    if isinstance(content, str):
        return {"role": role, "content": content}
    if not isinstance(content, list):
        raise ApiError(
            400, "invalid_request",
            f"{where}.content must be a string or a list of parts",
            {"field": f"{where}.content"},
        )
    texts: list[str] = []
    images: list[str] = []
    for number, part in enumerate(content):
        at = f"{where}.content[{number}]"
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
            continue
        if kind == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            match = _DATA_IMAGE.match(url) if isinstance(url, str) else None
            if match is None:
                raise ApiError(
                    400, "invalid_request",
                    f"{at} must carry its image inline as a `data:image/...;base64,` "
                    "URL: Ollama takes image bytes, and this server does not fetch "
                    "URLs on a caller's behalf",
                    {"field": at},
                )
            images.append(match.group("data"))
            continue
        raise _unsupported([at], f"a content part of type {kind!r} has no Ollama form")
    translated: dict[str, Any] = {"role": role, "content": "\n".join(texts)}
    if images:
        translated["images"] = images
    return translated


def _ollama_document(model_id: str, body: dict[str, Any]) -> dict[str, Any]:
    from .sampling import TEMPLATE_KWARGS, THINKING_KEY

    unread = sorted(set(body) - _OLLAMA_READS)
    if unread:
        hint = (
            f"; the context window is stated as `{CONTEXT_FIELD}` and sampling as "
            "OpenAI's own top-level fields"
            if {"num_ctx", "options", "num_predict"} & set(unread)
            else ""
        )
        raise _unsupported(unread, "this server would otherwise drop them "
                           "silently, and a dropped field is a setting the "
                           f"caller believes was applied{hint}")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ApiError(400, "invalid_request",
                       "messages must be a non-empty list", {"field": "messages"})

    options: dict[str, Any] = {}
    for field, option in _OLLAMA_OPTIONS.items():
        if field in body:
            options[option] = body[field]
    if "max_tokens" in body and "max_completion_tokens" in body:
        raise ApiError(
            400, "invalid_request",
            "state max_tokens or max_completion_tokens, not both",
            {"field": "max_completion_tokens"},
        )
    for field in ("max_tokens", "max_completion_tokens"):
        if field in body:
            options["num_predict"] = body[field]
    stop = body.get("stop")
    if isinstance(stop, str):
        options["stop"] = [stop]
    elif isinstance(stop, list):
        options["stop"] = stop
    elif stop is not None:
        raise ApiError(400, "invalid_request",
                       "stop must be a string or a list of strings", {"field": "stop"})

    document: dict[str, Any] = {
        "model": model_id,
        "messages": [_ollama_message(i, m) for i, m in enumerate(messages)],
        "stream": body.get("stream") is True,
        "options": options,
    }

    kwargs = body.get(TEMPLATE_KWARGS)
    if kwargs is not None:
        if not isinstance(kwargs, dict):
            raise ApiError(
                400, "invalid_request",
                f"{TEMPLATE_KWARGS} must be an object, got {type(kwargs).__name__}",
                {"field": TEMPLATE_KWARGS},
            )
        extra = sorted(set(kwargs) - {THINKING_KEY})
        if extra:
            raise _unsupported(
                [f"{TEMPLATE_KWARGS}.{key}" for key in extra],
                f"Ollama has no chat-template arguments; {THINKING_KEY} alone "
                "crosses, as `think`",
            )
        if THINKING_KEY in kwargs:
            if not isinstance(kwargs[THINKING_KEY], bool):
                raise ApiError(
                    400, "invalid_request",
                    f"{TEMPLATE_KWARGS}.{THINKING_KEY} must be a boolean",
                    {"field": f"{TEMPLATE_KWARGS}.{THINKING_KEY}"},
                )
            document["think"] = kwargs[THINKING_KEY]

    response_format = body.get("response_format")
    if response_format is not None:
        kind = response_format.get("type") if isinstance(response_format, dict) else None
        if kind == "json_object":
            document["format"] = "json"
        elif kind == "json_schema":
            spec = response_format.get("json_schema")
            schema = spec.get("schema") if isinstance(spec, dict) else None
            if not isinstance(schema, dict):
                raise ApiError(
                    400, "invalid_request",
                    "response_format.json_schema.schema must be an object",
                    {"field": "response_format.json_schema.schema"},
                )
            document["format"] = schema
        elif kind != "text":
            raise ApiError(
                400, "invalid_request",
                f"response_format.type must be text, json_object or json_schema, "
                f"got {kind!r}",
                {"field": "response_format.type"},
            )
    return document


async def forward_ollama(
    client: httpx.AsyncClient,
    record: UpstreamRecord,
    model_id: str,
    body: dict[str, Any],
    contexts: OllamaContexts,
) -> Forwarded:
    document = _ollama_document(model_id, body)
    context = await resolve_ollama_context(client, record, model_id, body, contexts)
    document["options"] = {"num_ctx": context.num_ctx, **document["options"]}
    stream_options = body.get("stream_options")
    include_usage = (
        isinstance(stream_options, dict) and stream_options.get("include_usage") is True
    )
    print(
        f"crucible: ollama chat {model_id!r} at {record.url}: num_ctx "
        f"{context.num_ctx} ({context.source})",
        file=sys.stderr,
    )
    return Forwarded(
        body=json.dumps(document).encode("utf-8"),
        sources=_sources(body, filled_max_tokens=False, forwards_thinking=True),
        dialect=DIALECT_OLLAMA,
        context={"num_ctx": context.num_ctx, "source": context.source},
        include_usage=include_usage,
    )


_DONE_REASON: dict[str, str] = {"stop": "stop", "length": "length"}


def _ollama_usage(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = payload.get("prompt_eval_count")
    completion = payload.get("eval_count")
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": (
            None if not isinstance(prompt, int) or not isinstance(completion, int)
            else prompt + completion
        ),
    }


def ollama_to_openai(payload: Any, model: str) -> dict[str, Any]:
    message = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ApiError(
            502,
            "upstream_rejected",
            "ollama answered 200 with a body that carries no message.content",
            {"upstream": "ollama", "upstream_status": 200},
        )
    answer: dict[str, Any] = {"role": "assistant", "content": message["content"]}
    thinking = message.get("thinking")
    if isinstance(thinking, str) and thinking != "":
        answer["reasoning"] = thinking
    reason = payload.get("done_reason")
    return {
        "id": _completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": answer,
                "finish_reason": _DONE_REASON.get(reason, reason)
                if isinstance(reason, str) else None,
            }
        ],
        "usage": _ollama_usage(payload),
    }


class OllamaStreamTranslator:

    def __init__(self, model: str, *, include_usage: bool) -> None:
        self.model = model
        self.include_usage = include_usage
        self._id = _completion_id()
        self._buffer = b""
        self._started = False
        self._done = False
        self._failed = False

    def _frame(self, payload: dict[str, Any]) -> bytes:
        return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"

    def _chunk(self, delta: dict[str, Any], finish_reason: str | None) -> bytes:
        return self._frame(
            {
                "id": self._id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.model,
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish_reason}
                ],
            }
        )

    def _translate(self, line: dict[str, Any]) -> Iterator[bytes]:
        if self._done or self._failed:
            return
        if "error" in line:
            self._failed = True
            error = line["error"]
            yield self._frame(
                {"error": {"message": error if isinstance(error, str) else json.dumps(error),
                           "upstream": "ollama"}}
            )
            return
        if not self._started:
            self._started = True
            yield self._chunk({"role": "assistant"}, None)
        message = line.get("message")
        if isinstance(message, dict):
            thinking = message.get("thinking")
            if isinstance(thinking, str) and thinking != "":
                yield self._chunk({"reasoning": thinking}, None)
            content = message.get("content")
            if isinstance(content, str) and content != "":
                yield self._chunk({"content": content}, None)
        if line.get("done") is True:
            self._done = True
            reason = line.get("done_reason")
            yield self._chunk(
                {},
                _DONE_REASON.get(reason, reason) if isinstance(reason, str) else None,
            )
            if self.include_usage:
                yield self._frame(
                    {
                        "id": self._id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": self.model,
                        "choices": [],
                        "usage": _ollama_usage(line),
                    }
                )
            yield b"data: [DONE]\n\n"

    def _lines(self, final: bool) -> Iterator[bytes]:
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            yield from self._parse(line)
        if final and self._buffer.strip():
            line, self._buffer = self._buffer, b""
            yield from self._parse(line)

    def _parse(self, line: bytes) -> Iterator[bytes]:
        if line.strip() == b"":
            return
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._failed = True
            yield self._frame(
                {"error": {"message": "ollama sent a stream line that is not JSON: "
                           + line[:200].decode("utf-8", "replace"),
                           "upstream": "ollama"}}
            )
            return
        if isinstance(event, dict):
            yield from self._translate(event)

    def feed(self, chunk: bytes) -> Iterator[bytes]:
        self._buffer += chunk
        yield from self._lines(final=False)

    def finish(self) -> Iterator[bytes]:
        yield from self._lines(final=True)
        if not self._done and not self._failed:
            yield self._frame(
                {"error": {"message": "ollama's stream ended before it said done; "
                           "the answer is truncated", "upstream": "ollama"}}
            )


__all__ = [
    "ANTHROPIC_BASE",
    "ANTHROPIC_DEFAULT_TEMPERATURE",
    "ANTHROPIC_EFFORTS",
    "ANTHROPIC_EFFORT_FIELD",
    "ANTHROPIC_MAX_TOKENS_DEFAULT",
    "ANTHROPIC_MODEL_RULES",
    "ANTHROPIC_SAMPLING",
    "ANTHROPIC_VERSION",
    "AnthropicModelRule",
    "AnthropicStreamTranslator",
    "CONTEXT_FIELD",
    "CONTEXT_HEADER",
    "DIALECT_ANTHROPIC",
    "DIALECT_OLLAMA",
    "DIALECT_OPENAI",
    "Forwarded",
    "OLLAMA_LOOKUP_ATTEMPTS",
    "OPENAI_BASE",
    "OllamaContext",
    "OllamaContexts",
    "OllamaStreamTranslator",
    "SOURCE_DROPPED",
    "SOURCE_UPSTREAM_DEFAULT",
    "anthropic_rule",
    "anthropic_to_openai",
    "anthropic_usage",
    "auth_headers",
    "chat_headers",
    "chat_url",
    "forward_body",
    "forward_ollama",
    "list_models",
    "ollama_to_openai",
    "resolve_ollama_context",
    "stated_context",
]
