"""The three upstreams, and everything that differs between them.

PHASE15-HOST.md sections 2, 3.2 and 3.4. An **upstream** is an HTTP
chat-completions service this server forwards to on the operator's account.
There are exactly three names — `anthropic`, `openai`, `ollama` — and the list
is closed on purpose: a fourth would be a provider with nowhere to say how its
body is shaped, which is the thing this module exists to be.

WHY THIS IS A MODULE AND NOT A BRANCH IN `api.py`
-------------------------------------------------
Owen, 2026-09-14: *"they dont have ollama fallbacks or cloud anything at all …
one contract, one SDK, one API, one communication method."* The provider code
leaves BookForge and Foundry, and the only way that is an improvement is if it
lands in ONE place rather than being sprinkled through the chat door. So every
sentence that is true of Anthropic and false of OpenAI is here: the URL, the
headers, the body shape, the response shape, the SSE framing.

THREE THINGS THIS MODULE REFUSES TO DO
--------------------------------------
**It never retries.** A rate limit comes back as `429` with the upstream's own
`Retry-After` and the CALLER waits (section 3.4). A request that reached the
upstream may already be billed, and a server that quietly sent it twice would be
spending somebody's money to make a graph look smoother.

**It never invents a model list.** `POST /v1/settings/upstreams/{name}/test`
asks the upstream what it serves and reports THAT. A table of cloud model names
in this repo would be stale within a month and would be a second owner of a fact
Anthropic already publishes.

**It never logs a key.** The key reaches exactly two places: the header of the
request it authenticates, and `key_hint`, which is its last four characters.
`tests/test_settings_api.py` greps every response body, every header, every log
record and every activity row for the whole key.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterator

import httpx

from .errors import ApiError

#: Exactly these three. A name outside the set is `unknown_upstream`, never a
#: fourth entry nothing knows how to call.
UPSTREAM_NAMES: tuple[str, ...] = ("anthropic", "openai", "ollama")

#: The ONE field each upstream takes, and the whole of what "configured" means.
#: Anthropic and OpenAI are reached at a fixed address with a secret; Ollama is
#: reached at an address and has no secret at all. A name given the other one's
#: field is `upstream_bad_field` — it is a request about a different upstream
#: than the one it named.
UPSTREAM_FIELD: dict[str, str] = {
    "anthropic": "key",
    "openai": "key",
    "ollama": "url",
}

#: Where the two hosted upstreams live. Not configurable, and that is the
#: absence of a feature rather than an oversight: a `base_url` per upstream is
#: how a key ends up posted to somebody else's host by a typo, and nobody has
#: asked for a proxy.
ANTHROPIC_BASE = "https://api.anthropic.com"
OPENAI_BASE = "https://api.openai.com"

#: Anthropic pins its wire format by date and requires the header on every call.
ANTHROPIC_VERSION = "2023-06-01"

#: Anthropic REQUIRES `max_tokens`; OpenAI and Ollama do not. A request that
#: states none still has to carry one, so this server states it and says so in
#: the audit header (`SOURCE_UPSTREAM_DEFAULT` below). 4096 is PHASE15-HOST.md
#: section 3.4's number.
ANTHROPIC_MAX_TOKENS_DEFAULT = 4096

#: The name the forced tool takes when a `response_format` JSON schema is
#: translated into Anthropic tool use. It is visible to the model, so it says
#: what the model is being asked for rather than naming this server.
ANTHROPIC_JSON_TOOL = "structured_answer"

#: How long `test` waits. It is one small GET against a hosted API with a person
#: watching a spinner, not a completion.
TEST_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class UpstreamRecord:
    """One configured upstream, as `config.toml` holds it.

    PRESENT MEANS CONFIGURED. There is no half-configured state: `load_config`
    refuses an `[upstreams.anthropic]` with no `key`, so a record that exists can
    always be called. That is what lets `GET /v1/settings` answer `configured`
    from the presence of the record instead of from a second flag that could
    disagree with it (ARCHITECTURE.md R1).
    """

    name: str
    #: `anthropic` and `openai`. None for `ollama`, which has no secret.
    key: str | None = None
    #: `ollama`. None for the two hosted upstreams, whose address is fixed.
    url: str | None = None

    @property
    def key_hint(self) -> str | None:
        """`…` then the last four characters, and never more.

        Enough to recognise WHICH key is in there — the question a person asks
        when two accounts are in play — and nothing else. A key shorter than
        four characters would be reported whole, so the whole of it is refused
        at the door instead (`require_key`).

        **The ellipsis is part of the value** (`U+2026`, one character, not
        three dots) and is pinned by the contract because Foundry renders the
        hint VERBATIM beside its key field. A server that returned the bare
        four characters would make every window either show `k3A9` as though
        it were the whole key, or prepend its own ellipsis — which is two
        clients inventing the same decoration, differently (R1).
        """
        if self.key is None:
            return None
        return f"…{self.key[-4:]}"


def blank(name: str) -> dict[str, Any]:
    """What `GET /v1/settings` says about an upstream nobody has configured.

    The three names are always present in the document, configured or not,
    because a window draws three cards and a key that came and went would make
    "not configured" and "this build does not know that upstream" the same
    reading.
    """
    if UPSTREAM_FIELD[name] == "key":
        return {"configured": False, "key_hint": None}
    return {"configured": False, "url": None}


def settings_entry(record: UpstreamRecord) -> dict[str, Any]:
    """What `GET /v1/settings` says about a configured one. **Never the key.**"""
    if UPSTREAM_FIELD[record.name] == "key":
        return {"configured": True, "key_hint": record.key_hint}
    return {"configured": True, "url": record.url}


def require_name(name: str, field: str) -> str:
    if name not in UPSTREAM_NAMES:
        raise ApiError(
            400,
            "unknown_upstream",
            f"{name!r} is not an upstream this server knows; it speaks to "
            f"{list(UPSTREAM_NAMES)}. A name it does not know cannot be stored: "
            "nothing would ever be able to call it",
            {"field": field, "upstream": name, "known": list(UPSTREAM_NAMES)},
        )
    return name


def require_key(name: str, value: Any, field: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ApiError(
            400,
            "invalid_request",
            f"{field} must be a non-empty string; {name} is reached with a key "
            "and a blank one is not a key",
            {"field": field},
        )
    key = value.strip()
    if len(key) < 8:
        raise ApiError(
            400,
            "invalid_request",
            f"{field} is {len(key)} characters, which is shorter than the four "
            "this server reports back as `key_hint` plus enough to be worth "
            f"hiding; {name} keys are far longer than that",
            {"field": field},
        )
    return key


def require_url(name: str, value: Any, field: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ApiError(
            400,
            "invalid_request",
            f"{field} must be a non-empty string; {name} is reached by address",
            {"field": field},
        )
    url = value.strip().rstrip("/")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ApiError(
            400,
            "invalid_request",
            f"{field} must be an http(s) URL, got {url!r}",
            {"field": field},
        )
    return url


def record_from_patch(name: str, patch: Any, field_path: str) -> UpstreamRecord:
    """One `upstreams.<name>` entry of a `PUT /v1/settings` body, validated.

    The ONE field this upstream takes, and nothing else. A `url` for `anthropic`
    is `upstream_bad_field` rather than an ignored key, because a caller that
    sent one believes it is pointing this server somewhere, and silently
    dropping it would leave them watching requests go to the address they
    thought they had replaced.
    """
    require_name(name, field_path)
    wanted = UPSTREAM_FIELD[name]
    if not isinstance(patch, dict):
        raise ApiError(
            400,
            "invalid_request",
            f"{field_path} must be an object with a {wanted!r}, or null to "
            f"remove it; got {type(patch).__name__}",
            {"field": field_path},
        )
    unknown = sorted(set(patch) - {wanted})
    if unknown:
        raise ApiError(
            400,
            "upstream_bad_field",
            f"{name} is configured with a {wanted!r} and nothing else; "
            f"{field_path} also carries {unknown}. Each upstream takes exactly "
            "one field, so a request carrying the other one is a request about "
            "a different upstream than the one it named",
            {"field": field_path, "upstream": name, "unknown": unknown,
             "takes": wanted},
        )
    if wanted not in patch:
        raise ApiError(
            400,
            "invalid_request",
            f"{field_path} must carry a {wanted!r} (or be null to remove the "
            "upstream); an empty object says nothing",
            {"field": field_path},
        )
    if wanted == "key":
        return UpstreamRecord(name=name, key=require_key(name, patch["key"],
                                                         f"{field_path}.key"))
    return UpstreamRecord(name=name, url=require_url(name, patch["url"],
                                                     f"{field_path}.url"))


def split_model(model: str) -> tuple[str, str] | None:
    """`<upstream>/<id>` split, or None because this is a local model id.

    The slash is the whole of the test (section 2), and it works because a local
    model id can never contain one — refused at manifest load,
    `manifest_model_id_slash`. One character, one owner, no table to keep.
    """
    if "/" not in model:
        return None
    name, _, rest = model.partition("/")
    return name, rest


def require_upstream_model(model: str) -> tuple[str, str]:
    """Split a chat's `model`, refusing a prefix that is not one of the three.

    `route_bad_model` and not a name of its own: section 3.2 already owns that
    word for exactly this malformation, and the door it arrives at does not
    change what the mistake is.
    """
    split = split_model(model)
    if split is None:
        raise ValueError(f"{model!r} has no '/' and is not an upstream model id")
    name, rest = split
    if name not in UPSTREAM_NAMES or rest == "":
        raise ApiError(
            400,
            "route_bad_model",
            f"{model!r} names no upstream this server knows. An upstream model "
            f"id is `<upstream>/<model>` with the upstream one of "
            f"{list(UPSTREAM_NAMES)}; a model id with no slash is a local model "
            "and is looked for on the card",
            {"field": "model", "model": model, "known": list(UPSTREAM_NAMES)},
        )
    return name, rest


# ------------------------------------------------------------------ listing


def _models_url(record: UpstreamRecord) -> str:
    if record.name == "anthropic":
        return f"{ANTHROPIC_BASE}/v1/models"
    if record.name == "openai":
        return f"{OPENAI_BASE}/v1/models"
    return f"{record.url}/api/tags"


def auth_headers(record: UpstreamRecord) -> dict[str, str]:
    """What proves this server may spend the operator's account.

    Ollama gets none: it is reached by address and has no account. That is not
    an unauthenticated hole this server opened — it is what Ollama is.
    """
    if record.name == "anthropic":
        return {
            "x-api-key": record.key or "",
            "anthropic-version": ANTHROPIC_VERSION,
        }
    if record.name == "openai":
        return {"Authorization": f"Bearer {record.key}"}
    return {}


def _read_model_ids(record: UpstreamRecord, payload: Any) -> list[str]:
    """The ids out of one upstream's own listing, in the order it gave them."""
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
            f"{record.name} answered its model listing with a body this server "
            f"cannot read: expected an object carrying a list, got "
            f"{type(payload).__name__}",
            {"upstream": record.name},
        )
    found: list[str] = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get(field), str):
            found.append(row[field])
    return found


async def list_models(client: httpx.AsyncClient, record: UpstreamRecord) -> list[str]:
    """What the upstream itself says it serves. Unbilled, and never cached.

    Not cached because the answer is somebody else's and changes without telling
    us; a stale list shown beside a key the operator just pasted is exactly the
    moment they would believe it.
    """
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
            f"{record.name} did not answer at {_models_url(record)}: "
            f"{type(exc).__name__}: {exc}",
            {"upstream": record.name},
        ) from None
    if response.status_code != 200:
        raise ApiError(
            401,
            "upstream_rejected",
            f"{record.name} refused this server's credentials with "
            f"{response.status_code}: {upstream_message(response.content)}",
            {"upstream": record.name, "upstream_status": response.status_code},
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise ApiError(
            502,
            "upstream_unreachable",
            f"{record.name} answered 200 with a body that is not JSON: {exc}",
            {"upstream": record.name},
        ) from None
    return _read_model_ids(record, payload)


def upstream_message(body: bytes) -> str:
    """The upstream's own sentence, or the bytes it sent instead of one.

    Never a sentence of this server's invention: an operator debugging a key is
    reading the provider's words, and a paraphrase is one more thing between
    them and the answer.
    """
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


# ------------------------------------------------------------------ chatting


def chat_url(record: UpstreamRecord) -> str:
    if record.name == "anthropic":
        return f"{ANTHROPIC_BASE}/v1/messages"
    if record.name == "openai":
        return f"{OPENAI_BASE}/v1/chat/completions"
    return f"{record.url}/v1/chat/completions"


def chat_headers(record: UpstreamRecord) -> dict[str, str]:
    return {"Content-Type": "application/json", **auth_headers(record)}


def _anthropic_tool(response_format: Any) -> dict[str, Any] | None:
    """A `response_format` JSON schema, as the forced tool Anthropic answers with.

    Anthropic has no `response_format`. What it has is tool use, and a tool with
    a forced choice is guided decoding wearing another name: the model must
    produce an argument object matching the schema. That is how `analysis` — the
    one act that sends a schema (CLIENT-SURFACES.md section 6.2) — gets a
    structured answer out of a cloud model.

    None when the request asked for no schema, which includes
    `{"type": "text"}` and `{"type": "json_object"}`: a bare json_object with no
    schema has nothing to force a tool WITH, and inventing an empty schema would
    make the model answer a question nobody asked.
    """
    if not isinstance(response_format, dict):
        return None
    if response_format.get("type") != "json_schema":
        return None
    spec = response_format.get("json_schema")
    if not isinstance(spec, dict):
        return None
    schema = spec.get("schema")
    if not isinstance(schema, dict):
        return None
    return {
        "name": ANTHROPIC_JSON_TOOL,
        "description": (
            "Return the answer as this object. Every field is required and "
            "nothing outside the schema is read."
        ),
        "input_schema": schema,
    }


def _anthropic_messages(messages: Any) -> tuple[list[Any], str | None]:
    """OpenAI `messages` split into Anthropic's `messages` and its `system`.

    Anthropic does not take a `system` ROLE; it takes a top-level `system`
    string. Every leading system message is lifted out and joined with blank
    lines — leading, because a system turn in the middle of a conversation is
    not something either app sends and guessing what it would mean is worse than
    leaving it where it is for Anthropic to refuse by name.
    """
    if not isinstance(messages, list):
        return [], None
    system_parts: list[str] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "system":
            break
        content = message.get("content")
        if isinstance(content, str):
            system_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    system_parts.append(part["text"])
        index += 1
    rest = messages[index:]
    system = "\n\n".join(system_parts) if system_parts else None
    return rest, system


#: The OpenAI knobs that survive the hop, per upstream. Anthropic's list is
#: short because its body is a different document; the other two speak OpenAI
#: and their bodies pass through with the one substitution and the one deletion.
_ANTHROPIC_PASSTHROUGH: tuple[str, ...] = ("temperature", "top_p", "top_k")


@dataclass(frozen=True)
class Forwarded:
    """One chat body translated for one upstream, and the audit of the change."""

    body: bytes
    #: `X-Crucible-Sampling`'s map, all six keys, every time (PHASE2 section 9).
    sources: dict[str, str]
    #: True when the upstream answers in its own shape and the reply has to be
    #: translated back. Only Anthropic does.
    translate_reply: bool


#: A fourth source, beside `request`, `manifest` and `engine`: the request stated
#: it and this server did NOT forward it, because the upstream does not take it.
#: `thinking` travels in `chat_template_kwargs` and none of the three reads that
#: table, so it is dropped and said (PHASE15-HOST.md section 3.4).
SOURCE_DROPPED = "dropped"

#: And a fifth, for the one gap this server fills on a hop: Anthropic requires
#: `max_tokens`. The number is in the string because a reader holding one
#: response must be able to see what was sent.
SOURCE_UPSTREAM_DEFAULT = f"upstream default {ANTHROPIC_MAX_TOKENS_DEFAULT}"


def _sources(body: dict[str, Any], *, filled_max_tokens: bool) -> dict[str, str]:
    """Where each of the six knobs' effective value came from, for an upstream.

    There is no manifest on this path — Crucible has no file describing somebody
    else's weights — so `manifest` never appears. What appears instead is
    `dropped` for a `thinking` this server refused to forward, and
    `upstream default 4096` for the one value it supplied.
    """
    from .manifests import DEFAULTS_KEYS, DEFAULTS_WIRE_KEYS
    from .sampling import SOURCE_ENGINE, SOURCE_REQUEST, TEMPLATE_KWARGS, THINKING_KEY

    sources: dict[str, str] = {}
    for key in DEFAULTS_WIRE_KEYS:
        if key in body:
            sources[key] = SOURCE_REQUEST
        else:
            sources[key] = SOURCE_ENGINE
    if filled_max_tokens and "max_tokens" not in body:
        sources["max_tokens"] = SOURCE_UPSTREAM_DEFAULT
    kwargs = body.get(TEMPLATE_KWARGS)
    if isinstance(kwargs, dict) and THINKING_KEY in kwargs:
        sources["thinking"] = SOURCE_DROPPED
    else:
        sources["thinking"] = SOURCE_ENGINE
    return {key: sources[key] for key in DEFAULTS_KEYS}


def forward_body(name: str, model_id: str, body: dict[str, Any]) -> Forwarded:
    """The caller's chat body, as this upstream reads it.

    OpenAI and Ollama speak OpenAI, so the document goes through with two
    changes: `model` becomes the id without this server's prefix, and
    `chat_template_kwargs` is removed. The second is the whole of "thinking is
    dropped for upstreams that do not know it" — it is the table `thinking`
    travels in, and none of the three reads it.

    Anthropic is a different document and is built rather than edited.
    """
    from .sampling import TEMPLATE_KWARGS

    if name != "anthropic":
        document = {k: v for k, v in body.items() if k != TEMPLATE_KWARGS}
        document["model"] = model_id
        return Forwarded(
            body=json.dumps(document).encode("utf-8"),
            sources=_sources(body, filled_max_tokens=False),
            translate_reply=False,
        )

    messages, system = _anthropic_messages(body.get("messages"))
    document: dict[str, Any] = {"model": model_id, "messages": messages}
    if system is not None:
        document["system"] = system
    max_tokens = body.get("max_tokens")
    document["max_tokens"] = (
        max_tokens if isinstance(max_tokens, int) else ANTHROPIC_MAX_TOKENS_DEFAULT
    )
    for key in _ANTHROPIC_PASSTHROUGH:
        if key in body:
            document[key] = body[key]
    stop = body.get("stop")
    if isinstance(stop, str):
        document["stop_sequences"] = [stop]
    elif isinstance(stop, list):
        document["stop_sequences"] = stop
    if body.get("stream") is True:
        document["stream"] = True
    tool = _anthropic_tool(body.get("response_format"))
    if tool is not None:
        document["tools"] = [tool]
        document["tool_choice"] = {"type": "tool", "name": ANTHROPIC_JSON_TOOL}
    return Forwarded(
        body=json.dumps(document).encode("utf-8"),
        sources=_sources(body, filled_max_tokens=True),
        translate_reply=True,
    )


#: Anthropic's reasons a generation ended, in OpenAI's vocabulary. `tool_use`
#: becomes `stop` because the tool was FORCED by this server to carry a JSON
#: answer — a caller that asked for a schema got what it asked for, and telling
#: it the model "called a tool" would describe a translation it never made.
_STOP_REASON: dict[str, str] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "stop",
    "max_tokens": "length",
    "refusal": "content_filter",
}


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def anthropic_to_openai(payload: dict[str, Any], model: str) -> dict[str, Any]:
    """One Anthropic message, in the completion shape the caller expects.

    The caller sent an OpenAI request to an OpenAI door and must read an OpenAI
    answer; that the hop went somewhere else is the operator's routing decision
    and not a protocol the client has to learn. `model` is Crucible's own
    `<upstream>/<id>`, which is what was asked for, for the same reason
    `_restore_model_id` puts the local id back.

    A forced tool's `input` is serialised into `content`, because that is where a
    caller that sent a `response_format` schema reads its JSON.
    """
    text_parts: list[str] = []
    for block in payload.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            text_parts.append(json.dumps(block.get("input")))
    usage = payload.get("usage") or {}
    prompt_tokens = usage.get("input_tokens")
    completion_tokens = usage.get("output_tokens")
    return {
        "id": payload.get("id") or _completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(text_parts)},
                "finish_reason": _STOP_REASON.get(
                    payload.get("stop_reason") or "", "stop"
                ),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (
                None
                if prompt_tokens is None or completion_tokens is None
                else prompt_tokens + completion_tokens
            ),
        },
    }


class AnthropicStreamTranslator:
    """Anthropic's SSE events, re-emitted as OpenAI chunks.

    Anthropic frames a stream as a message envelope with content blocks inside
    it; OpenAI frames it as a list of deltas and a `[DONE]`. The mapping is
    small and total:

    | Anthropic | OpenAI chunk |
    |---|---|
    | `message_start` | a chunk with `delta: {"role": "assistant"}` |
    | `content_block_delta` `text_delta` | `delta: {"content": <text>}` |
    | `content_block_delta` `input_json_delta` | the same — a forced tool's JSON IS the content |
    | `message_delta` with a `stop_reason` | a chunk with `finish_reason` |
    | `message_stop` | `data: [DONE]` |
    | `ping`, `content_block_start/stop` | nothing; they carry no token |
    | `error` | one `data:` frame carrying the upstream's own error object |

    STATEFUL, because `id` and `model` arrive in `message_start` and every later
    chunk has to repeat them. A stream that never sent one still emits chunks
    (with an id of this server's minting), because a caller reading tokens
    should not be made to care which frame the envelope was in.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self._id = _completion_id()
        self._buffer = b""
        self._done = False

    def _chunk(self, delta: dict[str, Any], finish_reason: str | None) -> bytes:
        payload = {
            "id": self._id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }
        return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"

    def _translate(self, event: dict[str, Any]) -> Iterator[bytes]:
        kind = event.get("type")
        if kind == "message_start":
            message = event.get("message")
            if isinstance(message, dict) and isinstance(message.get("id"), str):
                self._id = message["id"]
            yield self._chunk({"role": "assistant"}, None)
        elif kind == "content_block_delta":
            delta = event.get("delta")
            if not isinstance(delta, dict):
                return
            text = delta.get("text")
            if isinstance(text, str):
                yield self._chunk({"content": text}, None)
                return
            partial = delta.get("partial_json")
            if isinstance(partial, str):
                yield self._chunk({"content": partial}, None)
        elif kind == "message_delta":
            delta = event.get("delta")
            reason = delta.get("stop_reason") if isinstance(delta, dict) else None
            if isinstance(reason, str):
                yield self._chunk({}, _STOP_REASON.get(reason, "stop"))
        elif kind == "message_stop":
            self._done = True
            yield b"data: [DONE]\n\n"
        elif kind == "error":
            # Mid-stream there is nowhere to raise, and the caller is owed the
            # upstream's own words rather than a truncated stream that looks
            # like a finished answer (ARCHITECTURE.md R3).
            yield b"data: " + json.dumps({"error": event.get("error")}).encode(
                "utf-8"
            ) + b"\n\n"

    def feed(self, chunk: bytes) -> Iterator[bytes]:
        """Whatever complete OpenAI frames this chunk of Anthropic SSE completes."""
        self._buffer += chunk
        while b"\n\n" in self._buffer:
            frame, self._buffer = self._buffer.split(b"\n\n", 1)
            for line in frame.split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                payload = line[len(b"data:"):].strip()
                if payload == b"" or payload == b"[DONE]":
                    continue
                try:
                    event = json.loads(payload)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(event, dict):
                    yield from self._translate(event)

    def finish(self) -> Iterator[bytes]:
        """The `[DONE]` an upstream that stopped mid-envelope never sent.

        Not a repair of a broken stream: a caller's SSE reader waits for
        `[DONE]` and a relay that simply closed would leave it waiting on a
        socket that is already shut. What it did receive is exactly what
        arrived; this only closes the frame.
        """
        if not self._done:
            yield b"data: [DONE]\n\n"


__all__ = [
    "ANTHROPIC_BASE",
    "ANTHROPIC_JSON_TOOL",
    "ANTHROPIC_MAX_TOKENS_DEFAULT",
    "ANTHROPIC_VERSION",
    "AnthropicStreamTranslator",
    "Forwarded",
    "OPENAI_BASE",
    "SOURCE_DROPPED",
    "SOURCE_UPSTREAM_DEFAULT",
    "UPSTREAM_FIELD",
    "UPSTREAM_NAMES",
    "UpstreamRecord",
    "anthropic_to_openai",
    "auth_headers",
    "blank",
    "chat_headers",
    "chat_url",
    "forward_body",
    "list_models",
    "record_from_patch",
    "require_name",
    "require_upstream_model",
    "settings_entry",
    "split_model",
]
