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

OLLAMA IS SPOKEN NATIVELY (PHASE15-HOST.md section 3.4a)
--------------------------------------------------------
Until 2026-09-23 the `ollama` upstream was forwarded to Ollama's OpenAI shim,
`/v1/chat/completions`. That shim has no field for `num_ctx`, so every
`ollama/<id>` chat ran at Ollama's default context (4096 unless the host set
`OLLAMA_CONTEXT_LENGTH`), and a longer prompt was truncated from the FRONT
with a 200 and no field saying so (FITS-AND-THE-CARD.md 6.1 measured it).
`thinking` was dropped the same way. So Ollama is now reached at its native
`POST /api/chat`, translated in both directions the way Anthropic is, and
**every Ollama chat states `options.num_ctx`**: the request's
`context_tokens`, else the tag's own (`resolve_ollama_context`). Sending
nothing is the bug, so nothing is never sent.

(The ollama context lookup DOES retry — `/api/show` and `/api/tags` are
unbilled reads of the operator's own box, and a blip there is weather. The
chat itself still never retries.)
"""

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
            f"nothing answered at {_models_url(record)}, which is where this "
            f"server reaches {record.name}: {type(exc).__name__}: {exc}",
            {"upstream": record.name, "url": _models_url(record)},
        ) from None
    if response.status_code != 200:
        # **502, NOT 401**, and the difference is who is being talked about.
        # A 401 from a Crucible route means THIS server refused THIS client's
        # bearer token, and a client that saw one here would show a person
        # "your Crucible token is wrong" about a key the upstream rejected.
        # The chat door already answers 502 for every non-2xx but 429 (7.2);
        # this is the same decision at the other door, so one code has one
        # status.
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
    # NATIVE, not the `/v1/chat/completions` shim: the shim has no field for
    # `num_ctx` (section 3.4a), and a chat that cannot state its context runs
    # at 4096 and is silently truncated.
    return f"{record.url}/api/chat"


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
#: short because its body is a different document; OpenAI speaks OpenAI and its
#: body passes through with the one substitution and the two deletions.
_ANTHROPIC_PASSTHROUGH: tuple[str, ...] = ("temperature", "top_p", "top_k")

#: The three wire dialects a reply can arrive in. `openai` is relayed as it
#: stands (the `model` put back); the other two are translated into it.
DIALECT_OPENAI = "openai"
DIALECT_ANTHROPIC = "anthropic"
DIALECT_OLLAMA = "ollama"

#: How a client states the context window an `ollama/<id>` chat runs at — a
#: top-level integer on the OpenAI-shaped chat body. Crucible's word, not
#: Ollama's: it is the same word `GET /v1/capability?context_tokens=` uses for
#: the same quantity (the tokens of prompt plus answer one request needs), and
#: a client sizing work asks both questions in one vocabulary. Sent to Ollama as
#: `options.num_ctx`. Section 3.4a.
CONTEXT_FIELD = "context_tokens"

#: The response header that says which context an upstream chat ran at, and
#: where the number came from. Compact JSON, `{"num_ctx": N, "source": S}`,
#: on every upstream response — a header for `X-Crucible-Sampling`'s reason: a
#: streamed answer has nowhere else to carry it.
CONTEXT_HEADER = "X-Crucible-Context"

#: Where `num_ctx` came from. `request`: the client's `context_tokens`.
#: `modelfile`: the tag's own `PARAMETER num_ctx` (what `ollama show` prints
#: under Parameters — the tag author's statement of the window this tag runs
#: at, e.g. Owen's `qwen3.8:27b-24g` at 98304). `model`: the weights' trained
#: maximum, `model_info.<arch>.context_length`, for a tag that states none.
#: `dropped`: the client stated one and this upstream has no such knob (a
#: hosted model's window is its provider's). `upstream`: nothing stated, and
#: the window is the provider's.
CONTEXT_SOURCE_REQUEST = "request"
CONTEXT_SOURCE_MODELFILE = "modelfile"
CONTEXT_SOURCE_MODEL = "model"
CONTEXT_SOURCE_DROPPED = "dropped"
CONTEXT_SOURCE_UPSTREAM = "upstream"


@dataclass(frozen=True)
class OllamaContext:
    """The `num_ctx` one Ollama chat is sent, and where it came from."""

    num_ctx: int
    #: `request`, `modelfile` or `model`.
    source: str


@dataclass(frozen=True)
class Forwarded:
    """One chat body translated for one upstream, and the audit of the change."""

    body: bytes
    #: `X-Crucible-Sampling`'s map, all six keys, every time (PHASE2 section 9).
    sources: dict[str, str]
    #: Which wire shape the reply arrives in: `openai` (relayed), `anthropic`
    #: or `ollama` (translated back into OpenAI's).
    dialect: str
    #: `X-Crucible-Context`'s value (section 3.4a).
    context: dict[str, Any]
    #: OpenAI's `stream_options.include_usage`, which only a translating relay
    #: has to honour: an OpenAI upstream reads it itself.
    include_usage: bool = False


#: A fourth source, beside `request`, `manifest` and `engine`: the request stated
#: it and this server did NOT forward it, because the upstream does not take it.
#: `thinking` travels in `chat_template_kwargs` and neither Anthropic nor OpenAI
#: reads that table, so it is dropped and said (PHASE15-HOST.md section 3.4).
#: Ollama DOES take it, as `think` (section 3.4a).
SOURCE_DROPPED = "dropped"

#: And a fifth, for the one gap this server fills on a hop: Anthropic requires
#: `max_tokens`. The number is in the string because a reader holding one
#: response must be able to see what was sent.
SOURCE_UPSTREAM_DEFAULT = f"upstream default {ANTHROPIC_MAX_TOKENS_DEFAULT}"


def _sources(
    body: dict[str, Any], *, filled_max_tokens: bool, forwards_thinking: bool = False
) -> dict[str, str]:
    """Where each of the six knobs' effective value came from, for an upstream.

    There is no manifest on this path — Crucible has no file describing somebody
    else's weights — so `manifest` never appears. What appears instead is
    `dropped` for a `thinking` this server refused to forward, and
    `upstream default 4096` for the one value it supplied. Ollama forwards
    `thinking` (as `think`), so there a stated one is `request`.
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
        sources["thinking"] = SOURCE_REQUEST if forwards_thinking else SOURCE_DROPPED
    else:
        sources["thinking"] = SOURCE_ENGINE
    return {key: sources[key] for key in DEFAULTS_KEYS}


def stated_context(body: dict[str, Any]) -> int | None:
    """The request's `context_tokens`, or None — refused by name if malformed.

    Read on every upstream, not only Ollama's, so a client's mistake is the same
    400 whichever route its class happens to be on today.
    """
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
    """`X-Crucible-Context` for Anthropic and OpenAI: their window is theirs."""
    stated = stated_context(body)
    return {
        "num_ctx": None,
        "source": CONTEXT_SOURCE_UPSTREAM if stated is None else CONTEXT_SOURCE_DROPPED,
    }


def forward_body(name: str, model_id: str, body: dict[str, Any]) -> Forwarded:
    """The caller's chat body, as Anthropic or OpenAI reads it.

    OpenAI speaks OpenAI, so the document goes through with three changes:
    `model` becomes the id without this server's prefix, and
    `chat_template_kwargs` and `context_tokens` are removed. `thinking` travels
    in the first and OpenAI does not read it; the second is this server's field
    and OpenAI would refuse it as an unknown argument. Both are said in the
    audit headers as `dropped`.

    Anthropic is a different document and is built rather than edited.

    Ollama is NOT built here: its body needs a context this server has to ask
    Ollama for, which is I/O. `forward_ollama` is its door.
    """
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
        dialect=DIALECT_ANTHROPIC,
        context=context,
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


# ------------------------------------------------------- ollama, natively
#
# PHASE15-HOST.md section 3.4a. Everything below is the ONE place Ollama's own
# wire is spoken: the context lookup (`/api/tags`, `/api/show`), the body
# (`/api/chat`), the reply and the NDJSON stream.

#: How many times one context lookup is asked before it is refused, and the
#: waits between the asks. `/api/tags` and `/api/show` are unbilled reads of
#: the operator's own box — a restart, a model mid-load holding Ollama's lock,
#: a dropped socket are weather, and weather gets a budget. Worst case one
#: lookup is 3 x 10 s of timeout plus 2.5 s of waiting; then it is refused
#: `upstream_context_unknown`, never answered with a guess.
OLLAMA_LOOKUP_ATTEMPTS = 3
OLLAMA_LOOKUP_BACKOFF_SECONDS: tuple[float, ...] = (0.5, 2.0)
OLLAMA_LOOKUP_TIMEOUT_SECONDS = 10.0


class OllamaContexts:
    """What each Ollama tag's own context is, remembered PER DIGEST.

    THE INVALIDATION RULE: an entry is keyed by (Ollama address, tag) and is
    good only while `/api/tags` reports the SAME digest for that tag. Asked
    on every chat that does not state `context_tokens` — one small local GET —
    because the thing that changes a tag's context is Owen running
    `ollama create qwen3.8:27b-24g` with a new `PARAMETER num_ctx`, which keeps
    the name and changes the digest. A per-process cache would send the old
    number until Crucible restarted; a digest cannot be stale.

    In memory: a restart forgets, and the first chat after it asks `/api/show`
    once per tag.
    """

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
    """The name `/api/tags` lists a model under: Ollama's own `:latest` rule."""
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
    """One lookup against Ollama, within `OLLAMA_LOOKUP_ATTEMPTS`.

    Weather — no answer, a timeout, a 5xx — is asked again after a stated
    wait, and said in the server log by name each time. A 4xx is Ollama's
    answer about the REQUEST (a tag that is not pulled is a 404) and is passed
    back at once as `upstream_rejected` with Ollama's own words: asking again
    would get the same answer.
    """
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
        # Nothing answered at all: the same fact, and so the same name, the
        # chat itself would have met (`upstream_unreachable`). Stating
        # `context_tokens` would not help — the chat goes to the same address.
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
    """The tag's own context out of one `/api/show` answer.

    The tag's `PARAMETER num_ctx` first, because it is the more specific
    statement: the tag author wrote down the window this tag runs at, knowing
    the card (`qwen3.8:27b-24g`'s 98304 exists because the trained 262144 does
    not fit 24 GB). Sending the trained maximum over it would override that
    decision and push the KV cache into system RAM. A tag that states none runs
    at what its weights were trained for, `model_info.<arch>.context_length`.
    """
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
    """The `num_ctx` this chat is sent. Never absent: absent IS the bug.

    The request's `context_tokens` when it states one — the caller knows how
    long its prompt is and this server does not (it has no tokenizer for
    somebody else's weights). Otherwise the tag's own (`_context_from_show`),
    remembered per digest (`OllamaContexts`). Never 4096, never a number this
    server made up.
    """
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
    # A tag `/api/tags` did not list is still ASKED about rather than refused
    # here: Ollama's `/api/show` is the owner of "is that a model", and its own
    # 404 is the sentence the caller should read.
    shown = await _ollama_read(
        client, record, model_id, "POST", "/api/show", {"model": model_id}
    )
    context = _context_from_show(record, model_id, shown)
    if digest is not None:
        contexts.put(record.url or "", tag, digest, context)
    return context


#: OpenAI sampling fields and the `options` key Ollama reads each under.
_OLLAMA_OPTIONS: dict[str, str] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "seed": "seed",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    # vLLM's name for it, which the local door's manifests use.
    "repetition_penalty": "repeat_penalty",
}

#: Every top-level field the Ollama translation READS. Anything else is
#: refused by name (`upstream_field_unsupported`), because a field this
#: translation silently left behind is exactly how `num_ctx` was lost for a
#: month. `user` is OpenAI's end-user tag for abuse monitoring: accepted and
#: not sent, since it changes nothing about the answer.
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
    return ApiError(
        400,
        "upstream_field_unsupported",
        f"the ollama upstream does not carry {fields}: {why}",
        {"upstream": "ollama", "fields": fields, "carried": sorted(_OLLAMA_READS)},
    )


def _ollama_message(index: int, message: Any) -> dict[str, Any]:
    """One OpenAI message as Ollama reads it: text in `content`, pictures in `images`."""
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
    """The caller's OpenAI chat body as Ollama's native `/api/chat` document,
    every refusal made — all but `options.num_ctx`, which `forward_ollama`
    puts first once it is known.

    Built, not edited, the way Anthropic's is. `stream` is ALWAYS stated,
    because Ollama's default is to stream and a non-streamed request that left
    it out would get NDJSON back.
    """
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
    """The caller's chat body as Ollama's `/api/chat` reads it, context and all.

    The body is checked BEFORE the context is looked up, so a request this
    translation cannot carry costs no round trip to Ollama.
    """
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


#: Ollama's `done_reason`, in OpenAI's words. A reason not in this table is
#: passed through as Ollama said it rather than rounded to `stop`: a finish
#: this server does not recognise is not evidence the answer is whole.
_DONE_REASON: dict[str, str] = {"stop": "stop", "length": "length"}


def _ollama_usage(payload: dict[str, Any]) -> dict[str, Any]:
    """OpenAI `usage` from Ollama's counts.

    `prompt_eval_count` is what Ollama EVALUATED, not what was sent: a prompt
    longer than `num_ctx` is cut before evaluation and this count is of what
    survived (section 3.4a says what that does and does not let a caller see).
    """
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
    """One Ollama `/api/chat` answer, as the OpenAI completion the caller reads.

    Ollama's `message.thinking` becomes `message.reasoning` — the field the
    local engines already answer in and `@crucible/client` already reads
    (`client.ts`'s reasoning-without-content check).
    """
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
    """Ollama's NDJSON stream, re-emitted as OpenAI SSE chunks.

    | Ollama line | OpenAI |
    |---|---|
    | the first line | a chunk with `delta: {"role": "assistant"}` first |
    | `message.thinking` | `delta: {"reasoning": <text>}` |
    | `message.content` | `delta: {"content": <text>}` |
    | `done: true` | a chunk with `finish_reason`, a usage chunk if asked, `[DONE]` |
    | `{"error": ...}` | one `data:` frame carrying `{"error": {"message": ...}}` |

    A STREAM THAT ENDS WITHOUT `done: true` GETS NO `[DONE]`. Unlike
    Anthropic's translator, this one does not close the frame for an upstream
    that stopped: Ollama's last line is the only evidence the answer is whole,
    and a `[DONE]` here would make a cut-off answer read as a finished one. The
    caller gets an error frame naming it and a stream with no terminator, which
    `@crucible/client` already throws on.
    """

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
        """Whatever complete OpenAI frames this chunk of Ollama NDJSON completes."""
        self._buffer += chunk
        yield from self._lines(final=False)

    def finish(self) -> Iterator[bytes]:
        """The last line, if it had no newline — and a truncation said by name."""
        yield from self._lines(final=True)
        if not self._done and not self._failed:
            yield self._frame(
                {"error": {"message": "ollama's stream ended before it said done; "
                           "the answer is truncated", "upstream": "ollama"}}
            )


__all__ = [
    "ANTHROPIC_BASE",
    "ANTHROPIC_JSON_TOOL",
    "ANTHROPIC_MAX_TOKENS_DEFAULT",
    "ANTHROPIC_VERSION",
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
    "UPSTREAM_FIELD",
    "UPSTREAM_NAMES",
    "UpstreamRecord",
    "anthropic_to_openai",
    "auth_headers",
    "blank",
    "chat_headers",
    "chat_url",
    "forward_body",
    "forward_ollama",
    "list_models",
    "ollama_to_openai",
    "resolve_ollama_context",
    "stated_context",
    "record_from_patch",
    "require_name",
    "require_upstream_model",
    "settings_entry",
    "split_model",
]
