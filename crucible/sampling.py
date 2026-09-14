"""Per-model sampling defaults, applied on the chat door.

PHASE2-LLM.md section 9. One rule, three sources, and a header that says which
one each effective value came from:

| the request | the manifest | what the engine gets | source |
|---|---|---|---|
| states it | anything | **the request's value** | `request` |
| omits it | states it | **the manifest's value** | `manifest` |
| omits it | omits it | nothing is sent | `engine` |

Why the server applies them at all
----------------------------------
Because the alternative is every client carrying the same numbers. Foundry sends
`thinking: false` on every cleanup call today and BookForge sends it on its own,
and the reason is a property of the *weights* — Qwen3.5 thinks before it answers
and spends a bounded budget entirely on `reasoning`, returning a message with no
`content` at all. That is one fact with two owners in two repos (ARCHITECTURE.md
R1), and the owner it belongs to is the thing that knows which weights are
loaded. DESIGN.md section 3.1 already ruled the general case: *"everything that
tunes an engine to a model lives in Crucible's own configuration, not on the
wire."*

Why the wire fields stay
------------------------
They are not retired and this module does not touch a request that states one.
Foundry sends `temperature` and `max_tokens` on every call it makes; a proxy
that started overriding them would change the output of a shipping app on the
day it was deployed. **The manifest fills gaps; it never wins an argument.**

Why the source is REPORTED and not merely applied
-------------------------------------------------
A default that cannot be seen is a default nobody can debug. A client that got a
`max_tokens` it did not send has no way to tell a manifest default from an
engine default from a proxy bug, and "why is this answer 600 tokens" then costs
somebody an afternoon reading server source. So every chat response carries
`X-Crucible-Sampling`, a compact JSON object naming the source of all six keys —
on the streamed door too, where a body field could not have gone.

It is a HEADER and not a body field on purpose. PHASE2-LLM.md section 5's rule
is that the proxy is verbatim in both directions except for `model`; adding a
key to somebody's completion would break that, and there is no place to put one
in an SSE stream that a client would not have to learn to skip.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .errors import ApiError
from .manifests import DEFAULTS_KEYS, DEFAULTS_WIRE_KEYS, ModelDefaults

#: The response header that names where each effective value came from.
SAMPLING_HEADER = "X-Crucible-Sampling"

#: The three answers. `engine` means *nothing was sent for this key*, so what
#: happens is whatever the engine does on its own — which is a real, nameable
#: source and not an absence.
SOURCE_REQUEST = "request"
SOURCE_MANIFEST = "manifest"
SOURCE_ENGINE = "engine"

#: Where `thinking` actually travels. Both engines read it per request under
#: this name (PHASE2-LLM.md section 5); it is not a sampling field and has no
#: top-level spelling.
TEMPLATE_KWARGS = "chat_template_kwargs"
THINKING_KEY = "enable_thinking"


@dataclass(frozen=True)
class Applied:
    """A chat body with the manifest's gaps filled, and the audit of what filled them."""

    #: The body to forward. The SAME object as the input when nothing changed.
    body: dict[str, Any]
    #: One entry per key in `DEFAULTS_KEYS`, always all of them.
    sources: dict[str, str]
    #: True when this module put something in the body. When it is False the
    #: caller may forward the client's original bytes untouched, which is what
    #: keeps `response_format.json_schema.schema` a grammar nobody re-encoded.
    changed: bool

    def header(self) -> str:
        """The `X-Crucible-Sampling` value: compact JSON, stable key order."""
        return json.dumps(self.sources, separators=(",", ":"), sort_keys=True)

    def from_manifest(self) -> list[str]:
        """The keys this server filled in, for a log line or a test."""
        return [
            key
            for key, source in self.sources.items()
            if source == SOURCE_MANIFEST
        ]


def _template_kwargs(body: dict[str, Any]) -> dict[str, Any] | None:
    """The request's `chat_template_kwargs`, or a refusal if it is not an object.

    This is the one field of somebody else's body that has to be read INSIDE,
    because `thinking` lives in it. A non-object here would be refused by the
    engine anyway; refusing it by name at the door is the difference between a
    message that says which field is wrong and a 400 from vLLM about a pydantic
    model the caller has never heard of.
    """
    value = body.get(TEMPLATE_KWARGS)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ApiError(
            400,
            "invalid_request",
            f"{TEMPLATE_KWARGS} must be an object, got "
            f"{type(value).__name__}; it is where {THINKING_KEY} travels and this "
            "server has to read it to know whether the request stated one",
        )
    return value


def apply_defaults(body: dict[str, Any], defaults: ModelDefaults) -> Applied:
    """Fill the gaps this manifest has an answer for, and record every source.

    **A key present in the request is stated**, whatever its value — including
    an explicit `null`. That is deliberate: a client that wrote the key made a
    decision, and a server that treated `"temperature": null` as an absence
    would answer a stated request with a different number than it was asked for.
    Whether the engine likes the value is the engine's to say.
    """
    sources: dict[str, str] = {}
    additions: dict[str, Any] = {}

    for key in DEFAULTS_WIRE_KEYS:
        if key in body:
            sources[key] = SOURCE_REQUEST
            continue
        value = getattr(defaults, key)
        if value is None:
            sources[key] = SOURCE_ENGINE
            continue
        additions[key] = value
        sources[key] = SOURCE_MANIFEST

    kwargs = _template_kwargs(body)
    if kwargs is not None and THINKING_KEY in kwargs:
        sources["thinking"] = SOURCE_REQUEST
    elif defaults.thinking is None:
        sources["thinking"] = SOURCE_ENGINE
    else:
        # Merged into whatever else the client put in the table rather than
        # replacing it: `chat_template_kwargs` is the client's dictionary and
        # this server owns exactly one key inside it.
        additions[TEMPLATE_KWARGS] = {
            **(kwargs or {}),
            THINKING_KEY: defaults.thinking,
        }
        sources["thinking"] = SOURCE_MANIFEST

    # Every key, always, in one order. A map whose keys came and went would make
    # a reader guess whether a missing entry meant "engine" or "this build does
    # not report that one".
    ordered = {key: sources[key] for key in DEFAULTS_KEYS}
    if not additions:
        return Applied(body=body, sources=ordered, changed=False)
    return Applied(body={**body, **additions}, sources=ordered, changed=True)


__all__ = [
    "Applied",
    "SAMPLING_HEADER",
    "SOURCE_ENGINE",
    "SOURCE_MANIFEST",
    "SOURCE_REQUEST",
    "TEMPLATE_KWARGS",
    "THINKING_KEY",
    "apply_defaults",
]
