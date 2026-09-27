from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .errors import ApiError
from .manifests import DEFAULTS_KEYS, DEFAULTS_WIRE_KEYS, ModelDefaults

SAMPLING_HEADER = "X-Crucible-Sampling"

SOURCE_REQUEST = "request"
SOURCE_MANIFEST = "manifest"
SOURCE_ENGINE = "engine"

TEMPLATE_KWARGS = "chat_template_kwargs"
THINKING_KEY = "enable_thinking"


@dataclass(frozen=True)
class Applied:
    body: dict[str, Any]
    sources: dict[str, str]
    changed: bool

    def header(self) -> str:
        return json.dumps(self.sources, separators=(",", ":"), sort_keys=True)

    def from_manifest(self) -> list[str]:
        return [
            key
            for key, source in self.sources.items()
            if source == SOURCE_MANIFEST
        ]


def _template_kwargs(body: dict[str, Any]) -> dict[str, Any] | None:
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
        additions[TEMPLATE_KWARGS] = {
            **(kwargs or {}),
            THINKING_KEY: defaults.thinking,
        }
        sources["thinking"] = SOURCE_MANIFEST

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
