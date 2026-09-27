from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import ApiError

UPSTREAM_NAMES: tuple[str, ...] = ("anthropic", "openai", "ollama")

UPSTREAM_FIELD: dict[str, str] = {
    "anthropic": "key",
    "openai": "key",
    "ollama": "url",
}

UPSTREAM_DISPLAY: dict[str, str] = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "ollama": "Ollama",
}


@dataclass(frozen=True)
class UpstreamRecord:

    name: str
    key: str | None = None
    url: str | None = None

    @property
    def key_hint(self) -> str | None:
        if self.key is None:
            return None
        return f"…{self.key[-4:]}"


def blank(name: str) -> dict[str, Any]:
    if UPSTREAM_FIELD[name] == "key":
        return {"configured": False, "key_hint": None}
    return {"configured": False, "url": None}


def settings_entry(record: UpstreamRecord) -> dict[str, Any]:
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
    if "/" not in model:
        return None
    name, _, rest = model.partition("/")
    return name, rest


def require_upstream_model(model: str) -> tuple[str, str]:
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
