from __future__ import annotations

import threading
from typing import Any, Callable, Mapping

from . import capabilityclasses, classnames, memorybudget, upstreamrecord
from .backend import CardFacts
from .capabilitystore import decide_on, record_of
from .clock import utcnow
from .capabilityrecord import DESKTOP_BASIS_STATED, CapabilityRecord, desktop_reserve_words
from .config import Config, LocalModelRecord, RouteRecord, _advertised, load_config, write_config
from .errors import ApiError, ConfigError
from .upstreamrecord import UPSTREAM_DISPLAY, UPSTREAM_NAMES, UpstreamRecord

HISTORY_LIMIT = 20

PATCH_KEYS: frozenset[str] = frozenset(
    {
        "routes",
        "upstreams",
        "local_models",
        "desktop_allowance_bytes",
        "tailscale_advertise",
        "lan_advertise",
    }
)


class History:

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: list[dict[str, Any]] = []

    def record(
        self, *, act: str | None, client: str | None, changed: list[str]
    ) -> None:
        row = {
            "at": utcnow(),
            "act": act,
            "client": client,
            "changed": changed,
        }
        with self._lock:
            self._rows.append(row)
            del self._rows[:-HISTORY_LIMIT]

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(self._rows))


def local_selection(config: Config, name: str) -> str | None:
    record = config.capability
    if record is None:
        return None
    row = record.row(name)
    if row is None or row.selected == "":
        return None
    return row.selected


def _choices(
    config: Config, installed: Mapping[str, bool]
) -> dict[str, list[dict[str, Any]]]:
    record = config.capability
    if record is None:
        return {}
    budget = memorybudget.available_bytes(
        record.total_bytes, config.desktop_allowance_bytes
    )
    found: dict[str, list[dict[str, Any]]] = {}
    for name in classnames.SELECTABLE_CLASSES:
        entry = capabilityclasses.BY_NAME[name]
        assert entry.candidates is not None
        rows: list[dict[str, Any]] = []
        for candidate in entry.candidates(record.backend_kind):
            if candidate.id not in installed:
                raise ApiError(
                    500,
                    "catalog_incomplete",
                    f"the catalog has no row for {candidate.id!r}, which "
                    f"{name} offers as a candidate; these two read the same "
                    "manifests and must agree",
                    {"capability": name, "model": candidate.id},
                )
            rows.append(
                {
                    "id": candidate.id,
                    "memory_bytes_estimate": candidate.memory_bytes_estimate,
                    "fits": candidate.memory_bytes_estimate <= budget,
                    "installed": installed[candidate.id],
                }
            )
        found[name] = rows
    return found


def document(config: Config, *, installed: Mapping[str, bool]) -> dict[str, Any]:
    routes: dict[str, Any] = {}
    for name in classnames.ROUTABLE_CLASSES:
        model = config.route_model(name)
        if model is None:
            routes[name] = {"route": "local", "model": local_selection(config, name)}
        else:
            routes[name] = {"route": "upstream", "model": model}
    upstreams: dict[str, Any] = {}
    for name in UPSTREAM_NAMES:
        record = config.upstream(name)
        upstreams[name] = (
            upstreamrecord.blank(name)
            if record is None
            else upstreamrecord.settings_entry(record)
        )
    return {
        "local_models": {
            name: config.local_model(name)
            for name in classnames.SELECTABLE_CLASSES
        },
        "local_model_choices": _choices(config, installed),
        "routes": routes,
        "upstreams": upstreams,
        "upstream_labels": dict(UPSTREAM_DISPLAY),
        "desktop_allowance_bytes": config.desktop_allowance_bytes,
        "desktop_allowance_basis": config.desktop_allowance_basis,
        "desktop_reserve": desktop_reserve_words(
            config.desktop_allowance_bytes, config.desktop_allowance_basis
        ),
        "backend_kind": config.backend_kind,
        "tailscale_advertise": list(config.tailscale_advertise),
        "lan_advertise": list(config.lan_advertise),
    }


class Resolved:

    def __init__(self, config: Config) -> None:
        self.upstreams: dict[str, UpstreamRecord] = {
            entry.name: entry for entry in config.upstreams
        }
        self.routes: dict[str, str] = {
            entry.capability: entry.model for entry in config.routes
        }
        self.local_models: dict[str, str] = {
            entry.capability: entry.model for entry in config.local_models
        }
        self.desktop_allowance_bytes = config.desktop_allowance_bytes
        self.desktop_allowance_basis = config.desktop_allowance_basis
        self.desktop_allowance_note = config.desktop_allowance_note
        self.tailscale_advertise = config.tailscale_advertise
        self.lan_advertise = config.lan_advertise
        self.removed: set[str] = set()
        self.changed: list[str] = []
        self.touched_routes = False

    def as_records(
        self,
    ) -> tuple[
        tuple[RouteRecord, ...],
        tuple[UpstreamRecord, ...],
        tuple[LocalModelRecord, ...],
    ]:
        routes = tuple(
            RouteRecord(capability=name, model=self.routes[name])
            for name in classnames.ROUTABLE_CLASSES
            if name in self.routes
        )
        upstreams = tuple(
            self.upstreams[name] for name in UPSTREAM_NAMES if name in self.upstreams
        )
        local_models = tuple(
            LocalModelRecord(capability=name, model=self.local_models[name])
            for name in classnames.SELECTABLE_CLASSES
            if name in self.local_models
        )
        return routes, upstreams, local_models


ROOT_FIELD = "body"


def _require_object(patch: Any, field: str) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise ApiError(
            400,
            "invalid_request",
            f"{field} must be an object, got {type(patch).__name__}",
            {"field": field},
        )
    return patch


def _refuse_unknown_fields(body: dict[str, Any]) -> None:
    unknown = sorted(set(body) - PATCH_KEYS)
    if unknown:
        raise ApiError(
            400,
            "invalid_request",
            f"unknown settings field(s) {unknown}; this door takes "
            f"{sorted(PATCH_KEYS)}",
            {"field": unknown[0], "unknown": unknown},
        )


def _resolve_upstreams(config: Config, resolved: Resolved, value: Any) -> None:
    table = _require_object(value, "upstreams")
    for name in sorted(table):
        field = f"upstreams.{name}"
        upstreamrecord.require_name(name, field)
        _resolve_upstream(resolved, name, table[name], field)


def _resolve_upstream(resolved: Resolved, name: str, value: Any, field: str) -> None:
    if value is None:
        if name in resolved.upstreams:
            del resolved.upstreams[name]
            resolved.removed.add(name)
            resolved.changed.append(f"{field} removed")
        return
    resolved.upstreams[name] = upstreamrecord.record_from_patch(name, value, field)
    resolved.changed.append(f"{field} set")


def _resolve_routes(config: Config, resolved: Resolved, value: Any) -> None:
    table = _require_object(value, "routes")
    resolved.touched_routes = True
    for name in sorted(table):
        _resolve_route(resolved, name, table[name])


def _require_routable(name: str, field: str) -> None:
    if name in classnames.ROUTABLE_CLASSES:
        return
    raise ApiError(
        400,
        "route_not_routable",
        f"{name!r} cannot run anywhere but this server's card. The "
        f"classes a route may name are "
        f"{list(classnames.ROUTABLE_CLASSES)}; every other "
        "class is local and there is no upstream that does its kind "
        "of work",
        {
            "field": field,
            "capability": name,
            "routable": list(classnames.ROUTABLE_CLASSES),
        },
    )


def _require_upstream_model_id(value: str, field: str) -> None:
    upstream_name, _, rest = value.partition("/")
    if upstream_name in UPSTREAM_NAMES and rest != "":
        return
    raise ApiError(
        400,
        "route_bad_model",
        f"{value!r} is not an upstream model id. A route's value is "
        f"`<upstream>/<model>` with the upstream one of "
        f"{list(UPSTREAM_NAMES)}, or the word \"local\"",
        {
            "field": field,
            "model": value,
            "known": list(UPSTREAM_NAMES),
        },
    )


def _resolve_route(resolved: Resolved, name: str, value: Any) -> None:
    field = f"routes.{name}"
    _require_routable(name, field)
    if not isinstance(value, str) or value == "":
        raise ApiError(
            400,
            "invalid_request",
            f"{field} must be \"local\" or an upstream model id, got "
            f"{type(value).__name__}",
            {"field": field},
        )
    if value == "local":
        if name in resolved.routes:
            del resolved.routes[name]
            resolved.changed.append(f"{field} = local")
        return
    _require_upstream_model_id(value, field)
    resolved.routes[name] = value
    resolved.changed.append(f"{field} = {value}")


def _resolve_desktop_allowance(config: Config, resolved: Resolved, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ApiError(
            400,
            "invalid_request",
            "desktop_allowance_bytes must be a non-negative integer number "
            f"of bytes, got {value!r}",
            {"field": "desktop_allowance_bytes"},
        )
    if value == resolved.desktop_allowance_bytes:
        return
    resolved.desktop_allowance_bytes = value
    resolved.desktop_allowance_basis = DESKTOP_BASIS_STATED
    resolved.desktop_allowance_note = f"set in Settings on {utcnow()[:10]}"
    resolved.changed.append(f"desktop_allowance_bytes = {value}")


def _resolve_local_models(config: Config, resolved: Resolved, value: Any) -> None:
    table = _require_object(value, "local_models")
    record = config.capability
    for name in sorted(table):
        _resolve_local_model(record, resolved, name, table[name])


def _require_selectable(name: str, field: str) -> None:
    if name in classnames.SELECTABLE_CLASSES:
        return
    raise ApiError(
        400,
        "local_model_not_selectable",
        f"{name!r} has no local models to choose between. The classes "
        f"that do are "
        f"{list(classnames.SELECTABLE_CLASSES)}",
        {
            "field": field,
            "capability": name,
            "selectable": list(classnames.SELECTABLE_CLASSES),
        },
    )


def _require_decided(
    record: CapabilityRecord | None, name: str, field: str
) -> CapabilityRecord:
    if record is not None:
        return record
    raise ApiError(
        503,
        "capability_undecided",
        "This server has not decided its capability yet, so a local "
        "model cannot be chosen on it. Run `crucible capability "
        "--write` on the host first",
        {"field": field, "capability": name},
    )


def _offered_candidate(
    record: CapabilityRecord, name: str, value: str, field: str
) -> Any:
    entry = capabilityclasses.BY_NAME[name]
    assert entry.candidates is not None
    offered = entry.candidates(record.backend_kind)
    picked = next((c for c in offered if c.id == value), None)
    if picked is not None:
        return picked
    raise ApiError(
        400,
        "local_model_unknown",
        f"{value!r} is not among the {len(offered)} {entry.noun} "
        f"this build ships for {name} on {record.backend_kind}",
        {
            "field": field,
            "capability": name,
            "model": value,
            "choices": [c.id for c in offered],
        },
    )


def _require_fits(
    record: CapabilityRecord, resolved: Resolved, picked: Any, name: str, field: str
) -> None:
    value = picked.id
    budget = memorybudget.available_bytes(
        record.total_bytes, resolved.desktop_allowance_bytes
    )
    if picked.memory_bytes_estimate <= budget:
        return
    shortfall = picked.memory_bytes_estimate - budget
    raise ApiError(
        409,
        "local_model_does_not_fit",
        f"{value} needs "
        f"{picked.memory_bytes_estimate / 2**30:.1f} GiB and there "
        f"is {budget / 2**30:.1f} GiB available "
        f"({record.total_bytes / 2**30:.1f} GiB less a "
        f"{resolved.desktop_allowance_bytes / 2**30:.1f} GiB desktop "
        f"allowance) — short by {shortfall / 2**30:.1f} GiB",
        {
            "field": field,
            "capability": name,
            "model": value,
            "memory_bytes_estimate": picked.memory_bytes_estimate,
            "available_bytes": budget,
            "shortfall_bytes": shortfall,
        },
    )


def _resolve_local_model(
    record: CapabilityRecord | None, resolved: Resolved, name: str, value: Any
) -> None:
    field = f"local_models.{name}"
    _require_selectable(name, field)
    if value is None:
        if resolved.local_models.pop(name, None) is not None:
            resolved.changed.append(f"local_models.{name} = automatic")
        return
    if not isinstance(value, str) or value == "":
        raise ApiError(
            400,
            "invalid_request",
            f"{field} must be a model id, or null for automatic, got "
            f"{type(value).__name__}",
            {"field": field},
        )
    decided = _require_decided(record, name, field)
    picked = _offered_candidate(decided, name, value, field)
    _require_fits(decided, resolved, picked, name, field)
    if resolved.local_models.get(name) != value:
        resolved.local_models[name] = value
        resolved.changed.append(f"local_models.{name} = {value}")


def _advertise_list(value: Any, field: str) -> tuple[str, ...]:
    try:
        return _advertised({"server": {"advertise": value}})
    except ConfigError as exc:
        raise ApiError(400, "invalid_request", str(exc), {"field": field}) from exc


def _resolve_tailscale_advertise(config: Config, resolved: Resolved, value: Any) -> None:
    resolved.tailscale_advertise = _advertise_list(value, "tailscale_advertise")
    if resolved.tailscale_advertise != config.tailscale_advertise:
        resolved.changed.append("tailscale_advertise")


def _resolve_lan_advertise(config: Config, resolved: Resolved, value: Any) -> None:
    resolved.lan_advertise = _advertise_list(value, "lan_advertise")
    if resolved.lan_advertise != config.lan_advertise:
        resolved.changed.append("lan_advertise")


SectionResolver = Callable[[Config, Resolved, Any], None]

SECTION_RESOLVERS: tuple[tuple[str, SectionResolver], ...] = (
    ("upstreams", _resolve_upstreams),
    ("routes", _resolve_routes),
    ("desktop_allowance_bytes", _resolve_desktop_allowance),
    ("local_models", _resolve_local_models),
    ("tailscale_advertise", _resolve_tailscale_advertise),
    ("lan_advertise", _resolve_lan_advertise),
)


def resolve(config: Config, patch: Any) -> Resolved:
    body = _require_object(patch, ROOT_FIELD)
    _refuse_unknown_fields(body)
    resolved = Resolved(config)
    for key, resolver in SECTION_RESOLVERS:
        if key in body:
            resolver(config, resolved, body[key])
    _validate(resolved)
    return resolved


def _validate(resolved: Resolved) -> None:
    for name in classnames.ROUTABLE_CLASSES:
        model = resolved.routes.get(name)
        if model is None:
            continue
        upstream_name = model.partition("/")[0]
        if upstream_name in resolved.upstreams:
            continue
        if upstream_name in resolved.removed:
            using = sorted(
                other
                for other, value in resolved.routes.items()
                if value.partition("/")[0] == upstream_name
            )
            raise ApiError(
                409,
                "upstream_in_use",
                f"{upstream_name} cannot be removed while "
                f"{', '.join(using)} "
                + ("is" if len(using) == 1 else "are")
                + " routed to it. Re-route "
                + ("it" if len(using) == 1 else "them")
                + " first — in this same request if you like, the routes are "
                "applied after the upstreams",
                {
                    "field": f"upstreams.{upstream_name}",
                    "upstream": upstream_name,
                    "classes": using,
                },
            )
        raise ApiError(
            409,
            "route_upstream_unconfigured",
            f"{name} cannot be routed to {model!r}: [upstreams."
            f"{upstream_name}] is not configured on this server. Configure it "
            "in this same request (upstreams are applied before routes) or "
            "before. This server never stores a route it cannot serve",
            {
                "field": f"routes.{name}",
                "capability": name,
                "upstream": upstream_name,
                "model": model,
            },
        )


def recomputed_capability(
    config: Config,
    resolved: Resolved,
    *,
    gpu_vendor: str,
    card: "CardFacts | None",
) -> CapabilityRecord | None:
    record = config.capability
    if record is None:
        return None
    decisions = decide_on(
        record.backend_kind,
        total_bytes=record.total_bytes,
        desktop_allowance_bytes=resolved.desktop_allowance_bytes,
        gpu_vendor=gpu_vendor,
        card=card,
        chosen=resolved.local_models,
    )
    return record_of(
        record.backend_kind,
        total_bytes=record.total_bytes,
        desktop_allowance_bytes=resolved.desktop_allowance_bytes,
        decisions=decisions,
        routes=resolved.routes,
    )


def apply(
    config: Config,
    resolved: Resolved,
    *,
    gpu_vendor: str,
    card: "CardFacts | None",
) -> None:
    routes, upstreams, local_models = resolved.as_records()
    write_config(
        config.home,
        name=config.name,
        host=config.host,
        port=config.port,
        token=config.token,
        backend_kind=config.backend_kind,
        enable_echo=config.enable_echo,
        enable_llm=config.enable_llm,
        enable_asr=config.enable_asr,
        enable_tts=config.enable_tts,
        enable_align=config.enable_align,
        enable_rvc=config.enable_rvc,
        enable_denoise=config.enable_denoise,
        enable_image=config.enable_image,
        retention_days=config.retention_days,
        tts_engines=config.tts_engines,
        desktop_allowance_bytes=resolved.desktop_allowance_bytes,
        desktop_allowance_basis=resolved.desktop_allowance_basis,
        desktop_allowance_note=resolved.desktop_allowance_note,
        capability=recomputed_capability(
            config,
            resolved,
            gpu_vendor=gpu_vendor,
            card=card,
        ),
        routes=routes,
        upstreams=upstreams,
        local_models=local_models,
        advertise=config.advertise,
        tailscale_advertise=resolved.tailscale_advertise,
        lan_advertise=resolved.lan_advertise,
    )
    config.adopt(load_config(config.home))


__all__ = [
    "HISTORY_LIMIT",
    "History",
    "PATCH_KEYS",
    "Resolved",
    "apply",
    "document",
    "local_selection",
    "recomputed_capability",
    "resolve",
]
