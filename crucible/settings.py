from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from . import (
    capabilityclasses,
    classnames,
    llmconcurrency,
    lowvram,
    memorybudget,
    upstreamrecord,
    videodesktop,
)
from .backend import CardFacts
from .capabilityrecord import DESKTOP_BASIS_STATED, CapabilityRecord, desktop_reserve_words
from .capabilitystore import decide_on, record_of
from .capabilitywords import low_vram_refusal_note
from .clock import utcnow
from .config import (
    Config,
    LocalModelRecord,
    RouteRecord,
    _advertised,
    _cors_origins,
    check_bind_host,
    check_max_session_hold_s,
    check_port,
    check_retention_days,
    check_server_name,
    load_config,
    unowned_table,
    write_config,
)
from .errors import ApiError, ConfigError
from .events import SETTINGS, EventHub
from .fit import on_host
from .upstreamrecord import UPSTREAM_DISPLAY, UPSTREAM_NAMES, UpstreamRecord

HISTORY_LIMIT = 20

class History:

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: list[dict[str, Any]] = []
        self.events = EventHub()

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
        self.events.publish(SETTINGS, "settings.written", row)

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(self._rows))


def _low_vram_on(config: Config, desktop_allowance_bytes: int) -> lowvram.LowVram | None:
    """Decided against the card the record was decided on; None before any record,
    when there is no card to decide it against."""
    record = config.capability
    if record is None:
        return None
    return lowvram.on_card(
        config,
        record.backend_kind,
        total_bytes=record.total_bytes,
        desktop_allowance_bytes=desktop_allowance_bytes,
    )


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
        for candidate in on_host(entry.candidates(record.backend_kind), config.audio_low_vram):
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
                    "low_vram": candidate.held_low_vram,
                    "installed": installed[candidate.id],
                }
            )
        found[name] = rows
    return found


def document(
    config: Config, *, installed: Mapping[str, bool], resident: Any = None
) -> dict[str, Any]:
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
        "audio_low_vram": low_vram_entry(config),
        "llm_concurrency": llmconcurrency.rows(config, resident),
        **server_entries(config),
    }


# A table nothing in Settings writes, shown as the file has it: [video_trial] lifts the
# declared clip limits for a measurement and is set by the person running it.
READ_ONLY_TABLES: tuple[str, ...] = ("video_trial",)


def server_entries(config: Config) -> dict[str, Any]:
    """The [server], [auth], [jobs], [queue], [hf], [tts] and [video_desktop] keys, as
    Settings shows them. The bearer token and the Hugging Face token are never in it:
    `token_hint` and `hf.token_hint` are their last four characters."""
    return {
        "name": config.name,
        "host": config.host,
        "port": config.port,
        "advertise": list(config.advertise),
        "cors_origins": list(config.cors_origins),
        "open_pairing": config.open_pairing,
        "install_on_submit": config.install_on_submit,
        "retention_days": config.retention_days,
        "max_session_hold_s": config.max_session_hold_s,
        "token_hint": _hint(config.token),
        "hf": hf_entry(config),
        "video_desktop": video_desktop_entry(config),
        "tts_engines": [
            {"engine": entry.engine, **entry.to_dict()} for entry in config.tts_engines
        ],
        "read_only_tables": {
            name: table
            for name in READ_ONLY_TABLES
            if (table := unowned_table(config.path, name))
        },
    }


HINT_CHARS = 4


def _hint(secret: str) -> str:
    return f"…{secret[-HINT_CHARS:]}"


def hf_entry(config: Config) -> dict[str, Any]:
    """`[hf] token`, write-only. `from_environment`: $HF_TOKEN is set in this server's
    environment, which wins over the file (weights.hf_token_at)."""
    import os

    from .weights import HF_TOKEN_ENV

    token = unowned_table(config.path, "hf").get("token")
    configured = isinstance(token, str) and token.strip() != ""
    return {
        "configured": configured,
        "token_hint": _hint(token.strip()) if configured else None,
        "from_environment": os.environ.get(HF_TOKEN_ENV, "").strip() != "",
    }


def video_desktop_entry(config: Config) -> dict[str, Any] | None:
    """`[video_desktop]` on the one backend whose video engine reads it; None elsewhere."""
    if config.backend_kind != videodesktop.BACKEND:
        return None
    return {"rows": videodesktop.rows(unowned_table(config.path, videodesktop.TABLE))}


def live_document(
    config: Config,
    *,
    installed: Mapping[str, bool],
    resident: Any,
    resident_voice: Any,
    job_types: list[dict[str, Any]],
    bound: tuple[str, int],
) -> dict[str, Any]:
    """The settings document with what only the running server knows: the address it
    listens on now (`bound`, and `restart_pending`, the keys written that wait for a
    restart), every job type's flag and verdict, and the voice on the card with the
    [tts.<engine>] numbers it was started with."""
    found = document(config, installed=installed, resident=resident)
    host, port = bound
    found["bound"] = {"host": host, "port": port}
    found["restart_pending"] = [
        key for key, now in (("host", host), ("port", port)) if getattr(config, key) != now
    ]
    found["job_types"] = job_types
    for row in found["tts_engines"]:
        row["resident"] = _resident_on_engine(row, resident_voice)
    return found


ENGINE_LEVERS: tuple[str, ...] = (
    "memory_bytes_estimate", "max_num_seqs", "mem_fraction", "context_length",
)


def _resident_on_engine(row: dict[str, Any], voice: Any) -> dict[str, Any] | None:
    """The voice on the card served by this row's engine, the numbers it was started
    with, and whether those differ from the row's (it takes the row's at its next load)."""
    if voice is None or voice.narrator_engine != row["engine"] or voice.levers is None:
        return None
    return {
        "voice": voice.voice_id,
        "levers": dict(voice.levers),
        "reload_needed": any(voice.levers.get(k) != row.get(k) for k in ENGINE_LEVERS),
    }


def low_vram_entry(config: Config) -> dict[str, Any] | None:
    """`[audio] low_vram` for Settings: its state, who set it, and what the card the
    record was decided on makes of it. None before any record, and on a machine where no
    audio model can be split, so there is nothing to show or set."""
    decided = _low_vram_on(config, config.desktop_allowance_bytes)
    if decided is None or decided.need.verdict == lowvram.NOT_OFFERED:
        return None
    return decided.to_dict()


# Keys that take effect only when the server starts again: the address it listens on.
RESTART_KEYS: tuple[str, ...] = ("host", "port")


@dataclass
class ServerKeys:
    """The [server], [auth] and [jobs] keys Settings writes, as they will be written."""

    name: str
    host: str
    port: int
    advertise: tuple[str, ...]
    cors_origins: tuple[str, ...]
    open_pairing: bool
    install_on_submit: bool
    retention_days: int

    @classmethod
    def of(cls, config: Config) -> "ServerKeys":
        return cls(
            name=config.name,
            host=config.host,
            port=config.port,
            advertise=config.advertise,
            cors_origins=config.cors_origins,
            open_pairing=config.open_pairing,
            install_on_submit=config.install_on_submit,
            retention_days=config.retention_days,
        )


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
        # Not a settings key: `[audio] low_vram` as this card is decided with
        # (crucible/lowvram.py), which a new desktop allowance can change when Crucible
        # owns it. It makes an audio model's need.
        self.low_vram: lowvram.LowVram | None = _low_vram_on(
            config, config.desktop_allowance_bytes
        )
        self.audio_low_vram = (
            config.audio_low_vram if self.low_vram is None else self.low_vram.on
        )
        self.tailscale_advertise = config.tailscale_advertise
        self.lan_advertise = config.lan_advertise
        self.server = ServerKeys.of(config)
        # Keys set in tables this writer does not own ([queue], [hf], [video_desktop]);
        # None removes a key.
        self.unowned: dict[str, dict[str, Any]] = {}
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
    resolved.low_vram = _low_vram_on(config, value)
    if resolved.low_vram is not None:
        resolved.audio_low_vram = resolved.low_vram.on


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
    record: CapabilityRecord, resolved: Resolved, name: str, value: str, field: str
) -> Any:
    entry = capabilityclasses.BY_NAME[name]
    assert entry.candidates is not None
    offered = on_host(entry.candidates(record.backend_kind), resolved.audio_low_vram)
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
        f"allowance) — short by {shortfall / 2**30:.1f} GiB."
        + (low_vram_refusal_note(picked) if picked.would_fit_low_vram(budget) else ""),
        {
            "field": field,
            "capability": name,
            "model": value,
            "memory_bytes_estimate": picked.memory_bytes_estimate,
            "available_bytes": budget,
            "shortfall_bytes": shortfall,
            "low_vram": picked.held_low_vram,
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
    picked = _offered_candidate(decided, resolved, name, value, field)
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


def _refused(exc: ConfigError, field: str) -> ApiError:
    """A config reader's refusal, as the Settings door answers it: 400 with the field.
    A reader that names its refusal (`code: sentence`) keeps that name."""
    named, colon, rest = str(exc).partition(": ")
    if colon and named.replace("_", "").isalpha() and named.islower():
        return ApiError(400, named, rest, {"field": field})
    return ApiError(400, "invalid_request", str(exc), {"field": field})


def _checked(check: Callable[[Any, str], Any], value: Any, field: str) -> Any:
    try:
        return check(value, f"settings {field}")
    except ConfigError as exc:
        raise _refused(exc, field) from None


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ApiError(
            400, "invalid_request", f"{field} is true or false, got {value!r}",
            {"field": field},
        )
    return value


def _note(resolved: Resolved, old: Any, new: Any, words: str) -> None:
    if new != old:
        resolved.changed.append(words)


def _resolve_name(config: Config, resolved: Resolved, value: Any) -> None:
    resolved.server.name = _checked(check_server_name, value, "name")
    _note(resolved, config.name, value, f"[server] name = {value}")


def _resolve_host(config: Config, resolved: Resolved, value: Any) -> None:
    resolved.server.host = _checked(check_bind_host, value, "host")
    _note(resolved, config.host, value,
          f"[server] host = {value} (takes effect when Crucible restarts)")


def port_fixed_by_windows(config: Config) -> str | None:
    """Why this server's port is not this server's to choose, or None when it is. The
    Windows host reaches its engine (the WSL2 guest, or its own child) on one fixed
    port, platform/paths.py ENGINE_PORT, and has no setting to follow another."""
    from .backend import LLAMA_WINDOWS
    from .platform.paths import ENGINE_PORT
    from .service import in_wsl

    if config.backend_kind == LLAMA_WINDOWS or in_wsl():
        return (
            f"the Windows host reaches this engine on port {ENGINE_PORT} and has no "
            "setting to follow another, so the PC would lose its engine. On a PC the "
            f"port stays {ENGINE_PORT}"
        )
    return None


def _resolve_port(config: Config, resolved: Resolved, value: Any) -> None:
    port = _checked(check_port, value, "port")
    if port != config.port:
        fixed = port_fixed_by_windows(config)
        if fixed is not None:
            raise ApiError(409, "port_fixed_by_windows_host", fixed, {"field": "port"})
    resolved.server.port = port
    _note(resolved, config.port, port,
          f"[server] port = {port} (takes effect when Crucible restarts)")


def _resolve_advertise(config: Config, resolved: Resolved, value: Any) -> None:
    resolved.server.advertise = _advertise_list(value, "advertise")
    _note(resolved, config.advertise, resolved.server.advertise, "[server] advertise")


def _resolve_cors_origins(config: Config, resolved: Resolved, value: Any) -> None:
    try:
        resolved.server.cors_origins = _cors_origins({"server": {"cors_origins": value}})
    except ConfigError as exc:
        raise _refused(exc, "cors_origins") from None
    _note(resolved, config.cors_origins, resolved.server.cors_origins,
          "[server] cors_origins")


def _resolve_open_pairing(config: Config, resolved: Resolved, value: Any) -> None:
    resolved.server.open_pairing = _require_bool(value, "open_pairing")
    _note(resolved, config.open_pairing, value,
          f"[auth] open_pairing = {str(value).lower()}")


def _resolve_install_on_submit(config: Config, resolved: Resolved, value: Any) -> None:
    resolved.server.install_on_submit = _require_bool(value, "install_on_submit")
    _note(resolved, config.install_on_submit, value,
          f"[jobs] install_on_submit = {str(value).lower()}")


def _resolve_retention_days(config: Config, resolved: Resolved, value: Any) -> None:
    resolved.server.retention_days = _checked(check_retention_days, value, "retention_days")
    _note(resolved, config.retention_days, value, f"[jobs] retention_days = {value}")


def _resolve_max_session_hold_s(config: Config, resolved: Resolved, value: Any) -> None:
    held = _checked(check_max_session_hold_s, value, "max_session_hold_s")
    resolved.unowned.setdefault("queue", {})["max_session_hold_s"] = held
    _note(resolved, config.max_session_hold_s, held, f"[queue] max_session_hold_s = {held}")


HF_TOKEN_LEAST_CHARS = 8


def _resolve_hf_token(config: Config, resolved: Resolved, value: Any) -> None:
    if value is not None and (
        not isinstance(value, str)
        or len(value.strip()) < HF_TOKEN_LEAST_CHARS
        or any(c.isspace() for c in value.strip())
    ):
        raise ApiError(
            400,
            "invalid_request",
            "hf_token is a Hugging Face access token (hf_...), one word of at least "
            f"{HF_TOKEN_LEAST_CHARS} characters, or null to remove it",
            {"field": "hf_token"},
        )
    token = None if value is None else value.strip()
    resolved.unowned.setdefault("hf", {})["token"] = token
    resolved.changed.append("[hf] token " + ("removed" if token is None else "set"))


def _resolve_video_desktop(config: Config, resolved: Resolved, value: Any) -> None:
    table = _require_object(value, "video_desktop")
    if config.backend_kind != videodesktop.BACKEND:
        raise ApiError(
            409,
            "video_desktop_not_here",
            f"[video_desktop] is read only by the video engine on {videodesktop.BACKEND}; "
            f"this server runs {config.backend_kind}, so nothing would read it",
            {"field": "video_desktop"},
        )
    keys = resolved.unowned.setdefault(videodesktop.TABLE, {})
    for key in sorted(table):
        entry = table[key]
        if entry is not None:
            try:
                videodesktop.check_key(key, entry)
            except ConfigError as exc:
                raise _refused(exc, f"video_desktop.{key}") from None
        keys[key] = entry
        resolved.changed.append(
            f"[video_desktop] {key} = " + ("the default" if entry is None else repr(entry))
        )


SectionResolver = Callable[[Config, Resolved, Any], None]

SECTION_RESOLVERS: tuple[tuple[str, SectionResolver], ...] = (
    ("upstreams", _resolve_upstreams),
    ("routes", _resolve_routes),
    ("desktop_allowance_bytes", _resolve_desktop_allowance),
    ("local_models", _resolve_local_models),
    ("tailscale_advertise", _resolve_tailscale_advertise),
    ("lan_advertise", _resolve_lan_advertise),
    ("name", _resolve_name),
    ("host", _resolve_host),
    ("port", _resolve_port),
    ("advertise", _resolve_advertise),
    ("cors_origins", _resolve_cors_origins),
    ("open_pairing", _resolve_open_pairing),
    ("install_on_submit", _resolve_install_on_submit),
    ("retention_days", _resolve_retention_days),
    ("max_session_hold_s", _resolve_max_session_hold_s),
    ("hf_token", _resolve_hf_token),
    ("video_desktop", _resolve_video_desktop),
)

PATCH_KEYS: frozenset[str] = frozenset(key for key, _ in SECTION_RESOLVERS)


def resolve(config: Config, patch: Any) -> Resolved:
    body = _require_object(patch, ROOT_FIELD)
    _refuse_unknown_fields(body)
    resolved = Resolved(config)
    for key, resolver in SECTION_RESOLVERS:
        if key in body:
            resolver(config, resolved, body[key])
    _validate(resolved)
    if resolved.low_vram is not None and resolved.low_vram.changed:
        # Crucible's own `[audio] low_vram` follows the card it is written with.
        resolved.changed.append(resolved.low_vram.change_sentence)
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
        audio_low_vram=resolved.audio_low_vram,
    )
    return record_of(
        record.backend_kind,
        total_bytes=record.total_bytes,
        desktop_allowance_bytes=resolved.desktop_allowance_bytes,
        decisions=decisions,
        routes=resolved.routes,
    )


def _written_server_keys(config: Config, resolved: Resolved) -> dict[str, Any]:
    server = resolved.server
    return {
        "name": server.name,
        "host": server.host,
        "port": server.port,
        "token": config.token,
        "advertise": server.advertise,
        "cors_origins": server.cors_origins,
        "open_pairing": server.open_pairing,
        "install_on_submit": server.install_on_submit,
        "retention_days": server.retention_days,
        "tailscale_advertise": resolved.tailscale_advertise,
        "lan_advertise": resolved.lan_advertise,
        "unowned": resolved.unowned,
    }


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
        backend_kind=config.backend_kind,
        enable_echo=config.enable_echo,
        enable_llm=config.enable_llm,
        enable_asr=config.enable_asr,
        enable_tts=config.enable_tts,
        enable_align=config.enable_align,
        enable_rvc=config.enable_rvc,
        enable_denoise=config.enable_denoise,
        enable_image=config.enable_image,
        enable_audio=config.enable_audio,
        enable_segment=config.enable_segment,
        enable_video=config.enable_video,
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
        # Written with the record it was decided with, never apart from it.
        audio_low_vram=(
            resolved.low_vram.setting
            if resolved.low_vram is not None and resolved.low_vram.changed
            else None
        ),
        **_written_server_keys(config, resolved),
    )
    config.adopt(load_config(config.home))


__all__ = [
    "HISTORY_LIMIT",
    "History",
    "PATCH_KEYS",
    "Resolved",
    "apply",
    "document",
    "live_document",
    "server_entries",
    "local_selection",
    "recomputed_capability",
    "resolve",
]
