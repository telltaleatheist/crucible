from __future__ import annotations

import os
import secrets
import socket
import stat
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w

from .capabilityrecord import DESKTOP_BASES, CapabilityRecord, CapabilityRow
from .classnames import ROUTABLE_CLASSES, SELECTABLE_CLASSES
from .errors import ConfigError
from .narratorengines import ESTIMATE_BASES, NARRATOR_ENGINE_SAMPLING, EngineFootprint
from .tomltable import check_table
from .upstreamrecord import UPSTREAM_FIELD, UPSTREAM_NAMES, UpstreamRecord

CRUCIBLE_HOME_ENV = "CRUCIBLE_HOME"
DEFAULT_HOST = "127.0.0.1"

DEFAULT_OPEN_PAIRING = True

DEFAULT_RETENTION_DAYS = 7

DEFAULT_INSTALL_ON_SUBMIT = True

DEFAULT_ENABLE_IMAGE = False

DEFAULT_ENABLE_AUDIO = False

DEFAULT_ENABLE_SEGMENT = False
DEFAULT_ENABLE_VIDEO = False

DEFAULT_MAX_SESSION_HOLD_S = 0

DEFAULT_PORT = 7100
TOKEN_BYTES = 32

DEFAULT_DESKTOP_ALLOWANCE_BYTES = 3 * 1024 ** 3

MLX_DESKTOP_ALLOWANCE_FRACTION = 0.25


def default_desktop_allowance_bytes(backend_kind: str, total_bytes: int) -> int:
    if backend_kind == "mlx-darwin":
        return int(total_bytes * MLX_DESKTOP_ALLOWANCE_FRACTION)
    return DEFAULT_DESKTOP_ALLOWANCE_BYTES


WINDOWS_HOME_DIRNAME = "Crucible"


def crucible_home() -> Path:
    override = os.environ.get(CRUCIBLE_HOME_ENV)
    if override is not None and override != "":
        return Path(override).expanduser()
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if local is None or local == "":
            raise ConfigError(
                "%LOCALAPPDATA% is not set, so this Windows host cannot say "
                f"where Crucible's home is. Set {CRUCIBLE_HOME_ENV} to a "
                "directory on a disk with room for the weights"
            )
        return Path(local) / WINDOWS_HOME_DIRNAME
    return Path.home() / ".crucible"


def config_path(home: Path | None = None) -> Path:
    return (home if home is not None else crucible_home()) / "config.toml"


def mint_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def default_server_name() -> str:
    return f"crucible@{socket.gethostname()}"


@dataclass(frozen=True)
class RouteRecord:

    capability: str
    model: str


@dataclass(frozen=True)
class LocalModelRecord:

    capability: str
    model: str


@dataclass(frozen=True)
class Config:
    path: Path
    home: Path
    name: str
    host: str
    port: int
    token: str
    backend_kind: str
    enable_echo: bool
    enable_llm: bool
    enable_asr: bool
    enable_tts: bool
    enable_align: bool
    enable_rvc: bool
    enable_denoise: bool
    desktop_allowance_bytes: int
    desktop_allowance_basis: str
    enable_image: bool = DEFAULT_ENABLE_IMAGE
    enable_audio: bool = DEFAULT_ENABLE_AUDIO
    enable_segment: bool = DEFAULT_ENABLE_SEGMENT
    enable_video: bool = DEFAULT_ENABLE_VIDEO
    advertise: tuple[str, ...] = ()
    tailscale_advertise: tuple[str, ...] = ()
    lan_advertise: tuple[str, ...] = ()
    open_pairing: bool = True
    retention_days: int = DEFAULT_RETENTION_DAYS
    desktop_allowance_note: str = ""
    install_on_submit: bool = DEFAULT_INSTALL_ON_SUBMIT
    capability: CapabilityRecord | None = None
    routes: tuple[RouteRecord, ...] = ()
    upstreams: tuple[UpstreamRecord, ...] = ()
    local_models: tuple[LocalModelRecord, ...] = ()
    tts_engines: tuple[EngineFootprint, ...] = ()
    max_session_hold_s: int = DEFAULT_MAX_SESSION_HOLD_S
    stamp: tuple[int, int] | None = None

    def follow_file(self) -> bool:
        try:
            current = self.path.stat()
        except FileNotFoundError as exc:
            raise ConfigError(
                f"{self.path} is gone from under a running server; the last "
                "document read from it is still being served"
            ) from exc
        seen = (current.st_mtime_ns, current.st_size)
        if seen == self.stamp:
            return False
        self.adopt(load_config(self.home))
        return True

    def engine_footprint(self, narrator_engine: str) -> EngineFootprint | None:
        for entry in self.tts_engines:
            if entry.engine == narrator_engine:
                return entry
        return None

    def route_model(self, capability: str) -> str | None:
        for entry in self.routes:
            if entry.capability == capability:
                return entry.model
        return None

    def local_model(self, capability: str) -> str | None:
        for entry in self.local_models:
            if entry.capability == capability:
                return entry.model
        return None

    def upstream(self, name: str) -> UpstreamRecord | None:
        for entry in self.upstreams:
            if entry.name == name:
                return entry
        return None

    def classes_routed_to(self, name: str) -> tuple[str, ...]:
        return tuple(
            entry.capability
            for entry in self.routes
            if entry.model.partition("/")[0] == name
        )

    def adopt(self, fresh: "Config") -> None:
        if fresh.path != self.path or fresh.home != self.home:
            raise ConfigError(
                f"refusing to adopt a config from {fresh.path} into the one this "
                f"server loaded from {self.path}. A reload re-reads THIS server's "
                "own file; a different file is a different server"
            )
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, getattr(fresh, name))

    @property
    def jobs_dir(self) -> Path:
        return self.home / "jobs"

    @property
    def uploads_dir(self) -> Path:
        return self.home / "uploads"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"


def _open_pairing(table: dict[str, Any]) -> bool:
    auth = table.get("auth")
    if not isinstance(auth, dict) or "open_pairing" not in auth:
        return DEFAULT_OPEN_PAIRING
    value = auth["open_pairing"]
    if not isinstance(value, bool):
        raise ConfigError(
            "config [auth] open_pairing: must be true or false, not "
            f"{value!r}. A quoted string here would read as true and open a "
            "door you meant to close"
        )
    return value


def _retention_days(table: dict[str, Any]) -> int:
    section = table.get("jobs")
    if section is None:
        raise ConfigError("config is missing the [jobs] section")
    if "retention_days" not in section:
        return DEFAULT_RETENTION_DAYS
    value = _require(table, "jobs", "retention_days", int)
    if value < 1:
        raise ConfigError(
            f"config [jobs] retention_days: must be at least 1 day, got {value}. "
            "A server that kept a finished job for no days would delete its "
            "artifacts before the client that asked for them could fetch them"
        )
    return value


def _desktop_basis(table: dict[str, Any]) -> str:
    value = _require(table, "accelerator", "desktop_allowance_basis", str)
    if value not in DESKTOP_BASES:
        raise ConfigError(
            f"config [accelerator] desktop_allowance_basis: {value!r} is not one of "
            f"{list(DESKTOP_BASES)}"
        )
    return value


def _desktop_note(table: dict[str, Any]) -> str:
    section = table.get("accelerator") or {}
    if "desktop_allowance_note" not in section:
        return ""
    return _require(table, "accelerator", "desktop_allowance_note", str)


def _max_session_hold_s(table: dict[str, Any]) -> int:
    """`[queue] max_session_hold_s`: how long one queue session may hold the server.
    Absent or 0 is no limit; a session stays open while its client keeps using it."""
    section = table.get("queue")
    if section is None or "max_session_hold_s" not in section:
        return DEFAULT_MAX_SESSION_HOLD_S
    value = _require(table, "queue", "max_session_hold_s", int)
    if value < 0:
        raise ConfigError(
            f"config [queue] max_session_hold_s: must be 0 (no limit) or a number of "
            f"seconds, got {value}"
        )
    return value


def _install_on_submit(table: dict[str, Any]) -> bool:
    section = table.get("jobs")
    if section is None:
        raise ConfigError("config is missing the [jobs] section")
    if "install_on_submit" not in section:
        return DEFAULT_INSTALL_ON_SUBMIT
    return _require(table, "jobs", "install_on_submit", bool)


def _optional_jobs_flag(table: dict[str, Any], key: str, default: bool) -> bool:
    if key not in table.get("jobs", {}):
        return default
    return _require(table, "jobs", key, bool)


def _kept_jobs_flag(home: Path, key: str, default: bool) -> bool:
    try:
        with open(config_path(home), "rb") as handle:
            value = tomllib.load(handle).get("jobs", {}).get(key)
    except (OSError, tomllib.TOMLDecodeError):
        return default
    return value if isinstance(value, bool) else default


def _advertised(table: dict[str, Any]) -> tuple[str, ...]:
    server = table.get("server")
    if not isinstance(server, dict) or "advertise" not in server:
        return ()
    raw = server["advertise"]
    if not isinstance(raw, list) or not all(isinstance(entry, str) for entry in raw):
        raise ConfigError(
            "config [server] advertise: must be a list of strings, each an "
            "address something forwards to this server on "
            '(e.g. advertise = ["owens-pc.owenmorgan.com:7100"])'
        )
    cleaned: list[str] = []
    for entry in raw:
        authority = entry.strip()
        if authority == "":
            raise ConfigError(
                "config [server] advertise: an empty entry names no address"
            )
        if "://" in authority:
            raise ConfigError(
                f"config [server] advertise: {entry!r} carries a scheme; entries "
                "are authorities like `host` or `host:port`, and the scheme is "
                "the server's own"
            )
        if "/" in authority:
            raise ConfigError(
                f"config [server] advertise: {entry!r} carries a path; an address "
                "an app dials has nowhere to put one"
            )
        from urllib.parse import urlsplit
        try:
            parsed = urlsplit("http://" + authority)
            port = parsed.port
            if (not parsed.hostname or parsed.username is not None or parsed.password is not None
                    or parsed.query or parsed.fragment or any(c.isspace() for c in authority)
                    or "\\" in authority or parsed.hostname in ("0.0.0.0", "::")
                    or (port is not None and not 1 <= port <= 65535)):
                raise ValueError("not a dialable authority")
        except ValueError as exc:
            raise ConfigError(f"config [server] advertise: invalid authority {entry!r}: {exc}") from exc
        if authority not in cleaned:
            cleaned.append(authority)
    return tuple(cleaned)


def _require(table: dict[str, Any], section: str, key: str, kind: type) -> Any:
    if section not in table:
        raise ConfigError(f"config is missing the [{section}] section")
    if key not in table[section]:
        raise ConfigError(f"config is missing {section}.{key}")
    value = table[section][key]
    wrong_type = not isinstance(value, kind)
    if kind is int and isinstance(value, bool):
        wrong_type = True
    if wrong_type:
        raise ConfigError(
            f"config key {section}.{key} must be {kind.__name__}, got "
            f"{type(value).__name__}"
        )
    return value


CAPABILITY_FLAGS: tuple[str, ...] = (
    "enable_echo",
    "enable_llm",
    "enable_asr",
    "enable_tts",
    "enable_align",
    "enable_rvc",
    "enable_denoise",
    "enable_image",
    "enable_audio",
    "enable_segment",
    "enable_video",
)


_CAPABILITY_REQUIRED: dict[str, type] = {
    "backend_kind": str,
    "total_bytes": int,
    "desktop_allowance_bytes": int,
}
_CAPABILITY_ROW_REQUIRED: dict[str, type] = {
    "capability": str,
    "enabled": bool,
    "selected": str,
    "reason": str,
    "summary": str,
    "shortfall_bytes": int,
}


def _capability_record(table: dict[str, Any]) -> CapabilityRecord | None:
    section = table.get("capability")
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ConfigError("config key capability must be a table")
    allowed = set(_CAPABILITY_REQUIRED) | {"classes"}
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ConfigError(
            f"config [capability]: unknown key(s) {unknown}; this table takes "
            f"exactly {sorted(allowed)}"
        )
    for key, kind in _CAPABILITY_REQUIRED.items():
        if key not in section:
            raise ConfigError(f"config is missing capability.{key}")
        value = section[key]
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            raise ConfigError(
                f"config key capability.{key} must be {kind.__name__}, got "
                f"{type(value).__name__}"
            )
    raw_rows = section.get("classes")
    if raw_rows is None:
        raise ConfigError("config is missing capability.classes")
    if not isinstance(raw_rows, list):
        raise ConfigError(
            "config key capability.classes must be an array of tables, one per "
            "capability class"
        )
    rows: list[CapabilityRow] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rows):
        where = f"config [[capability.classes]][{index}]"
        if not isinstance(raw, dict):
            raise ConfigError(f"{where} must be a table")
        unknown = sorted(set(raw) - set(_CAPABILITY_ROW_REQUIRED))
        if unknown:
            raise ConfigError(
                f"{where}: unknown key(s) {unknown}; a row takes "
                f"{sorted(_CAPABILITY_ROW_REQUIRED)}"
            )
        for key, kind in _CAPABILITY_ROW_REQUIRED.items():
            if key not in raw:
                raise ConfigError(f"{where}: missing required key {key!r}")
            value = raw[key]
            if not isinstance(value, kind) or (
                kind is int and isinstance(value, bool)
            ):
                raise ConfigError(
                    f"{where}: {key} must be {kind.__name__}, got "
                    f"{type(value).__name__}"
                )
        name = raw["capability"]
        if name in seen:
            raise ConfigError(
                f"{where}: capability {name!r} is recorded twice; one class has "
                "one verdict"
            )
        seen.add(name)
        rows.append(
            CapabilityRow(
                capability=name,
                enabled=raw["enabled"],
                selected=raw["selected"],
                reason=raw["reason"],
                shortfall_bytes=raw["shortfall_bytes"],
                summary=raw["summary"],
            )
        )
    return CapabilityRecord(
        backend_kind=section["backend_kind"],
        total_bytes=section["total_bytes"],
        desktop_allowance_bytes=section["desktop_allowance_bytes"],
        rows=tuple(rows),
    )


def _upstream_records(table: dict[str, Any]) -> tuple[UpstreamRecord, ...]:
    section = table.get("upstreams")
    if section is None:
        return ()
    if not isinstance(section, dict):
        raise ConfigError("config key upstreams must be a table")
    unknown = sorted(set(section) - set(UPSTREAM_NAMES))
    if unknown:
        raise ConfigError(
            f"config [upstreams]: unknown_upstream {unknown}; this server speaks "
            f"to exactly {list(UPSTREAM_NAMES)}"
        )
    found: list[UpstreamRecord] = []
    for name in UPSTREAM_NAMES:
        entry = section.get(name)
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise ConfigError(f"config key upstreams.{name} must be a table")
        wanted = UPSTREAM_FIELD[name]
        extra = sorted(set(entry) - {wanted})
        if extra:
            raise ConfigError(
                f"config [upstreams.{name}]: upstream_bad_field {extra}; this "
                f"upstream is configured with a {wanted!r} and nothing else"
            )
        value = entry.get(wanted)
        if not isinstance(value, str) or value.strip() == "":
            raise ConfigError(
                f"config is missing upstreams.{name}.{wanted}; an upstream "
                "table that exists is one this server can call, and one it "
                "cannot call must be absent instead"
            )
        if wanted == "key":
            found.append(UpstreamRecord(name=name, key=value))
        else:
            found.append(UpstreamRecord(name=name, url=value.rstrip("/")))
    return tuple(found)


def _route_records(
    table: dict[str, Any], upstreams: tuple[UpstreamRecord, ...]
) -> tuple[RouteRecord, ...]:
    section = table.get("routes")
    if section is None:
        return ()
    if not isinstance(section, dict):
        raise ConfigError("config key routes must be a table")
    configured = {entry.name for entry in upstreams}
    found: list[RouteRecord] = []
    for name in sorted(section):
        if name not in ROUTABLE_CLASSES:
            raise ConfigError(
                f"config [routes]: route_not_routable {name!r}; only "
                f"{list(ROUTABLE_CLASSES)} may run anywhere but this card"
            )
        model = section[name]
        if not isinstance(model, str):
            raise ConfigError(
                f"config key routes.{name} must be a string, got "
                f"{type(model).__name__}"
            )
        if model == "local":
            raise ConfigError(
                f"config [routes]: routes.{name} is \"local\", which is the "
                "absence of a route and is never written; remove the key"
            )
        upstream_name, _, rest = model.partition("/")
        if upstream_name not in UPSTREAM_NAMES or rest == "":
            raise ConfigError(
                f"config [routes]: route_bad_model {model!r} for {name}; a "
                f"route's value is `<upstream>/<model>` with the upstream one "
                f"of {list(UPSTREAM_NAMES)}"
            )
        if upstream_name not in configured:
            raise ConfigError(
                f"config [routes]: route_upstream_unconfigured — {name} is "
                f"routed to {model!r} and [upstreams.{upstream_name}] is not in "
                "this config. This server never holds a route it cannot serve"
            )
        found.append(RouteRecord(capability=name, model=model))
    return tuple(found)


def _local_model_records(table: dict[str, Any]) -> tuple[LocalModelRecord, ...]:
    section = table.get("local_models")
    if section is None:
        return ()
    if not isinstance(section, dict):
        raise ConfigError("config key local_models must be a table")
    found: list[LocalModelRecord] = []
    for name in sorted(section):
        if name not in SELECTABLE_CLASSES:
            raise ConfigError(
                f"config [local_models]: local_model_not_selectable {name!r}; "
                f"only {list(SELECTABLE_CLASSES)} choose a local model"
            )
        model = section[name]
        if not isinstance(model, str):
            raise ConfigError(
                f"config key local_models.{name} must be a string, got "
                f"{type(model).__name__}"
            )
        if model == "":
            raise ConfigError(
                f"config [local_models]: local_models.{name} is empty, which is "
                "the absence of a selection and is never written; remove the key"
            )
        found.append(LocalModelRecord(capability=name, model=model))
    return tuple(found)


_TTS_ENGINE_REQUIRED: dict[str, type] = {
    "memory_bytes_estimate": int,
    "estimate_basis": str,
    "max_num_seqs": int,
    "max_num_seqs_note": str,
}
_TTS_ENGINE_OPTIONAL: dict[str, type] = {
    "estimate_note": str,
    "mem_fraction": object,
    "mem_fraction_note": str,
    "context_length": int,
    "context_length_note": str,
}


def _tts_engine_lever(where: str, block: dict[str, Any], key: str) -> Any:
    note = block.get(f"{key}_note")
    value = block.get(key)
    if value is None:
        if note is not None:
            raise ConfigError(
                f"{where}: states {key}_note and no {key}. The note says where a "
                "number came from and there is no number"
            )
        return None
    if note is None or note.strip() == "":
        raise ConfigError(
            f"{where}: {key} carries no note. It reconfigures the server "
            "narrator starts, so the next person to touch it has to be able to "
            "find out where the number came from"
        )
    return value


def _tts_engine_records(table: dict[str, Any]) -> tuple[EngineFootprint, ...]:
    section = table.get("tts")
    if section is None:
        return ()
    if not isinstance(section, dict):
        raise ConfigError("config key tts must be a table")
    found: list[EngineFootprint] = []
    for engine in sorted(section):
        where = f"config [tts.{engine}]"
        if engine not in NARRATOR_ENGINE_SAMPLING:
            raise ConfigError(
                f"{where}: {engine!r} is not one of narrator's engines; they are "
                f"{sorted(NARRATOR_ENGINE_SAMPLING)}. A footprint for an engine "
                "nothing serves is a number nothing reads"
            )
        block = section[engine]
        if not isinstance(block, dict):
            raise ConfigError(f"{where}: must be a table")
        check_table(
            where, block, _TTS_ENGINE_REQUIRED, _TTS_ENGINE_OPTIONAL, error=ConfigError
        )
        if block["memory_bytes_estimate"] <= 0:
            raise ConfigError(
                f"{where}: memory_bytes_estimate must be positive, got "
                f"{block['memory_bytes_estimate']}"
            )
        basis = block["estimate_basis"]
        if basis not in ESTIMATE_BASES:
            raise ConfigError(
                f"{where}: estimate_basis {basis!r} is not one of "
                f"{sorted(ESTIMATE_BASES)}"
            )
        note = block.get("estimate_note")
        if basis == "declared" and (note is None or note.strip() == ""):
            raise ConfigError(
                f"{where}: estimate_basis is 'declared' and there is no "
                "estimate_note. A declared number came from somewhere — an "
                "engine's configured reservation, a sibling machine's "
                "measurement — and the reader of a /v1/voices row has to be able "
                "to find out where"
            )
        if basis == "measured" and note is not None:
            raise ConfigError(
                f"{where}: estimate_basis is 'measured' and it also carries an "
                "estimate_note. Put the measurement in a comment beside the "
                "number; estimate_note is what a DECLARED number owes"
            )
        if block["max_num_seqs"] < 1:
            raise ConfigError(
                f"{where}: max_num_seqs must be at least 1, got "
                f"{block['max_num_seqs']}"
            )
        if block["max_num_seqs_note"].strip() == "":
            raise ConfigError(
                f"{where}: max_num_seqs carries no note. The number is contested "
                "— the deathstalker cap certificate was measured at 64 while the "
                "shipped width is 16 — so the next person to touch it has to be "
                "able to find out where it came from"
            )
        mem_fraction = _tts_engine_lever(where, block, "mem_fraction")
        if mem_fraction is not None:
            if isinstance(mem_fraction, bool) or not isinstance(
                mem_fraction, (int, float)
            ):
                raise ConfigError(
                    f"{where}: mem_fraction is {mem_fraction!r}, which is not a "
                    "fraction"
                )
            if not 0 < mem_fraction < 1:
                raise ConfigError(
                    f"{where}: mem_fraction must be a fraction in (0, 1), got "
                    f"{mem_fraction}. It is SGLang's --mem-fraction-static, and "
                    "narrator's launcher refuses anything else by name"
                )
            mem_fraction = float(mem_fraction)
        context_length = _tts_engine_lever(where, block, "context_length")
        if context_length is not None and context_length <= 0:
            raise ConfigError(
                f"{where}: context_length must be positive, got {context_length}"
            )
        found.append(
            EngineFootprint(
                engine=engine,
                memory_bytes_estimate=block["memory_bytes_estimate"],
                estimate_basis=basis,
                estimate_note=note,
                max_num_seqs=block["max_num_seqs"],
                max_num_seqs_note=block["max_num_seqs_note"],
                mem_fraction=mem_fraction,
                mem_fraction_note=block.get("mem_fraction_note"),
                context_length=context_length,
                context_length_note=block.get("context_length_note"),
            )
        )
    return tuple(found)




def tts_engine_footprints(home: Path | None = None) -> dict[str, EngineFootprint]:
    _root, _path, table = _read_document(home)
    return {entry.engine: entry for entry in _tts_engine_records(table)}


RECORD_COMMAND = "crucible capability --write"


def _record_agrees(
    record: CapabilityRecord | None, backend_kind: str, desktop_allowance_bytes: int
) -> None:
    if record is None:
        return
    if record.backend_kind != backend_kind:
        raise ConfigError(
            f"config [capability] backend_kind is {record.backend_kind!r} but "
            f"[backend] kind is {backend_kind!r}: the capability record was "
            f"decided for another backend. `{RECORD_COMMAND}` rewrites the "
            "record from this host"
        )
    if record.desktop_allowance_bytes != desktop_allowance_bytes:
        raise ConfigError(
            f"config [capability] desktop_allowance_bytes is "
            f"{record.desktop_allowance_bytes} but [accelerator] "
            f"desktop_allowance_bytes is {desktop_allowance_bytes}: the "
            "capability record was decided with another desktop reserve. "
            f"`{RECORD_COMMAND}` rewrites the record with the reserve"
        )


def _read_document(home: Path | None) -> tuple[Path, Path, dict[str, Any]]:
    root = home if home is not None else crucible_home()
    path = config_path(root)
    if not path.exists():
        raise ConfigError(
            f"no config at {path} — run `crucible init` (or set {CRUCIBLE_HOME_ENV})"
        )
    try:
        with path.open("rb") as handle:
            return root, path, tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc


def own_engine_backend(home: Path | None = None) -> str | None:
    _root, _path, table = _read_document(home)
    if "server" not in table:
        return None
    return _require(table, "backend", "kind", str)


def load_config(
    home: Path | None = None, *, tolerate_stale_record: bool = False
) -> Config:
    stamped = config_path(home if home is not None else crucible_home())
    try:
        before = stamped.stat()
        stamp: tuple[int, int] | None = (before.st_mtime_ns, before.st_size)
    except FileNotFoundError:
        stamp = None
    root, path, table = _read_document(home)

    upstreams = _upstream_records(table)
    record = _capability_record(table)
    if not tolerate_stale_record:
        _record_agrees(
            record,
            _require(table, "backend", "kind", str),
            _require(table, "accelerator", "desktop_allowance_bytes", int),
        )
    return Config(
        path=path,
        home=root,
        name=_require(table, "server", "name", str),
        host=_require(table, "server", "host", str),
        port=_require(table, "server", "port", int),
        advertise=_advertised(table),
        tailscale_advertise=_advertised({"server": {"advertise": table.get("server", {}).get("tailscale_advertise", [])}}),
        lan_advertise=_advertised({"server": {"advertise": table.get("server", {}).get("lan_advertise", [])}}),
        token=_require(table, "auth", "token", str),
        open_pairing=_open_pairing(table),
        backend_kind=_require(table, "backend", "kind", str),
        enable_echo=_require(table, "jobs", "enable_echo", bool),
        enable_llm=_require(table, "jobs", "enable_llm", bool),
        enable_asr=_require(table, "jobs", "enable_asr", bool),
        enable_tts=_require(table, "jobs", "enable_tts", bool),
        enable_align=_require(table, "jobs", "enable_align", bool),
        enable_rvc=_require(table, "jobs", "enable_rvc", bool),
        enable_denoise=_require(table, "jobs", "enable_denoise", bool),
        enable_image=_optional_jobs_flag(table, "enable_image", DEFAULT_ENABLE_IMAGE),
        enable_audio=_optional_jobs_flag(table, "enable_audio", DEFAULT_ENABLE_AUDIO),
        enable_segment=_optional_jobs_flag(table, "enable_segment", DEFAULT_ENABLE_SEGMENT),
        enable_video=_optional_jobs_flag(table, "enable_video", DEFAULT_ENABLE_VIDEO),
        install_on_submit=_install_on_submit(table),
        retention_days=_retention_days(table),
        desktop_allowance_bytes=_require(
            table, "accelerator", "desktop_allowance_bytes", int
        ),
        desktop_allowance_basis=_desktop_basis(table),
        desktop_allowance_note=_desktop_note(table),
        capability=record,
        routes=_route_records(table, upstreams),
        local_models=_local_model_records(table),
        upstreams=upstreams,
        tts_engines=_tts_engine_records(table),
        max_session_hold_s=_max_session_hold_s(table),
        stamp=stamp,
    )


WRITER_OWNED_TABLES = frozenset(
    {
        "server",
        "auth",
        "backend",
        "jobs",
        "accelerator",
        "capability",
        "routes",
        "tts",
        "local_models",
        "upstreams",
    }
)


def _unowned_tables(path: Path) -> dict[str, Any]:
    """Tables in the file on disk that write_config does not produce, kept as they are.

    [hf] token is the one that mattered: every `crucible install <type>` rewrites the
    config through write_config, and until 1.0.68 that dropped [hf], so the next gated
    pull (LTX-2.5 on 2026-09-30) was refused for want of a token that had been there.
    Anything a person or another version added survives the same way.
    """
    import tomllib

    try:
        existing = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return {
        name: table
        for name, table in existing.items()
        if name not in WRITER_OWNED_TABLES and isinstance(table, dict)
    }


def write_config(
    home: Path,
    *,
    name: str,
    host: str,
    port: int,
    token: str,
    backend_kind: str,
    enable_echo: bool,
    enable_llm: bool,
    enable_asr: bool,
    enable_tts: bool,
    enable_align: bool,
    enable_rvc: bool,
    enable_denoise: bool,
    desktop_allowance_bytes: int,
    retention_days: int,
    desktop_allowance_basis: str,
    desktop_allowance_note: str,
    install_on_submit: bool | None = None,
    enable_image: bool | None = None,
    enable_audio: bool | None = None,
    enable_segment: bool | None = None,
    enable_video: bool | None = None,
    capability: CapabilityRecord | None = None,
    routes: tuple[RouteRecord, ...] = (),
    local_models: tuple[LocalModelRecord, ...] = (),
    upstreams: tuple[UpstreamRecord, ...] = (),
    advertise: tuple[str, ...] = (),
    tailscale_advertise: tuple[str, ...] = (),
    lan_advertise: tuple[str, ...] = (),
    open_pairing: bool = DEFAULT_OPEN_PAIRING,
    tts_engines: tuple[EngineFootprint, ...] = (),
    carried_tables: dict[str, Any] | None = None,
) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    os.chmod(home, 0o700)
    path = config_path(home)
    document: dict[str, Any] = {
        "server": {"name": name, "host": host, "port": port},
        "auth": {"token": token, "open_pairing": open_pairing},
        "backend": {"kind": backend_kind},
        "jobs": {
            "enable_echo": enable_echo,
            "enable_llm": enable_llm,
            "enable_asr": enable_asr,
            "enable_tts": enable_tts,
            "enable_align": enable_align,
            "enable_rvc": enable_rvc,
            "enable_denoise": enable_denoise,
            "enable_image": (
                _kept_jobs_flag(home, "enable_image", DEFAULT_ENABLE_IMAGE)
                if enable_image is None
                else enable_image
            ),
            "enable_audio": (
                _kept_jobs_flag(home, "enable_audio", DEFAULT_ENABLE_AUDIO)
                if enable_audio is None
                else enable_audio
            ),
            "enable_segment": (
                _kept_jobs_flag(home, "enable_segment", DEFAULT_ENABLE_SEGMENT)
                if enable_segment is None
                else enable_segment
            ),
            "enable_video": (
                _kept_jobs_flag(home, "enable_video", DEFAULT_ENABLE_VIDEO)
                if enable_video is None
                else enable_video
            ),
            "install_on_submit": (
                _kept_jobs_flag(home, "install_on_submit", DEFAULT_INSTALL_ON_SUBMIT)
                if install_on_submit is None
                else install_on_submit
            ),
            "retention_days": retention_days,
        },
        "accelerator": {
            "desktop_allowance_bytes": desktop_allowance_bytes,
            "desktop_allowance_basis": desktop_allowance_basis,
        },
    }
    if desktop_allowance_basis not in DESKTOP_BASES:
        raise ConfigError(
            f"desktop_allowance_basis {desktop_allowance_basis!r} is not one of "
            f"{list(DESKTOP_BASES)}"
        )
    if desktop_allowance_note:
        document["accelerator"]["desktop_allowance_note"] = desktop_allowance_note
    if capability is not None:
        document["capability"] = capability.to_dict()
    if advertise:
        document["server"]["advertise"] = list(advertise)
    if tailscale_advertise:
        document["server"]["tailscale_advertise"] = list(tailscale_advertise)
    if lan_advertise:
        document["server"]["lan_advertise"] = list(lan_advertise)
    if routes:
        document["routes"] = {entry.capability: entry.model for entry in routes}
    if tts_engines:
        document["tts"] = {entry.engine: entry.to_dict() for entry in tts_engines}
    if local_models:
        document["local_models"] = {
            entry.capability: entry.model for entry in local_models
        }
    if upstreams:
        document["upstreams"] = {
            entry.name: (
                {"key": entry.key}
                if UPSTREAM_FIELD[entry.name] == "key"
                else {"url": entry.url}
            )
            for entry in upstreams
        }
    for table_name, table in _unowned_tables(path).items():
        document[table_name] = table
    for table_name, table in (carried_tables or {}).items():
        if table_name in document:
            raise ConfigError(
                f"carried_tables names [{table_name}], which this writer already "
                "owns. A table with two writers is a table whose value depends on "
                "which one ran last; carry the tables section 2 added and nothing "
                "else."
            )
        document[table_name] = table
    import tempfile
    data = tomli_w.dumps(document).encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix="config-", suffix=".tmp", dir=home)
    staged = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(staged, 0o600)
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)
    return path


def config_mode(path: Path) -> str:
    return oct(stat.S_IMODE(path.stat().st_mode))


