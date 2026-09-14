"""Config and the bearer token.

Everything Crucible keeps on disk lives under one root:

    <CRUCIBLE_HOME>/config.toml      mode 0600, holds the token
    <CRUCIBLE_HOME>/jobs/<id>/       job scratch (inputs/, artifacts/)
    <CRUCIBLE_HOME>/uploads/         blobs from POST /uploads

`CRUCIBLE_HOME` defaults to `~/.crucible` and is read from the environment on every
call, so a test (or a second server on one host) can point it somewhere else.
"""

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

from .errors import ConfigError
from .upstreams import UPSTREAM_FIELD, UPSTREAM_NAMES, UpstreamRecord

CRUCIBLE_HOME_ENV = "CRUCIBLE_HOME"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7100
TOKEN_BYTES = 32

#: VRAM the host's own desktop holds that is not anybody's job. A headless Linux
#: box wants 0; a Windows machine running WSL2 wants about 3 GiB, because the
#: desktop compositor, the browser and the editor are all on the same card and
#: the WSL2 driver shim does not list them as compute apps. The accelerator guard
#: subtracts this before calling unaccounted VRAM "somebody else's job"
#: (crucible/accelerator.py). It is a declared fact about the host, written by
#: `crucible init`, not a fudge factor the code picks.
#:
#: **This is the `cuda-linux` number and only that one.** See
#: `default_desktop_allowance_bytes` below for why the Mac cannot share it.
DEFAULT_DESKTOP_ALLOWANCE_BYTES = 3 * 1024 ** 3

#: The share of unified memory `mlx-darwin` reserves for the machine itself.
#:
#: A discrete card and a unified pool are not the same question wearing different
#: numbers. On `cuda-linux` the desktop's appetite is roughly CONSTANT — a
#: compositor and a browser want about the same VRAM on a 12 GB card as on a
#: 24 GB one — so a fixed byte count is the honest shape. On `mlx-darwin` the
#: allowance has to cover the entire operating system and every app on it, out of
#: the same pool the model allocates from, and that scales with the machine: 3 GiB
#: is defensible on a 16 GB Mac mini and absurd on a 192 GB Studio.
#:
#: 25% is not picked. It is the complement of Metal's own
#: `recommendedMaxWorkingSetSize`, which Apple reports as ~75% of physical memory
#: on Apple Silicon — the working set the platform itself says a GPU process may
#: take before the system starts suffering.
#:
#: The check that this is right is Owen's own long-standing configuration, which
#: predates the rule: he translates with a 4-bit 27B on the Mac and has for
#: months. A flat 3 GiB allowance leaves 60.8 GB "available" on his 64 GB Studio,
#: a best-first walk selects the **bf16** 27B at 55.5 GB, and macOS is left 8.5 GB.
#: At 25% the walk sees 48 GB, refuses bf16 and selects the 4-bit — which is what
#: he already runs. PHASE9-CAPABILITY.md section 1.1 records that disagreement:
#: the rule was wrong, not the operator.
MLX_DESKTOP_ALLOWANCE_FRACTION = 0.25


def default_desktop_allowance_bytes(backend_kind: str, total_bytes: int) -> int:
    """This backend's default host reserve, given the pool it is reserving from.

    `crucible init` calls this AFTER detection, because the answer depends on
    which backend was found and how big its pool is — an argparse default cannot
    know either. An explicit `--desktop-allowance-bytes` still wins over it: this
    is the default for an operator who does not state one, not a ceiling.
    """
    if backend_kind == "mlx-darwin":
        return int(total_bytes * MLX_DESKTOP_ALLOWANCE_FRACTION)
    return DEFAULT_DESKTOP_ALLOWANCE_BYTES


#: Where a Windows server keeps everything, under `%LOCALAPPDATA%`.
#:
#: PHASE15-HOST.md section 3.5: *"On Windows the server's home is
#: `%LOCALAPPDATA%\\Crucible\\` and every subject lives under it."* Not
#: `~/.crucible`, because on Windows a dot-directory in the user profile is
#: roamed by some configurations and backed up by others, and this directory
#: holds tens of gigabytes of GGUF that must never leave the machine. It is
#: also the directory the host pack unpacks beside (section 4.4), so the
#: engine and the weights it reads are under one root.
WINDOWS_HOME_DIRNAME = "Crucible"


def crucible_home() -> Path:
    """The root of this server's state. Honours $CRUCIBLE_HOME on every platform."""
    override = os.environ.get(CRUCIBLE_HOME_ENV)
    if override is not None and override != "":
        return Path(override).expanduser()
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if local is None or local == "":
            # Not a fallback to `~/.crucible`: a Windows session without
            # LOCALAPPDATA is broken in a way that would make every path this
            # server writes wrong, and putting tens of gigabytes somewhere
            # else quietly is worse than saying so.
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
    """A 32-byte urlsafe bearer token."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def default_server_name() -> str:
    return f"crucible@{socket.gethostname()}"


@dataclass(frozen=True)
class RouteRecord:
    """One `[routes]` entry: a capability class, and the upstream model it runs on.

    PHASE15-HOST.md section 2. There is no record for a class that runs
    LOCALLY: an absent key and `route = "local"` mean the same thing and
    `"local"` is never written, so "is this class routed" is one question with
    one answer — is there a record — rather than a value that could be spelled
    two ways.
    """

    capability: str
    model: str


@dataclass(frozen=True)
class CapabilityRow:
    """One capability class's verdict, as `crucible capability` decided it.

    A row is a RECORD, not an authority. `[jobs] enable_*` stays the single owner
    of what this server offers (ARCHITECTURE.md R1); this says what the numbers
    were when somebody decided it, so a refusal can name the number that turned
    the class off instead of telling an operator to flip a flag that will OOM
    (PHASE9-CAPABILITY.md section 2.1).

    `selected` is `""` rather than absent when nothing fit, and `shortfall_bytes`
    is `0` rather than absent when something did: TOML has no null, and a key that
    comes and goes would make "no candidate fit" and "this config predates the
    field" the same reading.
    """

    capability: str
    enabled: bool
    selected: str
    reason: str
    shortfall_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "enabled": self.enabled,
            "selected": self.selected,
            "reason": self.reason,
            "shortfall_bytes": self.shortfall_bytes,
        }


@dataclass(frozen=True)
class CapabilityRecord:
    """`[capability]` — what the card was, and what was decided from it.

    The three scalars are the INPUTS to the decision, kept so a reader can tell a
    stale record from a current one. `crucible doctor` compares `total_bytes`
    against the card it detects now, which is how a swapped GPU is noticed
    without anybody writing down a date: the number that matters is the one the
    decision was made on, not the day it was made.
    """

    backend_kind: str
    total_bytes: int
    desktop_allowance_bytes: int
    rows: tuple[CapabilityRow, ...]

    def row(self, capability: str) -> CapabilityRow | None:
        for entry in self.rows:
            if entry.capability == capability:
                return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_kind": self.backend_kind,
            "total_bytes": self.total_bytes,
            "desktop_allowance_bytes": self.desktop_allowance_bytes,
            "classes": [entry.to_dict() for entry in self.rows],
        }


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
    #: `denoise` shares the `rvc` env, and it still gets a flag of its own: a
    #: host may have the env and the RVC models and no separator checkpoint, or
    #: the other way round, and one flag for both would advertise a job type
    #: whose first request refuses.
    enable_denoise: bool
    desktop_allowance_bytes: int
    #: Capability flags this config did not carry, so they were read as off.
    #: Empty for a config written by this build. `crucible doctor` prints it, so
    #: "the type is not enabled" and "the config predates the type" are told
    #: apart by a reader rather than guessed at.
    flags_absent: tuple[str, ...] = ()
    #: What `crucible capability` decided on this host, or None when nothing has
    #: decided anything here yet — a config written by `crucible init` alone, or
    #: one written before this field existed. None is a REPORTED state, not a
    #: guess: the refusal in `crucible/jobs/__init__.py` says "no selection has
    #: been recorded here" rather than inventing a reason for a disabled type.
    capability: CapabilityRecord | None = None
    #: `[routes]` — where each routable class's work runs (PHASE15-HOST.md
    #: section 2). Empty means every class is local, which is what every server
    #: written before this phase says, so an old config needs no migration.
    routes: tuple[RouteRecord, ...] = ()
    #: `[upstreams.*]` — the services this server may forward a chat to on the
    #: operator's account. PRESENT MEANS CONFIGURED: `load_config` refuses an
    #: entry missing its one field, so nothing downstream has to ask twice.
    upstreams: tuple[UpstreamRecord, ...] = ()

    def route_model(self, capability: str) -> str | None:
        """The upstream model this class runs on, or None because it runs local."""
        for entry in self.routes:
            if entry.capability == capability:
                return entry.model
        return None

    def upstream(self, name: str) -> UpstreamRecord | None:
        """This upstream's record, or None because nobody configured it."""
        for entry in self.upstreams:
            if entry.name == name:
                return entry
        return None

    def classes_routed_to(self, name: str) -> tuple[str, ...]:
        """Every class whose route names this upstream, in `[routes]` order.

        What `upstream_in_use` reports: a caller removing a key is owed the list
        of things that would stop working, in one refusal, rather than one
        refusal per attempt.
        """
        return tuple(
            entry.capability
            for entry in self.routes
            if entry.model.partition("/")[0] == name
        )

    def adopt(self, fresh: "Config") -> None:
        """Take on a re-read of this same file, in place. **One Config per process.**

        PHASE13-OPERATOR.md section 3.4. `crucible install` rewrites `[jobs]` and
        `[capability]` while this server is running, and a client that asked for
        the install must see the new job type before the task says `done`. So the
        config has to change under a live app — and the only honest way to do
        that is for there to go on being exactly ONE config object.

        **Why not simply hand out a new one.** Every route in `crucible/api.py`
        closes over this object; so do `Residency`, `JobStore`, every job-type
        plugin and the streaming manager. Replacing the app's reference would
        leave all of those reading the old flags while `/v1/setup` read the new
        ones — one fact with two owners and nothing comparing them, which is the
        whole of ARCHITECTURE.md section 1. Rebinding every holder is the same
        bug with more places to forget.

        **Why this is not a licence to mutate a frozen dataclass.** `frozen=True`
        stays, and `object.__setattr__` appears exactly here, in a method whose
        name says what it is for. Nothing else in the package writes to a Config,
        and a test asserts that
        (`tests/test_tasks_api.py`). The immutability being
        protected is *"a config is not edited field by field from wherever"*, and
        that is intact: this replaces the whole document at once, from a file.

        **ROUTES AND UPSTREAMS TRAVEL THE SAME WAY** (PHASE15-HOST.md section
        2), and they are why this method matters twice as much as it did: a
        `PUT /v1/settings` writes the file and then adopts it, so the chat door
        forwards to the key that was pasted a millisecond ago without a
        restart. The loop below is over `__dataclass_fields__` rather than a
        list of names for exactly this reason — a field added to `Config` is
        adopted on the day it is added, and a phase that forgot to extend a
        hand-written list would leave half the server reading the old document.

        IDENTITY IS REFUSED, CAPABILITY IS ADOPTED. A fresh config from a
        different path or home is not a re-read of this one, it is a different
        server, and adopting it would silently move where this process keeps its
        jobs. The token, the name, the host and the port are adopted, because
        `crucible init --force` is the only thing that changes them and it tells
        the operator every client will need the new one.
        """
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


def _require(table: dict[str, Any], section: str, key: str, kind: type) -> Any:
    if section not in table:
        raise ConfigError(f"config is missing the [{section}] section")
    if key not in table[section]:
        raise ConfigError(f"config is missing {section}.{key}")
    value = table[section][key]
    wrong_type = not isinstance(value, kind)
    # bool is a subclass of int; a bool where an int is wanted is still wrong.
    if kind is int and isinstance(value, bool):
        wrong_type = True
    if wrong_type:
        raise ConfigError(
            f"config key {section}.{key} must be {kind.__name__}, got "
            f"{type(value).__name__}"
        )
    return value


def _capability_flag(table: dict[str, Any], key: str) -> bool:
    """`[jobs] enable_<type>`, where ABSENT means off and that is not a fallback.

    Every other key in this file is required, and stays required: a config that
    forgets its token or its backend is broken, and guessing one would hide the
    break. A capability flag is a different animal. `enable_rvc` was not missing
    from a config written in phase 2 — `rvc` did not exist. Demanding it means
    that adding a job type INVALIDATES EVERY CONFIG IN EXISTENCE, and the only
    repair on offer, `crucible init --force`, mints a new token and breaks every
    client that had one.

    Found on Owen's Mac on 2026-09-13: its server had been running since before
    `asr`, `tts`, `align` and `rvc` were built, and after an upgrade the CLI
    could not read its own config to print its own token.

    So: absent means off. It fails SAFELY (a capability cannot switch itself on)
    and it fails VISIBLY — `crucible doctor` lists which flags were absent, and
    `/v1/info`'s `job_types` shows the type is not there. A wrong type or an
    unknown key in `[jobs]` is still a refusal; it is only absence that is
    allowed to mean "written before this existed".
    """
    section = table.get("jobs")
    if section is None:
        raise ConfigError("config is missing the [jobs] section")
    if key not in section:
        return False
    return _require(table, "jobs", key, bool)


#: Every capability flag, in the order `crucible init` writes them.
CAPABILITY_FLAGS: tuple[str, ...] = (
    "enable_echo",
    "enable_llm",
    "enable_asr",
    "enable_tts",
    "enable_align",
    "enable_rvc",
    "enable_denoise",
)


#: The scalars of `[capability]`, and the keys of one `[[capability.classes]]`.
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
    "shortfall_bytes": int,
}


def _capability_record(table: dict[str, Any]) -> CapabilityRecord | None:
    """`[capability]`, or None when this config has never had one written.

    ABSENT means "nobody has decided", exactly as `_capability_flag`'s absence
    means "written before this existed", and for the same reason: a config from
    phase 8 must still load on a phase 9 build. Where the two differ is what an
    absence is allowed to become. A missing FLAG becomes `False`, because a
    capability that switches itself on is the dangerous direction. A missing
    RECORD becomes None and stays None — it must never become an empty record,
    because an empty record reads as "the card was probed and nothing fit", which
    is a different and false statement about the host.

    PRESENT and malformed is a refusal, like every other table in this file: a
    `[capability]` block with a misspelled key must not load with that class
    silently missing and have a refusal claim no selection was ever run.
    """
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
                f"{where}: unknown key(s) {unknown}; a row takes exactly "
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
            )
        )
    return CapabilityRecord(
        backend_kind=section["backend_kind"],
        total_bytes=section["total_bytes"],
        desktop_allowance_bytes=section["desktop_allowance_bytes"],
        rows=tuple(rows),
    )


def _upstream_records(table: dict[str, Any]) -> tuple[UpstreamRecord, ...]:
    """`[upstreams.*]`, in `UPSTREAM_NAMES` order, or a refusal naming the fault.

    ABSENT IS THE ONLY WAY TO BE UNCONFIGURED. An entry that exists carries the
    one field its upstream takes, or the config does not load — there is no
    `[upstreams.anthropic]` with no key, because a record like that would make
    `configured` a second fact beside the record's own existence and the two
    would eventually disagree (ARCHITECTURE.md R1).

    A whole `[upstreams]` table is optional: every config written before this
    phase has none, and "this server forwards nowhere" is the honest reading of
    that rather than a missing piece.
    """
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
    """`[routes]`, validated against the upstreams that were just read.

    A hand-edited config is refused HERE, with the same three names
    `PUT /v1/settings` refuses by (PHASE15-HOST.md section 3.2), because a
    server that started holding a route it cannot serve would spend the rest of
    its life refusing one capability with a sentence about the wrong thing.

    The order matters: upstreams first, then routes, exactly as one PUT applies
    them, so a config file and a patch cannot disagree about whether a route is
    servable.
    """
    from .capability import ROUTABLE_CLASSES

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


def load_config(home: Path | None = None) -> Config:
    """Read config.toml. Raises ConfigError naming the missing piece."""
    root = home if home is not None else crucible_home()
    path = config_path(root)
    if not path.exists():
        raise ConfigError(
            f"no config at {path} — run `crucible init` (or set {CRUCIBLE_HOME_ENV})"
        )
    try:
        with path.open("rb") as handle:
            table = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc

    upstreams = _upstream_records(table)
    return Config(
        path=path,
        home=root,
        name=_require(table, "server", "name", str),
        host=_require(table, "server", "host", str),
        port=_require(table, "server", "port", int),
        token=_require(table, "auth", "token", str),
        backend_kind=_require(table, "backend", "kind", str),
        enable_echo=_capability_flag(table, "enable_echo"),
        enable_llm=_capability_flag(table, "enable_llm"),
        enable_asr=_capability_flag(table, "enable_asr"),
        enable_tts=_capability_flag(table, "enable_tts"),
        enable_align=_capability_flag(table, "enable_align"),
        enable_rvc=_capability_flag(table, "enable_rvc"),
        enable_denoise=_capability_flag(table, "enable_denoise"),
        flags_absent=tuple(
            flag for flag in CAPABILITY_FLAGS if flag not in table.get("jobs", {})
        ),
        desktop_allowance_bytes=_require(
            table, "accelerator", "desktop_allowance_bytes", int
        ),
        capability=_capability_record(table),
        routes=_route_records(table, upstreams),
        upstreams=upstreams,
    )


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
    desktop_allowance_bytes: int,
    #: Defaulted, and it is the ONE flag that is, because a config's every other
    #: writer passes it: `_write_capability` builds its call from
    #: `CAPABILITY_FLAGS`, so it always states this, and `crucible init` states
    #: it too. What the default serves is a caller written before this job type
    #: existed — a test, a script — for which `False` is the same answer
    #: `_capability_flag` gives an absent key, and the safe direction.
    enable_denoise: bool = False,
    capability: CapabilityRecord | None = None,
    #: `[routes]` and `[upstreams.*]`. Defaulted to empty for the same reason
    #: `enable_denoise` is defaulted: a caller written before this phase states
    #: neither, and empty is what such a config already means. **Every caller
    #: that REWRITES an existing config must pass the loaded values**, or the
    #: rewrite silently unroutes a server — `cli._write_capability` does, and a
    #: test pins it.
    routes: tuple[RouteRecord, ...] = (),
    upstreams: tuple[UpstreamRecord, ...] = (),
) -> Path:
    """Write config.toml at mode 0600 under a 0700 home. Returns the path.

    `capability` is optional and the default writes NO `[capability]` table, which
    is the honest record for `crucible init`: init takes the operator's
    `--enable-*` flags at their word and probes nothing, so it has no verdict to
    write down. `crucible capability --write` and `crucible install` are the two
    doors that have one.

    **A key lands in this file and nowhere else.** The document is created at
    0600 under a 0700 home before a byte of it is written, which is the same
    protection the token has had since phase 1 — PHASE15-HOST.md section 2:
    *"a key here is no worse than the token"*.
    """
    home.mkdir(parents=True, exist_ok=True)
    os.chmod(home, 0o700)
    path = config_path(home)
    document: dict[str, Any] = {
        "server": {"name": name, "host": host, "port": port},
        "auth": {"token": token},
        "backend": {"kind": backend_kind},
        "jobs": {
            "enable_echo": enable_echo,
            "enable_llm": enable_llm,
            "enable_asr": enable_asr,
            "enable_tts": enable_tts,
            "enable_align": enable_align,
            "enable_rvc": enable_rvc,
            "enable_denoise": enable_denoise,
        },
        "accelerator": {"desktop_allowance_bytes": desktop_allowance_bytes},
    }
    if capability is not None:
        document["capability"] = capability.to_dict()
    if routes:
        # Only when there is one. An empty `[routes]` table and no table at all
        # read the same, and writing the empty one would put a section in every
        # config on earth to say nothing.
        document["routes"] = {entry.capability: entry.model for entry in routes}
    if upstreams:
        document["upstreams"] = {
            entry.name: (
                {"key": entry.key}
                if UPSTREAM_FIELD[entry.name] == "key"
                else {"url": entry.url}
            )
            for entry in upstreams
        }
    # Create with 0600 from the outset so the token is never briefly world-readable.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(tomli_w.dumps(document).encode("utf-8"))
    os.chmod(path, 0o600)
    return path


def config_mode(path: Path) -> str:
    return oct(stat.S_IMODE(path.stat().st_mode))


def pairing_path(home: Path | None = None) -> Path:
    """`<CRUCIBLE_HOME>/pairing` — the file an app on this machine reads."""
    return (home if home is not None else crucible_home()) / "pairing"


def write_pairing_file(home: Path, *, name: str, port: int, token: str) -> Path:
    """One loopback pairing line, mode 0600, with a trailing newline.

    PHASE15-HOST.md section 3.6. `crucible init`, `crucible service install`
    and a token rotation all write it, and an app on the same machine reads it
    instead of asking a person to type a secret it could have read.

    **The line is always the loopback one**, whatever the server is bound to.
    The file answers one question — *an app on THIS machine wants in* — and the
    answer to that is never a LAN address: a wildcard-bound server has no
    loopback entry in `reachable_urls` at all, so a file built from that list
    would hand a local app whichever interface the OS happened to list first.

    0600, like the config, because the line carries the token in its fragment.
    On Windows `os.chmod` cannot express that and the host writes the file with
    an ACL instead (section 4.3); the mode call is harmless there and the write
    still happens, so this function has ONE body rather than a platform branch
    that would let the two drift.
    """
    from .pairing import pairing_line

    home.mkdir(parents=True, exist_ok=True)
    path = pairing_path(home)
    line = pairing_line(name, f"http://{DEFAULT_HOST}:{port}", token)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write((line + "\n").encode("utf-8"))
    os.chmod(path, 0o600)
    return path
