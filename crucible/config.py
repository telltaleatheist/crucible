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

#: Whether reaching an engine is the whole of the authorisation.
#:
#: THE ONE OWNER OF THIS FACT. `_open_pairing` returns it for an absent key,
#: `write_config` writes it for a config that states nothing, and the test
#: fixture builds servers with it. Flipping this line flips the product, which
#: is the property a default is supposed to have and did not when three files
#: each said `True` on their own account.
#:
#: TRUE by ruling (Owen, 2026-09-17): *"ollama allows anybody to connect if they
#: can reach it. make that the case with crucible servers as well"*. See
#: `crucible/connect.py` for what that does and does not expose.
DEFAULT_OPEN_PAIRING = True
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
class LocalModelRecord:
    """One `[local_models]` entry: a class, and the local model an APP chose.

    The mirror of `RouteRecord` above, with the same absence rule: no record
    means the class takes whatever `capability` decides best-first, and
    "automatic" is never written as a value, so "did anyone choose" is one
    question with one answer rather than a value spelled two ways.

    INTENT.md gives the APP the choice of its own models — BookForge its
    voices, Foundry its reading and language models — and Crucible the running
    of them. This table is where that choice is kept so it survives a restart.
    Whether the choice still FITS is not recorded here: that is a fact about a
    card, `capability` owns it, and a stored answer would go stale the first
    time the config moved to another machine.
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
    #: ADDRESSES SOMETHING ELSE FORWARDS TO THIS SERVER FROM, stated because
    #: they cannot be derived.
    #:
    #: `reachable_urls` answers "where am I" by looking at the bind and, for a
    #: wildcard, at this machine's own interfaces. That is right and it is
    #: complete for a server whose reachability is its own. It is NOT complete
    #: when something outside the server's world creates the reachability: the
    #: engine in WSL binds 127.0.0.1, correctly says so, and is reachable from
    #: another machine anyway because `tailscale serve` on the Windows side
    #: forwards into the guest. The guest cannot see that and never will.
    #:
    #: So the fact is DECLARED, once, here — and it is added to what the server
    #: derives, never substituted for it. The loopback line is what an app on
    #: this machine needs; this is what an app on another machine needs; both
    #: are true at the same time and the console offers both.
    #:
    #: Empty is the normal case: a server whose bind is already reachable
    #: (0.0.0.0 on a Mac) enumerates its interfaces and needs no help.
    #:
    #: DEFAULTED, and placed here with the other defaulted fields for the
    #: reason dataclasses require: a field with a default cannot precede one
    #: without. It was briefly required, which broke every construction of a
    #: Config that is not the parser -- `Config.__init__() missing 1 required
    #: positional argument` in four llama-engine tests. `()` is the honest
    #: default anyway: nothing forwards here unless somebody says so.
    advertise: tuple[str, ...] = ()
    # A host-owned projection, kept separate from operator-authored addresses.
    tailscale_advertise: tuple[str, ...] = ()
    #: The SAME shape for the LAN door (`crucible lan`), and separate from
    #: `tailscale_advertise` for the same reason that one is separate from
    #: `advertise`: each is owned by a different thing, and one list holding
    #: two owners' entries cannot be withdrawn by either without guessing
    #: which rows were whose. Disabling the LAN door must not silently drop
    #: a tailnet address.
    lan_advertise: tuple[str, ...] = ()
    #: `[auth] open_pairing` — whether reaching this engine is the whole of the
    #: authorisation (Owen, 2026-09-17; see `crucible/connect.py`). TRUE is the
    #: ruled default and what an absent key means, so every config written before
    #: this field reads as open, which is the behaviour that was asked for. Set it
    #: false to put the approval step back.
    open_pairing: bool = True
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
    #: `[local_models]` — the local model an app CHOSE for a class, where it
    #: chose one. Empty means every class is decided automatically, which is
    #: what every config written before this field says, so an old one needs
    #: no migration.
    local_models: tuple[LocalModelRecord, ...] = ()

    def route_model(self, capability: str) -> str | None:
        """The upstream model this class runs on, or None because it runs local."""
        for entry in self.routes:
            if entry.capability == capability:
                return entry.model
        return None

    def local_model(self, capability: str) -> str | None:
        """The local model an app chose for this class, or None for automatic.

        None is not "nothing fits" — that is `capability`'s answer and it says
        so with an arithmetic reason. None here is the narrower fact that
        nobody has stated a preference, which is the common case.
        """
        for entry in self.local_models:
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


def _open_pairing(table: dict[str, Any]) -> bool:
    """`[auth] open_pairing`, defaulting to TRUE when the key is absent.

    Absent means open, and that is not a fallback hiding a missing value: it is
    the ruled default stated once. A config written before this field existed
    describes a server whose behaviour is now open, and reading it as closed
    would make every existing machine disagree with the ruling.

    A non-boolean is REFUSED rather than coerced. `open_pairing = "false"` is a
    string, is truthy, and would silently open a door its operator just tried to
    shut — which is the one mistake this field must never make quietly.
    """
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


def _advertised(table: dict[str, Any]) -> tuple[str, ...]:
    """`[server] advertise` — authorities something forwards to this server on.

    ABSENT IS THE NORMAL CASE and means "nothing does", which is why this is
    not `_require`: almost every server's reachability is its own, and only one
    whose address is manufactured outside itself has anything to declare.

    Each entry is an AUTHORITY — `host` or `host:port` — not a URL. The scheme
    is this server's own and a path would have nowhere to go, which is the same
    reasoning `pairing_line` gives for using only a URL's authority. A bare host
    takes the server's port, because the overwhelmingly common case is a
    forward that keeps the number.
    """
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
        # REFUSED, NOT TRIMMED. A scheme here means somebody believes this
        # field takes URLs, and quietly dropping it would leave them believing
        # it — including the day they write `https://`, which this would
        # silently serve over http.
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


def _local_model_records(table: dict[str, Any]) -> tuple[LocalModelRecord, ...]:
    """`[local_models]`, validated as classes that can have a selection.

    A hand-edited config is refused HERE for the reason `[routes]` is: a server
    that started holding a selection for a class which cannot have one would
    answer every settings read with a fact nothing can act on.

    What is NOT asked here is whether the model exists on this backend or fits
    this card. Both are questions about a machine this process has not probed
    yet, `capability` is their one owner, and answering them twice is how two
    doors come to disagree (ARCHITECTURE.md R1). A selection naming a model
    this backend cannot run is reported BY `capability`, with its own reason.
    """
    from .capability import BY_NAME, CLASSES

    section = table.get("local_models")
    if section is None:
        return ()
    if not isinstance(section, dict):
        raise ConfigError("config key local_models must be a table")
    selectable = [entry.name for entry in CLASSES if entry.candidates is not None]
    found: list[LocalModelRecord] = []
    for name in sorted(section):
        entry = BY_NAME.get(name)
        if entry is None or entry.candidates is None:
            raise ConfigError(
                f"config [local_models]: local_model_not_selectable {name!r}; "
                f"only {selectable} choose a local model"
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
        advertise=_advertised(table),
        tailscale_advertise=_advertised({"server": {"advertise": table.get("server", {}).get("tailscale_advertise", [])}}),
        lan_advertise=_advertised({"server": {"advertise": table.get("server", {}).get("lan_advertise", [])}}),
        token=_require(table, "auth", "token", str),
        open_pairing=_open_pairing(table),
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
        local_models=_local_model_records(table),
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
    #: `[local_models]`, on the same terms as `routes` above: defaulted to
    #: empty, and written only when there is one.
    local_models: tuple[LocalModelRecord, ...] = (),
    upstreams: tuple[UpstreamRecord, ...] = (),
    advertise: tuple[str, ...] = (),
    tailscale_advertise: tuple[str, ...] = (),
    lan_advertise: tuple[str, ...] = (),
    open_pairing: bool = DEFAULT_OPEN_PAIRING,
    #: Whole top-level tables to copy in VERBATIM, or None.
    #:
    #: `crucible init --config-from` (PHASE15-HOST.md 4.3) is the one caller:
    #: when the Windows host moves a Crucible into the WSL guest it carries
    #: `[routes]` and `[upstreams]` across, and those tables' SHAPE belongs to
    #: section 2 and to whatever reads them — not to this writer, which would
    #: otherwise have to grow a parameter per upstream and a second
    #: declaration of a document somebody else owns. Copied and never merged
    #: key by key: a key this build does not know about is still the
    #: operator's, and dropping it silently on an upgrade is how a
    #: configuration quietly stops meaning what it said.
    carried_tables: dict[str, Any] | None = None,
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
        },
        "accelerator": {"desktop_allowance_bytes": desktop_allowance_bytes},
    }
    if capability is not None:
        document["capability"] = capability.to_dict()
    if advertise:
        document["server"]["advertise"] = list(advertise)
    if tailscale_advertise:
        document["server"]["tailscale_advertise"] = list(tailscale_advertise)
    if lan_advertise:
        document["server"]["lan_advertise"] = list(lan_advertise)
    if routes:
        # Only when there is one. An empty `[routes]` table and no table at all
        # read the same, and writing the empty one would put a section in every
        # config on earth to say nothing.
        document["routes"] = {entry.capability: entry.model for entry in routes}
    if local_models:
        # Only when there is one, for the reason `routes` gives just above.
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
    # Carried tables go in AFTER `routes`/`upstreams`, so a caller that states a
    # table twice — once as the typed parameter, once as a carried copy — is
    # refused rather than silently having one of the two win.
    for table_name, table in (carried_tables or {}).items():
        if table_name in document:
            raise ConfigError(
                f"carried_tables names [{table_name}], which this writer already "
                "owns. A table with two writers is a table whose value depends on "
                "which one ran last; carry the tables section 2 added and nothing "
                "else."
            )
        document[table_name] = table
    # Serialize before touching the installed file, then replace atomically.
    # A failed write/reinstall must not truncate the token and provider keys.
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


# THE PAIRING FILE lives in `crucible/pairing.py` and nowhere else.
#
# It was briefly written twice — once here (POSIX, `os.chmod` 0600) and once
# there (both platforms, with `icacls` on Windows) — because two builds of
# PHASE15 section 3.6 landed on two branches. Two writers of one file is two
# answers to "who may read this token", so the POSIX-only one is gone and
# `pairing.write_pairing_file` / `pairing.pairing_file_path` are the names.
# `crucible init` and `crucible service install` call them through
# `cli._write_pairing_file`, which is what turns (name, port, token) into the
# loopback LINE those functions write.
