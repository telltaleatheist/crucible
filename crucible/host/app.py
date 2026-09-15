"""`crucible host` — the wiring. PHASE15-HOST.md section 4.

Everything this file does is decided somewhere else: `presence.py` decides what
state the machine is in, `menu.py` decides what the menu says, `installer.py`
decides what an install is, `door.py` decides who may ask for one. This is the
loop that holds them, and it is deliberately dull.

THE SHAPE, IN ORDER
--------------------
1. One host per machine (`host_already_running`): a second tray would boot the
   same distro twice and watch the first one's recoveries.
2. The Startup item, written if absent — 4.1 makes these verbs the ONE owner of
   that file and `install.ps1` calls the verb rather than writing a `.lnk`.
3. Presence: distro present → boot the guest; absent → start the
   `llama-windows` server as a child (section 0's amendment: Windows IS a
   backend, and the machine has an engine either way).
4. The pairing file, so an app on this machine never asks anybody to type a
   token.
5. The door, on loopback.
6. The tray, and a watch every 15 s.
"""

from __future__ import annotations

import os
import sys
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .. import API_VERSION, VERSION
from .. import peer as peer_module
from ..pairing import parse_pairing_line
from . import installer, menu, startup
from .catalog import CatalogPort, GuestCatalog, HttpCatalog
from .door import OrchestratorDoor, serve
from .errors import HostError
from .log import HostLog
from .menu import Distro, Engine, Owner
from .paths import (
    CONSOLE_CMD,
    ENGINE_PORT,
    console_cmd_path,
    crucible_root,
    door_url,
    engine_url,
    host_pack_dir,
    log_path,
    previous_log_path,
)
from .presence import Presence, PresenceWatcher
from .runner import ProcessRunner, Runner
from ..tasks import HOST_DOOR_ENV
from .wsl_states import CRUCIBLE_DISTRO

#: A second tray is refused by a file, not by a mutex: the file NAMES the
#: process that holds it, so "another one is running" is a sentence with a pid
#: in it rather than a silent exit.
LOCK_NAME = "host.pid"

#: The release the guest install fetches. The host ships AT the server's
#: version — the one legitimate default in here, for `install.ts`'s reason:
#: the bootstrapper ships at the server's version, so the default IS the
#: answer rather than a guess at one.
DEFAULT_RELEASE = VERSION

INSTALL_SH_URL = (
    "https://github.com/telltaleatheist/crucible/releases/download/"
    "v{release}/install.sh"
)


@dataclass
class HostContext:
    """Everything the loop holds. One object, so a test can build one."""

    runner: Runner
    log: HostLog
    home: Path
    watcher: PresenceWatcher
    presence: Presence
    release: str = DEFAULT_RELEASE
    #: PHASE17 3.2: what this orchestrator calls itself, on its `/v1/ping`
    #: and in the claim it makes. Composed once, at startup, from the machine
    #: name — the same shape `crucible init` gives a server.
    name: str = ""


def orchestrator_name() -> str:
    """`crucible-orchestrator@<machine>` — PHASE17 3.2.

    Lower-cased, because a name that differs from the same machine's engine
    name only by the case Windows reports is a name two log lines disagree
    about. The hostname and nothing assembled: `socket.gethostname()` is what
    `crucible init` reads for the engine's own name.
    """
    import socket

    return f"crucible-orchestrator@{socket.gethostname().lower()}"


ORCHESTRATOR_GPU = {"vendor": "none", "name": "", "vram_bytes": 0}
"""What an accelerator block says about a process that plays nothing.

Not omitted and not null: a client reading `host.gpu.vram_bytes` must get a
number, and the true number is zero. PHASE17 1: "what accelerator does the
thing that plays nothing have" has exactly one honest answer.
"""

#: PHASE15 4.1a's owner enum, as PHASE17 3.2 spells it on the wire.
OWNER_ON_THE_WIRE = {
    Owner.WSL_UNIT: peer_module.OWNER_WSL_UNIT,
    Owner.HOST_CHILD: peer_module.OWNER_CHILD,
    Owner.FOUND: peer_module.OWNER_FOUND,
}


def engine_token(context: "HostContext") -> str | None:
    """The ENGINE's bearer, from whichever source this machine's owner says.

    PHASE15 4.1a's rule, reused verbatim and for the same reason there is no
    fallback between the two sources: a guest engine's token lives in the line
    the orchestrator COPIED out of the distro, and a host-mode child's is the
    one in the config the orchestrator itself wrote. Reading the wrong one
    gives a 401 against a door that is working perfectly.
    """
    owner = context.presence.owner
    if owner in (Owner.WSL_UNIT, Owner.FOUND):
        line = _guest_line(context)
        if line is None:
            return None
        try:
            return parse_pairing_line(line).token
        except ValueError as exc:
            context.log.write(f"claim: the guest's pairing line will not parse ({exc})")
            return None
    if owner is Owner.HOST_CHILD:
        return read_token(context.home)
    return None


def read_token(home: Path) -> str | None:
    """The engine token out of the host's own config, or None before there is one.

    Parsed with the stdlib's TOML reader and not with a regex: a token is a
    quoted string and `tomllib` is what `crucible/config.py` reads the same
    file with, so the two cannot disagree about escaping.
    """
    import tomllib

    path = Path(home) / "config.toml"
    if not path.is_file():
        return None
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    auth = document.get("auth")
    if not isinstance(auth, dict):
        return None
    token = auth.get("token")
    return token if isinstance(token, str) and token else None


#: PHASE17 2.5 — the table and key a person writes to CONSENT to this
#: orchestrator managing a distro Crucible did not import. `[orchestrator]`
#: because that is what the process reading it IS, and `distro` because the
#: value is a distro's name: the setting is read by the orchestrator, about
#: the orchestrator's own reach, and it belongs to no server.
CONSENT_TABLE = "orchestrator"
CONSENT_KEY = "distro"


def consented_distro(home: Path) -> str | None:
    """The distro this orchestrator was GIVEN permission to manage, or None.

    PHASE17 2.5. Without it the orchestrator manages only the distro Crucible
    imported (`crucible`), and every other engine on the machine is `found` —
    watched, never claimed, never acted on (PHASE15 4.1a). That rule is right
    for a machine nobody has spoken about and wrong for Owen's PC, where the
    engine has lived in `Ubuntu` since before any of this existed: the
    orchestrator can see it, can read its pairing line, could restart its unit
    — and refuses, because it cannot tell that distro apart from a stranger's.
    Consent is how a person tells it apart, by name, once, in the one file on
    the Windows side that is already the orchestrator's own.

    Read with `tomllib` and not a regex, for `read_token`'s reason: this is
    the same document `crucible/config.py` reads, and two parsers for one file
    are two opinions about escaping.

    **A value that is present and unusable is REFUSED, never ignored.** A
    person who wrote `distro = 4` is a person who meant to grant something,
    and an orchestrator that shrugged at it would silently be the unconsented
    one while its config said otherwise — a fact with two owners and nothing
    comparing them (`docs/ARCHITECTURE.md`). Absent is the only quiet answer.
    """
    import tomllib

    path = Path(home) / "config.toml"
    if not path.is_file():
        return None
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise HostError(
            "orchestrator_distro_invalid",
            f"{path} could not be read as TOML ({exc}), so whether this "
            "orchestrator was given a distro to manage cannot be known. It "
            "manages none until the file parses.",
        ) from exc
    table = document.get(CONSENT_TABLE)
    if table is None:
        return None
    if not isinstance(table, dict):
        raise HostError(
            "orchestrator_distro_invalid",
            f"[{CONSENT_TABLE}] in {path} is a {type(table).__name__} and not "
            "a table.",
        )
    if CONSENT_KEY not in table:
        return None
    name = table[CONSENT_KEY]
    if not isinstance(name, str) or name.strip() == "":
        raise HostError(
            "orchestrator_distro_invalid",
            f"{CONSENT_TABLE}.{CONSENT_KEY} in {path} is "
            f"{name!r}; it names a WSL distribution, as `wsl -l -v` spells it "
            '(e.g. distro = "Ubuntu").',
        )
    return name.strip()


def acquire(home: Path) -> Path:
    """One host per machine. Refuses `host_already_running`, naming the pid."""
    home.mkdir(parents=True, exist_ok=True)
    lock = home / LOCK_NAME
    if lock.is_file():
        previous = lock.read_text(encoding="utf-8").strip()
        if previous.isdigit() and _alive(int(previous)):
            raise HostError(
                "host_already_running",
                f"another `crucible host` is running on this machine (pid {previous}). "
                "Two trays would boot the same distro twice and watch each other's "
                "recoveries. Quit that one from its menu, or end that process.",
            )
    lock.write_text(str(os.getpid()), encoding="utf-8")
    return lock


def _alive(pid: int) -> bool:
    """Is this pid a live process? A stale lock must not wedge the tray."""
    if sys.platform == "win32":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # type: ignore[attr-defined]
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        return True
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def server_argv(env: "os._Environ[str] | dict[str, str]") -> list[str]:
    """`crucible serve` as the host's CHILD — the `llama-windows` server.

    Through the pack's own `crucible.cmd`, which is the relocatable entry point
    (4.4): `Scripts\\crucible.exe` bakes the build tree's interpreter path into
    the binary and does not survive the move to `%LOCALAPPDATA%`.
    """
    return [str(console_cmd_path(env)), "serve"]


def server_environment(
    env: "os._Environ[str] | dict[str, str]",
) -> dict[str, str]:
    """The child server's environment: this one, plus where the door is.

    4.7's engine task is the server handing the move to THIS process, and it
    refuses `engine_move_needs_host` when there is nothing to hand it to. The
    fact "a host started me" is not one a server can probe for — 127.0.0.1:
    7101 can be answered by something that is not a host, and a host
    restarting its own door is still the host — so it is STATED, here, by the
    only thing that knows it.

    The token is deliberately NOT passed. The door's bearer is the engine's
    own token, which the child already holds in its config; a copy in an
    environment variable would be a secret with two owners.
    """
    environment = dict(env)
    environment[HOST_DOOR_ENV] = door_url("")
    return environment


def init_argv(env: "os._Environ[str] | dict[str, str]") -> list[str]:
    """The first-run `crucible init` for the host-mode server.

    `--backend llama-windows` and NOT `none`: section 0's amendment. This
    machine runs an engine, it just runs llama.cpp rather than vLLM.
    """
    return [str(console_cmd_path(env)), "init", "--backend", "llama-windows"]


def open_console(home: Path, log: HostLog) -> None:
    """4.2's Open console: the pairing line's URL with `#token=`.

    The DEFAULT browser, deliberately. PHASE13 5.3's hardened-window rule is
    about not putting a token into something that will navigate away with it;
    a browser the person chose is the one place a URL belongs.
    """
    from ..pairing import read_pairing_file

    line = read_pairing_file(home)
    if line is None:
        log.write(
            f"open console: there is no pairing file in {home}, so there is no "
            "server on this machine to open yet"
        )
        return
    # `crucible://name@host:port/#token` -> `http://host:port/#token=...`
    from urllib.parse import urlsplit

    parts = urlsplit(line)
    url = f"http://{parts.netloc.rsplit('@', 1)[-1]}/#token={parts.fragment}"
    log.write(f"open console: {parts.netloc.rsplit('@', 1)[-1]}")
    webbrowser.open(url)


def open_log(log: HostLog) -> None:
    """4.2's Open log. The default handler for a `.log`, which is Notepad."""
    if sys.platform == "win32":
        os.startfile(str(log.path))  # type: ignore[attr-defined]  # noqa: S606
        return
    log.write(f"the log is {log.path}")


def _step(name: str, index: int, total: int = 2) -> "installer.Event":
    """One `step` event, in the shape `crucible/tasks.py` relays verbatim.

    Spelled here rather than inline so the restart cannot drift from the
    move's shape: 4.7 says *"a relay that reshapes is a second owner of the
    shape"*, and two sequences emitting two nearly-identical dicts is how a
    shape acquires a second owner without anybody deciding to give it one.
    """
    return installer.Event("step", {"name": name, "index": index, "total": total})


class Host:
    """The orchestrator's loop. Started by `crucible orchestrator`, stopped by Quit.

    It is the ORCHESTRATOR (PHASE17): the one process on this machine with a
    role of its own, managing exactly one engine and serving no job types. The
    class and its package keep the name `host` — PHASE17 section 7 records why
    the rename stops at the wire.
    """

    def __init__(self, context: HostContext) -> None:
        self._c = context
        self._icon: object | None = None
        self._stop = threading.Event()
        self._door_server: object | None = None
        #: Whether THIS process holds a claim on this machine's engine
        #: (PHASE17 2.1). Not "whether the engine is claimed" — that is the
        #: engine's fact and it is read from `/v1/info`, never mirrored here.
        self._claimed = False

    # -------------------------------------------------------------- presence

    def start(self) -> Presence:
        """Boot whichever server this machine runs, and say which.

        **The ping comes FIRST on a machine with no Crucible distro**, and
        that order is the fix the first real run demanded (2026-09-15). Owen's
        PC runs a Crucible inside `Ubuntu`, installed by hand long before any
        of this: the distro probe truthfully answers `absent`, and the old
        order read that as "no server here" and spawned the `llama-windows`
        child onto a port another Crucible already held. Section 0 is one
        server per machine, and the cheapest way to keep that true is to look
        before starting anything.
        """
        distro, detail = self._c.watcher.probe_distro()
        self._c.log.write(f"presence: {detail}")
        if distro is Distro.PRESENT:
            self._c.presence = self._c.watcher.boot()
        elif self._c.watcher.ping():
            self._c.presence = self._c.watcher.adopt(distro)
        else:
            self._c.presence = self._start_host_mode(distro)
        self._c.log.write(
            f"presence: {self._c.presence.distro.value}/"
            f"{self._c.presence.engine.value}/{self._c.presence.owner.value} — "
            f"{self._c.presence.detail}"
        )
        self._hold()
        return self._c.presence

    def _hold(self) -> None:
        """7b.4c: hold the distro the engine is in, or it goes away by itself."""
        if self._c.presence.owner is Owner.WSL_UNIT:
            name = self._c.watcher._distro  # noqa: SLF001 - one object, one loop
        elif (
            self._c.presence.owner is Owner.FOUND
            and self._c.watcher.found is not None
        ):
            name = self._c.watcher.found.distro
        else:
            # A host-mode child is a Windows process; there is no VM to hold.
            return
        self._c.watcher.hold(name)

    def _start_host_mode(self, distro: Distro) -> Presence:
        """The `llama-windows` server, as this process's child."""
        env = dict(self._c.runner.env)
        cmd = console_cmd_path(env)
        if not Path(str(cmd)).is_file():
            return Presence(
                distro,
                Engine.FAILED,
                f"there is no {CONSOLE_CMD} in {host_pack_dir(env)}, so this host "
                "has no server to start. Reinstall with install.ps1.",
                Owner.NONE,
            )
        config = self._c.home / "config.toml"
        if not config.is_file():
            first = self._c.runner.run(init_argv(env), timeout_s=300.0)
            self._c.log.write(
                f"init: {'ok' if first.ok else first.said()}"
            )
            if not first.ok:
                return Presence(
                    distro, Engine.FAILED, f"crucible init: {first.said()}", Owner.NONE
                )
        self._c.watcher.respawn_host_mode(
            server_argv(env), server_environment(env)
        )
        if self._c.watcher._wait_for_ping(30.0):  # noqa: SLF001 - one object, one loop
            return Presence(
                distro,
                Engine.RUNNING,
                "the Windows engine answered /v1/ping",
                Owner.HOST_CHILD,
            )
        return Presence(
            distro,
            Engine.FAILED,
            "the Windows engine did not answer within 30 s — open the log",
            Owner.NONE,
        )

    # ----------------------------------------------------------- the relation

    def claim(self) -> bool:
        """Tell this machine's engine that this orchestrator manages it.

        **A `found` engine is NEVER claimed** (PHASE15 4.1a, PHASE17 2.1).
        The orchestrator did not start it, has no unit it may name and no
        child it may kill, so a claim would be a statement that is not true:
        `managed_by` would name a door that refuses every verb the field
        implies. It is watched, and that is the whole of the relation with it.

        A claim that fails is a LINE IN THE LOG and never a crash. An engine
        that will not be claimed is still an engine, and a tray that died
        telling it so would take the watch with it; the next down-to-up edge
        tries again.
        """
        owner = self._c.presence.owner
        if owner is Owner.FOUND:
            self._c.log.write(
                "claim: the engine on this machine was already answering when "
                "this orchestrator started, so it is watched and not claimed "
                "(owner=found, PHASE15 4.1a)"
            )
            return False
        if owner not in (Owner.WSL_UNIT, Owner.HOST_CHILD):
            return False
        token = engine_token(self._c)
        if token is None:
            self._c.log.write(
                "claim: this machine's engine token could not be read, so no "
                "claim was made - an engine is not less of an engine for "
                "being unclaimed"
            )
            return False
        try:
            answer = peer_module.claim_engine(
                engine_url(),
                token,
                self._orchestrator_ref(),
                api_version=API_VERSION,
            )
        except peer_module.PeerCallFailed as exc:
            self._c.log.write(f"claim: {exc.code}: {exc.message}")
            return False
        self._c.log.write(
            f"claim: {engine_url()} is managed by {self._c.name} "
            f"(owner={OWNER_ON_THE_WIRE[owner]}, claimed {answer.get('claimed')})"
        )
        self._claimed = True
        return True

    def release_claim(self) -> None:
        """Drop the claim on the way out (PHASE17 2.2).

        A tray that exits leaving `managed_by` pointing at a door that no
        longer answers is PHASE15 3.6's "a file that exists and disagrees is
        worse than none", one layer up.
        """
        if not self._claimed:
            return
        token = engine_token(self._c)
        if token is None:
            return
        try:
            peer_module.release_engine(
                engine_url(),
                token,
                self._orchestrator_ref(),
                api_version=API_VERSION,
            )
            self._c.log.write(f"claim: released {engine_url()}")
        except peer_module.PeerCallFailed as exc:
            # Quitting is not a thing that fails. An engine that could not be
            # told is an engine whose `managed_by` is stale until it restarts,
            # which is a display and not a behaviour.
            self._c.log.write(f"claim: release did not land: {exc.code}: {exc.message}")
        self._claimed = False

    def _orchestrator_ref(self) -> peer_module.Orchestrator:
        return peer_module.Orchestrator(
            name=self._c.name, url=door_url(""), version=VERSION
        )

    @property
    def name(self) -> str:
        """`OrchestratorPort`. What `/v1/ping` on the door calls this process."""
        return self._c.name

    def info(self) -> dict[str, Any]:
        """PHASE17 3.2 - this orchestrator, and its engine READ THROUGH.

        The engine's `/v1/info` is re-read on EVERY request and nothing is
        cached. A cached capability list is this system's one defect in a
        third place: the engine pulls a model, the orchestrator keeps
        answering yesterday's list, and a client picks a model the engine has
        and is told it does not.

        When the engine cannot be read, `capabilities` is `[]` and the
        engine's `name` and `backend` are `null` - the orchestrator does not
        invent an answer for a server that did not give one. `engine.url`
        still names where it should be, because that is a fact about this
        machine rather than about the engine's health.
        """
        import platform as platform_module

        owner = self._c.presence.owner
        engine: dict[str, Any] | None = None
        capabilities: list[Any] = []
        if owner in OWNER_ON_THE_WIRE:
            engine = {
                "name": None,
                "url": engine_url(),
                "backend": None,
                "owner": OWNER_ON_THE_WIRE[owner],
            }
            token = engine_token(self._c)
            if token is not None:
                try:
                    read = peer_module.read_info(
                        engine_url(), token, api_version=API_VERSION
                    )
                except peer_module.PeerCallFailed as exc:
                    self._c.log.write(f"info: the engine did not answer: {exc.code}")
                else:
                    server = read.get("server")
                    host = read.get("host")
                    if isinstance(server, dict):
                        engine["name"] = server.get("name")
                    if isinstance(host, dict):
                        engine["backend"] = host.get("backend")
                    rows = read.get("capabilities")
                    capabilities = rows if isinstance(rows, list) else []
        return {
            "server": {
                "name": self._c.name,
                "version": VERSION,
                "api_version": API_VERSION,
            },
            "host": {
                "platform": sys.platform,
                "arch": platform_module.machine(),
                "backend": peer_module.BACKEND_ORCHESTRATOR,
                "gpu": dict(ORCHESTRATOR_GPU),
            },
            "role": peer_module.ROLE_ORCHESTRATOR,
            # ZERO, and that is the DEFINITION of the role rather than a
            # property of this machine. An orchestrator serves none.
            "job_types": [],
            "engine": engine,
            "capabilities": capabilities,
        }

    def check_restartable(self) -> None:
        """`OrchestratorPort`. 4.1a's rule, refused before anything opens."""
        if self._c.presence.owner is Owner.FOUND:
            raise HostError(
                "engine_not_ours",
                "the engine on this machine was already answering when this "
                "orchestrator started: it did not start it, has no unit it "
                "may name and no child it may kill. Restarting it would mean "
                "guessing, and on the machine this rule was found on the "
                "guess (`systemctl restart user@1000`) would have killed a "
                "five-thousand-step LoRA trainer. Restart it where it was "
                "started (PHASE15-HOST.md 4.1a).",
            )

    def restart_engine(self, emit: Callable[[installer.Event], None]) -> None:
        """PHASE17 4.2 - restart by the owner-appropriate means.

        ONE implementation, two callers: the door's `POST /restart` and the
        tray's own Restart item both arrive here, so a person and a page
        cannot get two different restarts.
        """
        self.check_restartable()
        owner = self._c.presence.owner
        if owner is Owner.WSL_UNIT:
            emit(_step("restart the guest's unit", 1))
            came_back = self._c.watcher.restart_wsl_unit()
        elif owner is Owner.HOST_CHILD:
            emit(_step("respawn the Windows engine", 1))
            came_back = self._respawn_child()
        else:
            # No engine at all. A restart from here is a START, which is what
            # a person pressing Restart on a stopped machine is asking for,
            # and `start()` is the one place that decides which server this
            # machine runs.
            emit(_step("start this machine's engine", 1))
            self.start()
            came_back = self._c.presence.engine is Engine.RUNNING
        emit(_step("wait for /v1/ping", 2))
        if not came_back:
            emit(
                installer.Event(
                    "failed",
                    {
                        "code": "engine_did_not_return",
                        "message": (
                            "the engine was restarted and nothing answered "
                            f"{engine_url('/v1/ping')}. The orchestrator's log "
                            "says which recipe was tried; the engine's own log "
                            "says why it did not come up."
                        ),
                    },
                )
            )
            self._c.presence = Presence(
                self._c.presence.distro,
                Engine.FAILED,
                "a restart did not bring it back",
                owner,
            )
            self._refresh()
            return
        self._c.presence = Presence(
            self._c.presence.distro, Engine.RUNNING, "restarted", owner
        )
        # A restarted engine has forgotten who manages it (PHASE17 2.3), so
        # the claim is re-asserted here rather than waited for.
        self._claimed = False
        self.claim()
        self._refresh()
        emit(installer.Event("done", {"engine": engine_url()}))

    def _respawn_child(self) -> bool:
        """`host-mode-respawn`, as a RESTART: stop this one, then start one."""
        self._c.watcher.stop_child()
        env = dict(self._c.runner.env)
        self._c.watcher.respawn_host_mode(server_argv(env), server_environment(env))
        return self._c.watcher._wait_for_ping(30.0)  # noqa: SLF001 - one object, one loop

    # ------------------------------------------------------------------ menu

    def model(self) -> menu.MenuModel:
        return menu.menu_model(
            self._c.presence.distro,
            self._c.presence.engine,
            self._c.presence.owner,
        )

    def on_click(self, item_id: str) -> None:
        self._c.log.write(f"menu: {item_id}")
        if item_id == menu.OPEN_CONSOLE:
            open_console(self._c.home, self._c.log)
        elif item_id == menu.INSTALL_ENGINE:
            # 4.7: the tray does NOT run the sequence. The switch is a control
            # on the page, the page posts the task, the server relays to this
            # host's door. One door, one sequence, one place a person watches.
            open_console(self._c.home, self._c.log)
        elif item_id == menu.RESTART_ENGINE:
            if self._refuse_acting_on_a_found_engine("restart"):
                return
            # PHASE17 4.2: ONE restart, whether a person clicks it or the page
            # posts `engine-restart`. Before this the tray called `boot()`,
            # which on a RUNNING engine pings, succeeds and changes nothing —
            # a button that did nothing precisely when it was most obviously
            # pressed. The events go to the log, because a tray has no stream.
            self.restart_engine(
                lambda event: self._c.log.write(f"restart {event.event}: {event.data}")
            )
            self._hold()
        elif item_id == menu.STOP_ENGINE:
            if self._refuse_acting_on_a_found_engine("stop"):
                return
            self._stop_engine()
        elif item_id == menu.OPEN_LOG:
            open_log(self._c.log)
        elif item_id == menu.QUIT:
            self.quit()
        self._refresh()

    def _refuse_acting_on_a_found_engine(self, verb: str) -> bool:
        """An engine the host did not start is one it does not act on.

        The menu already disables both verbs (4.2's model), and this is the
        second half of the same rule: a disabled item is a drawing, and the
        thing that must not happen is the ACT. `systemctl --user stop
        crucible` sent into somebody's own distro because a click arrived
        anyway is exactly the class of surprise this host exists to avoid.
        """
        if self._c.presence.owner is not Owner.FOUND:
            return False
        self._c.log.write(
            f"menu: refusing to {verb} an engine this host did not start "
            "(owner=found)"
        )
        return True

    def _stop_engine(self) -> None:
        if self._c.presence.distro is Distro.PRESENT:
            # `systemctl --user stop`, which `Restart=always` respects: the
            # unit is STOPPED, not exited (the ruling in crucible/service.py).
            result = self._c.runner.run(
                [
                    "wsl.exe",
                    "-d",
                    self._c.watcher._distro,  # noqa: SLF001
                    "--exec",
                    "systemctl",
                    "--user",
                    "stop",
                    "crucible",
                ],
                timeout_s=60.0,
            )
            self._c.log.write(f"stop: {'ok' if result.ok else result.said()}")
        else:
            self._c.watcher.stop_child()
        self._c.presence = Presence(
            self._c.presence.distro,
            Engine.STOPPED,
            "stopped from the menu",
            self._c.presence.owner,
        )

    def _refresh(self) -> None:
        if self._icon is None:
            return
        from . import tray

        tray.update(self._icon, self.model(), self.on_click)

    # ----------------------------------------------------------------- watch

    def watch(self) -> None:
        """4.1's watch. One recovery per down-edge, then a state with a name."""
        while not self._stop.wait(self._c.watcher.watch_s):
            before = self._c.presence.engine
            self._c.presence = self._c.watcher.poll(
                self._c.presence.distro, self._c.presence.owner
            )
            # 7b.4c: the hold is what keeps the VM there at all, so it is
            # taken again the tick after it dies rather than at the next login.
            self._c.watcher.rehold()
            if self._c.presence.engine is not before:
                self._c.log.write(
                    f"watch: {before.value} -> {self._c.presence.engine.value} — "
                    f"{self._c.presence.detail}"
                )
                # PHASE17 2.3: a claim is LIVE state and an engine that
                # restarted has forgotten. Every down-to-up edge re-asserts
                # it, which is why nothing has to be remembered on disk — the
                # relation is re-stated within one 15-second tick instead.
                if self._c.presence.engine is Engine.RUNNING:
                    self._claimed = False
                    self.claim()
                self._refresh()

    def quit(self) -> None:
        self._stop.set()
        # BEFORE the hold and before the child: while the engine is still
        # answering. A release sent to a server this process is about to stop
        # would be a release nobody hears.
        self.release_claim()
        # The hold goes first: it is this process's session, and a wsl.exe
        # left running after the tray is gone is a VM nothing owns.
        self._c.watcher.release()
        if self._c.presence.owner is Owner.HOST_CHILD:
            # In host mode the server is this process's child and 4.2's Quit
            # label already said it goes too. An engine the host FOUND is not
            # its child even though the distro probe said `absent`, which is
            # why this asks the owner and not the distro.
            self._c.watcher.stop_child()
        if self._icon is not None:
            self._icon.stop()  # type: ignore[attr-defined]


def run(argv: list[str] | None = None) -> int:
    """`crucible host`. Returns an exit code; never raises past here."""
    env = os.environ
    home = Path(str(crucible_root(env)))
    log = HostLog(Path(str(log_path(env))), Path(str(previous_log_path(env))))
    runner = ProcessRunner(sys.platform, env)
    log.write(f"crucible host {VERSION} starting; CRUCIBLE_HOME={home}")
    acquire(home)

    # The Startup item, if absent. 4.1 makes these verbs its ONE owner, and a
    # host that has been started by hand is a host that should still be there
    # after the next login.
    try:
        outcome = startup.install(runner)
        log.write(f"startup: {outcome.detail}")
    except HostError as exc:
        log.write(f"startup: NOT written — {exc.code}: {exc.message}")

    # PHASE17 2.5, BEFORE the watcher, because consent decides which distro
    # the watcher is about. A malformed setting is named and then not used:
    # the tray still runs, unconsented, which is the behaviour of every
    # machine that never wrote one.
    consented: str | None = None
    try:
        consented = consented_distro(home)
    except HostError as exc:
        log.write(f"consent: NOT used — {exc.code}: {exc.message}")
    if consented is None:
        watcher = PresenceWatcher(runner, log)
    else:
        log.write(
            f'consent: config.toml names "{consented}" as the distro this '
            "orchestrator may manage (PHASE17 2.5); its engine is claimed and "
            "its unit restarted if there is one, and the recipes that would "
            "restart everything uid 1000 owns in it stay refused"
        )
        watcher = PresenceWatcher(runner, log, distro=consented, consented=True)
    context = HostContext(
        runner=runner,
        log=log,
        home=home,
        watcher=watcher,
        presence=Presence(Distro.UNKNOWN, Engine.STARTING, "starting", Owner.NONE),
        name=orchestrator_name(),
    )
    log.write(f"role: orchestrator, as {context.name} (PHASE17-ORCHESTRATOR.md)")
    host = Host(context)
    host.start()
    _write_pairing(context)
    # AFTER the presence and AFTER the pairing file, because the claim needs
    # both: the owner decides whether a claim is made at all (a `found` engine
    # is never claimed), and on a machine whose engine is a guest's, the only
    # place that engine's token exists on the Windows side is the line the
    # pairing step just copied.
    host.claim()

    door = OrchestratorDoor(
        log,
        _sequence(context),
        token=lambda: engine_token(context),
        orchestrator=host,
    )
    try:
        serve(door)
        log.write("door: listening on 127.0.0.1:7101")
    except OSError as exc:
        log.write(f"door: NOT listening ({exc}); the page's engine switch will refuse")

    threading.Thread(target=host.watch, name="crucible-watch", daemon=True).start()

    from . import tray

    icon = tray.make_icon(host.model(), host.on_click)
    host._icon = icon  # noqa: SLF001 - one object, one loop
    icon.run()
    return 0


def _sequence(context: HostContext) -> Callable[[Callable[[installer.Event], None]], None]:
    """Bind the install sequence to THIS host's two servers.

    The catalogs are built per RUN and not once at startup, because the token
    they both use is the one the config has at the moment the move begins —
    and `migrate-config`, a few steps earlier in that same run, is what makes
    the guest's token the Windows one. A port that captured a token at tray
    start would be a port holding a token the guest never had.

    `None` for either side is a fact rather than a fallback: a machine with no
    Windows config has no Windows engine, so there is no catalog to move from
    and `migrate-weights` says exactly that.
    """
    def run_sequence(emit: Callable[[installer.Event], None]) -> None:
        token = read_token(context.home)
        windows: CatalogPort | None = None
        guest: CatalogPort | None = None
        if token is not None:
            windows = HttpCatalog(
                engine_url(), token, where="the Windows engine"
            )
            guest = GuestCatalog(
                context.runner,
                CRUCIBLE_DISTRO,
                token,
                ENGINE_PORT,
                where=f'the "{CRUCIBLE_DISTRO}" engine',
            )
        installer.EngineInstall(
            context.runner,
            emit,
            release=context.release,
            home=context.home,
            install_sh_url=INSTALL_SH_URL.format(release=context.release),
            windows_catalog=windows,
            guest_catalog=guest,
        ).run()

    return run_sequence


def _guest_line(context: HostContext) -> str | None:
    """The line the GUEST wrote, copied — 3.6, and not a second composition.

    *"The Windows file is the host's COPY of the guest's line, because the
    guest's own home is inside the distro where no Windows app looks."* The
    old code composed a line here out of the host's OWN `config.toml`, which
    on a machine whose engine is the guest's is a different token entirely —
    a file that exists and disagrees, which 3.6 says is worse than none
    because it points an app at a door with the wrong key.
    """
    owner = context.presence.owner
    if owner is Owner.WSL_UNIT:
        return context.watcher.read_guest_pairing(
            context.watcher._distro  # noqa: SLF001 - one object, one loop
        )
    if owner is Owner.FOUND:
        return None if context.watcher.found is None else context.watcher.found.line
    return None


def _host_mode_line(context: HostContext) -> str | None:
    """The host-mode child's line, composed from the config the host wrote.

    The one case where composing is right: this server's config IS the host's
    config, so there is one owner of those four facts and not two.
    """
    from ..pairing import pairing_line

    import tomllib

    path = context.home / "config.toml"
    if not path.is_file():
        context.log.write("pairing: no config yet, so no pairing file")
        return None
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        return pairing_line(
            document["server"]["name"],
            engine_url(),
            document["auth"]["token"],
        )
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        context.log.write(f"pairing: {path} could not be read ({exc})")
        return None


def _write_pairing(context: HostContext) -> None:
    """3.6: the Windows-side pairing file, ACL'd to this user.

    WHERE THE LINE COMES FROM IS DECIDED BY WHO OWNS THE ENGINE, and that is
    the correction the first real run forced. Two owners, two sources, and no
    fallback between them: a guest engine's line is READ out of the guest, a
    host-mode child's line is COMPOSED from the host's own config, and when
    there is no engine at all there is no file — which 3.6 calls a fact an app
    knows how to handle.
    """
    from ..pairing import PairingFileError, write_pairing_file

    owner = context.presence.owner
    if owner in (Owner.WSL_UNIT, Owner.FOUND):
        line = _guest_line(context)
        if line is None:
            context.log.write(
                "pairing: the engine on this machine is a guest's and its own "
                "pairing line could not be read, so nothing was written — a "
                "file with the wrong token is worse than no file (3.6)"
            )
            return
    elif owner is Owner.HOST_CHILD:
        line = _host_mode_line(context)
        if line is None:
            return
    else:
        context.log.write("pairing: there is no engine on this machine, so no file")
        return
    try:
        written = write_pairing_file(context.home, line, env=context.runner.env)
    except PairingFileError as exc:
        context.log.write(f"pairing: {exc.code}: {exc.message}")
        return
    context.log.write(f"pairing: {written}")
