from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from .. import API_VERSION, VERSION
from .. import peer as peer_module
from ..controller_client import request
from ..pairing import parse_pairing_line
from ..platform import portholder, startup
from ..platform.errors import HostError
from ..platform.hostconfig import consented_distro
from ..platform.installation import publish_installation
from ..platform.paths import (
    CONSOLE_CMD,
    DOOR_PORT,
    HOST_DOOR_ENV,
    INSTALL_ONE_LINER,
    console_cmd_path,
    door_url,
    engine_url,
    host_pack_dir,
    log_path,
    previous_log_path,
)
from ..platform.runner import ProcessRunner, Runner
from ..wsl import CRUCIBLE_DISTRO
from . import installer, migration, move_policy, operator_stop, outcome, pairing_sync
from . import presence as presence_module
from .catalog import CatalogPort, HttpCatalog
from .context import HostContext
from .controller_door import OrchestratorDoor, serve
from .info import OWNER_ON_THE_WIRE, controller_info
from .log import HostLog
from .presence import Presence, PresenceWatcher
from .state import Distro, Engine, EngineDecision, Owner

LOCK_NAME = "host.pid"

PRESENCE_SETTLE_CEILING_SECONDS = (
    presence_module.WATCH_SECONDS
    + presence_module.RECIPE_TIMEOUT_SECONDS
    + presence_module.BOOT_WAIT_SECONDS
)

HOST_CHILD_START_WAIT_SECONDS = 30.0

INIT_TIMEOUT_SECONDS = 300.0

STOP_TIMEOUT_SECONDS = 60.0


def orchestrator_name() -> str:
    import socket

    return f"crucible-orchestrator@{socket.gethostname().lower()}"


def acquire(home: Path) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    lock = home / LOCK_NAME
    import atexit

    from ..processlock import ProcessLock
    guard = ProcessLock(home / "host.lock")
    if not guard.acquire():
        raise HostError(
            "host_already_running",
            "Another Crucible controller is starting or running. Run `crucible local status` "
            "to see it, or `crucible local shutdown` to stop it first",
        )
    try:
        lock.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        guard.close()
        raise
    atexit.register(guard.close)
    return lock


def server_argv(env: "os._Environ[str] | dict[str, str]") -> list[str]:
    return [str(host_pack_dir(env) / "python.exe"), "-m", "crucible.cli", "serve", "--controller-stdin"]


def server_environment(
    env: "os._Environ[str] | dict[str, str]",
) -> dict[str, str]:
    environment = dict(env)
    environment[HOST_DOOR_ENV] = door_url("")
    return environment


def init_argv(env: "os._Environ[str] | dict[str, str]") -> list[str]:
    return [str(console_cmd_path(env)), "init", "--backend", "llama-windows"]


def _step(name: str, index: int, total: int = 2) -> "installer.Event":
    return installer.Event("step", {"name": name, "index": index, "total": total})


def _read_engine_info(token: str) -> dict[str, Any]:
    return request(engine_url("/v1/info"), token=token)


class Host:
    def __init__(self, context: HostContext) -> None:
        self.context = context
        self.operation = threading.RLock()
        self._stop = threading.Event()
        self._shutdown_complete = threading.Event()
        self._presence_settled = threading.Event()
        self._install_door: OrchestratorDoor | None = None
        self._door_server: Any = None
        self._claimed = False
        self._cleanup = migration.ModelCleanup(
            context,
            lock=self.operation,
            windows_catalog=lambda: self.stopped_windows_catalog(),
            guest_catalog=lambda: self._active_guest_catalog(),
        )

    @property
    def stopped_by_operator(self) -> bool:
        return operator_stop.is_stopped(self.context.home)

    def start(self) -> Presence:
        context = self.context
        distro, detail = context.watcher.probe_distro()
        context.log.write(f"presence: {detail}")
        if distro is Distro.PRESENT:
            context.presence = context.watcher.boot()
        elif context.watcher.ping():
            context.presence = context.watcher.adopt(distro)
        else:
            context.presence = self._start_host_mode(distro)
        context.log.write(
            f"presence: {context.presence.distro.value}/"
            f"{context.presence.engine.value}/{context.presence.owner.value} — "
            f"{context.presence.detail}"
        )
        self._hold()
        return context.presence

    def presence(self) -> dict[str, object]:
        presence = self.context.presence
        return {
            "distro": presence.distro.value,
            "engine": presence.engine.value,
            "owner": presence.owner.value,
            "detail": presence.detail,
        }

    def install_outcome(self) -> dict[str, object] | None:
        recorded = outcome.read_or_quarantine(self.context.home, self.context.log.write)
        return None if recorded is None else recorded.to_dict()

    def local_status(self) -> dict[str, object]:
        stopped = self.stopped_by_operator
        return {"state": Engine.STOPPED.value if stopped else self.context.presence.engine.value,
                "intentional": stopped,
                "detail": self.context.presence.detail}

    def local_start(self) -> dict[str, object]:
        with self.operation:
            self.check_restartable()
            operator_stop.clear(self.context.home)
            if not self.context.watcher.ping():
                self.start()
                pairing_sync.write_pairing(self.context)
                self._claimed = False
                self.claim()
            return self.local_status()

    def local_stop(self) -> dict[str, object]:
        with self.operation:
            self.check_restartable()
            self._stop_engine()
            return self.local_status()

    def stop_windows_for_move(self) -> None:
        if self.context.presence.owner is Owner.HOST_CHILD:
            self.release_claim()
            self.context.watcher.stop_child()
        elif self.context.presence.owner not in (Owner.NONE, Owner.WSL_UNIT):
            raise HostError("engine_not_ours", "Cannot replace an unmanaged engine")

    def _booted_guest(self) -> tuple[PresenceWatcher, Presence, str]:
        self.context.watcher.release()
        watcher = PresenceWatcher(self.context.runner, self.context.log, distro=CRUCIBLE_DISTRO)
        presence = watcher.boot()
        if presence.engine is not Engine.RUNNING:
            raise HostError("engine_move_failed", presence.detail)
        line = watcher.read_guest_pairing(CRUCIBLE_DISTRO)
        if line is None:
            raise HostError("engine_move_failed", "The guest did not publish its pairing")
        pair = parse_pairing_line(line)
        ping = request(engine_url("/v1/ping"))
        if ping.get("crucible") is not True or ping.get("name") != pair.name:
            raise HostError("engine_move_failed", "Windows is not reaching the installed guest engine")
        info = request(engine_url("/v1/info"), token=pair.token)
        server, machine = info.get("server"), info.get("host")
        if (not isinstance(server, dict) or server.get("name") != pair.name
                or server.get("api_version") != API_VERSION or not isinstance(machine, dict)
                or machine.get("backend") != installer.GUEST_BACKEND):
            raise HostError("engine_move_failed", "Windows is not reaching the authenticated WSL engine with the expected API")
        return watcher, presence, line

    def _adopt_guest(self, watcher: PresenceWatcher, presence: Presence, line: str) -> None:
        context = self.context
        was = (context.watcher, context.presence)
        pairing = context.home / "pairing"
        was_pairing = pairing.read_text(encoding="utf-8")
        was_stopped = self.stopped_by_operator
        context.watcher, context.presence = watcher, presence
        try:
            pairing_sync.write_pairing(context)
            if pairing.read_text(encoding="utf-8").strip() != line.strip():
                raise HostError("engine_move_failed", "Windows pairing was not updated")
            operator_stop.clear(context.home)
            self._hold()
            self._claimed = False
            if not self.claim():
                raise HostError("engine_move_failed", "The guest could not be claimed by its Windows controller")
        except Exception:
            context.watcher, context.presence = was
            pairing.write_text(was_pairing, encoding="utf-8")
            operator_stop.restore(context.home, was_stopped)
            self._claimed = False
            watcher.release()
            raise

    def finish_wsl_move(self) -> None:
        watcher, presence, line = self._booted_guest()
        self._adopt_guest(watcher, presence, line)
        from ..sharing import SharingError, reconcile
        try:
            reconcile(self.context.home, self.context.runner)
        except SharingError as exc:
            raise HostError(
                "sharing_reconcile_failed",
                f"The WSL engine is running, but its saved network sharing could not be restored: {exc}. "
                "Run `crucible sharing enable` to set it up again",
            ) from exc

    def carry_guest_to_this_release(
        self, *, settle_ceiling_s: float = PRESENCE_SETTLE_CEILING_SECONDS
    ) -> None:
        log = self.context.log.write
        if not self._presence_settled.wait(settle_ceiling_s):
            log(
                f"guest release: presence never settled within "
                f"{settle_ceiling_s:.0f} s, so whether this machine's engine is "
                "a guest of ours is unknown and nothing was carried"
            )
            return
        if self.context.presence.owner is not Owner.WSL_UNIT:
            log(f"guest release: no guest to carry (owner={self.context.presence.owner.value})")
            self.decide_engine()
            return
        self._carry_guest()

    def _carry_guest(self) -> None:
        context = self.context
        walk = context.install_walk(context.logged_as("guest release"), distro=context.watcher.distro)
        try:
            carried = walk.upgrade_guest()
        except HostError as exc:
            context.log.write(f"guest release: {exc.code}: {exc.message}")
            return
        except Exception as exc:
            context.log.write(f"guest release: could not be read or carried: {exc}")
            return
        if carried is None:
            context.log.write(f"guest release: already {context.release}")
        else:
            context.log.write(f"guest release: carried the guest to {carried}")

    def decide_engine(self) -> EngineDecision:
        return move_policy.decide_engine(self.context, self._run_move)

    def _run_move(self, why: str) -> EngineDecision:
        return move_policy.run_move(self.context, self._install_door, why)

    def attach_install_door(self, door: OrchestratorDoor, server: Any = None) -> None:
        self._install_door = door
        self._door_server = server

    def stopped_windows_catalog(self) -> CatalogPort:
        self.verify_active_guest()
        return migration.stopped_windows_catalog(self.context.home)

    def verify_active_guest(self) -> None:
        migration.verify_active_guest(self.context.presence, pairing_sync.engine_token(self.context), _read_engine_info)

    def _active_guest_catalog(self) -> CatalogPort:
        token = pairing_sync.engine_token(self.context)
        if token is None:
            raise HostError(migration.NOT_READY, "The active guest has no credential")
        return HttpCatalog(engine_url(), token, where="the active WSL engine")

    def resume_model_cleanup(self, *, raise_errors: bool = False) -> None:
        self._cleanup.resume(raise_errors=raise_errors)

    def _hold(self) -> None:
        context = self.context
        if context.presence.owner is Owner.WSL_UNIT:
            name = context.watcher.distro
        elif context.presence.owner is Owner.FOUND and context.watcher.found is not None:
            name = context.watcher.found.distro
        else:
            return
        context.watcher.hold(name)

    def _host_mode_unready(self, distro: Distro, env: dict[str, str]) -> Presence | None:
        if not Path(str(console_cmd_path(env))).is_file():
            return Presence(
                distro,
                Engine.FAILED,
                f"there is no {CONSOLE_CMD} in {host_pack_dir(env)}, so this controller "
                "has no server to start. Reinstall from PowerShell with: "
                f"{INSTALL_ONE_LINER}",
                Owner.NONE,
            )
        if (self.context.home / "config.toml").is_file():
            return None
        first = self.context.runner.run(init_argv(env), timeout_s=INIT_TIMEOUT_SECONDS)
        self.context.log.write(f"init: {'ok' if first.ok else first.output_tail()}")
        if first.ok:
            return None
        return Presence(
            distro, Engine.FAILED,
            f"crucible init: {first.output_tail()}. Its log is {log_path(env)}; reinstall from "
            f"PowerShell with: {INSTALL_ONE_LINER}",
            Owner.NONE,
        )

    def _start_host_mode(self, distro: Distro) -> Presence:
        env = dict(self.context.runner.env)
        unready = self._host_mode_unready(distro, env)
        if unready is not None:
            return unready
        self.context.watcher.respawn_host_mode(server_argv(env), server_environment(env))
        if self.context.watcher.wait_for_ping(HOST_CHILD_START_WAIT_SECONDS):
            return Presence(distro, Engine.RUNNING, "the Windows engine answered /v1/ping", Owner.HOST_CHILD)
        return Presence(
            distro,
            Engine.FAILED,
            f"the Windows engine did not answer within {HOST_CHILD_START_WAIT_SECONDS:.0f} s; "
            f"why is in {log_path(env)} and in {self.context.home / 'logs'}",
            Owner.NONE,
        )

    def claim(self) -> bool:
        log = self.context.log.write
        owner = self.context.presence.owner
        if owner is Owner.FOUND:
            log(
                "claim: the engine on this machine was already answering when "
                "this controller started, so it is watched and not claimed "
                "(owner=found; docs/internals/host-and-platform.md, \"Ownership\")"
            )
            return False
        if owner not in (Owner.WSL_UNIT, Owner.HOST_CHILD):
            return False
        token = pairing_sync.engine_token(self.context)
        if token is None:
            log(
                "claim: this machine's engine token could not be read, so no "
                "claim was made - an engine is not less of an engine for "
                "being unclaimed"
            )
            return False
        try:
            answer = peer_module.claim_engine(engine_url(), token, self._controller_ref(), api_version=API_VERSION)
        except peer_module.PeerCallFailed as exc:
            log(f"claim: {exc.code}: {exc.message}")
            return False
        log(
            f"claim: {engine_url()} is managed by {self.context.name} "
            f"(owner={OWNER_ON_THE_WIRE[owner]}, claimed {answer.get('claimed')})"
        )
        self._claimed = True
        return True

    def release_claim(self) -> None:
        if not self._claimed:
            return
        token = pairing_sync.engine_token(self.context)
        if token is None:
            return
        log = self.context.log.write
        try:
            peer_module.release_engine(engine_url(), token, self._controller_ref(), api_version=API_VERSION)
            log(f"claim: released {engine_url()}")
        except peer_module.PeerCallFailed as exc:
            if exc.code == "peer_unreachable":
                log("claim: the engine is already stopped; nothing to release")
            else:
                log(f"claim: release did not land: {exc.code}: {exc.message}")
        self._claimed = False

    def _controller_ref(self) -> peer_module.Orchestrator:
        return peer_module.Orchestrator(name=self.context.name, url=door_url(""), version=VERSION)

    @property
    def name(self) -> str:
        return self.context.name

    def info(self) -> dict[str, Any]:
        return controller_info(
            self.context.name,
            self.context.presence.owner,
            engine_url=engine_url(),
            token=lambda: pairing_sync.engine_token(self.context),
            log=self.context.log.write,
        )

    def check_restartable(self) -> None:
        if self.context.presence.owner is Owner.FOUND:
            raise HostError(
                "engine_not_ours",
                "the engine on this machine was already answering when this "
                "orchestrator started: it did not start it, has no unit it "
                "may name and no child it may kill. Restarting it would mean "
                "guessing, and on the machine this rule was found on the "
                "guess (`systemctl restart user@1000`) would have killed a "
                "five-thousand-step LoRA trainer. Restart it where it was "
                "started (docs/internals/host-and-platform.md, \"Ownership\").",
            )

    def _restart_by_owner(self, owner: Owner, emit: Callable[[installer.Event], None]) -> bool:
        if owner is Owner.WSL_UNIT:
            emit(_step("restart the guest's unit", 1))
            return self.context.watcher.restart_wsl_unit()
        if owner is Owner.HOST_CHILD:
            emit(_step("respawn the Windows engine", 1))
            return self._respawn_child()
        emit(_step("start this machine's engine", 1))
        self.start()
        return self.context.presence.engine is Engine.RUNNING

    def _did_not_return(self, owner: Owner) -> installer.Event:
        self.context.presence = Presence(
            self.context.presence.distro, Engine.FAILED, "a restart did not bring it back", owner
        )
        return installer.Event(
            "failed",
            {
                "code": "engine_did_not_return",
                "message": (
                    "the engine was restarted and nothing answered "
                    f"{engine_url('/v1/ping')}. The controller's log "
                    f"({self.context.log.path}) says which recipe was tried; the "
                    f"engine's own log in {self.context.home / 'logs'} says why it "
                    "did not come up."
                ),
            },
        )

    def restart_engine(self, emit: Callable[[installer.Event], None]) -> None:
        self.check_restartable()
        operator_stop.clear(self.context.home)
        owner = self.context.presence.owner
        came_back = self._restart_by_owner(owner, emit)
        emit(_step("wait for /v1/ping", 2))
        if not came_back:
            emit(self._did_not_return(owner))
            return
        self.context.presence = Presence(self.context.presence.distro, Engine.RUNNING, "restarted", owner)
        self._claimed = False
        self.claim()
        emit(installer.Event("done", {"engine": engine_url()}))

    def _respawn_child(self) -> bool:
        watcher = self.context.watcher
        watcher.stop_child()
        env = dict(self.context.runner.env)
        watcher.respawn_host_mode(server_argv(env), server_environment(env))
        return watcher.wait_for_ping(HOST_CHILD_START_WAIT_SECONDS)

    def _stop_guest_unit(self) -> None:
        context = self.context
        probe = context.watcher.probe_unit()
        if not probe.readable:
            context.log.write(f"stop: NOT RUN — {probe.detail}")
            raise HostError("engine_stop_failed", probe.detail)
        result = context.runner.run(
            presence_module.system_systemctl_argv(context.watcher.distro, "stop"),
            timeout_s=STOP_TIMEOUT_SECONDS,
        )
        context.log.write(f"stop: {'ok' if result.ok else result.output_tail()}")
        if not result.ok:
            raise HostError("engine_stop_failed", result.output_tail())

    def _stop_engine(self) -> None:
        if self.context.presence.distro is Distro.PRESENT:
            self._stop_guest_unit()
        else:
            self.context.watcher.stop_child()
        self._mark_stopped()

    def _mark_stopped(self) -> None:
        operator_stop.record(self.context.home)
        self.context.presence = Presence(
            self.context.presence.distro,
            Engine.STOPPED,
            "stopped from the menu",
            self.context.presence.owner,
        )

    def _tick(self) -> None:
        context = self.context
        before = context.presence.engine
        context.presence = context.watcher.poll(context.presence.distro, context.presence.owner)
        if context.watcher.held_distro is None:
            self._hold()
        context.watcher.rehold()
        if self._cleanup.due():
            self._cleanup.start_in_background()
        if context.presence.engine is before:
            return
        context.log.write(f"watch: {before.value} -> {context.presence.engine.value} — {context.presence.detail}")
        if context.presence.engine is Engine.RUNNING:
            self._claimed = False
            self.claim()

    def watch(self) -> None:
        while not self._stop.wait(self.context.watcher.watch_s):
            with self.operation:
                if not self.stopped_by_operator:
                    self._tick()
                self._presence_settled.set()

    def quit(self, *, handover: bool = False) -> None:
        context = self.context
        owner = context.presence.owner
        claim = "released" if self._claimed else "not held, so nothing to release"
        engine = "stopped with it, being this process's child" if owner is Owner.HOST_CHILD else "left running"
        context.log.write(
            f"quit: stopping this controller (owner={owner.value}); the "
            f"claim is {claim} and the engine is {engine}"
        )
        self._stop.set()
        self.release_claim()
        if handover:
            context.watcher.hand_over()
        else:
            context.watcher.release()
        if owner is Owner.HOST_CHILD:
            context.watcher.stop_child()
        self._shutdown_complete.set()
        context.log.write("quit: shutdown complete; controller loop signalled")

    def wait_until_quit(self) -> None:
        self._shutdown_complete.wait()
        if self._door_server is not None:
            self._door_server.shutdown()
            self._door_server.server_close()


def _install_startup_item(runner: Runner, log: HostLog) -> None:
    try:
        written = startup.install(runner)
        log.write(f"startup: {written.detail}")
    except HostError as exc:
        log.write(f"startup: NOT written — {exc.code}: {exc.message}")


def _watcher_for(home: Path, runner: Runner, log: HostLog) -> PresenceWatcher:
    consented: str | None = None
    try:
        consented = consented_distro(home)
    except HostError as exc:
        log.write(f"consent: NOT used — {exc.code}: {exc.message}")
    if consented is None:
        return PresenceWatcher(runner, log)
    log.write(
        f'consent: config.toml names "{consented}" as the distro this '
        "controller may manage (docs/internals/host-and-platform.md, \"Ownership\"); its engine is claimed and "
        "its unit restarted if there is one, and the recipes that would "
        "restart everything uid 1000 owns in it stay refused"
    )
    return PresenceWatcher(runner, log, distro=consented, consented=True)


def _open_door(context: HostContext, host: Host) -> None:
    log = context.log
    door = OrchestratorDoor(
        log,
        move_policy.move_sequence(context, host),
        token=lambda: pairing_sync.engine_token(context),
        token_detail=lambda: pairing_sync.engine_token_detail(context),
        orchestrator=host,
    )
    host.attach_install_door(door)
    try:
        server = serve(door)
    except OSError as exc:
        held = portholder.held_sentence(DOOR_PORT)
        log.write(f"door: NOT listening ({exc}); {held}; shutting down this controller's owned child")
        host.quit()
        raise HostError(
            "host_door_unavailable",
            f"The local controller port is unavailable ({exc}): {held}. Its log is {log.path}.",
        ) from exc
    host.attach_install_door(door, server)
    log.write(f"door: listening on {door_url()}")


def _background(target: Callable[[], None], name: str) -> None:
    threading.Thread(target=target, name=name, daemon=True).start()


def run(argv: list[str] | None = None, *, headless: bool = True) -> int:
    env = os.environ
    from ..config import crucible_home
    home = crucible_home()
    log = HostLog(Path(str(log_path(env))), Path(str(previous_log_path(env))))
    runner = ProcessRunner(sys.platform, env, cwd=str(home))
    log.write(f"crucible controller {VERSION} starting; CRUCIBLE_HOME={home}")
    acquire(home)
    _install_startup_item(runner, log)
    context = HostContext(
        runner=runner,
        log=log,
        home=home,
        watcher=_watcher_for(home, runner, log),
        presence=Presence(Distro.UNKNOWN, Engine.STARTING, "starting", Owner.NONE),
        name=orchestrator_name(),
    )
    log.write(f"role: orchestrator, as {context.name}")
    host = Host(context)
    if host.stopped_by_operator:
        context.presence = Presence(Distro.UNKNOWN, Engine.STOPPED, operator_stop.REASON, Owner.NONE)
    else:
        host.start()
    pairing_sync.write_pairing(context)
    host.claim()
    _open_door(context, host)
    _background(host.watch, "crucible-watch")
    publish_installation(home)
    _background(host.carry_guest_to_this_release, "crucible-guest-release")
    host.wait_until_quit()
    return 0
