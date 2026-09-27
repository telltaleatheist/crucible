from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .. import API_VERSION, VERSION
from .. import peer as peer_module
from ..pairing import parse_pairing_line
from . import door as door_module
from . import installer, menu, outcome, startup, wslstate
from .catalog import CatalogPort, GuestCatalog, HttpCatalog, StoppedWindowsCatalog
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
from . import presence as presence_module
from .presence import Presence, PresenceWatcher
from .runner import ProcessRunner, Runner
from ..tasks import HOST_DOOR_ENV
from .wsl_states import CRUCIBLE_DISTRO

LOCK_NAME = "host.pid"

DEFAULT_RELEASE = VERSION

PRESENCE_SETTLE_CEILING_SECONDS = (
    presence_module.WATCH_SECONDS
    + presence_module.RECIPE_TIMEOUT_SECONDS
    + presence_module.BOOT_WAIT_SECONDS
)

INSTALL_SH_URL = (
    "https://github.com/telltaleatheist/crucible/releases/download/"
    "v{release}/install.sh"
)


@dataclass
class HostContext:
    runner: Runner
    log: HostLog
    home: Path
    watcher: PresenceWatcher
    presence: Presence
    release: str = DEFAULT_RELEASE
    name: str = ""


def orchestrator_name() -> str:
    import socket

    return f"crucible-orchestrator@{socket.gethostname().lower()}"


ORCHESTRATOR_GPU = {"vendor": "none", "name": "", "vram_bytes": 0}
"""What an accelerator block says about a process that plays nothing.

Not omitted and not null: a client reading `host.gpu.vram_bytes` must get a
number, and the true number is zero. PHASE17 1: "what accelerator does the
thing that plays nothing have" has exactly one honest answer.
"""

OWNER_ON_THE_WIRE = {
    Owner.WSL_UNIT: peer_module.OWNER_WSL_UNIT,
    Owner.HOST_CHILD: peer_module.OWNER_CHILD,
    Owner.FOUND: peer_module.OWNER_FOUND,
}


def engine_token(context: "HostContext") -> str | None:
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
    if owner is Owner.NONE and (context.home / "engine.stopped").exists():
        try:
            return parse_pairing_line((context.home / "pairing").read_text(encoding="utf-8").strip()).token
        except (OSError, ValueError) as exc:
            context.log.write(f"stopped engine pairing is invalid: {exc}")
    return None


def engine_token_detail(context: "HostContext") -> str:
    owner = context.presence.owner
    if owner in (Owner.WSL_UNIT, Owner.FOUND):
        return (
            f"the engine here is owner={owner.value} and its bearer comes from the "
            "guest's pairing line, which could not be read or would not parse. "
            "The host log says which."
        )
    if owner is Owner.HOST_CHILD:
        return (
            "this host runs its own engine and there is no token in its config "
            f"yet ({context.home}). It gets one the first time the Windows server "
            "is initialised, which is seconds after the host first starts."
        )
    return (
        "this orchestrator owns no engine (owner=none), so there is no bearer "
        "for it to check against. Its config is not the problem. An engine that "
        "is answering is adopted on the next watch tick; one that is not needs "
        "Restart engine, or `crucible local start`."
    )


def read_token(home: Path) -> str | None:
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


CONSENT_TABLE = "orchestrator"
CONSENT_KEY = "distro"

WSL_KEY = "wsl"
WSL_NEVER = "never"


def declined_wsl(home: Path) -> bool:
    import tomllib

    path = Path(home) / "config.toml"
    if not path.is_file():
        return False
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise HostError(
            "orchestrator_wsl_invalid",
            f"{path} could not be read as TOML ({exc}), so whether this machine "
            "declined the Linux engine cannot be known. Nothing is moved until "
            "the file parses.",
        ) from exc
    table = document.get(CONSENT_TABLE)
    if table is None or not isinstance(table, dict) or WSL_KEY not in table:
        return False
    value = table[WSL_KEY]
    if value != WSL_NEVER:
        raise HostError(
            "orchestrator_wsl_invalid",
            f"{CONSENT_TABLE}.{WSL_KEY} in {path} is {value!r}. The one value "
            f'this key takes is "{WSL_NEVER}", which keeps this machine on its '
            "native Windows engine; remove the key to let Crucible install the "
            "Linux one.",
        )
    return True


def consented_distro(home: Path) -> str | None:
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
    home.mkdir(parents=True, exist_ok=True)
    lock = home / LOCK_NAME
    from ..processlock import ProcessLock
    import atexit
    guard = ProcessLock(home / "host.lock")
    if not guard.acquire():
        raise HostError("host_already_running", "Another Crucible controller is starting or running")
    try:
        lock.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        guard.close()
        raise
    atexit.register(guard.close)
    return lock


HOST_CHILD_START_WAIT_SECONDS = 30.0
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_ERROR_ACCESS_DENIED = 5


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
        kernel.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except (OSError, ProcessLookupError):
        return False
    return True


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


def open_console(home: Path, log: HostLog) -> None:
    from ..pairing import read_pairing_file

    line = read_pairing_file(home)
    if line is None:
        log.write(
            f"open console: there is no pairing file in {home}, so there is no "
            "server on this machine to open yet"
        )
        return
    from urllib.parse import urlsplit

    parts = urlsplit(line)
    url = f"http://{parts.netloc.rsplit('@', 1)[-1]}/#token={parts.fragment}"
    log.write(f"open console: {parts.netloc.rsplit('@', 1)[-1]}")
    webbrowser.open(url)


def open_log(log: HostLog) -> None:
    if sys.platform == "win32":
        os.startfile(str(log.path))
        return
    log.write(f"the log is {log.path}")


def _step(name: str, index: int, total: int = 2) -> "installer.Event":
    return installer.Event("step", {"name": name, "index": index, "total": total})


class Host:
    def __init__(self, context: HostContext) -> None:
        self._c = context
        self._icon: object | None = None
        self._stop = threading.Event()
        self._shutdown_complete = threading.Event()
        self._door_server: object | None = None
        self._operation = threading.RLock()
        self._paused = (context.home / "engine.stopped").exists()
        self._cleanup_running = False
        self._cleanup_retry_at = 0.0
        self._presence_settled = threading.Event()
        self._install_door: "door_module.OrchestratorDoor | None" = None
        self._claimed = False


    def start(self) -> Presence:
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

    def presence(self) -> dict[str, object]:
        return {
            "distro": self._c.presence.distro.value,
            "engine": self._c.presence.engine.value,
            "owner": self._c.presence.owner.value,
            "detail": self._c.presence.detail,
        }

    def install_outcome(self) -> dict[str, object] | None:
        recorded = outcome.read(self._c.home)
        return None if recorded is None else recorded.to_dict()

    def local_status(self) -> dict[str, object]:
        return {"state": "stopped" if self._paused else self._c.presence.engine.value,
                "intentional": self._paused,
                "detail": self._c.presence.detail}

    def local_start(self) -> dict[str, object]:
        with self._operation:
            self.check_restartable()
            self._c.home.joinpath("engine.stopped").unlink(missing_ok=True)
            self._paused = False
            if not self._c.watcher.ping():
                self.start()
                _write_pairing(self._c)
                self._claimed = False
                self.claim()
            self._refresh()
            return self.local_status()

    def local_stop(self) -> dict[str, object]:
        with self._operation:
            self.check_restartable()
            self._stop_engine()
            return self.local_status()

    def stop_windows_for_move(self) -> None:
        if self._c.presence.owner is Owner.HOST_CHILD:
            self.release_claim()
            self._c.watcher.stop_child()
        elif self._c.presence.owner not in (Owner.NONE, Owner.WSL_UNIT):
            raise HostError("engine_not_ours", "Cannot replace an unmanaged engine")

    def finish_wsl_move(self) -> None:
        from ..local import request
        self._c.watcher.release()
        watcher = PresenceWatcher(self._c.runner, self._c.log, distro=CRUCIBLE_DISTRO)
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
                or server.get("api_version") != 1 or not isinstance(machine, dict)
                or machine.get("backend") != "cuda-linux"):
            raise HostError("engine_move_failed", "Windows is not reaching the authenticated WSL engine with the expected API")
        was = (self._c.watcher, self._c.presence, self._paused)
        was_pairing = (self._c.home / "pairing").read_text(encoding="utf-8")
        was_stopped = self._c.home.joinpath("engine.stopped").exists()
        self._c.watcher = watcher
        self._c.presence = presence
        try:
            _write_pairing(self._c)
            if (self._c.home / "pairing").read_text(encoding="utf-8").strip() != line.strip():
                raise HostError("engine_move_failed", "Windows pairing was not updated")
            self._paused = False
            self._c.home.joinpath("engine.stopped").unlink(missing_ok=True)
            self._hold()
            self._claimed = False
            if not self.claim():
                raise HostError("engine_move_failed", "The guest could not be claimed by its Windows controller")
        except Exception:
            self._c.watcher, self._c.presence, self._paused = was
            self._c.home.joinpath("pairing").write_text(was_pairing, encoding="utf-8")
            if was_stopped:
                self._c.home.joinpath("engine.stopped").touch()
            self._claimed = False
            watcher.release()
            raise
        self._refresh()
        from ..sharing import SharingError, reconcile
        try:
            reconcile(self._c.home, self._c.runner)
        except SharingError as exc:
            raise HostError(
                "sharing_reconcile_failed",
                f"The WSL engine is running, but its saved network sharing could not be restored: {exc}",
            ) from exc

    def carry_guest_to_this_release(
        self, *, settle_ceiling_s: float = PRESENCE_SETTLE_CEILING_SECONDS
    ) -> None:
        if not self._presence_settled.wait(settle_ceiling_s):
            self._c.log.write(
                f"guest release: presence never settled within "
                f"{settle_ceiling_s:.0f} s, so whether this machine's engine is "
                "a guest of ours is unknown and nothing was carried"
            )
            return
        if self._c.presence.owner is not Owner.WSL_UNIT:
            self._c.log.write(
                f"guest release: no guest to carry "
                f"(owner={self._c.presence.owner.value})"
            )
            self.decide_engine()
            return
        walk = installer.EngineInstall(
            self._c.runner,
            lambda event: self._c.log.write(f"guest release: {event.event}: {event.data}"),
            release=self._c.release,
            home=self._c.home,
            install_sh_url=INSTALL_SH_URL.format(release=self._c.release),
            distro=self._c.watcher.distro,
        )
        try:
            carried = walk.upgrade_guest()
        except HostError as exc:
            self._c.log.write(f"guest release: {exc.code}: {exc.message}")
            return
        except Exception as exc:
            self._c.log.write(f"guest release: could not be read or carried: {exc}")
            return
        if carried is None:
            self._c.log.write(f"guest release: already {self._c.release}")
        else:
            self._c.log.write(f"guest release: carried the guest to {carried}")
            self._refresh()


    def decide_engine(self) -> str:
        owner = self._c.presence.owner
        if owner is Owner.FOUND:
            self._c.log.write(
                "engine: this machine's engine is one this orchestrator did not "
                "start (owner=found), so nothing is moved (PHASE17 4.1a)"
            )
            return "found"
        try:
            declined = declined_wsl(self._c.home)
        except HostError as exc:
            self._c.log.write(f"engine: {exc.code}: {exc.message}")
            return "unreadable"
        try:
            previous = outcome.read(self._c.home)
        except HostError as exc:
            self._c.log.write(f"engine: {exc.code}: {exc.message}")
            return "unreadable"
        if declined:
            if previous is None or previous.state != outcome.DECLINED:
                outcome.write(
                    self._c.home,
                    state=outcome.DECLINED,
                    release=self._c.release,
                    attempts=0 if previous is None else previous.attempts,
                )
            self._c.log.write(
                'engine: this machine declined the Linux engine '
                f'([{CONSENT_TABLE}] {WSL_KEY} = "{WSL_NEVER}"); it stays native'
            )
            return outcome.DECLINED
        if (
            previous is not None
            and previous.state == outcome.CANNOT
            and previous.code in outcome.TRANSIENT_CANNOT_CODES
        ):
            try:
                live = wslstate.probe_live(self._c.runner)
            except Exception as exc:
                self._c.log.write(f"engine: the WSL re-check crashed: {type(exc).__name__}: {exc}")
                return outcome.CANNOT
            self._c.log.write(
                f"engine: {previous.code} was recorded {previous.at}; checked again "
                f"at this start: {live.line()}"
            )
            if live.live:
                return self._move("resumed: WSL is live now")
            return outcome.CANNOT
        if (
            previous is not None
            and previous.state == outcome.CANNOT
            and previous.code in outcome.FIRMWARE_CANNOT_CODES
        ):
            try:
                live = wslstate.probe_live(self._c.runner)
            except Exception as exc:
                self._c.log.write(f"engine: the virtualization re-check crashed: {type(exc).__name__}: {exc}")
                return outcome.CANNOT
            self._c.log.write(
                f"engine: {previous.code} was recorded {previous.at}; checked again "
                f"at this start: {live.line()}"
            )
            if live.answer.kind != "no_hypervisor":
                return self._move("resumed: virtualization is on now")
            return outcome.CANNOT
        if previous is not None and previous.state == outcome.CANNOT:
            self._c.log.write(
                f"engine: this machine cannot run the Linux engine "
                f"({previous.code}), recorded {previous.at}. Nothing is retried "
                "on its own; the apps offer Try again (2.5)"
            )
            return outcome.CANNOT
        if (
            previous is not None
            and previous.state == outcome.FAILED
            and previous.attempts >= outcome.FAILED_ATTEMPT_CEILING
        ):
            self._c.log.write(
                f"engine: the move has failed {previous.attempts} times in a row "
                f"({previous.code}); it stays failed until somebody presses Try "
                "again (2.2)"
            )
            return outcome.FAILED
        if previous is not None and previous.state == outcome.DONE:
            self._c.log.write(
                f"engine: the last move finished at {previous.at} and this "
                "machine's engine is not the guest's now; walking the sequence "
                "again"
            )
        return self._move("resumed" if previous is not None and previous.state == outcome.REBOOT_PENDING else "started")

    def _move(self, why: str) -> str:
        door = self._install_door
        if door is None:
            self._c.log.write(
                "engine: NOT moved — this orchestrator has no door yet, and the "
                "move runs under the door's claim so that a POST /install can "
                "be refused and attached rather than queued"
            )
            return "no_door"
        if not door.claim():
            self._c.log.write(
                "engine: a move is already running on this machine; this one is "
                "not a second walk over the same distro"
            )
            return "already_running"
        self._c.log.write(f"engine: the move is {why}")
        try:
            door.run_recorded()
        except HostError as exc:
            self._c.log.write(f"engine: {exc.code}: {exc.message}")
            return outcome.classify(exc.code)
        except Exception as exc:
            self._c.log.write(f"engine: the move crashed: {type(exc).__name__}: {exc}")
            return outcome.FAILED
        finally:
            door.release()
            self._refresh()
        return outcome.DONE

    def stopped_windows_catalog(self) -> CatalogPort:
        from ..backend import detect_backend
        from ..config import load_config
        self._verify_active_guest()
        return StoppedWindowsCatalog(load_config(self._c.home), detect_backend(), installer.cleanup_subjects(self._c.home))

    def _verify_active_guest(self) -> None:
        from ..local import request
        if self._c.presence.owner is not Owner.WSL_UNIT or self._c.presence.engine is not Engine.RUNNING:
            raise HostError("migration_cleanup_not_ready", "The WSL engine has not taken ownership; Windows models are kept")
        token = engine_token(self._c)
        if token is None:
            raise HostError("migration_cleanup_not_ready", "The WSL engine has no pairing; Windows models are kept")
        info = request(engine_url("/v1/info"), token=token)
        if info.get("host", {}).get("backend") != "cuda-linux":
            raise HostError("migration_cleanup_not_ready", "The authenticated endpoint is not the WSL engine; Windows models are kept")

    def _resume_model_cleanup(self, *, raise_errors: bool = False) -> None:
        try:
            with self._operation:
                record = self._c.home / installer.CLEANUP_RECORD
                if not record.exists():
                    return
                windows = self.stopped_windows_catalog()
                token = engine_token(self._c)
                if token is None:
                    raise HostError("migration_cleanup_not_ready", "The active guest has no credential")
                guest = HttpCatalog(engine_url(), token, where="the active WSL engine")
                walk = installer.EngineInstall(
                    self._c.runner, lambda event: self._c.log.write(f"model cleanup: {event.event}: {event.data}"),
                    release=self._c.release, home=self._c.home,
                    install_sh_url=INSTALL_SH_URL.format(release=self._c.release),
                    windows_catalog=windows, guest_catalog=guest,
                )
                walk._migrate_weights(allow_pull=False)
                record.unlink()
                self._c.log.write("model cleanup: completed; native runtime kept, migrated Windows model files removed")
        except Exception as exc:
            self._c.log.write(f"model cleanup pending: {exc}")
            if raise_errors:
                raise
        finally:
            self._cleanup_retry_at = time.monotonic() + 300
            self._cleanup_running = False

    def _hold(self) -> None:
        if self._c.presence.owner is Owner.WSL_UNIT:
            name = self._c.watcher.distro
        elif (
            self._c.presence.owner is Owner.FOUND
            and self._c.watcher.found is not None
        ):
            name = self._c.watcher.found.distro
        else:
            return
        self._c.watcher.hold(name)

    def _start_host_mode(self, distro: Distro) -> Presence:
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
        if self._c.watcher._wait_for_ping(HOST_CHILD_START_WAIT_SECONDS):
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


    def claim(self) -> bool:
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
            if exc.code == "peer_unreachable":
                self._c.log.write("claim: the engine is already stopped; nothing to release")
            else:
                self._c.log.write(f"claim: release did not land: {exc.code}: {exc.message}")
        self._claimed = False

    def _orchestrator_ref(self) -> peer_module.Orchestrator:
        return peer_module.Orchestrator(
            name=self._c.name, url=door_url(""), version=VERSION
        )

    @property
    def name(self) -> str:
        return self._c.name

    def info(self) -> dict[str, Any]:
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
            "local_lifecycle_version": 1,
            "job_types": [],
            "engine": engine,
            "capabilities": capabilities,
        }

    def check_restartable(self) -> None:
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
        self.check_restartable()
        self._c.home.joinpath("engine.stopped").unlink(missing_ok=True)
        self._paused = False
        owner = self._c.presence.owner
        if owner is Owner.WSL_UNIT:
            emit(_step("restart the guest's unit", 1))
            came_back = self._c.watcher.restart_wsl_unit()
        elif owner is Owner.HOST_CHILD:
            emit(_step("respawn the Windows engine", 1))
            came_back = self._respawn_child()
        else:
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
        self._claimed = False
        self.claim()
        self._refresh()
        emit(installer.Event("done", {"engine": engine_url()}))

    def _respawn_child(self) -> bool:
        self._c.watcher.stop_child()
        env = dict(self._c.runner.env)
        self._c.watcher.respawn_host_mode(server_argv(env), server_environment(env))
        return self._c.watcher._wait_for_ping(HOST_CHILD_START_WAIT_SECONDS)


    def model(self) -> menu.MenuModel:
        try:
            recorded = outcome.read(self._c.home)
        except HostError:
            recorded = None
        return menu.menu_model(
            self._c.presence.distro,
            self._c.presence.engine,
            self._c.presence.owner,
            None if recorded is None else recorded.state,
        )

    def try_again(self) -> None:
        threading.Thread(
            target=self._move, args=("tried again from the tray menu",),
            name="crucible-try-again", daemon=True,
        ).start()

    def on_click(self, item_id: str) -> None:
        self._c.log.write(f"menu: {item_id}")
        if item_id == menu.OPEN_CONSOLE:
            open_console(self._c.home, self._c.log)
        elif item_id == menu.INSTALL_ENGINE:
            open_console(self._c.home, self._c.log)
        elif item_id == menu.TRY_AGAIN:
            self.try_again()
        elif item_id == menu.RESTART_ENGINE:
            if self._refuse_acting_on_a_found_engine("restart"):
                return
            self.restart_engine(
                lambda event: self._c.log.write(f"restart {event.event}: {event.data}")
            )
            self._hold()
        elif item_id == menu.STOP_ENGINE:
            if self._refuse_acting_on_a_found_engine("stop"):
                return
            self.local_stop()
        elif item_id == menu.OPEN_LOG:
            open_log(self._c.log)
        elif item_id == menu.QUIT:
            self.quit()
        self._refresh()

    def _refuse_acting_on_a_found_engine(self, verb: str) -> bool:
        if self._c.presence.owner is not Owner.FOUND:
            return False
        self._c.log.write(
            f"menu: refusing to {verb} an engine this host did not start "
            "(owner=found)"
        )
        return True

    def _stop_engine(self) -> None:
        if self._c.presence.distro is Distro.PRESENT:
            probe = self._c.watcher.probe_unit()
            if not probe.readable:
                self._c.log.write(f"stop: NOT RUN — {probe.detail}")
                raise HostError("engine_stop_failed", probe.detail)
            result = self._c.runner.run(
                presence_module.system_systemctl_argv(
                    self._c.watcher.distro,
                    "stop",
                ),
                timeout_s=60.0,
            )
            self._c.log.write(f"stop: {'ok' if result.ok else result.said()}")
            if not result.ok:
                raise HostError("engine_stop_failed", result.said())
        else:
            self._c.watcher.stop_child()
        self._mark_stopped()

    def _mark_stopped(self) -> None:
        self._c.home.mkdir(parents=True, exist_ok=True)
        self._c.home.joinpath("engine.stopped").write_text("Stopped by the operator\n", encoding="utf-8")
        self._paused = True
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


    def watch(self) -> None:
        while not self._stop.wait(self._c.watcher.watch_s):
            with self._operation:
                if not self._paused:
                    before = self._c.presence.engine
                    self._c.presence = self._c.watcher.poll(
                        self._c.presence.distro, self._c.presence.owner
                    )
                    if self._c.watcher.held_distro is None:
                        self._hold()
                    self._c.watcher.rehold()
                    if (self._c.presence.owner is Owner.WSL_UNIT
                            and self._c.presence.engine is Engine.RUNNING
                            and (self._c.home / installer.CLEANUP_RECORD).exists()
                            and not self._cleanup_running and time.monotonic() >= self._cleanup_retry_at):
                        self._cleanup_running = True
                        threading.Thread(target=self._resume_model_cleanup, name="crucible-model-cleanup", daemon=True).start()
                    if self._c.presence.engine is not before:
                        self._c.log.write(
                            f"watch: {before.value} -> {self._c.presence.engine.value} — "
                            f"{self._c.presence.detail}"
                        )
                        if self._c.presence.engine is Engine.RUNNING:
                            self._claimed = False
                            self.claim()
                        self._refresh()
                self._presence_settled.set()

    def quit(self, *, handover: bool = False) -> None:
        owner = self._c.presence.owner
        claim = "released" if self._claimed else "not held, so nothing to release"
        engine = (
            "stopped with it, being this process's child"
            if owner is Owner.HOST_CHILD
            else "left running"
        )
        self._c.log.write(
            f"quit: stopping this orchestrator (owner={owner.value}); the "
            f"claim is {claim} and the engine is {engine}"
        )
        self._stop.set()
        self.release_claim()
        if handover:
            self._c.watcher.hand_over()
        else:
            self._c.watcher.release()
        if owner is Owner.HOST_CHILD:
            self._c.watcher.stop_child()
        self._shutdown_complete.set()
        if self._icon is None:
            self._c.log.write("quit: shutdown complete; controller loop signalled")
            return
        self._icon.stop()


def run(argv: list[str] | None = None, *, headless: bool = False) -> int:
    env = os.environ
    from ..config import crucible_home
    home = crucible_home()
    log = HostLog(Path(str(log_path(env))), Path(str(previous_log_path(env))))
    runner = ProcessRunner(sys.platform, env, cwd=str(home))
    log.write(f"crucible orchestrator {VERSION} starting; CRUCIBLE_HOME={home}")
    acquire(home)

    try:
        outcome = startup.install(runner)
        log.write(f"startup: {outcome.detail}")
    except HostError as exc:
        log.write(f"startup: NOT written — {exc.code}: {exc.message}")

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
    if host._paused:
        context.presence = Presence(Distro.UNKNOWN, Engine.STOPPED, "Stopped by the operator", Owner.NONE)
    else:
        host.start()
    _write_pairing(context)
    host.claim()

    door = OrchestratorDoor(
        log,
        _sequence(context, host),
        token=lambda: engine_token(context),
        token_detail=lambda: engine_token_detail(context),
        orchestrator=host,
    )
    host._install_door = door
    try:
        host._door_server = serve(door)
        log.write("door: listening on 127.0.0.1:7101")
    except OSError as exc:
        log.write(f"door: NOT listening ({exc}); shutting down this controller's owned child")
        host.quit()
        raise HostError("host_door_unavailable", f"The local controller port is unavailable: {exc}") from exc

    threading.Thread(target=host.watch, name="crucible-watch", daemon=True).start()

    from ..local import publish_installation
    publish_installation(home)
    threading.Thread(
        target=host.carry_guest_to_this_release,
        name="crucible-guest-release",
        daemon=True,
    ).start()
    if headless:
        host._shutdown_complete.wait()
        if host._door_server is not None:
            host._door_server.shutdown()
            host._door_server.server_close()
        return 0

    from . import tray

    icon = tray.make_icon(host.model(), host.on_click)
    host._icon = icon
    icon.run()
    return 0


def _sequence(context: HostContext, host: Host) -> Callable[[Callable[[installer.Event], None]], None]:
    def install_sequence(
        emit: Callable[[installer.Event], None],
        *,
        restarts: int,
        rebooted: bool,
        walks: list[installer.EngineInstall],
    ) -> None:
        if context.presence.owner is Owner.WSL_UNIT:
            if (context.home / installer.CLEANUP_RECORD).exists():
                host._resume_model_cleanup(raise_errors=True)
            else:
                host._verify_active_guest()
            walk = installer.EngineInstall(context.runner, emit, release=context.release,
                                           home=context.home,
                                           install_sh_url=INSTALL_SH_URL.format(release=context.release))
            walk._complete()
            return
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
        walk = installer.EngineInstall(
            context.runner,
            emit,
            release=context.release,
            home=context.home,
            install_sh_url=INSTALL_SH_URL.format(release=context.release),
            restarts=restarts,
            rebooted=rebooted,
            log=context.log.write,
            windows_catalog=windows,
            guest_catalog=guest,
            stop_windows_server=host.stop_windows_for_move,
            switch_pairing=host.finish_wsl_move,
            windows_after_switch=host.stopped_windows_catalog,
        )
        walks.append(walk)
        walk.run()

    def run_sequence(emit: Callable[[installer.Event], None]) -> None:
        with host._operation:
            host.check_restartable()
            previous = outcome.read(context.home)
            attempt = (
                previous.attempts + 1
                if previous is not None and previous.state == outcome.FAILED
                else 1
            )
            restarts = 0
            rebooted = True
            if previous is not None and previous.state == outcome.REBOOT_PENDING:
                restarts = previous.restarts
                rebooted = _booted_since(previous)
            walks: list[installer.EngineInstall] = []
            recorded = False

            def record(state: str, code: str | None, sentence: str | None) -> None:
                nonlocal recorded
                if recorded:
                    return
                recorded = True
                written = outcome.write(
                    context.home,
                    state=state,
                    code=code,
                    sentence=sentence,
                    release=context.release,
                    attempts=attempt,
                    restarts=walks[-1].restarts if walks else restarts,
                )
                context.log.write(
                    f"outcome: {outcome.path(context.home)} says {written.state}"
                    + (f" ({written.code})" if written.code else "")
                    + f", written {written.at}"
                )

            def emit_recorded(event: installer.Event) -> None:
                if event.event == "failed":
                    code = event.data.get("code")
                    message = event.data.get("message")
                    code = code if isinstance(code, str) else "task_failed"
                    record(
                        outcome.classify(code),
                        code,
                        message if isinstance(message, str) else None,
                    )
                elif event.event == "done":
                    record(outcome.DONE, None, None)
                emit(event)

            try:
                install_sequence(
                    emit_recorded, restarts=restarts, rebooted=rebooted, walks=walks
                )
            except HostError as exc:
                record(outcome.classify(exc.code), exc.code, exc.message)
                raise
            except Exception as exc:
                record(outcome.FAILED, "task_failed", f"{type(exc).__name__}: {exc}")
                raise
            record(outcome.DONE, None, None)

    return run_sequence


def _booted_since(previous: outcome.Outcome) -> bool:
    boot = wslstate.booted_at()
    at = previous.at_epoch()
    if boot is None or at is None:
        return True
    return boot > at


def _guest_line(context: HostContext) -> str | None:
    owner = context.presence.owner
    if owner is Owner.WSL_UNIT:
        return context.watcher.read_guest_pairing(
            context.watcher.distro
        )
    if owner is Owner.FOUND:
        return None if context.watcher.found is None else context.watcher.found.line
    return None


def _host_mode_line(context: HostContext) -> str | None:
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
