from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import catalog, service, wsl
from .errors import CrucibleError
from .platform import hostconfig
from .platform.paths import LOG_NAME
from .processlock import alive
from .wsl import CRUCIBLE_DISTRO, GUEST_CRUCIBLE


class UninstallError(CrucibleError):
    ...


STARTUP = "startup"

UNINSTALL_MECHANISM: dict[str, str] = {
    "linux": service.SYSTEMD,
    "darwin": service.LAUNCHD,
    "win32": STARTUP,
}


def mechanism_for_platform(platform: str) -> str:
    found = UNINSTALL_MECHANISM.get(platform)
    if found is None:
        raise UninstallError(
            f"uninstall_no_mechanism: there is no Crucible supervisor on "
            f"{platform!r}; this command undoes {sorted(UNINSTALL_MECHANISM)} "
            f"({service.SYSTEMD} on linux, {service.LAUNCHD} on darwin, the "
            "Startup shortcut and the tray on win32)"
        )
    return found


SUBJECT_DIRS: dict[str, str] = {
    "model": "models",
    "voice": "voices",
    "rvc": "rvc",
    "rvc-base": "rvc-base",
    "denoise": "denoise-models",
    "engine": "engines",
}

STATE_DIRS: tuple[str, ...] = ("logs", "downloads")
USER_DATA_DIRS: tuple[str, ...] = ("jobs", "uploads")

STATE_FILES: tuple[str, ...] = (
    "migration-cleanup.json",
    "installation.json",
    "wsl-outcome.json",
    "host.pid",
    "narrator-higgs-voices.json",
    "narrator-reference.wav",
)

ENVS_DIR = "envs"

PACK_DIRS: tuple[str, ...] = ("server", "host")

CONFIG_NAME = hostconfig.CONFIG_NAME
PAIRING_NAME = "pairing"

WSL_TIMEOUT_SECONDS = 600.0

LOCAL_VERB: tuple[str, ...] = ("-m", "crucible.cli", "local")

CONTROLLER_STEPS: frozenset[str] = frozenset({"stop-engine", "stop-controller"})

FATAL_BEFORE_REMOVAL: frozenset[str] = CONTROLLER_STEPS | {"remove-sharing", "remove-service"}


@dataclass(frozen=True)
class Refusal:
    code: str
    message: str
    fatal: bool

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "fatal": self.fatal}


REMOVE = "remove"
STOP = "stop"
KEEP = "keep"


@dataclass
class Step:
    name: str
    what: str
    action: str
    target: str
    bytes: int | None = None
    done: bool = False
    refused: Refusal | None = None
    detail: tuple[str, ...] = ()
    act: Callable[[], list[str]] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "name": self.name,
            "what": self.what,
            "action": self.action,
            "target": self.target,
            "done": self.done,
        }
        if self.bytes is not None:
            row["bytes"] = self.bytes
        if self.refused is not None:
            row["refused"] = self.refused.to_dict()
        if self.detail:
            row["detail"] = list(self.detail)
        return row


@dataclass
class Plan:
    home: Path
    platform: str
    mechanism: str
    purge_weights: bool
    wsl_too: bool
    backend_kind: str | None
    steps: list[Step]
    dry_run: bool = True

    @property
    def fatal(self) -> list[Step]:
        return [
            step
            for step in self.steps
            if step.refused is not None and step.refused.fatal
        ]

    def kept(self) -> dict[str, Any]:
        kept = [
            step
            for step in self.steps
            if step.action == KEEP and step.refused is None
        ]
        paths = [step.target for step in kept]
        weights = sum(
            step.bytes or 0 for step in kept if step.name.startswith("weights:")
        )
        return {"weights_bytes": weights, "paths": paths}

    def removed_bytes(self) -> int:
        return sum(
            step.bytes or 0
            for step in self.steps
            if step.action == REMOVE and step.done
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "home": str(self.home),
            "platform": self.platform,
            "mechanism": self.mechanism,
            "backend_kind": self.backend_kind,
            "purge_weights": self.purge_weights,
            "wsl_too": self.wsl_too,
            "steps": [step.to_dict() for step in self.steps],
            "kept": self.kept(),
            "removed_bytes": self.removed_bytes(),
            "ok": not self.fatal,
        }


def path_bytes(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    if not path.is_dir():
        return 0
    total = 0
    for root, _dirs, names in os.walk(path, onerror=lambda _exc: None):
        for name in names:
            try:
                total += (Path(root) / name).lstat().st_size
            except OSError:
                continue
    return total


def gib(value: int) -> str:
    return f"{value / 1024 ** 3:.2f} GiB"


def _remove_path(home: Path, path: Path) -> list[str]:
    resolved = path.resolve()
    root = home.resolve()
    if resolved != root and root not in resolved.parents:
        raise UninstallError(
            f"unsafe_target: {resolved} is not under {root}, and this command "
            "removes nothing outside CRUCIBLE_HOME"
        )
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return [f"removed directory {path}"]
    path.unlink()
    return [f"removed {path}"]


def wsl_list_argv() -> list[str]:
    return wsl.list_argv()


parse_wsl_list = wsl.parse_distro_list


def wsl_uninstall_argv(
    *, purge_weights: bool, dry_run: bool, distro: str = CRUCIBLE_DISTRO
) -> list[str]:
    flags = " --json"
    if purge_weights:
        flags += " --purge-weights"
    if dry_run:
        flags += " --dry-run"
    return wsl.guest_shell_argv(distro, f'"{GUEST_CRUCIBLE}" uninstall{flags}')


def local_argv(running_from: Path, action: str) -> list[str]:
    return [str(running_from), *LOCAL_VERB, action]


_alive = alive


def read_host_pid(home: Path) -> int | None:
    lock = home / "host.pid"
    if not lock.is_file():
        return None
    try:
        text = lock.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text.isdigit():
        return None
    pid = int(text)
    return pid if _alive(pid) else None


def read_backend_kind(home: Path) -> str | None:
    return hostconfig.read_backend_kind(home)


def known_entries() -> set[str]:
    return {
        CONFIG_NAME,
        PAIRING_NAME,
        ENVS_DIR,
        "launcher.json", "sharing.json", "bin",
        *PACK_DIRS,
        *STATE_DIRS,
        *USER_DATA_DIRS,
        *STATE_FILES,
        *SUBJECT_DIRS.values(),
    }


def plan(
    *,
    home: Path,
    platform: str,
    env: Mapping[str, str],
    runner: service.Runner,
    purge_weights: bool = False,
    wsl_too: bool = False,
    user_home: Path | None = None,
    executable: str | None = None,
) -> Plan:
    home = Path(home)
    mechanism = mechanism_for_platform(platform)
    operator_home = service.user_home() if user_home is None else user_home
    running_from = Path(executable if executable is not None else sys.executable)
    steps: list[Step] = []

    if (home / "sharing.json").is_file():
        from . import sharing
        from .platform.runner import ProcessRunner
        steps.append(Step(
            name="remove-sharing", what="withdraw the owned Tailscale address and forward",
            action=REMOVE, target=str(home / "sharing.json"),
            act=lambda: [str(sharing.disable(home, ProcessRunner(platform, env), sharing.PairedEngine(home)))],
        ))

    steps.append(_stop_step(mechanism, home, operator_home, runner, running_from))

    if wsl_too:
        steps.append(_wsl_step(platform, runner, purge_weights=purge_weights))

    if mechanism == STARTUP:
        steps.append(_controller_step(home, runner, running_from))

    steps.append(_service_step(mechanism, home, operator_home, env, runner))
    if platform in ("win32", "darwin") and (home / "installation.json").is_file():
        steps.append(Step(
            name="remove-desktop", what="remove Crucible's registered desktop presence",
            action=REMOVE, target="Crucible desktop registration",
            act=lambda: _run_or_raise(
                runner, [str(running_from), "-m", "crucible.cli", "local", "remove-desktop"],
                "desktop_remove_failed", "the desktop registration could not be removed",
            ),
        ))
    if (home / "launcher.json").is_file():
        from . import launcher
        steps.append(Step(
            name="remove-cli", what="remove the owned CLI launcher and user PATH entry",
            action=REMOVE, target=str(home / "launcher.json"),
            act=lambda: launcher.remove(home),
        ))

    steps.append(
        _path_step(
            name="remove-envs",
            what="the job-type environments; `crucible install <type>` rebuilds one",
            home=home,
            path=home / ENVS_DIR,
            absent_code="envs_absent",
        )
    )

    steps.append(
        _path_step(
            name="remove-pairing",
            what="the pairing file an app on this machine reads (3.6)",
            home=home,
            path=home / PAIRING_NAME,
            absent_code="pairing_absent",
        )
    )
    steps.append(
        _path_step(
            name="remove-config",
            what="config.toml — the bearer token goes with it",
            home=home,
            path=home / CONFIG_NAME,
            absent_code="config_absent",
        )
    )

    for name in STATE_DIRS:
        steps.append(
            _path_step(
                name=f"remove-{name}",
                what=f"<home>/{name}",
                home=home,
                path=home / name,
                absent_code=f"{name}_absent",
            )
        )
    for name in STATE_FILES:
        steps.append(
            _path_step(
                name=f"remove-{name}",
                what=f"<home>/{name}",
                home=home,
                path=home / name,
                absent_code="file_absent",
            )
        )

    for name in USER_DATA_DIRS:
        path = home / name
        if path.exists():
            steps.append(Step(
                name=f"keep-data:{name}", what="user inputs and partial job output survive uninstall",
                action=KEEP, target=str(path), bytes=path_bytes(path),
            ))
    for kind in catalog.KINDS:
        steps.append(
            _weights_step(home, kind, SUBJECT_DIRS[kind], purge_weights=purge_weights)
        )

    for name in PACK_DIRS:
        step = _pack_step(home, name, running_from)
        if step is not None:
            steps.append(step)

    for name in _strangers(home):
        path = home / name
        steps.append(
            Step(
                name=f"keep-unknown:{name}",
                what=(
                    "Crucible did not put this here, so it is not Crucible's to "
                    "delete"
                ),
                action=KEEP,
                target=str(path),
                bytes=path_bytes(path),
            )
        )

    steps.append(_home_step(home, steps))

    return Plan(
        home=home,
        platform=platform,
        mechanism=mechanism,
        purge_weights=purge_weights,
        wsl_too=wsl_too,
        backend_kind=read_backend_kind(home),
        steps=steps,
    )


def _strangers(home: Path) -> list[str]:
    if not home.is_dir():
        return []
    known = known_entries()
    try:
        return sorted(entry.name for entry in home.iterdir() if entry.name not in known)
    except OSError:
        return []


def _path_step(
    *, name: str, what: str, home: Path, path: Path, absent_code: str
) -> Step:
    if not path.exists() and not path.is_symlink():
        return Step(
            name=name,
            what=what,
            action=REMOVE,
            target=str(path),
            refused=Refusal(
                code=absent_code,
                message=f"there is no {path}; nothing to remove",
                fatal=False,
            ),
        )
    return Step(
        name=name,
        what=what,
        action=REMOVE,
        target=str(path),
        bytes=path_bytes(path),
        act=lambda: _remove_path(home, path),
    )


def _weights_step(home: Path, kind: str, dirname: str, *, purge_weights: bool) -> Step:
    path = home / dirname
    size = path_bytes(path)
    if not path.exists():
        return Step(
            name=f"weights:{kind}",
            what=f"{kind} subjects",
            action=REMOVE if purge_weights else KEEP,
            target=str(path),
            refused=Refusal(
                code="weights_absent",
                message=f"this server holds no {kind} subjects ({path} is not there)",
                fatal=False,
            ),
        )
    if not purge_weights:
        return Step(
            name=f"weights:{kind}",
            what=(
                f"{kind} subjects — KEPT ({gib(size)}). They are the expensive "
                "part (3.5); `--purge-weights` is what deletes them"
            ),
            action=KEEP,
            target=str(path),
            bytes=size,
        )
    return Step(
        name=f"weights:{kind}",
        what=f"{kind} subjects — {gib(size)}, removed because --purge-weights",
        action=REMOVE,
        target=str(path),
        bytes=size,
        act=lambda: _remove_path(home, path),
    )


def _pack_step(home: Path, name: str, running_from: Path) -> Step | None:
    path = home / name
    if not path.is_dir():
        return None
    try:
        inside = path.resolve() in running_from.resolve().parents
    except OSError:
        inside = False
    whose = (
        "the interpreter running this very command"
        if inside
        else "a relocatable interpreter this command did not unpack"
    )
    return Step(
        name=f"pack:{name}",
        what=(
            f"{whose} — kept. `install.sh --uninstall` (or `install.ps1 "
            "-Uninstall`) removes the pack, because the wrapper is what "
            "unpacked it and is still running when this exits"
        ),
        action=KEEP,
        target=str(path),
        bytes=path_bytes(path),
    )


def _no_controller(name: str, what: str, home: Path) -> Step:
    return Step(
        name=name,
        what=what,
        action=STOP,
        target=str(home / "host.pid"),
        refused=Refusal(
            code="engine_not_running",
            message=(
                f"no live `crucible orchestrator` is recorded in {home / 'host.pid'}; "
                "there is nothing to stop"
            ),
            fatal=False,
        ),
    )


def _stop_engine_through_controller(
    runner: service.Runner, running_from: Path, home: Path, pid: int
) -> list[str]:
    pairing = home / PAIRING_NAME
    if not pairing.is_file():
        return [f"no {pairing}: the controller (pid {pid}) owns no engine, so there is nothing to stop"]
    return _run_or_raise(
        runner,
        local_argv(running_from, "stop"),
        "stop_failed",
        f"the controller (pid {pid}) would not stop its engine; its log is "
        f"{home / LOG_NAME}. Run `crucible local stop` once it answers, then this "
        "uninstall again",
    )


def _end_controller(
    runner: service.Runner, running_from: Path, home: Path, pid: int
) -> list[str]:
    unfinished = (
        f"the controller (pid {pid}) did not exit; its log is {home / LOG_NAME}. "
        "Nothing was force-killed: its process tree holds the wsl.exe session that "
        "keeps the Linux engine's distro up and, on a native PC, the engine itself. "
        "Run `crucible local shutdown` once it answers; if it never exits, end pid "
        f"{pid} alone (not its tree) in Task Manager, then run this uninstall again"
    )
    lines = _run_or_raise(runner, local_argv(running_from, "shutdown"), "stop_failed", unfinished)
    if _alive(pid):
        raise UninstallError(f"stop_failed: {unfinished}")
    return lines


def _stop_step(
    mechanism: str, home: Path, operator_home: Path, runner: service.Runner, running_from: Path
) -> Step:
    if mechanism == STARTUP:
        what = (
            "ask the controller to stop the engine it owns (the guest's unit or the "
            "Windows child) and leave it stopped: a cooperative stop through "
            "`crucible local stop`, so a job holding the GPU finishes its shutdown"
        )
        pid = read_host_pid(home)
        if pid is None:
            return _no_controller("stop-engine", what, home)
        return Step(
            name="stop-engine",
            what=what,
            action=STOP,
            target=f"pid {pid}",
            act=lambda: _stop_engine_through_controller(runner, running_from, home, pid),
        )

    definition = service.definition_path(mechanism, operator_home)
    if not definition.is_file():
        return Step(
            name="stop-engine",
            what="stop the service before anything it reads is deleted",
            action=STOP,
            target=str(definition),
            refused=Refusal(
                code="service_not_installed",
                message=(
                    f"there is no {mechanism} definition at {definition}, so there "
                    "is no service to stop"
                ),
                fatal=False,
            ),
        )
    label = service.UNIT_NAME if mechanism == service.SYSTEMD else service.LAUNCHD_LABEL
    return Step(
        name="stop-engine",
        what="stop the service before anything it reads is deleted",
        action=STOP,
        target=label,
        act=lambda: service.stop(mechanism, home=operator_home, runner=runner),
    )


def _controller_step(home: Path, runner: service.Runner, running_from: Path) -> Step:
    what = (
        "end the tray and its controller through `crucible local shutdown`, after "
        "the engine and the guest are stopped; its tree is never force-killed"
    )
    pid = read_host_pid(home)
    if pid is None:
        return _no_controller("stop-controller", what, home)
    return Step(
        name="stop-controller",
        what=what,
        action=STOP,
        target=f"pid {pid}",
        act=lambda: _end_controller(runner, running_from, home, pid),
    )


def _service_step(
    mechanism: str,
    home: Path,
    operator_home: Path,
    env: Mapping[str, str],
    runner: service.Runner,
) -> Step:
    if mechanism == STARTUP:
        from .platform import startup as host_startup

        try:
            lnk = host_startup.shortcut_path(env)
        except CrucibleError as exc:
            return Step(
                name="remove-service",
                what="the Startup shortcut that runs `crucible orchestrator` at login",
                action=REMOVE,
                target="(unknown)",
                refused=Refusal(
                    code="host_no_localappdata",
                    message=str(exc),
                    fatal=True,
                ),
            )
        return Step(
            name="remove-service",
            what="the Startup shortcut that runs `crucible orchestrator` at login (4.1)",
            action=REMOVE,
            target=str(lnk),
            act=lambda: _remove_startup(env),
        )

    definition = service.definition_path(mechanism, operator_home)
    if not definition.is_file():
        return Step(
            name="remove-service",
            what=f"the {mechanism} definition and its registration",
            action=REMOVE,
            target=str(definition),
            refused=Refusal(
                code="service_not_installed",
                message=f"there is no {mechanism} definition at {definition}",
                fatal=False,
            ),
        )
    return Step(
        name="remove-service",
        what=(
            "disable and delete the systemd user unit, then daemon-reload"
            if mechanism == service.SYSTEMD
            else "bootout the launchd agent and delete its plist"
        ),
        action=REMOVE,
        target=str(definition),
        bytes=path_bytes(definition),
        act=lambda: service.uninstall(mechanism, home=operator_home, runner=runner),
    )


def _remove_startup(env: Mapping[str, str]) -> list[str]:
    from .platform import startup as host_startup
    from .platform.runner import ProcessRunner

    outcome = host_startup.remove(ProcessRunner("win32", env))
    return [outcome.detail]


def _wsl_step(
    platform: str, runner: service.Runner, *, purge_weights: bool
) -> Step:
    if platform != "win32":
        return Step(
            name="wsl-guest",
            what="run the guest's own uninstall inside the Crucible distro",
            action=REMOVE,
            target=CRUCIBLE_DISTRO,
            refused=Refusal(
                code="wsl_not_here",
                message=(
                    f"--wsl-too drives a WSL2 guest through wsl.exe, and this is "
                    f"{platform}. On linux and darwin the server runs on the "
                    "machine this command is already on"
                ),
                fatal=True,
            ),
        )
    listed = runner(wsl_list_argv())
    names = parse_wsl_list(listed.stdout) if listed.ok else []
    if not listed.ok:
        return Step(
            name="wsl-guest",
            what="run the guest's own uninstall inside the Crucible distro",
            action=REMOVE,
            target=CRUCIBLE_DISTRO,
            refused=Refusal(
                code="wsl_unreadable",
                message=(
                    "wsl.exe could not be asked what distros this machine has: "
                    f"`{' '.join(listed.argv)}` exited {listed.returncode}: "
                    f"{listed.text()}"
                ),
                fatal=True,
            ),
        )
    if CRUCIBLE_DISTRO not in names:
        return Step(
            name="wsl-guest",
            what="run the guest's own uninstall inside the Crucible distro",
            action=REMOVE,
            target=CRUCIBLE_DISTRO,
            refused=Refusal(
                code="wsl_distro_absent",
                message=(
                    f"this machine has no {CRUCIBLE_DISTRO!r} distro "
                    f"(wsl -l -v lists {names or ['nothing']}). --wsl-too "
                    "uninstalls the guest Crucible imported and no other: every "
                    "distro on this list that is not that one is yours"
                ),
                fatal=True,
            ),
        )
    argv = wsl_uninstall_argv(purge_weights=purge_weights, dry_run=False)
    return Step(
        name="wsl-guest",
        what=(
            f"`crucible uninstall` inside the {CRUCIBLE_DISTRO} distro, with these "
            "same flags. The distro itself is NOT unregistered — that is yours "
            f"to run: wsl --unregister {CRUCIBLE_DISTRO}"
        ),
        action=REMOVE,
        target=CRUCIBLE_DISTRO,
        act=lambda: _run_or_raise(
            runner,
            argv,
            "wsl_uninstall_failed",
            f"the guest's own uninstall in {CRUCIBLE_DISTRO} failed",
        ),
    )


def _home_step(home: Path, planned: Sequence[Step]) -> Step:
    if not home.is_dir():
        return Step(
            name="remove-home",
            what="CRUCIBLE_HOME itself",
            action=REMOVE,
            target=str(home),
            refused=Refusal(
                code="home_absent",
                message=f"there is no {home}",
                fatal=False,
            ),
        )
    survivors = sorted(
        Path(step.target).name
        for step in planned
        if step.action == KEEP
        and step.refused is None
        and Path(step.target).parent == home
    )
    if survivors:
        return Step(
            name="remove-home",
            what="CRUCIBLE_HOME itself, once it is empty",
            action=KEEP,
            target=str(home),
            refused=Refusal(
                code="home_not_empty",
                message=(
                    f"{home} still holds {survivors}. A home is removed when it is "
                    "empty and never cleared: an entry nothing named is somebody "
                    "else's"
                ),
                fatal=False,
            ),
        )
    return Step(
        name="remove-home",
        what="CRUCIBLE_HOME itself — every step above emptied it",
        action=REMOVE,
        target=str(home),
        act=lambda: _rmdir_if_empty(home),
    )


def _rmdir_if_empty(home: Path) -> list[str]:
    left = sorted(entry.name for entry in home.iterdir())
    if left:
        raise UninstallError(
            f"home_not_empty: {home} holds {left}, which was not true when this "
            "run was planned. Nothing has been removed from it"
        )
    home.rmdir()
    return [f"removed {home}"]


def _run_or_raise(
    runner: service.Runner, argv: Sequence[str], code: str, what: str
) -> list[str]:
    ran = runner(argv)
    if not ran.ok:
        raise UninstallError(
            f"{code}: {what} — `{' '.join(ran.argv)}` exited {ran.returncode}: "
            f"{ran.text()}"
        )
    return [f"ran {' '.join(ran.argv)}", *[line for line in ran.text().splitlines()]]


def run(plan_: Plan) -> Plan:
    plan_.dry_run = False
    for step in plan_.steps:
        if step.act is None:
            step.done = step.action == KEEP and step.refused is None
            continue
        try:
            step.detail = tuple(step.act())
            step.done = True
        except (CrucibleError, OSError) as exc:
            code, _, message = str(exc).partition(": ")
            step.refused = Refusal(
                code=code if message else "step_failed",
                message=message or str(exc),
                fatal=True,
            )
            if step.name in FATAL_BEFORE_REMOVAL:
                break
    return plan_


__all__ = [
    "CRUCIBLE_DISTRO",
    "ENVS_DIR",
    "KEEP",
    "PACK_DIRS",
    "Plan",
    "REMOVE",
    "Refusal",
    "STARTUP",
    "STATE_DIRS",
    "STATE_FILES",
    "STOP",
    "SUBJECT_DIRS",
    "Step",
    "UNINSTALL_MECHANISM",
    "UninstallError",
    "gib",
    "known_entries",
    "mechanism_for_platform",
    "parse_wsl_list",
    "path_bytes",
    "plan",
    "read_backend_kind",
    "read_host_pid",
    "run",
    "local_argv",
    "wsl_list_argv",
    "wsl_uninstall_argv",
]
