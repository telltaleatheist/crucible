"""`crucible uninstall` — install, run backwards, with every step named.

Owen, 2026-09-14: *"make sure there's an uninstall route for crucible as well.
Once everything is working end to end I'll probably uninstall it on the pc (wsl
and windows) and fully reinstall end to end as a test. Same with Mac."* That is
what this module is for, and it is the sentence that sets its shape: an
uninstall whose purpose is to be followed by a reinstall must leave the machine
in the state `install.sh` expects to find, and must not cost a 57 GB download to
undo.

THE ORDER IS THE INVERSE OF THE INSTALL ORDER, AND THAT IS THE WHOLE DESIGN
---------------------------------------------------------------------------
`sdk/bootstrap/src/steps.ts` is the one owner of what installing a Crucible is:

    host-facts → server-pack → init → install-<type> → service-install
    → linger → capability-write

Read it upwards and you have this module. The service goes before the envs it
starts, because a unit whose `ExecStart` has been deleted is a unit that
crash-loops; the envs go before the config, because `envs/` is named relative to
`CRUCIBLE_HOME` and a config that is gone cannot say where that was; the weights
go last because they are the one thing a reinstall cannot cheaply replace
(PHASE15-HOST.md 3.5: *"Crucible owns WHERE the weights are"*, and they are the
expensive part). `docs/MAC-PARITY-AUDIT-2026-09-14.md` §3 states the upgrade
order for the Mac and says "order is the risk"; this is that list reversed.

WEIGHTS ARE KEPT UNLESS SOMEBODY SAYS OTHERWISE
------------------------------------------------
`--purge-weights` is a flag and never a default, and the six directories it
governs are the six `catalog.KINDS` — model, voice, rvc, rvc-base, denoise,
engine — because a subject kind and a subject directory are the same fact and
this module must not invent a seventh list of them. A run that keeps them SAYS
SO, with the byte count, so a person who meant to free the disk finds out
immediately rather than a week later.

IT DELETES ONLY WHAT IT CAN NAME
---------------------------------
Every removal target is a path this package put there, and an entry under
`CRUCIBLE_HOME` that is not in the known layout is REPORTED and left alone —
never swept up by a recursive delete of the home directory. `~/.crucible` on
Owen's Mac held an `hf-token.txt` Crucible never wrote and never read; a home
that is cleared rather than dismantled is a home that eats things like that.
The same rule is why `<home>` itself only goes when it is EMPTY.

AND IT NEVER DELETES THE PACK IT IS RUNNING FROM
-------------------------------------------------
`<home>/server` (and `<home>/host` on Windows) hold the relocatable interpreter
whose `crucible` this process IS. Unlinking a running interpreter's own
`site-packages` mid-run is undefined on POSIX and refused outright by Windows,
so this command names those directories, keeps them, and leaves them to the
wrapper that unpacked them — `install.sh --uninstall` / `install.ps1
-Uninstall`, which run after this returns and are the one owner of the pack.
That is also why `<home>` can be left standing after a `--purge-weights` run:
the pack is still in it, the wrapper removes both, and this command refuses to
guess which of the two is running it.

NOTHING HERE ASKS A QUESTION
-----------------------------
No prompt, no confirmation, no "are you sure". The flags decide, `--dry-run`
prints the whole plan and touches nothing, and `--json` is the same plan for an
app. A destructive command that is interactive cannot be scripted, and a
destructive command that is scripted must be readable before it runs.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import catalog, service
from .errors import CrucibleError
# `crucible/host/wsl_states.py` is generated DATA and imports nothing heavier
# than `dataclasses` — the host package's own header pins that (pytest runs in
# WSL, where pystray and tkinter do not exist), which is what makes it safe to
# name the distro from here rather than spelling it a second time.
from .host.wsl_states import CRUCIBLE_DISTRO


class UninstallError(CrucibleError):
    """An uninstall could not be planned. Names what it could not answer."""


# ------------------------------------------------------------ the mechanism

#: The Windows "service": there is none. A tray program is a per-user login
#: item (PHASE15-HOST.md 4.1), so what an uninstall removes on win32 is the
#: Startup shortcut and the tray process, not a unit.
STARTUP = "startup"

#: Which supervisor to undo, per PLATFORM.
#:
#: `crucible/service.py` keys the same fact off the BACKEND, and both are
#: right: PHASE15-HOST.md 3.5 says *"a backend runs where its engine runs and
#: nowhere else"*, so `cuda-linux` IS linux and `mlx-darwin` IS darwin. This
#: one is keyed off the platform because an uninstall must work on a machine
#: whose `config.toml` has already been deleted by a previous half-run — and
#: because asking `detect_backend()` would probe a card to answer a question
#: about a service manager. `tests/test_uninstall.py` pins the two against each
#: other, the same tied-by-a-check seam `envpack.SMOKE_IMPORT` uses.
UNINSTALL_MECHANISM: dict[str, str] = {
    "linux": service.SYSTEMD,
    "darwin": service.LAUNCHD,
    "win32": STARTUP,
}


def mechanism_for_platform(platform: str) -> str:
    """This platform's supervisor, or a refusal naming the platforms there are."""
    found = UNINSTALL_MECHANISM.get(platform)
    if found is None:
        raise UninstallError(
            f"uninstall_no_mechanism: there is no Crucible supervisor on "
            f"{platform!r}; this command undoes {sorted(UNINSTALL_MECHANISM)} "
            f"({service.SYSTEMD} on linux, {service.LAUNCHD} on darwin, the "
            "Startup shortcut and the tray on win32)"
        )
    return found


# ---------------------------------------------------------------- the layout

#: The subject directory each `catalog.KINDS` entry keeps its weights in,
#: relative to `CRUCIBLE_HOME`. These are the six `--purge-weights` governs.
#: Every value is the same string its own module builds the path from
#: (`weights.weights_root`, `rvcbase`, `denoisemodels`, `llamacpp`), and
#: `tests/test_uninstall.py` asserts the KEYS are exactly `catalog.KINDS` —
#: a seventh kind arriving without a seventh directory here would otherwise
#: be silently kept forever.
SUBJECT_DIRS: dict[str, str] = {
    "model": "models",
    "voice": "voices",
    "rvc": "rvc",
    "rvc-base": "rvc-base",
    "denoise": "denoise-models",
    "engine": "engines",
}

#: Directories under `CRUCIBLE_HOME` that hold this server's working state.
#: Removed on every run: a reinstall regenerates all of them, and `jobs/` and
#: `uploads/` after an uninstall are scratch for a server that is gone.
STATE_DIRS: tuple[str, ...] = ("logs", "jobs", "uploads", "downloads")

#: Loose files under `CRUCIBLE_HOME` this package writes and can regenerate.
#: `host.pid` is the tray's lock (`crucible/host/app.py`), the other two are
#: documents the narrator engine rebuilds on demand.
STATE_FILES: tuple[str, ...] = (
    "host.pid",
    "narrator-higgs-voices.json",
    "narrator-reference.wav",
)

#: The env packs' directory. One step removes the lot: `crucible install`
#: rebuilds any of them from a published pack.
ENVS_DIR = "envs"

#: The two relocatable interpreters. NEVER removed here — see the header.
PACK_DIRS: tuple[str, ...] = ("server", "host")

CONFIG_NAME = "config.toml"
PAIRING_NAME = "pairing"

#: Where a Crucible installed by `install.sh` keeps its interpreter inside the
#: guest. `$HOME` is expanded by the guest's own bash, never by wsl.exe — see
#: `wsl_uninstall_argv`.
GUEST_CRUCIBLE = "$HOME/.crucible/server/bin/crucible"

#: How long `--wsl-too` waits for the guest's own uninstall. It removes
#: directories and stops a unit; it never downloads anything.
WSL_TIMEOUT_SECONDS = 600.0

#: How long the Windows tray gets to die.
TASKKILL_TIMEOUT_SECONDS = 60.0


# ------------------------------------------------------------------- results


@dataclass(frozen=True)
class Refusal:
    """Why a step did not happen, by a name a person can search for.

    `fatal` is the difference between *"there was nothing to remove"* and
    *"there was something and it would not go"*. The first is an ordinary
    outcome of uninstalling a machine that is already half clean, and an
    uninstall that exited 1 over it could never be run twice; the second is a
    failure and the command says so with its exit code.
    """

    code: str
    message: str
    fatal: bool

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "fatal": self.fatal}


#: What a step does to its target. Three words and no fourth: `keep` is a
#: RESULT, not the absence of one — a weights directory left on disk is a
#: decision this command made and reports, with its size.
REMOVE = "remove"
STOP = "stop"
KEEP = "keep"


@dataclass
class Step:
    """One act of an uninstall, against exactly one target.

    One target per step, deliberately: an app draws this list, and a step that
    quietly stood for four paths would give a person a progress bar that lies
    about what is being deleted.
    """

    name: str
    what: str
    action: str
    target: str
    #: Bytes on disk, for a path. `None` where the target is not a path (a
    #: unit, a pid, a distro) — which is not the same as zero.
    bytes: int | None = None
    #: Did THIS run perform the step? Always false after a `--dry-run`.
    done: bool = False
    refused: Refusal | None = None
    #: Lines the act produced, for the human transcript.
    detail: tuple[str, ...] = ()
    #: How the step is performed. Never called by a dry run, and never
    #: serialised.
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
    """Every step, in order, and the facts they were planned against."""

    home: Path
    platform: str
    mechanism: str
    purge_weights: bool
    wsl_too: bool
    #: The backend this home's `config.toml` recorded, or None when there is
    #: no config to read it from. REPORTED, never detected: an uninstall does
    #: not probe a card.
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
        """The paths this plan leaves behind, and what the weights among them cost.

        A REFUSED keep is not a path that was left behind — it is a path that
        was never there — so it is not listed. See `_home_step`.
        """
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


# ------------------------------------------------------------------ measuring


def path_bytes(path: Path) -> int:
    """How much disk this path holds. 0 for one that is not there.

    Metadata only — `st_size` per file, no reads — so measuring a 57 GB voices
    directory costs a directory walk and not a byte of I/O. A file that
    disappears mid-walk is skipped rather than raising: this number is for a
    sentence, and a half-deleted tree is still worth reporting the size of.
    """
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


# ------------------------------------------------------------------- removing


def _remove_path(home: Path, path: Path) -> list[str]:
    """Delete one file or directory, refusing anything outside `home`.

    The containment check is not paranoia about this module's own constants —
    it is about `CRUCIBLE_HOME`, which an operator sets and which a test sets,
    and about the day somebody adds a step whose target is composed rather
    than named. A delete that can only ever run inside one directory is a
    delete that can be read once and trusted afterwards.
    """
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


# ------------------------------------------------------------------ the steps


def wsl_list_argv() -> list[str]:
    """`wsl.exe -l -q` — which distros this machine has."""
    return ["wsl.exe", "-l", "-q"]


def parse_wsl_list(text: str) -> list[str]:
    """Distro names out of `wsl -l -q`'s UTF-16-ish output.

    The same shape `crucible/host/presence.py` parses, kept here rather than
    imported because that module's parser takes the verbose listing. What both
    have to survive is the NULs `wsl.exe` writes between characters when its
    output is redirected.
    """
    return [
        line.strip()
        for line in text.replace("\x00", "").splitlines()
        if line.strip() != ""
    ]


def wsl_uninstall_argv(
    *, purge_weights: bool, dry_run: bool, distro: str = CRUCIBLE_DISTRO
) -> list[str]:
    """The guest's own `crucible uninstall`, run through wsl.exe.

    `--exec` and then `bash -lc`, in that order, and both halves matter.
    `--exec` stops wsl.exe pre-expanding `$HOME` on the WINDOWS side (BookForge's
    `wsl-exe-implicit-shell-trap.md`: without it, `$var` is expanded by the
    Windows shell before bash exists); `bash -lc` is then what expands it inside
    the guest, where the answer is the guest user's home. Composing the guest
    path on the Windows side would mean this command knowing the guest's
    username, which it has no honest way to learn.

    `--wsl-too` is deliberately NOT passed on: there is no distro inside the
    distro, and a flag that recursed would be a flag that could.
    """
    flags = " --json"
    if purge_weights:
        flags += " --purge-weights"
    if dry_run:
        flags += " --dry-run"
    return [
        "wsl.exe",
        "-d",
        distro,
        "--exec",
        "bash",
        "-lc",
        f'"{GUEST_CRUCIBLE}" uninstall{flags}',
    ]


def taskkill_argv(pid: int) -> list[str]:
    """End the tray and everything it started.

    `/T` because the host-mode server is the tray's CHILD (PHASE15-HOST.md 4.1:
    *"in host mode the child server stops with it"*), and a tree kill is the
    only way to take a `llama-server` grandchild off the card with it. `/F`
    because a `pythonw.exe` with no window has nothing for a polite WM_CLOSE to
    arrive at, and because the tray holds no unflushed state — `host.log` is
    appended line by line as it goes.

    By PID, from the lock file the tray itself wrote, and never by image name:
    `pythonw.exe` is not Crucible's, and killing every one of them on the
    machine would end somebody's editor.
    """
    return ["taskkill.exe", "/PID", str(pid), "/T", "/F"]


def _alive(pid: int) -> bool:
    """Is this pid a live process? A stale lock must not look like a tray."""
    if sys.platform == "win32":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # type: ignore[attr-defined]
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_host_pid(home: Path) -> int | None:
    """The tray's pid out of `<home>/host.pid`, when one is alive there."""
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
    """`[backend] kind` out of this home's config, or None because there is none.

    Read with `tomllib` and not `load_config`, because a config that fails
    validation is still a config whose backend an uninstall would like to
    report — and because `load_config` refusing would turn "tell me what this
    machine had" into a failure.
    """
    import tomllib

    path = home / CONFIG_NAME
    if not path.is_file():
        return None
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    backend = document.get("backend")
    if not isinstance(backend, dict):
        return None
    kind = backend.get("kind")
    return kind if isinstance(kind, str) and kind else None


def known_entries() -> set[str]:
    """Every name this package puts directly under `CRUCIBLE_HOME`.

    Anything else found there is somebody's, and `plan` reports it as kept.
    """
    return {
        CONFIG_NAME,
        PAIRING_NAME,
        ENVS_DIR,
        *PACK_DIRS,
        *STATE_DIRS,
        *STATE_FILES,
        *SUBJECT_DIRS.values(),
    }


# -------------------------------------------------------------------- the plan


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
    """Every step this machine's uninstall is, in order, planned but not run.

    Nothing here touches the disk beyond reading it. `run()` is what performs
    the plan, and `--dry-run` is simply a plan that is never handed to it —
    which is the only definition of "dry run" that cannot drift from the real
    thing.

    `runner` is `crucible/service.py`'s injectable one and is used for the two
    steps that ask the machine a question at PLAN time (is there a `crucible`
    distro?) as well as by the acts. `env` and `executable` are parameters
    rather than reads of `os.environ` and `sys.executable` for the reason
    `crucible/host/paths.py` states: the suite runs on a machine that is not
    the one these rules are for.
    """
    home = Path(home)
    mechanism = mechanism_for_platform(platform)
    operator_home = service.user_home() if user_home is None else user_home
    running_from = Path(executable if executable is not None else sys.executable)
    steps: list[Step] = []

    # 1. STOP. Before anything is deleted: a unit whose ExecStart has gone is a
    #    unit systemd restarts every two seconds (`service.py`'s Restart=always).
    steps.append(_stop_step(mechanism, home, operator_home, runner))

    # 2. THE GUEST, when this Windows machine manages one. It goes here and not
    #    later because the tray is what boots and watches the guest (4.1): with
    #    the tray still alive, a guest whose unit has just been removed is a
    #    guest the tray tries its two recovery recipes on.
    if wsl_too:
        steps.append(_wsl_step(platform, runner, purge_weights=purge_weights))

    # 3. THE SERVICE ITSELF.
    steps.append(_service_step(mechanism, home, operator_home, env, runner))

    # 4. THE ENVS. `crucible install <type>` rebuilds any of them from a pack.
    steps.append(
        _path_step(
            name="remove-envs",
            what="the job-type environments; `crucible install <type>` rebuilds one",
            home=home,
            path=home / ENVS_DIR,
            absent_code="envs_absent",
        )
    )

    # 5. THE PAIRING FILE, then 6. THE CONFIG. This order and not the reverse:
    #    the pairing file is a COPY of four facts the config owns (3.6), and a
    #    machine left holding the copy with the original gone would point an
    #    app at a door that is not there.
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

    # 7. THE WORKING STATE.
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

    # 8. THE WEIGHTS — kept unless asked, and the size is said either way.
    for kind in catalog.KINDS:
        steps.append(
            _weights_step(home, kind, SUBJECT_DIRS[kind], purge_weights=purge_weights)
        )

    # 9. THE PACKS, which this command never removes. Named so the wrapper's
    #    job is visible in the plan rather than implied by its absence.
    for name in PACK_DIRS:
        step = _pack_step(home, name, running_from)
        if step is not None:
            steps.append(step)

    # 10. ANYTHING ELSE. Reported, kept, never swept.
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

    # 11. THE HOME, if the ten steps above emptied it.
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
    """Entries directly under `home` that this package did not put there."""
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
    """A remove of one path, or a non-fatal refusal saying it was not there."""
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
    """One subject directory: removed on `--purge-weights`, kept and priced otherwise."""
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
    """The relocatable interpreter. Always kept; see the module header."""
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


def _stop_step(
    mechanism: str, home: Path, operator_home: Path, runner: service.Runner
) -> Step:
    """Stop whatever is serving on this machine, by this platform's one means."""
    if mechanism == STARTUP:
        pid = read_host_pid(home)
        if pid is None:
            return Step(
                name="stop-engine",
                what="end the tray and the server it holds",
                action=STOP,
                target=str(home / "host.pid"),
                refused=Refusal(
                    code="engine_not_running",
                    message=(
                        f"no live `crucible host` is recorded in {home / 'host.pid'}; "
                        "there is nothing to stop"
                    ),
                    fatal=False,
                ),
            )
        return Step(
            name="stop-engine",
            what="end the tray and the server it holds (a tree kill, 4.1)",
            action=STOP,
            target=f"pid {pid}",
            act=lambda: _run_or_raise(
                runner,
                taskkill_argv(pid),
                "stop_failed",
                f"the tray (pid {pid}) would not stop",
            ),
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


def _service_step(
    mechanism: str,
    home: Path,
    operator_home: Path,
    env: Mapping[str, str],
    runner: service.Runner,
) -> Step:
    """Remove the unit / the plist / the Startup shortcut. Its own module does it."""
    if mechanism == STARTUP:
        from .host import startup as host_startup

        try:
            lnk = host_startup.shortcut_path(env)
        except CrucibleError as exc:
            return Step(
                name="remove-service",
                what="the Startup shortcut that runs `crucible host` at login",
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
            what="the Startup shortcut that runs `crucible host` at login (4.1)",
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
    """`crucible host --remove-startup`, through the module that owns that file."""
    from .host import startup as host_startup
    from .host.runner import ProcessRunner

    outcome = host_startup.remove(ProcessRunner("win32", env))
    return [outcome.detail]


def _wsl_step(
    platform: str, runner: service.Runner, *, purge_weights: bool
) -> Step:
    """`--wsl-too`: run the GUEST's own uninstall, and refuse when there is none.

    It runs the guest's `crucible uninstall` and stops there. It does NOT
    `wsl --unregister`, and that is a ruling rather than an omission: unregister
    destroys the distro's whole ext4, including anything the guest's own
    uninstall deliberately kept, and it is irreversible. So the guest is
    emptied by the same command, with the same flags, and the one line that
    would delete the distro itself is PRINTED for the operator to run — the
    same hand-over `install.sh` makes for `loginctl enable-linger`.
    """
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
                    f"(wsl -l -q lists {names or ['nothing']}). --wsl-too "
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
    """`<home>` itself, when nothing is left in it.

    "Empty" is computed from the PLAN and not from the disk, so a dry run
    answers the same question the real run will: a survivor is a step that
    KEEPS something directly under home.

    A REFUSED step keeps nothing, and that distinction is the whole of this
    function's correctness. `weights:voice` on a server with no voices is
    planned as a keep — the flag says keep — and refused `weights_absent`; a
    survivor list that counted it would report `~/.crucible` as non-empty
    because of six directories that are not there.
    """
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
    """Remove `home`, or refuse naming what appeared in it since the plan."""
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


# --------------------------------------------------------------------- running


def run(plan_: Plan) -> Plan:
    """Perform the plan. The same object comes back, with the outcomes filled in.

    A step that raises does not stop the run. R6 — partial work survives
    failure — is the whole point here: a machine whose systemd unit would not
    go must still get its config, its envs and its pairing file removed, and
    the operator must be told which single thing is left. The exit code carries
    the failure; the other nine steps carry the work.
    """
    plan_.dry_run = False
    for step in plan_.steps:
        if step.act is None:
            # A KEEP, or a step already refused at plan time. Performing a keep
            # IS leaving it there, so it is done.
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
    "taskkill_argv",
    "wsl_list_argv",
    "wsl_uninstall_argv",
]
