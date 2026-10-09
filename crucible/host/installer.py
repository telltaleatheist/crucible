from __future__ import annotations

import base64
import json
import re
import shlex
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .. import wsl
from ..errors import CrucibleError
from ..memorybudget import gib_text
from ..platform import lan_door
from ..platform.errors import HostError
from ..platform.paths import ENGINE_PORT, INSTALL_ONE_LINER, PROGRESS_INTERVAL_SECONDS, engine_url
from ..platform.quarantine import quarantine
from ..platform.runner import Runner, RunResult
from ..platform.wsl_table import WSL_CONF_MARKER
from ..wsl import CRUCIBLE_DISTRO, GUEST_CRUCIBLE, guest_argv
from . import outcome, wslstate
from .catalog import CatalogPort, CatalogRefusal, Subject
from .cleanup_record import CLEANUP_RECORD, record_cleanup

ENGINE_TARGET_WSL = "wsl"
IMPORT_ARTEFACTS: frozenset[str] = frozenset({"ext4.vhdx"})


STEPS: tuple[str, ...] = (
    "wsl-state",
    "import-distro",
    "guest-ready",
    "guest-install",
    "migrate-config",
    "install-job-types",
    "prepare-weights",
    "stop-windows-server",
    "switch-pairing",
    "lan-door",
    "migrate-weights",
)

STEP_WORDS: dict[str, str] = {
    "wsl-state": "checking Windows' Linux support (WSL)",
    "import-distro": "setting up the Linux system (Ubuntu)",
    "guest-ready": "checking the Linux system",
    "guest-install": "installing Crucible inside Linux (a few minutes)",
    "migrate-config": "carrying this PC's settings and pairing across",
    "install-job-types": "job types",
    "prepare-weights": "copying models to the Linux engine",
    "stop-windows-server": "stopping the Windows engine",
    "switch-pairing": "switching apps over to the Linux engine",
    "lan-door": "network sharing",
    "migrate-weights": "removing the Windows copies of moved models",
}

IMPORT_TIMEOUT_SECONDS = 30 * 60.0

IMAGE_DOWNLOAD_TIMEOUT_SECONDS = 3600.0
IMAGE_DOWNLOAD_ATTEMPTS = 3
GUEST_INSTALL_TIMEOUT_SECONDS = 120 * 60.0
QUICK_TIMEOUT_SECONDS = 5 * 60.0

MIGRATE_PULL_TIMEOUT_SECONDS = 6 * 60 * 60.0

MIGRATE_POLL_SECONDS = 5.0

MIGRATE_IN_USE_ROUNDS = 60


UPDATE_AND_RESTART_HOW = (
    'Save your work, open Start, click the power button and choose "Update and '
    'restart" (if there is no such option, choose "Restart"). Do not choose '
    '"Shut down".'
)

SIGN_IN_SENTENCE = (
    "After the restart, someone has to sign in to Windows on this PC: Crucible "
    "carries on by itself once somebody is signed in, and not before."
)

REBOOT_SENTENCE = (
    "Windows needs to restart to finish installing its Linux support (WSL). "
    + UPDATE_AND_RESTART_HOW
    + " Choose Update even if you usually put updates off: Windows installs WSL "
    "in the same step as its waiting updates, so a restart that skips or "
    "postpones them does not install it. It can take more than one restart. "
    "Nothing downloaded so far is lost. "
    + SIGN_IN_SENTENCE
)

TRY_AGAIN_HINT = (
    "right-click the Crucible icon by the clock (it may be behind the ^ arrow "
    'there) and choose "Try again"'
)

REBOOT_AGAIN_SENTENCE = (
    "Windows has restarted several times and has still not finished installing "
    "its Linux support (WSL). The usual reason is that the restarts skipped or "
    "postponed waiting Windows updates: Windows installs WSL in the same step "
    "as those updates. Open Start, then Settings, then Windows Update, and let "
    "it install everything it offers. Then restart once more. "
    + UPDATE_AND_RESTART_HOW
    + " "
    + SIGN_IN_SENTENCE
    + " If nothing "
    "has changed a few minutes after that, "
    + TRY_AGAIN_HINT
    + ". The Windows engine keeps working meanwhile."
)

REBOOT_STILL_OWED_SENTENCE = (
    "Windows restarted, but it has not finished installing its Linux support "
    "(WSL) yet. That is normal when Windows had updates waiting: it installs "
    "WSL in the same step as those updates, and sometimes that takes another "
    "restart. Restart once more. "
    + UPDATE_AND_RESTART_HOW
    + " "
    + SIGN_IN_SENTENCE
)

RESTART_BUDGET = 3


def _feature_word(was: int | None, now: int | None) -> str:
    if was == 1:
        return "was already on"
    if now == 1:
        return "turned on now"
    if now is None:
        return "state unreadable"
    return f"still {wslstate.FEATURE_STATES.get(now, 'unknown')}"


def _feature_report(before: wslstate.LiveWsl, after: wslstate.LiveWsl) -> str:
    words = [
        f"{name} {_feature_word(before.features.get(name), after.features.get(name))}"
        for name in wslstate.WSL_FEATURES
    ]
    return "; ".join(words) + f"; then: {after.answer.line()}"


GUEST_RESTART_BUDGET_SECONDS = 180.0

# `crucible capability --write` then `crucible service restart`, which waits up to
# its own RESTART_ANSWER_SECONDS (120) for the server to answer before it says
# restart_not_answering. This is the outer bound on both, so it must never be the
# one that ends the wait first (tests/test_guest_restart.py holds that).
GUEST_RESTART_TIMEOUT_SECONDS = QUICK_TIMEOUT_SECONDS


@dataclass
class Event:
    event: str
    data: dict[str, object]


Emit = Callable[[Event], None]


GUEST_BACKEND = "cuda-linux"


@dataclass
class StepRecord:
    name: str
    argv: list[str] = field(default_factory=list)
    status: str = "running"
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "argv": list(self.argv),
            "status": self.status,
            "detail": self.detail,
        }


@dataclass
class InstallOutcome:
    steps: list[StepRecord] = field(default_factory=list)
    distro: str = CRUCIBLE_DISTRO
    detail: str = ""
    server_name: str = ""
    server_url: str = ""
    config_path: str = ""
    crucible: str = ""
    release: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "server": {
                "name": self.server_name,
                "url": self.server_url,
                "config_path": self.config_path,
            },
            "release": self.release,
            "backend": GUEST_BACKEND,
            "crucible": self.crucible,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass
class _MigrationRound:
    held: dict[tuple[str, str], str] = field(default_factory=dict)
    deferred: list[str] = field(default_factory=list)
    removed: int = 0


class EngineInstall:
    def __init__(
        self,
        runner: Runner,
        emit: Emit,
        *,
        release: str,
        home: Path,
        install_sh_url: str,
        distro: str = CRUCIBLE_DISTRO,
        elevate: bool = True,
        restarts: int = 0,
        rebooted: bool = True,
        log: Callable[[str], None] | None = None,
        share_lan: bool | None = None,
        windows_catalog: CatalogPort | None = None,
        guest_catalog: CatalogPort | None = None,
        stop_windows_server: Callable[[], None] | None = None,
        switch_pairing: Callable[[], None] | None = None,
        windows_after_switch: Callable[[], CatalogPort] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._runner = runner
        self._emit = emit
        self._release = release
        self._home = Path(home)
        self._install_sh_url = install_sh_url
        self._distro = distro
        self._elevate = elevate
        self.restarts = restarts
        self._rebooted = rebooted
        self._log = log
        self._share_lan = share_lan
        self._windows = windows_catalog
        self._guest = guest_catalog
        self._stop_windows_callback = stop_windows_server
        self._switch_pairing_callback = switch_pairing
        self._windows_after_switch = windows_after_switch
        self._monotonic = monotonic
        self._sleep = sleep
        self._index = 0
        self._records: list[StepRecord] = []
        self._last_bytes = 0.0


    def _step(self, name: str) -> StepRecord:
        self._index += 1
        record = StepRecord(name=name)
        self._records.append(record)
        self._emit(
            Event("step", {"name": name, "index": self._index, "total": len(STEPS)})
        )
        return record

    def _finish(self, name: str, detail: str, *, argv: Sequence[str] = ()) -> None:
        for record in reversed(self._records):
            if record.name == name:
                record.status = "ok"
                record.detail = detail
                record.argv = list(argv)
                return
        raise HostError("wsl_state_unknown", f"no step called {name!r} was begun")

    def _line(self, text: str, stream: str = "stdout") -> None:
        from ..interpreter import parse_progress_line

        measured = parse_progress_line(text)
        if measured is not None:
            self._emit(Event("progress", measured))
            return
        self._emit(Event("line", {"text": text, "stream": stream}))

    def _bytes(self, done: int, total: int | None, name: str) -> None:
        now = self._monotonic()
        if done != total and now - self._last_bytes < PROGRESS_INTERVAL_SECONDS:
            return
        self._last_bytes = now
        self._emit(
            Event("progress", {"bytes_done": done, "bytes_total": total, "file": name})
        )

    def _state(self, state: wslstate.WslState) -> None:
        self._emit(
            Event(
                "state",
                {
                    "code": state.code,
                    "sentence": state.sentence,
                    "action": state.action_kind,
                },
            )
        )

    def _fail(self, code: str, message: str) -> HostError:
        self._emit(Event("failed", {"code": code, "message": message}))
        return HostError(code, message)

    def _said(self, text: str) -> None:
        self._line(text)
        if self._log is not None:
            self._log(text)

    def _restart_owed(self) -> HostError:
        if self.restarts == 0:
            self.restarts = 1
            code, sentence = outcome.REBOOT_REQUIRED_CODE, REBOOT_SENTENCE
            note = "the first restart this move asks for"
        elif not self._rebooted:
            code, sentence = (
                (outcome.REBOOT_REQUIRED_CODE, REBOOT_SENTENCE)
                if self.restarts == 1
                else (outcome.REBOOT_STILL_OWED_CODE, REBOOT_STILL_OWED_SENTENCE)
            )
            note = "Windows has not booted since it was asked, so the same ask stands"
        elif self.restarts >= RESTART_BUDGET:
            code, sentence = outcome.REBOOT_BUDGET_SPENT_CODE, REBOOT_AGAIN_SENTENCE
            note = "the budget is spent; the tray re-checks at every start"
        else:
            self.restarts += 1
            code, sentence = outcome.REBOOT_STILL_OWED_CODE, REBOOT_STILL_OWED_SENTENCE
            note = "Windows restarted and servicing still owes one"
        self._said(f"wsl: {code}: restart {self.restarts} of {RESTART_BUDGET} ({note})")
        return self._fail(code, sentence)

    def _guest_facts(self) -> tuple[str, str, str]:
        home = self._runner.run(
            wsl.guest_home_argv(self._distro),
            timeout_s=QUICK_TIMEOUT_SECONDS,
        )
        if not home.ok or home.stdout.strip() == "":
            raise self._fail(
                "step_failed",
                f'the guest would not say where its CRUCIBLE_HOME is: {home.output_tail()}',
            )
        guest_home = home.stdout.strip()
        crucible = f"{guest_home}/server/bin/crucible"
        named = self._runner.run(
            guest_argv(self._distro, ["bash", "-lc", f'"{crucible}" token --url']),
            timeout_s=QUICK_TIMEOUT_SECONDS,
        )
        name = ""
        for line in named.stdout.splitlines():
            if line.strip().startswith("crucible://"):
                from urllib.parse import unquote, urlsplit

                authority = urlsplit(line.strip()).netloc
                if "@" in authority:
                    name = unquote(authority.rsplit("@", 1)[0])
                break
        if name == "":
            raise self._fail(
                "step_failed",
                "the guest's server would not print its pairing line, so this "
                f"install cannot say what it installed: {named.output_tail()}",
            )
        return guest_home, crucible, name


    def run(self) -> InstallOutcome:
        self._wsl_state()
        self._import_distro()
        self._guest_ready()
        self._guest_install()
        self._migrate_config()
        self._install_job_types()
        prepared = self._prepare_weights()
        if self._windows is not None:
            if self._windows_after_switch is None:
                raise self._fail("migration_cleanup_unavailable", "The controller did not provide stopped-engine cleanup; Windows models are unchanged")
            record_cleanup(self._home, {row.key for row in prepared})
        self._stop_windows_server()
        self._switch_pairing()
        if self._windows is not None:
            self._windows = self._windows_after_switch()
        self._lan_door()
        self.migrate_weights()
        self._home.joinpath(CLEANUP_RECORD).unlink(missing_ok=True)
        return self.complete()

    def complete(self) -> InstallOutcome:
        guest_home, crucible, name = self._guest_facts()
        outcome = InstallOutcome(
            steps=list(self._records),
            distro=self._distro,
            detail=f'the "{self._distro}" engine answers {engine_url("/v1/ping")}',
            server_name=name,
            server_url=engine_url(),
            config_path=f"{guest_home}/config.toml",
            crucible=crucible,
            release=self._release,
        )
        self._emit(Event("done", outcome.to_dict()))
        return outcome


    def _wsl_state(self) -> None:
        self._walk("wsl-state", stop=("wsl_ready", "no_crucible_distro"))

    def _guest_ready(self) -> None:
        self._walk(
            "guest-ready",
            stop=("wsl_ready",),
            never_repair=("no_crucible_distro",),
            check_network=True,
        )

    def _walk(
        self,
        step: str,
        *,
        stop: tuple[str, ...],
        never_repair: tuple[str, ...] = (),
        **inputs: object,
    ) -> None:
        self._step(step)
        repaired: set[str] = set()
        while True:
            state = wslstate.detect(self._runner, release=self._release, **inputs)
            self._state(state)
            if state.code in stop:
                self._finish(step, state.sentence)
                return
            self._repair(state, repaired, never_repair)

    def _repair(self, state: wslstate.WslState, repaired: set[str], never_repair: tuple[str, ...]) -> None:
        if state.action_kind in ("instruct", "link"):
            raise self._fail(state.code, state.sentence + " " + state.action_text)
        if state.action_kind == "run-elevated":
            self._repair_elevated(state, repaired)
            return
        if state.code in never_repair:
            raise self._fail(
                state.code,
                f"{state.sentence} This step does not repair that — the step "
                "before it owns it, and it reported success.",
            )
        self._repair_by_command(state, repaired)

    def _repair_elevated(self, state: wslstate.WslState, repaired: set[str]) -> None:
        if state.code in repaired:
            raise self._fail(
                state.code,
                f"{state.sentence} WSL was enabled once in this run and "
                "the machine still answers the same way, so this is not "
                "something Crucible can repair here.",
            )
        before = wslstate.probe_live(self._runner)
        self._said(f"wsl: {before.line()}")
        if before.restart_owed:
            self._said(
                f"wsl: features {before.features_line()}: already on and "
                "waiting for a restart, so nothing was enabled and no "
                "administrator prompt was raised"
            )
            raise self._restart_owed()
        self._run_elevated(state, before)
        after = wslstate.probe_live(self._runner)
        self._said(
            "wsl: an administrator prompt was raised to enable WSL and "
            "accepted; " + _feature_report(before, after)
        )
        if not after.live:
            raise self._restart_owed()
        repaired.add(state.code)

    def _run_elevated(self, state: wslstate.WslState, before: wslstate.LiveWsl) -> None:
        if not self._elevate:
            raise self._fail(
                state.code,
                state.sentence
                + " This needs administrator and elevation is off for this run: "
                + " ".join(state.action_argv),
            )
        self._line(
            "Windows is asking for administrator permission to change "
            "this: click Yes on the prompt (if you cannot see it, look "
            f"for it flashing on the taskbar). [{' '.join(state.action_argv)}]"
        )
        result = self._runner.run(wslstate.elevated_argv(state), timeout_s=IMPORT_TIMEOUT_SECONDS)
        if result.ok:
            return
        self._said(
            "wsl: an administrator prompt was raised to enable WSL "
            f"and was refused, or the command failed; features "
            f"{before.features_line()}"
        )
        raise self._fail(
            state.code,
            f"{state.sentence} The permission prompt was refused or the "
            f"command failed: {result.output_tail()}",
        )

    def _repair_by_command(self, state: wslstate.WslState, repaired: set[str]) -> None:
        if state.code in repaired:
            raise self._fail(
                state.code,
                f"{state.sentence} `{' '.join(state.action_argv)}` ran and "
                "the machine still answers the same way, so this is not "
                "something Crucible can repair here.",
            )
        repaired.add(state.code)
        result = self._runner.run(list(state.action_argv), timeout_s=QUICK_TIMEOUT_SECONDS)
        self._line(f"{' '.join(state.action_argv)}: {'ok' if result.ok else result.output_tail()}")
        if not result.ok:
            raise self._fail(state.code, f"{state.sentence} {result.output_tail()}")

    def _import_distro(self) -> None:
        self._step("import-distro")
        listed = self._runner.run(wsl.list_argv(), timeout_s=QUICK_TIMEOUT_SECONDS)
        distros = wslstate.read_wsl_distros(listed)
        if distros is None:
            raise self._fail("wsl_read_failed", listed.output_tail())
        if self._distro in distros:
            self._keep_the_imported_distro()
            return
        downloads, destination = self._home / "downloads", self._home / "wsl"
        downloads.mkdir(parents=True, exist_ok=True)
        destination.mkdir(parents=True, exist_ok=True)
        self._clear_half_import(destination)
        archive = self._download_ubuntu_image(downloads)
        self._unpack_ubuntu_image(archive, destination)
        self._settle_wsl_conf()
        self._finish("import-distro", f'Imported {archive.name} as "{self._distro}"')

    def _default_user(self) -> str:
        asked = self._runner.run(wsl.whoami_argv(self._distro), timeout_s=QUICK_TIMEOUT_SECONDS)
        if not asked.ok:
            raise self._fail(
                "wsl_read_failed",
                f'"{self._distro}" would not say who it enters as: {asked.output_tail()}',
            )
        return asked.stdout.strip()

    def _settle_wsl_conf(self) -> None:
        # First the boot: WSL reads /etc/wsl.conf (systemd=true, and the default
        # user) only when a distro starts, and the finishing script booted it before
        # the file existed. Who it enters as is how that is read, so it is asked
        # BEFORE the registry is told the default user, which would answer it.
        if self._default_user() != wsl.GUEST_USER:
            self._line(
                f'Restarting "{self._distro}" so its /etc/wsl.conf takes effect: WSL reads '
                f"it only when a distro starts, and it is what turns on systemd and makes "
                f"{wsl.GUEST_USER} the user a plain `wsl -d {self._distro}` enters as"
            )
            stopped = self._runner.run(wsl.terminate_argv(self._distro), timeout_s=QUICK_TIMEOUT_SECONDS)
            if not stopped.ok:
                raise self._fail(
                    "distro_restart_failed",
                    f'"{self._distro}" would not stop to restart with its /etc/wsl.conf: '
                    f"{stopped.output_tail()}",
                )
        registered = self._register_default_user()
        user = self._default_user()
        if user != wsl.GUEST_USER:
            lever = (
                f"`wsl --manage {self._distro} --set-default-user {wsl.GUEST_USER}` ran and "
                "did not change it either"
                if registered
                else "this WSL cannot set a distro's default user itself (it has no "
                "`wsl --manage --set-default-user`; `wsl --update` brings a WSL that does)"
            )
            raise self._fail(
                "distro_default_user",
                f'"{self._distro}" enters as "{user}" after a restart, though its '
                f"/etc/wsl.conf says [user] default={wsl.GUEST_USER}, and {lever}. The engine "
                f"lives in /home/{wsl.GUEST_USER}, so a hand command typed at that distro would "
                f"look for it in the wrong home. Read the file with: wsl -d {self._distro} "
                f"-u root --exec cat /etc/wsl.conf  then {TRY_AGAIN_HINT}.",
            )

    def _register_default_user(self) -> bool:
        """Name the default user in WSL's own registration of the distro, where WSL can.

        `wsl --manage <distro> --set-default-user` writes the distro's DefaultUid, which
        is what WSL itself reads for a session that names no user; /etc/wsl.conf is read
        only inside a booting distro. On a WSL without the option, wsl.conf is the only
        thing naming the user, and the verification after this is what stands behind it.
        Whether the option exists is asked of wsl.exe's own usage text, never inferred."""
        helped = self._runner.run(wsl.help_argv(), timeout_s=QUICK_TIMEOUT_SECONDS)
        supported = wslstate.sets_default_user(helped)
        if supported is None:
            raise self._fail(
                "wsl_read_failed",
                f"`wsl --help` gave no usage text this build can read, so it is not known "
                f'whether this WSL can set the default user of "{self._distro}": '
                f"{helped.output_tail()}",
            )
        if not supported:
            self._line(
                f"This WSL has no `wsl --manage <distro> --set-default-user` (older WSL "
                f"releases do not; `wsl --update` brings it), so /etc/wsl.conf's [user] "
                f"default={wsl.GUEST_USER} is the only thing naming the user "
                f'"{self._distro}" enters as'
            )
            return False
        argv = wsl.set_default_user_argv(self._distro, wsl.GUEST_USER)
        set_user = self._runner.run(argv, timeout_s=QUICK_TIMEOUT_SECONDS)
        if not set_user.ok:
            raise self._fail(
                "distro_default_user",
                f"`{' '.join(argv)}` failed: {set_user.output_tail()}. Without it a plain "
                f'`wsl -d {self._distro}` may enter as someone other than {wsl.GUEST_USER}, '
                f"whose home the engine is in. Repair what wsl.exe said, then {TRY_AGAIN_HINT}.",
            )
        self._line(f'"{self._distro}" enters as {wsl.GUEST_USER}: `{" ".join(argv)}`')
        return True

    def _keep_the_imported_distro(self) -> None:
        marked = self._runner.run(
            guest_argv(self._distro, ["cat", "/etc/wsl.conf"]),
            timeout_s=QUICK_TIMEOUT_SECONDS,
        )
        if not marked.ok or WSL_CONF_MARKER not in marked.stdout:
            raise self._fail(
                "distro_unmarked",
                f'A WSL distro called "{self._distro}" already exists on this PC, '
                f"but Crucible did not make it: its /etc/wsl.conf has no "
                f'"{WSL_CONF_MARKER}" line. It was left untouched. If it is not '
                f"one you want to keep, remove it from PowerShell with: wsl "
                f"--unregister {self._distro}  (WARNING: that deletes every file "
                f"inside that distro), then {TRY_AGAIN_HINT}.",
            )
        self._line(f'"{self._distro}" is already imported')
        self._settle_wsl_conf()
        self._finish("import-distro", f'"{self._distro}" was already there')

    def _download_ubuntu_image(self, downloads: Path) -> Path:
        from ..platform.wsl_table import UBUNTU_WSL_ROOTFS, UBUNTU_WSL_ROOTFS_URL

        asset = UBUNTU_WSL_ROOTFS
        archive = downloads / asset
        self._line(f"Downloading Ubuntu's own WSL image ({asset})")
        fetched = self._runner.download(
            UBUNTU_WSL_ROOTFS_URL,
            archive,
            timeout_s=IMAGE_DOWNLOAD_TIMEOUT_SECONDS,
            on_progress=self._bytes,
            attempts=IMAGE_DOWNLOAD_ATTEMPTS,
        )
        self._verify_ubuntu_image(archive, fetched)
        return archive

    def _verify_ubuntu_image(self, archive: Path, fetched: RunResult) -> None:
        from ..platform.wsl_table import UBUNTU_WSL_SUMS_URL

        asset = archive.name
        digest = self._runner.run(["curl.exe", "-fsSL", "--retry", "3", UBUNTU_WSL_SUMS_URL], timeout_s=300) if fetched.ok else fetched
        if not fetched.ok or not digest.ok:
            raise self._fail("rootfs_download_failed", f"Ubuntu's WSL image or its SHA256SUMS could not be downloaded: {digest.output_tail()}")
        rows = [line.split() for line in digest.stdout.splitlines() if line.strip()]
        want = next(
            (row[0].lower() for row in rows if len(row) >= 2 and row[1].lstrip("*").strip() == asset),
            "",
        )
        hashed = self._runner.run(["certutil", "-hashfile", str(archive), "SHA256"], timeout_s=300)
        candidates = [line.replace(" ", "").strip().lower() for line in hashed.stdout.splitlines()]
        if not re.fullmatch(r"[0-9a-f]{64}", want):
            raise self._fail("rootfs_sha_mismatch", f"{UBUNTU_WSL_SUMS_URL} names no sha256 for {asset}; no distro was imported")
        if not hashed.ok or want not in candidates:
            raise self._fail("rootfs_sha_mismatch", "The downloaded image does not match Ubuntu's own checksum; no distro was imported")

    def _say_free_space(self, destination: Path) -> None:
        try:
            free = shutil.disk_usage(destination).free
        except OSError:
            return
        self._line(
            f"This PC's drive {destination.anchor or destination} has "
            f"{gib_text(free, 0)} free. The Linux engine's disk lives "
            "there and grows into it, so that is the real limit on what it can hold."
        )

    def _unpack_ubuntu_image(self, archive: Path, destination: Path) -> None:
        from ..platform.wsl_table import FINISH_IMPORT_SCRIPT

        self._line(f"Unpacking Ubuntu into {destination}; this takes about half a minute")
        self._say_free_space(destination)
        imported = self._runner.run(wsl.import_argv(self._distro, str(destination), str(archive)), timeout_s=IMPORT_TIMEOUT_SECONDS)
        if not imported.ok:
            raise self._fail("distro_import_failed", imported.output_tail())
        self._line("Preparing the Linux system for Crucible (its user and settings)")
        finished = self._runner.run(
            wsl.root_argv(self._distro, ["bash", "-c", FINISH_IMPORT_SCRIPT]),
            timeout_s=QUICK_TIMEOUT_SECONDS,
        )
        if not finished.ok:
            raise self._fail("distro_import_failed", f"The imported image could not be prepared: {finished.output_tail()}")
        marked = self._runner.run(guest_argv(self._distro, ["cat", "/etc/wsl.conf"]), timeout_s=300)
        if not marked.ok or WSL_CONF_MARKER not in marked.stdout:
            raise self._fail(
                "distro_import_invalid",
                f'The imported image did not contain its ownership marker; "{self._distro}" was kept '
                f"so nothing is lost. Remove it from PowerShell with: wsl --unregister {self._distro}  "
                f"(that deletes every file inside it), then {TRY_AGAIN_HINT}.",
            )

    def _clear_half_import(self, destination: Path) -> None:
        left = sorted(entry.name for entry in destination.iterdir())
        if not left:
            return
        remove_it = f'Remove-Item -Recurse -Force "{destination}"'
        if not set(name.lower() for name in left) <= IMPORT_ARTEFACTS:
            raise self._fail(
                "distro_import_incomplete",
                f'{destination} holds {left} and no "{self._distro}" distro is '
                "registered, so an earlier import did not finish; the files were "
                "kept because not all of them are ones an import writes. Remove the "
                f"directory from PowerShell with: {remove_it}  then "
                f"{TRY_AGAIN_HINT}.",
            )
        self._said(
            f"wsl: an earlier import of \"{self._distro}\" was interrupted and left "
            f"{left} in {destination} with no distro registered; removing them and "
            "importing again"
        )
        try:
            shutil.rmtree(destination)
            destination.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise self._fail(
                "distro_import_incomplete",
                f"{destination} holds {left} from an interrupted import and could "
                f"not be cleared ({exc}). Remove it from PowerShell with: "
                f"{remove_it}  then {TRY_AGAIN_HINT}.",
            )


    def guest_release(self) -> str | None:
        read = self._runner.run(
            wsl.guest_file_argv(self._distro, "installation.json"),
            timeout_s=QUICK_TIMEOUT_SECONDS,
        )
        if not read.ok or read.stdout.strip() == "":
            return None
        try:
            record = json.loads(read.stdout)
        except json.JSONDecodeError:
            return None
        release = record.get("release")
        return release if isinstance(release, str) and release else None

    def upgrade_guest(self) -> str | None:
        from ..platform.errors import LocalError
        from ..platform.installation import release_order

        theirs = self.guest_release()
        if theirs is not None:
            try:
                order = release_order(theirs, self._release)
            except LocalError as exc:
                raise self._fail(
                    "guest_release_unreadable",
                    f'the "{self._distro}" guest records a release this build cannot '
                    f"order against its own ({exc}). It was left alone rather than "
                    "carried: a version nothing can compare is not a version anything "
                    "should act on.",
                ) from exc
            if order == 0:
                return None
            if order > 0:
                raise self._fail(
                    "guest_ahead_of_host",
                    f'the "{self._distro}" guest is Crucible {theirs} and Crucible on '
                    f"Windows is {self._release}. It was left alone: Windows does not take "
                    "a guest backwards, and the two halves of this machine are meant to be "
                    "one release. Upgrade Crucible on Windows from PowerShell with: "
                    f"{INSTALL_ONE_LINER}",
                )
        self._guest_install()
        return self._release

    def _guest_install(self) -> None:
        self._step("guest-install")
        script = (
            'export XDG_RUNTIME_DIR="/run/user/$(id -u)"; '
            'export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"; '
            f"curl -fsSL {shlex.quote(self._install_sh_url)} -o /tmp/crucible-install.sh "
            f"&& sh /tmp/crucible-install.sh --release {shlex.quote(self._release)}"
        )
        result = self._stream_guest(["bash", "-c", script], GUEST_INSTALL_TIMEOUT_SECONDS)
        if not result.ok:
            raise self._fail(
                "step_failed",
                f"install.sh exited {result.code} inside \"{self._distro}\": {result.output_tail()}",
            )
        self._finish("guest-install", f"install.sh finished inside \"{self._distro}\"", argv=["bash", "-c", script])

    def _migrate_config(self) -> None:
        self._step("migrate-config")
        config = self._home / "config.toml"
        if not config.is_file():
            self._line(
                f"no {config}: this machine had no Windows server, so there is no "
                "token to carry over and the guest keeps the one install.sh minted",
                "stderr",
            )
            self._finish("migrate-config", "no Windows config to carry over")
            return
        self._carry_config_into_guest(config)
        self._restart_guest_engine()
        if self._guest is not None:
            self._wait_for_guest_catalog(self._guest)
        self._finish("migrate-config", "the Windows token, routes and upstreams are the guest's now")

    def _carry_config_into_guest(self, config: Path) -> None:
        carried = carried_config(config.read_text(encoding="utf-8"))
        remote = "/tmp/crucible-config-from.toml"
        payload = base64.b64encode(carried.encode("utf-8")).decode("ascii")
        written = self._runner.run(
            guest_argv(
                self._distro,
                ["bash", "-c", f"umask 077 && printf %s {payload} | base64 -d > {remote}"],
            ),
            timeout_s=QUICK_TIMEOUT_SECONDS,
        )
        if not written.ok:
            raise self._fail(
                "step_failed", f"could not write {remote} in the guest: {written.output_tail()}"
            )
        result = self._stream_guest(
            [
                "bash",
                "-lc",
                f'"{GUEST_CRUCIBLE}" init --force --config-from {remote}; '
                f"code=$?; rm -f {remote}; exit $code",
            ],
            QUICK_TIMEOUT_SECONDS,
        )
        if not result.ok:
            raise self._fail(
                "step_failed",
                "the Windows token could not be carried into the guest "
                f"(`crucible init --config-from` exited {result.code}): {result.output_tail()}. "
                "Every app that paired with this machine would have to pair again.",
            )

    def _restart_guest_engine(self) -> None:
        # `service restart`, never stop then start: between those two the tray's
        # watch sees a stopped unit and starts it itself. And restart is done when
        # the server ANSWERS (it waits up to RESTART_ANSWER_SECONDS, saying so),
        # not when systemctl returns, so a server that never comes back fails this
        # step by name (restart_not_answering, in the tail) instead of the next.
        restarted = self._stream_guest([
            "bash", "-c", 'export XDG_RUNTIME_DIR="/run/user/$(id -u)"; '
            'export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"; '
            f'"{GUEST_CRUCIBLE}" capability --write && '
            f'"{GUEST_CRUCIBLE}" service restart',
        ], GUEST_RESTART_TIMEOUT_SECONDS)
        if not restarted.ok:
            raise self._fail(
                "guest_restart_failed",
                f"`crucible capability --write && crucible service restart` failed "
                f"inside \"{self._distro}\": {restarted.output_tail()}",
            )

    def _wait_for_guest_catalog(self, guest: CatalogPort) -> None:
        deadline = self._monotonic() + GUEST_RESTART_BUDGET_SECONDS
        while True:
            try:
                guest.installed_subjects()
                return
            except HostError as exc:
                if self._monotonic() >= deadline:
                    raise self._guest_silent(exc)
            self._sleep(0.5)

    def _guest_silent(self, exc: HostError) -> HostError:
        code = "guest_authentication_failed" if exc.code == "unauthorized" else "guest_not_answering"
        return self._fail(
            code,
            f"The restarted guest engine did not answer its catalog "
            f"within {GUEST_RESTART_BUDGET_SECONDS:.0f} s. The last "
            f"answer was {exc.code}: {exc.message}",
        )

    def _install_job_types(self) -> None:
        self._step("install-job-types")
        self._line(
            "no job types were installed: the list comes from the coordinate "
            "records the Windows server keeps for each connected app, and "
            "this build has no door onto them yet. The apps' own coordinate step "
            "(docs/internals/host-and-platform.md, \"The Windows to WSL move\") installs what they need on first connect to the guest."
        )
        self._finish("install-job-types", "none: the coordinate records are the server's")

    def _prepare_weights(self) -> list[Subject]:
        self._step("prepare-weights")
        if self._windows is None or self._guest is None:
            self._finish("prepare-weights", "No Windows model catalog to move")
            return []
        source = [row for row in self._windows.installed_subjects() if row.kind != "engine"]
        target = {row.key for row in self._guest.installed_subjects()}
        for subject in source:
            if subject.key not in target:
                self._pull_into_guest(subject)
        installed = {row.key for row in self._guest.installed_subjects()}
        missing = [str(row) for row in source if row.key not in installed]
        if missing:
            raise self._fail("migration_destination_incomplete", "Windows models are unchanged; the guest is missing " + ", ".join(missing))
        self._finish("prepare-weights", f"Verified {len(source)} subject(s) in the guest; all Windows originals are still present")
        return source

    def migrate_weights(self, *, allow_pull: bool = True) -> None:
        self._step("migrate-weights")
        if self._windows is None or self._guest is None:
            self._line(
                "nothing to migrate: this machine had no Windows engine, so there "
                "is no catalog to move from. Whatever the apps need, the guest's "
                "own coordinate step pulls on first connect (docs/internals/host-and-platform.md, \"The Windows to WSL move\")."
            )
            self._finish("migrate-weights", "no Windows engine; nothing to move")
            return
        moved: list[str] = []
        held: dict[tuple[str, str], str] = {}
        for round_number in range(1, MIGRATE_IN_USE_ROUNDS + 1):
            source = {row.key: row for row in self._windows.installed_subjects() if row.kind != "engine"}
            if not source:
                self._finish_migration(moved)
                return
            held = self._migration_round(source, moved, allow_pull).held
            if not held:
                continue
            if round_number == MIGRATE_IN_USE_ROUNDS:
                break
            self._sleep(MIGRATE_POLL_SECONDS)
        raise self._still_held(held)

    def _finish_migration(self, moved: list[str]) -> None:
        detail = (
            f"moved {len(moved)} subject(s): {', '.join(moved)}"
            if moved
            else "the Windows engine had no installed subjects"
        )
        self._line(f"migrate-weights: {detail}")
        self._finish("migrate-weights", detail)

    def _migration_round(
        self, source: dict[tuple[str, str], Subject], moved: list[str], allow_pull: bool
    ) -> "_MigrationRound":
        assert self._guest is not None
        target = {row.key for row in self._guest.installed_subjects()}
        missing = [str(row) for key, row in source.items() if key not in target]
        if missing and not allow_pull:
            raise self._fail("migration_cleanup_destination_missing",
                             "Windows models are kept; the active guest must restore these subjects before cleanup: " + ", ".join(missing))
        this_round = _MigrationRound()
        for key in sorted(source):
            if self._retire_windows_copy(source[key], key in target, this_round):
                moved.append(str(source[key]))
        if this_round.deferred and not this_round.held and this_round.removed == 0:
            raise self._fail(
                "weights_shared",
                f"{', '.join(this_round.deferred)} could not be removed from the Windows "
                "engine because an alias still holds its weights, and no alias "
                "was removed this round to free them. The guest has every "
                "subject and nothing needs doing to keep using Crucible: the "
                f"controller retries the cleanup recorded in {self._home / CLEANUP_RECORD} "
                "on its own, and `crucible uninstall --purge-weights` on this Windows "
                "side removes the Windows copies with everything else.",
            )
        return this_round

    def _retire_windows_copy(self, subject: Subject, in_guest: bool, this_round: "_MigrationRound") -> bool:
        assert self._windows is not None
        if not in_guest:
            self._pull_into_guest(subject)
        try:
            self._windows.remove(subject)
        except CatalogRefusal as refusal:
            self._removal_refused(subject, refusal, this_round)
            return False
        this_round.removed += 1
        self._line(f"migrate-weights: {subject} is the guest's now, and gone from Windows")
        return True

    def _removal_refused(self, subject: Subject, refusal: CatalogRefusal, this_round: "_MigrationRound") -> None:
        if refusal.code == "weights_shared":
            this_round.deferred.append(str(subject))
            self._line(
                f"migrate-weights: {subject} is shared with an alias "
                "that goes first; retried on the next round"
            )
            return
        if refusal.code != "subject_in_use":
            raise self._fail(
                refusal.code,
                f"{subject} could not be removed from the Windows engine: "
                f"{refusal.message}. The guest has it; the Windows copy "
                "stays until this is answered, because a half-deleted "
                "subject is worse than a duplicated one.",
            )
        who = refusal.who or "something on the Windows engine"
        this_round.held[subject.key] = who
        self._line(
            f"migrate-weights: {subject} is held by {who} on the "
            "Windows engine; the guest already has it, so this is a "
            "wait and not a skip",
            "stderr",
        )

    def _still_held(self, held: dict[tuple[str, str], str]) -> HostError:
        names = ", ".join(f"{kind} {ident} (held by {who})" for (kind, ident), who in sorted(held.items()))
        return self._fail(
            "subject_in_use",
            f"after {MIGRATE_IN_USE_ROUNDS} attempts over "
            f"{MIGRATE_IN_USE_ROUNDS * MIGRATE_POLL_SECONDS / 60:.0f} minutes, the "
            f"Windows engine still holds {names}. The guest has its own copy of "
            "each, so nothing is lost — close whatever is named and run the move "
            f"again ({TRY_AGAIN_HINT}); it resumes from where it stopped.",
        )

    def _pull_into_guest(self, subject: Subject) -> None:
        assert self._guest is not None
        self._line(f"migrate-weights: pulling {subject} in the guest")
        self._guest.pull(subject)
        deadline = self._monotonic() + MIGRATE_PULL_TIMEOUT_SECONDS
        while True:
            self._sleep(MIGRATE_POLL_SECONDS)
            if any(row.key == subject.key for row in self._guest.installed_subjects()):
                self._line(f"migrate-weights: the guest has {subject}")
                return
            if self._monotonic() >= deadline:
                raise self._fail(
                    "subject_pull_timeout",
                    f"the guest did not report {subject} installed within "
                    f"{MIGRATE_PULL_TIMEOUT_SECONDS / 3600:.0f} h. The Windows copy "
                    "has NOT been removed; look at the guest's tasks for what "
                    "happened to the pull.",
                )

    def _lan_door(self) -> None:
        self._step("lan-door")
        if not self._lan_wanted():
            self._finish(
                "lan-door",
                "Local engine access is ready. Network sharing is optional and must be "
                "enabled explicitly; installation changes no port forwards or firewall rules.",
            )
            return
        from .. import lan

        door = lan_door.detect(self._runner, ENGINE_PORT)
        missing = [
            command
            for present, command in (
                (door.forward, lan_door.add_argv(ENGINE_PORT)),
                (door.firewall, lan_door.firewall_add_argv(ENGINE_PORT)),
            )
            if not present
        ]
        if missing and not self._elevate:
            self._finish(
                "lan-door",
                "network sharing was requested, but this install may not elevate; "
                "nothing was changed. Run `crucible lan enable` to open it.",
                argv=lan.elevated_argv(missing),
            )
            return
        if missing:
            self._line(lan_door.ELEVATION_SENTENCE)
        self._open_lan_door()

    def _lan_wanted(self) -> bool:
        from .. import lan

        try:
            return lan.read(self._home) is not None if self._share_lan is None else self._share_lan
        except CrucibleError as exc:
            aside = quarantine(self._home / lan.RECORD)
            self._said(
                f"lan-door: this machine's LAN sharing record could not be read "
                f"({exc}); it was moved to {aside}. Network sharing stays off; "
                "`crucible lan enable` turns it back on."
            )
            return False

    def _open_lan_door(self) -> None:
        from .. import lan
        from ..sharing import PairedEngine

        try:
            result = lan.enable(
                self._home, self._runner, PairedEngine(self._home, "lan"),
                port=ENGINE_PORT, adopt=True, say=self._line,
            )
        except CrucibleError as exc:
            raise self._fail(
                "lan_door_failed",
                "the engine is installed and working on this machine, but network "
                f"sharing could not be turned on: {exc}. Run `crucible lan enable` "
                "to retry; nothing else about this install is affected.",
            )
        for url in result["urls"]:
            self._line(f"other devices on this network can reach the engine at {url}")
        if result.get("next"):
            self._line(result["next"])
        self._finish("lan-door", result["detail"])

    def _stop_windows_server(self) -> None:
        self._step("stop-windows-server")
        if self._stop_windows_callback is None:
            raise self._fail("host_switch_unavailable", "The host did not provide its native-engine shutdown operation")
        self._stop_windows_callback()
        self._line(
            "The host stopped its native engine; activating the Linux engine."
        )
        self._finish("stop-windows-server", "the host stops its child when this returns")

    def _switch_pairing(self) -> None:
        self._step("switch-pairing")
        if self._switch_pairing_callback is None:
            raise self._fail("host_switch_unavailable", "The host did not provide its guest activation operation")
        self._switch_pairing_callback()
        self._line(
            "The host activated the guest and refreshed local pairing with the carried token."
        )
        self._finish("switch-pairing", "the same line, same token, same host, same port")


    def _stream_guest(self, argv: Sequence[str], timeout_s: float) -> RunResult:
        full = guest_argv(self._distro, argv)
        return self._runner.stream(
            full,
            timeout_s=timeout_s,
            on_line=lambda text, stream: self._line(
                re.sub(r"crucible://\S+", "<pairing code redacted>", text)
                if stream == "stdout"
                else text,
                stream,
            ),
        )


def carried_config(config_text: str) -> str:
    kept: list[str] = []
    section = ""
    for raw in config_text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip()
            if (
                section in ("auth", "routes", "accelerator")
                or section.startswith("upstreams")
            ):
                kept.append(line)
            continue
        if section == "auth":
            if stripped.startswith("token"):
                kept.append(line)
            continue
        if section in ("routes", "accelerator") or section.startswith("upstreams"):
            kept.append(line)
    text = "\n".join(kept).strip()
    if "token" not in text:
        raise HostError(
            "config_from_no_token",
            "the Windows config has no [auth] token, so there is nothing to carry "
            "into the guest. The point of --config-from is that every app that "
            "paired stays paired.",
        )
    return text + "\n"
