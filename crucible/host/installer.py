"""The Windows → WSL2 move — PHASE15-HOST.md 4.3 and 4.7.

ONE implementation of the sequence, and this is it. The page's engine switch
(`POST /v1/tasks {"type": "engine", "target": "wsl"}`) reaches it because the
Windows server relays to the host's door (`door.py`); `@crucible/bootstrap`'s
`install()` reaches it directly on a machine that has no server yet. Both get
the same events, because there is one sequence.

WHAT THE HOST DOES AND WHAT THE GUEST DOES
-------------------------------------------
Only the host can run `wsl.exe`, raise a UAC prompt and survive a reboot, so
the WINDOWS half — the 4c states, the import, the LAN door — is here. The
GUEST half is not: `install.sh` is generated from `sdk/bootstrap/src/steps.ts`
and is the one owner of "what installing a Crucible on a Linux machine is"
(PHASE14 4a: an app-driven install and a hand install "cannot differ"). So the
host RUNS that script inside the distro rather than restating its six steps in
Python, which would be the third copy of a list that already has two
spellings.

THE TOKEN, AND WHY THERE ARE TWO `init`s
-----------------------------------------
`install.sh` mints its own token, because on a bare Linux machine there is
nobody to inherit one from. On this path there IS: the Windows server has been
answering apps on `:7100` with a token they have already paired with, and 3.5
says that token survives the move. So after the guest's bare install the host
runs `crucible init --force --config-from <file>`, which takes exactly
`auth.token`, `[routes]` and `[upstreams]` out of a 0600 file the host wrote
and then deletes. Two inits and one token, rather than one init and an app
that silently stops being paired.

NOTHING IS DELETED BEFORE THE GUEST HAS IT
-------------------------------------------
3.5's weights rule, and `migrate-weights` below is shaped by it: pull the
guest's form of every subject the Windows engine has, and only then delete the
Windows copy. An interrupted move leaves both copies of the unfinished subject
and resumes from the catalog diff on the next attempt, which is why the step
reads the two catalogs every time rather than carrying a list across.
"""

from __future__ import annotations

import base64
import re
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from . import landoor, wslstate
from .catalog import CatalogPort, CatalogRefusal, Subject
from .errors import HostError
from .paths import engine_url
from .runner import RunResult, Runner
from .wsl_states import CRUCIBLE_DISTRO

#: The only target this door accepts. 4.7: the reverse move is not in this
#: phase, and a target nobody implemented is refused rather than ignored.
ENGINE_TARGET_WSL = "wsl"

#: 4.7's step names, in 4.7's order. The page draws these, the log carries
#: them, and `tests/test_host_installer.py` asserts the order — a sequence
#: whose order is only in prose is a sequence that gets reordered.
STEPS: tuple[str, ...] = (
    "wsl-state",
    "import-distro",
    "guest-install",
    "migrate-config",
    "install-job-types",
    "migrate-weights",
    "lan-door",
    "stop-windows-server",
    "switch-pairing",
)

#: Long enough for a `wsl --import` of a multi-gigabyte ext4 file, and for a
#: guest-side install that downloads a server pack over somebody's home line.
IMPORT_TIMEOUT_SECONDS = 30 * 60.0
GUEST_INSTALL_TIMEOUT_SECONDS = 120 * 60.0
QUICK_TIMEOUT_SECONDS = 5 * 60.0

#: How long the migration waits for ONE subject to arrive in the guest. A
#: narrator voice is a few hundred megabytes and a model is tens of gigabytes
#: over somebody's home line; the number is generous because the alternative
#: to waiting is deleting a Windows copy that has no replacement.
MIGRATE_PULL_TIMEOUT_SECONDS = 6 * 60 * 60.0

#: Between two polls of the guest's catalog, and between two rounds of the
#: migration when something was held.
MIGRATE_POLL_SECONDS = 5.0

#: How many rounds a `subject_in_use` may survive before the step fails by
#: name. BOUNDED on purpose: 3.5 says nothing is skipped, so the only two
#: honest ends are "it was removed" and "it is still held, and here is who" —
#: an unbounded wait would be a third, which is a migration that never
#: finishes and never says why. 60 x 5 s is five minutes of somebody closing
#: an app.
MIGRATE_IN_USE_ROUNDS = 60

#: The sentence 4.7 requires for the reboot states, verbatim in one place.
REBOOT_SENTENCE = (
    "reboot, then Crucible continues — this machine has to restart before "
    "Windows can start a Linux virtual machine. Crucible starts itself when "
    "you log back in and picks this up where it stopped."
)


@dataclass
class Event:
    """One line of the door's ndjson, shaped like `crucible/tasks.py`'s events.

    Same shape and not a similar one: 4.7 has the Windows server RELAY these
    under its own task id, and a relay that reshapes is a second owner of the
    shape.
    """

    event: str
    data: dict[str, object]


Emit = Callable[[Event], None]


#: The backend the GUEST is. Linux x86_64, which is what
#: `sdk/bootstrap`'s `backendFor('linux')` answers — the client compares
#: against that, so the two must be the same word and this is where it is
#: stated. It is NOT `llama-windows`: that is the machine being left.
GUEST_BACKEND = "cuda-linux"


@dataclass
class StepRecord:
    """One step, as the `done` event reports it.

    `argv` may be empty for a step that ran no command; `status` is `ok`,
    `running` or `skipped`, which is the vocabulary
    `sdk/bootstrap/src/install.ts`'s `InstallStep` already has, so the client
    maps it rather than translating it.
    """

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
    """What the sequence achieved. The `done` event's data.

    IT CARRIES A RESULT, and `crucible/tasks.py`'s `done` carries `{}`. The
    difference has a reason rather than being drift: this door has a caller
    tasks.py does not, `@crucible/bootstrap`'s `install()`, which is a library
    function that must RETURN an `InstallResult` — and on Windows it cannot go
    and read the guest's config for itself, because that `wsl.exe` door is one
    of the things this phase deletes. The page-driven caller (4.7) relays
    these events under a task id and may drop this payload; nothing on the
    page reads it.
    """

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


class EngineInstall:
    """4.7's sequence, driven. Every machine call goes through `runner`."""

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
        windows_catalog: CatalogPort | None = None,
        guest_catalog: CatalogPort | None = None,
        stop_windows_server: Callable[[], None] | None = None,
        switch_pairing: Callable[[], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._runner = runner
        self._emit = emit
        self._release = release
        self._home = Path(home)
        self._install_sh_url = install_sh_url
        self._distro = distro
        #: `False` in a test and in `--install --no-elevate`: the argv is still
        #: reported, and nothing raises a consent dialog.
        self._elevate = elevate
        #: The two servers the weights migration talks to (3.5, 3.5a). Both
        #: `None` on a machine that has no Windows server yet — the very first
        #: install — and that is a FACT the step states, not a fallback: there
        #: is nothing on this machine to move.
        self._windows = windows_catalog
        self._guest = guest_catalog
        self._stop_windows_callback = stop_windows_server
        self._switch_pairing_callback = switch_pairing
        self._monotonic = monotonic
        self._sleep = sleep
        self._index = 0
        self._records: list[StepRecord] = []

    # ---------------------------------------------------------------- events

    def _step(self, name: str) -> StepRecord:
        """Begin a step. ALWAYS before any `line` of that step.

        The client attributes each `line` to the last `step` it saw and
        refuses a line that arrives before any step, because a line with a
        made-up owner is worse than a refusal. So this is the first thing every
        `_step_name()` below does.
        """
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
        self._emit(Event("line", {"text": text, "stream": stream}))

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

    def _guest_facts(self) -> tuple[str, str, str]:
        """The guest's home, its console script, and its server's name.

        Asked of the guest rather than assembled from `/home/crucible`: the
        rootfs names that user today and `CRUCIBLE_HOME` can move the rest,
        and a path this side composed would be a second answer to a question
        the guest can be asked.
        """
        home = self._runner.run(
            ["wsl.exe", "-d", self._distro, "--exec", "bash", "-lc", 'printf %s "${CRUCIBLE_HOME:-$HOME/.crucible}"'],
            timeout_s=QUICK_TIMEOUT_SECONDS,
        )
        if not home.ok or home.stdout.strip() == "":
            raise self._fail(
                "step_failed",
                f'the guest would not say where its CRUCIBLE_HOME is: {home.said()}',
            )
        guest_home = home.stdout.strip()
        crucible = f"{guest_home}/server/bin/crucible"
        named = self._runner.run(
            ["wsl.exe", "-d", self._distro, "--exec", "bash", "-lc", f'"{crucible}" token --url'],
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
                f"install cannot say what it installed: {named.said()}",
            )
        return guest_home, crucible, name

    # ------------------------------------------------------------- the walk

    def run(self) -> InstallOutcome:
        """Walk 4.7's steps. A failure leaves the Windows engine untouched."""
        self._wsl_state()
        self._import_distro()
        self._guest_install()
        self._migrate_config()
        self._install_job_types()
        self._migrate_weights()
        self._lan_door()
        self._stop_windows_server()
        self._switch_pairing()
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

    # ------------------------------------------------------------ the steps

    def _wsl_state(self) -> None:
        """4c, answered. The rows that need admin run through UAC BY NAME."""
        self._step("wsl-state")
        while True:
            state = wslstate.detect(self._runner, release=self._release)
            self._state(state)
            if state.code in ("wsl_ready", "no_crucible_distro"):
                # Both mean "WSL itself is fine". The distro is the next step's.
                self._finish("wsl-state", state.sentence)
                return
            if state.action_kind == "instruct" or state.action_kind == "link":
                # Nothing software can do: firmware, a VPN, a disk, a hardened
                # distro. The sentence is the table's and the host adds none.
                raise self._fail(state.code, state.sentence + " " + state.action_text)
            if state.action_kind == "run-elevated":
                if not self._elevate:
                    raise self._fail(
                        state.code,
                        state.sentence
                        + " This needs administrator and elevation is off for this run: "
                        + " ".join(state.action_argv),
                    )
                self._line(f"asking for administrator: {' '.join(state.action_argv)}")
                result = self._runner.run(
                    wslstate.elevated_argv(state), timeout_s=IMPORT_TIMEOUT_SECONDS
                )
                if not result.ok:
                    raise self._fail(
                        state.code,
                        f"{state.sentence} The permission prompt was refused or the "
                        f"command failed: {result.said()}",
                    )
                # Enabling WSL always needs a restart, and there is no probe
                # that says so — `wsl --status` answers the same before and
                # after. 4.7: the task ends here and the Startup item is what
                # makes "Crucible continues" true.
                raise self._fail("wsl_reboot_required", REBOOT_SENTENCE)
            result = self._runner.run(list(state.action_argv), timeout_s=QUICK_TIMEOUT_SECONDS)
            self._line(f"{' '.join(state.action_argv)}: {'ok' if result.ok else result.said()}")
            if not result.ok:
                raise self._fail(state.code, f"{state.sentence} {result.said()}")
            # Re-detect: the table is walked until it answers a state that
            # nothing further can improve.

    def _import_distro(self) -> None:
        """The rootfs, verified, imported. Idempotent: present is a no-op."""
        self._step("import-distro")
        listed = self._runner.run(["wsl.exe", "-l", "-v"], timeout_s=QUICK_TIMEOUT_SECONDS)
        from .presence import parse_wsl_list

        if not listed.ok:
            raise self._fail("wsl_read_failed", listed.said())
        if self._distro in parse_wsl_list(listed.stdout):
            marked = self._runner.run(
                ["wsl.exe", "-d", self._distro, "--exec", "cat", "/etc/wsl.conf"],
                timeout_s=QUICK_TIMEOUT_SECONDS,
            )
            if not marked.ok or "# crucible-rootfs" not in marked.stdout:
                raise self._fail("distro_unmarked", f'The existing "{self._distro}" distro is not marked as a Crucible image; it was left untouched')
            self._line(f'"{self._distro}" is already imported')
            self._finish("import-distro", f'"{self._distro}" was already there')
            return
        from .wsl_states import ROOTFS_ASSET_TEMPLATE, RELEASE_REPOSITORY, WSL_CONF_MARKER
        if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?", self._release):
            raise self._fail("invalid_release", "The rootfs release must be a version")
        asset = ROOTFS_ASSET_TEMPLATE.replace("{version}", self._release)
        base = f"https://github.com/{RELEASE_REPOSITORY}/releases/download/v{self._release}"
        downloads, destination = self._home / "downloads", self._home / "wsl"
        downloads.mkdir(parents=True, exist_ok=True)
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise self._fail("distro_import_incomplete", f"{destination} is not empty but no distro is registered. Its files were kept for recovery")
        archive = downloads / asset
        self._line(f"Downloading the verified Crucible Linux image for {self._release}")
        fetched = self._runner.run(["curl.exe", "-fL", "--retry", "3", "-o", str(archive), f"{base}/{asset}"], timeout_s=3600)
        digest = self._runner.run(["curl.exe", "-fsSL", "--retry", "3", f"{base}/{asset}.sha256"], timeout_s=300) if fetched.ok else fetched
        if not fetched.ok or not digest.ok:
            raise self._fail("rootfs_download_failed", f"The release's rootfs or checksum could not be downloaded: {digest.said()}")
        want = digest.stdout.split()[0].lower() if digest.stdout.split() else ""
        hashed = self._runner.run(["certutil", "-hashfile", str(archive), "SHA256"], timeout_s=300)
        candidates = [line.replace(" ", "").strip().lower() for line in hashed.stdout.splitlines()]
        if not re.fullmatch(r"[0-9a-f]{64}", want) or not hashed.ok or want not in candidates:
            raise self._fail("pack_sha_mismatch", "The downloaded rootfs does not match the release checksum; no distro was imported")
        imported = self._runner.run(["wsl.exe", "--import", self._distro, str(destination), str(archive), "--version", "2"], timeout_s=IMPORT_TIMEOUT_SECONDS)
        if not imported.ok:
            raise self._fail("distro_import_failed", imported.said())
        marked = self._runner.run(["wsl.exe", "-d", self._distro, "--exec", "cat", "/etc/wsl.conf"], timeout_s=300)
        if not marked.ok or WSL_CONF_MARKER not in marked.stdout:
            raise self._fail("distro_import_invalid", "The imported image did not contain its ownership marker; it was preserved for inspection")
        self._finish("import-distro", f'Imported the verified {asset} as "{self._distro}"')

    def _guest_install(self) -> None:
        """`install.sh`, inside the distro. The guest half has ONE owner."""
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
                f"install.sh exited {result.code} inside \"{self._distro}\": {result.said()}",
            )
        self._finish("guest-install", f"install.sh finished inside \"{self._distro}\"", argv=["bash", "-c", script])

    def _migrate_config(self) -> None:
        """The token, the routes and the upstreams cross into the guest.

        The file is written into the GUEST (through `bash -c 'cat > …'` with a
        `umask 077`), not onto `/mnt/c`: a 0600 file on a DrvFs mount has no
        0600, because DrvFs synthesises permissions from the Windows ACL, and a
        token written there would be readable by every process on Windows.
        """
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
        carried = carried_config(config.read_text(encoding="utf-8"))
        remote = "/tmp/crucible-config-from.toml"
        # The runner has no stdin door, so the document goes through the
        # command line — base64 so that no quoting rule, on either side of
        # wsl.exe, can change a byte of somebody's key. `umask 077` before the
        # redirect, so the file is never briefly readable.
        payload = base64.b64encode(carried.encode("utf-8")).decode("ascii")
        written = self._runner.run(
            [
                "wsl.exe",
                "-d",
                self._distro,
                "--exec",
                "bash",
                "-c",
                f"umask 077 && printf %s {payload} | base64 -d > {remote}",
            ],
            timeout_s=QUICK_TIMEOUT_SECONDS,
        )
        if not written.ok:
            raise self._fail(
                "step_failed", f"could not write {remote} in the guest: {written.said()}"
            )
        result = self._stream_guest(
            [
                "bash",
                "-lc",
                f'"$HOME/.crucible/server/bin/crucible" init --force --config-from {remote}; '
                f"code=$?; rm -f {remote}; exit $code",
            ],
            QUICK_TIMEOUT_SECONDS,
        )
        if not result.ok:
            raise self._fail(
                "step_failed",
                "the Windows token could not be carried into the guest "
                f"(`crucible init --config-from` exited {result.code}): {result.said()}. "
                "Every app that paired with this machine would have to pair again.",
            )
        restarted = self._stream_guest([
            "bash", "-c", 'export XDG_RUNTIME_DIR="/run/user/$(id -u)"; '
            'export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"; '
            '"$HOME/.crucible/server/bin/crucible" capability --write && '
            '"$HOME/.crucible/server/bin/crucible" service stop && '
            '"$HOME/.crucible/server/bin/crucible" service start',
        ], QUICK_TIMEOUT_SECONDS)
        if not restarted.ok:
            raise self._fail("guest_restart_failed", restarted.said())
        if self._guest is not None:
            # A guest catalog request proves the migrated token is actually live.
            deadline = self._monotonic() + 30
            while True:
                try:
                    self._guest.installed_subjects()
                    break
                except HostError:
                    if self._monotonic() >= deadline:
                        raise self._fail("guest_authentication_failed", "The restarted guest did not accept the migrated token")
                    self._sleep(0.5)
        self._finish("migrate-config", "the Windows token, routes and upstreams are the guest's now")

    def _install_job_types(self) -> None:
        """4.7: the job types the connected apps' modules asked for.

        NOT GUESSED, and not built here: the coordinate records the server
        keeps from every app are the input, and reading them is the SERVER's
        (`crucible/modules.py`). Until the Windows server exposes that list the
        step installs nothing and says so — an install of "everything" would
        cost somebody thirty gigabytes nobody asked for.
        """
        self._step("install-job-types")
        self._line(
            "no job types were installed: the list comes from the coordinate "
            "records the Windows server keeps for each connected app (4.7), and "
            "this build has no door onto them yet. The apps' own coordinate step "
            "(PHASE14 4a) installs what they need on first connect to the guest."
        )
        self._finish("install-job-types", "none: the coordinate records are the server's")

    def _migrate_weights(self) -> None:
        """3.5 and 3.5a: pull in the guest, then delete on Windows. Never the reverse.

        THE ORDER IS THE WHOLE RULE. For each subject the Windows catalog
        reports installed: submit the guest's pull, WAIT until the guest's own
        catalog says it is installed there, and only then
        `DELETE /v1/catalog/{kind}/{id}` on the Windows server. A machine
        unplugged at any instant has the subject on one side or on both, never
        on neither.

        IDEMPOTENT BY RE-DIFFING, not by a journal. Every round re-reads BOTH
        catalogs and acts on the difference, so a resume after a crash, a
        reboot or a `Ctrl-C` needs no state that survived the crash — which is
        the only kind of resume that is true after a power cut.

        `subject_in_use` (3.5a) is WAITED OUT, never skipped. Something holds
        the subject — a lease, a resident model, a running task — and 3.5 says
        nothing is skipped, so the subject is retried on the next round with
        its holder named in the meantime, for a bounded number of rounds, and
        then the step fails BY THAT NAME. The two honest ends are "removed"
        and "still held, and here is who".

        The host never touches a file: every read, pull and delete is a
        request to the server that owns that disk (3.5a's reason for existing).
        """
        self._step("migrate-weights")
        if self._windows is None or self._guest is None:
            self._line(
                "nothing to migrate: this machine had no Windows engine, so there "
                "is no catalog to move from. Whatever the apps need, the guest's "
                "own coordinate step pulls on first connect (PHASE14 4a)."
            )
            self._finish("migrate-weights", "no Windows engine; nothing to move")
            return

        moved: list[str] = []
        held: dict[tuple[str, str], str] = {}
        for round_number in range(1, MIGRATE_IN_USE_ROUNDS + 1):
            source = {row.key: row for row in self._windows.installed_subjects()}
            if not source:
                detail = (
                    f"moved {len(moved)} subject(s): {', '.join(moved)}"
                    if moved
                    else "the Windows engine had no installed subjects"
                )
                self._line(f"migrate-weights: {detail}")
                self._finish("migrate-weights", detail)
                return
            target = {row.key for row in self._guest.installed_subjects()}
            held = {}
            for key in sorted(source):
                subject = source[key]
                if key not in target:
                    self._pull_into_guest(subject)
                try:
                    self._windows.remove(subject)
                except CatalogRefusal as refusal:
                    if refusal.code != "subject_in_use":
                        raise self._fail(
                            refusal.code,
                            f"{subject} could not be removed from the Windows engine: "
                            f"{refusal.message}. The guest has it; the Windows copy "
                            "stays until this is answered, because a half-deleted "
                            "subject is worse than a duplicated one.",
                        )
                    who = refusal.who or "something on the Windows engine"
                    held[key] = who
                    self._line(
                        f"migrate-weights: {subject} is held by {who} on the "
                        "Windows engine; the guest already has it, so this is a "
                        "wait and not a skip",
                        "stderr",
                    )
                    continue
                moved.append(str(subject))
                self._line(f"migrate-weights: {subject} is the guest's now, and gone from Windows")
            if not held:
                # Everything this round either moved or was already gone. The
                # next round re-reads and finds the catalog empty, which is
                # the one place this loop returns from.
                continue
            if round_number == MIGRATE_IN_USE_ROUNDS:
                break
            self._sleep(MIGRATE_POLL_SECONDS)

        names = ", ".join(f"{kind} {ident} (held by {who})" for (kind, ident), who in sorted(held.items()))
        raise self._fail(
            "subject_in_use",
            f"after {MIGRATE_IN_USE_ROUNDS} attempts over "
            f"{MIGRATE_IN_USE_ROUNDS * MIGRATE_POLL_SECONDS / 60:.0f} minutes, the "
            f"Windows engine still holds {names}. The guest has its own copy of "
            "each, so nothing is lost — close whatever is named and run the move "
            "again; it resumes from where it stopped.",
        )

    def _pull_into_guest(self, subject: Subject) -> None:
        """Submit the guest's pull and WAIT for the guest's catalog to say so.

        The task's own events are not read: the catalog is the fact
        (`installed`), a task is a report about it, and the thing that gates a
        deletion has to be the fact.
        """
        assert self._guest is not None  # only called from the step, which checked
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
        """Installing an engine is not consent to expose it on every interface."""
        self._step("lan-door")
        self._finish(
            "lan-door",
            "Local engine access is ready. Network sharing is optional and must be "
            "enabled explicitly; installation changes no port forwards or firewall rules.",
        )

    def _stop_windows_server(self) -> None:
        """4.7: the Windows engine stops only after the guest is serving."""
        self._step("stop-windows-server")
        if self._stop_windows_callback is None:
            raise self._fail("host_switch_unavailable", "The host did not provide its native-engine shutdown operation")
        self._stop_windows_callback()
        self._line(
            "The host stopped its native engine; activating the Linux engine."
        )
        self._finish("stop-windows-server", "the host stops its child when this returns")

    def _switch_pairing(self) -> None:
        """3.6: the Windows-side pairing file now names the guest's server."""
        self._step("switch-pairing")
        if self._switch_pairing_callback is None:
            raise self._fail("host_switch_unavailable", "The host did not provide its guest activation operation")
        self._switch_pairing_callback()
        self._line(
            "The host activated the guest and refreshed local pairing with the carried token."
        )
        self._finish("switch-pairing", "the same line, same token, same host, same port")

    # ------------------------------------------------------------- plumbing

    def _stream_guest(self, argv: Sequence[str], timeout_s: float) -> RunResult:
        """Run inside the distro and put every line on the event stream.

        `--exec`, always: wsl.exe pre-expands `$var` in its implicit-shell form
        and `--exec` is the spelling everything else in this system uses.
        """
        full = ["wsl.exe", "-d", self._distro, "--exec", *argv]
        result = self._runner.run(full, timeout_s=timeout_s)
        for line in result.stdout.splitlines():
            self._line(re.sub(r"crucible://\S+", "<pairing code redacted>", line))
        for line in result.stderr.splitlines():
            self._line(line, "stderr")
        return result


def elevated(argv: Sequence[str]) -> list[str]:
    """`Start-Process -Verb RunAs` around an argv. One spelling, two callers."""
    program, *rest = argv
    quoted = ",".join("'" + word.replace("'", "''") + "'" for word in rest)
    arguments = "" if quoted == "" else f" -ArgumentList {quoted}"
    return [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        f"Start-Process -Verb RunAs -Wait -FilePath '{program}'{arguments}",
    ]


def carried_config(config_text: str) -> str:
    """The THREE things `--config-from` takes, and nothing else (4.3).

    `auth.token`, `[routes]`, `[upstreams]`. The host, the port, the name, the
    backend and the job flags belong to the machine being initialised, not to
    the one being left — a guest that inherited `backend = "llama-windows"`
    would refuse to serve on its own card.

    Extracted textually rather than parsed and re-emitted, because a key is a
    secret and a round trip through a writer is a chance to mangle one. The
    sections are copied verbatim; anything outside them is dropped.
    """
    kept: list[str] = []
    section = ""
    for raw in config_text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip()
            if section == "auth" or section == "routes" or section.startswith("upstreams"):
                kept.append(line)
            continue
        if section == "auth":
            if stripped.startswith("token"):
                kept.append(line)
            continue
        if section == "routes" or section.startswith("upstreams"):
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
