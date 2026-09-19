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

MODEL RETIREMENT FOLLOWS VERIFIED ACTIVATION
-------------------------------------------
Prepare every guest model while retaining every native original, then stop the
native process and verify guest ownership/pairing. Retire native model files
through their catalog owner functions only afterward. A persistent subject-key
record resumes interrupted cleanup, even if a partial deletion removed its
installation stamp. The Windows executable is retained; it is not a model or
a portable Linux engine. Temporary copies during migration preserve recovery;
successful cleanup leaves only the active backend's model files.
"""

from __future__ import annotations

import base64
import json
import re
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from . import landoor, wslstate
from .catalog import CatalogPort, CatalogRefusal, Subject
from .errors import HostError
from ..errors import CrucibleError
from .paths import ENGINE_PORT, engine_url
from .runner import RunResult, Runner
from .wsl_states import CRUCIBLE_DISTRO

#: The only target this door accepts. 4.7: the reverse move is not in this
#: phase, and a target nobody implemented is refused rather than ignored.
ENGINE_TARGET_WSL = "wsl"
CLEANUP_RECORD = "migration-cleanup.json"

#: THE MARKER IS GONE. `wsl-reboot-pending` used to be written here and read
#: nowhere else; PHASE19 2.2 replaces it with `wsl-outcome.json`, which records
#: the reboot as one of five endings instead of being a file whose only meaning
#: was its own existence. `crucible/host/outcome.py` is its one owner, and
#: `app._sequence` is what writes it at every terminal point of a move — this
#: class raises, as it always did, and the code it raises is what chooses the
#: state (`outcome.classify`).


def cleanup_subjects(home: Path) -> set[tuple[str, str]]:
    """Read only named catalog subjects; never accept filesystem paths."""
    value = json.loads((home / CLEANUP_RECORD).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("subjects"), list):
        raise HostError("migration_cleanup_record_invalid", "The migration cleanup record is incompatible")
    result = set()
    for row in value["subjects"]:
        if (not isinstance(row, list) or len(row) != 2
                or not all(isinstance(part, str) and part for part in row) or row[0] == "engine"):
            raise HostError("migration_cleanup_record_invalid", "The migration cleanup record contains an invalid model subject")
        result.add(tuple(row))
    return result


def record_cleanup(home: Path, subjects: set[tuple[str, str]]) -> None:
    home.mkdir(parents=True, exist_ok=True)
    record = home / CLEANUP_RECORD
    staged = record.with_suffix(".tmp")
    staged.write_text(json.dumps({"schema_version": 1, "subjects": [list(row) for row in sorted(subjects)]}) + "\n", encoding="utf-8")
    staged.replace(record)

#: 4.7's step names, in 4.7's order. The page draws these, the log carries
#: them, and `tests/test_host_installer.py` asserts the order — a sequence
#: whose order is only in prose is a sequence that gets reordered.
STEPS: tuple[str, ...] = (
    "wsl-state",
    "import-distro",
    # 4c's guest rows, asked once there IS a guest for them to be about. See
    # `_guest_ready`: the first walk stops before the distro exists, so without
    # this one nothing ever asks the distro anything.
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

#: Long enough for a `wsl --import` of a multi-gigabyte ext4 file, and for a
#: guest-side install that pips a job type's recipe over somebody's home line.
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
#:
#: REWRITTEN BY PHASE19 2.3. It used to say "press Install once more and it will
#: go on from here", because nothing on the machine resumed by itself: the tray
#: came back at login and the APP had to ask again. The tray now decides at
#: every start (2.3) and resumes a `reboot-pending` on its own, so the sentence
#: no longer asks for a press that nothing is waiting for.
REBOOT_SENTENCE = (
    "this machine has to restart before Windows can start a Linux virtual "
    "machine. Nothing downloaded so far is lost: Crucible comes back by itself "
    "when you log in and goes on from here."
)

#: 2.4's second demand. `wsl --install` ran, the machine restarted, and
#: `wsl --status` asks for a restart again — which is not a state anything can
#: repair and not one to loop on.
REBOOT_AGAIN_SENTENCE = (
    "Windows asked for a restart twice. `wsl --install` has already run and "
    "this machine has already been restarted, and Windows still says it needs "
    "another one before it can start a Linux virtual machine — so Crucible has "
    "stopped rather than asking again. The Windows engine keeps working; this "
    "is a machine somebody has to look at."
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
        resuming: bool = False,
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
        #: WHICH GUEST. The default is the distro Crucible IMPORTS, and it is
        #: the right one for the walk this class was written for: a first
        #: install has no consented distro to ask about, because `_import_distro`
        #: is the step that brings the only one into existence.
        #:
        #: It is NOT the right one for a machine that already has an engine in a
        #: distro a person named in config.toml. Every such caller passes the
        #: name the watcher holds — `PresenceWatcher.distro`, the one owner of
        #: that fact — and 1.0.4 is what happens when one of them forgets to
        #: (`app.carry_guest_to_this_release`, 2026-09-19).
        self._distro = distro
        #: `False` in a test and in `--install --no-elevate`: the argv is still
        #: reported, and nothing raises a consent dialog.
        self._elevate = elevate
        #: PHASE19 2.4: is this run the one AFTER the reboot Windows demanded?
        #:
        #: Resume is "run the sequence from the top", because every step is
        #: already idempotent — so the only thing this changes is what a SECOND
        #: reboot demand means. The first is a machine doing what Windows asked;
        #: the second, on a machine that has already restarted, is a state
        #: nothing here can repair, and asking for a third restart would be a
        #: loop with a person in it.
        self._resuming = resuming
        #: Whether this install should open the LAN door (`crucible lan`).
        #:
        #: THREE STATES, and `None` is the useful one. `True`/`False` is an
        #: operator saying so for this install. `None` means *follow what this
        #: machine already decided* — the `landoor.json` record — so a machine
        #: whose door was opened once keeps it open across every later
        #: reinstall and upgrade, and a machine that never opened one is never
        #: silently exposed by an upgrade.
        #:
        #: The RECORD is the preference. A second setting saying the same thing
        #: is a second thing to disagree with it, which is the shape of defect
        #: `docs/ARCHITECTURE.md` was written about.
        self._share_lan = share_lan
        #: The two servers the weights migration talks to (3.5, 3.5a). Both
        #: `None` on a machine that has no Windows server yet — the very first
        #: install — and that is a FACT the step states, not a fallback: there
        #: is nothing on this machine to move.
        self._windows = windows_catalog
        self._guest = guest_catalog
        self._stop_windows_callback = stop_windows_server
        self._switch_pairing_callback = switch_pairing
        self._windows_after_switch = windows_after_switch
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
        """Prepare, activate, then retire; failed preparation preserves native models."""
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
        # AFTER the switch: the door publishes this machine's addresses into
        # the guest engine, which only answers once `_switch_pairing` has
        # pointed this machine at it. Before `_migrate_weights`, because that
        # step can run for hours and a consent prompt raised at its far end is
        # a prompt nobody is sitting in front of.
        self._lan_door()
        self._migrate_weights()
        self._home.joinpath(CLEANUP_RECORD).unlink(missing_ok=True)
        return self._complete()

    def _complete(self) -> InstallOutcome:
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
        self._walk("wsl-state", stop=("wsl_ready", "no_crucible_distro"))

    def _guest_ready(self) -> None:
        """4c AGAIN, once there is a distro for its guest rows to be about.

        `_wsl_state` runs before the import and stops the moment it can say
        "WSL itself is fine", so every row whose probe runs INSIDE the distro —
        `distro_not_systemd`, `guest_no_network`, `pack_disk`,
        `guest_root_unreachable` — was skipped on the one path that creates the
        distro. Measured 2026-09-16 on a fabricated fresh machine: the walk ran
        `--status`, `-l -v`, `-l -v` and then downloaded the rootfs, and asked
        the guest nothing.

        `guest_root_unreachable` is the row that now matters most. Since the guest's
        server became a SYSTEM unit (`crucible/service.py`), the install writes
        `/etc/systemd/system` and drives the system manager, both through
        `wsl.exe -u root` — so a distro that will not grant root cannot be
        installed into at all, and finding that out here costs one `id -u`
        instead of a failed install.

        `check_network` is passed for the first time by anybody. This is the
        caller `wsl-states.ts` describes: "the only caller that needs this row
        is one that is about to download gigabytes." It costs one `curl` in the
        guest and turns a VPN into a sentence instead of a failed download.

        `required_bytes` is deliberately NOT passed, and `pack_disk` therefore
        still never fires. SINCE PHASE20 THERE IS NOTHING HERE TO PRICE: what
        goes into the guest is a ~30 MB interpreter and a 1 MB wheel with its
        PyPI dependencies, and a disk guard for a hundred megabytes is a
        sentence nobody needs. The row stays in the table for a caller that
        knows a bigger number — one about to pull weights, or `crucible install
        tts`, which is where the gigabytes actually are.
        """
        self._walk(
            "guest-ready",
            stop=("wsl_ready",),
            # `_import_distro` has just run and said it succeeded. If the table
            # still cannot see the distro, importing it AGAIN is not a repair —
            # it is this step doing the previous step's job on a machine where
            # that job did not take. Two owners of one import, and the second
            # one loops.
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
        """Detect, repair, detect again — until a state nothing can improve.

        **A REPAIR THAT DID NOT CHANGE THE ANSWER ENDS THE WALK.** Without this
        the loop runs the same action forever: `distro_not_systemd` answers
        again, `wsl --terminate crucible` exits 0 again, nothing is different
        and nothing says so. Measured 2026-09-16 — `pytest tests/test_host.py`
        sat for forty minutes on exactly that, printing nothing, and a stranger
        whose distro will not take systemd would have watched an installer hang
        with no message at all. An action is allowed one attempt: a code that
        comes back after its own repair ran is a machine this cannot fix, and
        saying so is the whole point of the table.
        """
        self._step(step)
        repaired: set[str] = set()
        while True:
            state = wslstate.detect(self._runner, release=self._release, **inputs)  # type: ignore[arg-type]
            self._state(state)
            if state.code in stop:
                self._finish(step, state.sentence)
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
                # after. The task ends here, the tray's Startup item brings the
                # tray back, and PHASE19 2.3 is what brings the INSTALL back:
                # the tray reads `reboot-pending` out of `wsl-outcome.json` at
                # its next start and resumes. That used to be the app's job and
                # the sentence used to ask for a press; 2.3 ruled it the tray's,
                # because the tray is the process that is already there.
                #
                # A SECOND DEMAND IS NOT A SECOND RESTART (2.4). This run is
                # already the one after the reboot, and Windows asking again is
                # a machine a person has to look at rather than a loop.
                if self._resuming:
                    raise self._fail("wsl_reboot_again", REBOOT_AGAIN_SENTENCE)
                raise self._fail("wsl_reboot_required", REBOOT_SENTENCE)
            if state.code in never_repair:
                raise self._fail(
                    state.code,
                    f"{state.sentence} This step does not repair that — the step "
                    "before it owns it, and it reported success.",
                )
            if state.code in repaired:
                raise self._fail(
                    state.code,
                    f"{state.sentence} `{' '.join(state.action_argv)}` ran and "
                    "the machine still answers the same way, so this is not "
                    "something Crucible can repair here.",
                )
            repaired.add(state.code)
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
        # CANONICAL'S IMAGE, AND CANONICAL'S OWN DIGEST (PHASE20 section 2).
        # It used to be `crucible-rootfs-<version>.tar.zst` off our own release,
        # rebuilt by a Docker job on every tag for an image whose contents had
        # not moved in a month. Every constant below is GENERATED from
        # `sdk/bootstrap/src/distro.ts`, which is the one owner: this side and
        # `@crucible/bootstrap`'s `ensureDistro()` import the same bytes and
        # finish the import the same way, or they are two installs again.
        from .wsl_states import (
            FINISH_IMPORT_SCRIPT,
            UBUNTU_WSL_ROOTFS,
            UBUNTU_WSL_ROOTFS_URL,
            UBUNTU_WSL_SUMS_URL,
            WSL_CONF_MARKER,
        )
        asset = UBUNTU_WSL_ROOTFS
        downloads, destination = self._home / "downloads", self._home / "wsl"
        downloads.mkdir(parents=True, exist_ok=True)
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise self._fail("distro_import_incomplete", f"{destination} is not empty but no distro is registered. Its files were kept for recovery")
        archive = downloads / asset
        self._line(f"Downloading Ubuntu's own WSL image ({asset})")
        fetched = self._runner.run(["curl.exe", "-fL", "--retry", "3", "-o", str(archive), UBUNTU_WSL_ROOTFS_URL], timeout_s=3600)
        digest = self._runner.run(["curl.exe", "-fsSL", "--retry", "3", UBUNTU_WSL_SUMS_URL], timeout_s=300) if fetched.ok else fetched
        if not fetched.ok or not digest.ok:
            raise self._fail("rootfs_download_failed", f"Ubuntu's WSL image or its SHA256SUMS could not be downloaded: {digest.said()}")
        # THE ROW FOR OUR FILE, by name. The sums file lists every image in
        # that directory, and `current/` is a moving pointer — so a sums file
        # that does not name this download is exactly what upstream renaming
        # the file looks like, and it must refuse rather than compare nothing.
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
        imported = self._runner.run(["wsl.exe", "--import", self._distro, str(destination), str(archive), "--version", "2"], timeout_s=IMPORT_TIMEOUT_SECONDS)
        if not imported.ok:
            raise self._fail("distro_import_failed", imported.said())
        # WHAT `build-rootfs.sh` USED TO BAKE, done here instead: the crucible
        # user, passwordless sudo, and the `/etc/wsl.conf` whose first line is
        # the marker every later check looks for. Canonical's image has none of
        # them, and it is ours — just imported under our name into our
        # directory — so writing them is finishing the import.
        finished = self._runner.run(
            ["wsl.exe", "-d", self._distro, "-u", "root", "--exec", "bash", "-c", FINISH_IMPORT_SCRIPT],
            timeout_s=QUICK_TIMEOUT_SECONDS,
        )
        if not finished.ok:
            raise self._fail("distro_import_failed", f"The imported image could not be prepared: {finished.said()}")
        marked = self._runner.run(["wsl.exe", "-d", self._distro, "--exec", "cat", "/etc/wsl.conf"], timeout_s=300)
        if not marked.ok or WSL_CONF_MARKER not in marked.stdout:
            raise self._fail("distro_import_invalid", "The imported image did not contain its ownership marker; it was preserved for inspection")
        self._finish("import-distro", f'Imported {asset} as "{self._distro}"')

    # --------------------------------------------- one release per machine

    def guest_release(self) -> str | None:
        """What `installation.json` inside the distro says the guest is.

        None when there is no record to read. That is not the same as "up to
        date": `crucible/local.py`'s `publish_installation` writes the file
        when the RUNTIME STARTS, so an absent one means nothing has run in
        there — a guest to bring up to this release rather than one to leave at
        a version nobody can name.
        """
        read = self._runner.run(
            [
                "wsl.exe", "-d", self._distro, "--exec", "bash", "-lc",
                'cat "${CRUCIBLE_HOME:-$HOME/.crucible}/installation.json"',
            ],
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
        """Carry the guest to THIS host's release. Returns it, or None.

        ONE RELEASE PER MACHINE, AND THE HOST IS THE DRIVER (Owen, 2026-09-18:
        *"windows is the driver; the thing moving wsl forward."*).

        THE DEFECT THIS EXISTS FOR, measured rather than remembered.
        `crucible/host/app.py` handed the install sequence to the door and
        nowhere else, so the walk ran on `POST /install`; and on a machine the
        guest already owns, that sequence called `_complete()` — which emits a
        `done` describing the engine that is already there — instead of
        installing anything. `_guest_install` below, the ONE place
        `install.sh --release` runs inside the distro, is reached only from
        `run()`. So `install.ps1` upgraded the Windows half and the guest sat
        at whatever release it was installed at, which is why `deploy.sh` had
        grown a second driver for the same machine.

        THREE ANSWERS, AND ONLY ONE OF THEM DOES ANYTHING:

          * the guest names an OLDER release, or names none → it is carried, by
            the same `install.sh --release <this host's version>` the move runs.
            That script is generated from `sdk/bootstrap/src/steps.ts`, so what
            the guest gets is the pinned interpreter (skipped when its digest is
            already stamped), this release's wheel, and then the service,
            capability and readiness steps it already ends with — which is why
            nothing here repeats them.
          * the guest names THIS release → nothing. Re-running an install that
            has nothing to do is a minute of somebody's startup for no change.
          * the guest names a NEWER one → `guest_ahead_of_host`, by name, and
            the guest is left exactly as it is. A host that silently took a
            guest BACKWARDS would be the never-older rule the installers
            themselves refuse (INSTALL-UNINSTALL.md 6.5.4), broken by the one
            process that is supposed to enforce it.
        """
        from ..local import LocalError, release_order

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
                    f'the "{self._distro}" guest is Crucible {theirs} and this host is '
                    f"{self._release}. It was left alone: a host does not take a guest "
                    "backwards, and the two halves of this machine are meant to be one "
                    "release. Upgrade the host, or uninstall the guest and let this "
                    "install it.",
                )
        self._guest_install()
        return self._release

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

    def _prepare_weights(self) -> list[Subject]:
        """Prepare every destination before stopping or deleting any source."""
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

    def _migrate_weights(self, *, allow_pull: bool = True) -> None:
        """Retire native models after the guest owns the endpoint.

        THE ORDER IS THE WHOLE RULE. For each subject the Windows catalog
        reports installed: submit the guest's pull, WAIT until the guest's own
        catalog says it is installed there, and only then
        the stopped native catalog's removal operation. A machine
        unplugged at any instant has the subject on one side or on both, never
        on neither.

        Every round re-reads both catalogs. The persistent cleanup record also
        names incomplete deletions whose installation stamp disappeared. That
        lets the stopped native adapter finish removing their remaining files.
        Background retries refuse missing destinations promptly; only an
        explicit guided migration may download another destination subject.

        `subject_in_use` (3.5a) is WAITED OUT, never skipped. Something holds
        the subject — a lease, a resident model, a running task — and 3.5 says
        nothing is skipped, so the subject is retried on the next round with
        its holder named in the meantime, for a bounded number of rounds, and
        then the step fails BY THAT NAME. The two honest ends are "removed"
        and "still held, and here is who".

        The native adapter calls the same catalog/weights owner functions as
        the API. It never sends a native deletion to port 7100 after that port
        has become the guest's endpoint.
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
            source = {row.key: row for row in self._windows.installed_subjects() if row.kind != "engine"}
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
            missing = [str(row) for key, row in source.items() if key not in target]
            if missing and not allow_pull:
                raise self._fail("migration_cleanup_destination_missing",
                                 "Windows models are kept; the active guest must restore these subjects before cleanup: " + ", ".join(missing))
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
        """Installing an engine is not consent to expose it on every interface.

        Unless the operator said so, which `share_lan` is. The work itself is
        `crucible/lan.py`'s and is not restated here: an install that opened the
        door by its own second copy of the mechanism would be a door
        `crucible lan status` did not know about and `crucible lan disable`
        could not shut.
        """
        self._step("lan-door")
        from .. import lan as lan_door

        try:
            wanted = (
                lan_door.read(self._home) is not None
                if self._share_lan is None else self._share_lan
            )
        except CrucibleError as exc:
            # A record too broken to read is not a licence to guess which way
            # the operator wanted this. It names the file and stops.
            raise self._fail(
                "lan_door_failed",
                f"this machine's LAN sharing record cannot be read ({exc}), so this "
                "install will not guess whether to open the network door. Fix or delete "
                "the file and run the install again.",
            )
        if not wanted:
            self._finish(
                "lan-door",
                "Local engine access is ready. Network sharing is optional and must be "
                "enabled explicitly; installation changes no port forwards or firewall rules.",
            )
            return
        # Imported inside the method rather than at module scope: `crucible.lan`
        # reaches back into `crucible.host` for the mechanism, and the host
        # package is what this file belongs to. A deferred import is how
        # `cli.py` keeps the same two-way relation from becoming a cycle.
        from ..sharing import Engine

        door = landoor.detect(self._runner, ENGINE_PORT)
        missing = [
            command
            for present, command in (
                (door.forward, landoor.add_argv(ENGINE_PORT)),
                (door.firewall, landoor.firewall_add_argv(ENGINE_PORT)),
            )
            if not present
        ]
        if missing and not self._elevate:
            # `--no-elevate` REPORTS the argv and changes nothing. Saying "done"
            # here would be the one lie this whole step exists to avoid.
            self._finish(
                "lan-door",
                "network sharing was requested, but this install may not elevate; "
                "nothing was changed. Run `crucible lan enable` to open it.",
                argv=lan_door.elevated_argv(missing),
            )
            return
        if missing:
            self._line(landoor.ELEVATION_SENTENCE)
        try:
            # `adopt=True`: a forward this machine already had is not a reason to
            # stop an install the operator asked for. It is still VERIFIED and
            # still recorded, so `lan disable` remains able to shut what it opened.
            result = lan_door.enable(
                self._home, self._runner, Engine(self._home, "lan"),
                port=ENGINE_PORT, adopt=True,
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
        self._finish("lan-door", result["detail"])

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
