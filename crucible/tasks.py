"""Operator tasks: the three things a person does TO a server, over HTTP.

PHASE13-OPERATOR.md section 3.3. A **task** is one operator operation the server
runs on itself — `pull` a subject, `install` a job type, or a `module` (an
ordered list of both) — and it exists because everything a person does to a
Crucible after it exists used to require a shell on that machine. The operator
page is the consumer; the CLI verbs it drives are unchanged and still the thing
a terminal person uses.

A TASK IS NOT A JOB, AND THE DIFFERENCE IS NOT COSMETIC
-------------------------------------------------------
A job is work a CLIENT wants done with this server's accelerator; a task is work
done to the server itself. They share the SSE envelope and nothing else:

* **Admission is separate.** One task at a time (`task_busy`), and a task and a
  job may run together — a `pull` is disk and network and has no business
  blocking a render. The one exception is `install`, which reloads the registry
  (3.4) and so may not START while any of the four facts holds the card.
* **Tasks are not persisted.** `JobStore` writes a directory per job with
  inputs, artifacts and a provenance sidecar. A task produces no artifact: what
  it leaves behind is an env or a directory of weights, and `GET /v1/catalog`
  is the record of those. The last `HISTORY` tasks are kept in memory so a page
  that reconnects can see what just happened, and a restart forgets them, as
  3.3 says it should.
* **There is no `queued` state.** A task is admitted and running in the same
  act, because there is nothing for it to queue behind: the second POST is
  refused rather than parked.

HOW A PULL IS CANCELLED, WHICH IS THE ONLY HARD PART IN HERE
-------------------------------------------------------------
`weights.pull` is a `snapshot_download` on a worker thread, and a Python thread
cannot be killed. So a cancel that merely dropped the `await` would mark the
task `cancelled` while nineteen gigabytes went on arriving — the "maybe" R3
forbids, in the most expensive form available.

The cancel is therefore **cooperative, through the progress hook**: `DELETE`
sets a flag, the hook raises `PullCancelled` on its next call, and
`crucible/weights.py` removes the partial directory on the way out. The hook is
called for every chunk of every file, so "the next call" is milliseconds. The
same hook is what promotes the pull's progress from the hub's terminal bar to
an event (R4) — one mechanism, two uses, and neither is a retrofit of the other:
a pull with no progress to report would also be a pull nothing could stop.

An `install` is a SUBPROCESS, so its cancel is a SIGTERM and needs none of this.
That asymmetry is why the two are written out separately below rather than
folded behind a `_run_one`.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from . import capability, catalog, interpreter, jobenv, workerenv
from .backend import LLAMA_WINDOWS, Backend
from .config import Config
from .errors import ApiError, CrucibleError
from .hosttools import searched_note, which
from .jobs.base import utcnow
from .jobs.llm import llm_engine_status
from .settle import Held
from .voices import NARRATOR_ENGINE_SAMPLING
from .weights import PullCancelled, WeightsError

RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL_STATES = frozenset({DONE, FAILED, CANCELLED})

#: The four things a task can be. PHASE13-OPERATOR.md 3.3, plus PHASE15-HOST.md
#: 4.7's `engine`, which is the one this server does not RUN: it hands it to
#: the host's loopback door and relays the host's events under its own id,
#: because only the host can run `wsl.exe`, prompt UAC and survive the reboot.
#:
#: `engine-restart` (PHASE17-ORCHESTRATOR.md 4.2) is the second of the two that
#: belong to the RELATION rather than to this server: the orchestrator restarts
#: its engine by the owner-appropriate means, and this side relays.
TASK_TYPES: tuple[str, ...] = (
    "pull",
    "install",
    "module",
    "engine",
    "engine-restart",
)

#: The only place this build moves the engine TO. 4.7: the reverse (WSL2 back
#: to Windows) is an explicit operator act written into section 6, and is
#: refused here rather than half-done.
ENGINE_TARGETS: tuple[str, ...] = ("wsl",)

#: **The env var the host sets on the server it starts**, naming its own
#: loopback door. Added by this build because `crucible/host/` did not have
#: one: the host spawns `crucible serve` as a child (`host/app.py`'s
#: `server_argv`) and the child inherited nothing that said a host was there.
#:
#: ITS PRESENCE IS THE FACT. 4.7: *"a Windows server that was not started by a
#: host (a developer running `crucible serve` by hand) refuses
#: `engine_move_needs_host`"*, and that is exactly "this variable is not set".
#: A probe of 127.0.0.1:7101 would be the wrong question twice over — it can
#: be answered by something that is not a host, and a host that is momentarily
#: restarting its door is still the host.
#:
#: THE TOKEN IS NOT CARRIED. The door's bearer is the ENGINE's token
#: (`crucible/host/door.py`: "a caller that can reach the engine can reach
#: this"), which this server already holds in its own config. A second copy in
#: an environment variable would be a secret with two owners and one more
#: place for it to be stale.
HOST_DOOR_ENV = "CRUCIBLE_HOST_DOOR"

#: What the host's door answers on. The server POSTs `{"target": "wsl"}` here
#: and reads newline-delimited JSON back.
HOST_DOOR_PATH = "/install"

#: The orchestrator's OTHER door route (PHASE17-ORCHESTRATOR.md 4.2). Same
#: bearer, same ndjson, same relay — one implementation, two sequences.
HOST_DOOR_RESTART_PATH = "/restart"

#: The POST-time refusal when there is no orchestrator to hand a restart to.
#:
#: The same FACT `engine_move_needs_host` names — `$CRUCIBLE_HOST_DOOR` is not
#: set — under a name that does not say "move". A server refusing a restart
#: with a sentence about moving to WSL2 sends a person to the wrong page, and
#: one code carrying two operator instructions is the defect T10 already found
#: once on this very door.
ENGINE_RESTART_NEEDS_ORCHESTRATOR = "engine_restart_needs_orchestrator"

#: THREE ENDINGS OF ONE DOOR, THREE NAMES — and they are not this module's
#: names, they are the door's (PHASE15-HOST.md 4.3 and 4.7).
#:
#: **Found by the first Windows run, 2026-09-14 (T10).** All three used to be
#: `engine_move_needs_host`, which made one code carry three different facts
#: with three different answers: *start a host* (there is none),
#: *start your host's door again* (there is one and it is dead), and *read the
#: host's log* (it answered and then abandoned the stream). The door's OTHER
#: caller — `@crucible/bootstrap`'s `requestHostInstall`, `sdk/bootstrap/src/
#: hostdoor.ts` — already had two of those names, so this side takes THEM
#: rather than inventing a third set for the same door (ARCHITECTURE.md R1:
#: one owner per name on the wire).
#:
#:     no `$CRUCIBLE_HOST_DOOR`     `engine_move_needs_host`  (this side only)
#:     connection refused/timeout   `host_unreachable`
#:     a stream with no terminal    `host_install_failed`
#:     a `failed` event             the code IT carries, verbatim
#:     an HTTP refusal, no code     `engine_move_needs_host` — unchanged, and
#:                                  for its own reason: something answered
#:                                  7101 and it is not behaving like a host,
#:                                  which is the same fact as "no host here".
HOST_UNREACHABLE = "host_unreachable"
HOST_INSTALL_FAILED = "host_install_failed"

#: How long the server waits for the host to ACCEPT the move. The sequence
#: itself takes as long as it takes — a distro import and a pack download —
#: and is read line by line with no deadline of its own, because a deadline
#: here would abandon an install that is still running on the machine.
HOST_DOOR_CONNECT_SECONDS = 30.0

#: How many finished tasks a server remembers. In memory, and a restart forgets
#: — a task is not a record anybody keeps (3.3). Fifty is enough for a page that
#: reconnects to show what happened while it was away and small enough that a
#: server left running for a month does not accumulate a log nobody reads.
HISTORY = 50

#: Seconds between `progress` events for one pull, at most. The hub's hook fires
#: per chunk, which on a 19 GB snapshot is tens of thousands of calls; every one
#: of them appended to an in-memory event log and replayed to every attached SSE
#: reader would make the progress reporting cost more than the download. The
#: CANCEL check is not throttled — it runs on every call, which is what keeps a
#: cancel sub-second.
PROGRESS_INTERVAL_SECONDS = 0.5

#: How long a SIGTERMed install is given to stop before it is killed. The child
#: is pip; it has nothing to flush and no card to release, unlike the engines
#: `crucible/residency.py` waits three minutes for.
TERMINATE_GRACE_SECONDS = 10.0


class TaskCancelled(CrucibleError):
    """Raised inside a runner once a cancel has been asked for."""


class TaskFailedByHost(CrucibleError):
    """The host's sequence failed and has ALREADY said why on this stream.

    Its own type so `_run` can mark the task failed without writing a second
    description of one failure. 4.7's events are the host's verbatim, and the
    `failed` one among them is the sentence a person reads.
    """

    def __init__(self, task: "Task") -> None:
        super().__init__(f"task {task.id} failed on the host")


def _host_refusal_code(body: str) -> str:
    """The host's own `error.code` out of its refusal body, or ours.

    The host refuses with named codes of its own — `host_install_running`,
    `host_no_token`, `host_unauthorized`, `engine_target_unknown` and every
    state code from the 4c table — and those names are what a client acts on,
    so they travel rather than being flattened into one. A body this server
    cannot read becomes `engine_move_needs_host`, which is the honest answer
    about a door that is not behaving like the host's.
    """
    try:
        payload = json.loads(body)
        code = payload["error"]["code"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return "engine_move_needs_host"
    return str(code) if isinstance(code, str) and code else "engine_move_needs_host"


class ReloadRefused(CrucibleError):
    """The registry could not be swapped because something holds the card.

    Carries the holder so the task's `failed` event can name it. See
    PHASE13-OPERATOR.md section 3.4: the four facts gate an install at POST and
    are read again at the swap, and a job admitted in between is a loud refusal
    rather than a registry replaced underneath it.
    """

    def __init__(self, held: Held) -> None:
        super().__init__(str(held))
        self.held = held


@dataclass
class Task:
    """One operator operation, as `GET /v1/tasks/{id}` reports it."""

    id: str
    type: str
    #: The request body, echoed. A page that reconnects draws the row from this
    #: rather than re-deriving what was asked for from the events.
    request: dict[str, Any]
    created: str
    started: str
    state: str = RUNNING
    finished: str | None = None
    error: dict[str, str] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    #: Classes a `module` named that this engine does not serve, with the
    #: capability row's own reason (PHASE15-HOST.md 5.3a). Empty for every
    #: other task type and for a module whose every class resolved — an
    #: EMPTY LIST and not None, so "nothing was unmet" and "this server
    #: predates the field" are not one reading.
    unmet: list[dict[str, str]] = field(default_factory=list)
    cancel_requested: bool = False
    #: The install subprocess, while one is running. A cancel SIGTERMs it.
    process: subprocess.Popen[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.id,
            "type": self.type,
            "request": self.request,
            "state": self.state,
            "error": self.error,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "unmet": self.unmet,
        }


# ----------------------------------------------------------------- validation
#
# Every refusal below is made BEFORE the 202, and the order is the job door's
# (`crucible/api.py`'s `create_job`): what is wrong with the REQUEST first,
# because a typo is true whatever this server is doing; then what is wrong with
# the SERVER's state. A client with a misspelled subject id told "busy" would
# come back in ten minutes to be told about the typo.


def require_installable(job_type: str) -> None:
    """Is there an installer for this type? Names the one that builds it if not.

    `INSTALLER_FOR` is `crucible doctor`'s table and the owner of "which command
    installs this", and it is asked here for doctor's reason: `denoise` has no
    installer of its own because it SHARES the `rvc` env, so the bare refusal
    — "there is no installer for 'denoise'" — is true and sends its reader
    looking for a command that will never exist. A refusal that names the fix
    which actually works is the difference between a declaration somebody can
    correct and one they argue with.
    """
    # One owner of "what has an installer", and of "which one builds this".
    from .cli import INSTALLABLE_JOB_TYPES, INSTALLER_FOR

    if job_type in INSTALLABLE_JOB_TYPES:
        return
    installer = INSTALLER_FOR.get(job_type)
    if installer is not None:
        raise ApiError(
            400,
            "unknown_job_type",
            f"job type {job_type!r} has no installer of its own: it shares "
            f"{installer!r}'s env, so installing {installer!r} is what builds "
            f"it. Name {installer!r} instead",
            {"job_type": job_type, "installed_by": installer},
        )
    raise ApiError(
        400,
        "unknown_job_type",
        f"there is no installer for job type {job_type!r}; this build "
        f"installs {sorted(INSTALLABLE_JOB_TYPES)}",
    )


def require_narrator_engine(job_type: str, narrator_engine: str | None) -> None:
    """`tts` must say which engine; everything else must not.

    The same rule `crucible install` enforces and for the same reason
    (`crucible/cli.py`'s `_env_spec`): on `cuda-linux` the tts env is named per
    narrator engine, because two of them cannot share a venv — each pins its
    own serving stack against its own torch. A bare `tts` would build one of
    them and the operator would not know which.
    """
    if job_type == "tts":
        if narrator_engine is None:
            raise ApiError(
                400,
                "narrator_engine_required",
                "installing 'tts' needs narrator_engine: on cuda-linux two "
                "engines cannot share a venv, so there is one env per engine "
                f"and no default. This build knows "
                f"{sorted(NARRATOR_ENGINE_SAMPLING)}",
            )
        if narrator_engine not in NARRATOR_ENGINE_SAMPLING:
            raise ApiError(
                400,
                "narrator_engine_required",
                f"{narrator_engine!r} is not one of narrator's engines; they are "
                f"{sorted(NARRATOR_ENGINE_SAMPLING)}",
            )
        return
    if narrator_engine is not None:
        raise ApiError(
            400,
            "narrator_engine_refused",
            f"narrator_engine names which tts env to build and means nothing for "
            f"{job_type!r}, which has exactly one env per host",
        )


def env_installed(config: Config, backend: Backend, job_type: str, engine: str | None) -> bool:
    """Is this job type's env already built on this host?

    **The env, not the `[jobs]` flag.** A flag is one boolean for a whole
    capability, and `tts` has one env PER NARRATOR ENGINE behind it: with
    `enable_tts` true because one engine is installed, asking the flag would
    refuse the next engine's install as `job_type_installed` while there is no
    env for it anywhere on the disk. The env is the thing `install` actually
    builds, so the env is what decides whether there is anything to do.
    """
    try:
        if job_type == "llm":
            # Native Windows owns a llama.cpp executable, not a Python venv.
            # Use the same installed predicate as model loading and /v1/models.
            return llm_engine_status(config, backend).installed
        if job_type in workerenv.WORKER_JOB_TYPES:
            return workerenv.env_status(config.home, job_type, backend.kind).installed
        spec = jobenv.tts_env(engine or "", backend.kind)
        return jobenv.env_status(config.home, spec, backend.kind).installed
    except (jobenv.EnvError, workerenv.WorkerEnvError):
        # An env whose recipe or directory cannot be read is not an env that is
        # installed. Reported as "not installed" rather than raised, because the
        # install about to run is exactly what would fix it — and it will raise
        # the same error itself, with its own output, if it cannot.
        return False


def _validate_pull(config: Config, backend: Backend, kind: str, subject_id: str) -> None:
    if kind not in catalog.KINDS:
        raise ApiError(
            404,
            "unknown_subject",
            f"{kind!r} is not a subject kind; they are {list(catalog.KINDS)}",
            {"kind": kind, "id": subject_id},
        )
    subject = catalog.find(config, backend, kind, subject_id)
    if subject is None:
        raise ApiError(
            404,
            "unknown_subject",
            f"this server has no {kind} called {subject_id!r} for "
            f"{backend.kind}. GET /v1/catalog lists every subject it can hold",
            {"kind": kind, "id": subject_id},
        )
    if subject.installed() is not None:
        # REFUSED, not skipped, and the difference is written into
        # PHASE13-OPERATOR.md section 3.3 on purpose: a single pull is a
        # person asking for one specific thing, and answering "fine" without
        # doing anything is how somebody comes to believe a re-pull repaired
        # something. A MODULE is idempotent and skips instead, because there
        # the request is "make this true", not "do this".
        raise ApiError(
            409,
            "already_installed",
            f"{kind} {subject_id!r} is already installed on this server. A pull "
            f"of an installed subject is refused rather than skipped; to replace "
            f"it deliberately, run `{subject.pull_command} --force` on the server",
            {"kind": kind, "id": subject_id},
        )


def _validate_engine(backend: Backend, target: str) -> str:
    """The engine move's three refusals, in the door's order. 4.7.

    Returns the host door's base URL. Every refusal here is made BEFORE the
    202, like every other task's, and the order is the same: what is wrong
    with the REQUEST (`engine_target_unknown`), then what is wrong with this
    MACHINE (`engine_move_not_here`), then what is wrong with this PROCESS
    (`engine_move_needs_host`).
    """
    if target not in ENGINE_TARGETS:
        raise ApiError(
            400,
            "engine_target_unknown",
            f"{target!r} is not an engine this build moves to; the targets are "
            f"{list(ENGINE_TARGETS)}. Moving BACK to Windows is an explicit "
            "operator act (PHASE15-HOST.md section 6) and is refused rather "
            "than half-done",
            {"target": target, "targets": list(ENGINE_TARGETS)},
        )
    if backend.kind != LLAMA_WINDOWS:
        raise ApiError(
            409,
            "engine_move_not_here",
            f"this server runs the {backend.kind} backend on "
            f"{backend.platform}, and the engine move is a Windows machine "
            "swapping llama.cpp for the WSL2 guest. There is nothing here to "
            "move from",
            {"backend": backend.kind, "platform": backend.platform},
        )
    door = os.environ.get(HOST_DOOR_ENV, "").strip()
    if door == "":
        raise ApiError(
            409,
            "engine_move_needs_host",
            "this server was not started by `crucible host`, so there is "
            f"nothing to hand the move to (${HOST_DOOR_ENV} is not set). Only "
            "the host can run wsl.exe, prompt for administrator and survive "
            "the reboot the move may need — a server doing it itself would "
            "stop halfway through and take its own event stream with it. "
            "Start the host and press it again from the page",
            {"env": HOST_DOOR_ENV},
        )
    return door.rstrip("/")


def _validate_engine_restart() -> str:
    """The restart's ONE refusal. PHASE17-ORCHESTRATOR.md 4.2.

    Returns the orchestrator door's base URL. Unlike the MOVE, there is no
    backend check: every engine an orchestrator started can be restarted by
    it, whether it is the guest's unit on `cuda-linux` or a `llama-windows`
    child on Windows. `engine_move_not_here` is a fact about a machine that
    has a Windows engine to move FROM, and a restart moves nothing.

    Whether the engine is one the orchestrator may TOUCH at all is not asked
    here and cannot be: `owner` is PHASE15 4.1a's fact and the orchestrator is
    the only process that holds it. A `found` engine is refused
    `engine_not_ours` at the orchestrator's door, and that refusal arrives on
    this task's own event stream.
    """
    door = os.environ.get(HOST_DOOR_ENV, "").strip()
    if door == "":
        raise ApiError(
            409,
            ENGINE_RESTART_NEEDS_ORCHESTRATOR,
            "this server was not started by an orchestrator "
            f"(${HOST_DOOR_ENV} is not set), so there is nothing here that "
            "can restart it. Only the orchestrator can run wsl.exe, name the "
            "guest's unit or respawn a child — a server restarting itself "
            "would take its own event stream with it and leave nobody to say "
            "whether it came back. Start `crucible orchestrator` and press it "
            "again from the page",
            {"env": HOST_DOOR_ENV},
        )
    return door.rstrip("/")


def _validate_install(
    config: Config, backend: Backend, job_type: str, narrator_engine: str | None
) -> None:
    require_installable(job_type)
    require_narrator_engine(job_type, narrator_engine)
    if env_installed(config, backend, job_type, narrator_engine):
        raise ApiError(
            409,
            "job_type_installed",
            f"job type {job_type!r}"
            + (f" ({narrator_engine})" if narrator_engine else "")
            + " already has its env on this server. To rebuild it deliberately, "
            "run `crucible install` with --force on the server",
            {"job_type": job_type, "narrator_engine": narrator_engine},
        )


@dataclass(frozen=True)
class ModuleEntry:
    """One step of a module. A job type, a named subject, or a CLASS.

    The third arm arrived with PHASE15-HOST.md 5.3a: a module names classes
    and THIS server resolves them, through its own capability record, because
    the record is per machine and the generator that used to resolve them
    runs on one. A class entry becomes a pull of whatever this card selected,
    or an `unmet` row — never a refusal of the whole module.
    """

    name: str
    job_type: str | None = None
    narrator_engine: str | None = None
    kind: str | None = None
    subject_id: str | None = None
    capability_class: str | None = None


def validate_module(
    config: Config, backend: Backend, module: Any
) -> list[ModuleEntry]:
    """Read a whole module or refuse the whole of it. Never half.

    3.3: *"A module is validated WHOLE before anything starts."* Collecting
    every problem and naming them together is the difference between an app
    author fixing one typo per five-minute install and fixing all four at once
    — and, more importantly, between a server that installed two of a module's
    six entries before discovering the third was misspelled and one that did
    nothing.
    """
    problems: list[str] = []

    if not isinstance(module, dict):
        raise ApiError(
            400,
            "invalid_module",
            f"a module is a JSON object, got {type(module).__name__}",
        )
    for key in ("name", "version"):
        if not isinstance(module.get(key), str) or module[key].strip() == "":
            problems.append(f"{key}: a module needs a non-empty string {key}")
    unknown = sorted(
        set(module) - {"name", "version", "job_types", "needs", "subjects"}
    )
    if unknown:
        problems.append(
            f"unknown key(s) {unknown}; a module carries exactly name, version, "
            "job_types, needs and subjects"
        )

    entries: list[ModuleEntry] = []
    raw_types = module.get("job_types", [])
    if not isinstance(raw_types, list):
        problems.append("job_types: must be a list")
        raw_types = []
    for index, raw in enumerate(raw_types):
        where = f"job_types[{index}]"
        if not isinstance(raw, dict):
            problems.append(f"{where}: must be an object with a `type`")
            continue
        stray = sorted(set(raw) - {"type", "narrator_engine"})
        if stray:
            problems.append(f"{where}: unknown key(s) {stray}")
            continue
        job_type = raw.get("type")
        engine = raw.get("narrator_engine")
        if not isinstance(job_type, str):
            problems.append(f"{where}: `type` must be a string")
            continue
        if engine is not None and not isinstance(engine, str):
            problems.append(f"{where}: `narrator_engine` must be a string")
            continue
        try:
            require_installable(job_type)
            require_narrator_engine(job_type, engine)
        except ApiError as exc:
            problems.append(f"{where}: {exc.message}")
            continue
        entries.append(
            ModuleEntry(
                name=f"install {job_type}"
                + (f" ({engine})" if engine else ""),
                job_type=job_type,
                narrator_engine=engine,
            )
        )

    # NEEDS ARE CLASSES AND THIS SERVER RESOLVES THEM (5.3a). What is checked
    # here is only that the class EXISTS — a word this build does not have is
    # a defect in the module and is the same defect on every machine. Whether
    # this card can serve it is not checked at all: a class this backend has
    # disabled is `unmet` on the result, not a refusal, because a module is an
    # app saying what it needs and a Mac with no page reader is still a Mac
    # Foundry can use for text.
    raw_needs = module.get("needs", [])
    if not isinstance(raw_needs, list):
        problems.append("needs: must be a list")
        raw_needs = []
    for index, raw in enumerate(raw_needs):
        where = f"needs[{index}]"
        if not isinstance(raw, dict):
            problems.append(f"{where}: must be an object with a `class`")
            continue
        stray = sorted(set(raw) - {"class"})
        if stray:
            problems.append(
                f"{where}: unknown key(s) {stray}. A need is a CLASS and nothing "
                "else; an app that wants one specific model names it under "
                "`subjects`, which is a choice and says so"
            )
            continue
        capability_class = raw.get("class")
        if not isinstance(capability_class, str):
            problems.append(f"{where}: `class` must be a string")
            continue
        if capability_class not in capability.BY_NAME:
            problems.append(
                f"{where}: {capability_class!r} is not a capability class; they "
                f"are {sorted(capability.BY_NAME)}"
            )
            continue
        entries.append(
            ModuleEntry(
                name=f"resolve {capability_class}",
                capability_class=capability_class,
            )
        )

    raw_subjects = module.get("subjects", [])
    if not isinstance(raw_subjects, list):
        problems.append("subjects: must be a list")
        raw_subjects = []
    for index, raw in enumerate(raw_subjects):
        where = f"subjects[{index}]"
        if not isinstance(raw, dict):
            problems.append(f"{where}: must be an object with `kind` and `id`")
            continue
        stray = sorted(set(raw) - {"kind", "id"})
        if stray:
            problems.append(f"{where}: unknown key(s) {stray}")
            continue
        kind, subject_id = raw.get("kind"), raw.get("id")
        if not isinstance(kind, str) or not isinstance(subject_id, str):
            problems.append(f"{where}: `kind` and `id` must both be strings")
            continue
        if kind not in catalog.KINDS:
            problems.append(
                f"{where}: {kind!r} is not a subject kind; they are "
                f"{list(catalog.KINDS)}"
            )
            continue
        if catalog.find(config, backend, kind, subject_id) is None:
            problems.append(
                f"{where}: this server has no {kind} called {subject_id!r} for "
                f"{backend.kind}"
            )
            continue
        entries.append(
            ModuleEntry(
                name=f"pull {kind} {subject_id}", kind=kind, subject_id=subject_id
            )
        )

    if not entries and not problems:
        problems.append(
            "a module with no job_types and no subjects asks for nothing; if that "
            "is what this app needs, it needs no module"
        )
    if problems:
        raise ApiError(
            400,
            "invalid_module",
            "this module was not run because "
            + (
                "it has a problem: " if len(problems) == 1 else
                f"it has {len(problems)} problems: "
            )
            + "; ".join(problems),
            {"problems": problems},
        )
    return entries


# -------------------------------------------------------------- the installer


def install_command() -> str:
    """Where the `crucible` console script is, or a refusal naming both places.

    The CONSOLE SCRIPT and not `python -m crucible`, for the reason
    PHASE11-SERVICE.md found the hard way: `python -m crucible` run from a
    directory containing a `crucible/` folder imports that folder instead of the
    installed package, and the server's working directory is whatever the
    service manager gave it.

    Beside this interpreter FIRST, then `PATH`. A server running out of one venv
    must install into that venv's Crucible, not into whichever one happens to be
    earlier on a unit's bare PATH — and `crucible/hosttools.py` is the owner of
    "what PATH did we search", so the refusal says it.
    """
    sibling = Path(sys.executable).resolve().parent / "crucible"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    found = which("crucible")
    if found is not None:
        return found
    raise ApiError(
        503,
        "install_command_missing",
        f"this server cannot install anything: there is no `crucible` console "
        f"script at {sibling} and none on PATH {searched_note()}. It is the "
        "script `pip install crucible` writes beside the interpreter this "
        "server runs on",
    )


# ------------------------------------------------------------------ the store


class TaskStore:
    """Every task this server has seen in this process, and the one lane for them.

    One lane, and unlike `JobStore`'s there is no deque behind it: a second
    submission is refused, full stop. Tasks have no client-side queue to be the
    less informed half of — an operator page is one person pressing one button.
    """

    def __init__(
        self,
        config: Config,
        backend: Backend,
        *,
        reload: Callable[[], list[str]],
        holder: Callable[[], Held | None],
    ) -> None:
        self._config = config
        self._backend = backend
        #: The 3.4 swap, injected because it needs the app: the registry, the
        #: residency and the store all live there and a task module that reached
        #: for them would be a second owner of how a server is assembled.
        self._reload = reload
        #: The four facts, read through the one thing that owns them
        #: (`crucible/settle.py`). Injected for `reload`'s reason.
        self._holder = holder
        self._tasks: dict[str, Task] = {}
        self._order: list[str] = []
        self._running_id: str | None = None
        self._runner: asyncio.Task[None] | None = None
        self._subscribers: dict[str, list[asyncio.Event]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------- lifecycle

    async def stop(self) -> None:
        """Cancel whatever is running, on the way out of the app's lifespan.

        A pull is asked to stop through its own flag as well as through the
        `asyncio.Task`, because cancelling the task alone would leave the
        download thread running into a shutting-down process.
        """
        runner, self._runner = self._runner, None
        if runner is None:
            return
        running = self.running
        if running is not None:
            self._request_cancel(running)
        runner.cancel()
        try:
            await runner
        except asyncio.CancelledError:
            pass

    # ----------------------------------------------------------------- state

    @property
    def running(self) -> Task | None:
        return None if self._running_id is None else self._tasks[self._running_id]

    def get(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if task is None:
            raise ApiError(
                404,
                "unknown_task",
                f"no task {task_id} on this server. Tasks are held in memory and "
                f"the last {HISTORY} are kept, so this one may have finished "
                "before a restart or been pushed out by newer ones",
            )
        return task

    def recent(self) -> list[Task]:
        """The last `HISTORY` tasks, newest first."""
        return [self._tasks[task_id] for task_id in reversed(self._order)]

    # ------------------------------------------------------------- admission

    def refuse_if_busy(self) -> None:
        """One task at a time, naming the one that has the lane."""
        running = self.running
        if running is None:
            return
        raise ApiError(
            409,
            "task_busy",
            f"this server is already running task {running.id} ({running.type}), "
            f"started {running.started}. One operator task at a time: they write "
            "to the same config and the same weights tree, and two at once would "
            "be two answers about what is installed. Watch "
            f"GET /v1/tasks/{running.id}/events for its end",
            {"task_id": running.id, "type": running.type, "since": running.started},
        )

    def refuse_if_the_card_is_held(self) -> None:
        """The four facts, for an install only. `409 server_busy`.

        3.3: a `pull` may run beside a job, because it is disk and network. An
        `install` may not, because it ends by swapping this server's registry
        (3.4) — and the four facts are the one place that knows whether anything
        is mid-run (`crucible/settle.py`). The holder's own fields travel on the
        refusal so an app's row can name who is in the way verbatim rather than
        drawing a button that looks broken.
        """
        held = self._holder()
        if held is None:
            return
        raise ApiError(
            409,
            "server_busy",
            f"this server cannot install anything right now: {held}. An install "
            "rewrites config.toml and reloads this server's job registry, so it "
            "waits until nothing holds the card",
            {"fact": held.fact, "who": held.who, **held.details},
        )

    # ---------------------------------------------------------------- submit

    def submit(self, request: dict[str, Any]) -> Task:
        """Validate, admit, and start. **Event loop only.**

        Returns the admitted task; the work runs in a background task on this
        loop. Nothing awaits between the admission check and the start, so the
        two are one atomic stretch — `JobStore.enqueue`'s property, for
        `JobStore.enqueue`'s reason.
        """
        task_type = request["type"]
        if task_type == "pull":
            _validate_pull(self._config, self._backend, request["kind"], request["id"])
        elif task_type == "install":
            _validate_install(
                self._config,
                self._backend,
                request["job_type"],
                request.get("narrator_engine"),
            )
        elif task_type == "module":
            validate_module(self._config, self._backend, request["module"])
        elif task_type == "engine":
            _validate_engine(self._backend, request["target"])
        elif task_type == "engine-restart":
            _validate_engine_restart()
        else:  # unreachable: the request model closes the vocabulary
            raise ApiError(
                400,
                "invalid_request",
                f"{task_type!r} is not a task type; they are {list(TASK_TYPES)}",
            )

        self.refuse_if_busy()
        if _touches_the_registry(task_type, request):
            self.refuse_if_the_card_is_held()

        now = utcnow()
        task = Task(
            id=uuid.uuid4().hex,
            type=task_type,
            request=request,
            created=now,
            # Admitted and running in the same act: there is no queue for tasks,
            # so a `created` that preceded `started` would be a duration that is
            # always zero pretending to mean something.
            started=now,
        )
        self._tasks[task.id] = task
        self._order.append(task.id)
        self._prune()
        self._running_id = task.id
        self._loop = asyncio.get_running_loop()
        self._runner = self._loop.create_task(
            self._run(task), name=f"crucible-task-{task.id}"
        )
        return task

    def _prune(self) -> None:
        while len(self._order) > HISTORY:
            oldest = self._order[0]
            if self._tasks[oldest].state not in TERMINAL_STATES:
                # Unreachable while one runs at a time; asserted rather than
                # assumed, because dropping a running task's record would make
                # its own `done` unreportable.
                raise RuntimeError(
                    f"task {oldest} is {self._tasks[oldest].state} and would be "
                    "pruned; the history must never drop a live task"
                )
            self._order.pop(0)
            self._tasks.pop(oldest, None)
            self._subscribers.pop(oldest, None)

    # ---------------------------------------------------------------- cancel

    def cancel(self, task: Task) -> str:
        """`DELETE /v1/tasks/{id}`. Refuses a task that has already finished."""
        if task.state in TERMINAL_STATES:
            raise ApiError(
                409,
                "not_running",
                f"task {task.id} is already {task.state}; there is nothing to "
                "cancel",
                {"task_id": task.id, "state": task.state},
            )
        self._request_cancel(task)
        return "cancelling"

    def _request_cancel(self, task: Task) -> None:
        task.cancel_requested = True
        process = task.process
        if process is not None and process.poll() is None:
            # An install is a subprocess, so its cancel is a signal and lands at
            # once. pip has nothing to flush and holds no card, which is why the
            # grace here is ten seconds rather than the three minutes
            # `crucible/residency.py` gives an engine.
            process.terminate()

    # ---------------------------------------------------------------- events

    def append_event(self, task: Task, kind: str, data: dict[str, Any]) -> None:
        """Append one SSE event. **Event loop thread only.**"""
        task.events.append(
            {"id": len(task.events) + 1, "event": kind, "data": data}
        )
        for waiter in self._subscribers.get(task.id, []):
            waiter.set()

    def subscribe(self, task: Task) -> asyncio.Event:
        waiter = asyncio.Event()
        self._subscribers.setdefault(task.id, []).append(waiter)
        return waiter

    def unsubscribe(self, task: Task, waiter: asyncio.Event) -> None:
        waiters = self._subscribers.get(task.id)
        if waiters is None:
            return
        if waiter in waiters:
            waiters.remove(waiter)
        if not waiters:
            self._subscribers.pop(task.id, None)

    def _from_thread(self, task: Task, kind: str, data: dict[str, Any]) -> None:
        """Append an event from a worker thread, on the loop.

        `append_event` sets `asyncio.Event`s, which are not thread-safe, so a
        download thread cannot call it. `call_soon_threadsafe` is the whole of
        the marshalling and it is not optional: the alternative — appending from
        the thread and hoping the reader notices — is a stream that stalls under
        exactly the load it exists to report on.
        """
        loop = self._loop
        if loop is None:  # pragma: no cover - a task always has one
            return
        loop.call_soon_threadsafe(self.append_event, task, kind, data)

    # ------------------------------------------------------------- the runner

    async def _run(self, task: Task) -> None:
        try:
            self.append_event(task, "started", {"type": task.type})
            if task.type == "pull":
                await self._run_pull(task)
            elif task.type == "install":
                await self._run_install(task)
            elif task.type == "engine":
                await self._run_engine(task)
            elif task.type == "engine-restart":
                await self._run_engine_restart(task)
            else:
                await self._run_module(task)
        except (TaskCancelled, PullCancelled):
            self._finish(task, CANCELLED, None)
        except TaskFailedByHost:
            # The host's own `failed` event is already on this stream, with
            # its own code and its own sentence. `_finish` is called with no
            # error dict so nothing writes a second one; the task is FAILED
            # and the reason is the event above it.
            self._finish(task, FAILED, None)
        except ReloadRefused as exc:
            self._finish(
                task,
                FAILED,
                {
                    "code": "reload_refused",
                    "message": (
                        f"the env is installed, but this server would not swap "
                        f"its job registry while {exc.held}. Nothing was undone "
                        f"(R6); run the install again and it will find the env "
                        f"built and reach the reload in seconds"
                    ),
                },
            )
        except ApiError as exc:
            self._finish(task, FAILED, {"code": exc.code, "message": exc.message})
        except asyncio.CancelledError:
            # The server is shutting down. The task is not `failed` — nothing
            # about it went wrong — and nobody is left to read either answer.
            self._finish(task, CANCELLED, None)
            raise
        except Exception as exc:
            self._finish(
                task,
                FAILED,
                {"code": "task_failed", "message": f"{type(exc).__name__}: {exc}"},
            )
        else:
            self._finish(task, DONE, None)

    def _finish(
        self, task: Task, state: str, error: dict[str, str] | None
    ) -> None:
        task.state = state
        task.finished = utcnow()
        task.error = error
        task.process = None
        self._running_id = None
        if state == DONE:
            self.append_event(task, "done", {"unmet": task.unmet})
        elif state == FAILED:
            if error is not None:
                self.append_event(task, "failed", error)
        else:
            self.append_event(task, "cancelled", {})

    def _raise_if_cancelled(self, task: Task) -> None:
        if task.cancel_requested:
            raise TaskCancelled(f"task {task.id} was cancelled")

    # ------------------------------------------------------------------ pull

    async def _run_pull(self, task: Task) -> None:
        await self._pull_one(
            task, task.request["kind"], task.request["id"], index=1, total=1
        )

    async def _pull_one(
        self, task: Task, kind: str, subject_id: str, *, index: int, total: int
    ) -> None:
        self._raise_if_cancelled(task)
        subject = catalog.find(self._config, self._backend, kind, subject_id)
        if subject is None:  # pragma: no cover - validated at submit
            raise ApiError(
                404, "unknown_subject", f"no {kind} called {subject_id!r}"
            )
        self.append_event(
            task,
            "step",
            {"name": f"pull {kind} {subject_id}", "index": index, "total": total},
        )
        await self._pull(task, subject)

    async def _pull(self, task: Task, subject: catalog.Subject) -> None:
        """The download, with the hub's refusals given a name of their own.

        `WeightsError` already says exactly what went wrong — a gated repo, a
        revision the manifest names and the repo does not, a digest that did
        not match — and it would otherwise reach a client as `task_failed`,
        which is the generic bucket `crucible/errors.py` says this server does
        not have. The message is the weights module's, verbatim.
        """
        try:
            await asyncio.to_thread(self._pull_blocking, task, subject)
        except WeightsError as exc:
            raise ApiError(
                500,
                "pull_failed",
                f"pulling {subject.kind} {subject.id!r} failed: {exc}",
                {"kind": subject.kind, "id": subject.id},
            ) from None

    def _pull_blocking(self, task: Task, subject: catalog.Subject) -> None:
        """**Worker thread.** The hub's download, reported and interruptible."""
        last = 0.0

        def on_progress(done: int, total: int | None, name: str) -> None:
            # UNTHROTTLED, because this is the cancel check: a pull that has
            # been cancelled must stop at the next chunk, not at the next
            # half-second.
            if task.cancel_requested:
                raise PullCancelled(f"task {task.id} was cancelled")
            nonlocal last
            now = time.monotonic()
            if now - last < PROGRESS_INTERVAL_SECONDS:
                return
            last = now
            self._from_thread(
                task,
                "progress",
                {"bytes_done": done, "bytes_total": total, "file": name},
            )

        def on_line(line: str) -> None:
            # NOT an event. The pull's events carry bytes, and interleaving a
            # second `progress` shape into one stream would make a typed reader
            # branch on which keys arrived. The line goes where every other
            # server line goes, for an operator tailing the unit (R4: promote
            # the fact, leave the log alone).
            print(f"crucible: task {task.id}: {line}", file=sys.stderr)

        subject.pull(force=False, on_line=on_line, on_progress=on_progress)

    # ---------------------------------------------------------------- engine

    async def _run_engine(self, task: Task) -> None:
        """Hand the move to the host and RELAY what it says. 4.7.

        This server runs none of it. The host's door emits newline-delimited
        JSON whose lines are already shaped like this module's events — that
        is `crucible/host/installer.py`'s `Event`, written that way on
        purpose, because *"a relay that reshapes is a second owner of the
        shape"*. So the whole of the relay is: read a line, append it.

        THE STREAM ENDS WHEN THE HOST ENDS IT. There is no deadline on the
        read: the sequence imports a distro and downloads a pack, and a
        server that gave up on it would leave an install running on the
        machine with nobody watching. A connection that CLOSES before a
        terminal event is a failure the host did not report, and is named
        here rather than reported as success.
        """
        door = _validate_engine(self._backend, task.request["target"])
        self.append_event(
            task,
            "step",
            {"name": "hand the move to the host", "index": 1, "total": 1},
        )
        terminal = await asyncio.to_thread(
            self._relay_blocking, task, door, HOST_DOOR_PATH, {"target": task.request["target"]}
        )
        if terminal is None:
            raise ApiError(
                502,
                HOST_INSTALL_FAILED,
                f"the host's door at {door}{HOST_DOOR_PATH} closed its stream "
                "without saying whether the move finished. Nothing here can "
                "tell a completed install from an abandoned one, so it is "
                "reported as a failure; the host's log says what happened",
                {"door": door},
            )
        if terminal == "failed":
            # The host already emitted its own `failed` event with its own
            # code and sentence, and that event is on this task's stream
            # verbatim. Raising a SECOND description of it would put two
            # sentences about one failure in one place.
            raise TaskFailedByHost(task)

    def _relay_blocking(
        self, task: Task, door: str, path: str, body_fields: dict[str, Any]
    ) -> str | None:
        """**Worker thread.** POST, then one appended event per line read.

        Returns the name of the terminal event the host sent (`done` or
        `failed`), or None when the stream ended without one.

        ONE RELAY, TWO SEQUENCES (PHASE17 4.2). `path` and `body_fields` are
        the only things the move and the restart differ by; everything below —
        the bearer, the line-by-line append, the unparseable-line rule, the
        cancel check, the three endings — is the same contract for both. A
        second copy of it would be a second owner of the ending names, which
        is the defect T10 found the first time these names were written twice.
        """
        import urllib.error
        import urllib.request

        body = json.dumps(body_fields).encode("utf-8")
        request = urllib.request.Request(
            f"{door}{path}",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                # The ENGINE's token, which this server already holds. The
                # door takes it because a caller that can reach the engine
                # can reach the door and nothing else can.
                "Authorization": f"Bearer {self._config.token}",
            },
        )
        terminal: str | None = None
        try:
            with urllib.request.urlopen(
                request, timeout=HOST_DOOR_CONNECT_SECONDS
            ) as stream:
                for raw in stream:
                    if task.cancel_requested:
                        # 4.7: cancellable BETWEEN STEPS. Dropping the read is
                        # what this side can do; the host's own sequence
                        # finishes the step it is in and stops, which is why
                        # the cancel is not a kill.
                        raise TaskCancelled(f"task {task.id} was cancelled")
                    line = raw.decode("utf-8", "replace").strip()
                    if line == "":
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        # A line this server cannot read is still evidence,
                        # and it is put on the stream as one rather than
                        # dropped: the alternative is an install whose events
                        # silently thin out.
                        self._from_thread(task, "progress", {"line": line})
                        continue
                    name = str(event.get("event") or "progress")
                    data = event.get("data")
                    self._from_thread(
                        task, name, data if isinstance(data, dict) else {}
                    )
                    if name in ("done", "failed", "cancelled"):
                        terminal = name
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise ApiError(
                502 if exc.code >= 500 else 409,
                _host_refusal_code(detail),
                f"the host refused the move with HTTP {exc.code}: {detail}",
                {"door": door, "host_status": exc.code},
            ) from None
        except (urllib.error.URLError, OSError) as exc:
            raise ApiError(
                502,
                HOST_UNREACHABLE,
                f"the orchestrator's door at {door}{path} did not answer: "
                f"{type(exc).__name__}: {exc}. A host started this server "
                f"(${HOST_DOOR_ENV} is set) and its door is not answering "
                "now. Start it from the Startup item, or run `crucible host` "
                "from the host runtime, and press it again",
                {"door": door},
            ) from None
        return terminal

    # ------------------------------------------------------- engine-restart

    async def _run_engine_restart(self, task: Task) -> None:
        """Hand the restart to the orchestrator and RELAY. PHASE17 4.2.

        **THE LAST EVENT MAY NEVER ARRIVE, AND THAT IS NOT A DEFECT.** The
        relay runs in the process being restarted, so the stream this task is
        writing to dies with it. PHASE15 4.7 set the precedent for the move —
        *"the page, which lost its server for a few seconds at the
        switch-over, re-reads `/v1/info`"* — and a restart is the same shape
        in less time. The client believes `/v1/info`, not the stream.

        A stream that ends with no terminal event is therefore reported with
        4.7's `host_install_failed`. That name is slightly wrong for a
        restart, and it is KEPT rather than forked: one relay with one set of
        endings is worth more than a second table, and the alternative is the
        thing ARCHITECTURE.md R1 forbids — one fact with two names depending
        on which route read it.
        """
        door = _validate_engine_restart()
        self.append_event(
            task,
            "step",
            {"name": "hand the restart to the orchestrator", "index": 1, "total": 1},
        )
        terminal = await asyncio.to_thread(
            self._relay_blocking, task, door, HOST_DOOR_RESTART_PATH, {}
        )
        if terminal is None:
            raise ApiError(
                502,
                HOST_INSTALL_FAILED,
                f"the orchestrator's door at {door}{HOST_DOOR_RESTART_PATH} "
                "closed its stream without saying whether the engine came "
                "back. That is the EXPECTED shape when the engine being "
                "restarted is the one relaying: read `GET /v1/info` for the "
                "answer, which is the only place it is reliably true",
                {"door": door},
            )
        if terminal == "failed":
            raise TaskFailedByHost(task)

    # --------------------------------------------------------------- install

    async def _run_install(self, task: Task) -> None:
        await self._install_one(
            task,
            task.request["job_type"],
            task.request.get("narrator_engine"),
            index=1,
            total=2,
        )
        await self._reload_step(task, index=2, total=2)

    async def _install_one(
        self,
        task: Task,
        job_type: str,
        narrator_engine: str | None,
        *,
        index: int,
        total: int,
    ) -> None:
        self._raise_if_cancelled(task)
        command = install_command()
        argv = [command, "install", job_type, "--verbose"]
        if narrator_engine is not None:
            argv += ["--narrator-engine", narrator_engine]
        self.append_event(
            task,
            "step",
            {
                "name": f"install {job_type}"
                + (f" ({narrator_engine})" if narrator_engine else ""),
                "index": index,
                "total": total,
            },
        )
        code = await asyncio.to_thread(self._run_install_process, task, argv)
        self._raise_if_cancelled(task)
        if code != 0:
            raise ApiError(
                500,
                "install_failed",
                f"`{' '.join(argv)}` exited {code}. Its output is on this task's "
                "event stream, line by line, and in the server's log; the env "
                "that was built is left on disk (R6) so a re-run does not start "
                "again from nothing",
            )

    def _run_install_process(self, task: Task, argv: list[str]) -> int:
        """**Worker thread.** Run the console script, one `progress` per line.

        `CRUCIBLE_HOME` is stated rather than inherited, because a server may
        have been started with one and this child must install into the home
        THIS server loaded its config from — not into whichever one the service
        manager's environment happens to name.
        """
        environment = dict(os.environ)
        environment["CRUCIBLE_HOME"] = str(self._config.home)
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            # Into one stream, because pip writes to both and two pipes read by
            # one thread is a deadlock waiting for a big enough error message.
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        task.process = process
        assert process.stdout is not None
        last_bytes = 0.0
        try:
            for line in process.stdout:
                stripped = line.rstrip("\n")
                # THE CANCEL CHECK IS FIRST, before anything below can `continue`
                # past it. A download emits a sentinel line per megabyte and
                # most of them are throttled away; a cancel tested only on the
                # lines that survive the throttle is a cancel that waits half a
                # second at best and, on a quiet stretch, never fires.
                if task.cancel_requested and process.poll() is None:
                    process.terminate()
                # AN INSTALL THAT DOWNLOADS AN INTERPRETER REPORTS ITS BYTES.
                # `crucible install` prints a sentinel line carrying the three
                # fields the PULL task already emits
                # (`interpreter.PROGRESS_PREFIX` owns the shape and says why
                # the child's stdout is the transport), so the operator page
                # draws that part of an env install with exactly the code that
                # draws a weights pull instead of a second progress shape. The
                # recipe's own wheels come from pip, whose prose goes through
                # the `line` branch below.
                measured = interpreter.parse_progress_line(stripped)
                if measured is not None:
                    now = time.monotonic()
                    if now - last_bytes < PROGRESS_INTERVAL_SECONDS:
                        continue
                    last_bytes = now
                    self._from_thread(task, "progress", measured)
                    continue
                self._from_thread(task, "progress", {"line": stripped})
            code = process.wait(timeout=TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait()
        finally:
            task.process = None
        return code

    async def _reload_step(self, task: Task, *, index: int, total: int) -> None:
        """Make what was just installed reachable, before `done`. Section 3.4.

        On the event loop and in one synchronous stretch with the four-facts
        re-read inside `self._reload`, so nothing can be admitted between the
        two.
        """
        job_types = self._reload()
        self.append_event(
            task,
            "step",
            {
                "name": "reload",
                "index": index,
                "total": total,
                # What became reachable, so a client is told rather than having
                # to diff two `/v1/info` reads (3.4).
                "job_types": job_types,
            },
        )

    # ---------------------------------------------------------------- module

    async def _run_module(self, task: Task) -> None:
        entries = validate_module(self._config, self._backend, task.request["module"])
        # One extra step for the reload, and only when something is actually
        # installed: a module of pulls changes no registry and a `reload` step
        # in its stream would be a step that did nothing.
        total = len(entries)
        installs = [entry for entry in entries if entry.job_type is not None]
        if installs:
            total += 1
        index = 0
        installed_anything = False
        #: The classes this server could NOT resolve, with the capability
        #: row's own sentence. 5.3a: a class this backend has disabled is not
        #: a refusal — the module is done, and the app shows "not on this
        #: engine" beside the pulls it did make.
        unmet: list[dict[str, str]] = []
        for entry in entries:
            index += 1
            self._raise_if_cancelled(task)
            if entry.capability_class is not None:
                resolved = self._resolve_need(task, entry, index, total)
                if resolved is None:
                    unmet.append(self._unmet_row(entry.capability_class))
                    continue
                if resolved.installed() is not None:
                    self.append_event(
                        task,
                        "skipped",
                        {
                            "reason": f"{entry.name}: this card selected "
                            f"{resolved.id!r} and it is already installed"
                        },
                    )
                    continue
                await self._pull(task, resolved)
                continue
            if entry.job_type is not None:
                self.append_event(
                    task,
                    "step",
                    {"name": entry.name, "index": index, "total": total},
                )
                if env_installed(
                    self._config, self._backend, entry.job_type, entry.narrator_engine
                ):
                    self.append_event(
                        task,
                        "skipped",
                        {
                            "reason": f"{entry.name}: this server already has that "
                            "env. A module says what must be true, so an entry "
                            "that is already true is skipped rather than refused"
                        },
                    )
                    continue
                await self._install_step(task, entry, index, total)
                installed_anything = True
                continue

            assert entry.kind is not None and entry.subject_id is not None
            subject = catalog.find(
                self._config, self._backend, entry.kind, entry.subject_id
            )
            assert subject is not None  # validated whole, above
            self.append_event(
                task, "step", {"name": entry.name, "index": index, "total": total}
            )
            if subject.installed() is not None:
                self.append_event(
                    task,
                    "skipped",
                    {
                        "reason": f"{entry.name}: already installed. A module says "
                        "what must be true, so an entry that is already true is "
                        "skipped rather than refused"
                    },
                )
                continue
            await self._pull(task, subject)

        if installs:
            index += 1
            if installed_anything:
                await self._reload_step(task, index=index, total=total)
            else:
                self.append_event(
                    task, "step", {"name": "reload", "index": index, "total": total}
                )
                self.append_event(
                    task,
                    "skipped",
                    {
                        "reason": "reload: every job type this module names was "
                        "already installed, so this server's registry is already "
                        "what the module asks for"
                    },
                )
        # WHAT THIS ENGINE CANNOT DO, on the task's own record (5.3a). Carried
        # on the task rather than only on an event, because an app that
        # attached late reads `GET /v1/tasks/{id}` and must still learn that
        # `pages` is not on this machine — and a `done` with no `unmet` and a
        # `done` whose `unmet` is empty have to be the same answer.
        task.unmet = unmet

    def _resolve_need(
        self, task: Task, entry: ModuleEntry, index: int, total: int
    ) -> "catalog.Subject | None":
        """Which subject THIS card selected for this class, or None.

        PHASE9: the capability record is the one place a class is resolved,
        and the record is per machine. None means the class is not served
        here — disabled, or selected onto a model this backend has no block
        for — and the caller turns that into an `unmet` row rather than a
        failure.
        """
        assert entry.capability_class is not None
        self.append_event(
            task, "step", {"name": entry.name, "index": index, "total": total}
        )
        row = self._capability_row(entry.capability_class)
        if row is None or not row.enabled or row.selected == "":
            return None
        subject = catalog.find(
            self._config, self._backend, "model", row.selected
        )
        if subject is None:
            # The record names a model this backend has no block for. That is
            # a stale record rather than a disabled class, and it is reported
            # as `unmet` for the same reason: the app's answer is the same,
            # and `crucible capability --write` is the operator's fix.
            return None
        self.append_event(
            task,
            "progress",
            {
                "line": f"{entry.capability_class}: this card selected "
                f"{row.selected}"
            },
        )
        return subject

    def _capability_row(self, capability_class: str) -> Any:
        record = self._config.capability
        return None if record is None else record.row(capability_class)

    def _unmet_row(self, capability_class: str) -> dict[str, str]:
        """`{class, reason}` — the capability row's OWN sentence, or why not.

        Never a sentence written here: the row said why the class is off and
        that is the sentence an app shows. A server with NO record at all
        says so by name, because "nothing has probed this card" and "this
        card cannot do it" are different things for an operator to fix.
        """
        row = self._capability_row(capability_class)
        if row is None:
            return {
                "class": capability_class,
                "reason": (
                    "this server has no capability record, so it cannot say "
                    "which model serves this class. Run `crucible capability "
                    "--write` on it"
                ),
            }
        if row.selected != "" and row.enabled:
            return {
                "class": capability_class,
                "reason": (
                    f"this card selected {row.selected!r} and this backend has "
                    "no block for it; the capability record is stale. Run "
                    "`crucible capability --write`"
                ),
            }
        return {"class": capability_class, "reason": row.reason}

    async def _install_step(
        self, task: Task, entry: ModuleEntry, index: int, total: int
    ) -> None:
        """One module entry's install, with R6's promise in its failure.

        The `step` event is already out, so a failure here names the index by
        being the last step anybody saw — and the message says it too, because
        an SSE reader that attached late has the index and a log reader does
        not.
        """
        assert entry.job_type is not None
        command = install_command()
        argv = [command, "install", entry.job_type, "--verbose"]
        if entry.narrator_engine is not None:
            argv += ["--narrator-engine", entry.narrator_engine]
        code = await asyncio.to_thread(self._run_install_process, task, argv)
        self._raise_if_cancelled(task)
        if code != 0:
            raise ApiError(
                500,
                "install_failed",
                f"step {index} of {total} ({entry.name}) exited {code}. The "
                "module stops here and every step before it STAYS — the envs "
                "and weights are on disk (R6). Re-posting the module skips what "
                "is already installed and resumes at this step",
            )


def _touches_the_registry(task_type: str, request: dict[str, Any]) -> bool:
    """Would this task end by swapping the job registry (3.4)?

    A `pull` never does, so it is admitted beside a running render. A `module`
    does exactly when it names at least one job type — a module of pure pulls is
    a pull, and gating it on the four facts would refuse a download because
    somebody is reading a book.
    """
    if task_type == "install":
        return True
    if task_type != "module":
        return False
    module = request.get("module")
    if not isinstance(module, dict):
        return False
    declared = module.get("job_types")
    return isinstance(declared, list) and len(declared) > 0


def module_document(entries: Iterable[ModuleEntry]) -> list[str]:
    """The step names a module will run, in order. For a caller that wants a plan."""
    return [entry.name for entry in entries]


__all__ = [
    "CANCELLED",
    "DONE",
    "FAILED",
    "HISTORY",
    "RUNNING",
    "ENGINE_RESTART_NEEDS_ORCHESTRATOR",
    "HOST_DOOR_RESTART_PATH",
    "TASK_TYPES",
    "TERMINAL_STATES",
    "ModuleEntry",
    "ReloadRefused",
    "Task",
    "TaskCancelled",
    "TaskStore",
    "env_installed",
    "install_command",
    "require_installable",
    "require_narrator_engine",
    "module_document",
    "validate_module",
]
