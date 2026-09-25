"""The job-type contract and the per-job run context.

A job type is a plugin module under `crucible/jobs/<type>/` declaring what it can
serve and how to run it (DESIGN.md section 3). Phase 1 ships exactly one: `echo`.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..errors import JobCancelled

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
#: THE SERVER STOPPED WHILE THIS WAS WORKING, and it is not `failed`.
#:
#: A job is `running` only while a process is running it, so a `running` job
#: found ON DISK at startup is one whose server went away underneath it —
#: nothing else can leave that state written down. Until 2026-09-20 the store
#: was in memory only, so such a job was not `failed`, it was GONE: a 404, with
#: its finished chunks sitting unreachable in `artifacts/`. Six minutes of a
#: fine-tuning ladder's GPU time was lost that way to a deploy.
#:
#: DISTINCT FROM `failed` BECAUSE CLIENTS ACT ON THE DIFFERENCE. `failed` is
#: this server judging the work — bad input, an engine that would not start —
#: and BookForge sends such a row to a person. An interruption is weather: the
#: right response is to collect what landed and re-ask for the rest, which is
#: what `artifacts-owed.ts` already does for a dropped stream. Reporting one as
#: the other would put a human in front of a queue that could have healed
#: itself.
INTERRUPTED = "interrupted"
TERMINAL_STATES = frozenset({DONE, FAILED, CANCELLED, INTERRUPTED})


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ModelDescriptor:
    """One model a job type can serve, as advertised by GET /info.

    DESIGN.md section 4's row. `installed` and `resident` are two facts and
    neither implies the other: `installed` is "the weights are on disk at the
    revision the manifest pins" — the puller's own stamp, read by the same
    predicate the doctor and the load path read — and `resident` is "an engine
    is serving it right now". It joined the row on 2026-09-14, when a puller
    reading `/v1/info` to decide whether to pull found the row could not say.
    """

    id: str
    revision: str
    source: str
    installed: bool
    resident: bool
    vram_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "revision": self.revision,
            "source": self.source,
            "installed": self.installed,
            "resident": self.resident,
            "vram_bytes": self.vram_bytes,
        }


@dataclass(frozen=True)
class JobTypeStatus:
    """What `crucible doctor` prints for one job type."""

    ready: bool
    detail: str


@dataclass
class Job:
    id: str
    type: str
    model: str | None
    params: dict[str, Any]
    dir: Path
    created: str
    status: str = QUEUED
    progress: float = 0.0
    started: str | None = None
    finished: str | None = None
    error: dict[str, str] | None = None
    artifacts: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False
    #: The last `progress` event's message, so a whole-server read
    #: (`GET /v1/activity`, PHASE7-LANES.md section 5) can say what a job is
    #: doing without replaying its event log. The event log stays the truth;
    #: this is the latest line off it.
    message: str | None = None
    #: Who submitted this job, from the User-Agent the SDK already sends
    #: (`<clientName> crucible-client/<version>`), or None when something spoke
    #: to this server without one.
    #:
    #: IDENTIFICATION, NOT AUTHORISATION. Everything holding the token is one
    #: trust domain (DESIGN.md section 8) and a client that lies about its name
    #: is lying to a bench widget. It exists because two BookForge instances can
    #: point at one server, and a bench that drew somebody else's render as its
    #: own would offer Owen a cancel button for a chapter his other machine is
    #: rendering. PHASE7-LANES.md section 5.
    client: str | None = None
    #: THE CLIENT'S OWN NAME FOR THIS WORK, echoed back and never read by this
    #: server. `POST /v1/jobs` takes it as `client_ref`.
    #:
    #: It exists for the restart that this record exists for. After BOTH sides
    #: have restarted, a client holding its own ledger has to match an
    #: `interrupted` job to the step that submitted it, and a job id it may
    #: have lost with everything else is a poor key to do it with. BookForge
    #: puts its queue step id here.
    #:
    #: `client_ref` and not `render_id`: this server runs `align`, `asr` and
    #: `rvc` jobs too, and a field called `render_id` on an `asr` job would be
    #: a name that lies. What it means is the client's, which is the whole
    #: point — nothing here parses it.
    client_ref: str | None = None
    #: When the server was found to have stopped while this was running. Set
    #: only on the restart that recovers it, never by the job itself.
    interrupted_at: str | None = None
    #: The chunk index of every artifact this job has published, for a job
    #: whose artifacts ARE indexed chunks. A resume is then a set difference
    #: rather than filename parsing — `<index>.flac` is a documented contract
    #: (`jobs/tts/render.py`), and a contract every client re-implements is a
    #: contract that drifts.
    chunks_done: list[int] = field(default_factory=list)
    #: How many indexed chunks this job was asked for, stated by the job type
    #: once it knows (`JobContext.expect_chunks`); None for a job whose
    #: artifacts are not chunks. With `chunks_done` this makes done/total ONE
    #: read of the record rather than a client's memory of what it sent — the
    #: ladder's ask of 2026-09-21, after the only mid-run pace it could measure
    #: was sglang's own log in /tmp, which died with the engine.
    chunks_total: int | None = None
    #: When the LAST chunk artifact landed (ISO-8601 UTC), None before any did.
    #: Two reads of the record a minute apart are a pace, with no engine log.
    chunk_at: str | None = None
    #: Extra keys a job type adds to its own `done` event. `load-model` puts
    #: `resident` here (PHASE2-LLM.md section 5); `artifacts` is always present.
    done_extra: dict[str, Any] = field(default_factory=dict)
    #: Every member of `artifacts/` a client has asked for through
    #: `GET /v1/jobs/{id}/artifacts/{name}`, sidecars included.
    #:
    #: WHAT THE REAPER READS (Owen's ruling, 2026-09-18): a job whose artifacts
    #: have all been collected is a job whose directory is a second copy of
    #: something the client now holds, and `JobStore.reap` deletes it. Nothing
    #: else reads this — it is not on the wire and it is not provenance.
    fetched: set[str] = field(default_factory=set)
    #: HELD FOR A CHAIN (Owen, 2026-09-25): *"keep all working files on the
    #: crucible side until the chain is complete. then remove them"*. The client
    #: that asked, while held; None when not. A held job is not reaped for being
    #: fetched, only released (`POST`/`DELETE /v1/jobs/{id}/hold`) or aged past
    #: `retention_days` (*"a garbage collector clean up files older than 7
    #: days"*). Persisted in the record, so a hold survives a restart.
    #: Each indexed artifact's chunk index, by artifact name, as the job type
    #: published it (`JobContext.artifact(..., index=)`). What lets a sidecar
    #: state only its own chunk (`JobStore.provenance`). In memory: it is read
    #: when a sidecar is written, which only happens in the process running the
    #: job.
    artifact_index: dict[str, int] = field(default_factory=dict)
    held_by: str | None = None
    #: When the hold was taken (ISO-8601 UTC), None when not held.
    held_since: str | None = None

    @property
    def held(self) -> bool:
        return self.held_since is not None

    @property
    def inputs_dir(self) -> Path:
        return self.dir / "inputs"

    @property
    def artifacts_dir(self) -> Path:
        return self.dir / "artifacts"

    @property
    def collected(self) -> bool:
        """Has the client taken every artifact of this job, sidecars included?

        FALSE FOR A JOB THAT PUBLISHED NOTHING, and that is the whole of why
        this is a method and not `set(artifacts) <= fetched`. `load-model`,
        `unload-voice` and a job that failed before it wrote anything all have
        an empty `artifacts` list, and an empty list is vacuously "all
        fetched" — which would reap a job the instant it ended, out from under
        the client still reading its event stream. A job with nothing to
        collect is the retention window's business, never this rule's.

        THE SIDECAR COUNTS. `SdkClient.#writeArtifact` fetches `<name>` and
        `<name>.provenance.json` with one `Promise.all`, so the two requests
        are in flight together; a rule that reaped on the artifact alone would
        race the sidecar's own GET and answer it `job_reaped` about a job the
        client was in the middle of collecting. DESIGN.md section 7 requires
        the client to keep the sidecar, so waiting for it costs nothing real.
        """
        if not self.artifacts:
            return False
        return all(
            name in self.fetched and f"{name}.provenance.json" in self.fetched
            for name in self.artifacts
        )


@runtime_checkable
class JobType(Protocol):
    """What every job-type plugin declares."""

    name: str

    def describe_models(self) -> list[ModelDescriptor]:
        """Models this type can serve on this host. Empty means the type takes no model."""

    def vram_estimate(self, model: str | None) -> int:
        """Bytes of accelerator memory a run of `model` needs."""

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        """The `model` block of this job's provenance sidecar, or None.

        DESIGN.md section 7: `{id, revision}` — and, since phase 3a, the
        `fingerprint` that joins them, because that is the string a client
        records (PHASE2-LLM.md section 5). It lives here rather than in the queue
        because the queue holds a model id and nothing that could turn it into a
        revision. A type that serves no models answers None, and the sidecar says
        `model: null`.
        """

    def check(self, backend: Any) -> JobTypeStatus:
        """Whether this type can run here right now, and why not if it cannot."""

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        """Refuse, by name, before the job is queued.

        Raises ApiError so the client gets an HTTP error naming the thing rather
        than a job that fails a minute later. A type with nothing to check here
        does nothing.
        """

    def run(self, job: Job, ctx: "JobContext") -> None:
        """Do the work. Blocking; the queue runs it on a worker thread."""


class JobContext:
    """Handed to `run()`. Progress and artifacts go back through here.

    `run()` executes on a worker thread, so every mutation of the job is marshalled
    back onto the event loop with `call_soon_threadsafe`. That also keeps event
    ordering exactly as the job emitted it.
    """

    def __init__(self, store: Any, job: Job, loop: asyncio.AbstractEventLoop) -> None:
        self._store = store
        self._job = job
        self._loop = loop

    @property
    def job(self) -> Job:
        return self._job

    @property
    def scratch(self) -> Path:
        return self._job.dir

    def inputs(self) -> dict[str, Path]:
        """The job's inputs, already materialised on disk, by name."""
        directory = self._job.inputs_dir
        if not directory.is_dir():
            return {}
        return {p.name: p for p in sorted(directory.iterdir()) if p.is_file()}

    @property
    def cancelled(self) -> bool:
        return self._job.cancel_requested

    def raise_if_cancelled(self) -> None:
        if self._job.cancel_requested:
            raise JobCancelled(f"job {self._job.id} was cancelled")

    def progress(self, fraction: float, message: str, **extra: Any) -> None:
        """Emit `progress {fraction, message, ...}`.

        `extra` is a job type's own measurements alongside the fraction, and it
        exists because a fraction is not always the useful number. `asr` sends
        processed seconds, total seconds and a running segment count
        (PHASE4-AUDIO.md section 3), so that BookForge's existing progress parser
        has exactly the information it has today — where those numbers come off
        the worker's own PROGRESS line and the percentage is still rounding to
        zero six minutes into an eighteen-hour book.

        `extra` cannot shadow `fraction` or `message`, and there is no check for
        it here because Python already refuses: both are named parameters, so
        `progress(0.5, "x", message="y")` is a TypeError before this body runs.
        """
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
            raise TypeError(f"progress fraction must be a number, got {fraction!r}")
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError(f"progress fraction must be in [0, 1], got {fraction!r}")
        self._loop.call_soon_threadsafe(
            self._store.append_event,
            self._job,
            "progress",
            {"fraction": float(fraction), "message": message, **extra},
        )

    def warming(self, message: str) -> None:
        """Emit a `warming {message}` event (PHASE2-LLM.md section 5).

        What a load job reports while an engine reads weights and captures CUDA
        graphs. Distinct from `progress`, which carries a fraction: nothing can
        honestly say how far through a model load it is.
        """
        if not isinstance(message, str):
            raise TypeError(f"a warming message must be a string, got {message!r}")
        self._loop.call_soon_threadsafe(
            self._store.append_event, self._job, "warming", {"message": message}
        )

    def note(self, message: str) -> None:
        """Emit a `note {message}` event — a fact about the MACHINE, not the work.

        The same event the lane already appends when a settlement could not
        clear the card (`crucible/jobs/queue.py:_settle`), given a door a job
        type can reach. A job says `progress` about what it is doing and
        `warming` about a load; a note is for the thing that happened AROUND the
        job and that a reader of its stream would otherwise never learn — "the
        engine would not stop, so its voice was taken off the card" being the
        one that asked for it (2026-09-15).

        It is not a terminal event and does not end a stream.
        """
        if not isinstance(message, str):
            raise TypeError(f"a note must be a string, got {message!r}")
        self._loop.call_soon_threadsafe(
            self._store.append_event, self._job, "note", {"message": message}
        )

    def chunk(
        self,
        *,
        index: int,
        seconds: float,
        chars: int,
        chars_per_sec: float,
        tokens: int | None,
        capped: bool | None,
        take: int,
        guard: dict[str, Any] | None,
    ) -> None:
        """Emit `chunk {index, seconds, chars, chars_per_sec, tokens, capped,
        take, guard}`.

        PHASE6-REMOTE-RENDER.md sections 3 and 4, amending PHASE3-TTS.md section
        6. **The model judges, this server forwards, the client orders.** The
        first seven keys are Crucible's own measurements of the bytes that
        arrived. `guard` is the verdict narrator's engine reached about the
        chunk, forwarded **verbatim** and `null` when narrator did not send one.

        Crucible does not read inside `guard`, does not validate its contents
        beyond "it is an object", and never acts on it — the same discipline
        `model_provenance` has. That statelessness is deliberate rather than
        lazy: a schema here that mirrored the retake ladder's internals would
        break the first time the ladder's vocabulary grew, and it would break at
        the first guard fire on a real book rather than at build time.

        Every argument is keyword-only and none has a default, because each one
        is a measurement and a measurement that defaulted would be a number
        nobody took. `tokens` and `capped` are `None`-able for the reason
        PHASE3-TTS.md section 6 gives: narrator does not report either on its
        wire at the pinned sha, and `None` means *narrator did not say*. It is
        never to be read as `false`. `guard` carries the same rule one level up:
        `null` means narrator sent no verdict — an engine that does not guard its
        own batch, or a row that never reached a decision — and is never to be
        read as "the take was clean".

        `chunk` is an addition to DESIGN.md section 4's event vocabulary and
        `api_version` does not move: a client that does not know the kind still
        sees every `progress`, `artifact` and `done` it saw before.
        """
        if capped is not None and not isinstance(capped, bool):
            raise TypeError(f"capped must be a bool or None, got {capped!r}")
        if guard is not None and not isinstance(guard, dict):
            # The ONE thing this server checks about a guard, and it checks it
            # because the event is JSON and an object is what the field is
            # declared to be. What is INSIDE it is narrator's business.
            raise TypeError(f"guard must be an object or None, got {guard!r}")
        self._loop.call_soon_threadsafe(
            self._store.append_event,
            self._job,
            "chunk",
            {
                "index": index,
                "seconds": seconds,
                "chars": chars,
                "chars_per_sec": chars_per_sec,
                "tokens": tokens,
                "capped": capped,
                "take": take,
                # VERBATIM: the object narrator sent, not a copy this server
                # reshaped. Nothing below this line reads a key inside it.
                "guard": guard,
            },
        )

    def cue(self, data: dict[str, Any]) -> None:
        """Emit a `cue {...}` event — one unit of the answer, as it lands.

        PHASE4-AUDIO.md section 2. `align` sends one per chunk, so a run killed
        at chunk 900 of 1,400 has cost the client the 500 it had not reached and
        not the 900 it had. Distinct from `progress`, which says how far along
        something is and carries no answer, and from `artifact`, which arrives
        once at the end and is all-or-nothing.

        It takes a dict rather than `**keys` because the payload is a *row of the
        answer* and its shape is the job type's, not this method's — an align cue
        is `{index, items}` or `{index, error}`, and a keyword signature here
        would invite the next type to invent a fourth spelling of `index`.
        """
        if not isinstance(data, dict):
            raise TypeError(f"a cue's data must be a dict, got {type(data).__name__}")
        self._loop.call_soon_threadsafe(
            self._store.append_event, self._job, "cue", dict(data)
        )

    def done_extra(self, **keys: Any) -> None:
        """Add keys to this job's `done` event, e.g. `resident` on a load."""
        self._job.done_extra.update(keys)

    def expect_chunks(self, total: int) -> None:
        """State how many indexed chunks this job will publish.

        Called once, before the first `artifact(..., index=...)`, by a job type
        whose artifacts are chunks. Written through with the record, so a
        client reading `GET /v1/jobs/<id>` after a restart still sees the
        denominator beside `chunks_done`.
        """
        if not isinstance(total, int) or isinstance(total, bool) or total < 0:
            raise ValueError(f"chunks_total must be a non-negative int, got {total!r}")
        self._loop.call_soon_threadsafe(self._store.record_chunks_total, self._job, total)

    def artifact(self, name: str, path: Path, *, index: int | None = None) -> Path:
        """Publish `path` as artifact `name`, with its provenance sidecar.

        Returns the artifact's final path. The sidecar is written immediately, so
        `<name>.provenance.json` exists for as long as the artifact does.
        """
        validate_member_name(name)
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"artifact source {source} is not a file")
        destination = self._job.artifacts_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        sidecar = self._job.artifacts_dir / f"{name}.provenance.json"
        sidecar.write_text(
            json.dumps(self._store.provenance(self._job, index=index), indent=2) + "\n",
            encoding="utf-8",
        )
        # `index` is the CHUNK this artifact is, for a job whose artifacts are
        # indexed chunks. Optional because most job types have no such notion —
        # an `asr` transcript is not chunk 12 of anything — and passing it is
        # how a resume becomes a set difference instead of every client parsing
        # `<index>.flac` for itself.
        self._loop.call_soon_threadsafe(
            self._store.record_artifact, self._job, name, index
        )
        return destination


_FORBIDDEN_NAME_PARTS = ("/", "\\", "\x00")


def validate_member_name(name: str) -> None:
    """Input and artifact names are single path members, never traversals."""
    if name == "" or name in (".", ".."):
        raise ValueError(f"invalid name {name!r}")
    for part in _FORBIDDEN_NAME_PARTS:
        if part in name:
            raise ValueError(f"invalid name {name!r}: contains {part!r}")
    if name.startswith("."):
        raise ValueError(f"invalid name {name!r}: must not start with a dot")
