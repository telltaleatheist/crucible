"""The job store and the single exclusive lane that drains it.

The server owns the accelerator (DESIGN.md section 6): one job runs at a time, on
one worker task. Clients never see a lock; they see `position` and the event
stream.

**ADMISSION, NOT QUEUEING (Owen, 2026-09-13; ARCHITECTURE.md section 3).** *"i
think all queuing logic should exist in the clients, not the server. if the
server is busy, it cant receive a new job. if its not busy, it receives the next
job requested."* So a submission arriving while the lane is occupied is
**refused** — `refuse_if_busy` — rather than appended behind the running job.

The reason is not simplicity. The client is the only thing that knows what the
user wants: the chain, the pin, the priority, which book is being watched. A
server-side FIFO can only ever be a dumb queue, and having one forces the smart
client queue to *model* it — two arbitrators, one strictly less informed. It also
ends an inconsistency that was already here: the streaming door has always
refused (`ttsstream.py`, `409 stream_session_open`, naming the holder) while this
door queued. One server, two policies, no reason.

**The lane, the deque, `position`, `queue_depth`, cancel, events and provenance
all stay exactly as they were.** What changed is one policy decision at
admission, so `_pending` never grows past the job the lane is about to pick up —
and if queueing is ever wanted back it is the same one line. Ripping out tested
machinery to get behaviour a policy gives you is the expensive version of the
right idea.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import VERSION
from ..errors import ApiError, JobCancelled, JobError
from .base import (
    CANCELLED,
    DONE,
    FAILED,
    QUEUED,
    RUNNING,
    TERMINAL_STATES,
    Job,
    JobContext,
    JobType,
    utcnow,
)

#: How often the lane looks for job directories to delete while it is idle.
#:
#: A MINUTE BECAUSE NOTHING HERE IS URGENT. The two reasons a job is reaped are
#: "the client has it" and "it is a week old", and neither becomes wrong by
#: being acted on a minute late. The cost of the tick is a `listdir` of the
#: jobs directory, so a shorter one would buy nothing and spin a loop that is
#: otherwise asleep on an event.
REAP_INTERVAL_SECONDS = 60.0


def _now() -> datetime:
    """The reaper's clock, in one place so a test can move it.

    Its own function for `crucible/residency.py`'s reason — a module that has
    to reason about elapsed time needs one place the time comes from, or a
    keeper about a seven-day window has to wait seven days.
    """
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Reaped:
    """The tombstone a reaped job leaves behind, so its id still answers.

    A REAPED JOB IS NOT AN UNKNOWN ONE. `GET /v1/jobs/{id}` answering
    `unknown_job` about a job this server ran an hour ago tells the client it
    got the id wrong, which sends it looking in the wrong place; `job_reaped`
    with the moment and the reason tells it what actually happened and what to
    do differently (fetch sooner, or keep the bytes it was given).

    THEY ARE NOT CAPPED, and the reason is arithmetic rather than principle:
    `JobStore` has always held every job it has ever seen in this process, and
    a tombstone is a few hundred bytes where the `Job` it replaces holds its
    params, its artifact list and its whole event log. Reaping strictly
    reduces what this process keeps; capping the tombstones would put the
    `unknown_job` lie back for the oldest of them to save the smallest thing
    here.
    """

    job_id: str
    #: `fetched` or `aged`. Two words rather than prose because the message is
    #: built from them and the pair is the whole vocabulary.
    why: str
    #: When the reaper took it, in `Job.finished`'s format.
    when: str
    #: The sentence `GET /v1/jobs/{id}` answers with, written once here so the
    #: stderr line and the 404 cannot tell two stories about one deletion.
    detail: str


def busy_details(job: Job) -> dict[str, Any]:
    """What a `409 server_busy` says about the job that is in the way.

    Its own function because there are now TWO doors that refuse for this job:
    `POST /v1/jobs`, below, and `POST /v1/tasks` (PHASE13-OPERATOR.md section
    3.3), which reads it through `Settlement.holder()`. Two hand-written copies
    of these eight fields would be two answers about one job the first time one
    of them was edited — ARCHITECTURE.md R1 in the smallest possible space.

    `holder` is `Job.client`, the recorded User-Agent, and it is null when
    something spoke to this server without one: **null means "it did not say"**,
    and inventing a name here would make a bench confidently wrong about whose
    render is on the card (PHASE7-LANES.md section 5).
    """
    return {
        "holder": job.client,
        "job_id": job.id,
        "type": job.type,
        "model": job.model,
        # Named so `since` is unambiguous: "running" dates from `started`,
        # "queued" from `created`. Without it a client cannot tell a job that
        # has been rendering for an hour from one admitted 2 ms ago.
        "status": job.status,
        # `started` for a running job, `created` for one the lane has not
        # reached: both answer "since when", and a null `started` reported as
        # `since` would read as "it has been busy since never".
        "since": job.started if job.started is not None else job.created,
        "progress": job.progress,
        # The holder's latest progress line, which is what turns "busy" into
        # "rendering 118 of 280" on somebody else's bench. Null until the job
        # has said anything.
        "message": job.message,
    }


class JobStore:
    """Holds every job this server has seen in this process, plus the run lane."""

    def __init__(self, config: Any, backend: Any, registry: dict[str, JobType]) -> None:
        self._config = config
        self._backend = backend
        self._registry = registry
        self._jobs: dict[str, Job] = {}
        #: Every job this store has reaped, by id. See `Reaped`: it is what
        #: keeps `GET /v1/jobs/{id}` from calling a job it deleted an unknown
        #: one, and it is never emptied.
        self._reaped: dict[str, Reaped] = {}
        #: Every upload blob a job has CONSUMED, by blob id, to the job that
        #: took it. An upload is MOVED into the job that names it (Owen's
        #: ruling, 2026-09-18), so there is exactly one copy of those bytes on
        #: this disk and the second job to name the same blob has to be told
        #: why the bytes are not there rather than that they never were.
        self._consumed_blobs: dict[str, str] = {}
        self._pending: deque[str] = deque()
        self._wake = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._running_id: str | None = None
        self._subscribers: dict[str, list[asyncio.Event]] = {}
        #: Owen's ruling, 2026-09-14 (`crucible/settle.py`). Injected rather than
        #: constructed here, because the settlement has to read three things this
        #: store has never heard of — the lease, the claim and the chats in
        #: flight — and a queue that built it would have to learn all three.
        #: None is a store with no server around it: `crucible doctor` and the
        #: admission unit tests build one, and neither has a card to clear.
        self._settlement: Any | None = None

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._worker is not None:
            raise RuntimeError("the job worker is already running")
        self._worker = asyncio.create_task(self._run_lane(), name="crucible-job-lane")

    async def stop(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            # The lane died on its own before shutdown reached it. `_run_lane`
            # guards against every way that can happen today, so this is the
            # backstop behind a backstop — but re-raising here would turn one
            # dead worker into a server that cannot shut down, which is how a
            # WSL2 guest ends up with a CUDA process nobody can SIGTERM. Say it
            # loudly on the way out instead.
            print(
                f"crucible: the job lane had already died: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        self._worker = None

    # ------------------------------------------------------------------- state

    def attach_settlement(self, settlement: Any) -> None:
        """Hand the lane the thing that clears the card when a job is done."""
        self._settlement = settlement

    @property
    def registry(self) -> dict[str, JobType]:
        return self._registry

    @property
    def queue_depth(self) -> int:
        """Jobs on the lane: running plus waiting.

        Keeps its name, its shape and its arithmetic. Under the admission policy
        it is honestly 0 or 1 — the only way it reads 1 with nothing running is
        the sub-millisecond window between a submission being admitted and the
        lane task waking to pick it up. Every phase-2 client reads this field;
        nothing about it needed to change for the number to become truthful.
        """
        return len(self._pending) + (1 if self._running_id is not None else 0)

    @property
    def running_id(self) -> str | None:
        return self._running_id

    @property
    def running(self) -> Job | None:
        """The job on the lane right now, or None."""
        return None if self._running_id is None else self._jobs[self._running_id]

    def occupied_by_anything_but(self, job_id: str | None) -> Job | None:
        """The job holding the lane other than this one, or None.

        The lane's half of the settlement's four facts (`crucible/settle.py`).
        `job_id` is the job asking, and it is asking **while it is still the
        running job** — the card is cleared before its terminal event, so the
        note lands on a stream its client is still reading. Without the
        exclusion every job would find itself and nothing would ever unload.

        Reads the deque as well as `_running_id`, for `refuse_if_busy`'s reason:
        a job admitted and not yet picked up is as much a hold on the card as one
        already running, and clearing the card out from under it would cost it a
        reload it never asked for.
        """
        running = self.running
        if running is not None and running.id != job_id:
            return running
        for pending_id in self._pending:
            if pending_id != job_id:
                return self._jobs[pending_id]
        return None

    def queued(self) -> list[Job]:
        """Everything waiting, in the order it will run.

        A LIST AND NOT THE DEQUE. `_pending` is the lane's own structure and a
        caller holding it could mutate the queue by accident; this is a snapshot
        for reading. PHASE7-LANES.md section 5.
        """
        return [self._jobs[job_id] for job_id in self._pending]

    def get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is not None:
            return job
        reaped = self._reaped.get(job_id)
        if reaped is not None:
            # NAMED, NEVER "UNKNOWN". A client told `unknown_job` about a job
            # this server finished an hour ago concludes it has the wrong id or
            # the wrong server, and goes looking for the bug somewhere it is
            # not. See `Reaped`.
            raise ApiError(
                404,
                "job_reaped",
                reaped.detail,
                {"job_id": job_id, "reaped_at": reaped.when, "why": reaped.why},
            )
        raise ApiError(404, "unknown_job", f"no job {job_id} on this server")

    def position(self, job: Job) -> int | None:
        """0 while running, 1-based place in line while queued, null once terminal.

        Unchanged by the admission ruling, and deliberately so: `position` is
        still what a client reads, it still means the same thing, and under the
        ruling the only value it can take besides 0 and null is 1 — "admitted,
        the lane has not picked it up yet". A client that already understands
        `position` needs to learn nothing.
        """
        if job.status == RUNNING:
            return 0
        if job.status == QUEUED:
            return self._pending.index(job.id) + 1
        return None

    # --------------------------------------------------------------- admission

    def refuse_if_busy(self) -> None:
        """Refuse, by name and with the facts, when the lane is occupied.

        Owen's ruling (ARCHITECTURE.md section 3): the server answers *"is there
        room now"* and nothing else. This is that answer, and it lives here
        rather than in the HTTP layer because **the lane's own state is the only
        authority on it** — an admission check written against a snapshot taken
        somewhere else is the two-arbitrators problem in miniature.

        IT READS `_pending` AND NOT ONLY `_running_id`, which is the whole of the
        race handling. `enqueue` appends and sets `_wake`; the lane is a task on
        the same event loop and does not resume until the current one yields, so
        for a moment there is an admitted job that nothing is running yet. A
        check that asked only "is something running" would admit a second job in
        that window and `_pending` would reach 2 — precisely the queue the ruling
        removes, reachable by two clients submitting a millisecond apart. Reading
        the deque closes it without a lock: the check and the append happen in one
        synchronous stretch on the event loop (see `enqueue`).

        THE REFUSAL CARRIES FACTS BECAUSE A BARE "BUSY" IS USELESS. It forces
        clients to poll, and polling is a *worse* queue than FIFO — the winner
        becomes whoever polls at the luckiest moment rather than whoever asked
        first. So the body names the holder, the job, what it is doing and how far
        along it is: enough for a client to back off intelligently, and exactly
        the *"GPU busy: foundry"* line BookForge wants, for free.

        `holder` is `Job.client`, the recorded User-Agent. It is null when
        something spoke to this server without one — **null means "it did not
        say"**, and inventing a name here would make a bench confidently wrong
        about whose render is on the card (PHASE7-LANES.md section 5).
        """
        holder = self.running
        if holder is None:
            if not self._pending:
                return
            # Admitted, not yet picked up. Reported as the holder because it is:
            # the lane is spoken for, and saying otherwise would be the polite
            # lie that lets the second client in.
            holder = self._jobs[self._pending[0]]

        who = "an unnamed client" if holder.client is None else repr(holder.client)
        what = holder.type if holder.model is None else f"{holder.type} {holder.model!r}"
        details = busy_details(holder)
        doing = "" if not holder.message else f" — {holder.message}"
        raise ApiError(
            409,
            "server_busy",
            f"this server is busy with job {holder.id} ({what}), {holder.status} "
            f"since {details['since']}, submitted by {who}, "
            f"{holder.progress:.0%} done"
            f"{doing}. Crucible admits one job at a time and does not queue: the "
            "client owns the queue, the server owns admission (ARCHITECTURE.md "
            "section 3). Read GET /v1/activity to see when it is finished.",
            details,
        )

    # ------------------------------------------------------------------ submit

    def create(
        self,
        job_type: str,
        model: str | None,
        params: dict[str, Any],
        client: str | None = None,
    ) -> Job:
        job_id = uuid.uuid4().hex
        directory = Path(self._config.jobs_dir) / job_id
        (directory / "inputs").mkdir(parents=True, exist_ok=False)
        (directory / "artifacts").mkdir(parents=True, exist_ok=False)
        job = Job(
            id=job_id,
            type=job_type,
            model=model,
            params=params,
            dir=directory,
            created=utcnow(),
            client=client,
        )
        self._jobs[job_id] = job
        return job

    def enqueue(self, job: Job) -> None:
        """Put an admitted job on the lane, or refuse if the lane took one first.

        `refuse_if_busy` again, and not as a belt-and-braces repeat of the
        caller's: this is where the append happens, so this is where the decision
        has to be final. The two calls are one implementation of one fact — the
        caller's is an early exit that avoids building a job directory and
        materialising inputs for a submission that will be refused anyway.

        Today nothing can slip between them: `POST /v1/jobs` runs from its check
        to this call without an `await`, so it is one atomic stretch on the event
        loop. This makes that a property of the queue rather than of a handler
        somebody may later add an `await` to.
        """
        self.refuse_if_busy()
        self._pending.append(job.id)
        self.append_event(job, "queued", {"position": self.position(job)})
        self._wake.set()

    def discard(self, job: Job) -> None:
        """Forget a job that was never admitted, and delete its scratch.

        A job exists from `create()`, which is before its inputs are written and
        before `enqueue`. Anything refused in between must leave nothing behind:
        a record left in `_jobs` would sit at `queued` forever without being on
        the lane, and `position()` would raise `ValueError` off `deque.index` for
        anyone who asked about it.
        """
        if job.id in self._pending:  # unreachable: only `enqueue` appends
            raise RuntimeError(
                f"job {job.id} is on the lane and cannot be discarded; cancel it"
            )
        self._jobs.pop(job.id, None)
        shutil.rmtree(job.dir, ignore_errors=True)

    # ----------------------------------------------------------------- uploads

    def refuse_if_blob_consumed(self, blob_id: str) -> None:
        """Refuse, by name, an upload some job has already taken.

        OWEN'S RULING, 2026-09-18: an upload is MOVED into the job that names
        it, not copied — one copy of those bytes on this disk. The move is
        `crucible/api.py`'s `_materialise_inputs`; what is here is the fact
        that a blob is consumed ONCE. After the move there is nothing left in
        `uploads/` for a second job to take, and the honest answer to that
        second job is not `unknown_blob` — this server did hold those bytes,
        and it can say where they went.

        The record survives the job, including one discarded before it ran:
        the bytes went with the job either way, and a client that gets this
        refusal has to upload them again whichever it was.
        """
        taken = self._consumed_blobs.get(blob_id)
        if taken is None:
            return
        raise ApiError(
            409,
            "blob_consumed",
            f"blob {blob_id!r} was consumed by job {taken}. An upload is moved "
            "into the job that names it, so this server holds one copy of "
            "those bytes and that job has it. Upload them again for this job",
            {"blob_id": blob_id, "job_id": taken},
        )

    def consume_blob(self, blob_id: str, job: Job) -> None:
        """Record that `job` has taken this upload. Refuses a second taker.

        The check is `refuse_if_blob_consumed` and is made again here rather
        than trusted from the caller's earlier pass: this is the line that
        writes the record, so this is where it has to be true.
        """
        self.refuse_if_blob_consumed(blob_id)
        self._consumed_blobs[blob_id] = job.id

    # ----------------------------------------------------------------- reaping

    def mark_fetched(self, job: Job, name: str) -> None:
        """Record that a client has asked for this member of `artifacts/`.

        Called by the artifact route once the file is known to exist. It is
        what turns `Job.collected` true, and `reap` deletes a collected job.
        """
        job.fetched.add(name)

    def reap(self) -> list[Reaped]:
        """Delete the job directories nothing needs any more. **Event loop only.**

        OWEN'S RULING, 2026-09-18 (ledger C5), and the measurement behind it:
        9.3 GB in 88 job directories on the PC since 09-12, 60 of them with an
        empty `artifacts/`, plus 2.4 GB of uploads. `store.discard` deleted a
        directory on the create-then-refused path and nothing else ever did:
        `_jobs` never evicted, so every render this server has ever done was
        still on the disk twice — once in the client's library and once here.

        TWO REASONS, BOTH LOUD, NEITHER A GUESS:

        - **fetched.** Every artifact and every sidecar has been GET. The
          client holds the bytes; this directory is the second copy.
        - **aged.** It finished more than `retention_days` ago
          (`crucible/config.py`). This is the backstop for the job nobody came
          back for, and for the one that published nothing to fetch.

        A QUEUED OR RUNNING JOB IS NEVER TOUCHED, which is why the terminal
        check comes first and is not an age comparison: a render that has been
        going for eight days is not a week-old job, it is this afternoon's
        work, and deleting its scratch would take the book with it.

        THE DIRECTORIES OF DEAD PROCESSES ARE REAPED TOO, by age alone. `_jobs`
        lives in this process, so every directory under `jobs/` that no live
        job owns belongs to a server that has already exited — its record is
        gone, nothing can ever fetch it, and it is the whole of the 9.3 GB. It
        is reaped by AGE and never immediately, so a second Crucible sharing
        this home (a different port on the same machine) cannot delete a
        directory the first one is writing into.

        Returns what it took, for the keepers and for a caller that wants to
        say so. Never raises on a directory it cannot delete: see the per-job
        `OSError` below.
        """
        now = _now()
        horizon = float(self._config.retention_days) * 86_400.0
        taken: list[Reaped] = []
        for job in list(self._jobs.values()):
            if job.status not in TERMINAL_STATES:
                continue
            if job.collected:
                record = self._reap_one(
                    job,
                    "fetched",
                    now,
                    f"its {len(job.artifacts)} artifact(s) and their sidecars "
                    "had all been fetched",
                )
            elif self._age_seconds(job, now) > horizon:
                record = self._reap_one(
                    job,
                    "aged",
                    now,
                    f"it finished at {job.finished} and this server keeps a "
                    f"finished job for {self._config.retention_days} day(s) "
                    "([jobs] retention_days)",
                )
            else:
                continue
            if record is not None:
                taken.append(record)
        taken.extend(self._reap_orphan_directories(now, horizon))
        return taken

    def _age_seconds(self, job: Job, now: datetime) -> float:
        """How long this terminal job has been finished, in seconds.

        `finished` is set by `_finish` and by `_fail_out_of_band` before either
        publishes a terminal status, so a terminal job without one is this
        module's own bug and is raised rather than given a substitute age —
        a missing timestamp read as "just now" would keep the directory for
        ever and a missing one read as 1970 would delete it at once.
        """
        if job.finished is None:
            raise RuntimeError(
                f"job {job.id} is {job.status} with no `finished` stamp; the "
                "reaper cannot say how old a job that never recorded an end is"
            )
        return (now - datetime.fromisoformat(job.finished)).total_seconds()

    def _reap_one(
        self, job: Job, why: str, now: datetime, because: str
    ) -> Reaped | None:
        """Delete one job's directory and replace its record with a tombstone.

        Returns None when the directory would not go. A CLEANUP FAILURE IS NOT
        AN OPERATION FAILURE: the job still ran and the client still has what
        it fetched, so this says what happened and leaves the record alone —
        the next tick tries again, and until it succeeds the id goes on
        answering about a job rather than about a deletion that did not
        happen. The one that really occurs is a Windows file still open in a
        `FileResponse` that is being streamed.
        """
        when = now.isoformat()
        detail = (
            f"job {job.id} was reaped at {when} because {because}. Crucible "
            "keeps a finished job's directory until its artifacts are fetched "
            "or it ages out; it is not an unknown job and this server did run "
            "it"
        )
        try:
            shutil.rmtree(job.dir)
        except OSError as exc:
            print(
                f"crucible: could not reap job {job.id} ({why}) at {job.dir}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return None
        del self._jobs[job.id]
        record = Reaped(job_id=job.id, why=why, when=when, detail=detail)
        self._reaped[job.id] = record
        print(
            f"crucible: reaped job {job.id} ({job.type}) — {because}",
            file=sys.stderr,
        )
        return record

    def _reap_orphan_directories(
        self, now: datetime, horizon: float
    ) -> list[Reaped]:
        """Delete aged job directories no live job owns. See `reap`.

        No tombstone is left, because there is no id to answer for: the record
        these directories belonged to died with the process that made them, so
        `GET /v1/jobs/{id}` was already answering `unknown_job` about every one
        of them and will go on doing so.
        """
        root = Path(self._config.jobs_dir)
        if not root.is_dir():
            return []
        taken: list[Reaped] = []
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name in self._jobs:
                continue
            try:
                age = now.timestamp() - entry.stat().st_mtime
            except OSError as exc:
                print(
                    f"crucible: could not read the age of {entry}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                continue
            if age <= horizon:
                continue
            when = now.isoformat()
            because = (
                f"it belongs to no job this process is running, and it was "
                f"last written {age / 86_400.0:.1f} day(s) ago — past this "
                f"server's {self._config.retention_days}-day window "
                "([jobs] retention_days)"
            )
            try:
                shutil.rmtree(entry)
            except OSError as exc:
                print(
                    f"crucible: could not reap the orphan directory {entry}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                continue
            print(f"crucible: reaped {entry} — {because}", file=sys.stderr)
            taken.append(
                Reaped(
                    job_id=entry.name,
                    why="aged",
                    when=when,
                    detail=f"job {entry.name} was reaped at {when} because "
                    f"{because}",
                )
            )
        return taken

    # ------------------------------------------------------------------ events

    def append_event(self, job: Job, kind: str, data: dict[str, Any]) -> None:
        """Append one SSE event. Called on the event loop thread only."""
        event = {"id": len(job.events) + 1, "event": kind, "data": data}
        job.events.append(event)
        if kind == "progress":
            job.progress = float(data["fraction"])
            # The latest line, for the whole-server read. The event log stays
            # the truth; a `progress` without a message leaves the last one
            # standing rather than blanking it, because "rendering 118 of 280"
            # followed by an empty bench row reads as a stall.
            message = data.get("message")
            if isinstance(message, str) and message:
                job.message = message
        for waiter in self._subscribers.get(job.id, []):
            waiter.set()

    def record_artifact(self, job: Job, name: str) -> None:
        if name not in job.artifacts:
            job.artifacts.append(name)
        self.append_event(job, "artifact", {"name": name})

    def subscribe(self, job: Job) -> asyncio.Event:
        """One waiter per open event stream, so streams never steal each other's wakeup."""
        waiter = asyncio.Event()
        self._subscribers.setdefault(job.id, []).append(waiter)
        return waiter

    def unsubscribe(self, job: Job, waiter: asyncio.Event) -> None:
        waiters = self._subscribers.get(job.id)
        if waiters is None:
            return
        if waiter in waiters:
            waiters.remove(waiter)
        if not waiters:
            del self._subscribers[job.id]

    # -------------------------------------------------------------- provenance

    def provenance(self, job: Job, finished: str | None = None) -> dict[str, Any]:
        """DESIGN.md section 7. Written beside every artifact.

        The `model` block comes from the job type, because the job type is what
        knows its models: the queue has a model *id* and nothing else, and until
        this was fixed it wrote `revision: null` on every artifact Crucible had
        ever produced — a sidecar that named a model and then declined to say
        which one. A model-less job type answers None, and the sidecar says
        `model: null`, which is the honest shape for `echo`.
        """
        return {
            "server": {"name": self._config.name, "version": VERSION},
            "backend": self._backend.kind,
            "job_type": job.type,
            "model": self._registry[job.type].model_provenance(job.model),
            "params": job.params,
            "started": job.started,
            "finished": finished if finished is not None else utcnow(),
        }

    def _restamp_provenance(self, job: Job) -> None:
        """Rewrite each sidecar with the job's real finish time."""
        document = json.dumps(self.provenance(job, job.finished), indent=2) + "\n"
        for name in job.artifacts:
            sidecar = job.artifacts_dir / f"{name}.provenance.json"
            sidecar.write_text(document, encoding="utf-8")

    # ------------------------------------------------------------------ cancel

    def cancel(self, job: Job) -> str:
        if job.status in TERMINAL_STATES:
            raise ApiError(
                409,
                "job_not_cancellable",
                f"job {job.id} is already {job.status}",
            )
        job.cancel_requested = True
        if job.status == QUEUED:
            self._pending.remove(job.id)
            self._finish(job, CANCELLED)
            return CANCELLED
        # Running: cooperative. The job ends as cancelled when it next checks.
        return "cancelling"

    # -------------------------------------------------------------------- lane

    async def _run_lane(self) -> None:
        while True:
            if not self._pending:
                self._wake.clear()
                # THE STORE TICK, and it is here rather than in a second task
                # because there is exactly one thing in this server allowed to
                # decide a job is finished with, and it is the thing that runs
                # them. It happens with the lane IDLE — the first time on the
                # way into the wait, which is the startup sweep, and again
                # every time the wait times out — so a reaper never competes
                # with a render for the disk.
                try:
                    self.reap()
                except Exception as exc:
                    # THE LANE OUTLIVES THE REAPER. A cleanup failure is not an
                    # operation failure (`_settle` says the same about the
                    # card), and letting one out here would kill the coroutine
                    # that IS the lane: every later job would sit at `queued`
                    # for ever while the server went on answering 202. Said
                    # loudly, and the next tick tries again.
                    print(
                        f"crucible: the job reaper failed: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                # AND THE LEASE NOBODY CAME BACK FOR. A lease that is RELEASED
                # settles at the release; a lease that simply runs out is read
                # and never swept (`crucible/leases.py`), so until this line
                # nothing ran at the moment it lapsed and the card sat resident
                # and unheld. It belongs on this tick rather than on a clock of
                # its own for the reason the reap does: this is the thing that
                # already runs when the lane is idle, which is exactly when a
                # lapse goes unnoticed.
                await self._settle_lapsed_lease()
                try:
                    await asyncio.wait_for(
                        self._wake.wait(), timeout=REAP_INTERVAL_SECONDS
                    )
                except asyncio.TimeoutError:
                    # Nothing arrived; go round and reap again. The wait is
                    # still an event and not a poll — a submission wakes it in
                    # microseconds, and the timeout only exists so that a
                    # server nobody is using still cleans up after itself.
                    pass
                continue
            job_id = self._pending.popleft()
            job = self._jobs[job_id]
            try:
                if job.cancel_requested:
                    self._finish(job, CANCELLED)
                    continue
                await self._execute(job)
            except asyncio.CancelledError:
                # The server is shutting the lane down. Nothing to salvage.
                raise
            except Exception as exc:
                # `_execute` already turns anything the PLUGIN raises into a
                # failed job. Reaching here means the queue's own bookkeeping
                # raised — writing a provenance sidecar, appending an event —
                # and the damage of letting it out is out of all proportion to
                # the bug: this coroutine IS the lane, so an escape kills the
                # worker, every later job sits at `queued` forever, and the
                # server goes on answering 200 to submissions it will never run.
                #
                # That is not hypothetical. A job type written against a
                # six-method `JobType` met a seventh added in another branch;
                # the AttributeError surfaced inside `_finish`, the job emitted
                # no terminal event, and the SSE stream its client was reading
                # never ended. `build_registry` now refuses that mismatch at
                # startup, and this keeps any future one to a single failed job.
                self._fail_out_of_band(job, exc)

    async def _execute(self, job: Job) -> None:
        plugin = self._registry[job.type]
        job.status = RUNNING
        job.started = utcnow()
        self._running_id = job.id
        self.append_event(job, "progress", {"fraction": 0.0, "message": "started"})
        loop = asyncio.get_running_loop()
        ctx = JobContext(self, job, loop)
        status = DONE
        error: dict[str, str] | None = None
        try:
            await asyncio.to_thread(plugin.run, job, ctx)
        except JobCancelled:
            status = CANCELLED
        except JobError as exc:
            status, error = FAILED, {"code": exc.code, "message": exc.message}
        except Exception as exc:  # a plugin bug; surface it, never swallow it
            status = FAILED
            error = {"code": "job_failed", "message": f"{type(exc).__name__}: {exc}"}
        else:
            if job.cancel_requested:
                status = CANCELLED
        try:
            # OWEN'S RULING, 2026-09-14: this job is done with the card, so if
            # nothing else holds it the card is cleared NOW (crucible/settle.py).
            #
            # BEFORE THE TERMINAL EVENT AND WHILE THIS JOB STILL HOLDS THE LANE,
            # which is two properties in one placement. The note lands on a
            # stream the client is still reading — `_event_stream` returns at the
            # terminal event, so a line appended after `done` is a line nobody is
            # told. And `_running_id` is still set, so `refuse_if_busy` refuses a
            # submission that would otherwise race the unload; a job admitted
            # here waits behind this one instead of failing against a dying
            # engine.
            await self._settle(job, status)
        finally:
            self._finish(job, status, error)
            self._running_id = None

    async def _settle(self, job: Job, outcome: str) -> None:
        """Clear the card if this job was the last thing holding it.

        A CLEANUP FAILURE IS NOT AN OPERATION FAILURE. An engine that will not
        stop is a real fact and is said in both places a reader looks, but it
        does not rewrite the outcome of the render that finished: a book that
        rendered is a book that rendered, whatever happened to the card
        afterwards.

        `outcome` is carried in rather than read off the job because this runs
        BEFORE `_finish` — which is the whole point of where it sits — so
        `job.status` is still `running`. A load is exempt from settling only
        when it ended `done` (`crucible/settle.py`), and that is the one fact
        the settlement cannot get from the document in front of it.
        """
        if self._settlement is None:
            return
        try:
            settled = await asyncio.to_thread(
                self._settlement.settle_for_job, job, outcome
            )
        except Exception as exc:
            line = (
                f"could not clear the card after job {job.id} ({job.type}): "
                f"{type(exc).__name__}: {exc}"
            )
            print(f"crucible: {line}", file=sys.stderr)
            self.append_event(job, "note", {"message": line})
            return
        if settled is not None:
            self.append_event(job, "note", settled.to_dict())

    async def _settle_lapsed_lease(self) -> None:
        """Clear the card when a lease ran out and nobody heartbeated it.

        No job to hang a note on — that is the whole difficulty with a lapse and
        the reason `settle.py` logs every clearance it makes. A failure here is a
        cleanup failure and says so loudly without touching the lane, which is
        `settle_quietly`'s rule applied to the one trigger that has no caller.
        """
        if self._settlement is None:
            return
        try:
            await asyncio.to_thread(self._settlement.settle_for_lapsed_lease)
        except Exception as exc:
            # THE LANE OUTLIVES THIS, exactly as it outlives the reaper: letting
            # one out here would kill the coroutine that IS the lane, and every
            # later job would sit at `queued` for ever while the server went on
            # answering 202.
            print(
                f"crucible: could not clear the card after a lease lapsed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    def _fail_out_of_band(self, job: Job, exc: BaseException) -> None:
        """Mark a job failed when the queue's own machinery is what broke.

        Deliberately does not go through `_finish`: whatever raised is most
        likely still there (provenance, or the event append itself), and a
        second attempt down the same path would take the lane with it. So this
        sets the terminal state directly, appends the one event a client is
        waiting for, and treats a failure even to do that as survivable — the
        lane matters more than the message.
        """
        job.status = FAILED
        job.finished = utcnow()
        job.error = {
            "code": "queue_failed",
            "message": (
                f"the job lane could not finish this job: {type(exc).__name__}: "
                f"{exc}. This is a bug in Crucible, not in the request."
            ),
        }
        try:
            self.append_event(job, "failed", {"error": job.error})
        except Exception:
            # Nothing further can be told to the client, whose stream will end
            # when it disconnects. The lane lives, which is the point.
            pass
        self._running_id = None

    def _finish(self, job: Job, status: str, error: dict[str, str] | None = None) -> None:
        job.status = status
        job.finished = utcnow()
        job.error = error
        if status == DONE:
            job.progress = 1.0
            self._restamp_provenance(job)
            self.append_event(
                job, "done", {"artifacts": list(job.artifacts), **job.done_extra}
            )
        elif status == FAILED:
            self._restamp_provenance(job)
            self.append_event(job, "failed", {"error": error})
        else:
            self._restamp_provenance(job)
            self.append_event(job, "cancelled", {"status": CANCELLED})
