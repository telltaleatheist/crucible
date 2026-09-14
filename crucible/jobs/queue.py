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


class JobStore:
    """Holds every job this server has seen in this process, plus the run lane."""

    def __init__(self, config: Any, backend: Any, registry: dict[str, JobType]) -> None:
        self._config = config
        self._backend = backend
        self._registry = registry
        self._jobs: dict[str, Job] = {}
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
        if job is None:
            raise ApiError(404, "unknown_job", f"no job {job_id} on this server")
        return job

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
        # `started` for a running job, `created` for one the lane has not reached:
        # both answer "since when", and a null `started` reported as `since` would
        # read as "it has been busy since never".
        since = holder.started if holder.started is not None else holder.created
        doing = "" if not holder.message else f" — {holder.message}"
        raise ApiError(
            409,
            "server_busy",
            f"this server is busy with job {holder.id} ({what}), {holder.status} "
            f"since {since}, submitted by {who}, {holder.progress:.0%} done"
            f"{doing}. Crucible admits one job at a time and does not queue: the "
            "client owns the queue, the server owns admission (ARCHITECTURE.md "
            "section 3). Read GET /v1/activity to see when it is finished.",
            {
                "holder": holder.client,
                "job_id": holder.id,
                "type": holder.type,
                "model": holder.model,
                # Named so `since` is unambiguous: "running" dates from `started`,
                # "queued" from `created`. Without it a client cannot tell a job
                # that has been rendering for an hour from one admitted 2 ms ago.
                "status": holder.status,
                "since": since,
                "progress": holder.progress,
                # The holder's latest progress line, which is what turns "busy"
                # into "rendering 118 of 280" on somebody else's bench. Null until
                # the job has said anything.
                "message": holder.message,
            },
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
                await self._wake.wait()
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
            await self._settle(job)
        finally:
            self._finish(job, status, error)
            self._running_id = None

    async def _settle(self, job: Job) -> None:
        """Clear the card if this job was the last thing holding it.

        A CLEANUP FAILURE IS NOT AN OPERATION FAILURE. An engine that will not
        stop is a real fact and is said in both places a reader looks, but it
        does not rewrite the outcome of the render that finished: a book that
        rendered is a book that rendered, whatever happened to the card
        afterwards.
        """
        if self._settlement is None:
            return
        try:
            settled = await asyncio.to_thread(self._settlement.settle_for_job, job)
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
