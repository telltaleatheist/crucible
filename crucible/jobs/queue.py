"""The job store and the single exclusive lane that drains it.

The server owns the accelerator (DESIGN.md section 6): one job runs at a time, in
submission order, on one worker task. Clients never see a lock; they see `position`
and the event stream.
"""

from __future__ import annotations

import asyncio
import json
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

    @property
    def registry(self) -> dict[str, JobType]:
        return self._registry

    @property
    def queue_depth(self) -> int:
        return len(self._pending) + (1 if self._running_id is not None else 0)

    @property
    def running_id(self) -> str | None:
        return self._running_id

    @property
    def running(self) -> Job | None:
        """The job on the lane right now, or None."""
        return None if self._running_id is None else self._jobs[self._running_id]

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
        """0 while running, 1-based place in line while queued, null once terminal."""
        if job.status == RUNNING:
            return 0
        if job.status == QUEUED:
            return self._pending.index(job.id) + 1
        return None

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
        self._pending.append(job.id)
        self.append_event(job, "queued", {"position": self.position(job)})
        self._wake.set()

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
        try:
            await asyncio.to_thread(plugin.run, job, ctx)
        except JobCancelled:
            self._finish(job, CANCELLED)
        except JobError as exc:
            self._finish(job, FAILED, {"code": exc.code, "message": exc.message})
        except Exception as exc:  # a plugin bug; surface it, never swallow it
            self._finish(
                job,
                FAILED,
                {"code": "job_failed", "message": f"{type(exc).__name__}: {exc}"},
            )
        else:
            if job.cancel_requested:
                self._finish(job, CANCELLED)
            else:
                self._finish(job, DONE)
        finally:
            self._running_id = None

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
