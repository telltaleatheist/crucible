"""The job store and the single exclusive lane that drains it.

The server owns the accelerator (DESIGN.md section 6): one job runs at a time, in
submission order, on one worker task. Clients never see a lock; they see `position`
and the event stream.
"""

from __future__ import annotations

import asyncio
import json
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

    def create(self, job_type: str, model: str | None, params: dict[str, Any]) -> Job:
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
            if job.cancel_requested:
                self._finish(job, CANCELLED)
                continue
            await self._execute(job)

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
