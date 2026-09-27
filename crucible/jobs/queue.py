from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import VERSION
from ..errors import ApiError, JobCancelled, JobError
from ..journal import Journals
from .base import (
    CANCELLED,
    DONE,
    FAILED,
    QUEUED,
    RUNNING,
    INTERRUPTED,
    TERMINAL_STATES,
    Job,
    JobContext,
    JobType,
    utcnow,
)

REAP_INTERVAL_SECONDS = 60.0


def _params_for_artifact(params: dict[str, Any], index: int | None) -> dict[str, Any]:
    if index is None:
        return params
    trimmed: dict[str, Any] = {}
    for key, value in params.items():
        if (
            isinstance(value, list)
            and value
            and all(isinstance(item, dict) and "index" in item for item in value)
        ):
            trimmed[key] = [item for item in value if item.get("index") == index]
        else:
            trimmed[key] = value
    return trimmed


def _params_sha256(params: dict[str, Any]) -> str:
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Reaped:
    job_id: str
    why: str
    when: str
    detail: str


def busy_details(job: Job) -> dict[str, Any]:
    return {
        "door": "job",
        "holder": job.client,
        "job_id": job.id,
        "type": job.type,
        "model": job.model,
        "status": job.status,
        "since": job.started if job.started is not None else job.created,
        "progress": job.progress,
        "message": job.message,
    }


class JobStore:
    def __init__(self, config: Any, backend: Any, registry: dict[str, JobType]) -> None:
        self._config = config
        self._backend = backend
        self._registry = registry
        self._jobs: dict[str, Job] = {}
        self._reaped: dict[str, Reaped] = {}
        self._consumed_blobs: dict[str, str] = {}
        self._pending: deque[str] = deque()
        self._wake = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._running_id: str | None = None
        self._lane_lock = threading.Lock()
        self._subscribers: dict[str, list[asyncio.Event]] = {}
        self._settlement: Any | None = None
        home = getattr(config, "home", None)
        self._journals = Journals(
            None if home is None else Path(home) / "journals",
            lambda: float(self._config.retention_days),
            self._live_state,
        )


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
            print(
                f"crucible: the job lane had already died: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        self._worker = None


    def attach_settlement(self, settlement: Any) -> None:
        self._settlement = settlement

    @property
    def registry(self) -> dict[str, JobType]:
        return self._registry

    @property
    def journals(self) -> Journals:
        return self._journals

    def _live_state(self, job_id: str) -> str | None:
        job = self._jobs.get(job_id)
        if job is None or job.status not in (QUEUED, RUNNING):
            return None
        return job.status

    @property
    def queue_depth(self) -> int:
        return len(self._pending) + (1 if self._running_id is not None else 0)

    @property
    def running_id(self) -> str | None:
        return self._running_id

    @property
    def running(self) -> Job | None:
        return None if self._running_id is None else self._jobs[self._running_id]

    def occupied_by_anything_but(self, job_id: str | None) -> Job | None:
        with self._lane_lock:
            running = self.running
            if running is not None and running.id != job_id:
                return running
            for pending_id in self._pending:
                if pending_id != job_id:
                    return self._jobs[pending_id]
            return None

    def queued(self) -> list[Job]:
        return [self._jobs[job_id] for job_id in self._pending]

    def get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is not None:
            return job
        reaped = self._reaped.get(job_id)
        if reaped is not None:
            raise ApiError(
                404,
                "job_reaped",
                reaped.detail,
                {"job_id": job_id, "reaped_at": reaped.when, "why": reaped.why},
            )
        raise ApiError(404, "unknown_job", f"no job {job_id} on this server")

    def position(self, job: Job) -> int | None:
        if job.status == RUNNING:
            return 0
        if job.status == QUEUED:
            return self._pending.index(job.id) + 1
        return None


    def refuse_if_busy(self) -> None:
        holder = self.running
        if holder is None:
            if not self._pending:
                return
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


    def create(
        self,
        job_type: str,
        model: str | None,
        params: dict[str, Any],
        client: str | None = None,
        client_ref: str | None = None,
        hold: bool = False,
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
            client_ref=client_ref,
            held_by=client if hold else None,
            held_since=utcnow() if hold else None,
        )
        self._jobs[job_id] = job
        self._persist(job)
        return job

    def enqueue(self, job: Job) -> None:
        self.refuse_if_busy()
        with self._lane_lock:
            self._pending.append(job.id)
        self.append_event(job, "queued", {"position": self.position(job)})
        self._wake.set()

    def discard(self, job: Job) -> None:
        if job.id in self._pending:
            raise RuntimeError(
                f"job {job.id} is on the lane and cannot be discarded; cancel it"
            )
        self._jobs.pop(job.id, None)
        shutil.rmtree(job.dir, ignore_errors=True)


    def refuse_if_blob_consumed(self, blob_id: str) -> None:
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
        self.refuse_if_blob_consumed(blob_id)
        self._consumed_blobs[blob_id] = job.id


    def hold(self, job: Job, client: str | None) -> dict[str, Any]:
        if job.status in TERMINAL_STATES and not job.artifacts:
            raise ApiError(
                409,
                "nothing_to_hold",
                f"job {job.id} ended {job.status} and published no artifacts, so "
                "there is nothing a later job could take from it",
                {"job_id": job.id, "status": job.status},
            )
        if job.held_since is None:
            job.held_since = utcnow()
        job.held_by = client
        self._persist(job)
        return self.hold_record(job)

    def hold_record(self, job: Job) -> dict[str, Any]:
        horizon = float(self._config.retention_days) * 86_400.0
        gc_at: str | None = None
        if job.finished is not None:
            finished = datetime.fromisoformat(str(job.finished))
            gc_at = datetime.fromtimestamp(
                finished.timestamp() + horizon, tz=timezone.utc
            ).isoformat()
        return {
            "job_id": job.id,
            "status": job.status,
            "held": job.held,
            "held_by": job.held_by,
            "held_since": job.held_since,
            "gc_at": gc_at,
            "artifacts": list(job.artifacts),
        }

    def release(self, job: Job) -> bool:
        was = job.held
        job.held_by = None
        job.held_since = None
        self._persist(job)
        if job.status in TERMINAL_STATES:
            self._reap_one(
                job,
                "released",
                _now(),
                "its hold was released: the chain that held it is complete",
            )
        return was

    def mark_fetched(self, job: Job, name: str) -> None:
        job.fetched.add(name)

    def reap(self) -> list[Reaped]:
        now = _now()
        horizon = float(self._config.retention_days) * 86_400.0
        taken: list[Reaped] = []
        for job in list(self._jobs.values()):
            if job.status not in TERMINAL_STATES:
                continue
            if job.held and self._age_seconds(job, now) <= horizon:
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
        try:
            self._journals.reap(now)
        except Exception as exc:
            print(
                f"crucible: the journal reaper failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        return taken

    def _age_seconds(self, job: Job, now: datetime) -> float:
        if job.finished is None:
            raise RuntimeError(
                f"job {job.id} is {job.status} with no `finished` stamp; the "
                "reaper cannot say how old a job that never recorded an end is"
            )
        return (now - datetime.fromisoformat(job.finished)).total_seconds()

    def _reap_one(
        self, job: Job, why: str, now: datetime, because: str
    ) -> Reaped | None:
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


    def append_event(self, job: Job, kind: str, data: dict[str, Any]) -> None:
        event = {"id": len(job.events) + 1, "event": kind, "data": data}
        job.events.append(event)
        if kind == "progress":
            job.progress = float(data["fraction"])
            message = data.get("message")
            if isinstance(message, str) and message:
                job.message = message
        for waiter in self._subscribers.get(job.id, []):
            waiter.set()

    def attach_journal(self, job: Job, resume_id: str, *, resumed: bool) -> None:
        job.resume_id = resume_id
        job.resumed = resumed
        self._persist(job)

    def record_chunks_total(self, job: Job, total: int) -> None:
        job.chunks_total = int(total)
        self._persist(job)

    def record_artifact(self, job: Job, name: str,
                        index: int | None = None) -> None:
        if name not in job.artifacts:
            job.artifacts.append(name)
        if index is not None:
            job.artifact_index[name] = index
            if index not in job.chunks_done:
                job.chunks_done.append(index)
            job.chunk_at = utcnow()
        self.append_event(job, "artifact", {"name": name})
        self._persist(job)

    def subscribe(self, job: Job) -> asyncio.Event:
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


    def restore(self) -> list[str]:
        root = Path(self._config.jobs_dir)
        if not root.is_dir():
            return []
        recovered: list[str] = []
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name in self._jobs:
                continue
            path = entry / self.RECORD_NAME
            if not path.is_file():
                continue
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(
                    f"crucible: could not read the record in {entry}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                continue
            job = self._job_from_record(entry, document)
            if job is None:
                continue
            self._jobs[job.id] = job
            recovered.append(job.id)
            if job.resume_id is not None and job.status == INTERRUPTED:
                self._journals.ended(job.resume_id, job.id, INTERRUPTED)
        if recovered:
            print(
                f"crucible: recovered {len(recovered)} job(s) from disk; "
                f"{sum(1 for i in recovered if self._jobs[i].status == INTERRUPTED)}"
                " were interrupted by a restart",
                file=sys.stderr,
            )
        return recovered

    def _job_from_record(self, directory: Path, document: Any) -> Job | None:
        if not isinstance(document, dict) or not document.get("job_id"):
            return None
        status = str(document.get("status") or QUEUED)
        interrupted_at = document.get("interrupted_at")
        if status not in TERMINAL_STATES:
            status = INTERRUPTED
            interrupted_at = interrupted_at or utcnow()
        job = Job(
            id=str(document["job_id"]),
            type=str(document.get("type") or ""),
            model=document.get("model"),
            params={},
            dir=directory,
            created=str(document.get("created") or utcnow()),
            status=status,
            progress=float(document.get("progress") or 0.0),
            started=document.get("started"),
            finished=document.get("finished"),
            error=document.get("error"),
            artifacts=list(document.get("artifacts") or []),
            client=document.get("client"),
            client_ref=document.get("client_ref"),
            interrupted_at=interrupted_at,
            chunks_done=[int(i) for i in (document.get("chunks_done") or [])],
            chunks_total=(
                None if document.get("chunks_total") is None
                else int(document["chunks_total"])
            ),
            chunk_at=document.get("chunk_at"),
            done_extra=dict(document.get("done_extra") or {}),
            held_by=document.get("held_by"),
            held_since=document.get("held_since"),
            resume_id=document.get("resume_id"),
            resumed=bool(document.get("resumed")),
        )
        return job


    RECORD_NAME = "job.json"

    def _persist(self, job: Job) -> None:
        try:
            job.dir.mkdir(parents=True, exist_ok=True)
            document = json.dumps(self._record_of(job), indent=2) + chr(10)
            path = job.dir / self.RECORD_NAME
            temporary = path.with_suffix(".json.writing")
            temporary.write_text(document, encoding="utf-8")
            os.replace(temporary, path)
        except Exception as exc:
            print(
                f"crucible: could not record job {job.id} on disk: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    def _record_of(self, job: Job) -> dict[str, Any]:
        return {
            "job_id": job.id,
            "type": job.type,
            "model": job.model,
            "status": job.status,
            "progress": job.progress,
            "error": job.error,
            "artifacts": list(job.artifacts),
            "chunks_done": sorted(job.chunks_done),
            "chunks_total": job.chunks_total,
            "chunk_at": job.chunk_at,
            "created": job.created,
            "started": job.started,
            "finished": job.finished,
            "interrupted_at": job.interrupted_at,
            "client": job.client,
            "client_ref": job.client_ref,
            "done_extra": job.done_extra,
            "held_by": job.held_by,
            "held_since": job.held_since,
            "resume_id": job.resume_id,
            "resumed": job.resumed,
        }

    def provenance(
        self, job: Job, finished: str | None = None, index: int | None = None
    ) -> dict[str, Any]:
        return {
            "server": {"name": self._config.name, "version": VERSION},
            "backend": self._backend.kind,
            "job_type": job.type,
            "job_id": job.id,
            "model": self._registry[job.type].model_provenance(job.model),
            "params": _params_for_artifact(job.params, index),
            "params_sha256": _params_sha256(job.params),
            "started": job.started,
            "finished": finished if finished is not None else utcnow(),
        }

    def _restamp_provenance(self, job: Job) -> None:
        for name in job.artifacts:
            sidecar = job.artifacts_dir / f"{name}.provenance.json"
            document = self.provenance(
                job, job.finished, index=job.artifact_index.get(name)
            )
            sidecar.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


    def cancel(self, job: Job) -> str:
        if job.status in TERMINAL_STATES:
            raise ApiError(
                409,
                "job_not_cancellable",
                f"job {job.id} is already {job.status}",
            )
        job.cancel_requested = True
        if job.status == QUEUED:
            with self._lane_lock:
                self._pending.remove(job.id)
            self._finish(job, CANCELLED)
            return CANCELLED
        return "cancelling"


    async def _run_lane(self) -> None:
        while True:
            if not self._pending:
                self._wake.clear()
                try:
                    self.reap()
                except Exception as exc:
                    print(
                        f"crucible: the job reaper failed: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                await self._settle_lapsed_lease()
                try:
                    await asyncio.wait_for(
                        self._wake.wait(), timeout=REAP_INTERVAL_SECONDS
                    )
                except asyncio.TimeoutError:
                    pass
                continue
            with self._lane_lock:
                job_id = self._pending.popleft()
                job = self._jobs[job_id]
                cancelled = job.cancel_requested
                if not cancelled:
                    self._running_id = job_id
            try:
                if cancelled:
                    self._finish(job, CANCELLED)
                    continue
                await self._execute(job)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._fail_out_of_band(job, exc)

    async def _execute(self, job: Job) -> None:
        plugin = self._registry[job.type]
        job.status = RUNNING
        self._persist(job)
        job.started = utcnow()
        self.append_event(job, "progress", {"fraction": 0.0, "message": "started"})
        if job.resume_id is not None:
            self._journal_started(job)
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
        except Exception as exc:
            status = FAILED
            error = {"code": "job_failed", "message": f"{type(exc).__name__}: {exc}"}
        else:
            if job.cancel_requested:
                status = CANCELLED
        try:
            await self._settle(job, status)
        finally:
            self._finish(job, status, error)
            self._running_id = None

    def _journal_started(self, job: Job) -> None:
        assert job.resume_id is not None
        try:
            journal = self._journals.open(job.resume_id)
            before = journal.manifest
            journal.set_writer(job.id, RUNNING, resumed=job.resumed)
        except Exception as exc:
            self.append_event(
                job,
                "note",
                {"message": f"could not open journal {job.resume_id}: "
                            f"{type(exc).__name__}: {exc}"},
            )
            return
        if not job.resumed:
            return
        done = before.get("units_done") or 0
        total = before.get("units_total")
        sentence = before.get("progress") or f"{done:,} unit(s) done"
        previous = [row.get("job_id") for row in before.get("jobs") or []]
        self.append_event(
            job,
            "note",
            {
                "message": f"resumed: {sentence}, from journal {job.resume_id} "
                f"(last saved {before.get('last_saved')})",
                "resume_id": job.resume_id,
                "resumed_from": previous[-2] if len(previous) >= 2 else None,
                "units_done": done,
                "units_total": total,
            },
        )

    async def _settle(self, job: Job, outcome: str) -> None:
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
        if self._settlement is None:
            return
        try:
            await asyncio.to_thread(self._settlement.settle_for_lapsed_lease)
        except Exception as exc:
            print(
                f"crucible: could not clear the card after a lease lapsed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    def _fail_out_of_band(self, job: Job, exc: BaseException) -> None:
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
            pass
        if job.resume_id is not None:
            self._journals.ended(job.resume_id, job.id, FAILED)
        self._running_id = None

    def _finish(self, job: Job, status: str, error: dict[str, str] | None = None) -> None:
        job.status = status
        job.finished = utcnow()
        job.error = error
        self._persist(job)
        if job.resume_id is not None:
            self._journals.ended(job.resume_id, job.id, status)
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
