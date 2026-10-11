from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .. import VERSION, clock
from ..clock import utcnow
from ..errors import ApiError, JobCancelled, JobError
from ..events import JOB, EventHub
from ..journal import JOURNALS_DIRNAME, Journals
from ..procgroup import STOP_TIMEOUT_SECONDS, stop_budget_seconds
from ..updating import UpdateHold
from .base import (
    CANCELLED,
    DONE,
    FAILED,
    INTERRUPTED,
    QUEUED,
    REMOVED,
    RUNNING,
    TERMINAL_STATES,
    Job,
    JobContext,
    JobFailure,
    JobType,
)

REAP_INTERVAL_SECONDS = 60.0

SECONDS_PER_DAY = 86_400.0

# The done-event key in which a job type reports what is resident on the card
# (audio, image, segment, video, align, denoise, the load and unload types).
RESIDENT_KEY = "resident"

# The events whose message becomes the job's `message` and its `job.progress`. A load
# says what it is doing with `warming` (the accelerator check, then the engine starting,
# every 2 s), and that is the job's whole progress until it is resident: when only its
# own stream carried it, a client reading the job saw "loading <model>" for the minutes
# a first load spends compiling, and took it for hung (Victoria's laptop, 2026-10-09).
SAYS_WHAT_IT_IS_DOING = frozenset({"progress", "warming"})


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


def _removed_by_restart(waiting: Any, at: str) -> dict[str, Any]:
    submitted = waiting.get("submitted") if isinstance(waiting, dict) else None
    waited: float | None = None
    if isinstance(submitted, str):
        try:
            waited = round(
                (datetime.fromisoformat(at) - datetime.fromisoformat(submitted))
                .total_seconds(), 3,
            )
        except ValueError:
            waited = None
    return {
        "reason": "server_restart",
        "message": "the server restarted while this job waited in its queue; a "
        "queued job is not run hours later by a server that has forgotten who "
        "asked for it. Submit it again",
        "waited_s": waited,
        "at": at,
    }


def _job_change(job: Job, kind: str, data: dict[str, Any]) -> dict[str, Any]:
    change: dict[str, Any] = {
        "job_id": job.id,
        "type": job.type,
        "model": job.model,
        "client": job.client,
        "client_ref": job.client_ref,
        "status": job.status,
    }
    if job.status == QUEUED:
        change["position"] = data["position"] if kind == "queued" else None
        change["waiting"] = job.waiting is not None
    elif job.status == RUNNING:
        change["started"] = job.started
    elif job.status == DONE:
        change["artifacts"] = list(job.artifacts)
    elif job.status == FAILED:
        change["error"] = job.error
    elif job.status == REMOVED:
        change["removal"] = job.removal
    elif job.status == INTERRUPTED:
        change["interrupted_at"] = job.interrupted_at
    return change


class ReapReason(str, Enum):
    FETCHED = "fetched"
    AGED = "aged"
    RELEASED = "released"


@dataclass(frozen=True)
class Reaped:
    job_id: str
    why: ReapReason
    when: str
    detail: str


@dataclass(frozen=True)
class HoldRecord:
    job_id: str
    status: str
    held: bool
    held_by: str | None
    held_since: str | None
    gc_at: str | None
    artifacts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "held": self.held,
            "held_by": self.held_by,
            "held_since": self.held_since,
            "gc_at": self.gc_at,
            "artifacts": list(self.artifacts),
        }


@dataclass(frozen=True)
class JournalProgress:
    units_done: int
    units_total: int | None
    sentence: str
    last_saved: Any
    writers: tuple[Any, ...]

    @classmethod
    def of(cls, manifest: dict[str, Any]) -> "JournalProgress":
        done = manifest.get("units_done") or 0
        return cls(
            units_done=done,
            units_total=manifest.get("units_total"),
            sentence=manifest.get("progress") or f"{done:,} unit(s) done",
            last_saved=manifest.get("last_saved"),
            writers=tuple(row.get("job_id") for row in manifest.get("jobs") or []),
        )

    @property
    def resumed_from(self) -> Any:
        return self.writers[-2] if len(self.writers) >= 2 else None


class LaneSlot:
    def __init__(self) -> None:
        self.job_id: str | None = None

    def admit(self, job_id: str) -> None:
        if self.job_id is not None and self.job_id != job_id:
            raise RuntimeError(
                f"the lane already holds job {self.job_id}; it admits one job at a time"
            )
        self.job_id = job_id

    def release(self, job_id: str) -> None:
        if self.job_id != job_id:
            raise ValueError(f"job {job_id} is not the one the lane holds")
        self.job_id = None

    def take(self) -> str | None:
        job_id, self.job_id = self.job_id, None
        return job_id


class JobStore:
    def __init__(self, config: Any, backend: Any, registry: dict[str, JobType]) -> None:
        self._config = config
        self._backend = backend
        self._registry = registry
        self._jobs: dict[str, Job] = {}
        self._reaped: dict[str, Reaped] = {}
        self._consumed_blobs: dict[str, str] = {}
        self._admitted = LaneSlot()
        self._wake = asyncio.Event()
        # The deploy's hold (crucible/updating.py); the app hands every work owner the same one.
        self.updating = UpdateHold()
        self._worker: asyncio.Task[None] | None = None
        self._running_id: str | None = None
        self._lane_lock = threading.Lock()
        self._subscribers: dict[str, list[asyncio.Event]] = {}
        self._settlement: Any | None = None
        self._stopping = False
        self._interrupted_by_stop: set[str] = set()
        self._lane_idle = asyncio.Event()
        self._lane_idle.set()
        self._line: Any | None = None
        self._on_idle: Any = lambda: None
        self.events = EventHub()
        self._announced: dict[str, str] = {}
        home = getattr(config, "home", None)
        self._journals = Journals(
            None if home is None else Path(home) / JOURNALS_DIRNAME,
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
        with self._lane_lock:
            self._stopping = True
        await self._stop_the_running_job()
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


    async def _stop_the_running_job(self) -> None:
        job = self.running
        if job is None or self._worker is None or self._worker.done():
            return
        self._interrupted_by_stop.add(job.id)
        job.cancel_requested = True
        budget = stop_budget_seconds(STOP_TIMEOUT_SECONDS)
        idle = asyncio.ensure_future(self._lane_idle.wait())
        try:
            await asyncio.wait(
                {idle, self._worker}, timeout=budget,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            idle.cancel()
        if not self._lane_idle.is_set() and not self._worker.done():
            print(
                f"crucible: job {job.id} ({job.type}) did not stop within "
                f"{budget:.0f}s of the server asking it to; the server stops its "
                "engines next, and any engine that will not stop is named there "
                f"with its pid and log. The job's files are in {job.dir}",
                file=sys.stderr,
            )

    def attach_settlement(self, settlement: Any) -> None:
        self._settlement = settlement

    def attach_line(self, line: Any) -> None:
        self._line = line

    @property
    def line(self) -> Any | None:
        return self._line

    def when_idle(self, callback: Any) -> None:
        self._on_idle = callback

    def _idle(self) -> None:
        self._lane_idle.set()
        try:
            self._on_idle()
        except Exception as exc:
            print(
                f"crucible: could not wake the queue: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    @property
    def lane_free(self) -> bool:
        return self._running_id is None and self._admitted.job_id is None

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
    def admitted(self) -> LaneSlot:
        return self._admitted

    def _retention_seconds(self) -> float:
        return float(self._config.retention_days) * SECONDS_PER_DAY

    @property
    def queue_depth(self) -> int:
        on_lane = int(self._admitted.job_id is not None or self._running_id is not None)
        return on_lane + (0 if self._line is None else len(self._line))

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
            admitted = self._admitted.job_id
            if admitted is not None and admitted != job_id:
                return self._jobs[admitted]
            return None

    def queued(self, *, calls: bool = False) -> list[Any]:
        admitted = self._admitted.job_id
        waiting = [] if self._line is None else [
            w.job for w in self._line.ordered()
            if calls or not (w.is_call or w.is_session)
        ]
        return ([] if admitted is None else [self._jobs[admitted]]) + waiting

    def followed(self) -> tuple[set[str], set[str | None]]:
        jobs = {job_id for job_id, waiters in self._subscribers.items() if waiters}
        clients = {self._jobs[job_id].client for job_id in jobs if job_id in self._jobs}
        return jobs, clients

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
                {"job_id": job_id, "reaped_at": reaped.when, "why": reaped.why.value},
            )
        raise ApiError(404, "unknown_job", f"no job {job_id} on this server")

    def position(self, job: Job) -> int | None:
        if job.status == RUNNING:
            return 0
        if job.waiting is not None and self._line is not None:
            return self._line.position(job.id)
        if job.status == QUEUED and job.id == self._admitted.job_id:
            return 1
        return None


    def refuse_if_busy(self) -> None:
        holder = self.running
        if holder is None:
            if self._admitted.job_id is None:
                return
            holder = self._jobs[self._admitted.job_id]

        who = "an unnamed client" if holder.client is None else repr(holder.client)
        what = holder.type if holder.model is None else f"{holder.type} {holder.model!r}"
        busy = holder.busy()
        doing = "" if not holder.message else f" — {holder.message}"
        raise ApiError(
            409,
            "server_busy",
            f"this server is busy with job {holder.id} ({what}), {holder.status} "
            f"since {busy.since}, submitted by {who}, "
            f"{holder.progress:.0%} done"
            f"{doing}. Crucible admits one job at a time. Leave out "
            '"queue": false to wait in the queue on this server instead of being '
            "refused (docs/QUEUE.md), or read GET /v1/activity to see when it "
            "is finished.",
            busy.to_dict(),
        )


    def create(
        self,
        job_type: str,
        model: str | None,
        params: dict[str, Any],
        client: str | None = None,
        client_ref: str | None = None,
        hold: bool = False,
        session: str | None = None,
    ) -> Job:
        self.updating.refuse_if_holding(f"a {job_type} job")
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
            session=session,
        )
        self._jobs[job_id] = job
        self._persist(job)
        return job

    def enqueue(self, job: Job, *, announce: bool = True) -> None:
        self.refuse_if_busy()
        with self._lane_lock:
            self._admitted.admit(job.id)
        if announce:
            self.append_event(job, "queued", {"position": self.position(job)})
        self._wake.set()

    def mark_waiting(self, job: Job, waiting: dict[str, Any] | None) -> None:
        job.waiting = waiting
        self._persist(job)

    def end_waiting(
        self,
        item: Any,
        *,
        failure: JobFailure | None = None,
        removal: dict[str, Any] | None = None,
    ) -> None:
        job = item.job
        if item.fresh_journal is not None:
            self._journals.forget_new(item.fresh_journal)
            job.resume_id = None
        if failure is not None:
            self._finish(job, FAILED, failure)
            return
        assert removal is not None
        job.status = REMOVED
        job.finished = utcnow()
        job.removal = {**removal, "at": job.finished}
        self._persist(job)
        if job.resume_id is not None:
            self._journals.ended(job.resume_id, job.id, REMOVED)
        shutil.rmtree(job.inputs_dir, ignore_errors=True)
        self.append_event(job, "removed", job.removal)

    def discard(self, job: Job) -> None:
        if job.id == self._admitted.job_id:
            raise RuntimeError(
                f"job {job.id} is on the lane and cannot be discarded; cancel it"
            )
        self._jobs.pop(job.id, None)
        self._announced.pop(job.id, None)
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
        gc_at: str | None = None
        if job.finished is not None:
            finished = datetime.fromisoformat(str(job.finished))
            gc_at = datetime.fromtimestamp(
                finished.timestamp() + self._retention_seconds(), tz=timezone.utc
            ).isoformat()
        return HoldRecord(
            job_id=job.id,
            status=job.status,
            held=job.held,
            held_by=job.held_by,
            held_since=job.held_since,
            gc_at=gc_at,
            artifacts=tuple(job.artifacts),
        ).to_dict()

    def release(self, job: Job) -> bool:
        was = job.held
        job.held_by = None
        job.held_since = None
        self._persist(job)
        if job.status in TERMINAL_STATES:
            self._reap_one(
                job,
                ReapReason.RELEASED,
                clock.now(),
                "its hold was released: the chain that held it is complete",
            )
        return was

    def mark_fetched(self, job: Job, name: str) -> None:
        job.fetched.add(name)

    def reap(self) -> list[Reaped]:
        now = clock.now()
        horizon = self._retention_seconds()
        taken: list[Reaped] = []
        for job in list(self._jobs.values()):
            try:
                record = self._reap_if_due(job, now, horizon)
            except Exception as exc:
                print(
                    f"crucible: the reaper skipped job {job.id} ({job.dir}) and "
                    f"went on with the rest: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
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

    def _reap_if_due(self, job: Job, now: datetime, horizon: float) -> Reaped | None:
        if job.status not in TERMINAL_STATES:
            return None
        if job.held and self._age_seconds(job, now) <= horizon:
            return None
        if job.collected:
            return self._reap_one(
                job,
                ReapReason.FETCHED,
                now,
                f"its {len(job.artifacts)} artifact(s) and their sidecars "
                "had all been fetched",
            )
        if self._age_seconds(job, now) > horizon:
            return self._reap_one(
                job,
                ReapReason.AGED,
                now,
                f"it finished at {job.finished} and this server keeps a "
                f"finished job for {self._config.retention_days} day(s) "
                "([jobs] retention_days)",
            )
        return None

    def _age_seconds(self, job: Job, now: datetime) -> float:
        if job.finished is None:
            raise RuntimeError(
                f"job {job.id} is {job.status} with no `finished` stamp; the "
                "reaper cannot say how old a job that never recorded an end is"
            )
        return (now - datetime.fromisoformat(job.finished)).total_seconds()

    def _reap_one(
        self, job: Job, why: ReapReason, now: datetime, because: str
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
                f"crucible: could not reap job {job.id} ({why.value}) at {job.dir}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return None
        del self._jobs[job.id]
        self._announced.pop(job.id, None)
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
                f"last written {age / SECONDS_PER_DAY:.1f} day(s) ago — past this "
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
                    why=ReapReason.AGED,
                    when=when,
                    detail=f"job {entry.name} was reaped at {when} because "
                    f"{because}",
                )
            )
        return taken


    def append_event(self, job: Job, kind: str, data: dict[str, Any]) -> None:
        event = {"id": len(job.events) + 1, "event": kind, "data": data}
        job.events.append(event)
        if kind in SAYS_WHAT_IT_IS_DOING:
            if kind == "progress":
                job.progress = float(data["fraction"])
            message = data.get("message")
            if isinstance(message, str) and message:
                job.message = message
        for waiter in self._subscribers.get(job.id, []):
            waiter.set()
        self._publish(job, kind, data)

    def _publish(self, job: Job, kind: str, data: dict[str, Any]) -> None:
        """Tell the server-wide stream: `job.<status>` once each time the job's status
        changes, and `job.progress`, throttled, while it runs (docs/EVENTS.md)."""
        key = f"job:{job.id}"
        if self._announced.get(job.id) != job.status:
            self._announced[job.id] = job.status
            if job.status in TERMINAL_STATES:
                self.events.forget(key)
            self.events.publish(JOB, f"job.{job.status}", _job_change(job, kind, data))
        elif kind in SAYS_WHAT_IT_IS_DOING and job.status == RUNNING:
            self.events.publish_throttled(JOB, "job.progress", key, {
                "job_id": job.id, "fraction": job.progress, "message": job.message,
            })

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
            self._read_request(job)
            if (document.get("status"), document.get("finished")) != (
                job.status, job.finished
            ):
                self._persist(job)
            recovered.append(job.id)
            if job.resume_id is not None and job.status == INTERRUPTED:
                self._journals.ended(job.resume_id, job.id, INTERRUPTED)
            if document.get("waiting") and job.status == REMOVED:
                shutil.rmtree(job.inputs_dir, ignore_errors=True)
                if job.resume_id is not None:
                    self._journals.ended(job.resume_id, job.id, REMOVED)
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
        finished = document.get("finished")
        removal = document.get("removal")
        if status not in TERMINAL_STATES and document.get("waiting"):
            status, finished = REMOVED, utcnow()
            removal = _removed_by_restart(document["waiting"], finished)
        if status not in TERMINAL_STATES:
            status = INTERRUPTED
        if status == INTERRUPTED:
            interrupted_at = interrupted_at or finished or utcnow()
            finished = finished or interrupted_at
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
            finished=finished,
            failure=JobFailure.from_dict(document.get("error")),
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
            removal=removal if isinstance(removal, dict) else None,
            session=document.get("session"),
        )
        self._replay_ending(job)
        return job

    def _replay_ending(self, job: Job) -> None:
        """Give a job read back after a restart the ending its stream had.

        Events live in memory, so a restart lost every recovered job's history. Its
        stream then sent nothing and never closed: an SDK client following it (and
        reconnecting with Last-Event-ID) waited forever for a job the record says
        failed (B-Side, 2026-10-04: a song killed by a deploy). So the ending is
        rebuilt from the record exactly as `_finish` wrote it live - `done` with its
        artifacts and done_extra (what readAudioResult and the like read), `failed`
        with its error, `cancelled`, `removed` - and an interrupted job gets the note
        it got live. The events are final either way: the stream ends once sent.
        """
        if job.status == DONE:
            self.append_event(job, "done", {"artifacts": list(job.artifacts), **job.done_extra})
        elif job.status == FAILED:
            self.append_event(job, "failed", {"error": job.error})
        elif job.status == CANCELLED:
            self.append_event(job, "cancelled", {"status": CANCELLED})
        elif job.status == REMOVED and job.removal is not None:
            self.append_event(job, "removed", job.removal)
        elif job.status == INTERRUPTED:
            self.append_event(
                job,
                "note",
                {"message": f"the server stopped while job {job.id} ran, so it "
                            "ended interrupted. Collect what landed from "
                            f"GET /v1/jobs/{job.id} and submit the rest again."},
            )
        job.events_final = True


    RECORD_NAME = "job.json"

    # What a job type keeps of its request beside the record (keep_request): today the
    # audio type, whose params and chosen seed are what reproduce a song that failed.
    REQUEST_NAME = "request.json"

    def keep_request(self, job: Job, document: dict[str, Any]) -> None:
        """Write what `job` runs with to `request.json` in its directory, and hold it on
        the job (GET /v1/jobs/{id} `request`).

        It lives until the job ends `done` - the done event's own record says the same
        then (an audio job's `done_extra.audio`), so `_finish` drops it - or until the
        job's directory is reaped. A job that ends failed, cancelled or interrupted keeps
        it: that is the job it exists for (Victoria's 1f3da14c, 2026-10-10: a song OOM'd
        in synthesizing and nothing on disk said its seed). `job.json` still never carries
        `params`; a type that does not call this keeps nothing of its request on disk.
        Called on the job's thread before its work starts, and written through before it
        returns; a write that fails is said on stderr and the job goes on, as `_persist`.
        """
        job.request = document
        try:
            job.dir.mkdir(parents=True, exist_ok=True)
            path = job.dir / self.REQUEST_NAME
            temporary = path.with_suffix(".json.writing")
            temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, path)
        except Exception as exc:
            print(
                f"crucible: could not keep job {job.id}'s request on disk: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    def _drop_request(self, job: Job) -> None:
        job.request = None
        try:
            (job.dir / self.REQUEST_NAME).unlink(missing_ok=True)
        except OSError as exc:
            print(
                f"crucible: could not remove job {job.id}'s kept request "
                f"({job.dir / self.REQUEST_NAME}): {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    def _read_request(self, job: Job) -> None:
        """A recovered job takes back the request it kept; a `done` job's is removed, the
        drop a stop between `_finish`'s record and its removal never reached."""
        path = job.dir / self.REQUEST_NAME
        if not path.is_file():
            return
        if job.status == DONE:
            self._drop_request(job)
            return
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(
                f"crucible: could not read job {job.id}'s kept request ({path}): "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return
        job.request = document if isinstance(document, dict) else None

    def _persist(self, job: Job) -> None:
        try:
            job.dir.mkdir(parents=True, exist_ok=True)
            document = json.dumps(self._record_of(job), indent=2) + "\n"
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
            "waiting": job.waiting,
            "removal": job.removal,
            "session": job.session,
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
        if job.waiting is not None and self._line is not None:
            self._line.remove(
                job.id,
                "client",
                "the client that queued it cancelled it (DELETE /v1/jobs/{id})",
            )
            return REMOVED
        job.cancel_requested = True
        if job.status == QUEUED:
            with self._lane_lock:
                self._admitted.release(job.id)
            self._finish(job, CANCELLED)
            return CANCELLED
        return "cancelling"


    async def _run_lane(self) -> None:
        while True:
            if self._admitted.job_id is None:
                self._wake.clear()
                try:
                    self.reap()
                except Exception as exc:
                    print(
                        f"crucible: the job reaper failed: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                try:
                    await asyncio.wait_for(
                        self._wake.wait(), timeout=REAP_INTERVAL_SECONDS
                    )
                except asyncio.TimeoutError:
                    pass
                continue
            with self._lane_lock:
                if self._stopping:
                    return
                job_id = self._admitted.take()
                if job_id is None:
                    continue
                job = self._jobs[job_id]
                cancelled = job.cancel_requested
                if not cancelled:
                    self._running_id = job_id
                    self._lane_idle.clear()
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
        error: JobFailure | None = None
        try:
            await asyncio.to_thread(plugin.run, job, ctx)
        except JobCancelled:
            status = CANCELLED
        except JobError as exc:
            status, error = FAILED, JobFailure(exc.code, exc.message)
        except Exception as exc:
            status = FAILED
            error = JobFailure("job_failed", f"{type(exc).__name__}: {exc}")
        else:
            if job.cancel_requested:
                status = CANCELLED
        if status == CANCELLED and job.id in self._interrupted_by_stop:
            status = INTERRUPTED
        try:
            await self._settle(job, status)
        finally:
            self._finish(job, status, error)
            self._running_id = None
            self._idle()

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
        progress = JournalProgress.of(before)
        self.append_event(
            job,
            "note",
            {
                "message": f"resumed: {progress.sentence}, from journal {job.resume_id} "
                f"(last saved {progress.last_saved})",
                "resume_id": job.resume_id,
                "resumed_from": progress.resumed_from,
                "units_done": progress.units_done,
                "units_total": progress.units_total,
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
        else:
            if settled is not None:
                self.append_event(job, "note", settled.to_dict())
        if RESIDENT_KEY in job.done_extra:
            # A job that reports what is resident read it when it finished; the
            # settlement runs after that and may have taken it off the card (a note
            # says so), even when it then failed stopping the process. The done event
            # written next says what the residency holds now, whatever the outcome.
            job.done_extra[RESIDENT_KEY] = self._settlement.resident_id

    def _fail_out_of_band(self, job: Job, exc: BaseException) -> None:
        job.status = FAILED
        job.finished = utcnow()
        job.failure = JobFailure(
            "queue_failed",
            f"the job lane could not finish this job: {type(exc).__name__}: "
            f"{exc}. This is a bug in Crucible, not in the request.",
        )
        try:
            self.append_event(job, "failed", {"error": job.error})
        except Exception:
            pass
        self._persist(job)
        if job.resume_id is not None:
            self._journals.ended(job.resume_id, job.id, FAILED)
        self._running_id = None
        self._idle()

    def _finish(self, job: Job, status: str, error: JobFailure | None = None) -> None:
        job.status = status
        job.finished = utcnow()
        job.failure = error
        if status == INTERRUPTED:
            job.interrupted_at = job.finished
        self._persist(job)
        if status == DONE:
            # The done record says what it ran with (done_extra); the copy kept for a
            # failure has done its work. Dropped after the record says `done`, so a stop
            # between the two leaves it for `_read_request` to drop, never a done-less job
            # without it.
            self._drop_request(job)
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
            self.append_event(job, "failed", {"error": job.error})
        elif status == INTERRUPTED:
            self._restamp_provenance(job)
            self.append_event(
                job,
                "note",
                {"message": f"the server stopped while job {job.id} ran, so it "
                            "ended interrupted. Collect what landed from "
                            f"GET /v1/jobs/{job.id} and submit the rest again "
                            "once `crucible serve` is back."},
            )
        else:
            self._restamp_provenance(job)
            self.append_event(job, "cancelled", {"status": CANCELLED})
