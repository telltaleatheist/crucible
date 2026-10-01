"""The server's waiting line: jobs a client asked to queue while the lane was busy.

A queued job is a normal job (status ``queued``) that holds its inputs and waits here
until the queue pump (crucible/queuepump.py) walks the line and admits it through the
same admission path a fresh submit takes. A queued chat or decision is a *call*: no job
record, just a ticket whose HTTP request is held open until the pump gives it a slot
on the resident model (crucible/callqueue.py). This module is the line's state and its
announcements; it never admits anything itself.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from .. import clock
from ..errors import ApiError
from .base import Job, JobFailure

if TYPE_CHECKING:
    from .queue import JobStore

DEFAULT_MAX_WAIT_S = 3600
MIN_MAX_WAIT_S = 10
MAX_MAX_WAIT_S = 86_400
PER_CLIENT_LIMIT = 50
TOTAL_LIMIT = 200
ABANDON_AFTER_S = 300
EVENT_MEMORY = 500

OPERATOR = "operator"
CLIENT = "client"
EXPIRED = "expired"
SERVER_RESTART = "server_restart"
REFUSED = "refused"
REMOVAL_REASONS = (OPERATOR, CLIENT, EXPIRED, SERVER_RESTART)

UNNAMED = "an unnamed client"

CALL_PREFIX = "call-"
ADMITTED = "admitted"
LEASE = "lease"


def limits() -> dict[str, Any]:
    return {
        "per_client": PER_CLIENT_LIMIT,
        "total": TOTAL_LIMIT,
        "max_wait_s": {
            "default": DEFAULT_MAX_WAIT_S,
            "min": MIN_MAX_WAIT_S,
            "max": MAX_MAX_WAIT_S,
        },
        "abandon_after_s": ABANDON_AFTER_S,
    }


@dataclass
class Call:
    """What a queued chat or decision shows in the line where a job shows its record."""

    type: str
    model: str
    client: str | None
    act: str | None = None
    client_ref: str | None = None
    ttl_seconds: int | None = None
    id: str = field(default_factory=lambda: CALL_PREFIX + uuid.uuid4().hex)
    status: str = "queued"

    def busy_details(self) -> dict[str, Any]:
        return {"call_id": self.id, "type": self.type, "model": self.model,
                "client": self.client}


@dataclass
class Waiting:
    is_call: ClassVar[bool] = False

    job: Job
    request: Any
    max_wait_s: int
    submitted: datetime
    seen: datetime
    fresh_journal: Any = None
    position: int | None = None
    gone: bool = False

    @property
    def expires_at(self) -> datetime:
        return self.submitted + timedelta(seconds=self.max_wait_s)

    def waited_s(self, now: datetime) -> float:
        return round(max(0.0, (now - self.submitted).total_seconds()), 3)

    def row(self, now: datetime, holder: str | None) -> dict[str, Any]:
        job = self.job
        return {
            "position": self.position,
            "job_id": job.id,
            "type": job.type,
            "model": job.model,
            "client": job.client,
            "client_ref": job.client_ref,
            "submitted": self.submitted.isoformat(),
            "waited_s": self.waited_s(now),
            "max_wait_s": self.max_wait_s,
            "expires_at": self.expires_at.isoformat(),
            "lease_holder": holder is not None and job.client == holder,
            "kind": (
                ("lease" if job.type == LEASE else "call") if self.is_call else "job"
            ),
        }


@dataclass
class WaitingCall(Waiting):
    """A held-open chat or decision. ``outcome`` resolves once: ``(ADMITTED, entry)``
    with the in-flight slot the pump opened for it, or ``(status, ApiError)`` when it
    left the line any other way. ``load_job`` is the load-model job the pump started
    for it when its model was not resident."""

    is_call: ClassVar[bool] = True

    outcome: "asyncio.Future[tuple[str, Any]] | None" = None
    load_job: str | None = None
    loads: int = 0

    def settle(self, status: str, value: Any) -> None:
        if self.outcome is not None and not self.outcome.done():
            self.outcome.set_result((status, value))


class WaitingLine:
    """FIFO, except that the client holding the open lease goes ahead of the line."""

    def __init__(self, store: "JobStore", lease_holder: Callable[[], str | None]) -> None:
        self._store = store
        self._lease_holder = lease_holder
        self._items: list[Waiting] = []
        self._client_seen: dict[str | None, datetime] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=EVENT_MEMORY)
        self._last_id = 1
        self._waiters: list[asyncio.Event] = []
        self._wake: Callable[[], None] = lambda: None
        store.attach_line(self)

    def when_changed(self, wake: Callable[[], None]) -> None:
        self._wake = wake

    def __len__(self) -> int:
        return len(self._items)

    def holder(self) -> str | None:
        return self._lease_holder()

    def ordered(self) -> list[Waiting]:
        holder = self.holder()
        if holder is None:
            return list(self._items)
        first = [item for item in self._items if item.job.client == holder]
        return first + [item for item in self._items if item.job.client != holder]

    def calls_waiting(self) -> dict[str, int]:
        """How many calls wait for each model."""
        counts: dict[str, int] = {}
        for item in self._items:
            if item.is_call:
                counts[item.job.model] = counts.get(item.job.model, 0) + 1
        return counts

    def get(self, job_id: str) -> Waiting | None:
        for item in self._items:
            if item.job.id == job_id:
                return item
        return None

    def position(self, job_id: str) -> int | None:
        item = self.get(job_id)
        return None if item is None else item.position

    def rows(self, now: datetime | None = None) -> list[dict[str, Any]]:
        now = clock.now() if now is None else now
        holder = self.holder()
        return [item.row(now, holder) for item in self.ordered()]


    def refuse_if_full(self, client: str | None) -> None:
        total = len(self._items)
        if total >= TOTAL_LIMIT:
            raise ApiError(
                409,
                "queue_full",
                f"this server's queue already holds {total} job(s), its limit. "
                "Wait for some to run, or remove some with DELETE /v1/queue/{job_id}",
                {"scope": "server", "limit": TOTAL_LIMIT, "depth": total},
            )
        mine = sum(1 for item in self._items if item.job.client == client)
        if mine >= PER_CLIENT_LIMIT:
            who = UNNAMED if client is None else repr(client)
            raise ApiError(
                409,
                "queue_full",
                f"{who} already has {mine} job(s) waiting in this server's queue, "
                f"the limit for one client. Wait for some to run before queueing more",
                {"scope": "client", "limit": PER_CLIENT_LIMIT, "depth": mine,
                 "client": client},
            )

    def join(self, job: Job, request: Any, max_wait_s: int, fresh_journal: Any) -> Waiting:
        self.refuse_if_full(job.client)
        now = clock.now()
        item = Waiting(
            job=job, request=request, max_wait_s=max_wait_s,
            submitted=now, seen=now, fresh_journal=fresh_journal,
        )
        self._items.append(item)
        self._store.mark_waiting(job, {"max_wait_s": max_wait_s,
                                       "submitted": now.isoformat()})
        self._client_seen[job.client] = now
        self.reorder(announce_new=item)
        self._wake()
        return item

    def join_call(self, call: Call, max_wait_s: int) -> WaitingCall:
        self.refuse_if_full(call.client)
        now = clock.now()
        item = WaitingCall(
            job=call,  # type: ignore[arg-type]
            request=None, max_wait_s=max_wait_s, submitted=now, seen=now,
            outcome=asyncio.get_running_loop().create_future(),
        )
        self._items.append(item)
        self.reorder(announce_new=item)
        self._wake()
        return item


    def reorder(self, announce_new: Waiting | None = None) -> None:
        depth = len(self._items)
        for index, item in enumerate(self.ordered(), start=1):
            if item.position == index and item is not announce_new:
                continue
            was, item.position = item.position, index
            data: dict[str, Any] = {"position": index, "of": depth}
            if item is announce_new:
                data.update(max_wait_s=item.max_wait_s,
                            expires_at=item.expires_at.isoformat())
            if not item.is_call:
                self._store.append_event(item.job, "queued", data)
            if item is announce_new:
                self._announce("added", item, **self._added(item))
            elif was is not None:
                self._announce("moved", item, position=index)

    @staticmethod
    def _added(item: Waiting) -> dict[str, Any]:
        job = item.job
        return {
            "position": item.position,
            "type": job.type,
            "model": job.model,
            "client": job.client,
            "submitted": item.submitted.isoformat(),
            "max_wait_s": item.max_wait_s,
        }

    def _take(self, item: Waiting) -> None:
        if item in self._items:
            self._items.remove(item)
        item.gone = True
        if not item.is_call:
            self._store.mark_waiting(item.job, None)

    def started(self, item: Waiting, entry: Any = None) -> None:
        waited = item.waited_s(clock.now())
        self._take(item)
        if isinstance(item, WaitingCall):
            item.job.status = "running"
            item.settle(ADMITTED, entry)
        else:
            self._store.append_event(item.job, "started", {"waited_s": waited})
        self._announce("started", item, waited_s=waited)
        self.reorder()

    def fail(self, item: Waiting, refusal: ApiError) -> None:
        self._take(item)
        failure = JobFailure(refusal.code, refusal.message)
        if isinstance(item, WaitingCall):
            item.job.status = "failed"
            item.settle("failed", refusal)
        else:
            self._store.end_waiting(item, failure=failure)
        self._announce("removed", item, reason=REFUSED, error=failure.to_dict())
        self.reorder()

    def remove(self, job_id: str, reason: str, message: str) -> Waiting:
        item = self.get(job_id)
        if item is None:
            raise self.not_waiting(job_id)
        waited = item.waited_s(clock.now())
        self._take(item)
        removal = {"reason": reason, "message": message, "waited_s": waited}
        if isinstance(item, WaitingCall):
            item.job.status = "removed"
            item.settle("removed", removed_call(item.job, removal))
        else:
            self._store.end_waiting(item, removal=removal)
        self._announce("removed", item, reason=reason, message=message)
        self.reorder()
        self._wake()
        return item

    def not_waiting(self, job_id: str) -> ApiError:
        if job_id.startswith(CALL_PREFIX):
            return ApiError(
                409,
                "not_queued",
                f"{job_id} is not waiting in this server's queue: the chat or "
                "decision it named has been answered, is being answered, or has left",
                {"job_id": job_id, "status": None},
            )
        job = self._store.get(job_id)
        return ApiError(
            409,
            "not_queued",
            f"job {job_id} is not waiting in this server's queue; it is {job.status}. "
            "Cancel a job that has started with DELETE /v1/jobs/{job_id}",
            {"job_id": job_id, "status": job.status},
        )

    def drain(self, reason: str, message: str) -> int:
        taken = [item.job.id for item in list(self._items)]
        for job_id in taken:
            self.remove(job_id, reason, message)
        return len(taken)


    def touch(self, job_id: str | None = None, client: str | None = None) -> None:
        now = clock.now()
        if job_id is not None:
            item = self.get(job_id)
            if item is not None:
                item.seen = now
                client = item.job.client if client is None else client
        if client is not None:
            self._client_seen[client] = now

    def expire(self, now: datetime | None = None) -> list[Waiting]:
        now = clock.now() if now is None else now
        followed_jobs, followed_clients = self._store.followed()
        gone: list[Waiting] = []
        for item in list(self._items):
            why = self._why_expired(item, now, followed_jobs, followed_clients)
            if why is not None:
                gone.append(self.remove(item.job.id, EXPIRED, why))
        return gone

    def _why_expired(
        self, item: Waiting, now: datetime,
        followed_jobs: set[str], followed_clients: set[str | None],
    ) -> str | None:
        if now >= item.expires_at:
            return (
                f"it waited its whole max_wait_s ({item.max_wait_s} s) without "
                "reaching the lane"
            )
        if item.is_call:
            return None
        job = item.job
        if job.id in followed_jobs:
            return None
        if job.client is not None and job.client in followed_clients:
            return None
        seen = max(item.seen, self._client_seen.get(job.client, item.seen))
        if (now - seen).total_seconds() < ABANDON_AFTER_S:
            return None
        return (
            f"nobody followed it for {ABANDON_AFTER_S} s: no open event stream on it "
            "or on another job of the same client, no GET /v1/jobs/{id}, and no "
            "POST /v1/queue/{id}/heartbeat, so the client that queued it is gone"
        )


    def _announce(self, kind: str, item: Waiting, **data: Any) -> None:
        self._last_id += 1
        self._events.append({
            "id": self._last_id,
            "event": kind,
            "data": {"job_id": item.job.id, **data, "depth": len(self._items)},
        })
        for waiter in self._waiters:
            waiter.set()

    @property
    def last_event_id(self) -> int:
        return self._last_id

    def events_after(self, cursor: int) -> list[dict[str, Any]]:
        return [event for event in self._events if event["id"] > cursor]

    def subscribe(self) -> asyncio.Event:
        waiter = asyncio.Event()
        self._waiters.append(waiter)
        return waiter

    def unsubscribe(self, waiter: asyncio.Event) -> None:
        if waiter in self._waiters:
            self._waiters.remove(waiter)


def removed_call(call: Call, removal: dict[str, Any]) -> ApiError:
    reason = removal["reason"]
    return ApiError(
        409,
        "removed_from_queue",
        f"this {call.type} waited {removal['waited_s']} s in the server's queue for "
        f"{call.model!r} and was removed ({reason}): {removal['message']}. Nothing "
        "was sent to the engine",
        {"call_id": call.id, "reason": reason, "waited_s": removal["waited_s"]},
    )
