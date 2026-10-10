"""The server's waiting line: what a request found busy and waits for (waiting is the
default; a request sent with ``"queue": false`` is refused instead, crucible/queuerequest.py).

A queued job is a normal job (status ``queued``) that holds its inputs and waits here
until the queue pump (crucible/queuepump.py) walks the line and admits it through the
same admission path a fresh submit takes. A queued chat or decision is a *call*: no job
record, just a ticket whose HTTP request is held open until the pump gives it a slot
on the resident model (crucible/callqueue.py). A queue session (crucible/queuesessions.py) waits
here too until the pump opens it (crucible/sessionqueue.py); while one is open its own
items go ahead of everything else. This module is the line's state and its
announcements; it never admits anything itself.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from .. import clock, keeptogether
from ..accelerator import WAITS_FOR_THE_CARD
from ..errors import ApiError
from ..events import QUEUE
from ..keeptogether import KEEPING_CALLS_TOGETHER, Hold, Kept
from ..queuesessions import LOAD_FAILED, QueueSession, QueueSessions, is_session_id
from .base import TERMINAL_STATES, Job, JobFailure

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
SESSION_CLOSED = "session_closed"
REMOVAL_REASONS = (OPERATOR, CLIENT, EXPIRED, SERVER_RESTART, SESSION_CLOSED)

UNNAMED = "an unnamed client"

CALL_PREFIX = "call-"
ADMITTED = "admitted"

CARD_RECHECK_S = 5.0
"""How often an item waiting for the accelerator (``accelerator_busy``: memory on it is
held by a process this Crucible does not own) has its card checked again. Each check is
a full admission (nvidia-smi and the process table), so it is paced, never per tick."""

CARD_WAIT_REPEAT_S = 60.0
"""How often a card wait whose holder has not changed is said again on the job's, the
session's and the queue's streams, so a client watching for silence sees the wait is
still alive. A changed holder is said at once. A wait behind a model's queued calls
(``KeepWait``) is repeated on the same pace."""


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
    session: str | None = None
    id: str = field(default_factory=lambda: CALL_PREFIX + uuid.uuid4().hex)
    status: str = "queued"

    def busy_details(self) -> dict[str, Any]:
        return {"call_id": self.id, "type": self.type, "model": self.model,
                "client": self.client}


@dataclass(frozen=True)
class CardWait:
    """Why an item at the front waits for the accelerator rather than for the lane: the
    refusal its last admission met (``accelerator_busy``), naming who holds the card."""

    code: str
    message: str
    details: dict[str, Any] | None
    since: datetime
    next_check: datetime
    said_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "since": self.since.isoformat(),
            "next_check_at": self.next_check.isoformat(),
        }


@dataclass(frozen=True)
class KeepWait:
    """Why an item waits behind calls that arrived after it: the resident model keeps
    its queued calls together, and ``ahead`` of them run on it first
    (crucible/keeptogether.py)."""

    model: str
    ahead: tuple[str, ...]
    turn_taken: bool
    message: str
    since: datetime
    said_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": KEEPING_CALLS_TOGETHER,
            "message": self.message,
            "details": {
                "model": self.model,
                "ahead": len(self.ahead),
                "ahead_ids": list(self.ahead),
                "turn_taken": self.turn_taken,
            },
            "since": self.since.isoformat(),
            "next_check_at": None,
        }


@dataclass
class Waiting:
    is_call: ClassVar[bool] = False
    is_session: ClassVar[bool] = False

    job: Job
    request: Any
    max_wait_s: int
    submitted: datetime
    seen: datetime
    fresh_journal: Any = None
    position: int | None = None
    gone: bool = False
    card_wait: CardWait | None = None
    kept: Kept | None = None
    keep_wait: KeepWait | None = None

    @property
    def expires_at(self) -> datetime:
        return self.submitted + timedelta(seconds=self.max_wait_s)

    @property
    def session(self) -> str | None:
        """The session this item belongs to: the open session's items go first."""
        return getattr(self.job, "session", None)

    @property
    def kind(self) -> str:
        return "session" if self.is_session else "call" if self.is_call else "job"

    def waited_s(self, now: datetime) -> float:
        return round(max(0.0, (now - self.submitted).total_seconds()), 3)

    def row(self, now: datetime) -> dict[str, Any]:
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
            "session": self.session,
            "kind": self.kind,
            "waiting_for": self.waiting_for(),
        }

    def waiting_for(self) -> dict[str, Any] | None:
        """What it waits for beyond its turn: a held card, or a model's queued calls."""
        if self.card_wait is not None:
            return self.card_wait.to_dict()
        if self.keep_wait is not None:
            return self.keep_wait.to_dict()
        return None

    def card_due(self, now: datetime) -> bool:
        """Whether this item may meet admission now: always, unless it is waiting for
        the accelerator and its next check has not come."""
        return self.card_wait is None or now >= self.card_wait.next_check


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


@dataclass
class WaitingSession(Waiting):
    """A queue session waiting to open. ``load_job`` is the load-model job the pump
    started to open it with its model."""

    is_session: ClassVar[bool] = True

    load_job: str | None = None
    loads: int = 0


class WaitingLine:
    """FIFO, except that the items of the open queue session go ahead of the line, and
    the queued calls on a resident model that keeps its calls together go ahead of what
    would take it off the card (crucible/keeptogether.py)."""

    def __init__(
        self,
        store: "JobStore",
        sessions: QueueSessions,
        resident: Callable[[], Any] = lambda: None,
    ) -> None:
        self._store = store
        self._sessions = sessions
        self._resident = resident
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

    @property
    def sessions(self) -> QueueSessions:
        return self._sessions

    def open_session(self) -> str | None:
        session = self._sessions.current()
        return None if session is None else session.id

    def ordered(self) -> list[Waiting]:
        return self._ordering()[0]

    def _ordering(self) -> tuple[list[Waiting], Hold | None]:
        items = list(self._items)
        first_id = self.open_session()
        first = [] if first_id is None else [i for i in items if i.session == first_id]
        rest = items if first_id is None else [i for i in items if i.session != first_id]
        kept, hold = keeptogether.order(rest, self._resident())
        return first + kept, hold

    def take_kept_turn(self) -> bool:
        """The lane is free: an item that would take a resident that keeps its calls
        together off the card, and is the first still waiting to have arrived, takes its
        turn now (keeptogether.take_turn). Not while a session is open: nothing else
        runs then."""
        if self.open_session() is not None:
            return False
        return keeptogether.take_turn(list(self._items), self._resident())

    def kept_on_card(self) -> tuple[str, int] | None:
        """The resident and how many queued calls run on it next, when it keeps its
        calls together and some do: the settlement does not unload it between them."""
        resident = self._resident()
        count = keeptogether.runs_next_on(self.ordered(), resident)
        return None if count == 0 else (resident.id, count)

    def items_of(self, session_id: str) -> list[Waiting]:
        return [item for item in self._items if item.session == session_id]

    def calls_waiting(self) -> dict[str, int]:
        """How many calls, and sessions that open on a model, wait for each model."""
        counts: dict[str, int] = {}
        for item in self._items:
            if (item.is_call or item.is_session) and item.job.model is not None:
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
        return [item.row(now) for item in self.ordered()]


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
        self._store.updating.refuse_if_holding(f"a {call.type} call")
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

    def join_session(self, session: QueueSession) -> WaitingSession:
        self.refuse_if_full(session.client)
        now = clock.now()
        item = WaitingSession(
            job=session,  # type: ignore[arg-type]
            request=None, max_wait_s=session.max_wait_s, submitted=now, seen=now,
        )
        self._items.append(item)
        self._client_seen[session.client] = now
        self.reorder(announce_new=item)
        self._wake()
        return item


    def not_yet(self, item: Waiting, refusal: ApiError) -> None:
        """The item met a refusal that keeps it waiting in its place. One that says the
        accelerator is held (``accelerator_busy``) is recorded on the item, said on its
        stream, the session's and the queue's whenever who holds the card changes and
        again every ``CARD_WAIT_REPEAT_S`` while it does not, and checked again only
        after ``CARD_RECHECK_S``; any other ("busy") clears it."""
        if refusal.code not in WAITS_FOR_THE_CARD:
            item.card_wait = None
            return
        now = clock.now()
        was = item.card_wait
        repeat = (
            was is not None
            and was.message == refusal.message
            and now - was.said_at < timedelta(seconds=CARD_WAIT_REPEAT_S)
        )
        item.card_wait = CardWait(
            code=refusal.code,
            message=refusal.message,
            details=refusal.details,
            since=now if was is None else was.since,
            next_check=now + timedelta(seconds=CARD_RECHECK_S),
            said_at=was.said_at if repeat and was is not None else now,
        )
        if repeat:
            return
        said = {
            "code": refusal.code,
            "message": (
                f"waiting for the accelerator, checked again every {CARD_RECHECK_S:g} s "
                f"until {item.expires_at.isoformat()}: {refusal.message}"
            ),
            "details": refusal.details,
            "since": item.card_wait.since.isoformat(),
            "next_check_at": item.card_wait.next_check.isoformat(),
        }
        if item.is_session:
            self._sessions.waiting(item.job, said)  # type: ignore[arg-type]
        elif not item.is_call:
            self._store.append_event(item.job, "waiting", said)
        self._announce("waiting", item, code=refusal.code, message=said["message"])

    def reorder(self, announce_new: Waiting | None = None) -> None:
        depth = len(self._items)
        ordered, hold = self._ordering()
        for index, item in enumerate(ordered, start=1):
            if item.position == index and item is not announce_new:
                continue
            was, item.position = item.position, index
            data: dict[str, Any] = {"position": index, "of": depth}
            if item is announce_new:
                data.update(max_wait_s=item.max_wait_s,
                            expires_at=item.expires_at.isoformat())
            if item.is_session:
                self._sessions.positioned(
                    item.job, index, depth,  # type: ignore[arg-type]
                    first=item is announce_new,
                )
            elif not item.is_call:
                self._store.append_event(item.job, "queued", data)
            if item is announce_new:
                self._announce("added", item, **self._added(item))
            elif was is not None:
                self._announce("moved", item, position=index)
        self._say_kept(ordered, hold)

    def _say_kept(self, ordered: list[Waiting], hold: Hold | None) -> None:
        """Say why the item held behind a model's queued calls waits: when the hold
        begins, whenever which calls go ahead of it changes, and every
        ``CARD_WAIT_REPEAT_S`` while nothing does."""
        for item in ordered:
            if item.keep_wait is not None and (hold is None or item is not hold.item):
                item.keep_wait = None
        if hold is None:
            return
        item = hold.item
        now = clock.now()
        ahead = tuple(other.job.id for other in hold.ahead)
        was = item.keep_wait
        if (
            was is not None
            and was.ahead == ahead
            and was.turn_taken == hold.turn_taken
            and now - was.said_at < timedelta(seconds=CARD_WAIT_REPEAT_S)
        ):
            return
        what = "session" if item.is_session else item.job.type
        later = (
            f"calls for {hold.model} sent from now on wait behind this {what}"
            if hold.turn_taken
            else f"when this {what} would be next, the calls for {hold.model} waiting "
            "then go first too, and any sent after that wait behind it"
        )
        message = (
            f"waiting: {hold.model} has {len(ahead)} queued call(s) ahead, run first "
            f"because {hold.model} keeps its calls together (it is slow to load, and "
            f"this {what} would take it off the card between them); {later}"
        )
        item.keep_wait = KeepWait(
            model=hold.model,
            ahead=ahead,
            turn_taken=hold.turn_taken,
            message=message,
            since=now if was is None else was.since,
            said_at=now,
        )
        said = item.keep_wait.to_dict()
        if item.is_session:
            self._sessions.waiting(item.job, said)  # type: ignore[arg-type]
        elif not item.is_call:
            self._store.append_event(item.job, "waiting", said)
        self._announce("waiting", item, code=KEEPING_CALLS_TOGETHER, message=message)

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
        if not item.is_call and not item.is_session:
            self._store.mark_waiting(item.job, None)

    def started(self, item: Waiting, entry: Any = None) -> None:
        waited = item.waited_s(clock.now())
        self._take(item)
        if isinstance(item, WaitingCall):
            item.job.status = "running"
            item.settle(ADMITTED, entry)
        elif isinstance(item, WaitingSession):
            self._sessions.opened(item.job)  # type: ignore[arg-type]
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
        elif isinstance(item, WaitingSession):
            reason = LOAD_FAILED if refusal.code == "session_load_failed" else REFUSED
            self._sessions.removed(
                item.job, reason, refusal.message, failure.to_dict()  # type: ignore[arg-type]
            )
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
        elif isinstance(item, WaitingSession):
            self._sessions.removed(item.job, reason, message)  # type: ignore[arg-type]
            self._abandon_load(item)
        else:
            self._store.end_waiting(item, removal=removal)
        self._announce("removed", item, reason=reason, message=message)
        self.reorder()
        self._wake()
        return item

    def _abandon_load(self, item: WaitingSession) -> None:
        """A session that leaves the line while the load opening it runs takes that load
        with it: the load was its own, and finishing it would leave a model on the card
        that nothing asked for (a fleet's losing server, a client that gave up)."""
        if item.load_job is None:
            return
        job = self._store.get(item.load_job)
        item.load_job = None
        if job.status not in TERMINAL_STATES:
            self._store.cancel(job)

    def not_waiting(self, job_id: str) -> ApiError:
        if is_session_id(job_id):
            session = self._sessions.get(job_id)
            how = "" if session.reason is None else f" ({session.reason})"
            return ApiError(
                409,
                "not_queued",
                f"session {job_id} is not waiting in this server's queue; it is "
                f"{session.status}{how}",
                {"job_id": job_id, "status": session.status},
            )
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

    def remove_items_of(self, session_id: str, message: str) -> int:
        """The items a closed session left waiting: nothing runs them as its items any
        more, so they leave the line rather than wait as anyone else's."""
        taken = [item.job.id for item in self.items_of(session_id)]
        for job_id in taken:
            self.remove(job_id, SESSION_CLOSED, message)
        return len(taken)

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
            if item.card_wait is not None:
                return (
                    f"it waited its whole max_wait_s ({item.max_wait_s} s) without "
                    "reaching the lane; since "
                    f"{item.card_wait.since.isoformat()} it was waiting for the "
                    f"accelerator, and it was still held: {item.card_wait.message}"
                )
            if item.keep_wait is not None:
                return (
                    f"it waited its whole max_wait_s ({item.max_wait_s} s) without "
                    "reaching the lane; since "
                    f"{item.keep_wait.since.isoformat()} it was waiting behind the "
                    f"queued calls for {item.keep_wait.model}: {item.keep_wait.message}"
                )
            return (
                f"it waited its whole max_wait_s ({item.max_wait_s} s) without "
                "reaching the lane"
            )
        if item.is_call:
            return None
        if item.is_session and item.job.followed:  # type: ignore[attr-defined]
            return None
        job = item.job
        if job.id in followed_jobs:
            return None
        if job.client is not None and job.client in followed_clients:
            return None
        seen = max(item.seen, self._client_seen.get(job.client, item.seen))
        if (now - seen).total_seconds() < ABANDON_AFTER_S:
            return None
        if item.is_session:
            return (
                f"nobody followed it for {ABANDON_AFTER_S} s: no open event stream on "
                "it or on a job of the same client, no GET /v1/queue/sessions/{id}, and "
                "no POST /v1/queue/sessions/{id}/touch, so the client that asked for it "
                "is gone"
            )
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
        self._store.events.publish(QUEUE, f"queue.{kind}", {
            **self._events[-1]["data"], "kind": item.kind})
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
