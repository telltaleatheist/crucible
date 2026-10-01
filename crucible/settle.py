from __future__ import annotations

import asyncio
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from .clock import now
from .errors import JobError
from .jobs.base import DONE
from .jobtypes import JOB_TYPE_SPECS

if TYPE_CHECKING:
    from .inflight import InFlight
    from .leases import Leases
    from .residency import Residency

SETTLEMENT_HOLDER = "the settlement clearing the card"

LEAVES_IT_RESIDENT: frozenset[str] = frozenset(
    spec.name for spec in JOB_TYPE_SPECS if spec.leaves_it_resident
)


@dataclass(frozen=True)
class Held:
    fact: str
    who: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.fact} holds it: {self.who}"

    def to_dict(self) -> dict[str, Any]:
        return {"fact": self.fact, "who": self.who, "details": self.details}


@dataclass(frozen=True)
class Settled:
    subject_id: str
    kind: str
    trigger: str

    @property
    def line(self) -> str:
        return (
            f"unloaded {self.subject_id} (the resident {self.kind}): nothing "
            f"holds it — {self.trigger}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.line,
            "unloaded": self.subject_id,
            "kind": self.kind,
            "trigger": self.trigger,
        }


def _to_stderr(message: str) -> None:
    print(f"crucible: {message}", file=sys.stderr)


class Settlement:
    def __init__(
        self,
        *,
        residency: "Residency",
        store: Any,
        leases: "Leases",
        inflight: "InFlight",
        waiting_calls: Callable[[], dict[str, int]] = dict,
        log: Callable[[str], None] = _to_stderr,
    ) -> None:
        self._waiting_calls = waiting_calls
        self._residency = residency
        self._store = store
        self._leases = leases
        self._inflight = inflight
        self._log = log
        self._unheld_since: datetime | None = None
        self._lock = threading.Lock()
        self._deadline: asyncio.TimerHandle | None = None
        self._running: set[asyncio.Task[Any]] = set()


    def arm_for_lease_expiry(self) -> None:
        loop = asyncio.get_running_loop()
        handle, self._deadline = self._deadline, None
        if handle is not None:
            handle.cancel()
        lease = self._leases.current()
        if lease is None:
            return
        seconds = max(
            0.0,
            (lease.expires_at - now()).total_seconds(),
        )
        self._deadline = loop.call_later(seconds, self._deadline_passed)

    def _deadline_passed(self) -> None:
        self._deadline = None
        loop = asyncio.get_running_loop()
        task = loop.create_task(
            asyncio.to_thread(
                self.settle_quietly, "the lease expired and was not renewed"
            )
        )
        self._running.add(task)

        def finished(done: asyncio.Task[Any]) -> None:
            self._running.discard(done)
            self.arm_for_lease_expiry()

        task.add_done_callback(finished)


    def holder(self, *, excluding_job: str | None = None) -> Held | None:
        job = self._store.occupied_by_anything_but(excluding_job)
        if job is not None:
            return Held(
                "a job", f"{job.type} {job.id} ({job.status})", job.busy_details()
            )
        lease = self._leases.current()
        if lease is not None:
            who = "an unnamed client" if lease.client is None else repr(lease.client)
            return Held(
                "a lease",
                f"{who} for {lease.act!r}, until {lease.expires_at.isoformat()}",
                lease.receipt(),
            )
        claim = self._residency.claimed_by
        if claim is not None and claim != SETTLEMENT_HOLDER:
            return Held("the claim", claim, {"held_by": claim})
        chats = len(self._inflight)
        if chats:
            return Held(
                "a chat", f"{chats} completion(s) in flight", {"in_flight": chats}
            )
        models = self._waiting_calls()
        resident = None if not models else self._residency.resident
        waiting = 0 if resident is None else models.get(resident.id, 0)
        if waiting:
            return Held(
                "a queued call",
                f"{waiting} chat(s) or decision(s) waiting for it in the queue",
                {"waiting": waiting},
            )
        return None


    def settle(
        self, trigger: str, *, excluding_job: str | None = None
    ) -> Settled | None:
        with self._lock:
            try:
                claimed = self._residency.claim_to_clear(
                    SETTLEMENT_HOLDER,
                    held=lambda: self.holder(excluding_job=excluding_job),
                )
            except JobError:
                return None
            if not claimed:
                return None
            try:
                resident = self._residency.resident
                if resident is None:
                    return None
                self._residency.unload(resident.id)
            finally:
                self._residency.release(SETTLEMENT_HOLDER)
        settled = Settled(
            subject_id=resident.id, kind=resident.kind, trigger=trigger
        )
        self._log(settled.line)
        return settled

    def unheld_since(self) -> datetime | None:
        if self.holder() is not None:
            return None
        if self._residency.resident is None:
            return None
        lapsed = self._leases.lapsed_at()
        stamped = self._unheld_since
        if lapsed is None:
            return stamped
        if stamped is None:
            return lapsed
        return max(lapsed, stamped)

    def held_by(self) -> Held | None:
        return self.holder()

    def settle_for_lapsed_lease(self) -> Settled | None:
        if self._leases.lapsed_at() is None:
            return None
        try:
            return self.settle("a lease lapsed and nothing heartbeated it")
        finally:
            self._leases.forget_lapse()

    def settle_quietly(self, trigger: str) -> Settled | None:
        try:
            return self.settle(trigger)
        except Exception as exc:
            self._log(
                f"could not clear the card ({trigger}): {type(exc).__name__}: {exc}"
            )
            return None

    def settle_for_job(self, job: Any, outcome: str) -> Settled | None:
        if job.type in LEAVES_IT_RESIDENT and outcome == DONE:
            self._unheld_since = now()
            return None
        return self.settle(
            f"job {job.id} ({job.type}) finished", excluding_job=job.id
        )
