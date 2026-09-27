from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, Callable

from .cardkinds import KIND_NOUNS
from .clock import now as _utcnow
from .errors import ApiError
from .jobtypes import JOB_TYPE_SPECS, CardEffect

MIN_TTL_SECONDS = 30
MAX_TTL_SECONDS = 3600

CARD_EFFECTS: dict[str, CardEffect] = {spec.name: spec.card for spec in JOB_TYPE_SPECS}


def require_ttl(ttl_seconds: int) -> int:
    if MIN_TTL_SECONDS <= ttl_seconds <= MAX_TTL_SECONDS:
        return ttl_seconds
    raise ApiError(
        400,
        "invalid_ttl",
        f"a lease's ttl_seconds must be between {MIN_TTL_SECONDS} and "
        f"{MAX_TTL_SECONDS} seconds, and {ttl_seconds} is not. Shorter than "
        f"{MIN_TTL_SECONDS}s is a lease that expires between two heartbeats — "
        "the eviction a lease exists to prevent, on a schedule; longer than "
        f"{MAX_TTL_SECONDS}s is a lease that outlives its own client's crash, "
        "which is the one thing expiry is for. Heartbeat a short lease rather "
        "than asking for a long one",
        {
            "ttl_seconds": ttl_seconds,
            "min_ttl_seconds": MIN_TTL_SECONDS,
            "max_ttl_seconds": MAX_TTL_SECONDS,
        },
    )


@dataclass(frozen=True)
class Lease:
    id: str
    kind: str
    subject: str
    act: str
    client: str | None
    since: datetime
    expires_at: datetime
    ttl_seconds: int

    @property
    def noun(self) -> str:
        return KIND_NOUNS[self.kind]

    def expired(self, now: datetime) -> bool:
        return now >= self.expires_at

    def evicted_by(self, job_type: str, model: str | None) -> bool:
        effect = CARD_EFFECTS[job_type]
        if effect.makes_resident is not None:
            reuses_the_leased_thing = (
                effect.reuses_what_it_names
                and effect.makes_resident == self.kind
                and model == self.subject
            )
            if not reuses_the_leased_thing:
                return True
        return effect.takes_off == self.kind

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.id,
            "kind": self.kind,
            "client": self.client,
            "act": self.act,
            "since": self.since.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }

    def receipt(self) -> dict[str, Any]:
        return {**self.to_dict(), "subject": self.subject}


class Leases:
    def __init__(self, now: Callable[[], datetime] | None = None) -> None:
        self._now = _utcnow if now is None else now
        self._lock = threading.Lock()
        self._lease: Lease | None = None
        self._closed: str | None = None
        self._lapse_handled: str | None = None


    def current(self) -> Lease | None:
        with self._lock:
            return self._open_locked()

    def lapsed_at(self) -> datetime | None:
        with self._lock:
            lease = self._lease
            if lease is None or self._closed is not None:
                return None
            if lease.id == self._lapse_handled:
                return None
            if not lease.expired(self._now()):
                return None
            return lease.expires_at

    def _open_locked(self) -> Lease | None:
        lease = self._lease
        if lease is None or self._closed is not None:
            return None
        if lease.expired(self._now()):
            return None
        return lease


    def open(
        self,
        *,
        kind: str,
        subject: str,
        act: str,
        client: str | None,
        ttl_seconds: int,
    ) -> Lease:
        with self._lock:
            held = self._open_locked()
            if held is not None:
                raise leased_error(held, f"leasing {subject!r}")
            since = self._now()
            lease = Lease(
                id=uuid.uuid4().hex,
                kind=kind,
                subject=subject,
                act=act,
                client=client,
                since=since,
                expires_at=since + timedelta(seconds=ttl_seconds),
                ttl_seconds=ttl_seconds,
            )
            self._lease = lease
            self._closed = None
            self._lapse_handled = None
            return lease

    def forget_lapse(self) -> None:
        with self._lock:
            if self._lease is not None:
                self._lapse_handled = self._lease.id

    def heartbeat(self, lease_id: str) -> Lease:
        with self._lock:
            held = self._open_locked()
            if held is None or held.id != lease_id:
                raise self._unknown_locked(lease_id)
            extended = replace(
                held, expires_at=self._now() + timedelta(seconds=held.ttl_seconds)
            )
            self._lease = extended
            return extended

    def release(self, lease_id: str) -> None:
        with self._lock:
            held = self._open_locked()
            if held is None or held.id != lease_id:
                raise self._unknown_locked(lease_id)
            self._closed = "it was released"


    def refuse_if_leased(self, job_type: str, model: str | None) -> None:
        held = self.current()
        if held is None:
            return
        if job_type not in CARD_EFFECTS:
            raise ApiError(
                500,
                "lease_scope_unknown",
                f"{job_type!r} is a job type crucible/leases.py has no ruling "
                "about: nothing says what it does to the card, so this server "
                "cannot tell whether the open lease should refuse it. Give it a "
                "JobTypeSpec in crucible/jobtypes.py, which CARD_EFFECTS is "
                "derived from",
                {"type": job_type},
            )
        if not held.evicted_by(job_type, model):
            return
        raise leased_error(held, f"a {job_type} job")

    def _unknown_locked(self, lease_id: str) -> ApiError:
        lease = self._lease
        if lease is not None and lease.id == lease_id:
            if self._closed is not None:
                why = f"{self._closed}"
            else:
                why = (
                    f"it expired at {lease.expires_at.isoformat()} and nothing "
                    "heartbeated it"
                )
            return ApiError(
                404,
                "unknown_lease",
                f"lease {lease_id} is no longer open: {why}. Take a new one with "
                f"POST /v1/models/{lease.subject}/lease — a client may re-lease "
                "a thing it let go of, provided nobody else took it and it is "
                "still resident",
                {"lease_id": lease_id, "reason": why},
            )
        return ApiError(
            404,
            "unknown_lease",
            f"this server has no lease {lease_id}. Leases are held in memory and "
            "a restart forgets them, and only the most recent one is remembered "
            "well enough to say how it ended",
            {"lease_id": lease_id, "reason": "this server never had it, or has "
             "forgotten it"},
        )


def leased_error(lease: Lease, what: str) -> ApiError:
    who = "an unnamed client" if lease.client is None else repr(lease.client)
    return ApiError(
        409,
        "leased",
        f"{lease.subject!r} (the resident {lease.noun}) is leased by {who} for "
        f"{lease.act!r} since {lease.since.isoformat()}, until at least "
        f"{lease.expires_at.isoformat()} — so {what} is refused rather than "
        f"taking the {lease.noun} off the card underneath a run in progress. A "
        f"lease is the client saying it intends more work on this {lease.noun}; "
        "wait for it to expire or be released, or use another server. "
        "GET /v1/activity reports it as `lease`",
        lease.to_dict(),
    )
