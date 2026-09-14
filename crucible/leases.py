"""A client says it intends a run, and the card stops moving under it.

THE HOLE THIS FILLS. Three kinds of work reach this server and they hold three
different amounts of it. A **job** takes the lane: it has a row in
`GET /v1/activity`, and a second submission is refused `server_busy` naming the
holder. A **streaming session** holds the resident engine's exclusive claim
without taking the lane (`crucible/residency.py`, `refuse_if_claimed`). A **chat
completion** holds NOTHING — deliberately, because a vLLM engine batches and
taking the lane to fix a reporting bug would serialise work the engine exists to
overlap (`crucible/inflight.py`).

That last one is right for one chat and wrong for two thousand. Foundry
translates a book as a sequence of chat completions against a resident 27B; each
one is a few seconds, and between any two of them this server is, by every
measure it publishes, idle. So BookForge submits a `load-voice` at block 400 of
2000, the guard sees a card it may reclaim, the translator is evicted, and
Foundry's run dies in the middle with nothing having gone wrong anywhere.

WHY NOT A TIMER. The obvious patch is "refuse a loader if a chat finished within
N seconds". That is a fact standing in for a guess: a chat that ended eight
seconds ago is evidence of nothing — the client may have finished the book, or
crashed, or be about to send another nine hundred. N would be tuned against one
workload and wrong for the next, and the failure would be silent in both
directions (a loader refused forever because a dead client's last chat is still
inside the window; a run evicted because its next block was slow to arrive).

The fact that actually exists is **the client's intention**, and only the client
has it. So the client says so: it takes a LEASE, heartbeats while the run is
alive, and releases when it is done. Everything here is that sentence made
enforceable.

WHAT A LEASE IS AND IS NOT
--------------------------
It is a refusal, not a reservation. Holding one does not admit anything, does not
reserve the lane, and does not make this server accept a job it would otherwise
refuse. It says exactly one thing: **while this is open, nothing may move the
model off the card.** `slots.accelerated.accepts_work` is untouched, because a
leased server really will take a render that does not need the card's contents to
change — and it is `POST /v1/jobs` that still decides, as ever (R5).

It does not gate chats. Chats are what it protects; a lease that blocked them
would protect the run from itself.

ONE AT A TIME, PER SERVER. There is one card and one resident model, so there is
one lease. A second is refused `model_leased` naming the holder — the same
refusal a loader gets, because it is the same fact.

EXPIRY IS READ, NEVER SWEPT. A lease past `expires_at` is simply not open. There
is no background task, no timer and nothing to cancel: every read compares the
stored instant to the clock, so a client that dies mid-run stops blocking the
card the moment its ttl runs out, whether or not anything was watching. A
heartbeat is what a live client sends instead of dying.

IN MEMORY, AND A RESTART FORGETS. A lease is worth exactly as much as the
residency it protects, and a restarted server holds no model — so carrying a
lease across a restart would protect nothing, and would hand the next operator a
refusal whose subject no longer exists.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .errors import ApiError

#: The sane range for a ttl, in seconds, and the one the refusal states.
#:
#: The floor is not politeness: a ttl shorter than the gap between two heartbeats
#: is a lease that expires while its holder is alive and working, which is the
#: eviction this module exists to prevent, arriving on a schedule. The ceiling is
#: the other direction — an hour is longer than any single run's gap, and a
#: client that wanted a day is really asking for a lease that outlives its own
#: crash, which is the one thing expiry is for.
MIN_TTL_SECONDS = 30
MAX_TTL_SECONDS = 3600

#: The job types that can take the resident model off the card, which is the
#: whole set a lease refuses. Decided by reading what each one does to
#: `Residency`, not by what its name suggests:
#:
#: - `load-model` / `load-voice` — load something else, which evicts (one card,
#:   one resident thing: `Residency._evict`).
#: - `unload-model` — evicts by definition.
#: - `tts` — a render LOADS ITS VOICE if it is not already resident
#:   (`crucible/jobs/tts/render.py`, `_engine_for`), so it evicts a model exactly
#:   as `load-voice` does. Leaving it out would have left the hole open in the
#:   shape clients actually hit it: BookForge renders by submitting `tts`, not by
#:   loading a voice and then rendering.
#: - `align` — loads an aligner, which is a third resident kind and evicts the
#:   other two (`Residency.load_aligner`).
EVICTS_THE_RESIDENT_MODEL: frozenset[str] = frozenset(
    {"load-model", "unload-model", "load-voice", "tts", "align"}
)

#: The job types that cannot move the resident MODEL, and so are never refused
#: for a lease. Listed rather than implied, so that `tests/test_leases.py` can
#: prove every job type this build knows is in exactly one of the two sets — a
#: new job type that touches the card is then a failing test rather than a hole.
#:
#: - `echo` — never touches the accelerator.
#: - `asr` / `rvc` / `denoise` — run the guard with **no** `reclaimable_bytes`,
#:   which is their own deliberate note: they never unload somebody's resident
#:   engine to make room, they refuse instead.
#: - `unload-voice` / `unload-aligner` — each can only unload its OWN kind. With
#:   a model resident, `is_resident(KIND_TTS, ...)` is false and the job refuses
#:   `not_resident` on its own, so it cannot reach the leased model to evict it.
#:   Refusing them `model_leased` would report the wrong reason for the right
#:   outcome.
KEEPS_THE_RESIDENT_MODEL: frozenset[str] = frozenset(
    {"echo", "asr", "rvc", "denoise", "unload-voice", "unload-aligner"}
)


def _utcnow() -> datetime:
    """The clock, as an instant rather than as a string.

    `jobs.base.utcnow()` is `datetime.now(timezone.utc).isoformat()`: the same
    instant and the same rendering, collapsed because nothing else on this wire
    needs to compare two of them. A lease does — expiry is decided at read time —
    so the two halves are separate here and `Lease` renders with `.isoformat()`,
    which is what keeps `since` and `expires_at` the same shape as every other
    timestamp this server publishes.
    """
    return datetime.now(timezone.utc)


def require_ttl(ttl_seconds: int) -> int:
    """The ttl, or a refusal that states the range rather than implying it."""
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
    """One client's declared intention to keep using the resident model."""

    id: str
    model: str
    act: str
    client: str | None
    since: datetime
    expires_at: datetime
    #: Kept so a heartbeat extends by what was asked for rather than by a second
    #: number the client would have to send again — and would eventually send
    #: differently.
    ttl_seconds: int

    def expired(self, now: datetime) -> bool:
        return now >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """The five fields `/v1/activity` reports and a refusal carries.

        **No `model`**, and that is not an oversight: a lease is only ever on the
        RESIDENT model, which `/v1/activity` already reports as `resident.id`.
        Repeating it here would be one fact with two owners in a single document,
        able to disagree the day anything is read out of order (R1). The lease's
        own receipt does name it, because a receipt has no `resident` beside it.
        """
        return {
            "lease_id": self.id,
            "client": self.client,
            "act": self.act,
            "since": self.since.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }

    def receipt(self) -> dict[str, Any]:
        """`201` — what the holder gets back, which names what it leased."""
        return {
            "lease_id": self.id,
            "model": self.model,
            "client": self.client,
            "act": self.act,
            "since": self.since.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }


class Leases:
    """The one lease this server will hold at a time.

    Locked for `InFlight`'s reason: the routes are one event loop today, and a
    registry whose correctness depends on that staying true is a registry that
    breaks the first time something calls it from a thread.
    """

    def __init__(self, now: Callable[[], datetime] | None = None) -> None:
        # Injected rather than read from the module, so a test can advance a
        # clock instead of sleeping through a ttl. Nothing here ever sleeps.
        self._now = _utcnow if now is None else now
        self._lock = threading.Lock()
        #: The most recent lease, open or not. Kept after it closes so that a
        #: `404` can say WHICH of the two things happened to it.
        self._lease: Lease | None = None
        #: Why the lease above is closed, or None while it is open. Expiry is not
        #: recorded here — it is derived from the clock, so that a lease stops
        #: being open at its deadline whether or not anybody looked.
        self._closed: str | None = None

    # ------------------------------------------------------------------ reads

    def current(self) -> Lease | None:
        """The open lease, or None. Expiry is decided here, from the clock."""
        with self._lock:
            return self._open_locked()

    def _open_locked(self) -> Lease | None:
        lease = self._lease
        if lease is None or self._closed is not None:
            return None
        if lease.expired(self._now()):
            return None
        return lease

    # ----------------------------------------------------------------- writes

    def open(
        self, *, model: str, act: str, client: str | None, ttl_seconds: int
    ) -> Lease:
        """Take the lease, or refuse naming who has it."""
        with self._lock:
            held = self._open_locked()
            if held is not None:
                raise leased_error(held, f"leasing {model!r}")
            since = self._now()
            lease = Lease(
                id=uuid.uuid4().hex,
                model=model,
                act=act,
                client=client,
                since=since,
                expires_at=since + timedelta(seconds=ttl_seconds),
                ttl_seconds=ttl_seconds,
            )
            self._lease = lease
            self._closed = None
            return lease

    def heartbeat(self, lease_id: str) -> Lease:
        """Push the deadline out by the ttl the lease was opened with."""
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
        """Give the card back. The polite half of expiry, and the usual one."""
        with self._lock:
            held = self._open_locked()
            if held is None or held.id != lease_id:
                raise self._unknown_locked(lease_id)
            self._closed = "it was released"

    # ------------------------------------------------------------- refusals

    def refuse_if_leased(self, job_type: str) -> None:
        """Refuse a job that would evict the leased model, before the lane.

        Called at the job door rather than inside `Residency`, which is the same
        placement `server_busy` has and for the same reason: this is an ADMISSION
        question, and admission is answered where submissions arrive. The
        residency's job is to be the authority on what is on the card, not on who
        is allowed to ask for it to change.
        """
        held = self.current()
        if held is None:
            # Nothing to protect, so nothing to decide — including for a job
            # type this module has never heard of. The question below only has
            # consequences while somebody is mid-run.
            return
        if job_type in KEEPS_THE_RESIDENT_MODEL:
            return
        if job_type not in EVICTS_THE_RESIDENT_MODEL:
            # A job type this module has no ruling about, asked for while a
            # lease is open. Admitting it would silently reopen the hole and
            # refusing it would report a reason nobody decided, so it says
            # exactly what is wrong. `tests/test_leases.py` proves every type
            # this build knows is in one of the two sets, so reaching this in
            # production means a type was added without the ruling — and that is
            # the one moment the ambiguity costs somebody a run.
            raise ApiError(
                500,
                "lease_scope_unknown",
                f"{job_type!r} is a job type crucible/leases.py has no ruling "
                "about: nothing says whether it can take the resident model off "
                "the card, so this server cannot tell whether the open lease "
                "should refuse it. Add it to EVICTS_THE_RESIDENT_MODEL or to "
                "KEEPS_THE_RESIDENT_MODEL",
                {"type": job_type},
            )
        raise leased_error(held, f"a {job_type} job")

    def _unknown_locked(self, lease_id: str) -> ApiError:
        """`404`, saying which kind of gone this lease is when it can tell."""
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
                f"POST /v1/models/{lease.model}/lease — a client may re-lease a "
                "model it let go of, provided nobody else took it meanwhile",
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
    """`409 model_leased` — the one refusal, whoever asked for it.

    A loader and a second lease get the SAME code and the same details, because
    they are the same fact: somebody has said they are mid-run on this model.
    Splitting it into two codes would make a client handle one shape twice.
    """
    who = "an unnamed client" if lease.client is None else repr(lease.client)
    return ApiError(
        409,
        "model_leased",
        f"{lease.model!r} is leased by {who} for {lease.act!r} since "
        f"{lease.since.isoformat()}, until at least {lease.expires_at.isoformat()} "
        f"— so {what} is refused rather than taking the model off the card "
        "underneath a run in progress. A lease is the client saying it intends "
        "more work on this model; wait for it to expire or be released, or use "
        "another server. GET /v1/activity reports it as `lease`",
        lease.to_dict(),
    )
