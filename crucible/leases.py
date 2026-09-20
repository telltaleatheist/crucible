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
refuse. It says exactly one thing: **while this is open, nothing may take the
leased thing off the card.** `slots.accelerated.accepts_work` is untouched,
because a leased server really will take a render that does not need the card's
contents to change — and it is `POST /v1/jobs` that still decides, as ever (R5).

It does not gate chats. Chats are what it protects; a lease that blocked them
would protect the run from itself.

ONE AT A TIME, PER SERVER. There is one card and one resident thing, so there is
one lease. A second is refused `leased` naming the holder — the same refusal a
loader gets, because it is the same fact.

A LEASE NAMES THE RESIDENT THING, OF ANY KIND
---------------------------------------------
**Extended 2026-09-14**, and it is the same hole one room along. A voice and an
aligner are resident kinds too (`crucible/residency.py`), and the unload ruling
clears the card the moment nothing holds it — so a book rendered as ONE `tts` job
loaded its voice once, while the same book rendered **chapter by chapter**, which
is how the app actually works, paid a narrator load per chapter. A book aligned
chapter by chapter paid an aligner load per chapter, which is worse: the resident
aligner exists precisely so hundreds of chunks pay one load (PHASE4-AUDIO.md
section 2). The lease is the thing that prevents exactly that, and until tonight
it could not reach either kind.

So a lease names the **resident thing**, whatever kind it is, and
`POST /v1/models/{id}/lease` takes a voice id or an aligner id as readily as a
model id. **The server supplies the kind; the client never states one**, and no
second route family is needed, because at the moment a lease is taken there is
exactly ONE candidate: the card holds one thing. Model ids and voice ids are
genuinely separate namespaces — `Residency.is_resident` takes a kind precisely
because nothing stops a voice being called `qwen3.5-9b` — but a collision cannot
reach this door, because only one of the two colliding things can be on the card,
and the lease is only ever on what is on it. Adding a `kind` to the body would
therefore be a field with no question to answer, and a second thing able to
disagree with `resident.kind` (R1).

WHAT A LEASE REFUSES IS DERIVED, NOT LISTED
-------------------------------------------
With three kinds the pairs are twenty-odd, and a hand-written list of them is a
fact with as many owners as it has rows. So each job type declares what it does
to the card — `CARD_EFFECTS` below — and `Lease.evicted_by` derives the answer
for every (lease kind, job) pair from those two facts. That is what makes `tts`
under a VOICE lease for the voice it names an **admission** rather than a
refusal: it reuses what is resident instead of loading, which is the whole point
of holding the lease. The same derivation, unchanged, still refuses `tts` under a
MODEL lease, because there the render really does evict.

EXPIRY IS READ, NEVER SWEPT. A lease past `expires_at` is simply not open. There
is no background task, no timer and nothing to cancel: every read compares the
stored instant to the clock, so a client that dies mid-run stops blocking the
card the moment its ttl runs out, whether or not anything was watching. A
heartbeat is what a live client sends instead of dying.

IN MEMORY, AND A RESTART FORGETS. A lease is worth exactly as much as the
residency it protects, and a restarted server holds nothing — so carrying a
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
from .residency import KIND_ALIGN, KIND_DENOISE, KIND_LLM, KIND_NOUNS, KIND_TTS

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


@dataclass(frozen=True)
class CardEffect:
    """What one job type does to whatever is on the card.

    Two facts and no opinion about leases, which is what lets one table answer
    every (lease kind, job) pair without anybody writing the pairs down. Both are
    decided by **reading what the job does to `Residency`**, never by what its
    name suggests — that reading is in the comment beside each row.
    """

    #: The kind this job MAKES resident, or None if it makes nothing resident.
    #: Loading anything evicts whatever was there, of any kind: one card, one
    #: resident thing (`Residency._evict`).
    makes_resident: str | None = None
    #: True when this job REUSES a resident thing of `makes_resident` whose id it
    #: names, instead of restarting it. `tts` and `align` do
    #: (`render.py`'s `_make_resident`, `align/__init__.py`'s `_session`); the
    #: two loaders deliberately do not — `load-voice` of the resident voice is a
    #: full narrator restart, and `Residency.load` evicts before it starts.
    reuses_what_it_names: bool = False
    #: The kind this job can TAKE OFF the card by name, or None. An unloader can
    #: only ever unload its own kind, and with another kind resident it refuses
    #: `*_not_resident` on its own before reaching anything.
    takes_off: str | None = None


#: What each job type this build knows does to the card. **The one owner of that
#: fact** — `tests/test_leases.py` proves every name in `ALL_JOB_TYPES` has a row
#: here, so a job type added later that touches the card is a failing test rather
#: than a silently reopened hole, and at runtime an unruled type asked for while a
#: lease is open is answered `lease_scope_unknown` rather than guessed at.
CARD_EFFECTS: dict[str, CardEffect] = {
    # Loads a model. `Residency.load` calls `_evict` unconditionally — reloading
    # the resident model really is a restart — so it never reuses.
    "load-model": CardEffect(makes_resident=KIND_LLM),
    # Loads a voice, the same way and for the same reason: a Higgs v3 voice
    # change IS a worker restart (`Residency.load_voice`).
    "load-voice": CardEffect(makes_resident=KIND_TTS),
    # A render LOADS ITS VOICE if it is not already resident, so under a model
    # lease it evicts exactly as `load-voice` does — leaving it out would have
    # left the hole open in the shape clients actually hit it, since BookForge
    # renders by submitting `tts`, not by loading a voice first. But
    # `_make_resident` REUSES a voice it finds resident under its own id, which
    # is what makes a chapter-by-chapter book under a voice lease one load.
    "tts": CardEffect(makes_resident=KIND_TTS, reuses_what_it_names=True),
    # Loads an aligner, the third resident kind, and reuses one it finds under
    # its own id (`AlignJobType._session`) — the reason the resident aligner
    # exists at all (PHASE4-AUDIO.md section 2).
    "align": CardEffect(makes_resident=KIND_ALIGN, reuses_what_it_names=True),
    # Loads a separator, the FOURTH resident kind (2026-09-15), and reuses one it
    # finds under its own id (`DenoiseJobType._session`) — which is the whole
    # reason the resident separator exists: a book is ~44 blocks and one load.
    # It is `align`'s row for `align`'s reason and not a new idea.
    "denoise": CardEffect(makes_resident=KIND_DENOISE, reuses_what_it_names=True),
    # The four unloaders, each of which can reach its own kind and no other.
    "unload-model": CardEffect(takes_off=KIND_LLM),
    "unload-voice": CardEffect(takes_off=KIND_TTS),
    "unload-aligner": CardEffect(takes_off=KIND_ALIGN),
    "unload-denoiser": CardEffect(takes_off=KIND_DENOISE),
    # `echo` never touches the accelerator at all.
    "echo": CardEffect(),
    # `asr` and `rvc` run the accelerator guard with **no** `reclaimable_bytes`,
    # which is their own deliberate note: they never unload somebody's resident
    # engine to make room, they refuse instead. `denoise` USED to be in this
    # sentence and moved up with the residency ruling — it now loads over a
    # previous resident exactly as `align` does.
    "asr": CardEffect(),
    "rvc": CardEffect(),
    # `align-longform` LOADS AN ALIGNER AND STILL LEAVES NOTHING RESIDENT, which
    # is the one row here that needs its reasoning written down because the
    # obvious ruling is wrong.
    #
    # It runs the same Qwen3 weights as `align` and it is not
    # `makes_resident=KIND_ALIGN`: it starts and stops its OWN `WorkerSession`
    # (`jobs/alignlongform/stages.py`) rather than putting one in the holder, and
    # the `stop` is in a `finally`, so the card is back before the job answers.
    # Nothing is left for a later job to reuse, and — the half that matters for
    # leases — nothing of somebody else's is evicted to make room for it.
    #
    # Deliberate rather than incidental. A long-form align is ONE pass over one
    # book, so the weights are read once either way; borrowing the resident
    # holder would buy nothing and would evict whatever a client had loaded,
    # which is exactly what a lease exists to prevent.
    "align-longform": CardEffect(),
}


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
    """One client's declared intention to keep using the resident thing."""

    id: str
    #: Which resident kind it holds — `llm`, `tts` or `align`. Read off
    #: `Residency.resident` at the open, never sent by the client: the card holds
    #: one thing, so there is nothing for the client to disambiguate.
    kind: str
    #: The resident thing's id. Named `subject` rather than `model` because it is
    #: a voice id as often as a model id, and a field called `model` holding
    #: `mistborn` is a fact that lies to everything downstream of it.
    subject: str
    act: str
    client: str | None
    since: datetime
    expires_at: datetime
    #: Kept so a heartbeat extends by what was asked for rather than by a second
    #: number the client would have to send again — and would eventually send
    #: differently.
    ttl_seconds: int

    @property
    def noun(self) -> str:
        """`model` / `voice` / `aligner`, for a sentence a reader can act on."""
        return KIND_NOUNS[self.kind]

    def expired(self, now: datetime) -> bool:
        return now >= self.expires_at

    def evicted_by(self, job_type: str, model: str | None) -> bool:
        """Would this job take THIS lease's subject off the card?

        The whole of what a lease refuses, derived from `CARD_EFFECTS` rather
        than from a table of pairs. Two clauses, and each is one sentence of the
        residency's own rule:

        1. **A job that loads evicts whatever is there**, of any kind — one card,
           one resident thing — *unless* it is a job that reuses what it names
           and what it names is exactly this lease's subject. That exception is
           the point of the whole extension: a `tts` render of the leased voice,
           or an `align` on the leased aligner, runs against what is already on
           the card and is admitted.
        2. **An unloader reaches its own kind and no other.** With another kind
           resident it refuses `*_not_resident` on its own, so refusing it
           `leased` would report the wrong reason for the right outcome.

        `model` is the job's subject as `resolve_model` settled it — a voice id
        for `tts`, an aligner id for `align`. It is passed rather than looked up
        because the job does not exist yet: this is asked at the door, before
        `store.create`.
        """
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
        """The six fields `/v1/activity` reports and a refusal carries.

        **No `subject`**, and that is not an oversight: a lease is only ever on
        the RESIDENT thing, which `/v1/activity` already reports as
        `resident.id`. Repeating it here would be one fact with two owners in a
        single document, able to disagree the day anything is read out of order
        (R1). The lease's own receipt does name it, because a receipt has no
        `resident` beside it.

        **`kind` IS here**, which is not the same call made twice. The id is the
        thing `resident.id` already owns; the kind is what decides *which jobs
        this lease refuses*, and these same six fields are the `details` of every
        `409 leased` — a document with no `resident` beside it at all. A bench
        shown "leased" and refused a `load-voice` can say why from the refusal it
        was handed, instead of needing a second read of a server whose card may
        have moved since.
        """
        return {
            "lease_id": self.id,
            "kind": self.kind,
            "client": self.client,
            "act": self.act,
            "since": self.since.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }

    def receipt(self) -> dict[str, Any]:
        """`201` — what the holder gets back, which names what it leased."""
        return {**self.to_dict(), "subject": self.subject}


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
        #: The id of a lapsed lease whose settlement has already been EVALUATED.
        #: See `forget_lapse()`; without it a lease that lapsed once would go on
        #: offering itself as a reason to clear the card for ever, and would
        #: eventually unload something loaded long after it died.
        self._lapse_handled: str | None = None

    # ------------------------------------------------------------------ reads

    def current(self) -> Lease | None:
        """The open lease, or None. Expiry is decided here, from the clock."""
        with self._lock:
            return self._open_locked()

    def lapsed_at(self) -> datetime | None:
        """When the stored lease LAPSED, or None if none did.

        A released lease is not a lapse: release triggers a settlement, so the
        card is already clear and there is nothing left unheld to date. A lapse
        is the other exit — "EXPIRY IS READ, NEVER SWEPT" (module docstring), so
        a lease whose ttl ran out with nothing asking again leaves the resident
        thing unheld at a moment NO code path observed. `expires_at` is that
        moment exactly, which is why this can be answered without a timer and
        without a sweep: the number was written when the lease was opened.

        Read-only. It reports a lapse; it does not end one, and nothing here
        clears `_lease`, because a reader arriving later must still be able to
        say when the card went quiet.
        """
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

    # ----------------------------------------------------------------- writes

    def open(
        self,
        *,
        kind: str,
        subject: str,
        act: str,
        client: str | None,
        ttl_seconds: int,
    ) -> Lease:
        """Take the lease, or refuse naming who has it.

        `kind` is the route's reading of `Residency.resident`, not anything the
        client sent: one card, one resident thing, so there is nothing to
        disambiguate and nothing for a client to get wrong.
        """
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
        """This lapse has been acted on; stop offering it as a reason.

        Called by the settlement AFTER it has evaluated a lapsed lease, whatever
        the outcome — including "there was nothing resident to clear". A lapse is
        a one-shot fact: its whole remaining job was to trigger one settlement.

        WITHOUT THIS IT IS A LOADED GUN. `_lease` is kept after expiry on
        purpose, so `_unknown_locked` can still say *"it expired at … and nothing
        heartbeated it"* rather than "unknown lease". But a lapsed lease that
        goes on being reported would, on the next idle tick after somebody loads
        a model WITHOUT a lease, be read as a holder letting go — and unload a
        model that had nothing to do with it. Evaluated once, then silent.

        `_closed` is deliberately untouched: that field is the SENTENCE a client
        gets back, and "it lapsed" is a worse answer than the expiry time it
        gives today.
        """
        with self._lock:
            if self._lease is not None:
                self._lapse_handled = self._lease.id

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

    def refuse_if_leased(self, job_type: str, model: str | None) -> None:
        """Refuse a job that would evict the leased thing, before the lane.

        Called at the job door rather than inside `Residency`, which is the same
        placement `server_busy` has and for the same reason: this is an ADMISSION
        question, and admission is answered where submissions arrive. The
        residency's job is to be the authority on what is on the card, not on who
        is allowed to ask for it to change.

        `model` is the job's subject, already resolved by the door. It is needed
        because the answer is not a property of the type alone: a `tts` render of
        the leased voice is admitted and a `tts` render of any other voice is
        refused, and that difference is the reason a book renders chapter by
        chapter for one load.
        """
        held = self.current()
        if held is None:
            # Nothing to protect, so nothing to decide — including for a job
            # type this module has never heard of. The question below only has
            # consequences while somebody is mid-run.
            return
        if job_type not in CARD_EFFECTS:
            # A job type this module has no ruling about, asked for while a
            # lease is open. Admitting it would silently reopen the hole and
            # refusing it would report a reason nobody decided, so it says
            # exactly what is wrong. `tests/test_leases.py` proves every type
            # this build knows has a row, so reaching this in production means a
            # type was added without the ruling — and that is the one moment the
            # ambiguity costs somebody a run.
            raise ApiError(
                500,
                "lease_scope_unknown",
                f"{job_type!r} is a job type crucible/leases.py has no ruling "
                "about: nothing says what it does to the card, so this server "
                "cannot tell whether the open lease should refuse it. Give it a "
                "row in CARD_EFFECTS",
                {"type": job_type},
            )
        if not held.evicted_by(job_type, model):
            return
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
    """`409 leased` — the one refusal, whoever asked for it.

    A loader and a second lease get the SAME code and the same details, because
    they are the same fact: somebody has said they are mid-run on this thing.
    Splitting it into two codes would make a client handle one shape twice.

    The code is `leased` and not `model_leased` because the leased thing is a
    voice or an aligner as often as a model, and a client branching on
    `model_leased` while narrator holds the card would be branching on a word
    that is not true of what it was refused for.
    """
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
