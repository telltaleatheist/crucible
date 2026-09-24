"""When the last holder lets go, the card is cleared. Owen's ruling, 2026-09-14.

> **Owen, 2026-09-14:** *"Models should always be unloaded when we're done with
> them. Every time."*

This **overrules** PHASE5-APPS.md section 7, which proposed the opposite — *"no
idle unload, and the Servers row shows what is resident and for how long"* — on
the grounds that an idle unload is a 110-second reload the next time anyone
types. The proposal was answered by what happened: a 9B sat resident after the
Foundry proof until a person unloaded it, and Owen saw *"something loaded and
nothing happening"*. The card is not storage.

"DONE" IS A FACT, NOT A TIMER
-----------------------------
The obvious shape is a keep-warm window — `keepServerWarmMinutes`, which is what
Foundry had. It is the shape ARCHITECTURE.md's audit found seven times: **a fact
standing in for a guess**. Ninety idle seconds is evidence of nothing. N would be
tuned against one workload and silently wrong for the next, in both directions.

So there is no window, no timer and no config key. There are **four facts**, and
the card is cleared the moment the last of them goes false:

1. **the lane** — no job is running or queued (`JobStore`);
2. **the lease** — no client has said it intends a run (`crucible/leases.py`);
3. **the claim** — no streaming session holds narrator's wire
   (`Residency.claimed_by`);
4. **the chats** — no chat completion is in flight (`crucible/inflight.py`).

Each of those four is somebody's declared hold on the card, and together they are
the whole of what this server can know about being *used*. Nothing else is
consulted, and in particular nothing is measured against a clock.

`Residency.warming` is deliberately not a fifth: it is only ever set from inside
a load job, which is on the lane, so it is a sub-state of fact 1. Testing it
separately would be a guard against a bug rather than against a state — the same
ruling `JobStore.refuse_if_busy` made (ARCHITECTURE.md section 3.1).

WHAT THIS COSTS A CLIENT THAT DOES NOT LEASE
--------------------------------------------
Exactly what the lease was built for, and it must be said plainly rather than
discovered: **a run of chat completions with no lease open reloads its model.**
Foundry leases a whole cleanup and pays nothing. BookForge's doors do not lease
yet (`electron/ai-bridge.ts`'s `crucible` provider loads nothing and leases
nothing), so a BookForge cleanup against a Crucible today will find the model
gone the moment its previous chat returned, and be answered `not_resident`
until somebody submits another `load-model`. That is not a bug in this rule; it
is the bill for not stating an intention, and the fix is a lease at BookForge's
door, never an exception here.

**The same bill, one kind along.** A book rendered as one `tts` job loads its
voice once; a book rendered CHAPTER BY CHAPTER — which is how Owen works — pays a
narrator load per chapter unless a lease is open on that voice, and a book
aligned chapter by chapter pays an aligner load per chapter unless a lease is
open on that aligner. Since 2026-09-14 a lease can name either
(`crucible/leases.py`), so the bill is avoidable by stating the intention; it is
not avoidable by this rule making an exception for a render, because "one more
chapter is coming" is the client's fact and nothing here can see it.

THE TRIGGER IS A HOLDER LETTING GO — AND A LOAD IS NOT A HOLDER LETTING GO
--------------------------------------------------------------------------
`load-model` and `load-voice` exist to make something resident and nothing else.
Their whole content is *"be resident"*, so their own completion cannot be the
moment the card is cleared: the thing would be gone before the operator's next
request, and neither the chat door nor the streaming door ever loads
(PHASE2-LLM.md section 5, PHASE3-TTS.md section 6), so nothing downstream could
bring it back. A server whose `load-model` is a no-op is not a stricter server,
it is a broken one.

AND ONLY A LOAD THAT GOT THERE (2026-09-18). The sentence above is about a load
that SUCCEEDED, and the code read it as a sentence about a load's TYPE.
`DELETE /v1/jobs/{id}` sets `cancel_requested`; neither loader read it, so
the engine came up anyway, the lane stamped the job `cancelled` because the flag
was set, and the exemption then let that job off settling BY NAME. The client
had been told `cancelled`, so it would never send an unload; the lane was empty,
no lease, no claim, no chat. A 21 GB model sat on the card with all four facts
false and nobody left who knew it was there — which is the exact overnight card
this ruling exists to prevent, arrived at through the ruling's own exemption.

So the exemption now reads type AND outcome (`Settlement.settle_for_job`), and
the loaders read `ctx.cancelled` on both sides of the load: before, to refuse to
start one nobody wants any more, and after, because `WorkerSession.start` has no
cancel hook and a DELETE has the whole of a load to land inside. A
cancelled load gets NO teardown of its own — it raises `JobCancelled` and the
settlement takes the card off it through the one unload door, exactly as
PHASE7-LANES.md says of a cancelled render.

That is not an exception to the rule — it is the rule read correctly. A load is
the *start* of a resident thing's life. What ends it is the last holder letting
go, and the doors below are the ones that have to say so:

# RULING OWED, SHARPENED 2026-09-14 — and half of what it used to say was wrong.
#   It read: `load-model` and `load-voice` must open a lease in the same job,
#   atomically, because "a load that is never used sits on the card until the
#   next thing finishes."
#
#   THE WRONG HALF. A lease would not free the operator who typed `crucible
#   load` and walked away, because **a lease is another holder** — it is fact 2.
#   A load that took one would hold the card for its whole ttl AND refuse
#   everybody else meanwhile, which is strictly worse than a load that holds it
#   quietly. Nothing here can free that card, because "is this operator done?"
#   is the one question this server cannot have an answer to, and
#   `LEAVES_IT_RESIDENT` is the ruling that a load's own end is not it. What that
#   operator has is `unload-model`, and that is the right shape for it.
#
#   THE HALF THAT IS STILL OPEN is narrower and is a real race: the WINDOW
#   between a load's `done` and its own client's `POST /v1/models/{id}/lease`. A
#   book rendered chapter by chapter must `load-voice` and THEN lease, because
#   the first chapter's `tts` job settles at its own end — a render is not
#   `LEAVES_IT_RESIDENT`, and making it so would put the stranded card back one
#   door along. In that window a third party's loader can evict what was just
#   loaded, and the holder learns about it as a `not_resident` on its lease.
#
#   IT NEEDS OWEN, because closing it is a new shape on the JOB wire rather than
#   a refinement of this rule: the load job would carry `lease: {act,
#   ttl_seconds}` in its params and its `done` would hand back a `lease_id` its
#   client must then heartbeat — the first time a job returns a handle with a
#   life of its own. That is a client-facing contract (the SDK, BookForge,
#   Foundry), not a settlement question, and it is not what tonight's regression
#   needed.
#
# CLOSED 2026-09-14: the render door and the `align` door CAN lease.
#   They lease by naming what they made resident, because a lease now names the
#   RESIDENT THING of any kind rather than the resident model
#   (`crucible/leases.py`, PHASE7-LANES.md section 5.2). A voice lease and an
#   aligner lease refuse every job that would evict them and ADMIT the job they
#   were taken for — a `tts` render of the leased voice, an `align` on the
#   leased aligner — so a book rendered chapter by chapter pays one narrator
#   load and a book aligned chapter by chapter pays one aligner load, which is
#   what they cost before this rule and what they must cost after it.
# RULING OWED: the streaming door is safe only because `load-voice` is.
#   A session claims the card, so a session keeps the voice; but the gap between
#   `load-voice` finishing and `POST /v1/tts/stream` opening is held by nothing.
#   It survives today only because a load is not a trigger. The honest fix is the
#   same lease.

WHERE THE UNLOAD RUNS, AND WHY IT TAKES THE CLAIM
-------------------------------------------------
**Never on the event loop.** `SubprocessEngine.stop()` SIGTERMs and waits up to
180 s, and a server that stops answering `/v1/activity` for three minutes is
indistinguishable from a dead one. So `settle()` is synchronous and is called
from a worker thread — `asyncio.to_thread` at the loop-side triggers, directly at
the one trigger that is already a thread (a streaming session closing).

Running off the loop means admission can move underneath it, so the settlement
**claims the card** for the duration, exactly as a render does — and since
2026-09-24 it reads the four facts and takes the claim as ONE step
(`Residency.claim_to_clear`), under the lock every client door records its hold
under. See "EVERY DOOR WAITS IT OUT" below.

WITH ONE EXCEPTION, AND IT IS NOT A LOOSENING (T6, 2026-09-15). An
`unload-model` / `unload-voice` / `unload-aligner` for **the very subject this
settlement is clearing** is not a second holder — it is this settlement, asked
for by name. On a live card the page reader loaded `dots-ocr`, read its page,
and its own `finally` unload landed milliseconds after the last chat completion
triggered the clearance; it came back `409 engine_in_use`, held by *"the
settlement clearing the card"*, and the failed tidy-up overwrote the page that
had just been read. So the claim now says what it is FOR (`clears=True`), the
three unload doors ask `Residency.being_cleared` first, and such a job is
admitted and waits the clearance out (`Residency.await_clearance`), ending
`done` on the empty card it asked for. Every other holder — a lease, a session,
a render's claim, a running job — refuses exactly what it refused before, under
exactly the name it used.

EVERY DOOR WAITS IT OUT (2026-09-24, Briefcase). The exception above turned out
to be the rule read too narrowly. On 1.0.25 Briefcase started a run straight
after a SIGINT'd one, and the first run's release set a clearance going. Inside
it the second run opened a lease (201, the model still published), sent a chat
(`model_not_resident`, unpublished a moment later) and submitted `load-model` —
`409 engine_in_use`, held by *"the settlement clearing the card"*. Nobody was
using the card. The clearance is a SIGTERM and a wait, it ends, and the answer
a client is owed is the one the card gives when it has: that is WEATHER, with a
stated budget, not a refusal (CLAUDE.md's hardening ruling, 2026-09-20).

So every client door that can arrive in a clearance waits it out, within
`CLEARANCE_TIMEOUT_SECONDS` (the engine's own SIGTERM deadline plus a margin),
and then answers from the settled card: the job door preflights and admits
against it (a `load-model` is `202` and ends `done`), the lease door says
`not_resident` because that is now true, the chat and decide doors say
`model_not_resident` instead of proxying to an engine being SIGTERMed, and the
streaming door opens or says `voice_not_resident`. A clearance that outlives
the budget is a wedge and is raised as one, `engine_in_use`, by name.

Waiting alone would only narrow the window, so the door's check and its hold are
made atomically against this module's check and claim (`Residency.settled_for`
and `Residency.claim_to_clear`, one lock). Either the settlement claims first
and the door waits, or the door's hold is recorded first and the settlement sees
it and declines.

AND THE WINDOW THAT USED TO REMAIN IS CLOSED. This paragraph used to say that a
job whose `preflight` passed before the claim went up and whose `enqueue` landed
after the re-read fails loudly at the mutation, and called it R3-shaped and not
new. Both halves are gone: the job door's preflight and enqueue now happen under
the lock and never inside a clearance, and the lane hands a queued job to
`running` in one step under its own lock (`JobStore._lane_lock`), so there is no
instant at which a job admitted before the clearance is invisible to `holder()`.
A job that is on the lane when a clearance would begin stops that clearance
from beginning; nothing on the lane has to wait for one. What is unchanged is
the window between a job's `preflight` and a streaming session opening — that is
two USERS of the card, not a clearance, and it is still answered by name.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from .errors import JobError
from .jobs.base import DONE
from .jobs.queue import busy_details

if TYPE_CHECKING:  # pragma: no cover - imports for annotations only
    from .inflight import InFlight
    from .leases import Leases
    from .residency import Residency

#: The name the settlement holds the card under. It appears in
#: `GET /v1/activity`'s `claim.held_by` and in the `engine_in_use` refusal a job
#: that raced it gets, so it is a sentence a reader can act on rather than an
#: internal token.
SETTLEMENT_HOLDER = "the settlement clearing the card"

#: The job types whose completion is NOT a moment to clear the card, because
#: making something resident is the whole of what they do. See the module
#: docstring's RULING OWED block — this is a statement about what a load MEANS.
#:
#: **A NAME HERE IS HALF THE EXEMPTION; the other half is `done`.** The
#: exemption says a load's whole content was *"be resident"*, and that is only
#: true of a load that got there. `Settlement.settle_for_job` therefore asks for
#: the type AND the outcome, and a `load-...` that ended `cancelled` or `failed`
#: settles like every other job.
#:
#: **`tts` and `align` are deliberately NOT here**, although both make something
#: resident: making it resident is not the whole of what they do, and a render
#: that left its voice on the card would strand it exactly as an unused load
#: does. What holds a voice across twenty chapters is a lease on that voice
#: (`crucible/leases.py`), which is the client saying more is coming — the fact
#: this file has no way to invent.
#:
#: Listed rather than derived, and `tests/test_settle.py` proves every name in it
#: is a job type this build knows AND one that `CARD_EFFECTS` agrees makes
#: something resident, so a rename is a failing test rather than a
#: silently-never-exempt loader.
LEAVES_IT_RESIDENT: frozenset[str] = frozenset({"load-model", "load-voice"})


@dataclass(frozen=True)
class Held:
    """Why the card was not cleared: which fact, who it names, and its own facts.

    A triple rather than a sentence, because the same value is read three times
    — once to decide, once to say, and since PHASE13-OPERATOR.md section 3.3
    once to put on the wire — and a decision made on a string is a decision
    nobody can test.

    `details` is that third reading, and it is **the holding fact's own shape,
    not a shape invented here**: a job's is `POST /v1/jobs`' `server_busy` body
    verbatim (`crucible/jobs/queue.py`'s `busy_details`), a lease's is the
    receipt `POST /v1/models/{id}/lease` hands back (`Lease.receipt`), and the
    other two have one field each because there is one fact to state. An app
    refused `server_busy` on an operator task shows the holder verbatim, and it
    can only do that if the fields it reads are the ones it already knows.
    """

    fact: str
    who: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.fact} holds it: {self.who}"

    def to_dict(self) -> dict[str, Any]:
        """For `/v1/activity`'s `resident.held_by`.

        `fact` and `who` are the two the refusals already print; `details` is
        the holding fact's OWN shape, unchanged from the class docstring's rule
        — a job's `server_busy` body, a lease's receipt — so a client that can
        read a refusal can read this without learning a second vocabulary.
        """
        return {"fact": self.fact, "who": self.who, "details": self.details}


@dataclass(frozen=True)
class Settled:
    """One unload, as the log and the triggering job's events report it."""

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
    """The one place that decides the resident thing has nothing holding it.

    ONE PLACE, deliberately. There are five moments a holder can let go — a job
    ending, a lease released, a lease expiring, a streaming session closing, the
    last chat returning — and each of them calls exactly this. Five copies of
    "is anything still using the card" would be five answers the day a sixth kind
    of holder arrives, which is the one-fact-two-owners shape ARCHITECTURE.md
    exists to stop.
    """

    def __init__(
        self,
        *,
        residency: "Residency",
        store: Any,
        leases: "Leases",
        inflight: "InFlight",
        log: Callable[[str], None] = _to_stderr,
    ) -> None:
        self._residency = residency
        self._store = store
        self._leases = leases
        self._inflight = inflight
        self._log = log
        #: When a SUCCEEDED load left something resident that nothing holds.
        #: See `unheld_since()`; the one state this module's ruling does not
        #: reach, recorded rather than inferred.
        self._unheld_since: datetime | None = None
        # One settlement at a time. Two threads finding an empty card together
        # would have the second one raise `KeyError` out of `Residency.unload`
        # for a subject the first already took off.
        self._lock = threading.Lock()
        #: The one-shot armed at the open lease's own `expires_at`, or None.
        self._deadline: asyncio.TimerHandle | None = None
        #: Settlements started by that one-shot, held so the loop cannot collect
        #: a task nobody awaits (`asyncio.create_task`'s documented hazard).
        self._running: set[asyncio.Task[Any]] = set()

    # -------------------------------------------------- the lease's own clock
    #
    # THE ONE PLACE A DEADLINE IS READ, AND IT IS NOT A POLICY. Four of the five
    # ways a holder lets go are edges this server sees: a job ends, a session
    # closes, a chat returns, a lease is released. The fifth has no edge — a
    # lease is *read* against the clock and never swept (`crucible/leases.py`),
    # so a client that crashed mid-run stops holding the card at a moment nothing
    # is watching, and its 21 GB would sit there until somebody happened to
    # submit something. That is precisely the overnight card Owen's ruling is
    # about.
    #
    # So the deadline is armed — at the lease's OWN `expires_at`, which is the
    # number the client chose and heartbeats forward, never an interval tuned
    # here. It is cancelled and re-armed at the three moments that move it
    # (open, heartbeat, release), and a firing that finds the lease still open
    # simply re-arms, which is what makes a missed heartbeat self-correcting.

    def arm_for_lease_expiry(self) -> None:
        """Re-arm the one-shot to the open lease's deadline. **Event loop only.**"""
        loop = asyncio.get_running_loop()
        handle, self._deadline = self._deadline, None
        if handle is not None:
            handle.cancel()
        lease = self._leases.current()
        if lease is None:
            return
        seconds = max(
            0.0,
            (lease.expires_at - datetime.now(timezone.utc)).total_seconds(),
        )
        self._deadline = loop.call_later(seconds, self._deadline_passed)

    def _deadline_passed(self) -> None:
        """**Event loop.** Ask, off the loop, then re-arm on what is left."""
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
            # A heartbeat that landed after this was armed leaves the lease open
            # and the settlement declining; the deadline it pushed out is the one
            # to wait for now.
            self.arm_for_lease_expiry()

        task.add_done_callback(finished)

    # ------------------------------------------------------------- the facts

    def holder(self, *, excluding_job: str | None = None) -> Held | None:
        """The first of the four facts that still holds the card, or None.

        `excluding_job` is the job asking. A job settles **while it is still the
        running job on the lane** — before its terminal event, so the note lands
        on a stream its client is still reading — so it must not find itself.
        """
        job = self._store.occupied_by_anything_but(excluding_job)
        if job is not None:
            return Held(
                "a job", f"{job.type} {job.id} ({job.status})", busy_details(job)
            )
        lease = self._leases.current()
        if lease is not None:
            who = "an unnamed client" if lease.client is None else repr(lease.client)
            return Held(
                "a lease",
                f"{who} for {lease.act!r}, until {lease.expires_at.isoformat()}",
                # The RECEIPT and not `to_dict()`: a refusal arrives with no
                # `resident` beside it, so the leased thing has to be named here
                # or the reader cannot say what is held (the same call
                # `Lease.to_dict`'s docstring makes about `kind`).
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
        return None

    # ------------------------------------------------------------ the ruling

    def settle(
        self, trigger: str, *, excluding_job: str | None = None
    ) -> Settled | None:
        """Clear the card if nothing holds it. **Never the event loop.**

        Returns what went away, or None — which is the honest answer for all
        three ways nothing happens: something still holds it, nothing was
        resident, or another thread got there first.
        """
        with self._lock:
            try:
                # THE FOUR FACTS AND THE CLAIM IN ONE STEP (2026-09-24,
                # Briefcase). `claim_to_clear` reads `holder()` under the lock
                # every client door records its hold under
                # (`Residency.settled_for`), so either a door's lease, chat or
                # job is seen here and nothing is claimed, or the door arrives
                # after the claim, sees the clearance and waits it out. This
                # used to be a read, a claim, and a re-read under the claim —
                # which left a door the gap between the re-read and the unload,
                # and Briefcase's lease landed in it.
                #
                # `may_mutate=True` (inside) binds the claim to THIS thread,
                # which is what lets the unload below through
                # `_refuse_mutation_if_claimed`. `clears=True` says what the
                # claim is FOR: an `unload-...` of the very thing this is taking
                # off the card is the same intent (T6), and every door arriving
                # meanwhile waits rather than being refused `engine_in_use`.
                claimed = self._residency.claim_to_clear(
                    SETTLEMENT_HOLDER,
                    held=lambda: self.holder(excluding_job=excluding_job),
                )
            except JobError:
                # `engine_still_stopping`: a process an earlier unload asked to
                # go is still on the card. There is nothing this can clear, and
                # the dying slot is already saying so to every load.
                return None
            if not claimed:
                # Something holds it, nothing is resident, or a session has the
                # claim. Each is the answer this was asking for.
                return None
            try:
                # Unload what is there rather than what was there: `unload`
                # answers a stale id with `KeyError` rather than with the wrong
                # engine, so the id is read under the claim.
                resident = self._residency.resident
                if resident is None:
                    return None
                self._residency.unload(resident.id)
            finally:
                self._residency.release(SETTLEMENT_HOLDER)
        settled = Settled(
            subject_id=resident.id, kind=resident.kind, trigger=trigger
        )
        # SAID, not merely done. A reader who finds the next request paying a
        # load must be able to see why the card was empty, and the log is the
        # only place a chat-triggered or lease-triggered unload can say it —
        # there is no job to carry an event.
        self._log(settled.line)
        return settled

    def unheld_since(self) -> datetime | None:
        """Since when has the resident thing been held by NOTHING? None if held.

        Two moments produce a resident-but-unheld card and this reports the
        later of them, because both are real events with real timestamps:

        1. **A load that succeeded** — `settle_for_job` stamps it above.
        2. **A lease that LAPSED** — `Leases.lapsed_at()`. Expiry is read and
           never swept, so no code path runs at the moment it happens; the
           lease's own `expires_at` is that moment, written when it opened.

        Every other way a holder lets go runs `settle()`, which clears the card,
        so there is nothing left to date.

        **Guarded by a live read of `holder()`**, so a stamp can never be
        reported while something actually holds the card: the stamp is history,
        the holder is now, and `None` from here means "held, or nothing is
        resident". That is also why nothing clears the stamp — it is only ever
        readable in the state that produced it.

        This is a READ. It starts no timer and ends no residency; what may be
        done about a card that has been unheld for a while is a ruling
        (`docs/BUG-HUNT-2026-09-20.md` §F.8), not this function.

        **IT DOES NOT TAKE `_lock`, and that is deliberate.** A settlement holds
        that lock for the whole of an unload, and `SubprocessEngine.stop()`
        waits up to 180 s for SIGTERM — so a reader that queued behind it would
        make `/v1/activity` hang for three minutes, which this module's own
        docstring calls indistinguishable from a dead server. The four sources
        below each lock themselves; what this cannot promise is that they were
        read in the same instant, and it does not need to: this is the bench
        read, and *"display and admission are different questions and only one
        may be answered from a poll"*.
        """
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
        """The public name for `holder()`: what holds the card right now.

        One owner of this fact, as the class docstring insists — `/v1/activity`
        reads it here rather than re-deriving four facts of its own, which is
        the one-fact-two-owners shape `docs/ARCHITECTURE.md` says this repo
        keeps finding.

        No `_lock`, for the reason `unheld_since()` gives at length: a bench
        read must never queue behind a 180 s unload.
        """
        return self.holder()

    def settle_for_lapsed_lease(self) -> Settled | None:
        """A lease that ran out is a holder letting go that NOBODY OBSERVED.

        THE ONE HOLDER WHOSE END FIRES NOTHING. Every other way a holder lets go
        is an edge this server sees — a job ends, a session closes, a chat
        returns, a lease is RELEASED — and each of those calls `settle()`. A
        lease that simply runs out is read and never swept (`crucible/leases.py`),
        so the card sits resident and unheld with no code path having run.

        That was harmless while a lease was only a refusal: the thing it
        protected had been loaded by somebody who was still expected to unload
        it. It stops being harmless the moment a lease is what HOLDS a load
        (`lease` on `load-model`), because then the lease running out is the
        whole of the client's disappearance, and nothing else is coming.

        **This is not the timer this module rejected.** That rejection was about
        answering *"is this operator done?"*, which is a guess. A ttl is not a
        guess — it is a number the client stated, about itself, and extended
        every time it heartbeated. Acting when it runs out is taking the client
        at its word, which is the opposite of inventing a policy.

        Evaluated ONCE per lapse, whatever the outcome: see
        `Leases.forget_lapse`. A lapse that kept being offered would, on the
        first idle tick after somebody loaded a model without a lease, be read
        as a holder letting go and unload a model that had nothing to do with it.
        """
        if self._leases.lapsed_at() is None:
            return None
        try:
            return self.settle("a lease lapsed and nothing heartbeated it")
        finally:
            # In the `finally` so that a settlement which RAISES still spends the
            # lapse. A lapse retried for ever against a card it cannot clear is
            # the loaded gun the docstring above describes.
            self._leases.forget_lapse()

    def settle_quietly(self, trigger: str) -> Settled | None:
        """`settle`, for a caller that has nothing to fail. **Never the loop.**

        A CLEANUP FAILURE IS NOT AN OPERATION FAILURE. A streaming session that
        closed did close; a lease that was released is released; a chat that
        answered answered. An engine that then refuses to stop is a real fact and
        is said, loudly, in the server log — it does not turn any of those three
        into an error the client sees, and it must never take down the stream
        watchdog, which is the one thread with no caller to report to.
        """
        try:
            return self.settle(trigger)
        except Exception as exc:
            self._log(
                f"could not clear the card ({trigger}): {type(exc).__name__}: {exc}"
            )
            return None

    def settle_for_job(self, job: Any, outcome: str) -> Settled | None:
        """The lane's trigger. **Never the event loop.**

        A load that SUCCEEDED is not a holder letting go (module docstring), so
        it is the one job whose end asks nothing. A load that ended any other
        way asks like everything else, because what it left on the card is a
        thing nobody is coming back for.

        `outcome` is the status the lane is ABOUT to stamp, handed over rather
        than read off `job.status` — which at this moment is still `running`.
        The lane settles BEFORE `_finish` on purpose, so that the note lands on
        a stream the client is still reading (`jobs/queue.py:_execute`), and a
        `job.status` read here would be a fact that has not been written yet:
        every load would find itself un-exempt and clear its own card.
        """
        if job.type in LEAVES_IT_RESIDENT and outcome == DONE:
            # THE ONE MOMENT THIS MODULE'S RULING DOES NOT REACH, and now the
            # one moment it is written down. A load that succeeded is not a
            # holder letting go, so the card is left loaded on purpose — and
            # until something leases it, chats it or renders on it, it is
            # resident and held by NOTHING. On 2026-09-20 a hosted runner was
            # stopped one second after its `load-model` reached `done`; the
            # cancel was refused `job_not_cancellable` (the job was terminal),
            # no lease was ever opened, and a 21 GB model sat on the card with
            # all four facts false until a person noticed.
            #
            # This does not decide anything — deciding is `docs/BUG-HUNT`'s
            # §F.8 and needs Owen. It makes the state VISIBLE, which is the
            # half that needs no ruling: `/v1/activity` can now say "resident,
            # held by nothing, since 18:29:37Z" and a reconciler on either side
            # can act on a fact instead of a poll's timing.
            self._unheld_since = datetime.now(timezone.utc)
            return None
        return self.settle(
            f"job {job.id} ({job.type}) finished", excluding_job=job.id
        )
