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
gone the moment its previous chat returned, and be answered `model_not_resident`
until somebody submits another `load-model`. That is not a bug in this rule; it
is the bill for not stating an intention, and the fix is a lease at BookForge's
door, never an exception here.

THE TRIGGER IS A HOLDER LETTING GO — AND A LOAD IS NOT A HOLDER LETTING GO
--------------------------------------------------------------------------
`load-model` and `load-voice` exist to make something resident and nothing else.
Their whole content is *"be resident"*, so their own completion cannot be the
moment the card is cleared: the thing would be gone before the operator's next
request, and neither the chat door nor the streaming door ever loads
(PHASE2-LLM.md section 5, PHASE3-TTS.md section 6), so nothing downstream could
bring it back. A server whose `load-model` is a no-op is not a stricter server,
it is a broken one.

That is not an exception to the rule — it is the rule read correctly. A load is
the *start* of a resident thing's life. What ends it is the last holder letting
go, and the doors below are the ones that have to say so:

# RULING OWED: `load-model` and `load-voice` are the doors that must lease.
#   Until a load can open a lease in the same job — atomically, so nothing can
#   slip between "it is resident" and "somebody holds it" — a load that is never
#   used sits on the card until the next thing finishes. The four facts cannot
#   see an operator who typed `crucible load` and walked away, because that
#   operator declared nothing. The lease is the declaration, and the load door is
#   where it is missing.
# RULING OWED: the render door (`tts`) must be able to lease A VOICE.
#   A render loads its own voice, so a book rendered as one job loads once and
#   unloads at the end, which is right. A book rendered as twenty jobs reloads
#   twenty times, and the lease cannot help because `POST /v1/models/{id}/lease`
#   leases the resident MODEL — `Leases.open` is refused for a resident voice.
# RULING OWED: the `align` door must be able to lease AN ALIGNER.
#   Same shape, and sharper: the resident aligner exists precisely so that
#   hundreds of chunks pay one load (PHASE4-AUDIO.md section 2). Within one job
#   they still do. Across a book aligned chapter by chapter they now do not.
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
**claims the card** for the duration, exactly as a render does, and re-reads the
four facts under that claim. A job that reaches the lane while the claim is up is
refused `engine_in_use` by `Residency._refuse_mutation_if_claimed` rather than
racing a dying engine. The window that remains is the one this server already
has everywhere: a job whose `preflight` passed before the claim went up and whose
`enqueue` landed after the re-read fails loudly at the mutation instead of being
refused at the door. That is R3-shaped (a loud wrong answer, never a quiet one)
and it is not new — the same window exists between `preflight` and a streaming
session opening.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from .errors import JobError

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
#: docstring's RULING OWED block — this is a statement about what a load MEANS,
#: and the way out of it is a lease at the load door, not a second name here.
#:
#: Listed rather than derived, and `tests/test_settle.py` proves every name in it
#: is a job type this build knows, so a rename is a failing test rather than a
#: silently-never-exempt loader.
LEAVES_IT_RESIDENT: frozenset[str] = frozenset({"load-model", "load-voice"})


@dataclass(frozen=True)
class Held:
    """Why the card was not cleared: which fact, and who it names.

    A pair rather than a sentence, because the same value is read twice — once to
    decide, once to say — and a decision made on a string is a decision nobody
    can test.
    """

    fact: str
    who: str

    def __str__(self) -> str:
        return f"{self.fact} holds it: {self.who}"


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
            return Held("a job", f"{job.type} {job.id} ({job.status})")
        lease = self._leases.current()
        if lease is not None:
            who = "an unnamed client" if lease.client is None else repr(lease.client)
            return Held("a lease", f"{who} for {lease.act!r}, until {lease.expires_at.isoformat()}")
        claim = self._residency.claimed_by
        if claim is not None and claim != SETTLEMENT_HOLDER:
            return Held("the claim", claim)
        chats = len(self._inflight)
        if chats:
            return Held("a chat", f"{chats} completion(s) in flight")
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
            if self.holder(excluding_job=excluding_job) is not None:
                return None
            if self._residency.resident is None:
                return None
            try:
                # `may_mutate=True` binds the claim to THIS thread, which is what
                # lets the unload below through `_refuse_mutation_if_claimed`
                # while every other thread is refused by name.
                self._residency.claim(SETTLEMENT_HOLDER, may_mutate=True)
            except JobError:
                # Somebody claimed the card between the read above and here. They
                # are using it, which is the answer this was asking for.
                return None
            try:
                if self.holder(excluding_job=excluding_job) is not None:
                    # Admitted while the claim was going up. The card is spoken
                    # for again and the next release will ask again.
                    return None
                # Re-read UNDER the claim, and unload what is there rather than
                # what was there: an id read before the claim could name an
                # engine something else has since replaced, and `unload` answers
                # a stale id with `KeyError` rather than with the wrong engine.
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

    def settle_for_job(self, job: Any) -> Settled | None:
        """The lane's trigger. **Never the event loop.**

        A load is not a holder letting go (module docstring), so it is the one
        job whose end asks nothing.
        """
        if job.type in LEAVES_IT_RESIDENT:
            return None
        return self.settle(
            f"job {job.id} ({job.type}) finished", excluding_job=job.id
        )
