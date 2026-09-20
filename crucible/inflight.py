"""What this server is doing that is NOT a job.

A `POST /v1/jobs` submission is visible: it takes the lane, it has a row in
`GET /v1/activity`, and a second submission is refused with its id. A chat
completion is none of those things. It is a synchronous proxy to the resident
engine — no lane, no job, no record — so until this module existed a server
grinding through a 27B translation reported `running: []`, `accepts_work: true`
and, to any bench polling it, looked idle.

That is the same defect that was fixed for TTS streaming on 2026-09-13, in a
third place, and it is the one that matters most for Owen's naming ruling: you
cannot *"accurately represent the job that's running"* if the job never appears.

WHY THIS DOES NOT TAKE THE LANE, WHICH IS THE WHOLE DESIGN
---------------------------------------------------------
Making chat take the lane would have fixed the display and broken the server. A
vLLM engine BATCHES: two cleanup passes on one resident model genuinely run at
once and finish sooner than they would in sequence, and that is not an accident
of the current deployment, it is what the engine is for. Taking the lane would
serialise them to fix a reporting bug.

So this is a RECORD, not a claim. It takes no lane and reserves nothing.
`slots.accelerated.accepts_work` stays true while chats are in flight, because
the server really will accept more work of every other kind.
`/v1/activity`'s own contract — *"it reports and nothing else... display and
admission are different questions and only one may be answered from a poll"* —
is what makes that the right shape rather than a compromise.

AMENDED 2026-09-20: IT DOES NOW BOUND ONE THING, AND ONLY FOR A SERIAL ENGINE
-----------------------------------------------------------------------------
This paragraph used to end "it gates nothing, refuses nothing", and for a
batching engine it still behaves exactly that way — vLLM states no
`chat_concurrency`, so `engines.chat_admission()` returns None and the door
refuses nobody. Nothing above is reversed: taking the LANE would still serialise
work the engine exists to overlap.

What the paragraph missed is that not every engine overlaps. mlx-lm serves on a
`ThreadingHTTPServer`, so it ACCEPTS every connection and looks concurrent, and
then generates on one thread draining one queue. Twelve accepted requests are
one running and eleven waiting, with nothing on the wire saying so. Foundry's
clean pass died in that gap: 12 in flight, a 300 s client deadline, a request
that had not started when it passed.

So `crucible/api.py`'s chat door refuses `chat_queue_full` (503) past the
engine's OWN measured concurrency plus one, and `/v1/activity` publishes that
number as `chat.max_in_flight` with the basis beside it, so a client sizes its
pool from the server rather than discovering the limit as a starved socket. The
count is still a record; the ENGINE is what sets the bound, and an engine that
has not been measured sets none.

THE ACT IS THE CLIENT'S TO STATE
--------------------------------
Crucible cannot infer a simplify from a translate: both are a chat completion
against the same 27B, and the only difference is the prompt, which is Foundry's
(see PHASE9's class table). So the client says, in `X-Crucible-Act`, and a value
that is not a capability class is **refused by name** rather than recorded.

That refusal is the point. A silently-accepted typo would put a wrong act name on
a bench, which is precisely what Owen ruled out on 2026-09-13: *"they can't lie to
the user and say a translate job is running when it's actually a simplify job."*
A wrong name is worse than no name.

An ABSENT header records `null` — "it did not say" — the same rule `Job.client`
and a streaming session's holder already follow. The header is optional because
this door is OpenAI-shaped and a generic client cannot be expected to know
Crucible's vocabulary; BookForge and Foundry are expected to send it.
"""

from __future__ import annotations

import itertools
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from .capability import CLASSES
from .errors import ApiError
from .jobs.base import utcnow

#: The header a client names its act in. One spelling, exported, because the
#: refusal message and the reader must not disagree about it.
ACT_HEADER = "X-Crucible-Act"

#: How many recent completion durations are kept for `retry_after()`. Small on
#: purpose: what a refused caller wants to know is how long the work in front of
#: it takes NOW, and a long window would answer with a model that was unloaded
#: an hour ago.
RECENT_DURATIONS = 20

#: The acts a client may name: exactly the capability classes, because those are
#: the names `GET /v1/capability` already answers with and a second vocabulary
#: would be a second owner of what an act is called.
ACT_NAMES: frozenset[str] = frozenset(entry.name for entry in CLASSES)


def require_act_name(act: str, source: str, advice: str = "") -> str:
    """The act, or a refusal by name. One validator, one vocabulary.

    `source` names where the act came from — the header here, the lease body in
    `crucible/leases.py` — so the sentence tells the caller which thing to fix.
    A second copy of "is this a capability class" would be a second place for
    the vocabulary to drift, and the two doors record the same field.
    """
    if act not in ACT_NAMES:
        raise ApiError(
            400,
            "unknown_act",
            f"{act!r} is not an act this server knows. {source} must name a "
            f"capability class: {sorted(ACT_NAMES)}. It is refused rather than "
            "recorded because a bench showing the wrong act name is worse than "
            f"one showing none{advice}",
            {"act": act, "known": sorted(ACT_NAMES)},
        )
    return act


def read_act(headers: Any) -> str | None:
    """The act this request declares, or None because it did not say.

    Raises rather than guessing on an unknown name — see the module docstring.
    """
    raw = headers.get(ACT_HEADER)
    if raw is None:
        return None
    act = raw.strip()
    if act == "":
        return None
    return require_act_name(
        act, ACT_HEADER, " — send no header if you would rather not say"
    )


@dataclass(frozen=True)
class Entry:
    """One chat completion, while it is happening."""

    id: int
    act: str | None
    model: str
    client: str | None
    since: str
    #: `time.monotonic()` when this completion opened. `since` is the wall clock
    #: a reader sees; this is what a DURATION is measured from, because the wall
    #: clock can step and a negative completion time would be reported as fact.
    #: Not in `to_dict()`: it is an implementation detail of the recent-duration
    #: record below, not part of the activity contract.
    started: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            # Null means the client did not say. Never a guess: Crucible cannot
            # tell a simplify from a translate, and inventing one here would make
            # a bench confidently wrong about what is running.
            "act": self.act,
            "model": self.model,
            "client": self.client,
            "since": self.since,
        }


class InFlight:
    """Every chat completion currently open on this server.

    Locked even though the routes are one event loop: `_proxy_stream` hands a
    body to a background task, and a registry whose correctness depends on
    nothing ever calling it from a thread is a registry that breaks the first
    time something does.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[int, Entry] = {}
        self._ids = itertools.count(1)
        #: Seconds each of the last `RECENT_DURATIONS` completions took. Empty
        #: until this server has finished one, which is why `retry_after()`
        #: answers None rather than a number on a cold server.
        self._recent: list[float] = []

    def open(self, *, act: str | None, model: str, client: str | None) -> Entry:
        """Record a completion that has started. Pair it with `close`.

        THE PAIR EXISTS BECAUSE ONE COMPLETION OUTLIVES ITS HANDLER. A streamed
        chat returns a `StreamingResponse` and the tokens are relayed afterwards,
        so `with tracked(...)` around the handler stopped counting at the moment
        the relay began — a reporting wart while nothing acted on the count, and
        a real hazard since 2026-09-14, when "no chat is in flight" became one of
        the four facts that let the card be cleared (`crucible/settle.py`).
        Unloading a model out from under a stream that is still producing tokens
        is exactly the eviction this server refuses to do to anyone else.

        So the streamed door opens the record here and closes it where the relay
        really ends (`_RelayResponse.__call__`'s `finally`, which runs on every
        path including a caller who walked away). The synchronous door keeps the
        context manager below, which is these two with a `try`.
        """
        entry = Entry(
            id=next(self._ids),
            act=act,
            model=model,
            client=client,
            since=utcnow(),
            started=time.monotonic(),
        )
        with self._lock:
            self._entries[entry.id] = entry
        return entry

    def close(self, entry: Entry) -> None:
        """This completion is over. Idempotent: closing twice is not an error."""
        with self._lock:
            removed = self._entries.pop(entry.id, None)
            if removed is not None and removed.started > 0.0:
                # HOW LONG COMPLETIONS ACTUALLY TAKE ON THIS ENGINE, kept only so
                # that a `Retry-After` can be a measurement instead of a guess.
                # `_rate_limited` states the rule this follows: a Retry-After is
                # copied when there is one and absent when there is not, NEVER
                # invented. An upstream's number belongs to the upstream; this
                # door's number has to come from somewhere, and the only honest
                # source is what this engine has been doing.
                self._recent.append(time.monotonic() - removed.started)
                del self._recent[:-RECENT_DURATIONS]

    @contextmanager
    def tracked(
        self, *, act: str | None, model: str, client: str | None
    ) -> Iterator[Entry]:
        entry = self.open(act=act, model=model, client=client)
        try:
            yield entry
        finally:
            # A `finally` rather than a happy-path removal: a caller that
            # disconnects mid-completion, an engine that dies and an abort all
            # end here, and an entry that outlives its request would make the
            # server look permanently busy with work that stopped.
            self.close(entry)

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            entries = sorted(self._entries.values(), key=lambda e: e.id)
        return [entry.to_dict() for entry in entries]

    def retry_after(self) -> int | None:
        """Seconds a refused caller should wait, or None when nothing is known.

        The MEDIAN of the recent completions on this engine, rounded up, floored
        at one second because `Retry-After: 0` reads as "immediately" and would
        turn a refusal into a spin. The median rather than the mean: one 27B
        translation among a run of short cleanups should not tell every refused
        caller to wait a minute.

        None on a server that has not finished a completion yet. The header is
        then absent, and absent is the honest answer — `_rate_limited` in
        `crucible/api.py` states the rule this follows.
        """
        with self._lock:
            recent = sorted(self._recent)
        if not recent:
            return None
        middle = recent[len(recent) // 2]
        return max(1, math.ceil(middle))

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
