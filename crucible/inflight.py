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

So this is a RECORD, not a claim. It gates nothing, refuses nothing and reserves
nothing. `slots.accelerated.accepts_work` stays true while chats are in flight,
because the server really will accept more. `/v1/activity`'s own contract — *"it
reports and nothing else... display and admission are different questions and
only one may be answered from a poll"* — is what makes that the right shape
rather than a compromise.

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
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from .capability import CLASSES
from .errors import ApiError
from .jobs.base import utcnow

#: The header a client names its act in. One spelling, exported, because the
#: refusal message and the reader must not disagree about it.
ACT_HEADER = "X-Crucible-Act"

#: The acts a client may name: exactly the capability classes, because those are
#: the names `GET /v1/capability` already answers with and a second vocabulary
#: would be a second owner of what an act is called.
ACT_NAMES: frozenset[str] = frozenset(entry.name for entry in CLASSES)


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
    if act not in ACT_NAMES:
        raise ApiError(
            400,
            "unknown_act",
            f"{act!r} is not an act this server knows. {ACT_HEADER} must name a "
            f"capability class: {sorted(ACT_NAMES)}. It is refused rather than "
            "recorded because a bench showing the wrong act name is worse than "
            "one showing none — send no header if you would rather not say",
            {"act": act, "known": sorted(ACT_NAMES)},
        )
    return act


@dataclass(frozen=True)
class Entry:
    """One chat completion, while it is happening."""

    id: int
    act: str | None
    model: str
    client: str | None
    since: str

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

    @contextmanager
    def tracked(
        self, *, act: str | None, model: str, client: str | None
    ) -> Iterator[Entry]:
        entry = Entry(
            id=next(self._ids), act=act, model=model, client=client, since=utcnow()
        )
        with self._lock:
            self._entries[entry.id] = entry
        try:
            yield entry
        finally:
            # A `finally` rather than a happy-path removal: a caller that
            # disconnects mid-completion, an engine that dies and an abort all
            # end here, and an entry that outlives its request would make the
            # server look permanently busy with work that stopped.
            with self._lock:
                self._entries.pop(entry.id, None)

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            entries = sorted(self._entries.values(), key=lambda e: e.id)
        return [entry.to_dict() for entry in entries]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
