"""The hold a deploy takes on this server before it restarts it.

A deploy used to ask `GET /v1/activity` whether the server was idle and then spend
half a minute downloading and installing before the restart. Work admitted in that
half minute died with the old server: B-Side's phone song, 2026-10-04, admitted 8 s
after an idle answer and killed by the 1.0.102 restart.

So the question is asked the other way round. `POST /v1/server/updating` takes the
hold FIRST - from that moment every door that creates work (a job, a queued call, a
chat completion, a queue session, a TTS stream, an operator task) refuses with a
retryable `503 server_updating` - and only then reads whether anything is still in
flight. Idle: the hold stands and the deploy may restart. Busy: the hold is let go in
the same breath and the deploy is told what is working. Either way there is no window
in which work is admitted and then killed.

The check lives in the owners that create the work, not at the route doors: an
admission can await (a model settling) between its door and the moment the job
exists, and a door check would let that job through after the hold was taken.

Every hold has an owner and a way out: the restart ends it (it lives in memory), the
deploy lets it go with `DELETE` when its install fails, and it lapses by itself at
its `until` if neither happens - a deploy that died mid-way does not leave a server
refusing work forever.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from . import clock
from .errors import ApiError

SERVER_UPDATING = "server_updating"
# Long enough for a slow install's download, short enough that a deploy that died
# mid-way stops costing clients their work within minutes.
DEFAULT_HOLD_S = 600
MAX_HOLD_S = 1800
# What a refused client is told to wait before asking again: a restart takes tens of
# seconds, so a client polling faster only adds refusals.
RETRY_AFTER_S = 15


@dataclass(frozen=True)
class Holding:
    release: str | None
    by: str | None
    since: datetime
    until: datetime
    deadline: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "release": self.release,
            "by": self.by,
            "since": self.since.isoformat(),
            "until": self.until.isoformat(),
        }


class UpdateHold:
    """Whether this server is refusing new work because a deploy is about to restart it."""

    def __init__(self, *, monotonic: Any = time.monotonic) -> None:
        self._lock = threading.Lock()
        self._holding: Holding | None = None
        self._monotonic = monotonic

    def begin(self, *, seconds: int, release: str | None, by: str | None) -> Holding:
        if not 1 <= seconds <= MAX_HOLD_S:
            raise ApiError(
                400,
                "invalid_hold",
                f"a hold lasts 1 to {MAX_HOLD_S} seconds, and {seconds} was asked for",
                {"seconds": seconds, "max_seconds": MAX_HOLD_S},
            )
        now = clock.now()
        holding = Holding(
            release=release,
            by=by,
            since=now,
            until=now + timedelta(seconds=seconds),
            deadline=self._monotonic() + seconds,
        )
        with self._lock:
            self._holding = holding
        return holding

    def end(self) -> Holding | None:
        with self._lock:
            was, self._holding = self._current(), None
        return was

    def current(self) -> Holding | None:
        with self._lock:
            return self._current()

    def _current(self) -> Holding | None:
        holding = self._holding
        if holding is not None and self._monotonic() >= holding.deadline:
            # Lapsed: whoever took it never restarted the server and never let go.
            self._holding = None
            return None
        return holding

    def refuse_if_holding(self, what: str) -> None:
        """Refuse `what` (a job, a chat completion, ...) while a deploy holds the server."""
        holding = self.current()
        if holding is None:
            return
        for_release = "" if holding.release is None else f" to {holding.release}"
        raise ApiError(
            503,
            SERVER_UPDATING,
            f"this server is about to restart for an update{for_release}, so it is "
            f"not starting {what}. Nothing was admitted: ask again in "
            f"{RETRY_AFTER_S}s, and once the new server answers the request is taken "
            "as usual",
            {**holding.to_dict(), "retry_after": RETRY_AFTER_S},
            headers={"Retry-After": str(RETRY_AFTER_S)},
        )


__all__ = [
    "DEFAULT_HOLD_S",
    "Holding",
    "MAX_HOLD_S",
    "RETRY_AFTER_S",
    "SERVER_UPDATING",
    "UpdateHold",
]
