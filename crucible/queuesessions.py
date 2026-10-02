"""Queue sessions: one client's claim on the lane for a run of requests it cannot know in
advance (a Briefcase video analysis: ASR, then many chats and decisions). Not to be
confused with a TTS *stream* session (crucible/ttsstream), which is something else.

A session is asked for with ``POST /v1/queue/sessions`` and waits in the same line as
everything else (crucible/jobs/line.py, a line item of kind ``session``). At the front the
queue pump opens it, loading its ``model`` first when it names one that is not resident
(crucible/sessionqueue.py). While it is open:

- requests carrying ``X-Crucible-Session: <id>`` are its items, and so is every request
  from the client that holds it (an implicit item, header or not): they are admitted
  ahead of everything waiting and are never refused on account of other clients;
- nothing from anyone else runs: other clients' queued work waits, their unqueued work is
  refused by name;
- the settlement treats it as a holder, so what its items leave on the card stays there.

It closes when its client deletes it (``client``), when ``idle_s`` passes with nothing in
flight and no item or touch (``idle``), when an operator removes it (``operator``), when
the configured maximum hold passes (``max_hold``; none by default), or when the server
stops (``server_restart``). One session is open at a time.

Every change of a session's state goes through ``QueueSessions._say``: it records the event on the
session's own stream and wakes its followers.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

from . import clock
from .errors import ApiError
from .protocol import CLIENT_HEADER

SESSION_PREFIX = "ses-"
DEFAULT_IDLE_S = 300
STREAM_IDLE_S = 900
MIN_IDLE_S = 10
MAX_IDLE_S = 86_400
REMEMBERED = 200

QUEUED = "queued"
OPEN = "open"
CLOSED = "closed"

CLIENT = "client"
IDLE = "idle"
MAX_HOLD = "max_hold"
OPERATOR = "operator"
SERVER_RESTART = "server_restart"
EXPIRED = "expired"
LOAD_FAILED = "load_failed"
REFUSED = "refused"

ENDING_EVENTS = frozenset({"closed", "removed"})

UNNAMED = "an unnamed client"


def is_session_id(value: str) -> bool:
    return value.startswith(SESSION_PREFIX)


def who(client: str | None) -> str:
    return UNNAMED if client is None else repr(client)


@dataclass
class QueueSession:
    """One session, queued, open or closed. ``type``, ``model``, ``client`` and
    ``client_ref`` are what the waiting line reads off every item it holds."""

    act: str
    client: str | None
    model: str | None
    idle_s: int
    max_wait_s: int
    created: datetime
    id: str = field(default_factory=lambda: SESSION_PREFIX + uuid.uuid4().hex)
    type: str = "session"
    client_ref: str | None = None
    status: str = QUEUED
    position: int | None = None
    opened_at: datetime | None = None
    closed_at: datetime | None = None
    reason: str | None = None
    message: str | None = None
    error: dict[str, Any] | None = None
    seen: datetime | None = None
    items_run: int = 0
    load_job: str | None = None
    jobs: list[str] = field(default_factory=list)
    stream_sessions: set[str] = field(default_factory=set)
    opened_for_stream: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    waiters: list[asyncio.Event] = field(default_factory=list)

    @property
    def followed(self) -> bool:
        return bool(self.waiters)

    def busy_details(self) -> dict[str, Any]:
        since = self.opened_at if self.opened_at is not None else self.created
        return {"door": "session", "holder": self.client, "session_id": self.id,
                "type": self.type, "act": self.act, "model": self.model,
                "status": self.status, "since": since.isoformat()}

    def describe(self) -> str:
        since = "" if self.opened_at is None else f", open since {self.opened_at.isoformat()}"
        return f"session {self.id} of {who(self.client)} for {self.act!r}{since}"


class QueueSessions:
    """Every session this server knows: the one open now, those waiting in the line, and
    the most recent closed ones (so a client can still read how its session ended)."""

    def __init__(
        self,
        max_hold_s: Callable[[], int] = lambda: 0,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._max_hold_s = max_hold_s
        self._now = clock.now if now is None else now
        self._sessions: OrderedDict[str, QueueSession] = OrderedDict()
        self._open: QueueSession | None = None
        self._in_flight: Callable[[QueueSession], list[dict[str, Any]]] = lambda session: []
        self._stream: Callable[[QueueSession], dict[str, Any] | None] = lambda session: None
        self._publish: Callable[[str, dict[str, Any]], None] = lambda event, data: None

    def when_said(self, publish: Callable[[str, dict[str, Any]], None]) -> None:
        """Where every state change is also published (GET /v1/events)."""
        self._publish = publish

    def watch(
        self,
        in_flight: Callable[[QueueSession], list[dict[str, Any]]],
        stream: Callable[[QueueSession], dict[str, Any] | None],
    ) -> None:
        """How to read what a session has in flight (its jobs on or waiting for the
        lane, its chats and decisions answering or waiting, rows its TTS stream session
        is saying), and the TTS stream session open inside it, if any."""
        self._in_flight = in_flight
        self._stream = stream


    def current(self) -> QueueSession | None:
        return self._open

    def get(self, session_id: str) -> QueueSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise ApiError(
                404,
                "unknown_queue_session",
                f"this server has no queue session {session_id}. Queue sessions are held "
                "in memory: a restart closes every one and forgets it, and only the most recent "
                f"{REMEMBERED} are remembered after they close. Open one with "
                "POST /v1/queue/sessions",
                {"session_id": session_id},
            )
        return session

    def queued(self) -> list[QueueSession]:
        return [session for session in self._sessions.values() if session.status == QUEUED]

    def in_flight(self, session: QueueSession) -> list[dict[str, Any]]:
        return self._in_flight(session)


    def create(
        self, *, act: str, client: str | None, model: str | None, idle_s: int,
        max_wait_s: int,
    ) -> QueueSession:
        session = QueueSession(
            act=act, client=client, model=model, idle_s=idle_s,
            max_wait_s=max_wait_s, created=self._now(),
        )
        self._sessions[session.id] = session
        self._forget_old()
        return session

    def _forget_old(self) -> None:
        closed = [g.id for g in self._sessions.values() if g.status == CLOSED]
        for session_id in closed[: max(0, len(closed) - REMEMBERED)]:
            del self._sessions[session_id]


    def item(self, session_id: str, client: str | None) -> QueueSession:
        """The open session a request's ``X-Crucible-Session`` names, refused by name when
        it is not open or belongs to another client."""
        session = self.get(session_id)
        if session.status == QUEUED:
            where = "" if session.position is None else f" at position {session.position}"
            raise ApiError(
                409,
                "session_not_open",
                f"session {session_id} is still waiting in this server's line{where}; it "
                "has no items until it opens. Follow GET /v1/queue/sessions/"
                f"{session_id}/events and send its items after `opened`",
                {"session_id": session_id, "status": session.status,
                 "position": session.position},
            )
        if session.status == CLOSED:
            raise ApiError(
                409,
                "session_closed",
                f"session {session_id} closed ({session.reason}): {session.message}. "
                "Nothing more runs in it; open a new session with POST /v1/queue/sessions",
                {"session_id": session_id, "status": session.status, "reason": session.reason},
            )
        if session.client != client:
            raise ApiError(
                409,
                "session_not_yours",
                f"session {session_id} belongs to {who(session.client)}, and this request "
                f"comes from {who(client)} ({CLIENT_HEADER}, else User-Agent). Only "
                "the client that opened a session sends its items",
                {"session_id": session_id, "client": session.client, "caller": client},
            )
        return session

    def member(self, session_id: str | None, client: str | None) -> QueueSession | None:
        """The session a request is an item of. ``X-Crucible-Session`` is the explicit
        form and is checked by ``item``. Without it, a request from the client that
        holds the open session is an implicit item of it: an app making standalone calls
        beside its own long run never waits behind itself."""
        if session_id is not None:
            return self.item(session_id, client)
        held = self._open
        if held is not None and client is not None and held.client == client:
            return held
        return None

    def item_arrived(self, session: QueueSession) -> None:
        session.items_run += 1
        session.seen = self._now()

    def adopt_job(self, session_id: str, job_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is not None:
            session.jobs.append(job_id)

    def adopt_stream_session(
        self, session: QueueSession, stream_session_id: str, *, opened_for_it: bool
    ) -> None:
        """A TTS stream session runs inside this queue session. ``opened_for_it``: the
        queue session was opened by the stream's own open, so it ends with the stream."""
        session.stream_sessions.add(stream_session_id)
        if opened_for_it:
            session.opened_for_stream = stream_session_id

    def of_stream_session(self, stream_session_id: str) -> QueueSession | None:
        held = self._open
        if held is not None and stream_session_id in held.stream_sessions:
            return held
        return None

    def touch(self, session: QueueSession) -> None:
        """"Still here": an in-memory timestamp, nothing written anywhere."""
        if session.status == OPEN:
            session.seen = self._now()


    def refuse_if_held(self, session_id: str | None, what: str) -> None:
        """``server_busy`` naming the open session, for work that is not one of its
        items; a queued request takes it as "wait in the line"."""
        held = self._open
        if held is None or session_id == held.id:
            return
        raise ApiError(
            409,
            "server_busy",
            f"{held.describe()} holds this server, and nothing else runs until it "
            f"closes, so {what} is not admitted now. Submit with \"queue\": {{}} to "
            "wait in the line, or read GET /v1/activity to see the session",
            held.busy_details(),
        )

    def refuse_call_if_held(
        self, session_id: str | None, what: str, *, queueable: bool = True
    ) -> None:
        """``session_open`` for an unqueued chat or decision (or a TTS stream session)
        that is not one of the open session's items."""
        held = self._open
        if held is None or session_id == held.id:
            return
        wait = (
            "Send it with \"queue\": {} to wait in the line until the session closes"
            if queueable else "Try again once it closes (GET /v1/activity shows it)"
        )
        raise ApiError(
            409,
            "session_open",
            f"{held.describe()} holds this server, and nothing else runs until it "
            f"closes, so {what} is not answered now. {wait}, or, if this is the "
            f"session's own work, send it with the header X-Crucible-Session: {held.id}",
            held.busy_details(),
        )


    def positioned(self, session: QueueSession, position: int, of: int, *, first: bool) -> None:
        session.position = position
        self._say(session, "queued" if first else "moved", {"position": position, "of": of})

    def opened(self, session: QueueSession) -> None:
        if self._open is not None and self._open is not session:
            raise RuntimeError(
                f"session {session.id} cannot open while {self._open.id} is open; the "
                "queue pump opens one session at a time"
            )
        now = self._now()
        session.status, session.opened_at, session.seen, session.position = OPEN, now, now, None
        self._open = session
        self._say(session, "opened", {
            "opened_at": now.isoformat(), "model": session.model, "load_job": session.load_job,
        })

    def removed(
        self, session: QueueSession, reason: str, message: str,
        error: dict[str, Any] | None = None,
    ) -> None:
        """A session that never opened leaves the line."""
        if session.status != QUEUED:
            return
        session.status, session.closed_at = CLOSED, self._now()
        session.reason, session.message, session.error = reason, message, error
        session.position = None
        data: dict[str, Any] = {"reason": reason, "message": message}
        if error is not None:
            data["error"] = error
        self._say(session, "removed", data)

    def close(self, session: QueueSession, reason: str, message: str) -> bool:
        """An open session ends. False when it was not open."""
        if session.status != OPEN:
            return False
        now = self._now()
        session.status, session.closed_at = CLOSED, now
        session.reason, session.message = reason, message
        if self._open is session:
            self._open = None
        self._say(session, "closed", {
            "reason": reason, "message": message, "items_run": session.items_run,
            "held_s": round((now - session.opened_at).total_seconds(), 3)
            if session.opened_at is not None else None,
        })
        return True


    def max_hold_s(self) -> int:
        return self._max_hold_s()

    def idle_deadline(self, session: QueueSession) -> datetime | None:
        if session.status != OPEN or session.seen is None or self.in_flight(session):
            return None
        return session.seen + timedelta(seconds=session.idle_s)

    def max_hold_deadline(self, session: QueueSession) -> datetime | None:
        limit = self._max_hold_s()
        if session.opened_at is None or limit <= 0:
            return None
        return session.opened_at + timedelta(seconds=limit)

    def due(self, now: datetime | None = None) -> tuple[QueueSession, str, str] | None:
        """The open session and why it should close now, if it should. Anything in flight
        counts as presence: a day-long job inside a session never idles it out."""
        session = self._open
        if session is None:
            return None
        now = self._now() if now is None else now
        held_until = self.max_hold_deadline(session)
        if held_until is not None and now >= held_until:
            return session, MAX_HOLD, (
                f"it was open for the server's maximum hold, {self._max_hold_s()} s "
                "([queue] max_session_hold_s in config.toml)"
            )
        if session.opened_for_stream is not None and self._stream(session) is None:
            return session, CLIENT, (
                f"the TTS stream session it was opened for ({session.opened_for_stream}) "
                "has closed"
            )
        if self.in_flight(session):
            session.seen = now
            return None
        assert session.seen is not None
        if (now - session.seen).total_seconds() >= session.idle_s:
            return session, IDLE, (
                f"nothing arrived, ran or was in flight for its idle_s ({session.idle_s} s), "
                "and nothing touched it"
            )
        return None


    def state(self, session: QueueSession) -> dict[str, Any]:
        idle = self.idle_deadline(session)
        hold = self.max_hold_deadline(session)
        return {
            "session_id": session.id,
            "status": session.status,
            "act": session.act,
            "client": session.client,
            "model": session.model,
            "position": session.position,
            "idle_s": session.idle_s,
            "max_wait_s": session.max_wait_s,
            "created": session.created.isoformat(),
            "opened_at": None if session.opened_at is None else session.opened_at.isoformat(),
            "idle_deadline": None if idle is None else idle.isoformat(),
            "max_hold_deadline": None if hold is None else hold.isoformat(),
            "items_run": session.items_run,
            "in_flight": self.in_flight(session) if session.status == OPEN else [],
            "stream_session": self._stream(session) if session.status == OPEN else None,
            "load_job": session.load_job,
            "closed_at": None if session.closed_at is None else session.closed_at.isoformat(),
            "reason": session.reason,
            "message": session.message,
            "error": session.error,
        }


    def _say(self, session: QueueSession, event: str, data: dict[str, Any]) -> None:
        """The one place a session's state change is announced."""
        session.events.append({
            "id": len(session.events) + 1,
            "event": event,
            "data": {"session_id": session.id, **data},
        })
        for waiter in session.waiters:
            waiter.set()
        self._publish(event, {
            "session_id": session.id, "client": session.client, "act": session.act, **data,
        })

    def subscribe(self, session: QueueSession) -> asyncio.Event:
        waiter = asyncio.Event()
        session.waiters.append(waiter)
        return waiter

    def unsubscribe(self, session: QueueSession, waiter: asyncio.Event) -> None:
        if waiter in session.waiters:
            session.waiters.remove(waiter)


__all__ = [
    "CLIENT", "CLOSED", "DEFAULT_IDLE_S", "ENDING_EVENTS", "EXPIRED", "SESSION_PREFIX",
    "QueueSession", "QueueSessions", "IDLE", "LOAD_FAILED", "MAX_HOLD", "MAX_IDLE_S", "MIN_IDLE_S",
    "OPEN", "OPERATOR", "QUEUED", "REFUSED", "SERVER_RESTART", "is_session_id", "who",
]
