"""The server-wide event stream's hub: one place every change is published to, and every
open GET /v1/events stream reads from.

The owners of the server's state publish here, each at its own single point of truth:
the job store's event append, the waiting line's announcements, residency, the in-flight
registry, the task store and the settings history. Publishing is one call:

    hub.publish(QUEUE, "queue.added", {...})

and never blocks. It takes a lock, gives the event the next id, keeps it in a bounded
history and hands it to each subscriber's bounded buffer; a subscriber whose buffer is
full is marked overflowed and is ended by its stream, not waited for. It may be called
from any thread: a residency change happens on a job's worker thread, and the
subscriber's waiter is woken on its own loop.

Ids are integers that only grow. They start at the hub's creation time in microseconds,
so an id from before a restart is never mistaken for one after it: it is older than
anything the history holds and the client is sent a fresh snapshot.

The event names, payloads and topics are documented in docs/EVENTS.md. A new owner adds
one `publish` call; a new topic is added to TOPICS and to that document.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from .clock import utcnow
from .errors import ApiError

JOB = "job"
QUEUE = "queue"
CARD = "card"
CHAT = "chat"
TASK = "task"
SETTINGS = "settings"
SERVER = "server"
SESSION = "session"
TOPICS: tuple[str, ...] = (JOB, QUEUE, SESSION, CARD, CHAT, TASK, SETTINGS, SERVER)
ALWAYS = frozenset({SERVER})

SNAPSHOT = "snapshot"
OVERFLOW = "overflow"
STOPPING = "server.stopping"
ENDS = frozenset({OVERFLOW, STOPPING})

HISTORY = 1000
SUBSCRIBER_LIMIT = 2000
THROTTLE_S = 1.0


def parse_topics(raw: str | None) -> frozenset[str]:
    """`?topics=job,queue` as a set; absent or empty means every topic."""
    if raw is None or raw.strip() == "":
        return frozenset(TOPICS)
    named = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(named) - set(TOPICS))
    if unknown:
        raise ApiError(
            400,
            "unknown_topic",
            f"{unknown} is not a topic of GET /v1/events. Name some of {list(TOPICS)}, "
            "comma-separated, or leave `topics` out for all of them",
            {"unknown": unknown, "known": list(TOPICS)},
        )
    return frozenset(named) | ALWAYS


def _on(loop: asyncio.AbstractEventLoop) -> bool:
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


@dataclass(eq=False)
class Subscriber:
    """One open stream: the topics it asked for and the events waiting to be sent."""

    topics: frozenset[str]
    limit: int
    loop: asyncio.AbstractEventLoop
    waiter: asyncio.Event = field(default_factory=asyncio.Event)
    buffer: deque[dict[str, Any]] = field(default_factory=deque)
    overflowed: bool = False

    def offer(self, topic: str, event: dict[str, Any]) -> None:
        if topic not in self.topics or self.overflowed:
            return
        if len(self.buffer) >= self.limit:
            self.overflowed = True
            self.buffer.clear()
        else:
            self.buffer.append(event)
        self.wake()

    def wake(self) -> None:
        _wake(self.loop, self.waiter)


def _wake(loop: asyncio.AbstractEventLoop, waiter: asyncio.Event) -> None:
    if _on(loop):
        waiter.set()
    elif not loop.is_closed():
        loop.call_soon_threadsafe(waiter.set)


@dataclass(frozen=True)
class Opening:
    """What `subscribe` found: the subscriber, the id its snapshot stands at, and
    whether it needs one (`snapshot`) because it asked for none to resume from or
    because the history no longer reaches back to the id it gave (`gap`)."""

    subscriber: Subscriber
    id: int
    snapshot: bool
    gap: bool


@dataclass
class _Throttle:
    topic: str
    name: str
    sent: Any = None
    at: float = float("-inf")
    pending: Any = None
    timer: bool = False


class EventHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_id = time.time_ns() // 1000
        self._history: deque[tuple[str, dict[str, Any]]] = deque(maxlen=HISTORY)
        self._subscribers: list[Subscriber] = []
        self._throttles: dict[str, _Throttle] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopped = False
        self._reason: str | None = None
        self._watchers: list[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """The loop the server runs on; a throttled event's trailing send is timed on it."""
        self._loop = loop

    @property
    def last_id(self) -> int:
        with self._lock:
            return self._last_id

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def subscribers(self) -> int:
        """How many streams are open on this hub right now."""
        with self._lock:
            return len(self._subscribers)

    def publish(self, topic: str, name: str, data: dict[str, Any]) -> None:
        """Say that something changed. `name` is `<topic>.<what>`; `data` is its payload,
        to which the hub adds `at`. Never blocks and never raises for a slow reader."""
        if topic not in TOPICS:
            raise ValueError(f"{topic!r} is not one of the event topics {TOPICS}")
        with self._lock:
            if not self._stopped:
                self._append(topic, name, data)

    def _append(self, topic: str, name: str, data: dict[str, Any]) -> None:
        self._last_id += 1
        event = {"id": self._last_id, "event": name, "data": {**data, "at": utcnow()}}
        self._history.append((topic, event))
        for subscriber in self._subscribers:
            subscriber.offer(topic, event)

    def publish_throttled(self, topic: str, name: str, key: str, data: dict[str, Any]) -> None:
        """Publish `data` for `key` at most once every THROTTLE_S, and only when it differs
        from what was last sent for that key. A change inside the window is held and the
        latest one is sent when the window closes, so the last word is never lost."""
        loop = self._loop
        if loop is None:
            self.publish(topic, name, data)
            return
        with self._lock:
            state = self._throttles.get(key)
            if state is None:
                state = self._throttles[key] = _Throttle(topic, name)
            if data == state.sent:
                state.pending = None
                return
            now = time.monotonic()
            if not state.timer and now - state.at >= THROTTLE_S:
                state.sent, state.at = data, now
                send = True
            else:
                state.pending = data
                send = False
                if not state.timer:
                    state.timer = True
                    self._later(loop, state.at + THROTTLE_S - now, key, state)
        if send:
            self.publish(topic, name, data)

    def forget(self, key: str) -> None:
        """Drop `key`'s throttle, so a change held back for it is never sent: its subject
        has ended and the event that said so is the last word."""
        with self._lock:
            self._throttles.pop(key, None)

    def _later(
        self, loop: asyncio.AbstractEventLoop, delay: float, key: str, state: _Throttle
    ) -> None:
        if _on(loop):
            loop.call_later(max(0.0, delay), self._flush, key, state)
        elif not loop.is_closed():
            loop.call_soon_threadsafe(loop.call_later, max(0.0, delay), self._flush, key, state)

    def _flush(self, key: str, state: _Throttle) -> None:
        with self._lock:
            if self._throttles.get(key) is not state:
                return
            state.timer = False
            data, state.pending = state.pending, None
            if data is None or data == state.sent:
                return
            state.sent, state.at = data, time.monotonic()
        self.publish(state.topic, state.name, data)

    def subscribe(self, topics: Iterable[str], after: int | None) -> Opening:
        """Open a subscriber. With `after`, every kept event since that id is queued for
        it at once; when the history does not reach back that far it is told to send a
        snapshot instead, with `gap` set."""
        loop = asyncio.get_running_loop()
        wanted = frozenset(topics) | ALWAYS
        with self._lock:
            if self._stopped:
                raise ApiError(
                    503,
                    "server_stopping",
                    "this server is stopping, so it opens no new event streams. "
                    "Reconnect once it is back",
                )
            subscriber = Subscriber(topics=wanted, limit=SUBSCRIBER_LIMIT, loop=loop)
            snapshot, gap = True, False
            if after is not None:
                oldest = self._history[0][1]["id"] if self._history else self._last_id + 1
                if oldest - 1 <= after <= self._last_id:
                    snapshot = False
                    for topic, event in self._history:
                        if event["id"] > after:
                            subscriber.offer(topic, event)
                else:
                    gap = True
            self._subscribers.append(subscriber)
            return Opening(subscriber, self._last_id, snapshot, gap)

    def unsubscribe(self, subscriber: Subscriber) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def pending(self, subscriber: Subscriber, cursor: int) -> list[dict[str, Any]]:
        with self._lock:
            return [event for event in subscriber.buffer if event["id"] > cursor]

    def delivered(self, subscriber: Subscriber, cursor: int) -> None:
        with self._lock:
            buffer = subscriber.buffer
            while buffer and buffer[0]["id"] <= cursor:
                buffer.popleft()

    def stop(self, reason: str) -> None:
        """Say `server.stopping` to every stream and end them, so none holds the server's
        shutdown open. Nothing is published after it. Safe from any thread, and once."""
        with self._lock:
            if self._stopped:
                return
            self._append(SERVER, STOPPING, {"reason": reason})
            self._stopped = True
            self._reason = reason
            subscribers, self._subscribers = self._subscribers, []
            watchers, self._watchers = self._watchers, []
        for subscriber in subscribers:
            subscriber.wake()
        for loop, waiter in watchers:
            _wake(loop, waiter)

    @property
    def stop_reason(self) -> str | None:
        return self._reason

    def watch_stop(self, waiter: asyncio.Event) -> bool:
        """Have `stop` set `waiter`, for a stream that is not a subscriber (a job's, a
        task's, the queue's, a narration session's), so it too ends when the server
        stops. False when the hub has stopped already: end now."""
        with self._lock:
            if self._stopped:
                return False
            self._watchers.append((asyncio.get_running_loop(), waiter))
            return True

    def unwatch_stop(self, waiter: asyncio.Event) -> None:
        with self._lock:
            self._watchers = [pair for pair in self._watchers if pair[1] is not waiter]


__all__ = [
    "ALWAYS",
    "CARD",
    "CHAT",
    "ENDS",
    "EventHub",
    "JOB",
    "OVERFLOW",
    "Opening",
    "QUEUE",
    "SERVER",
    "SETTINGS",
    "SNAPSHOT",
    "STOPPING",
    "Subscriber",
    "TASK",
    "TOPICS",
    "parse_topics",
]
