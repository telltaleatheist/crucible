from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from .. import ttsstream
from ..errors import ApiError

GRACE_SECONDS = 15.0


@dataclass
class Frame:
    id: int
    event: str
    data: dict[str, Any]
    at: float


@dataclass
class Reader:
    waiter: asyncio.Event
    delivered: int


class EventLog:
    def __init__(self, session_id: str, loop: asyncio.AbstractEventLoop) -> None:
        self._session_id = session_id
        self._loop = loop
        self._events: deque[Frame] = deque()
        self._next_event_id = 0
        self._floor = 0
        self._attached: list[Reader] = []
        self._ever_attached = False
        self._detached_at: float | None = time.monotonic()

    def emit(self, event: str, data: dict[str, Any]) -> None:
        self._loop.call_soon_threadsafe(self._append, event, data)

    def _append(self, event: str, data: dict[str, Any]) -> None:
        self._next_event_id += 1
        self._events.append(
            Frame(id=self._next_event_id, event=event, data=data, at=time.monotonic())
        )
        if self._floor == 0:
            self._floor = self._next_event_id
        for reader in list(self._attached):
            reader.waiter.set()
        self._prune()

    def _prune(self) -> None:
        if not self._events:
            return
        horizon = time.monotonic() - ttsstream.GRACE_SECONDS
        cursors = [reader.delivered for reader in self._attached]
        floor_cursor = min(cursors) if cursors else self._next_event_id
        while self._events:
            oldest = self._events[0]
            if oldest.at >= horizon or oldest.id > floor_cursor:
                break
            self._events.popleft()
            self._floor = oldest.id + 1

    def check_replayable(self, delivered: int) -> None:
        if delivered < self._floor - 1:
            raise ApiError(
                409,
                "replay_unavailable",
                f"session {self._session_id} can no longer replay from event "
                f"{delivered}: the oldest frame it still holds is {self._floor}. "
                f"A stream is replayable for {ttsstream.GRACE_SECONDS:.0f}s after "
                "the last reader leaves",
                {"session_id": self._session_id, "oldest": self._floor},
            )

    def attach(self, delivered: int) -> Reader:
        reader = Reader(waiter=asyncio.Event(), delivered=delivered)
        self._attached.append(reader)
        self._ever_attached = True
        self._detached_at = None
        return reader

    def detach(self, reader: Reader) -> None:
        if reader in self._attached:
            self._attached.remove(reader)
        if not self._attached:
            self._detached_at = time.monotonic()

    @property
    def ever_attached(self) -> bool:
        return self._ever_attached

    def grace_expired(self, now: float) -> bool:
        detached = self._detached_at
        return detached is not None and now - detached > ttsstream.GRACE_SECONDS

    def frames_after(self, delivered: int) -> list[Frame]:
        return [event for event in self._events if event.id > delivered]
