from __future__ import annotations

import itertools
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from .capabilityclasses import CLASSES
from .clock import utcnow
from .errors import ApiError
from .protocol import ACT_HEADER

RECENT_DURATIONS = 20

ACT_NAMES: frozenset[str] = frozenset(entry.name for entry in CLASSES)


def require_act_name(act: str, source: str, advice: str = "") -> str:
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
    id: int
    act: str | None
    model: str
    client: str | None
    since: str
    started: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "act": self.act,
            "model": self.model,
            "client": self.client,
            "since": self.since,
        }


class InFlight:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[int, Entry] = {}
        self._ids = itertools.count(1)
        self._recent: list[float] = []
        self._on_close: Any = lambda: None

    def when_closed(self, callback: Any) -> None:
        self._on_close = callback

    def open(self, *, act: str | None, model: str, client: str | None) -> Entry:
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
        with self._lock:
            removed = self._entries.pop(entry.id, None)
            if removed is not None and removed.started > 0.0:
                self._recent.append(time.monotonic() - removed.started)
                del self._recent[:-RECENT_DURATIONS]
        if removed is not None:
            self._on_close()

    @contextmanager
    def tracked(
        self, *, act: str | None, model: str, client: str | None
    ) -> Iterator[Entry]:
        entry = self.open(act=act, model=model, client=client)
        try:
            yield entry
        finally:
            self.close(entry)

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            entries = sorted(self._entries.values(), key=lambda e: e.id)
        return [entry.to_dict() for entry in entries]

    def retry_after(self) -> int | None:
        with self._lock:
            recent = sorted(self._recent)
        if not recent:
            return None
        middle = recent[len(recent) // 2]
        return max(1, math.ceil(middle))

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
