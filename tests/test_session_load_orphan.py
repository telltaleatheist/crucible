"""A waiting session that leaves the line after its own load finished takes the model off
the card (crucible/sessionqueue.py settle_a_left_sessions_load).

2026-10-25 on owens-pc (bookforge pc 1's report): BookForge quit while its pronunciation
guide's queue session waited on qwen3.8-27b-4bit. The load finished DONE at 16:49:53, the
session was gone before it opened, and the 27B sat on the card held by nothing until it was
unloaded by hand. The line cancels a load still running; one already done was left."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from crucible.jobs.base import DONE
from crucible.jobs.line import WaitingLine
from crucible.queuesessions import QueueSessions
from crucible.sessionqueue import settle_a_left_sessions_load
from crucible.updating import UpdateHold

from .test_keep_calls_together import _Store

MODEL = "qwen3.8-27b-4bit"


@dataclass
class _Settlement:
    settled: list[str]

    def settle_quietly(self, trigger: str) -> None:
        self.settled.append(trigger)


def _item(model: str | None = MODEL, load_job: str | None = "load-1") -> Any:
    session = SimpleNamespace(id="ses-1", model=model, load_job=load_job)
    return SimpleNamespace(job=session)


def _store(status: str) -> Any:
    return SimpleNamespace(get=lambda job_id: SimpleNamespace(id=job_id, status=status))


def _residency(model_id: str | None) -> Any:
    resident = None if model_id is None else SimpleNamespace(model_id=model_id)
    return SimpleNamespace(resident_model=resident)


def _settle(item: Any, *, status: str = DONE, resident: str | None = MODEL) -> list[str]:
    settlement = _Settlement([])
    worker = settle_a_left_sessions_load(
        item, store=_store(status), residency=_residency(resident), settlement=settlement
    )
    if worker is not None:
        worker.join(timeout=5)
    return settlement.settled


def test_a_session_gone_after_its_load_finished_settles_the_card() -> None:
    settled = _settle(_item())
    assert len(settled) == 1
    assert "ses-1" in settled[0] and MODEL in settled[0]


def test_a_load_that_did_not_finish_is_the_lines_to_cancel_not_this() -> None:
    assert _settle(_item(), status="cancelled") == []
    assert _settle(_item(), status="running") == []


def test_another_model_on_the_card_is_never_touched() -> None:
    """An operator's own load, or a model something else put there since, stays."""
    assert _settle(_item(), resident="qwen3.5-9b") == []
    assert _settle(_item(), resident=None) == []


def test_a_session_that_started_no_load_settles_nothing() -> None:
    assert _settle(_item(load_job=None)) == []
    assert _settle(_item(model=None)) == []


def test_the_line_hands_every_removed_waiting_session_to_the_hook() -> None:
    store = _Store()
    sessions = QueueSessions()
    sessions.updating = UpdateHold()
    line = WaitingLine(store, sessions, lambda: None)  # type: ignore[arg-type]
    left: list[Any] = []
    line.when_session_leaves(left.append)
    session = sessions.create(act="clean", client="bookforge", model=MODEL, idle_s=60,
                              max_wait_s=60)
    item = line.join_session(session)
    line.remove(session.id, "client", "the client went away")
    assert left == [item]


def test_the_settle_runs_off_the_callers_thread() -> None:
    """The line removes synchronously, on the event loop; a settle unloads an engine."""
    seen: list[str] = []
    started = threading.Event()

    class _Slow:
        def settle_quietly(self, trigger: str) -> None:
            seen.append(threading.current_thread().name)
            started.set()

    worker = settle_a_left_sessions_load(
        _item(), store=_store(DONE), residency=_residency(MODEL), settlement=_Slow()
    )
    assert worker is not None
    assert started.wait(timeout=5)
    assert seen == ["settle-ses-1"] and seen[0] != threading.current_thread().name
