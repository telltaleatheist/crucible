"""A queue session states the least context its model must be started at
(`context_tokens`), and the load that opens it is held to it (load-model `min_context`).

bookforge pc 2, 2026-11-11: Simplify sends one paragraph at a time and needs a few
thousand tokens, but a session could not say so. The server picked the model's default or
the most the card afforded, and on owens-pc that load was sized wrong and failed. With the
need stated, a tight card loads at what it affords at or above it, or refuses by name."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from crucible import sessionqueue
from crucible.api.schemas import SessionOpen
from crucible.errors import ApiError
from crucible.queuesessions import QueueSessions
from crucible.sessionqueue import WAIT, _load_for, _resident
from crucible.updating import UpdateHold

MODEL = "qwen3.8-27b-4bit"


def _sessions() -> QueueSessions:
    sessions = QueueSessions()
    sessions.updating = UpdateHold()
    return sessions


def _ctx(resident_len: int | None) -> Any:
    @asynccontextmanager
    async def settled_for(what: str):  # type: ignore[no-untyped-def]
        yield

    resident = (
        None
        if resident_len is None
        else SimpleNamespace(model_id=MODEL, max_model_len=resident_len)
    )
    return SimpleNamespace(
        residency=SimpleNamespace(resident_model=resident, settled_for=settled_for)
    )


def test_the_wire_takes_a_positive_whole_number_of_tokens() -> None:
    assert SessionOpen(act="clean", model=MODEL, context_tokens=6000).context_tokens == 6000
    for wrong in (0, -1, 6000.5, "6000"):
        with pytest.raises(ValidationError):
            SessionOpen(act="clean", model=MODEL, context_tokens=wrong)


def test_a_need_with_no_model_to_start_at_it_is_refused() -> None:
    with pytest.raises(ApiError) as refused:
        _sessions().create(
            act="clean", client="bookforge", model=None, idle_s=60, max_wait_s=60,
            context_tokens=6000,
        )
    assert refused.value.code == "context_without_model"


def test_a_resident_started_at_enough_opens_the_session_at_its_context() -> None:
    session = _sessions().create(
        act="clean", client="bookforge", model=MODEL, idle_s=60, max_wait_s=60,
        context_tokens=6000,
    )
    assert asyncio.run(_resident(_ctx(11_317), session)) is True
    assert session.context == 11_317


def test_a_resident_started_short_of_the_need_is_loaded_again() -> None:
    session = _sessions().create(
        act="clean", client="bookforge", model=MODEL, idle_s=60, max_wait_s=60,
        context_tokens=16_000,
    )
    assert asyncio.run(_resident(_ctx(11_317), session)) is False
    assert session.context is None


def _load_request(context_tokens: int | None, monkeypatch: pytest.MonkeyPatch) -> Any:
    session = _sessions().create(
        act="clean", client="bookforge", model=MODEL, idle_s=60, max_wait_s=60,
        context_tokens=context_tokens,
    )
    asked: list[Any] = []

    async def admit(request: Any, ctx: Any) -> Any:
        asked.append(request)
        return SimpleNamespace(job=SimpleNamespace(id="load-1"))

    monkeypatch.setattr(sessionqueue, "admit", admit)
    waiting = SimpleNamespace(
        job=session, loads=0, card_wait=None, load_job=None, card_due=lambda now: True
    )
    assert asyncio.run(_load_for(waiting, SimpleNamespace(), [])) == WAIT  # type: ignore[arg-type]
    assert session.load_job == "load-1"
    (request,) = asked
    return request


def test_the_load_that_opens_it_carries_the_need_as_its_minimum(monkeypatch) -> None:
    request = _load_request(6000, monkeypatch)
    assert request.type == "load-model"
    assert request.params == {"min_context": 6000}


def test_a_session_with_no_stated_need_loads_as_before(monkeypatch) -> None:
    assert _load_request(None, monkeypatch).params == {}
