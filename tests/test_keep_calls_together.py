"""A model that keeps its queued calls together (crucible/keeptogether.py).

Victoria's laptop, 2026-10-10: B-Sides queued YuE2 songs, and a CLI ``load-model`` sent
mid-batch would have taken YuE2 off the card between two of them. Owen: "keep yue's calls
together". These tests drive the waiting line, the pump and the settlement with a fake
resident, so nothing loads and no card is touched.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from crucible import queuepump
from crucible.audiomodels import AudioManifestError, load_audio_manifest, parse_audio_manifest
from crucible.cardkinds import KIND_AUDIO, KIND_LLM
from crucible.events import EventHub
from crucible.inflight import InFlight
from crucible.jobs.line import Call, WaitingLine
from crucible.keeptogether import KEEPING_CALLS_TOGETHER, runs_on, takes_it_off
from crucible.queuepump import QueuePump
from crucible.queuesessions import QueueSessions
from crucible.settle import Settlement
from crucible.updating import UpdateHold

YUE = "yue2-3b"


@dataclass
class _Resident:
    id: str
    kind: str
    keeps_calls_together: bool


@dataclass
class _Job:
    id: str
    type: str
    model: str | None
    client: str | None = "b-sides"
    client_ref: str | None = None
    session: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def busy_details(self) -> dict[str, Any]:
        return {"job_id": self.id, "type": self.type, "model": self.model}


class _Store:
    def __init__(self) -> None:
        self.line: Any = None
        self.events = EventHub()
        self.updating = UpdateHold()
        self.lane_free = False

    def attach_line(self, line: Any) -> None:
        self.line = line

    def mark_waiting(self, job: _Job, waiting: dict[str, Any] | None) -> None:
        pass

    def append_event(self, job: _Job, kind: str, data: dict[str, Any]) -> None:
        job.events.append({"event": kind, "data": data})

    def end_waiting(self, item: Any, **_: Any) -> None:
        pass

    def followed(self) -> tuple[set[str], set[str | None]]:
        # Every job is followed: nothing here is abandoned.
        return {item.job.id for item in self.line.ordered()}, set()


class _Card:
    claimed_by = None

    def __init__(self, resident: _Resident | None) -> None:
        self.resident = resident


def _line(resident: _Resident | None) -> tuple[WaitingLine, _Store, _Card]:
    store, card = _Store(), _Card(resident)
    line = WaitingLine(store, QueueSessions(), lambda: card.resident)  # type: ignore[arg-type]
    return line, store, card


def _yue(keeps: bool = True) -> _Resident:
    return _Resident(YUE, KIND_AUDIO, keeps)


def _song(name: str) -> _Job:
    return _Job(name, "audio", YUE)


def _load(name: str = "load") -> _Job:
    return _Job(name, "load-model", "qwen3.5-9b", client="cli")


def _join(line: WaitingLine, job: _Job) -> Any:
    return line.join(job, SimpleNamespace(), 3600, None)  # type: ignore[arg-type]


def _ids(line: WaitingLine) -> list[str]:
    return [item.job.id for item in line.ordered()]


def _waits(job: _Job) -> list[dict[str, Any]]:
    return [e["data"] for e in job.events if e["event"] == "waiting"]


def test_yue2_declares_it_and_a_manifest_that_says_nothing_does_not() -> None:
    assert load_audio_manifest(YUE).keep_calls_together is True
    assert load_audio_manifest("stable-audio-3-small-sfx").keep_calls_together is False


def test_keep_calls_together_must_be_a_boolean() -> None:
    path = Path(__file__).resolve().parent.parent / "crucible" / "audio" / f"{YUE}.toml"
    text = path.read_text(encoding="utf-8").replace(
        "keep_calls_together = true", 'keep_calls_together = "yes"'
    )
    with pytest.raises(AudioManifestError, match="keep_calls_together must be bool"):
        parse_audio_manifest(text, path, YUE)


@pytest.mark.parametrize(
    ("job", "runs", "takes_off"),
    [
        (_Job("s", "audio", YUE), True, False),
        (_Job("l", "load-audio", YUE), True, False),
        (_Job("o", "audio", "stable-audio-3-medium"), False, True),
        (_Job("m", "load-model", "qwen3.5-9b"), False, True),
        (_Job("u", "unload-audio", YUE), False, True),
        (_Job("x", "unload-model", "qwen3.5-9b"), False, False),
        (_Job("a", "asr", "qwen3-asr-1.7b"), False, True),
        (_Job("e", "echo", None), False, False),
        (_Job("?", "a-type-nobody-knows", None), False, True),
    ],
)
def test_what_runs_on_yue2_and_what_would_take_it_off(
    job: _Job, runs: bool, takes_off: bool
) -> None:
    item = SimpleNamespace(job=job, is_call=False, is_session=False)
    assert (runs_on(item, _yue()), takes_it_off(item, _yue())) == (runs, takes_off)


def test_calls_and_sessions_take_it_off_only_when_they_name_another_model() -> None:
    chat = SimpleNamespace(job=Call("chat", "qwen3.5-9b", "x"), is_call=True, is_session=False)
    assert takes_it_off(chat, _yue()) and not runs_on(chat, _yue())
    named = SimpleNamespace(job=SimpleNamespace(model="qwen3.5-9b"), is_call=False,
                            is_session=True)
    unnamed = SimpleNamespace(job=SimpleNamespace(model=None), is_call=False,
                              is_session=True)
    assert takes_it_off(named, _yue())
    assert not takes_it_off(unnamed, _yue())
    llm = _Resident("qwen3.5-9b", KIND_LLM, True)
    assert runs_on(chat, llm) and not takes_it_off(chat, llm)


def test_a_song_sent_while_one_runs_goes_ahead_of_the_load_and_says_why() -> None:
    asyncio.run(_song_sent_while_one_runs())


async def _song_sent_while_one_runs() -> None:
    line, _, _ = _line(_yue())
    load = _load()
    _join(line, load)
    song = _song("song2")
    _join(line, song)
    assert _ids(line) == ["song2", "load"]
    rows = line.rows()
    assert [row["position"] for row in rows] == [1, 2]
    waiting_for = rows[1]["waiting_for"]
    assert waiting_for["code"] == KEEPING_CALLS_TOGETHER
    assert waiting_for["details"]["ahead"] == 1
    assert waiting_for["message"].startswith(f"waiting: {YUE} has 1 queued call(s) ahead")
    said = _waits(load)
    assert said and said[-1]["code"] == KEEPING_CALLS_TOGETHER
    assert line.kept_on_card() == (YUE, 1)


def test_only_the_songs_waiting_when_its_turn_comes_go_ahead_of_it() -> None:
    asyncio.run(_the_bound())


async def _the_bound() -> None:
    line, _, _ = _line(_yue())
    load = _load()
    _join(line, load)
    early = _song("early")
    _join(line, early)
    assert line.take_kept_turn() is True
    assert line.take_kept_turn() is False, "the turn is taken once"
    line.reorder()
    late = _song("late")
    _join(line, late)
    assert _ids(line) == ["early", "load", "late"]
    assert "sent from now on wait behind this load-model" in _waits(load)[-1]["message"]
    line.started(line.get("early"))
    assert _ids(line) == ["load", "late"]
    assert line.rows()[0]["waiting_for"] is None
    assert line.kept_on_card() is None, "the next thing to run takes it off the card"


def test_a_model_that_does_not_keep_its_calls_together_stays_first_come() -> None:
    asyncio.run(_first_come(_yue(keeps=False)))
    asyncio.run(_first_come(None))


async def _first_come(resident: _Resident | None) -> None:
    line, _, _ = _line(resident)
    _join(line, _load())
    _join(line, _song("song2"))
    assert _ids(line) == ["load", "song2"]
    assert line.take_kept_turn() is False
    assert line.kept_on_card() is None


def test_when_the_model_leaves_the_card_the_line_is_first_come_again() -> None:
    asyncio.run(_model_left())


async def _model_left() -> None:
    line, _, card = _line(_yue())
    _join(line, _load())
    _join(line, _song("song2"))
    assert _ids(line) == ["song2", "load"]
    card.resident = None
    line.reorder()
    assert _ids(line) == ["load", "song2"]
    assert line.rows()[0]["waiting_for"] is None


def test_a_song_removed_from_ahead_of_it_is_no_longer_waited_for() -> None:
    asyncio.run(_removed_ahead())


async def _removed_ahead() -> None:
    line, _, _ = _line(_yue())
    load = _load()
    _join(line, load)
    _join(line, _song("a"))
    _join(line, _song("b"))
    assert line.rows()[2]["waiting_for"]["details"]["ahead"] == 2
    line.remove("a", "client", "cancelled")
    assert _ids(line) == ["b", "load"]
    assert _waits(load)[-1]["details"]["ahead"] == 1
    line.remove("b", "operator", "removed")
    assert _ids(line) == ["load"]
    assert line.rows()[0]["waiting_for"] is None


def test_things_that_do_not_touch_the_card_keep_their_place() -> None:
    asyncio.run(_echo_keeps_its_place())


async def _echo_keeps_its_place() -> None:
    line, _, _ = _line(_yue())
    _join(line, _Job("echo", "echo", None))
    _join(line, _load())
    _join(line, _song("song2"))
    assert _ids(line) == ["echo", "song2", "load"]
    assert line.take_kept_turn() is False, "the load is not first to have arrived"
    assert line.kept_on_card() == (YUE, 1), "echo does not take YuE2 off the card"


def test_the_open_sessions_items_stay_first_and_in_their_order() -> None:
    asyncio.run(_open_session())


async def _open_session() -> None:
    line, _, _ = _line(_yue())
    session = line.sessions.create(act="tts", client="bookforge", model=None,
                                   idle_s=300, max_wait_s=3600)
    line.sessions.opened(session)
    _join(line, _Job("mine-load", "load-model", "qwen3.5-9b", client="bookforge",
                     session=session.id))
    _join(line, _Job("mine-song", "audio", YUE, client="bookforge", session=session.id))
    _join(line, _load())
    _join(line, _song("song2"))
    assert _ids(line) == ["mine-load", "mine-song", "song2", "load"]
    assert line.take_kept_turn() is False


def test_the_settlement_keeps_it_for_the_songs_that_run_next() -> None:
    kept: list[tuple[str, int] | None] = [(YUE, 2)]
    settlement = Settlement(
        residency=_Card(_yue()),  # type: ignore[arg-type]
        store=SimpleNamespace(occupied_by_anything_but=lambda _job: None),
        sessions=QueueSessions(),
        inflight=InFlight(),
        kept_together=lambda: kept[0],
        log=lambda _line: None,
    )
    held = settlement.holder()
    assert held is not None
    assert held.details == {"waiting": 2, "model": YUE, "kept_together": True}
    kept[0] = None
    assert settlement.holder() is None


def test_the_pump_runs_the_batch_before_the_load_and_the_load_before_later_songs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(_pump(monkeypatch))


async def _pump(monkeypatch: pytest.MonkeyPatch) -> None:
    line, store, _ = _line(_yue())
    ran: list[str] = []

    async def admit(waiting: Any, _ctx: Any) -> None:
        ran.append(waiting.job.id)
        store.lane_free = False
        line.started(waiting)
        return None

    monkeypatch.setattr(queuepump, "admit_waiting", admit)
    ctx = SimpleNamespace(store=store)
    pump = QueuePump(line, lambda: ctx)  # type: ignore[arg-type,return-value]

    # song1 is on the lane; the CLI's load arrives, then two more songs.
    _join(line, _load())
    _join(line, _song("song2"))
    _join(line, _song("song3"))
    await pump.step()
    assert ran == []

    store.lane_free = True  # song1 finished: the load's turn comes
    await pump.step()
    assert ran == ["song2"]
    _join(line, _song("song4"))  # sent after the turn: waits behind the load
    assert _ids(line) == ["song3", "load", "song4"]

    for expected in ("song3", "load", "song4"):
        store.lane_free = True
        await pump.step()
        assert ran[-1] == expected
    assert ran == ["song2", "song3", "load", "song4"]
