from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from crucible.settle import Held, Settlement


class _FakeStore:
    def __init__(self, job: Any = None) -> None:
        self.job = job

    def occupied_by_anything_but(self, _excluding: str | None) -> Any:
        return self.job


class _FakeLeases:
    def __init__(self, current: Any = None, lapsed: datetime | None = None) -> None:
        self._current = current
        self._lapsed = lapsed
        self.forgotten = 0

    def current(self) -> Any:
        return self._current

    def lapsed_at(self) -> datetime | None:
        return self._lapsed

    def forget_lapse(self) -> None:
        self.forgotten += 1
        self._lapsed = None


class _FakeResidency:
    def __init__(self, resident: Any = "a-model") -> None:
        self.resident = resident
        self.claimed_by = None
        self.unloaded: list[str] = []

    def claim(self, _who: str, **_kwargs: Any) -> None:
        return None

    def release(self, _who: str) -> None:
        return None

    def claim_to_clear(self, _who: str, *, held: Any) -> bool:
        return held() is None and self.resident is not None

    def unload(self, subject_id: str) -> None:
        self.unloaded.append(subject_id)
        self.resident = None


class _FakeInFlight:
    def __init__(self, count: int = 0) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count


class _Job:

    id = "j1"
    type = "tts"
    status = "running"
    client = "a-test"
    model = "mistborn"
    started = "2026-09-20T18:29:37+00:00"
    created = "2026-09-20T18:29:30+00:00"
    progress = 0.25
    message = "rendering 118 of 280"


def _settlement(
    *,
    resident: Any = "a-model",
    job: Any = None,
    lease: Any = None,
    lapsed: datetime | None = None,
    chats: int = 0,
) -> Settlement:
    return Settlement(
        residency=_FakeResidency(resident),
        store=_FakeStore(job),
        leases=_FakeLeases(lease, lapsed),
        inflight=_FakeInFlight(chats),
        log=lambda _line: None,
    )


class _LoadJob:
    id = "load-1"
    type = "load-model"
    status = "running"


def test_a_load_that_succeeded_dates_the_unheld_card() -> None:
    settlement = _settlement()
    before = datetime.now(timezone.utc)

    assert settlement.settle_for_job(_LoadJob(), "done") is None
    unclaimed = settlement.unheld_since()

    assert unclaimed is not None
    assert unclaimed >= before
    assert settlement.held_by() is None


def test_a_load_that_did_not_succeed_is_not_dated_here() -> None:
    settlement = _settlement(resident=None)
    settlement.settle_for_job(_LoadJob(), "cancelled")
    assert settlement.unheld_since() is None


def test_a_held_card_reports_no_unclaimed_time_however_old_the_stamp() -> None:
    settlement = _settlement()
    settlement.settle_for_job(_LoadJob(), "done")
    assert settlement.unheld_since() is not None

    settlement._store.job = _Job()
    assert settlement.held_by() is not None
    assert settlement.unheld_since() is None

    settlement._store.job = None
    assert settlement.unheld_since() is not None


def test_an_empty_card_has_nothing_to_date() -> None:
    settlement = _settlement(resident=None)
    settlement.settle_for_job(_LoadJob(), "done")
    assert settlement.unheld_since() is None


def test_a_lapsed_lease_dates_the_card_from_its_own_expiry() -> None:
    lapsed = datetime.now(timezone.utc) - timedelta(minutes=4)
    settlement = _settlement(lapsed=lapsed)
    assert settlement.unheld_since() == lapsed


def test_the_later_of_the_two_moments_wins() -> None:
    settlement = _settlement()
    settlement.settle_for_job(_LoadJob(), "done")
    stamped = settlement.unheld_since()
    assert stamped is not None

    later = stamped + timedelta(minutes=2)
    settlement._leases._lapsed = later
    assert settlement.unheld_since() == later

    earlier = stamped - timedelta(minutes=2)
    settlement._leases._lapsed = earlier
    assert settlement.unheld_since() == stamped


@pytest.mark.parametrize(
    "kwargs,fact",
    [
        ({"job": _Job()}, "a job"),
        ({"chats": 3}, "a chat"),
    ],
)
def test_held_by_reports_the_holding_fact(kwargs: dict[str, Any], fact: str) -> None:
    settlement = _settlement(**kwargs)
    held = settlement.held_by()
    assert held is not None
    assert held.fact == fact
    assert settlement.unheld_since() is None


def test_held_by_is_the_same_answer_the_refusals_print() -> None:
    held = Held("a chat", "3 completion(s) in flight", {"in_flight": 3})
    assert held.to_dict() == {
        "fact": "a chat",
        "who": "3 completion(s) in flight",
        "details": {"in_flight": 3},
    }
    assert str(held) == "a chat holds it: 3 completion(s) in flight"


def test_the_readers_do_not_take_the_settlement_lock() -> None:
    settlement = _settlement()
    with settlement._lock:
        assert settlement.held_by() is None
        assert settlement.unheld_since() is None


class _Resident:

    def __init__(self, ident: str = "a-model") -> None:
        self.id = ident
        self.kind = "llm"


def test_a_lapsed_lease_clears_the_card() -> None:
    settlement = _settlement(
        resident=_Resident(), lapsed=datetime.now(timezone.utc) - timedelta(minutes=1)
    )
    settled = settlement.settle_for_lapsed_lease()

    assert settled is not None
    assert settled.subject_id == "a-model"
    assert settlement._residency.unloaded == ["a-model"]


def test_a_lease_that_has_not_lapsed_clears_nothing() -> None:
    settlement = _settlement(resident=_Resident(), lapsed=None)
    assert settlement.settle_for_lapsed_lease() is None
    assert settlement._residency.unloaded == []


def test_a_lapse_is_spent_once_even_when_it_cleared_nothing() -> None:
    settlement = _settlement(
        resident=None, lapsed=datetime.now(timezone.utc) - timedelta(minutes=1)
    )
    assert settlement.settle_for_lapsed_lease() is None
    assert settlement._leases.forgotten == 1

    settlement._residency.resident = _Resident("a-later-model")
    assert settlement.settle_for_lapsed_lease() is None
    assert settlement._residency.unloaded == []


def test_a_settlement_that_raises_still_spends_the_lapse() -> None:

    class _Exploding(_FakeResidency):
        def unload(self, subject_id: str) -> None:
            raise RuntimeError("the engine would not stop")

    settlement = Settlement(
        residency=_Exploding(_Resident()),
        store=_FakeStore(None),
        leases=_FakeLeases(None, datetime.now(timezone.utc) - timedelta(minutes=1)),
        inflight=_FakeInFlight(0),
        log=lambda _line: None,
    )
    with pytest.raises(RuntimeError):
        settlement.settle_for_lapsed_lease()
    assert settlement._leases.forgotten == 1
