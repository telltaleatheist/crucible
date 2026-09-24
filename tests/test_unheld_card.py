"""`/v1/activity` says when the card is resident and held by NOTHING.

On 2026-09-20 a hosted-Foundry runner submitted `load-model qwen3.5-9b` on the PC
and Owen pressed Stop one second after it reached `done`. The cancel was refused
`job_not_cancellable` — the job was already terminal — no lease was ever opened,
no chat was ever sent, and a 21 GB model sat on the card for five minutes with
all four of `settle.py`'s facts false. `/v1/activity` reported `resident` set,
`lease: null`, `claim: null`, `chat.in_flight: 0`, `running: []`: every field
correct, and not one of them said "nobody is coming back for this".

That state is produced two ways and both are by construction, not by accident:

1. **A load that succeeded.** `LEAVES_IT_RESIDENT` — a load's whole content is
   "be resident", so its own completion cannot clear the card.
2. **A lease that LAPSED.** Expiry is read and never swept, so nothing runs at
   the moment the ttl passes.

These tests are about SAYING it. Nothing here unloads anything and there is no
timer: what should be done about an unheld card is a ruling
(`docs/BUG-HUNT-2026-09-20.md` §F.8). They also pin the two properties that make
the field trustworthy — it is never reported while something actually holds the
card, and it is read from `settle.py` rather than derived a second time.
"""

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
        # The real one's contract without its lock: the settlement's own reading
        # of the four facts decides, and an empty card is nothing to clear.
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
    """Only what `Settlement.holder` reads off a job.

    It reaches `jobs/queue.py`'s `busy_details`, so this carries that door's
    whole shape — `held_by.details` IS the `server_busy` body, deliberately, and
    a double that stubbed a shorter one would prove a contract nobody ships.
    """

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


# ------------------------------------------------------- the stamp is taken


def test_a_load_that_succeeded_dates_the_unheld_card() -> None:
    """THE INCIDENT. The load is exempt from settling, so nothing else records
    the moment the card became nobody's."""
    settlement = _settlement()
    before = datetime.now(timezone.utc)

    assert settlement.settle_for_job(_LoadJob(), "done") is None
    unclaimed = settlement.unheld_since()

    assert unclaimed is not None
    assert unclaimed >= before
    # And the card is genuinely unheld, which is the other half of the report.
    assert settlement.held_by() is None


def test_a_load_that_did_not_succeed_is_not_dated_here() -> None:
    """A cancelled or failed load settles like everything else, so the card is
    cleared rather than left unheld — there is nothing to date."""
    settlement = _settlement(resident=None)
    settlement.settle_for_job(_LoadJob(), "cancelled")
    assert settlement.unheld_since() is None


# ------------------------------------------ the stamp is never reported wrongly


def test_a_held_card_reports_no_unclaimed_time_however_old_the_stamp() -> None:
    """THE PROPERTY THAT MAKES THE FIELD SAFE. The stamp is history and the
    holder is now; a reconciler acting on a stale stamp would unload a model
    somebody had just leased."""
    settlement = _settlement()
    settlement.settle_for_job(_LoadJob(), "done")
    assert settlement.unheld_since() is not None

    # Something takes the card afterwards. Nothing clears the stamp — it simply
    # stops being readable, because the live holder is checked first.
    settlement._store.job = _Job()  # noqa: SLF001
    assert settlement.held_by() is not None
    assert settlement.unheld_since() is None

    # And when that holder lets go the stamp is readable again.
    settlement._store.job = None  # noqa: SLF001
    assert settlement.unheld_since() is not None


def test_an_empty_card_has_nothing_to_date() -> None:
    settlement = _settlement(resident=None)
    settlement.settle_for_job(_LoadJob(), "done")
    assert settlement.unheld_since() is None


# ------------------------------------------------------- the lapsed lease


def test_a_lapsed_lease_dates_the_card_from_its_own_expiry() -> None:
    """The second producer. Expiry is READ, never swept, so no code path runs at
    the moment the ttl passes — but `expires_at` was written when the lease
    opened, so the moment is known exactly and needs no timer to observe."""
    lapsed = datetime.now(timezone.utc) - timedelta(minutes=4)
    settlement = _settlement(lapsed=lapsed)
    assert settlement.unheld_since() == lapsed


def test_the_later_of_the_two_moments_wins() -> None:
    """A load, then a lease that was taken and lapsed: the card went quiet at
    the lapse, not at the load."""
    settlement = _settlement()
    settlement.settle_for_job(_LoadJob(), "done")
    stamped = settlement.unheld_since()
    assert stamped is not None

    later = stamped + timedelta(minutes=2)
    settlement._leases._lapsed = later  # noqa: SLF001
    assert settlement.unheld_since() == later

    earlier = stamped - timedelta(minutes=2)
    settlement._leases._lapsed = earlier  # noqa: SLF001
    assert settlement.unheld_since() == stamped


# ------------------------------------------------- held_by is not a second owner


@pytest.mark.parametrize(
    "kwargs,fact",
    [
        ({"job": _Job()}, "a job"),
        ({"chats": 3}, "a chat"),
    ],
)
def test_held_by_reports_the_holding_fact(kwargs: dict[str, Any], fact: str) -> None:
    """`/v1/activity` reads this rather than re-deriving four facts of its own."""
    settlement = _settlement(**kwargs)
    held = settlement.held_by()
    assert held is not None
    assert held.fact == fact
    assert settlement.unheld_since() is None


def test_held_by_is_the_same_answer_the_refusals_print() -> None:
    """One owner, one vocabulary: the wire shape is the refusal's shape."""
    held = Held("a chat", "3 completion(s) in flight", {"in_flight": 3})
    assert held.to_dict() == {
        "fact": "a chat",
        "who": "3 completion(s) in flight",
        "details": {"in_flight": 3},
    }
    assert str(held) == "a chat holds it: 3 completion(s) in flight"


def test_the_readers_do_not_take_the_settlement_lock() -> None:
    """A BENCH READ MUST NEVER QUEUE BEHIND AN UNLOAD.

    `settle()` holds `_lock` for the whole of a stop, and `SubprocessEngine.stop()`
    waits up to 180 s for SIGTERM. A reader that acquired the same lock would make
    `/v1/activity` hang for three minutes, which `settle.py` itself calls
    indistinguishable from a dead server. This holds the lock and proves both
    readers answer anyway.
    """
    settlement = _settlement()
    with settlement._lock:  # noqa: SLF001
        assert settlement.held_by() is None
        assert settlement.unheld_since() is None


# ---------------------------------------------------------------------------
# The lease that ran out — the one holder whose end fires nothing
# ---------------------------------------------------------------------------


class _Resident:
    """What `settle()` reads off the resident thing."""

    def __init__(self, ident: str = "a-model") -> None:
        self.id = ident
        self.kind = "llm"


def test_a_lapsed_lease_clears_the_card() -> None:
    """THE HOLDER WHOSE END NOBODY OBSERVES.

    A released lease settles at the release. A lease that simply runs out is
    read and never swept, so until this existed nothing ran at the moment it
    lapsed — which is survivable while a lease is only a refusal, and is not
    once a lease is what HOLDS a load.
    """
    settlement = _settlement(
        resident=_Resident(), lapsed=datetime.now(timezone.utc) - timedelta(minutes=1)
    )
    settled = settlement.settle_for_lapsed_lease()

    assert settled is not None
    assert settled.subject_id == "a-model"
    assert settlement._residency.unloaded == ["a-model"]  # noqa: SLF001


def test_a_lease_that_has_not_lapsed_clears_nothing() -> None:
    settlement = _settlement(resident=_Resident(), lapsed=None)
    assert settlement.settle_for_lapsed_lease() is None
    assert settlement._residency.unloaded == []  # noqa: SLF001


def test_a_lapse_is_spent_once_even_when_it_cleared_nothing() -> None:
    """THE LOADED GUN. `_lease` is kept after expiry so the 404 can still say
    when it expired — so a lapse that went on being offered would, on the first
    idle tick after somebody loaded a model WITHOUT a lease, be read as a holder
    letting go and unload a model that had nothing to do with it.
    """
    settlement = _settlement(
        resident=None, lapsed=datetime.now(timezone.utc) - timedelta(minutes=1)
    )
    # Nothing resident, so nothing to clear — and the lapse is still spent.
    assert settlement.settle_for_lapsed_lease() is None
    assert settlement._leases.forgotten == 1  # noqa: SLF001

    # A model loaded afterwards, by somebody who took no lease, is SAFE.
    settlement._residency.resident = _Resident("a-later-model")  # noqa: SLF001
    assert settlement.settle_for_lapsed_lease() is None
    assert settlement._residency.unloaded == []  # noqa: SLF001


def test_a_settlement_that_raises_still_spends_the_lapse() -> None:
    """A lapse retried for ever against a card it cannot clear is the same gun
    with a slower trigger, so the spend is in a `finally`."""

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
    assert settlement._leases.forgotten == 1  # noqa: SLF001
