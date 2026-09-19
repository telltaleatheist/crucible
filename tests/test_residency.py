"""The dying slot: a process is Crucible's until its stop is CONFIRMED.

`Residency.unload` unpublishes the resident thing before it signals it, and that
ordering is deliberate — from that line on nothing new is proxied to a dying
engine and no align job is handed a session that is being stopped. What it must
not also do is FORGET the process. `engines/base.py` gives a stop
`STOP_TIMEOUT_SECONDS` and **never SIGKILLs** (a killed CUDA process wedges WSL2
until Windows reboots), so "asked to stop, still running" is a documented
outcome rather than an accident; and a residency that had already nulled its two
holder slots reported its own orphan to the accelerator guard as a FOREIGN
process, left `_evict` with nothing to evict, and let the next load start a
second engine onto an occupied card.

These drive `Residency` directly with engine doubles rather than through the job
doors, because what is under test is the holder's own bookkeeping across a stop
that raises — and a real engine that will not honour SIGTERM costs a 180 s wait
and a wedged card, which is the very thing the code under test exists to stop
Crucible causing.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from crucible.config import load_config, write_config
from crucible.engines import EngineError
from crucible.errors import JobError
from crucible.residency import Residency
from crucible.voices import load_voice

from .conftest import FAKE_BACKEND
from .test_tts_api import VOICE, resident_voice

#: A pid nothing on this machine has. The doubles never spawn anything — what is
#: being asserted is which number the residency still calls its own, and a real
#: process would make that assertion depend on the machine.
STUBBORN_PID = 424_242


class StubbornEngine:
    """An engine whose SIGTERM is not honoured: `SubprocessEngine.stop()`'s
    timeout path, without the 180 s wait or a process to wedge.

    `pids` goes on answering after the failed stop for the same reason the real
    one does — `SubprocessEngine.stop()` only clears `_process` after `wait()`
    returns, so a timed-out stop leaves the handle still naming a live child.
    """

    name = "stubborn"

    def __init__(self) -> None:
        self.stops = 0

    @property
    def pids(self) -> frozenset[int]:
        return frozenset({STUBBORN_PID})

    def stop(self) -> None:
        self.stops += 1
        raise EngineError(
            f"{self.name} (pid {STUBBORN_PID}) did not exit within 180s of "
            "SIGTERM. Crucible does not SIGKILL a process holding CUDA"
        )


class ObedientEngine:
    """An engine that goes when it is told, so the clean path stays pinned."""

    name = "obedient"

    def __init__(self) -> None:
        self.stops = 0
        self._alive = True

    @property
    def pids(self) -> frozenset[int]:
        return frozenset({STUBBORN_PID}) if self._alive else frozenset()

    def stop(self) -> None:
        self.stops += 1
        self._alive = False


@pytest.fixture
def holder(home: Path) -> Residency:
    """A `Residency` over a throwaway home, with nothing on the card yet."""
    write_config(
        home,
        name="crucible@test",
        host="127.0.0.1",
        port=7100,
        token="test-token-not-minted",
        backend_kind=FAKE_BACKEND.kind,
        enable_echo=False,
        enable_llm=True,
        enable_asr=False,
        enable_tts=True,
        enable_align=False,
        enable_rvc=False,
        desktop_allowance_bytes=3 * 1024 ** 3,
    )
    return Residency(load_config(home))


def a_resident_voice_on(holder: Residency, engine: object) -> None:
    """Put `engine` on the card serving the catalog's `deathstalker`.

    Straight into the slots, exactly as `crucible/residency.py`'s own tests in
    test_tts_api.py do it: `load_voice` would need narrator, and the fact under
    test begins at `unload`.
    """
    holder._resident = resident_voice(VOICE)
    holder._engine = engine  # type: ignore[assignment]


@contextmanager
def a_process_that_will_not_stop(holder: Residency) -> Iterator[StubbornEngine]:
    """Put a stubborn engine on `holder`'s card and fail to unload it.

    The dying slot is reached through `unload` rather than written into,
    because "asked to stop, still running" is a state `Residency` composes out
    of two facts, and a test that assembled it by hand would pin the assembly
    instead of the behaviour.

    **THE SLOT IS EMPTIED ON THE WAY OUT**, which matters only to the callers
    that hold a whole server (tests/test_activity.py, tests/test_tts_stream.py):
    `Residency.shutdown` asks the dying slot a second time and lets the refusal
    out — the right thing over a real CUDA process, and the reason a double
    that can NEVER go would turn those tests' teardown into an error about the
    double rather than a result about the server. Cleared rather than stopped,
    because being unstoppable is this double's whole character. The tests that
    drive a bare `Residency` do not need it and do not use it.
    """
    engine = StubbornEngine()
    a_resident_voice_on(holder, engine)
    with pytest.raises(EngineError):
        holder.unload(VOICE)
    try:
        yield engine
    finally:
        holder._dying = None


def test_a_stop_that_never_completes_leaves_the_pid_crucibles(
    holder: Residency,
) -> None:
    """The ledger C1 defect: unpublished is not the same as gone."""
    engine = StubbornEngine()
    a_resident_voice_on(holder, engine)
    assert holder.owned_pids() == frozenset({STUBBORN_PID})

    with pytest.raises(EngineError) as refusal:
        holder.unload(VOICE)
    assert "did not exit" in str(refusal.value)

    # Unpublished, because nothing may be proxied to it any more...
    assert holder.resident is None
    assert holder.resident_voice is None
    # ...but NOT disowned. The accelerator guard subtracts `owned_pids` before
    # it calls the card foreign-held, and this pid is Crucible's own child.
    assert holder.owned_pids() == frozenset({STUBBORN_PID})
    stopping = holder.stopping
    assert stopping is not None
    assert stopping.subject_id == VOICE
    assert stopping.pids == frozenset({STUBBORN_PID})


def test_a_load_behind_a_process_that_will_not_stop_is_refused_by_name(
    holder: Residency,
) -> None:
    """Never evict, never proceed: a second engine on a full card is the damage."""
    engine = StubbornEngine()
    a_resident_voice_on(holder, engine)
    with pytest.raises(EngineError):
        holder.unload(VOICE)
    assert engine.stops == 1

    manifest = load_voice(VOICE)
    with pytest.raises(JobError) as refusal:
        holder.load_voice(
            manifest,
            manifest.spec(FAKE_BACKEND.kind),
            Path("/nonexistent/weights"),
            Path("/nonexistent/python"),
        )
    assert refusal.value.code == "engine_still_stopping"
    assert VOICE in refusal.value.message
    assert str(STUBBORN_PID) in refusal.value.message
    # And it refused before it touched anything: no second SIGTERM at the dying
    # engine, which is what an `_evict` on the way past would have sent.
    assert engine.stops == 1


def test_a_stop_that_completes_leaves_nothing_behind(holder: Residency) -> None:
    """The existing behaviour, pinned: a confirmed stop owns no pids."""
    engine = ObedientEngine()
    a_resident_voice_on(holder, engine)

    unloaded = holder.unload(VOICE)
    assert unloaded.id == VOICE
    assert holder.resident is None
    assert holder.stopping is None
    assert holder.owned_pids() == frozenset()

    # And the card is loadable again: the very guard that refused in the test
    # above has nothing to refuse for, so it returns instead of raising.
    holder.refuse_if_stopping(f"load {VOICE}")


def test_shutdown_asks_the_dying_process_again(holder: Residency) -> None:
    """Nothing else ever will: the child is `start_new_session=True`, so the
    server exiting does not reach a process this module has forgotten."""
    engine = StubbornEngine()
    a_resident_voice_on(holder, engine)
    with pytest.raises(EngineError):
        holder.unload(VOICE)
    assert engine.stops == 1

    with pytest.raises(EngineError):
        holder.shutdown()
    assert engine.stops == 2
    # Still not confirmed, so still Crucible's. A shutdown that reported a clean
    # exit over a live CUDA process is the one lie that costs a reboot.
    assert holder.owned_pids() == frozenset({STUBBORN_PID})


def test_the_claim_is_refused_behind_a_process_that_will_not_stop(
    holder: Residency,
) -> None:
    """Ledger R14. The four `load*` doors have refused since the dying slot
    existed and this one did not, so a render or a settlement could take the
    card while a process Crucible had told to go was still on it — and then be
    refused `engine_still_stopping` by the load it took the card in order to
    make. It is also the ONLY place a claimant that loads nothing (a streaming
    session) can be told at all."""
    engine = StubbornEngine()
    a_resident_voice_on(holder, engine)
    with pytest.raises(EngineError):
        holder.unload(VOICE)

    with pytest.raises(JobError) as refusal:
        holder.claim("tts render", may_mutate=True)
    assert refusal.value.code == "engine_still_stopping"
    assert str(STUBBORN_PID) in refusal.value.message
    # Refused, not half-taken: the next claimant is told about the dying
    # process rather than about a holder that never got the card.
    assert holder.claimed_by is None
    assert engine.stops == 1


def test_a_card_with_nothing_dying_on_it_still_claims(holder: Residency) -> None:
    """The clean path, pinned beside the refusal: the guard returns."""
    holder.claim("tts render", may_mutate=True)
    assert holder.claimed_by == "tts render"
    holder.release("tts render")
