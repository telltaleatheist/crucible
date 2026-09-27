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

STUBBORN_PID = 424_242


class StubbornEngine:

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
        enable_denoise=False,
        retention_days=7,
        desktop_allowance_basis="stated",
        desktop_allowance_note="",
    )
    return Residency(load_config(home))


def a_resident_voice_on(holder: Residency, engine: object) -> None:
    holder._resident = resident_voice(VOICE)
    holder._engine = engine


@contextmanager
def a_process_that_will_not_stop(holder: Residency) -> Iterator[StubbornEngine]:
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
    engine = StubbornEngine()
    a_resident_voice_on(holder, engine)
    assert holder.owned_pids() == frozenset({STUBBORN_PID})

    with pytest.raises(EngineError) as refusal:
        holder.unload(VOICE)
    assert "did not exit" in str(refusal.value)

    assert holder.resident is None
    assert holder.resident_voice is None
    assert holder.owned_pids() == frozenset({STUBBORN_PID})
    stopping = holder.stopping
    assert stopping is not None
    assert stopping.subject_id == VOICE
    assert stopping.pids == frozenset({STUBBORN_PID})


def test_a_load_behind_a_process_that_will_not_stop_is_refused_by_name(
    holder: Residency,
) -> None:
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
    assert engine.stops == 1


def test_a_stop_that_completes_leaves_nothing_behind(holder: Residency) -> None:
    engine = ObedientEngine()
    a_resident_voice_on(holder, engine)

    unloaded = holder.unload(VOICE)
    assert unloaded.id == VOICE
    assert holder.resident is None
    assert holder.stopping is None
    assert holder.owned_pids() == frozenset()

    holder.refuse_if_stopping(f"load {VOICE}")


def test_shutdown_asks_the_dying_process_again(holder: Residency) -> None:
    engine = StubbornEngine()
    a_resident_voice_on(holder, engine)
    with pytest.raises(EngineError):
        holder.unload(VOICE)
    assert engine.stops == 1

    with pytest.raises(EngineError):
        holder.shutdown()
    assert engine.stops == 2
    assert holder.owned_pids() == frozenset({STUBBORN_PID})


def test_the_claim_is_refused_behind_a_process_that_will_not_stop(
    holder: Residency,
) -> None:
    engine = StubbornEngine()
    a_resident_voice_on(holder, engine)
    with pytest.raises(EngineError):
        holder.unload(VOICE)

    with pytest.raises(JobError) as refusal:
        holder.claim("tts render", may_mutate=True)
    assert refusal.value.code == "engine_still_stopping"
    assert str(STUBBORN_PID) in refusal.value.message
    assert holder.claimed_by is None
    assert engine.stops == 1


def test_a_card_with_nothing_dying_on_it_still_claims(holder: Residency) -> None:
    holder.claim("tts render", may_mutate=True)
    assert holder.claimed_by == "tts render"
    holder.release("tts render")
