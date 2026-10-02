from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from crucible import accelerator, memorybudget
from crucible import residency as residency_module
from crucible.cardkinds import KIND_TTS
from crucible.config import load_config, write_config
from crucible.engines import EngineError
from crucible.errors import ApiError, JobError
from crucible.residency import Residency, ResidentVoice

from .conftest import FAKE_BACKEND
from .test_tts_api import VOICE

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
    holder._resident = ResidentVoice(
        voice_id=VOICE,
        backend=FAKE_BACKEND.kind,
        narrator_engine="higgs-v3",
        revision="0" * 40,
        fingerprint="f" * 16,
        sample_rate=24_000,
        max_chars=None,
        memory_bytes_estimate=1,
        log_path=Path("/tmp/engine-test.log"),
        loaded_at="2026-09-13T02:00:00+00:00",
    )
    holder._engine = engine


def _alive_while_stubborn(original: object) -> object:
    return lambda pid: pid == STUBBORN_PID or original(pid)


@contextmanager
def a_process_that_will_not_stop(holder: Residency) -> Iterator[StubbornEngine]:
    engine = StubbornEngine()
    a_resident_voice_on(holder, engine)
    original = residency_module.process_alive
    residency_module.process_alive = _alive_while_stubborn(original)
    with pytest.raises(EngineError):
        holder.unload(VOICE)
    try:
        yield engine
    finally:
        residency_module.process_alive = original
        holder._dying = None


@pytest.fixture
def stubborn_is_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        residency_module,
        "process_alive",
        _alive_while_stubborn(residency_module.process_alive),
    )


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
    holder: Residency, stubborn_is_running: None
) -> None:
    engine = StubbornEngine()
    a_resident_voice_on(holder, engine)
    with pytest.raises(EngineError):
        holder.unload(VOICE)
    assert engine.stops == 1

    def start() -> residency_module.Occupant:
        raise AssertionError("a refused load must not start anything")

    with pytest.raises(JobError) as refusal:
        holder.occupy(KIND_TTS, VOICE, start, say=lambda _message: None)
    assert refusal.value.code == "engine_still_stopping"
    assert VOICE in refusal.value.message
    assert f"`kill {STUBBORN_PID}` (never -9)" in refusal.value.message
    assert "engine-test.log" in refusal.value.message
    assert "by hand" not in refusal.value.message
    assert engine.stops == 2


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
    holder: Residency, stubborn_is_running: None
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
    assert engine.stops == 2


def test_a_card_with_nothing_dying_on_it_still_claims(holder: Residency) -> None:
    holder.claim("tts render", may_mutate=True)
    assert holder.claimed_by == "tts render"
    holder.release("tts render")


def test_a_dying_process_that_has_since_exited_no_longer_wedges_the_card(
    holder: Residency,
) -> None:
    engine = StubbornEngine()
    a_resident_voice_on(holder, engine)
    with pytest.raises(EngineError):
        holder.unload(VOICE)
    assert holder.stopping is not None

    holder.refuse_if_stopping(f"load {VOICE}")
    assert holder.stopping is None
    assert engine.stops == 1
    holder.claim("tts render", may_mutate=True)
    holder.release("tts render")


class SlowToStopEngine(StubbornEngine):

    def __init__(self) -> None:
        super().__init__()
        self.alive = True

    def stop(self) -> None:
        if self.stops == 0:
            super().stop()
        self.stops += 1
        self.alive = False


def test_a_second_stop_that_works_frees_the_card(
    holder: Residency, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = SlowToStopEngine()
    monkeypatch.setattr(
        residency_module,
        "process_alive",
        lambda pid: pid == STUBBORN_PID and engine.alive,
    )
    a_resident_voice_on(holder, engine)
    with pytest.raises(EngineError):
        holder.unload(VOICE)

    holder.refuse_if_stopping(f"load {VOICE}")
    assert engine.stops == 2
    assert holder.stopping is None


class ThisProcessEngine:

    name = "this-process"

    def __init__(self, stop_budget_seconds: float | None = None) -> None:
        self.running = True
        if stop_budget_seconds is not None:
            self.stop_budget_seconds = stop_budget_seconds

    @property
    def pids(self) -> frozenset[int]:
        return frozenset({os.getpid()}) if self.running else frozenset()

    def stop(self) -> None:
        self.running = False


def test_the_resident_is_recorded_on_disk_and_forgotten_on_unload(
    holder: Residency,
) -> None:
    engine = ThisProcessEngine()
    a_resident_voice_on(holder, engine)
    holder._record_residents()

    record = json.loads(holder.record_path.read_text(encoding="utf-8"))
    assert holder.record_path.parent.name == "run"
    assert record["subject"] == VOICE
    assert [entry["pid"] for entry in record["processes"]] == [os.getpid()]
    assert record["processes"][0]["command"]
    assert record["processes"][0]["started"]

    holder.unload(VOICE)
    assert not holder.record_path.exists()


def test_the_clearance_wait_is_the_engines_own_stop_budget(holder: Residency) -> None:
    assert holder.clearance_timeout() == residency_module.CLEARANCE_TIMEOUT_SECONDS
    a_resident_voice_on(holder, ThisProcessEngine())
    assert holder.clearance_timeout() == residency_module.CLEARANCE_TIMEOUT_SECONDS
    a_resident_voice_on(holder, ThisProcessEngine(stop_budget_seconds=750.0))
    assert holder.clearance_timeout() == 780.0


@pytest.fixture
def orphan() -> Iterator["subprocess.Popen[bytes]"]:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        yield child
    finally:
        child.terminate()
        child.wait(timeout=30)
        accelerator.forget_leftover(child.pid)


def _a_dead_pid() -> int:
    pid = 999_983
    while accelerator.process_alive(pid):
        pid -= 2
    return pid


def _left_by_a_crash(
    holder: Residency, child: "subprocess.Popen[bytes]", budget: float
) -> None:
    identity = accelerator.process_identity(child.pid)
    assert identity is not None
    holder.record_path.parent.mkdir(parents=True, exist_ok=True)
    holder.record_path.write_text(
        json.dumps(
            {
                "crucible_pid": _a_dead_pid(),
                "recorded_at": "2026-09-27T00:00:00+00:00",
                "subject": VOICE,
                "stop_budget_seconds": budget,
                "processes": [identity.to_dict()],
            }
        ),
        encoding="utf-8",
    )


def test_an_engine_a_crashed_crucible_left_is_asked_to_stop_on_startup(
    holder: Residency,
    orphan: "subprocess.Popen[bytes]",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[int] = []

    def ask(pid: int) -> bool:
        asked.append(pid)
        orphan.terminate()
        orphan.wait(timeout=30)
        return True

    monkeypatch.setattr(residency_module, "ask_pid_to_stop", ask)
    _left_by_a_crash(holder, orphan, budget=30.0)

    assert holder.reclaim_leftovers() == [orphan.pid]
    assert asked == [orphan.pid]
    assert not holder.record_path.exists()
    assert accelerator.leftovers() == []


def test_a_survivor_is_named_by_the_guard_with_the_command_to_stop_it(
    holder: Residency,
    orphan: "subprocess.Popen[bytes]",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(residency_module, "ask_pid_to_stop", lambda pid: True)
    _left_by_a_crash(holder, orphan, budget=0.2)

    assert holder.reclaim_leftovers() == []
    kept = json.loads(holder.record_path.read_text("utf-8"))
    assert [entry["pid"] for entry in kept["processes"]] == [orphan.pid]

    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(
        accelerator, "probe_vram", lambda: (4 * memorybudget.GIB, 24 * memorybudget.GIB)
    )
    with pytest.raises(ApiError) as refusal:
        accelerator.guard("cuda-linux", model_id="m", need_bytes=1)
    message = refusal.value.message
    assert f"A previous Crucible left {orphan.pid} " in message
    assert "it was asked to stop at 20" in message
    assert f"`kill {orphan.pid}` (never -9)" in message
    assert refusal.value.details["left_by_previous_run"][0]["pid"] == orphan.pid


def test_a_record_whose_pid_is_now_someone_else_is_left_alone(
    holder: Residency,
    orphan: "subprocess.Popen[bytes]",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[int] = []
    monkeypatch.setattr(residency_module, "ask_pid_to_stop", asked.append)
    _left_by_a_crash(holder, orphan, budget=30.0)
    record = json.loads(holder.record_path.read_text("utf-8"))
    record["processes"][0]["started"] = "not when it started"
    holder.record_path.write_text(json.dumps(record), encoding="utf-8")

    assert holder.reclaim_leftovers() == []
    assert asked == []
    assert orphan.poll() is None
    assert not holder.record_path.exists()


def test_a_corrupt_resident_record_is_quarantined(holder: Residency) -> None:
    holder.record_path.parent.mkdir(parents=True, exist_ok=True)
    holder.record_path.write_text("{half a record", encoding="utf-8")

    assert holder.reclaim_leftovers() == []
    assert not holder.record_path.exists()
    quarantined = list(holder.record_path.parent.glob("resident.json.bad-*"))
    assert len(quarantined) == 1
