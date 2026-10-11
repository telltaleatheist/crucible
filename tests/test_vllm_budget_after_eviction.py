"""The budget vLLM is started with rests on the card as it is after the eviction.

Victoria's RTX 3070 Laptop (8 GiB, WSL2, Crucible 1.0.140), 2026-10-11, jobs 3e27dab7 and
a2640dc7: yue2-3b resident, a load of qwen3.5-4b-8bit planned BEFORE yue2 was evicted, from
the card's free memory plus yue2's reclaimable figure (its estimate, not what it held). That
came to more than an idle 8 GiB card has, so the budget was the whole 7.0 GiB the desktop
allowance leaves (0.875); yue2 was then evicted, and vLLM, seeing the idle card's 6.95 GiB,
refused to start. The same model loaded with nothing to evict."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pytest

from crucible import accelerator
from crucible import engines as engines_module
from crucible.cardkinds import KIND_IMAGE
from crucible.config import load_config
from crucible.errors import JobError
from crucible.jobs.llm import LoadModelJobType, LoadParams
from crucible.memorybudget import GIB, MIB
from crucible.residency import Occupant, Residency, ResidentImage
from crucible.vram import VLLM_CUDA_CONTEXT_BYTES

from .conftest import FAKE_BACKEND, configure_box
from .fake_engine import FakeEngine

MODEL = "qwen3.5-4b-8bit"
TOTAL = 8 * GIB
ALLOWANCE = 1 * GIB
IDLE_FREE = 8017 * MIB  # 7.83 GiB: what Victoria's idle card reads
EIGHT_GIB_BACKEND = replace(
    FAKE_BACKEND,
    gpu=replace(FAKE_BACKEND.gpu, name="NVIDIA GeForce RTX 3070 Laptop GPU", vram_bytes=TOTAL),
)


class Card:
    """nvidia-smi's free figure: what the resident holds until it is stopped, then the
    readings the driver gives while what it held is handed back."""

    def __init__(self, held_bytes: int, after_stop: list[int]) -> None:
        self.free = IDLE_FREE - held_bytes
        self._after_stop = list(after_stop)
        self.stopped = False
        self.readings: list[int] = []

    def stop(self) -> None:
        self.stopped = True

    def probe(self) -> tuple[int, int]:
        if self.stopped and self._after_stop:
            self.free = self._after_stop.pop(0)
        self.readings.append(self.free)
        return self.free, TOTAL


class Held:
    """The resident's engine: stopping it is what the card's readings follow."""

    def __init__(self, card: Card) -> None:
        self._card = card
        self.pids: frozenset[int] = frozenset()

    def stop(self) -> None:
        self._card.stop()


class Ctx:
    def __init__(self) -> None:
        self.said: list[str] = []
        self.cancelled = False

    def warming(self, message: str) -> None:
        self.said.append(message)

    def progress(self, fraction: float, message: str, **extra: Any) -> None:
        self.said.append(message)

    def raise_if_cancelled(self) -> None:
        return None

    def done_extra(self, **extra: Any) -> None:
        return None


@pytest.fixture
def loader(
    home: Path,
    fake_env: Path,
    fake_weights: Callable[[str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Card | None], tuple[LoadModelJobType, list[FakeEngine], Ctx]]:
    configure_box(
        home, enable_llm=True, backend=EIGHT_GIB_BACKEND, desktop_allowance_bytes=ALLOWANCE
    )
    config = load_config(home)
    fake_weights(MODEL)
    monkeypatch.setattr("crucible.accelerator.refuse_if_card_lacks", lambda **_: None)
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    built: list[FakeEngine] = []

    def build(
        engine_name: str, python: Path, log_path: Path, *, library_dirs: tuple[Path, ...] = ()
    ) -> FakeEngine:
        engine = FakeEngine(python, log_path)
        built.append(engine)
        return engine

    monkeypatch.setattr(engines_module, "build_engine", build)
    monkeypatch.setattr(
        engines_module, "engine_model_name", lambda engine_name, model_dir, model_id: model_id
    )

    def make(card: Card | None) -> tuple[LoadModelJobType, list[FakeEngine], Ctx]:
        residency = Residency(config)
        if card is None:
            monkeypatch.setattr(accelerator, "probe_vram", lambda: (IDLE_FREE, TOTAL))
        else:
            monkeypatch.setattr(accelerator, "probe_vram", card.probe)
            resident = ResidentImage(
                model_id="dreamshaper-xl-lightning",
                backend=FAKE_BACKEND.kind,
                engine="diffusers",
                revision="0" * 40,
                fingerprint="fixture",
                device="cuda",
                dtype="float16",
                versions={},
                memory_bytes_estimate=7_300_000_000,
                log_path=home / "logs" / "resident.log",
                loaded_at="2026-10-11T17:20:06Z",
            )
            residency.occupy(
                KIND_IMAGE,
                resident.model_id,
                lambda: Occupant(resident, engine=Held(card)),  # type: ignore[arg-type]
                say=lambda _m: None,
            )
        return LoadModelJobType(config, EIGHT_GIB_BACKEND, residency), built, Ctx()

    return make


def _flag(args: list[str], name: str) -> str:
    return args[len(args) - 1 - args[::-1].index(name) + 1]


def _budget(engine: FakeEngine) -> int:
    return round(float(_flag(engine.args, "--gpu-memory-utilization")) * TOTAL)


def test_the_incident_eviction_then_load_plans_within_what_vllm_will_see(loader) -> None:
    """The resident holds 4.5 GiB and reclaims 6.8 GiB by its estimate: free plus that is
    past the idle card, and the old plan was 0.875. Stopping it hands the memory back over
    three readings."""
    card = Card(held_bytes=4608 * MIB, after_stop=[6 * GIB, 7 * GIB, IDLE_FREE])
    job, built, ctx = loader(card)

    forecast = job.requirements(MODEL, LoadParams()).forecast
    assert forecast.budget_bytes == TOTAL - ALLOWANCE, (
        "the forecast is what the incident's plan was: free plus the resident's figure "
        "reaches the allowance's 7.0 GiB"
    )

    job._load(ctx, MODEL, LoadParams())  # type: ignore[arg-type]

    assert card.stopped and len(built) == 1
    vllm_sees = IDLE_FREE - VLLM_CUDA_CONTEXT_BYTES
    assert abs(_budget(built[0]) - vllm_sees) <= TOTAL // 10_000, (
        "the budget is the settled idle card less vLLM's own CUDA context"
    )
    assert _budget(built[0]) <= vllm_sees + TOTAL // 10_000
    assert float(_flag(built[0].args, "--gpu-memory-utilization")) < 0.875
    assert card.readings[-2:] == [IDLE_FREE, IDLE_FREE], "planned once the card stopped rising"


def test_a_reading_taken_mid_release_is_not_the_one_planned_from(loader) -> None:
    card = Card(held_bytes=4608 * MIB, after_stop=[5 * GIB, IDLE_FREE])
    job, built, ctx = loader(card)
    job._load(ctx, MODEL, LoadParams())  # type: ignore[arg-type]
    assert abs(_budget(built[0]) - (IDLE_FREE - VLLM_CUDA_CONTEXT_BYTES)) <= TOTAL // 10_000


def test_a_load_with_nothing_to_evict_is_planned_exactly_as_before(loader) -> None:
    """The 1.0.121 subtraction still applies: 7.83 GiB free, so 7.83 less the CUDA context,
    which is under the allowance's 7.0."""
    job, built, ctx = loader(None)
    forecast = job.requirements(MODEL, LoadParams()).forecast
    job._load(ctx, MODEL, LoadParams())  # type: ignore[arg-type]
    assert forecast.budget_bytes == IDLE_FREE - VLLM_CUDA_CONTEXT_BYTES
    args = built[0].args
    assert _flag(args, "--gpu-memory-utilization") == f"{forecast.budget_bytes / TOTAL:.4f}"
    assert _flag(args, "--kv-cache-memory-bytes") == str(forecast.pool_bytes)
    assert any(MODEL in line and "KV pool" in line for line in ctx.said)


def test_a_card_that_after_the_eviction_cannot_hold_the_cache_refuses_by_name(loader) -> None:
    """Something else took the room while the resident was stopped: the load says so
    rather than starting vLLM on a budget the card does not have."""
    card = Card(held_bytes=4608 * MIB, after_stop=[4 * GIB])
    job, built, ctx = loader(card)
    with pytest.raises(JobError) as refused:
        job._load(ctx, MODEL, LoadParams())  # type: ignore[arg-type]
    assert refused.value.code == "insufficient_kv_cache"
    assert built == []


def test_the_settled_read_returns_once_free_stops_rising(monkeypatch) -> None:
    readings = iter([1 * GIB, 3 * GIB, 5 * GIB, 5 * GIB, 9 * GIB])
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (next(readings), 24 * GIB))
    said: list[str] = []
    state = accelerator.read_settled_state("cuda-linux", GIB, say=said.append)
    assert state.free_bytes == 5 * GIB and said == []


def test_a_card_still_rising_at_the_budget_is_planned_from_its_latest_reading_and_said(
    monkeypatch,
) -> None:
    rising = iter(range(1, 100))
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (next(rising) * GIB // 8, 24 * GIB))
    said: list[str] = []
    state = accelerator.read_settled_state(
        "cuda-linux", GIB, say=said.append, budget_seconds=0.0
    )
    assert state.free_bytes == 2 * GIB // 8
    assert len(said) == 1 and "still rising" in said[0]


def test_the_forecast_never_counts_more_back_than_the_card_has_in_use() -> None:
    from crucible.accelerator import AcceleratorState
    from crucible.manifests import load_manifest
    from crucible.vram import plan_vllm_memory

    spec = load_manifest(MODEL).spec(FAKE_BACKEND.kind)
    held = AcceleratorState(
        backend="cuda-linux", total_bytes=TOTAL, free_bytes=IDLE_FREE - 4608 * MIB,
        compute_apps=(), detail="fixture",
    )
    plan = plan_vllm_memory(
        model_id=MODEL, spec=spec, context=8192, card=held,
        desktop_allowance_bytes=0, reclaimable_bytes=7_300_000_000,
    )
    assert plan is not None and plan.free_bytes == TOTAL
