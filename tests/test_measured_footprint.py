"""A resident's reclaimable bytes are what its load measured, not only its estimate.

B-Side, 2026-10-06: qwen3.5-4b-bside's manifest estimates 13.4 GB, but vLLM sizes its KV
pool to the free card and the engine held 19 GB. Crucible credited itself only the estimate,
read the rest as a process it did not start, and refused its own session's image item as
accelerator_busy for 18 minutes instead of evicting the LLM.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible import accelerator
from crucible.backend import CUDA_LINUX, MLX_DARWIN
from crucible.memorybudget import GIB
from crucible.residency import Occupant, Residency, ResidentModel

ESTIMATE = 13_393_958_393
CARD = 24 * GIB
DESKTOP = 3 * GIB
# The real measurement; tests/conftest.py stubs it out for every other test.
MEASURE = Residency._card_used_bytes


def _resident(home: Path) -> ResidentModel:
    return ResidentModel(
        model_id="qwen3.5-4b-bside",
        backend=CUDA_LINUX,
        engine="vllm",
        engine_model_name="qwen3.5-4b-bside",
        base_url="http://127.0.0.1:1",
        port=1,
        revision="r",
        max_model_len=16384,
        memory_bytes_estimate=ESTIMATE,
        log_path=home / "engine.log",
        loaded_at="2026-10-06T04:53:56Z",
        engine_args=(),
    )


def _load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, took: int):
    monkeypatch.setattr(Residency, "_card_used_bytes", MEASURE)
    card = {"used": int(1.4 * GIB)}
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (CARD - card["used"], CARD))
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    residency = Residency(SimpleNamespace(backend_kind=backend, home=tmp_path))  # type: ignore[arg-type]

    def start() -> Occupant:
        card["used"] += took
        return Occupant(_resident(tmp_path))

    residency.occupy("llm", "qwen3.5-4b-bside", start, say=lambda _: None)
    return residency


def test_an_engine_that_takes_more_than_its_estimate_is_reclaimable_in_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    took = int(19.3 * GIB)
    residency = _load(tmp_path, monkeypatch, CUDA_LINUX, took)
    assert residency.reclaimable_bytes() == took
    assert residency.reclaimable_bytes(excluding="qwen3.5-4b-bside") == 0

    state = accelerator.guard(
        CUDA_LINUX,
        model_id="qwen-image-2.1",
        need_bytes=20 * GIB,
        desktop_allowance_bytes=DESKTOP,
        reclaimable_bytes=residency.reclaimable_bytes(excluding="qwen-image-2.1"),
    )
    assert state.total_bytes == CARD


def test_counting_only_the_estimate_is_what_refused_the_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _load(tmp_path, monkeypatch, CUDA_LINUX, int(19.3 * GIB))
    with pytest.raises(Exception) as refused:
        accelerator.guard(
            CUDA_LINUX,
            model_id="qwen-image-2.1",
            need_bytes=20 * GIB,
            desktop_allowance_bytes=DESKTOP,
            reclaimable_bytes=ESTIMATE,
        )
    assert getattr(refused.value, "code", None) == accelerator.ACCELERATOR_BUSY


def test_an_engine_under_its_estimate_keeps_the_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    residency = _load(tmp_path, monkeypatch, CUDA_LINUX, 2 * GIB)
    assert residency.reclaimable_bytes() == ESTIMATE


def test_the_mac_measures_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    residency = _load(tmp_path, monkeypatch, MLX_DARWIN, int(19.3 * GIB))
    assert residency.reclaimable_bytes() == ESTIMATE
