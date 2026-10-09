from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from crucible import (
    accelerator,
    asrplan,
    capabilityclasses,
    capabilityquery,
    fit,
    memorybudget,
    servingplan,
    ttsplan,
    verdict,
    vram,
)
from crucible.backend import CUDA_LINUX, LLAMA_WINDOWS
from crucible.capabilityrecord import CapabilityRecord, CapabilityRow
from crucible.engines import narrator
from crucible.manifests import BackendSpec, ManifestError

GIB = memorybudget.GIB
REPO = Path(__file__).resolve().parents[1]


def test_the_capability_shim_is_gone() -> None:
    assert importlib.util.find_spec("crucible.capability") is None


def test_the_class_verdict_has_a_name_apart_from_the_decision_door() -> None:
    assert verdict.decide is verdict.decide_capabilities


def test_one_owner_for_the_memory_arithmetic() -> None:
    assert vram.engine_budget_bytes is memorybudget.engine_budget_bytes
    assert accelerator.available_bytes is memorybudget.available_bytes
    assert memorybudget.available_bytes(4 * GIB, 8 * GIB) == 0
    assert memorybudget.engine_budget_bytes(24 * GIB, 3 * GIB, 30 * GIB) == 21 * GIB
    assert memorybudget.engine_budget_bytes(24 * GIB, 3 * GIB, 10 * GIB) == 10 * GIB
    assert memorybudget.engine_budget_bytes(24 * GIB, 3 * GIB, -1) == 0
    assert memorybudget.gib_text(3 * GIB) == "3.0 GiB"
    assert memorybudget.gib_text(3 * GIB, 2) == "3.00 GiB"


def test_accelerator_imports_neither_capability_nor_the_engines() -> None:
    probe = (
        "import sys, crucible.accelerator; "
        "print('\\n'.join(sorted(sys.modules)))"
    )
    loaded = set(
        subprocess.run(
            [sys.executable, "-c", probe],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    )
    assert "crucible.verdict" not in loaded
    assert "crucible.engines" not in loaded


def test_serving_variant_has_one_home_both_ladders_use() -> None:
    ladder = ttsplan.variants(1, GIB)
    assert ladder
    assert all(isinstance(variant, servingplan.ServingVariant) for variant in ladder)
    assert asrplan.ServingVariant is servingplan.ServingVariant
    assert asrplan.LADDER_BACKEND == CUDA_LINUX


def test_the_batched_classes_share_one_working_context_and_one_goal() -> None:
    by_name = capabilityclasses.BY_NAME
    for name in ("translate", "simplify", "analysis"):
        assert by_name[name].work is capabilityclasses.BATCHED_BLOCKS_WORK
        assert by_name[name].goal is capabilityclasses.CHAT_GOAL
    assert by_name["generate"].candidates is capabilityclasses.TEXT_MODELS


def _spec(args: tuple[str, ...]) -> BackendSpec:
    return BackendSpec(
        backend=CUDA_LINUX,
        engine="vllm",
        hf_repo="fixture/repo",
        revision="0" * 40,
        memory_bytes_estimate=GIB,
        engine_args=args,
        context_default=None,
        memory=None,
    )


def test_max_num_seqs_reads_like_vllm_and_refuses_a_non_number_by_name() -> None:
    assert vram.max_num_seqs(_spec(("--max-num-seqs", "8", "--max-num-seqs=12"))) == 12
    with pytest.raises(ManifestError, match=r"qwen-x's engine_args give --max-num-seqs 'many'.*qwen-x\.toml"):
        vram.max_num_seqs(_spec(("--max-num-seqs", "many")), "qwen-x")


def test_both_cuda_readings_describe_the_card_the_same_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(accelerator, "nvidia_smi_path", lambda: "nvidia-smi")
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (20 * GIB, 24 * GIB))
    linux = accelerator.read_state(CUDA_LINUX, 3 * GIB)
    windows = accelerator.read_state(LLAMA_WINDOWS, 3 * GIB)
    assert linux.detail == windows.detail == (
        "20.0 GiB free of 24.0 GiB, 0 compute app(s), desktop allowance 3.0 GiB"
    )
    assert windows.backend == LLAMA_WINDOWS


def test_narrator_walks_the_process_table_through_accelerator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for pid, environ in ((11, b"A=1\0NARRATOR_HIGGS3_OWNER=7\0"), (12, b"B=2\0")):
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "environ").write_bytes(environ)
    (tmp_path / "self").mkdir()
    monkeypatch.setattr(accelerator, "PROC", tmp_path)
    assert narrator.processes_launched_by(7) == frozenset({11})


def _counting_catalog(root: Path) -> tuple[fit.CatalogCandidates, list[Path]]:
    loads: list[Path] = []

    def load(directory: Path | None = None) -> dict[str, object]:
        assert directory is not None
        loads.append(directory)
        return {}

    return fit.CatalogCandidates(load, directory=lambda: root), loads


def test_candidates_are_parsed_once_until_a_manifest_changes(tmp_path: Path) -> None:
    manifest = tmp_path / "one.toml"
    manifest.write_text("x = 1\n", encoding="utf-8")
    source, loads = _counting_catalog(tmp_path)
    source(CUDA_LINUX)
    source(CUDA_LINUX)
    assert len(loads) == 1
    stat = manifest.stat()
    os.utime(manifest, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    source(CUDA_LINUX)
    assert len(loads) == 2
    (tmp_path / "two.toml").write_text("y = 2\n", encoding="utf-8")
    source(CUDA_LINUX)
    assert len(loads) == 3


def _generate_record() -> CapabilityRecord:
    return CapabilityRecord(
        backend_kind=CUDA_LINUX,
        total_bytes=24 * GIB,
        desktop_allowance_bytes=3 * GIB,
        rows=(
            CapabilityRow(
                capability="generate",
                enabled=True,
                selected="",
                reason="fixture",
                summary="fixture",
                shortfall_bytes=0,
            ),
        ),
    )


def _sized_capability_call() -> list[dict[str, object]]:
    return capabilityquery.served_rows(
        _generate_record(),
        gpu_vendor="nvidia",
        chosen={},
        routes={},
        capability_class="generate",
        context_tokens="16384",
        concurrency=None,
        audio_low_vram=False,
    )


def test_a_repeated_capability_query_reads_no_manifest_twice() -> None:
    fit.forget_cached_catalogs()
    started = time.perf_counter()
    cold = _sized_capability_call()
    cold_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(10):
        warm = _sized_capability_call()
    warm_seconds = (time.perf_counter() - started) / 10
    print(f"GET /v1/capability?class=generate cold {cold_seconds * 1000:.1f} ms, warm {warm_seconds * 1000:.1f} ms")
    assert warm == cold
    assert warm_seconds < cold_seconds
