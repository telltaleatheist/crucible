"""What every vLLM process Crucible starts runs under (crucible/engines/vllm.py
``ENVIRONMENT``): the resident llm engine, and the ASR worker that reads the same env."""

from __future__ import annotations

from pathlib import Path

from crucible.engines.vllm import ENVIRONMENT, VllmEngine
from crucible.jobs.asr import qwen


def test_vllm_never_probes_deep_gemm() -> None:
    """DeepGEMM needs a Hopper or Blackwell card and nvcc, and Crucible places no CUDA
    toolkit. With the switch on, vLLM 0.29.0's warmup trial-imports the vendored
    deep_gemm, which asserts a CUDA home, and logs the traceback on every start."""
    assert ENVIRONMENT["VLLM_USE_DEEP_GEMM"] == "0"


def test_the_engine_and_the_asr_worker_both_carry_it(tmp_path: Path) -> None:
    engine = VllmEngine(python=tmp_path / "python", log_path=tmp_path / "e.log")
    assert engine.environment()["VLLM_USE_DEEP_GEMM"] == "0"
    assert qwen.WORKER_ENVIRONMENT_FOR_ENGINE["vllm"]["VLLM_USE_DEEP_GEMM"] == "0"
