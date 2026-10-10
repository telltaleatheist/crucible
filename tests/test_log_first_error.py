"""A failed load's refusal leads with the error that stopped it, then the log's tail.

On Victoria's laptop (2026-10-09) a vLLM load died on its KV cache, and what B-Sides
showed was the last 40 lines of the engine log: the API server's own traceback, ending
"Engine core initialization failed. See root cause above", with the engine core's
``ValueError`` that named the cause cut off above it."""

from __future__ import annotations

from pathlib import Path

from crucible import workers
from crucible.engines.vllm import VllmEngine
from crucible.logtail import first_error_of_last_run, led_by_first_error

KV = (
    "ValueError: To serve at least one request with the models's max seq len (8192), "
    "(0.35 GiB KV cache is needed, which is larger than the available KV cache memory "
    "(0.29 GiB). Based on the available memory, the estimated maximum model length is "
    "6656. Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when "
    "initializing the engine."
)

CORE = "(EngineCore pid=4242) "
API = "(APIServer pid=4100) "
ERROR = "ERROR 10-09 21:14:07 [core.py:1100] "
WARNING = "WARNING 10-09 21:14:02 [import_utils.py:408] "


def vllm_failure() -> list[str]:
    """The shape vLLM 0.29.0 writes when its engine core cannot start: a WARNING
    traceback it carried on from, the core's logged ERROR traceback (with an API
    server line written between its frames), the core's raw traceback, then the API
    server's wrapper, and enough after it that a 40-line tail holds none of the cause."""
    lines = [
        "=== crucible vllm engine, 2026-10-09 21:13:40",
        "=== /home/v/.crucible/envs/llm/bin/python -m vllm.entrypoints.openai.api_server",
        f"{CORE}{WARNING}Module vllm.third_party.deep_gemm was found but failed to import",
        f"{CORE}{WARNING}Traceback (most recent call last):",
        f'{CORE}{WARNING}  File "deep_gemm/__init__.py", line 120, in _find_cuda_home',
        f"{CORE}{WARNING}    assert cuda_home is not None",
        f"{CORE}{WARNING}AssertionError",
        f"{CORE}{ERROR}EngineCore failed to start.",
        f"{CORE}{ERROR}Traceback (most recent call last):",
        f'{CORE}{ERROR}  File "vllm/v1/engine/core.py", line 1090, in run_engine_core',
        f"{API}INFO 10-09 21:14:07 [api_server.py:12] waiting for the engine core",
        f"{CORE}{ERROR}    engine_core = EngineCoreProc(*args, **kwargs)",
        f'{CORE}{ERROR}  File "vllm/v1/core/kv_cache_utils.py", line 700, in check',
        f"{CORE}{ERROR}    raise ValueError(",
        f"{CORE}{ERROR}{KV}",
        f"{CORE}Process EngineCore:",
        f"{CORE}Traceback (most recent call last):",
        f'{CORE}  File "multiprocessing/process.py", line 314, in _bootstrap',
        f"{CORE}{KV}",
        f"{API}Traceback (most recent call last):",
    ]
    lines += [f'{API}  File "vllm/entrypoints/cli/serve.py", line {n}, in run' for n in range(40)]
    lines += [
        f"{API}RuntimeError: Engine core initialization failed. See root cause above. "
        "Failed core proc(s): {}",
    ]
    return lines


def write_log(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_first_error_is_the_engine_cores_not_the_wrapper(tmp_path: Path) -> None:
    log = write_log(tmp_path / "engine.log", vllm_failure())
    assert first_error_of_last_run(log) == KV


def test_a_traceback_logged_as_a_warning_is_not_the_cause(tmp_path: Path) -> None:
    lines = vllm_failure()[:7]
    log = write_log(tmp_path / "engine.log", lines)
    assert first_error_of_last_run(log) is None


def test_only_the_last_run_is_read(tmp_path: Path) -> None:
    earlier = [
        "=== crucible vllm engine, 2026-10-08 10:00:00",
        "Traceback (most recent call last):",
        '  File "x.py", line 1, in <module>',
        "RuntimeError: an earlier run's failure",
    ]
    log = write_log(tmp_path / "engine.log", earlier + vllm_failure())
    assert first_error_of_last_run(log) == KV
    clean = earlier + ["=== crucible vllm engine, 2026-10-09 09:00:00", "INFO started"]
    assert first_error_of_last_run(write_log(tmp_path / "clean.log", clean)) is None


def test_a_bare_exception_line_is_used_when_there_is_no_traceback(tmp_path: Path) -> None:
    lines = [
        "=== crucible worker align_worker.py, 2026-10-09 21:00:00",
        "loading the aligner",
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "exiting",
    ]
    log = write_log(tmp_path / "worker.log", lines)
    assert first_error_of_last_run(log) == lines[2]


def test_the_engine_refusal_leads_with_it_then_the_tail(tmp_path: Path) -> None:
    log = write_log(tmp_path / "engine.log", vllm_failure())
    report = VllmEngine(python=tmp_path / "python", log_path=log).log_report()
    first, second = report.split("\n")[:2]
    assert first == f"First error in its log: {KV}"
    assert second == f"Last 40 lines of {log}:"
    assert "KV cache is needed" not in "\n".join(report.split("\n")[2:]), (
        "the tail alone would not have said it"
    )
    assert report.endswith("Failed core proc(s): {}")


def test_a_worker_refusal_leads_with_it_too(tmp_path: Path) -> None:
    log = write_log(tmp_path / "worker.log", vllm_failure())
    report = workers._log_tail(log)
    assert report.startswith(f"First error in its log: {KV}\nLast 40 lines of the latest run")


def test_a_log_with_no_error_is_quoted_as_before(tmp_path: Path) -> None:
    log = write_log(tmp_path / "engine.log", ["=== crucible vllm engine, now", "INFO up"])
    assert led_by_first_error(log, "the tail") == "the tail"
    assert first_error_of_last_run(tmp_path / "missing.log") is None
