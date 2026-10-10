"""A load says what its engine is doing while it starts, where a client reads the job.

The first load after an install spends minutes compiling kernels. vLLM said so only in
its own log, and the load job's `message` read "loading <model>" the whole time, so
B-Sides on Victoria's laptop (2026-10-09) took it for hung. Now the engine's start
phase, read from its log and timed from when Crucible first saw it, is the warming
message, and a warming message is the job's `message` and its `job.progress`. Nothing
here starts an engine: the vLLM log is written by the test.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from crucible.engines import base as engine_base
from crucible.engines.vllm import COLD_CACHE, VllmEngine

from .test_events import _drain, _open_on
from .test_queue import body, status, wait_for

HEADER = "=== crucible vllm engine, 2026-10-09 21:13:40"
CORE = "(EngineCore pid=4242) INFO 10-09 21:14:0{} [{}] "


def started_log(path: Path, *lines: str) -> VllmEngine:
    path.write_text("\n".join([HEADER, *lines]) + "\n", encoding="utf-8")
    return VllmEngine(python=path.parent / "python", log_path=path)


def deadline_in(seconds: float) -> float:
    return time.monotonic() + seconds


def test_a_first_load_says_it_is_compiling_and_for_how_long(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "engine.log"
    engine = started_log(
        log,
        CORE.format(1, "default_loader.py:430") + "Loading weights took 15.64 seconds",
        CORE.format(4, "qwen_triton_warmup.py:270")
        + "Warming up Qwen Triton kernels for model_type=qwen3_5_text.",
    )
    first = engine.warming_message(10, deadline_in(500))
    assert first.startswith("vllm loading; 20s elapsed, ")
    assert first.endswith(
        f"compiling Qwen's linear-attention Triton kernels ({COLD_CACHE}), 0s so far"
    )
    later = time.monotonic() + 42
    monkeypatch.setattr(engine_base.time, "monotonic", lambda: later)
    assert engine.warming_message(31, deadline_in(500)).endswith(", 42s so far")


def test_the_latest_phase_in_the_log_is_the_one_said(tmp_path: Path) -> None:
    log = tmp_path / "engine.log"
    engine = started_log(
        log,
        CORE.format(4, "qwen_triton_warmup.py:270") + "Warming up Qwen Triton kernels.",
        "(EngineCore pid=4242) Capturing CUDA graphs (PIECEWISE):  40%|####  | 2/5",
    )
    said = engine.warming_message(1, deadline_in(500))
    assert said.endswith("capturing CUDA graphs, 0s so far"), said


def test_a_log_with_no_known_phase_still_quotes_its_last_line(tmp_path: Path) -> None:
    engine = started_log(tmp_path / "engine.log", "something vLLM printed")
    assert engine.starting_phase() is None
    assert engine.warming_message(1, deadline_in(500)).endswith(" — something vLLM printed")


def test_only_the_last_run_is_read_for_its_phase(tmp_path: Path) -> None:
    log = tmp_path / "engine.log"
    log.write_text(
        "\n".join([
            HEADER,
            "(EngineCore pid=1) Capturing CUDA graphs (FULL): 100%",
            HEADER,
            "(APIServer pid=2) INFO starting",
        ]) + "\n",
        encoding="utf-8",
    )
    engine = VllmEngine(python=tmp_path / "python", log_path=log)
    assert engine.starting_phase() is None


def test_a_warming_message_is_the_jobs_message_and_its_progress(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    feed = client.portal.call(_open_on, client, "job")
    echo = client.app.state.store.registry["echo"]
    said = "vllm loading; 96s elapsed, 504s before give-up — compiling kernels, 80s so far"
    release = threading.Event()

    def run(job: Any, ctx: Any) -> None:
        ctx.progress(0.0, "loading echo")
        ctx.warming(said)
        release.wait(10)
        ctx.progress(1.0, "echo is resident")
        ctx.done_extra()

    monkeypatch.setattr(echo, "run", run)
    answer = client.post("/v1/jobs", headers=auth, json=body())
    assert answer.status_code == 202, answer.text
    job_id = answer.json()["job_id"]
    try:
        wait_for(
            lambda: client.get(f"/v1/jobs/{job_id}", headers=auth).json()["message"] == said,
            "the job's message to be the warming message",
        )
        state = client.get(f"/v1/jobs/{job_id}", headers=auth).json()
        assert state["progress"] == 0.0, "a warming message leaves the fraction alone"
        messages: list[str] = []
        cursor = 0

        def announced() -> bool:
            nonlocal cursor
            seen, cursor = _drain(feed, cursor)
            messages.extend(
                e["data"]["message"] for e in seen
                if e["event"] == "job.progress" and e["data"]["job_id"] == job_id
            )
            return said in messages

        wait_for(announced, "job.progress to carry the warming message")
    finally:
        release.set()
        feed.close()
    wait_for(lambda: status(client, auth, job_id) == "done", "the job to finish")
