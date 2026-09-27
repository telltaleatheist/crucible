from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from crucible import procgroup, workers
from crucible.errors import JobCancelled

from .conftest import end_process_tree

FAKE_WORKER = Path(__file__).resolve().parent / "fake_asr_worker.py"

REQUEST = {
    "model_dir": "/nowhere",
    "ffmpeg": "/usr/bin/ffmpeg",
    "audio": "/nowhere/audio.m4b",
    "language": None,
    "vad_filter": True,
    "word_timestamps": True,
    "initial_prompt": None,
    "device": "cuda",
    "compute_type": "float16",
    "window_s": 900,
    "overlap_s": 15,
}


def run(tmp_path: Path, **kwargs):
    import sys

    parameters = {
        "python": Path(sys.executable),
        "script": FAKE_WORKER,
        "request": REQUEST,
        "log_path": tmp_path / "worker.log",
        "ready_silence_timeout": 30.0,
    }
    parameters.update(kwargs)
    return workers.run_worker(**parameters)


def test_a_run_returns_the_ready_line_and_every_result(tmp_path: Path) -> None:
    outcome = run(tmp_path)
    assert outcome.ready["windows"] == 2
    assert outcome.ready["duration_s"] == 1800.0
    assert outcome.ready["compute_type"] == "float16"
    assert len(outcome.results) == 2
    assert "window 0" in outcome.results[0]["segments"][0]["text"]
    assert "window 1" in outcome.results[1]["segments"][0]["text"]


def test_the_request_reaches_the_worker_verbatim(tmp_path: Path) -> None:
    transcript = tmp_path / "sent.jsonl"
    run(tmp_path, environment={"CRUCIBLE_FAKE_ASR_TRANSCRIPT": str(transcript)})
    sent = json.loads(transcript.read_text(encoding="utf-8").strip())
    assert sent == REQUEST


def test_ready_and_progress_are_handed_to_the_caller(tmp_path: Path) -> None:
    seen: list[dict] = []
    ready: list[dict] = []
    run(tmp_path, on_progress=seen.append, on_ready=ready.append)
    assert len(ready) == 1
    stages = [message["stage"] for message in seen]
    assert stages[:2] == ["decoding", "decoding"]
    assert "transcribing" in stages


def test_a_failed_unit_is_a_result_and_the_run_continues(tmp_path: Path) -> None:
    outcome = run(tmp_path, environment={"CRUCIBLE_FAKE_ASR_FAIL_WINDOW": "0"})
    assert len(outcome.results) == 2
    assert "error" in outcome.results[0]
    assert "segments" in outcome.results[1]


def test_a_non_json_line_on_fd_1_is_refused_naming_it(tmp_path: Path) -> None:
    with pytest.raises(workers.WorkerError) as caught:
        run(
            tmp_path,
            environment={"CRUCIBLE_FAKE_ASR_JUNK_LINE": "INFO: loading model weights"},
        )
    message = str(caught.value)
    assert "not JSON" in message
    assert "INFO: loading model weights" in message


def test_an_unknown_message_type_is_refused(tmp_path: Path) -> None:
    with pytest.raises(workers.WorkerError) as caught:
        run(
            tmp_path,
            environment={"CRUCIBLE_FAKE_ASR_JUNK_LINE": '{"type": "chunk"}'},
        )
    assert "type 'chunk'" in str(caught.value)


def test_a_worker_that_dies_early_is_refused_with_its_log(tmp_path: Path) -> None:
    with pytest.raises(workers.WorkerError) as caught:
        run(tmp_path, environment={"CRUCIBLE_FAKE_ASR_EXIT_CODE": "3"})
    message = str(caught.value)
    assert "exited 3" in message
    assert "told to exit before saying anything" in message


def test_stopping_without_done_is_an_unfinished_answer(tmp_path: Path) -> None:
    with pytest.raises(workers.WorkerError) as caught:
        run(tmp_path, environment={"CRUCIBLE_FAKE_ASR_NO_DONE": "1"})
    assert "without saying done" in str(caught.value)


def test_silence_before_ready_gives_up_by_name(tmp_path: Path) -> None:
    started = time.monotonic()
    with pytest.raises(workers.WorkerError) as caught:
        run(
            tmp_path,
            ready_silence_timeout=1.0,
            environment={"CRUCIBLE_FAKE_ASR_SILENT": "1"},
        )
    assert "said nothing at all for 1s" in str(caught.value)
    assert time.monotonic() - started < 30.0


def test_progress_keeps_the_silence_clock_from_firing(tmp_path: Path) -> None:
    outcome = run(
        tmp_path,
        ready_silence_timeout=2.0,
        environment={
            "CRUCIBLE_FAKE_ASR_DECODE_LINES": "4",
            "CRUCIBLE_FAKE_ASR_READY_DELAY_S": "0.5",
        },
    )
    assert outcome.ready["windows"] == 2


def test_a_short_stream_is_not_a_partial_answer(tmp_path: Path) -> None:
    outcome = run(tmp_path, environment={"CRUCIBLE_FAKE_ASR_SHORT": "1"})
    with pytest.raises(workers.WorkerError) as caught:
        workers.require_positional_results(outcome, outcome.ready["windows"], "window")
    assert "returned 1 window result(s) for 2 window(s)" in str(caught.value)


def test_the_counts_matching_returns_the_results(tmp_path: Path) -> None:
    outcome = run(tmp_path)
    assert workers.require_positional_results(outcome, 2, "window") == outcome.results


def test_a_cancel_terminates_the_worker(tmp_path: Path) -> None:
    calls = {"n": 0}

    def cancelled() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(JobCancelled):
        run(
            tmp_path,
            cancelled=cancelled,
            environment={"CRUCIBLE_FAKE_ASR_SLOW_S": "5"},
        )


def test_a_worker_that_ignores_sigterm_is_reported_never_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workers, "STOP_TIMEOUT_SECONDS", 1.0)
    pids: list[int] = []
    real_popen = workers.subprocess.Popen

    def watched(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        pids.append(process.pid)
        return process

    monkeypatch.setattr(workers.subprocess, "Popen", watched)

    polls = {"n": 0}

    def cancelled() -> bool:
        polls["n"] += 1
        return polls["n"] > 3

    environment = {
        "CRUCIBLE_FAKE_ASR_IGNORE_SIGTERM": "1",
        "CRUCIBLE_FAKE_ASR_SLOW_S": "30",
    }
    if procgroup.platform_kind() == procgroup.WIN32:
        with pytest.raises(JobCancelled):
            run(tmp_path, cancelled=cancelled, environment=environment)
        spawned = list(pids)
        assert spawned and all(not _alive(pid) for pid in spawned)
        return
    with pytest.raises(workers.WorkerError) as caught:
        run(tmp_path, cancelled=cancelled, environment=environment)
    assert "does not SIGKILL" in str(caught.value)
    for pid in list(pids):
        end_process_tree(pid)


def test_a_missing_interpreter_is_named(tmp_path: Path) -> None:
    with pytest.raises(workers.WorkerError) as caught:
        run(tmp_path, python=tmp_path / "no-such-python")
    assert "env is not installed" in str(caught.value)


def test_a_missing_worker_script_is_named(tmp_path: Path) -> None:
    with pytest.raises(workers.WorkerError) as caught:
        run(tmp_path, script=tmp_path / "no-such-worker.py")
    assert "no worker script at" in str(caught.value)


def _alive(pid: int) -> bool:
    import subprocess

    listed = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True
    ).stdout
    return str(pid) in listed
