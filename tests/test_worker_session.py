"""`crucible.workers.WorkerSession` — the worker that outlives a job.

Run against `tests/fake_align_worker.py` as a **real subprocess**, spawned exactly
the way the residency spawns the real one. Nothing here imports the worker: a
worker's env has no `crucible` in it, and a test that imported it would be
testing a relationship that does not exist at runtime.

What is being asserted is the one thing a session has that `run_worker` does not:
the process survives an exchange, the SAME process answers the next one, and it
goes when it is asked — without ever being SIGKILLed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from crucible import workers
from crucible.errors import JobCancelled

from .conftest import end_process_tree

FAKE_WORKER = Path(__file__).resolve().parent / "fake_align_worker.py"

LOAD = {"op": "load", "model_dir": "/nowhere", "device": "cuda", "dtype": "bfloat16"}


def align(*texts: str) -> dict:
    return {
        "op": "align",
        "language": "English",
        "max_audio_s": 300.0,
        "ffmpeg": "/usr/bin/ffmpeg",
        "chunks": [{"audio": f"/nowhere/{n}.flac", "text": t}
                   for n, t in enumerate(texts)],
    }


def session(tmp_path: Path, **kwargs) -> workers.WorkerSession:
    return workers.WorkerSession(
        python=Path(sys.executable),
        script=FAKE_WORKER,
        log_path=tmp_path / "session.log",
        **kwargs,
    )


# ---------------------------------------------------------------- it holds


def test_one_process_answers_every_request(tmp_path: Path) -> None:
    """The whole reason this class exists: hundreds of chunks, one model load."""
    held = session(tmp_path)
    loaded = held.start(LOAD, ready_silence_timeout=30.0)
    assert loaded.ready["dtype"] == "bfloat16"
    assert loaded.results == ()
    pids = held.pids
    assert len(pids) == 1

    first = held.send(align("one two"), ready_silence_timeout=30.0)
    second = held.send(align("three", "four five"), ready_silence_timeout=30.0)
    assert first.ready["chunks"] == 1
    assert second.ready["chunks"] == 2
    # The same process, after three exchanges.
    assert held.pids == pids
    assert held.alive
    held.stop()
    assert not held.alive
    assert held.pids == frozenset()


def test_results_come_back_positionally_across_exchanges(tmp_path: Path) -> None:
    held = session(tmp_path)
    held.start(LOAD, ready_silence_timeout=30.0)
    outcome = held.send(align("alpha beta", "gamma"), ready_silence_timeout=30.0)
    held.stop()
    texts = [[item["text"] for item in result["items"]] for result in outcome.results]
    assert texts == [["alpha", "beta"], ["gamma"]]


def test_every_request_reaches_the_worker_on_the_same_stdin(tmp_path: Path) -> None:
    transcript = tmp_path / "sent.jsonl"
    held = session(
        tmp_path,
        environment={"CRUCIBLE_FAKE_ALIGN_TRANSCRIPT": str(transcript)},
    )
    held.start(LOAD, ready_silence_timeout=30.0)
    held.send(align("one"), ready_silence_timeout=30.0)
    held.stop()
    ops = [json.loads(line)["op"] for line in transcript.read_text().splitlines()]
    assert ops == ["load", "align"]


def test_stopping_closes_stdin_first_and_the_worker_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The polite door: EOF on stdin ends the worker's own loop, so CUDA is
    released the way its own code expects. SIGTERM is only the backstop."""
    # Watched at `procgroup.ask_to_stop`, the one place the polite signal is
    # sent on every platform — `os.killpg` does not exist on win32.
    signalled: list[int] = []
    real_ask = workers.procgroup.ask_to_stop
    monkeypatch.setattr(
        workers.procgroup,
        "ask_to_stop",
        lambda process: (signalled.append(process.pid), real_ask(process))[1],
    )
    held = session(tmp_path)
    held.start(LOAD, ready_silence_timeout=30.0)
    held.stop()
    assert signalled == []


def test_stopping_twice_is_not_an_error(tmp_path: Path) -> None:
    held = session(tmp_path)
    held.start(LOAD, ready_silence_timeout=30.0)
    held.stop()
    held.stop()


# ---------------------------------------------------------------- it refuses


def test_a_load_that_fails_is_reported_and_nothing_is_left_running(
    tmp_path: Path,
) -> None:
    held = session(tmp_path, environment={"CRUCIBLE_FAKE_ALIGN_LOAD_FAIL": "1"})
    with pytest.raises(workers.WorkerError) as caught:
        held.start(LOAD, ready_silence_timeout=30.0)
    assert "told not to load" in str(caught.value)
    assert not held.alive
    assert held.pids == frozenset()


def test_sending_before_starting_is_refused(tmp_path: Path) -> None:
    with pytest.raises(workers.WorkerError) as caught:
        session(tmp_path).send(align("x"), ready_silence_timeout=30.0)
    assert "has not been started" in str(caught.value)


def test_starting_twice_is_refused(tmp_path: Path) -> None:
    held = session(tmp_path)
    held.start(LOAD, ready_silence_timeout=30.0)
    try:
        with pytest.raises(workers.WorkerError) as caught:
            held.start(LOAD, ready_silence_timeout=30.0)
        assert "already started" in str(caught.value)
    finally:
        held.stop()


def test_a_worker_that_died_between_jobs_is_named_not_written_to(
    tmp_path: Path,
) -> None:
    """A held process can go without anybody asking; the next send must say so
    rather than write a request into a closed pipe."""
    held = session(tmp_path, environment={"CRUCIBLE_FAKE_ALIGN_DIE_AFTER": "0"})
    held.start(LOAD, ready_silence_timeout=30.0)
    with pytest.raises(workers.WorkerError) as caught:
        held.send(align("one"), ready_silence_timeout=30.0)
    assert "in the middle of a request" in str(caught.value)
    assert not held.alive

    with pytest.raises(workers.WorkerError) as caught:
        held.send(align("two"), ready_silence_timeout=30.0)
    assert "no longer running" in str(caught.value)


def test_a_non_json_line_on_fd_1_is_refused_naming_it(tmp_path: Path) -> None:
    held = session(
        tmp_path,
        environment={
            "CRUCIBLE_FAKE_ALIGN_JUNK_LINE": "whisperx.alignment - WARNING - Failed"
        },
    )
    held.start(LOAD, ready_silence_timeout=30.0)
    with pytest.raises(workers.WorkerError) as caught:
        held.send(align("one"), ready_silence_timeout=30.0)
    message = str(caught.value)
    assert "not JSON" in message
    assert "whisperx.alignment" in message


def test_silence_before_ready_gives_up_by_name(tmp_path: Path) -> None:
    held = session(tmp_path, environment={"CRUCIBLE_FAKE_ALIGN_SILENT": "1"})
    with pytest.raises(workers.WorkerError) as caught:
        held.start(LOAD, ready_silence_timeout=1.0)
    assert "said nothing at all for 1s" in str(caught.value)


def test_a_load_delay_does_not_trip_the_silence_clock(tmp_path: Path) -> None:
    """It is a silence timeout, not a deadline — and a model load is the one
    thing that is legitimately quiet for a long time."""
    held = session(tmp_path, environment={"CRUCIBLE_FAKE_ALIGN_LOAD_DELAY_S": "1.5"})
    outcome = held.start(LOAD, ready_silence_timeout=5.0)
    held.stop()
    assert outcome.ready["device"] == "cuda"


def test_stopping_without_done_is_an_unfinished_answer(tmp_path: Path) -> None:
    held = session(tmp_path, environment={"CRUCIBLE_FAKE_ALIGN_NO_DONE": "1"})
    held.start(LOAD, ready_silence_timeout=30.0)
    with pytest.raises(workers.WorkerError) as caught:
        held.send(align("one"), ready_silence_timeout=30.0)
    assert "in the middle of a request" in str(caught.value)


def test_a_cancel_terminates_the_session_and_kills_it_for_reuse(
    tmp_path: Path,
) -> None:
    """A cancel mid-exchange cannot leave a worker whose stdin is half a request."""
    held = session(tmp_path, environment={"CRUCIBLE_FAKE_ALIGN_SLOW_S": "5"})
    held.start(LOAD, ready_silence_timeout=30.0)
    calls = {"n": 0}

    def cancelled() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(JobCancelled):
        held.send(align("one", "two"), ready_silence_timeout=30.0, cancelled=cancelled)
    assert not held.alive
    with pytest.raises(workers.WorkerError):
        held.send(align("three"), ready_silence_timeout=30.0)


def test_a_worker_that_ignores_sigterm_is_reported_never_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Crucible does not SIGKILL a process that may be holding CUDA.

    The real wait is three minutes; shortened here so the suite does not spend
    them. What is asserted is that the timeout produces a named refusal rather
    than an escalation.
    """
    monkeypatch.setattr(workers, "STOP_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(workers, "STOP_ON_EOF_SECONDS", 1.0)
    held = session(
        tmp_path,
        environment={
            # Refuses BOTH doors: it will not exit when stdin closes, and it
            # ignores the SIGTERM that follows.
            "CRUCIBLE_FAKE_ALIGN_IGNORE_EOF": "1",
            "CRUCIBLE_FAKE_ALIGN_IGNORE_SIGTERM": "1",
        },
    )
    held.start(LOAD, ready_silence_timeout=30.0)
    pids = set(held.pids)
    if workers.procgroup.platform_kind() == workers.procgroup.WIN32:
        # win32 has no WSL2 wedge to protect: a worker deaf to both doors has
        # its tree terminated and `stop()` returns (`crucible/procgroup.py`).
        held.stop()
        assert not held.alive
        return
    with pytest.raises(workers.WorkerError) as caught:
        held.stop()
    assert "does not SIGKILL" in str(caught.value)

    # This test made the process; this test cleans it up. Nothing in Crucible
    # will, by design.
    # A SNAPSHOT: every `taskkill` below is itself a Popen, and the recorder
    # this test installed would otherwise append it to the list being walked.
    for pid in list(pids):
        end_process_tree(pid)


def test_a_missing_interpreter_is_named(tmp_path: Path) -> None:
    held = workers.WorkerSession(
        python=tmp_path / "no-such-python",
        script=FAKE_WORKER,
        log_path=tmp_path / "session.log",
    )
    with pytest.raises(workers.WorkerError) as caught:
        held.start(LOAD, ready_silence_timeout=5.0)
    assert "env is not installed" in str(caught.value)
