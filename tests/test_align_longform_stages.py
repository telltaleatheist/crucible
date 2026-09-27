from __future__ import annotations

import pytest

from crucible.jobs.alignlongform import stages
from crucible.jobs.asr import OVERLAP_SECONDS, WINDOW_SECONDS


def window(words):
    return {"segments": [{"words": [{"word": w, "start": t, "end": t + 0.2} for w, t in words]}]}


def run(monkeypatch, results):
    outcome = stages.workers.WorkerOutcome(ready={}, results=tuple(results))
    monkeypatch.setattr(stages.workers, "run_worker", lambda **_: outcome)
    monkeypatch.setattr(
        stages.workers, "worker_environment", lambda *_args, **_kw: {}
    )
    return stages.transcribe(
        python=stages.Path("/nowhere/python"),
        weights_dir=stages.Path("/nowhere/weights"), ffmpeg="ffmpeg",
        audio=stages.Path("/nowhere/book.m4b"), language="en",
        log_path=stages.Path("/nowhere/log"),
    )


def test_the_constants_are_the_asr_job_s_own(monkeypatch) -> None:
    assert stages.WINDOW_SECONDS is WINDOW_SECONDS
    assert stages.OVERLAP_SECONDS is OVERLAP_SECONDS
    assert isinstance(WINDOW_SECONDS, int), "the worker refuses a float by name"


def test_one_window_is_unshifted(monkeypatch) -> None:
    words = run(monkeypatch, [window([("one", 0.0), ("two", 1.5)])])
    assert [t for _, t in words] == [0.0, 1.5]


def test_the_SECOND_window_is_shifted_by_a_whole_window(monkeypatch) -> None:
    words = run(monkeypatch, [
        window([("a", 0.0)]),
        window([("b", 0.0), ("c", 10.0)]),
    ])
    assert [t for _, t in words] == [0.0, float(WINDOW_SECONDS), float(WINDOW_SECONDS) + 10.0]


def test_the_shift_is_index_times_window_not_a_running_sum(monkeypatch) -> None:
    words = run(monkeypatch, [window([(f"w{i}", 0.0)]) for i in range(5)])
    assert [t for _, t in words] == [float(i * WINDOW_SECONDS) for i in range(5)]


def test_words_come_back_in_book_order(monkeypatch) -> None:
    words = run(monkeypatch, [
        window([("a", 1.0), ("b", 2.0)]),
        window([("c", 1.0), ("d", 2.0)]),
    ])
    times = [t for _, t in words]
    assert times == sorted(times)


def test_a_FAILED_window_fails_the_job_rather_than_leaving_a_hole(monkeypatch) -> None:
    with pytest.raises(stages.StageFailed) as caught:
        run(monkeypatch, [window([("a", 0.0)]), {"error": "cuda oom"}])
    assert caught.value.code == "transcribe_window_failed"
    assert "window 1" in str(caught.value)
    assert "dropped from the VTT" in str(caught.value)


def test_a_window_with_no_words_is_not_an_error(monkeypatch) -> None:
    words = run(monkeypatch, [window([("a", 0.0)]), {"segments": []}, window([("b", 0.0)])])
    assert [t for _, t in words] == [0.0, 2.0 * WINDOW_SECONDS]
