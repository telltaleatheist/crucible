"""The multi-window assembly, which a 15-second probe cannot exercise.

`align-longform` was proven end to end on the card against a 15.09 s clip. That
clip is ONE window, so the arithmetic that turns per-window timings into
book-absolute ones never ran — and it is the piece whose failure is silent and
grows with the book: every word in window two lands 900 s wrong, every word in
window three 1800 s wrong, and the VTT still looks like a VTT.

The `asr` worker returns WINDOW-RELATIVE timings and NO index: position in the
stream is the window's whole identity, because an index a worker reports is an
index a worker can get wrong. So the shift is the server's arithmetic, and this
suite holds it to the same expression the `asr` job type uses — those two must
not drift, since they are the same fact about the same worker.
"""

from __future__ import annotations

import pytest

from crucible.jobs.alignlongform import stages
from crucible.jobs.asr import OVERLAP_SECONDS, WINDOW_SECONDS


class FakeOutcome:
    def __init__(self, results):
        self.results = results


def window(words):
    """One `result` as the asr worker sends it: segments of window-relative words."""
    return {"segments": [{"words": [{"word": w, "start": t, "end": t + 0.2} for w, t in words]}]}


def run(monkeypatch, results):
    monkeypatch.setattr(stages.workers, "run_worker", lambda **_: FakeOutcome(results))
    monkeypatch.setattr(
        stages.workerenv, "worker_environment", lambda *_args, **_kw: {}
    )
    return stages.transcribe(
        home=stages.Path("/nowhere"), python=stages.Path("/nowhere/python"),
        weights_dir=stages.Path("/nowhere/weights"), ffmpeg="ffmpeg",
        audio=stages.Path("/nowhere/book.m4b"), language="en",
        log_path=stages.Path("/nowhere/log"),
    )


def test_the_constants_are_the_asr_job_s_own(monkeypatch) -> None:
    """Imported, not copied. This file once carried 600.0/5.0, both invented."""
    assert stages.WINDOW_SECONDS is WINDOW_SECONDS
    assert stages.OVERLAP_SECONDS is OVERLAP_SECONDS
    assert isinstance(WINDOW_SECONDS, int), "the worker refuses a float by name"


def test_one_window_is_unshifted(monkeypatch) -> None:
    words = run(monkeypatch, [window([("one", 0.0), ("two", 1.5)])])
    assert [t for _, t in words] == [0.0, 1.5]


def test_the_SECOND_window_is_shifted_by_a_whole_window(monkeypatch) -> None:
    """The case the 15-second probe could not reach.

    Unshifted, every word here would land 900 s early and the VTT would still
    parse — a whole book silently wrong from the fifteen-minute mark.
    """
    words = run(monkeypatch, [
        window([("a", 0.0)]),
        window([("b", 0.0), ("c", 10.0)]),
    ])
    assert [t for _, t in words] == [0.0, float(WINDOW_SECONDS), float(WINDOW_SECONDS) + 10.0]


def test_the_shift_is_index_times_window_not_a_running_sum(monkeypatch) -> None:
    """Five windows, so an off-by-one compounds rather than cancelling."""
    words = run(monkeypatch, [window([(f"w{i}", 0.0)]) for i in range(5)])
    assert [t for _, t in words] == [float(i * WINDOW_SECONDS) for i in range(5)]


def test_words_come_back_in_book_order(monkeypatch) -> None:
    """The coarse stage walks them forward; out of order it still aligns and is
    wrong, which is the failure nothing downstream is looking for."""
    words = run(monkeypatch, [
        window([("a", 1.0), ("b", 2.0)]),
        window([("c", 1.0), ("d", 2.0)]),
    ])
    times = [t for _, t in words]
    assert times == sorted(times)


def test_a_FAILED_window_fails_the_job_rather_than_leaving_a_hole(monkeypatch) -> None:
    """A wordless stretch reads to the coarse stage as text the narrator never
    spoke, and those sentences are DROPPED from the VTT rather than placed. So a
    hole is not a transcript, and the job says so instead of continuing."""
    with pytest.raises(stages.StageFailed) as caught:
        run(monkeypatch, [window([("a", 0.0)]), {"error": "cuda oom"}])
    assert caught.value.code == "transcribe_window_failed"
    assert "window 1" in str(caught.value)
    assert "dropped from the VTT" in str(caught.value)


def test_a_window_with_no_words_is_not_an_error(monkeypatch) -> None:
    """Silence is a real answer — a music bridge, a gap between chapters. It is
    only a FAILED window that is fatal."""
    words = run(monkeypatch, [window([("a", 0.0)]), {"segments": []}, window([("b", 0.0)])])
    assert [t for _, t in words] == [0.0, 2.0 * WINDOW_SECONDS]
