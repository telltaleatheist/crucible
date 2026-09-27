from __future__ import annotations

import pytest

from crucible.jobs.alignlongform import plan as P

CHUNK_S = 240.0


def test_sentences_are_grouped_into_windows_of_about_chunk_s() -> None:
    rough = [0.0, 100.0, 200.0, 300.0, 400.0]
    out = P.plan_chunks(rough, 0, 5, duration=420.0, chunk_s=250.0)
    assert [c.sentences for c in out.chunks] == [[0, 1, 2], [3, 4]]


def test_a_window_is_padded_at_head_and_tail() -> None:
    rough = [100.0, 400.0]
    out = P.plan_chunks(rough, 0, 2, duration=500.0, chunk_s=CHUNK_S)
    assert len(out.chunks) == 2
    first = out.chunks[0]
    assert first.start == pytest.approx(100.0 - P.PAD_HEAD)
    assert first.end == pytest.approx(400.0 + P.PAD_TAIL)
    assert (P.PAD_HEAD, P.PAD_TAIL) == (4.0, 20.0), (
        "the pads are PORTED constants, not tuning knobs. This file once carried "
        "invented values of 0.30/0.60, which would have cut every window short "
        "of the audio its last sentence needs."
    )


def test_padding_never_runs_past_the_audio() -> None:
    out = P.plan_chunks([0.1], 0, 1, duration=5.0, chunk_s=CHUNK_S)
    assert out.chunks[0].start == 0.0, "no chunk may start before the file does"
    assert out.chunks[0].end == 5.0, "nor end after it"


def test_the_last_window_runs_to_the_end_of_the_file() -> None:
    out = P.plan_chunks([10.0, 20.0], 0, 2, duration=95.0, chunk_s=CHUNK_S)
    assert out.chunks[-1].end == pytest.approx(95.0)


def test_unnarrated_sentences_are_left_OUT_of_every_window() -> None:
    rough = [0.0, None, None, 300.0]
    out = P.plan_chunks(rough, 0, 4, duration=360.0, chunk_s=500.0)
    placed = [i for c in out.chunks for i in c.sentences]
    assert placed == [0, 3]


def test_a_runaway_span_is_capped_because_aligner_memory_is_quadratic() -> None:
    out = P.plan_chunks([0.0, 10_000.0], 0, 2, duration=20_000.0, chunk_s=30.0)
    assert out.capped >= 1
    for chunk in out.chunks:
        assert chunk.span <= 2 * 30.0 + 1e-9, "a memory-bomb chunk escaped the cap"


def test_the_capped_warning_names_AUDIO_TIMES_not_chunk_indexes() -> None:
    out = P.plan_chunks([0.0, 10_000.0], 0, 2, duration=20_000.0, chunk_s=30.0)
    message = P.capped_warning(out, 30.0)
    assert message is not None
    assert "00:00:00.000" in message
    assert "coarse alignment is likely off" in message


def test_no_warning_when_nothing_was_capped() -> None:
    out = P.plan_chunks([0.0, 50.0], 0, 2, duration=120.0, chunk_s=CHUNK_S)
    assert out.capped == 0
    assert P.capped_warning(out, CHUNK_S) is None


def test_a_book_with_nothing_narrated_plans_no_chunks() -> None:
    out = P.plan_chunks([None, None], 0, 2, duration=600.0, chunk_s=CHUNK_S)
    assert out.chunks == []


def test_every_narrated_sentence_appears_exactly_once_in_order() -> None:
    rough = [float(i * 30) for i in range(40)]
    out = P.plan_chunks(rough, 0, 40, duration=40 * 30 + 30.0, chunk_s=CHUNK_S)
    assert out.chunks, "a fully narrated book must produce chunks"
    seen: list[int] = []
    for chunk in out.chunks:
        assert chunk.sentences, "an empty chunk would be an aligner call for nothing"
        assert chunk.end > chunk.start
        seen += chunk.sentences
    assert seen == list(range(40))
