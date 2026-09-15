"""Chunk planning: the glue between `coarse-align` and the resident aligner.

Each chunk becomes one `{audio, text}` entry on the `align` worker's wire. Three
properties carry real weight:

  * a sentence the narrator never read (`rough is None`) must never end up inside
    a window — unspoken words in a chunk drag the alignment, which is the Well of
    Ascension failure arriving one stage later;
  * no window may exceed `2 * chunk_s`, because wav2vec2 memory is QUADRATIC in
    span, so one bad coarse result is a memory bomb rather than a bad chunk;
  * every narrated sentence appears exactly once, in reading order.

TWO BEHAVIOURS THESE TESTS HAD TO LEARN, both of them the original's and both
easy to write a wrong expectation against:

  1. the pads are 4 s head and 20 s TAIL, so a toy `chunk_s` of 5 caps every
     window before padding can even be observed. Real books use 240;
  2. the LAST chunk always runs to the end of the file, so a short book inside a
     long file caps its final window by construction. That is the original's
     behaviour (`b = DUR`), not a defect — the tail of a book is whatever audio
     remains.
"""

from __future__ import annotations

import pytest

from crucible.jobs.alignlongform import plan as P

#: The job's own default, and far enough clear of PAD_TAIL to be realistic.
CHUNK_S = 240.0


def test_sentences_are_grouped_into_windows_of_about_chunk_s() -> None:
    # Sentences every 100 s; a 250 s window takes three, then the rest.
    rough = [0.0, 100.0, 200.0, 300.0, 400.0]
    out = P.plan_chunks(rough, 0, 5, duration=420.0, chunk_s=250.0)
    assert [c.sentences for c in out.chunks] == [[0, 1, 2], [3, 4]]


def test_a_window_is_padded_at_head_and_tail() -> None:
    # TWO chunks are needed to observe PAD_TAIL at all: the last window always
    # runs to the end of the file, so its tail is the duration rather than a pad.
    # Sentences 300 s apart with a 240 s window split into two.
    rough = [100.0, 400.0]
    out = P.plan_chunks(rough, 0, 2, duration=500.0, chunk_s=CHUNK_S)
    assert len(out.chunks) == 2
    first = out.chunks[0]
    assert first.start == pytest.approx(100.0 - P.PAD_HEAD)
    # The tail reaches the NEXT sentence's rough start plus the tail pad.
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
    """The original's own behaviour: the tail of a book is whatever remains."""
    out = P.plan_chunks([10.0, 20.0], 0, 2, duration=95.0, chunk_s=CHUNK_S)
    assert out.chunks[-1].end == pytest.approx(95.0)


def test_unnarrated_sentences_are_left_OUT_of_every_window() -> None:
    """`None` is text nobody read aloud. Putting it in a chunk drags the align."""
    rough = [0.0, None, None, 300.0]
    out = P.plan_chunks(rough, 0, 4, duration=360.0, chunk_s=500.0)
    placed = [i for c in out.chunks for i in c.sentences]
    assert placed == [0, 3]


def test_a_runaway_span_is_capped_because_aligner_memory_is_quadratic() -> None:
    # A coarse regression: two adjacent sentences implausibly far apart.
    out = P.plan_chunks([0.0, 10_000.0], 0, 2, duration=20_000.0, chunk_s=30.0)
    assert out.capped >= 1
    for chunk in out.chunks:
        assert chunk.span <= 2 * 30.0 + 1e-9, "a memory-bomb chunk escaped the cap"


def test_the_capped_warning_names_AUDIO_TIMES_not_chunk_indexes() -> None:
    """A cap means the coarse anchors are wrong somewhere, and the only way to
    check is to go and listen. A chunk index is not a place you can listen to."""
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
