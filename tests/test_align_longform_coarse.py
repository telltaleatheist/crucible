"""`coarse-align`, and the four books that put each rule in it.

This is a PORT of bookforge's `coarse_align`, so the tests that matter are the
ones pinning behaviour that came from a specific failure. A port that drops one
of these still aligns, still produces cues, and is wrong in a way nothing
downstream is looking for — which is exactly how each of them was found.

  * the 10k-sentence run that flatlined at an 8.6 % match rate  -> LIS anchoring
  * a GraphicAudio recap measuring 212 tok/s (85x reality)      -> rate clamp,
                                                                  adjacent-only pairs
  * Well of Ascension's ~90 unspoken front-matter sentences     -> the fit tests
  * a 16 s music bridge putting cues ~10 s late                 -> word-weighted
                                                                  interpolation
"""

from __future__ import annotations

from crucible.jobs.alignlongform import coarse


def stream(text: str, start: float = 0.0, step: float = 0.5):
    """A rough transcript: one word every `step` seconds, as `asr` returns."""
    return [(coarse._norm(w), start + i * step) for i, w in enumerate(text.split())]


BOOK = (
    "It was the best of times it was the worst of times. "
    "We had everything before us we had nothing before us. "
    "The year was seventeen seventy five and the age was one of wisdom."
)


def sentences_of(book: str) -> list[str]:
    return [s.strip() + "." for s in book.split(". ") if s.strip()]


# ── The ordinary case ───────────────────────────────────────────────────────


def test_sentences_spoken_in_order_get_real_times_and_are_marked_direct() -> None:
    sents = sentences_of(BOOK)
    words = stream(BOOK)
    out = coarse.coarse_align(sents, words)
    assert len(out.rough) == len(sents)
    assert all(t is not None for t in out.rough), out.rough
    # Every one was found on its own opening, so each is audio truth rather than
    # an interpolation — which is what lets the align stage trust it later.
    assert all(out.direct)
    assert out.rough == sorted(out.rough)


def test_the_result_is_monotonic_even_when_inputs_conspire() -> None:
    sents = sentences_of(BOOK)
    out = coarse.coarse_align(sents, stream(BOOK))
    times = [t for t in out.rough if t is not None]
    assert times == sorted(times), "a later sentence may never start before an earlier one"


def test_no_match_at_all_is_a_real_answer_and_not_a_crash() -> None:
    out = coarse.coarse_align(sentences_of(BOOK), stream("completely unrelated audio here"))
    assert all(t is None for t in out.rough)
    assert out.rate == coarse.DEFAULT_RATE


# ── The rate, and the recap that poisoned it ────────────────────────────────


def test_an_implausible_rate_is_clamped_rather_than_trusted() -> None:
    """The GraphicAudio recap measured 212 tok/s — 85x reality.

    Left alone it made the time fit test roomy enough to let a never-narrated
    39-sentence run smear itself over a music bridge.
    """
    # Sentences far apart in the book but seconds apart in audio.
    matched = [0, 40]
    rough: list[float | None] = [None] * 41
    rough[0], rough[40] = 0.0, 1.0
    tk = [["word"] * 20 for _ in range(41)]
    rate = coarse._rate(matched, rough, tk)
    assert coarse.RATE_MIN <= rate <= coarse.RATE_MAX


def test_only_pairs_ADJACENT_in_sentence_space_may_set_the_rate() -> None:
    """`b - a <= 3` is the guard, and it is why the recap cannot poison it.

    A pair 40 sentences apart contributes nothing, so with no adjacent pair to
    measure the rate falls back to the default rather than to 800 tok/s.
    """
    rough: list[float | None] = [None] * 41
    rough[0], rough[40] = 0.0, 1.0
    tk = [["word"] * 20 for _ in range(41)]
    assert coarse._rate([0, 40], rough, tk) == coarse.DEFAULT_RATE


def test_a_plausible_measured_rate_is_kept() -> None:
    rough: list[float | None] = [0.0, 10.0]
    tk = [["w"] * 30, ["w"] * 30]
    rate = coarse._rate([0, 1], rough, tk)
    assert rate == 3.0  # 30 tokens over 10 s, inside the band, used as measured


# ── Unnarrated text, and Well of Ascension ──────────────────────────────────


def test_front_matter_the_narrator_skipped_stays_None_instead_of_smearing() -> None:
    """~90 unspoken sentences dragged chapter 1's cues ~85 s late.

    The audio goes straight from one spoken line to the next; between them sits
    a pile of text nobody read aloud. It must come back None — excluded from the
    VTT — rather than interpolated across real narration.
    """
    spoken_a = "It was the best of times it was the worst of times."
    spoken_b = "The year was seventeen seventy five and the age was wisdom."
    unspoken = [
        "All rights reserved under international copyright conventions here."
        for _ in range(12)
    ]
    sents = [spoken_a, *unspoken, spoken_b]
    # The audio contains ONLY the two spoken sentences, back to back.
    words = stream(spoken_a + " " + spoken_b)
    out = coarse.coarse_align(sents, words)

    assert out.rough[0] is not None, "the first spoken sentence anchors"
    assert out.rough[-1] is not None, "so does the last"
    interior = out.rough[1:-1]
    assert all(t is None for t in interior), (
        "unspoken front matter was given times; it will be smeared over real "
        "audio and drag every later cue late"
    )
    assert out.dropped == len(unspoken)


def test_a_short_gap_is_treated_as_a_transcription_MISS_and_interpolated() -> None:
    """The other half of the same rule: a small run IS narrated, just misheard.

    `run_tok >= 12` gates the drop, so a short run is filled rather than dropped
    — otherwise every ASR hiccup would punch a hole in the VTT.
    """
    a = "It was the best of times it was the worst of times."
    middle = "A short line."
    b = "The year was seventeen seventy five and the age was wisdom."
    words = stream(a + " " + middle + " " + b)
    out = coarse.coarse_align([a, middle, b], words)
    assert out.rough[1] is not None, "a short interior run must be filled, not dropped"
    assert out.dropped == 0


# ── The tokeniser the whole thing rests on ──────────────────────────────────


def test_normalisation_folds_case_accents_and_punctuation() -> None:
    assert coarse._norm("Café,") == "cafe"
    assert coarse._norm("DON'T") == "dont"
    assert coarse._norm("—") == ""
    assert coarse.toks("The  café's  door!") == ["the", "cafes", "door"]
