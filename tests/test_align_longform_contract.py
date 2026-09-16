"""`align-longform`'s contract, and the refusals that must happen BEFORE the pool.

§B7, ruled 2026-09-15. This is the whole-audiobook align: the client sends the
book's sentences and the m4b, and the server discovers which seconds hold which
sentence — `transcribe` (CPU) → `coarse-align` (CPU) → `align` (card) → `write`.

The two CPU stages are most of the wall clock, and they run on the server by the
same ruling that sends a whole TTS step there: *"That includes anything the step
needs to do even if it's cpu."*

WHAT THIS SUITE IS FOR. Every check below is a refusal that, if it did not
happen here, would surface as a book's worth of bad cues an hour later — or
worse, as a VTT that looks fine and is wrong. A mis-aligned book reports success.

It also pins that the type is NOT yet queueable. The worker behind `run()` does
not exist, and a job type a client can select but not complete is the
"offered then refused" shape this codebase is built to avoid.
"""

from __future__ import annotations

import pytest

from crucible.jobs import alignlongform as alf


def sentences(n: int = 3, start: int = 0) -> list[dict]:
    return [
        {"index": start + i, "text": f"Sentence number {start + i}."}
        for i in range(n)
    ]


def params(**over) -> dict:
    base = {"language": "en", "sentences": sentences()}
    base.update(over)
    return base


# ── The shape of a legal request ────────────────────────────────────────────


def test_a_minimal_request_is_accepted_and_the_defaults_are_stated() -> None:
    parsed = alf.validate(params())
    assert parsed.language == "en"
    assert parsed.language_name == "English"
    assert [s.index for s in parsed.sentences] == [0, 1, 2]
    # Defaults exist so a caller need not restate the script's own arguments,
    # but each is a real number rather than "whatever the code did".
    assert parsed.rough_model == "small"
    assert parsed.chunk_s == 240.0
    assert parsed.silence_source == "decoded"


def test_a_sentence_carries_its_kind_and_is_never_asked_to_infer_one() -> None:
    parsed = alf.validate(
        params(sentences=[{"index": 0, "text": "Chapter One", "kind": "heading"}])
    )
    assert parsed.sentences[0].kind == "heading"
    # The default is prose, not "unknown": a row that does not say is ordinary text.
    assert alf.validate(params()).sentences[0].kind == "prose"


def test_the_stage_names_are_the_contract_the_app_draws_bars_from() -> None:
    # BookForge's generate-sentences row matches these exact strings to fill its
    # stacked bars. Renaming one silently blanks a bar, so they are pinned.
    assert alf.STAGES == ("transcribe", "coarse-align", "align", "write")
    assert alf.ARTIFACTS == ("alignment.vtt", "align-report.json")
    assert alf.JOB_TYPE_NAME == "align-longform"
    assert alf.ALIGNER_MODEL == "qwen3-aligner"


# ── The refusals, each naming what it refuses ───────────────────────────────


def test_an_untrained_language_is_refused_before_the_pool() -> None:
    with pytest.raises(ValueError) as caught:
        alf.validate(params(language="sv"))
    message = str(caught.value)
    assert "does not fall back to English" in message
    assert "before the pool" in message


def test_no_sentences_is_refused_rather_than_producing_an_empty_vtt() -> None:
    with pytest.raises(ValueError) as caught:
        alf.validate(params(sentences=[]))
    assert "nothing to place" in str(caught.value)


def test_a_blank_sentence_is_refused_because_it_shifts_every_later_cue() -> None:
    with pytest.raises(ValueError) as caught:
        alf.validate(params(sentences=[{"index": 0, "text": "   "}]))
    assert "shift every cue after it" in str(caught.value)


def test_a_duplicate_index_is_refused_because_the_vtt_is_keyed_by_it() -> None:
    with pytest.raises(ValueError) as caught:
        alf.validate(
            params(sentences=[
                {"index": 0, "text": "one"},
                {"index": 0, "text": "two"},
            ])
        )
    assert "more than once" in str(caught.value)


def test_out_of_order_sentences_are_refused_and_the_reason_is_the_dangerous_one() -> None:
    """Order is not tidiness — it is the premise of the whole coarse stage.

    Shuffled input still aligns and still produces cues. Nothing downstream is
    looking for it, so this is the last place it can be caught.
    """
    with pytest.raises(ValueError) as caught:
        alf.validate(
            params(sentences=[
                {"index": 0, "text": "one"},
                {"index": 5, "text": "six"},
                {"index": 2, "text": "three"},
            ])
        )
    message = str(caught.value)
    assert "reading order" in message
    assert "they are wrong" in message


def test_a_window_past_the_aligner_ceiling_is_refused_not_split() -> None:
    with pytest.raises(ValueError) as caught:
        alf.validate(params(chunk_s=301))
    message = str(caught.value)
    assert "300" in message
    assert "refused rather than split" in message
    # And the ceiling itself is the same fact the `align` job enforces.
    assert alf.QWEN3_MAX_AUDIO_S == 300.0


def test_a_window_at_the_ceiling_exactly_is_allowed() -> None:
    assert alf.validate(params(chunk_s=300)).chunk_s == 300.0


def test_an_unknown_silence_source_is_refused_rather_than_ignored() -> None:
    with pytest.raises(ValueError) as caught:
        alf.validate(params(silence_source="from-epub"))
    # The consequence is what makes this worth refusing: edges placed with no
    # silence map are the old build wearing a new label.
    assert "under a new label" in str(caught.value)


def test_an_unknown_field_is_refused_rather_than_silently_dropped() -> None:
    with pytest.raises(ValueError):
        alf.validate(params(gate_shift_s=0.2))


# ── The door is shut until there is something behind it ─────────────────────


def test_the_type_IS_registered_now_that_the_worker_exists() -> None:
    """The inverse of what this asserted while the orchestrator was unbuilt.

    It used to pin the ABSENCE — a job a client can queue and cannot finish is
    worse than one that is missing — and it was written to fail the day
    registration arrived without a worker. The worker arrived (jobtype.py), so
    the assertion turns over rather than being deleted: what it guards now is
    that the type is reachable at all.
    """
    from crucible.jobs import ALL_JOB_TYPES

    assert "align" in ALL_JOB_TYPES, (
        "ALL_JOB_TYPES no longer looks like the registry this test reads. Re-read "
        "jobs/__init__.py before trusting the assertion below."
    )
    assert ALL_JOB_TYPES.get(alf.JOB_TYPE_NAME) == "align", (
        "align-longform must share the `align` flag: it drives that job type's "
        "worker, and a host with the aligner off cannot run it."
    )


def test_it_needs_BOTH_envs_and_says_which_is_missing() -> None:
    """`check()` must not report ready on the aligner alone.

    The rough pass runs in the `asr` env, so a host with `align` installed and
    `asr` absent can queue this job and fail in its first stage — the exact shape
    the contract's refusals exist to prevent, arriving through the health probe
    instead of the door.
    """
    import inspect

    from crucible.jobs.alignlongform import jobtype

    source = inspect.getsource(jobtype.AlignLongformJobType.check)
    assert '"asr"' in source and '"align"' in source, (
        "check() no longer asks about both envs, so this job type can report ready "
        "on a host where its first stage cannot run."
    )
