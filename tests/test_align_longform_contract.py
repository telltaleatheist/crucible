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


def test_a_minimal_request_is_accepted_and_the_defaults_are_stated() -> None:
    parsed = alf.validate(params())
    assert parsed.language == "en"
    assert parsed.language_name == "English"
    assert [s.index for s in parsed.sentences] == [0, 1, 2]
    assert parsed.rough_model == "small"
    assert parsed.chunk_s == 240.0
    assert parsed.silence_source == "decoded"


def test_a_sentence_carries_its_kind_and_is_never_asked_to_infer_one() -> None:
    parsed = alf.validate(
        params(sentences=[{"index": 0, "text": "Chapter One", "kind": "heading"}])
    )
    assert parsed.sentences[0].kind == "heading"
    assert alf.validate(params()).sentences[0].kind == "prose"


def test_the_stage_names_are_the_contract_the_app_draws_bars_from() -> None:
    assert alf.STAGES == ("transcribe", "coarse-align", "align", "write")
    assert alf.ARTIFACTS == ("alignment.vtt", "align-report.json")
    assert alf.JOB_TYPE_NAME == "align-longform"
    assert alf.ALIGNER_MODEL == "qwen3-aligner"


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
    assert alf.QWEN3_MAX_AUDIO_S == 300.0


def test_a_window_at_the_ceiling_exactly_is_allowed() -> None:
    assert alf.validate(params(chunk_s=300)).chunk_s == 300.0


def test_an_unknown_silence_source_is_refused_rather_than_ignored() -> None:
    with pytest.raises(ValueError) as caught:
        alf.validate(params(silence_source="from-epub"))
    assert "under a new label" in str(caught.value)


def test_an_unknown_field_is_refused_rather_than_silently_dropped() -> None:
    with pytest.raises(ValueError):
        alf.validate(params(gate_shift_s=0.2))


def test_the_type_IS_registered_now_that_the_worker_exists() -> None:
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
    import inspect

    from crucible.jobs.alignlongform import jobtype

    source = inspect.getsource(jobtype.AlignLongformJobType.check)
    assert '"asr"' in source and '"align"' in source, (
        "check() no longer asks about both envs, so this job type can report ready "
        "on a host where its first stage cannot run."
    )
