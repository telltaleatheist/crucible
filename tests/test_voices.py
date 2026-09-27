from __future__ import annotations

import os
from pathlib import Path

import pytest

from crucible.voices import (
    VOICES_DIR_ENV,
    CLIPS_FROM_REQUEST,
    NARRATOR_ENGINE_SAMPLING,
    VoiceError,
    load_all_voices,
    load_voice,
    parse_voice,
    voices_dir,
)

from .conftest import configure_box

GOOD = """
[voice]
id = "probe"
display = "Probe"
kind = "checkpoint"
narrator_engine = "higgs-v3"
language = "en"
sample_rate = 24000

[voice.pace]
pace_chars_per_sec = 16.0
max_chars_per_sec = 20.8
min_chars_per_sec = 12.3
safe_min_chars = 600
safe_max_chars = 800

[voice.serving]
max_num_seqs = 16
max_num_seqs_note = "vllm-omni's own stage-0 value, and a measured ceiling at 0.35 + 0.10."

[voice.backends.cuda-linux]
hf_repo = "owenmorgan/probe-higgs-v3"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 19_000_000_000
estimate_basis = "measured"
max_chars = 800
sampling = { temperature = 0.8, top_p = 0.95, top_k = 50 }
"""


@pytest.fixture(autouse=True)
def a_configured_box() -> None:
    configure_box(Path(os.environ["CRUCIBLE_HOME"]))


def parse(text: str, voice_id: str = "probe"):
    return parse_voice(text, Path(f"{voice_id}.toml"), voice_id)


def refused(text: str, voice_id: str = "probe") -> str:
    with pytest.raises(VoiceError) as caught:
        parse(text, voice_id)
    return str(caught.value)


def swap(old: str, new: str) -> str:
    assert old in GOOD, f"the good manifest does not contain {old!r}"
    return GOOD.replace(old, new)


SERVING_TABLE = GOOD[GOOD.index("[voice.serving]"):GOOD.index("[voice.backends")]


def test_the_good_manifest_loads() -> None:
    voice = parse(GOOD)
    assert voice.id == "probe"
    assert voice.kind == "checkpoint"
    assert voice.narrator_engine == "higgs-v3"
    assert voice.supports("cuda-linux")
    assert not voice.supports("mlx-darwin")
    assert voice.spec("cuda-linux").max_chars == 800
    assert voice.pace.safe_min_chars == 600
    assert voice.pace.target_chars is None


def test_a_fingerprint_binds_the_id_to_the_revision() -> None:
    voice = parse(GOOD)
    assert voice.fingerprint("cuda-linux") == (
        "probe@0123456789abcdef0123456789abcdef01234567"
    )


def test_a_backend_with_no_block_is_refused_by_name() -> None:
    voice = parse(GOOD)
    with pytest.raises(VoiceError) as caught:
        voice.spec("mlx-darwin")
    assert "has no mlx-darwin block" in str(caught.value)
    assert "['cuda-linux']" in str(caught.value)


def test_an_unknown_top_level_table_is_refused() -> None:
    assert "unknown top-level table(s) ['serving']" in refused(
        GOOD + '\n[serving]\nport = 8095\n'
    )


def test_a_manifest_with_no_voice_table_is_refused() -> None:
    assert "missing the [voice] table" in refused("")


def test_an_unknown_key_in_voice_is_refused() -> None:
    assert "unknown key(s) ['sampel_rate']" in refused(
        swap("sample_rate = 24000", "sample_rate = 24000\nsampel_rate = 22050")
    )


def test_a_missing_required_key_is_refused() -> None:
    assert "missing required key(s) ['sample_rate']" in refused(
        swap("sample_rate = 24000\n", "")
    )


def test_the_id_must_be_the_filename() -> None:
    assert "the id and the filename are the same thing" in refused(
        swap('id = "probe"', 'id = "other"')
    )


def test_an_upper_case_id_is_refused() -> None:
    assert "must be lower-case" in refused(
        swap('id = "probe"', 'id = "Probe"'), "Probe"
    )


def test_an_unknown_kind_is_refused() -> None:
    message = refused(swap('kind = "checkpoint"', 'kind = "adapter"'))
    assert "'adapter' is not a voice kind" in message
    assert "['checkpoint', 'token', 'zeroshot']" in message


def test_an_unknown_narrator_engine_is_refused() -> None:
    message = refused(swap('narrator_engine = "higgs-v3"', 'narrator_engine = "xtts"'))
    assert "'xtts' is not one of narrator's engines" in message
    assert "['higgs-v3']" in message


def test_a_zero_sample_rate_is_refused() -> None:
    assert "sample_rate must be positive" in refused(
        swap("sample_rate = 24000", "sample_rate = 0")
    )


def test_a_voice_may_omit_the_pace_table_entirely() -> None:
    voice = parse(
        GOOD[: GOOD.index("[voice.pace]")]
        + GOOD[GOOD.index("[voice.serving]"):]
    )
    assert voice.pace.pace_chars_per_sec is None
    assert voice.pace.max_chars_per_sec is None
    assert voice.pace.min_chars_per_sec is None
    assert voice.pace.safe_min_chars is None
    assert voice.pace.safe_max_chars is None
    assert voice.pace.target_chars is None


def test_a_pace_that_is_not_a_table_is_still_refused() -> None:
    without_pace = (
        GOOD[: GOOD.index("[voice.pace]")] + GOOD[GOOD.index("[voice.serving]"):]
    )
    assert "[voice.pace] must be a table" in refused(
        without_pace.replace("sample_rate = 24000", "sample_rate = 24000\npace = 16.0")
    )


def test_half_a_band_is_refused() -> None:
    message = refused(swap("min_chars_per_sec = 12.3\n", ""))
    assert "declares only part of its rate band" in message
    assert "['min_chars_per_sec']" in message


def test_a_voice_may_state_no_rates_at_all() -> None:
    voice = parse(
        swap(
            "pace_chars_per_sec = 16.0\nmax_chars_per_sec = 20.8\n"
            "min_chars_per_sec = 12.3\n",
            "",
        )
    )
    assert voice.pace.pace_chars_per_sec is None
    assert voice.pace.max_chars_per_sec is None
    assert voice.pace.min_chars_per_sec is None
    assert voice.pace.safe_min_chars == 600
    assert voice.pace.safe_max_chars == 800


def test_a_pace_outside_its_own_edges_is_refused() -> None:
    message = refused(swap("pace_chars_per_sec = 16.0", "pace_chars_per_sec = 24.0"))
    assert "out of order" in message
    assert "the band is min < pace < max" in message


SPLICED = swap(
    "pace_chars_per_sec = 16.0\nmax_chars_per_sec = 20.8\nmin_chars_per_sec = 12.3",
    "pace_chars_per_sec = 15.0\nmax_chars_per_sec = 20.0\nmin_chars_per_sec = 14.5",
)


def test_a_lopsided_triple_is_refused_naming_both_ratios() -> None:
    message = refused(SPLICED)
    assert "1.333" in message
    assert "1.034" in message
    assert "edges" in message


def test_a_percentile_band_may_be_lopsided() -> None:
    voice = parse(
        SPLICED.replace(
            "min_chars_per_sec = 14.5",
            'min_chars_per_sec = 14.5\nedges = "percentile"',
        )
    )
    assert voice.pace.pace_chars_per_sec == 15.0
    assert voice.pace.max_chars_per_sec == 20.0
    assert voice.pace.min_chars_per_sec == 14.5


def test_an_unknown_edges_word_is_refused() -> None:
    message = refused(
        SPLICED.replace(
            "min_chars_per_sec = 14.5",
            'min_chars_per_sec = 14.5\nedges = "percentiles"',
        )
    )
    assert "edges" in message
    assert "'percentiles'" in message


def test_edges_without_a_band_is_refused() -> None:
    message = refused(
        swap(
            "pace_chars_per_sec = 16.0\nmax_chars_per_sec = 20.8\n"
            "min_chars_per_sec = 12.3\n",
            'edges = "percentile"\n',
        )
    )
    assert "edges" in message
    assert "states no rate band" in message


def test_a_negative_rate_is_refused() -> None:
    assert "min_chars_per_sec must be positive" in refused(
        swap("min_chars_per_sec = 12.3", "min_chars_per_sec = -1.0")
    )


def test_a_boolean_where_a_number_belongs_is_refused() -> None:
    assert "pace_chars_per_sec must be a number, got bool" in refused(
        swap("pace_chars_per_sec = 16.0", "pace_chars_per_sec = true")
    )


def test_declaring_both_a_target_and_a_band_is_refused() -> None:
    message = refused(
        swap("safe_min_chars = 600", "target_chars = 700\nsafe_min_chars = 600")
    )
    assert "declares both target_chars and a safe band" in message


def test_half_a_safe_band_is_refused() -> None:
    assert "a safe band needs both edges and is missing safe_max_chars" in refused(
        swap("safe_max_chars = 800\n", "")
    )


def test_a_floor_at_the_ceiling_is_refused() -> None:
    message = refused(swap("safe_min_chars = 600", "safe_min_chars = 800"))
    assert "is not below safe_max_chars" in message


def test_a_voice_declaring_neither_packs_to_the_backend_cap() -> None:
    voice = parse(swap("safe_min_chars = 600\nsafe_max_chars = 800\n", ""))
    assert voice.pace.target_chars is None
    assert voice.pace.safe_max_chars is None
    assert voice.spec("cuda-linux").max_chars == 800


def test_a_band_above_the_backend_cap_is_refused() -> None:
    message = refused(swap("safe_max_chars = 800", "safe_max_chars = 900"))
    assert "caps the voice at 800 characters" in message
    assert "safe_max_chars 900" in message


def test_a_target_above_the_backend_cap_is_refused() -> None:
    message = refused(
        swap(
            "safe_min_chars = 600\nsafe_max_chars = 800",
            "target_chars = 1200",
        )
    )
    assert "packs to target_chars 1200" in message


def test_a_voice_with_no_backend_block_is_refused() -> None:
    assert "missing every [voice.backends.<kind>] table" in refused(
        GOOD[: GOOD.index("[voice.backends")]
    )


def test_an_empty_backends_table_is_refused() -> None:
    assert "no backend blocks" in refused(
        GOOD[: GOOD.index("[voice.backends")] + "[voice.backends]"
    )


def test_an_unknown_backend_is_refused() -> None:
    message = refused(
        swap("[voice.backends.cuda-linux]", "[voice.backends.rocm-linux]")
    )
    assert "'rocm-linux' is not a Crucible backend" in message


def test_a_branch_name_is_not_a_pin() -> None:
    message = refused(
        swap(
            'revision = "0123456789abcdef0123456789abcdef01234567"',
            'revision = "main"',
        )
    )
    assert "must be a full 40-character commit sha" in message


def test_a_bare_repo_name_is_refused() -> None:
    assert "is not an <owner>/<name>" in refused(
        swap('hf_repo = "owenmorgan/probe-higgs-v3"', 'hf_repo = "probe"')
    )


def test_a_zero_estimate_is_refused() -> None:
    assert "memory_bytes_estimate must be positive" in refused(
        swap("memory_bytes_estimate = 19_000_000_000", "memory_bytes_estimate = 0")
    )


def test_a_zero_cap_is_refused() -> None:
    assert "max_chars must be positive" in refused(
        swap("safe_min_chars = 600\nsafe_max_chars = 800\n", "").replace(
            "max_chars = 800", "max_chars = 0"
        )
    )


def test_a_backend_may_state_no_cap_at_all() -> None:
    voice = parse(
        swap("safe_min_chars = 600\nsafe_max_chars = 800\n", "").replace(
            "max_chars = 800\n", ""
        )
    )
    assert voice.spec("cuda-linux").max_chars is None
    assert voice.spec("cuda-linux").to_dict()["max_chars"] is None


def test_a_packing_band_above_a_cap_is_still_refused_when_a_cap_is_stated() -> None:
    assert "The band may never exceed the arm's cap" in refused(
        swap("safe_max_chars = 800", "safe_max_chars = 900")
    )


def test_an_unknown_estimate_basis_is_refused() -> None:
    message = refused(swap('estimate_basis = "measured"', 'estimate_basis = "guessed"'))
    assert "estimate_basis 'guessed' is not one of ['declared', 'measured']" in message


def test_a_declared_estimate_needs_a_note() -> None:
    message = refused(swap('estimate_basis = "measured"', 'estimate_basis = "declared"'))
    assert "estimate_basis is 'declared' and there is no estimate_note" in message


def test_a_declared_estimate_with_a_note_is_accepted() -> None:
    voice = parse(
        swap(
            'estimate_basis = "measured"',
            'estimate_basis = "declared"\nestimate_note = "SGLang reserves 0.6"',
        )
    )
    spec = voice.spec("cuda-linux")
    assert spec.estimate_basis == "declared"
    assert spec.estimate_note == "SGLang reserves 0.6"


def test_a_measured_estimate_may_not_carry_a_note() -> None:
    message = refused(
        swap(
            'estimate_basis = "measured"',
            'estimate_basis = "measured"\nestimate_note = "about right"',
        )
    )
    assert "estimate_basis is 'measured' and it also carries an estimate_note" in message


def test_the_boson_default_needs_no_reason() -> None:
    assert parse(GOOD).spec("cuda-linux").sampling_reason is None


def test_a_deviation_without_a_reason_is_refused() -> None:
    message = refused(
        swap("temperature = 0.8, top_p = 0.95", "temperature = 0.7, top_p = 0.95")
    )
    assert "deviates from the higgs-v3 default" in message
    assert "temperature 0.7 (engine default 0.8)" in message
    assert "carries no sampling_reason" in message


def test_a_deviation_with_a_reason_is_accepted() -> None:
    voice = parse(
        swap(
            "sampling = { temperature = 0.8, top_p = 0.95, top_k = 50 }",
            "sampling = { temperature = 0.7, top_p = 0.95, top_k = 50 }\n"
            'sampling_reason = "measured 2026-09-11 over the same 88 chunks"',
        )
    )
    spec = voice.spec("cuda-linux")
    assert spec.sampling["temperature"] == 0.7
    assert spec.sampling_reason.startswith("measured")


def test_an_empty_reason_does_not_count_as_one() -> None:
    message = refused(
        swap(
            "sampling = { temperature = 0.8, top_p = 0.95, top_k = 50 }",
            "sampling = { temperature = 0.7, top_p = 0.95, top_k = 50 }\n"
            'sampling_reason = "   "',
        )
    )
    assert "carries no sampling_reason" in message


def test_a_reason_with_nothing_to_explain_is_refused() -> None:
    message = refused(
        swap(
            "sampling = { temperature = 0.8, top_p = 0.95, top_k = 50 }",
            "sampling = { temperature = 0.8, top_p = 0.95, top_k = 50 }\n"
            'sampling_reason = "because"',
        )
    )
    assert "carries a sampling_reason but its sampling is the higgs-v3 default" in message


def test_a_partial_sampling_block_is_refused() -> None:
    message = refused(
        swap(
            "sampling = { temperature = 0.8, top_p = 0.95, top_k = 50 }",
            "sampling = { temperature = 0.8 }",
        )
    )
    assert "missing required key(s) ['top_k', 'top_p']" in message


def test_the_serving_width_is_read() -> None:
    voice = parse(GOOD)
    assert voice.serving is not None
    assert voice.serving.max_num_seqs == 16
    assert "stage-0" in voice.serving.max_num_seqs_note
    assert voice.to_dict()["serving"]["max_num_seqs"] == 16


def test_a_higgs_voice_without_a_serving_table_is_refused() -> None:
    message = refused(GOOD.replace(SERVING_TABLE, ""))
    assert "[voice.serving]" in message
    assert "max_num_seqs" in message


def test_a_serving_width_below_one_is_refused() -> None:
    assert "at least 1" in refused(swap("max_num_seqs = 16", "max_num_seqs = 0"))


def test_a_serving_width_with_no_note_is_refused() -> None:
    quoted = GOOD[GOOD.index("max_num_seqs_note = "):].splitlines()[0]
    assert "carries no note" in refused(
        GOOD.replace(quoted, 'max_num_seqs_note = "   "')
    )


def test_an_unknown_serving_key_is_refused() -> None:
    assert "unknown key(s) ['stack']" in refused(
        swap(
            "max_num_seqs = 16",
            'stack = "vllm-omni"\nmax_num_seqs = 16',
        )
    )


SCREENING_SERVING = """mem_fraction = 0.48
mem_fraction_note = "0.48 + width 4 is ~20 GB; 0.55 measured 24.0-24.1 GB on this 24 GB card and WDDM then pages to host RAM"
context_length = 8192
context_length_note = "the Third Reich bank tops at 2,008 chars and 4096 tokens holds ~2,000, so the top rungs truncate on the context"
"""


def test_a_voice_may_state_a_mem_fraction_and_a_context_length() -> None:
    voice = parse(swap("max_num_seqs = 16\n", "max_num_seqs = 16\n" + SCREENING_SERVING))
    assert voice.serving.mem_fraction == 0.48
    assert voice.serving.context_length == 8192
    assert "24 GB card" in voice.serving.mem_fraction_note
    assert "2,008" in voice.serving.context_length_note


def test_a_voice_stating_neither_reports_them_as_null_on_its_row() -> None:
    row = parse(GOOD).serving.to_dict()
    assert row["mem_fraction"] is None and row["mem_fraction_note"] is None
    assert row["context_length"] is None and row["context_length_note"] is None


@pytest.mark.parametrize("key", ["mem_fraction", "context_length"])
def test_a_serving_lever_with_no_note_is_refused(key: str) -> None:
    stated = {"mem_fraction": "0.48", "context_length": "8192"}[key]
    message = refused(
        swap("max_num_seqs = 16\n", f"max_num_seqs = 16\n{key} = {stated}\n")
    )
    assert f"{key} carries no note" in message


@pytest.mark.parametrize("key", ["mem_fraction", "context_length"])
def test_a_serving_note_with_no_number_is_refused_as_the_leftover_it_is(
    key: str,
) -> None:
    message = refused(
        swap("max_num_seqs = 16\n", f'max_num_seqs = 16\n{key}_note = "x"\n')
    )
    assert f"states {key}_note and no {key}" in message


def test_a_mem_fraction_outside_zero_to_one_is_refused() -> None:
    message = refused(
        swap(
            "max_num_seqs = 16\n",
            'max_num_seqs = 16\nmem_fraction = 1.2\nmem_fraction_note = "x"\n',
        )
    )
    assert "must be a fraction in (0, 1)" in message


def test_a_zero_context_length_is_refused() -> None:
    message = refused(
        swap(
            "max_num_seqs = 16\n",
            'max_num_seqs = 16\ncontext_length = 0\ncontext_length_note = "x"\n',
        )
    )
    assert "context_length must be positive" in message


def test_the_two_levers_are_not_refused_on_a_voice_with_an_mlx_arm() -> None:
    mlx = GOOD + """
[voice.backends.mlx-darwin]
hf_repo = "owenmorgan/probe-higgs-v3"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 12_133_000_000
estimate_basis = "measured"
max_chars = 800
sampling = { temperature = 0.8, top_p = 0.95, top_k = 50 }
"""
    voice = parse(
        mlx.replace("max_num_seqs = 16\n", "max_num_seqs = 16\n" + SCREENING_SERVING)
    )
    assert voice.serving.context_length == 8192
    assert sorted(voice.backends) == ["cuda-linux", "mlx-darwin"]


def test_applied_sampling_is_the_whole_triple_and_not_the_rungs_override() -> None:
    voice = parse(GOOD + LADDER)
    assert voice.applied_sampling("cuda-linux", 0) == {
        "temperature": 0.8, "top_p": 0.95, "top_k": 50,
    }
    assert voice.applied_sampling("cuda-linux", 1) == {
        "temperature": 0.7, "top_p": 0.95, "top_k": 50,
    }
    assert voice.applied_sampling("cuda-linux", 5) == {
        "temperature": 0.8, "top_p": 0.95, "top_k": 50,
    }


def test_every_shipped_higgs_voice_declares_one() -> None:
    for voice in load_all_voices().values():
        if voice.narrator_engine == "higgs-v3":
            assert voice.serving is not None, voice.id
            assert voice.serving.max_num_seqs >= 1, voice.id
            assert voice.serving.max_num_seqs_note.strip(), voice.id


def zeroshot(clips: str) -> str:
    return swap('kind = "checkpoint"', 'kind = "zeroshot"').replace(
        "sampling = { temperature = 0.8, top_p = 0.95, top_k = 50 }",
        "sampling = { temperature = 0.8, top_p = 0.95, top_k = 50 }\n" + clips,
    )


def test_a_checkpoint_may_not_declare_clips() -> None:
    message = refused(
        swap(
            "sampling = { temperature = 0.8, top_p = 0.95, top_k = 50 }",
            "sampling = { temperature = 0.8, top_p = 0.95, top_k = 50 }\n"
            'clips = "from-request"',
        )
    )
    assert "a checkpoint voice declares clips" in message


def test_a_zeroshot_voice_must_declare_clips() -> None:
    message = refused(swap('kind = "checkpoint"', 'kind = "zeroshot"'))
    assert "must declare its reference clips" in message
    assert "'from-request'" in message


def test_from_request_is_the_only_string_clips_may_be() -> None:
    assert "the only string it may be is 'from-request'" in refused(
        zeroshot('clips = "from-the-disk"')
    )


def test_from_request_is_accepted() -> None:
    voice = parse(zeroshot('clips = "from-request"'))
    spec = voice.spec("cuda-linux")
    assert spec.clips == CLIPS_FROM_REQUEST
    assert spec.clips_from_request is True


def test_a_declared_clip_is_read() -> None:
    voice = parse(
        zeroshot(
            'clips = [{ file = "stranger-01.wav", transcript = "He had been '
            'walking.", seconds = 8.4 }]'
        )
    )
    clips = voice.spec("cuda-linux").clips
    assert len(clips) == 1
    assert clips[0].file == "stranger-01.wav"
    assert clips[0].seconds == 8.4


def test_a_clip_with_no_transcript_is_refused() -> None:
    message = refused(
        zeroshot(
            'clips = [{ file = "stranger-01.wav", transcript = "  ", seconds = 8.4 }]'
        )
    )
    assert "has no transcript" in message
    assert "book-exact text spoken in it" in message


def test_a_clip_with_no_duration_is_refused() -> None:
    message = refused(
        zeroshot(
            'clips = [{ file = "x.wav", transcript = "Hello.", seconds = 0.0 }]'
        )
    )
    assert "seconds must be positive" in message


def test_an_empty_clip_list_is_refused() -> None:
    assert "must be a non-empty list" in refused(zeroshot("clips = []"))


def test_a_voice_with_no_ladder_still_has_take_zero() -> None:
    voice = parse(GOOD)
    assert len(voice.takes) == 1
    assert voice.take(0).overrides == {}
    assert voice.take(0).reason is None


def test_a_take_past_the_end_is_a_seed_lane_at_take_zeros_sampling() -> None:
    rung = parse(GOOD).take(3)
    assert rung.index == 3
    assert rung.overrides == {}
    assert rung.reason is None
    assert len(parse(GOOD).takes) == 1


def test_a_take_past_a_DECLARED_ladder_is_not_the_last_rungs_numbers() -> None:
    rung = parse(GOOD + LADDER).take(4)
    assert rung.overrides == {}
    assert parse(GOOD + LADDER).take(1).overrides == {"temperature": 0.7}


def test_a_negative_take_is_still_refused() -> None:
    with pytest.raises(VoiceError) as caught:
        parse(GOOD).take(-1)
    assert "is below take 0" in str(caught.value)


LADDER = """
[[voice.takes]]

[[voice.takes]]
temperature = 0.7
reason = "measured 2026-09-11 over the same 88 chunks: 0.8 gave 4 guard fires, 0.7 gave 8"
"""


def test_a_ladder_is_read_in_order() -> None:
    voice = parse(GOOD + LADDER)
    assert len(voice.takes) == 2
    assert voice.take(0).overrides == {}
    assert voice.take(1).overrides == {"temperature": 0.7}
    assert voice.take(1).reason.startswith("measured")


def test_take_zero_may_not_deviate() -> None:
    message = refused(GOOD + '\n[[voice.takes]]\ntemperature = 0.7\nreason = "x"\n')
    assert "take 0 is the engine default and may not deviate" in message


def test_a_rung_that_changes_nothing_is_a_different_draw_and_is_allowed() -> None:
    voice = parse(GOOD + "\n[[voice.takes]]\n\n[[voice.takes]]\n")
    assert len(voice.takes) == 2
    assert voice.take(1).overrides == {}
    assert voice.take(1).reason is None


def test_take_zero_still_may_not_deviate_and_a_deviation_still_owes_a_reason() -> None:
    assert "take 0 is the engine default and may not deviate" in refused(
        GOOD + '\n[[voice.takes]]\ntemperature = 0.7\nreason = "x"\n'
    )
    assert "each one owes the measurement that chose it" in refused(
        GOOD + "\n[[voice.takes]]\n\n[[voice.takes]]\ntemperature = 0.7\n"
    )


def test_a_rung_without_a_reason_is_refused() -> None:
    message = refused(GOOD + "\n[[voice.takes]]\n\n[[voice.takes]]\ntemperature = 0.7\n")
    assert "deviates from the higgs-v3 default (temperature 0.7)" in message
    assert "each one owes the measurement that chose it" in message


def test_the_voices_dir_env_is_honoured(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CRUCIBLE_VOICES_DIR", str(tmp_path))
    (tmp_path / "probe.toml").write_text(GOOD, encoding="utf-8")
    assert sorted(load_all_voices()) == ["probe"]
    assert load_voice("probe").display == "Probe"


def test_a_voices_dir_that_is_not_a_directory_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_VOICES_DIR", str(tmp_path / "nowhere"))
    with pytest.raises(VoiceError) as caught:
        voices_dir()
    assert "is not a directory" in str(caught.value)


def test_an_unknown_voice_lists_what_this_build_ships(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_VOICES_DIR", str(tmp_path))
    (tmp_path / "probe.toml").write_text(GOOD, encoding="utf-8")
    with pytest.raises(VoiceError) as caught:
        load_voice("nobody")
    assert "no manifest for voice 'nobody'" in str(caught.value)
    assert "['probe']" in str(caught.value)


def test_voices_are_listed_by_id_and_not_by_path(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_VOICES_DIR", str(tmp_path))
    (tmp_path / "zeroshot.toml").write_text(
        GOOD.replace('id = "probe"', 'id = "zeroshot"'), encoding="utf-8"
    )
    (tmp_path / "zeroshot-x.toml").write_text(
        GOOD.replace('id = "probe"', 'id = "zeroshot-x"'), encoding="utf-8"
    )
    assert list(load_all_voices()) == ["zeroshot", "zeroshot-x"]


def test_bad_toml_is_refused_by_name(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CRUCIBLE_VOICES_DIR", str(tmp_path))
    (tmp_path / "probe.toml").write_text("[voice\n", encoding="utf-8")
    with pytest.raises(VoiceError) as caught:
        load_voice("probe")
    assert "not valid TOML" in str(caught.value)


SHIPPED = {
    "deathstalker": ("checkpoint", 800),
    "mistborn": ("checkpoint", 800),
    "owen": ("checkpoint", 800),
    "thirdreich": ("checkpoint", 1000),
    "sigma": ("checkpoint", 1000),
    "higgs-default": ("token", 600),
    "zeroshot": ("zeroshot", 600),
}


UNMEASURED_PACE = ("higgs-default", "zeroshot")


def test_this_build_ships_the_voices_it_says_it_does() -> None:
    assert sorted(load_all_voices()) == sorted(SHIPPED)


@pytest.mark.parametrize("voice_id", sorted(SHIPPED))
def test_a_manifest_states_the_pace_it_measured_and_no_other(voice_id: str) -> None:
    pace = load_voice(voice_id).pace
    rates = (pace.pace_chars_per_sec, pace.max_chars_per_sec, pace.min_chars_per_sec)
    if voice_id in UNMEASURED_PACE:
        assert rates == (None, None, None)
        if voice_id == "zeroshot":
            assert pace.target_chars == 600
        return
    assert all(rate is not None for rate in rates), voice_id
    assert pace.min_chars_per_sec < pace.pace_chars_per_sec < pace.max_chars_per_sec


@pytest.mark.parametrize("voice_id", sorted(set(SHIPPED) - set(UNMEASURED_PACE)))
def test_every_measured_band_in_this_catalog_is_symmetric(voice_id: str) -> None:
    pace = load_voice(voice_id).pace
    long_side = pace.max_chars_per_sec / pace.pace_chars_per_sec
    short_side = pace.pace_chars_per_sec / pace.min_chars_per_sec
    assert round(long_side, 2) == 1.3, voice_id
    assert round(short_side, 2) == 1.3, voice_id


@pytest.mark.parametrize("voice_id", sorted(SHIPPED))
def test_each_shipped_voice_carries_the_catalog_numbers(voice_id: str) -> None:
    kind, cap = SHIPPED[voice_id]
    voice = load_voice(voice_id)
    assert voice.kind == kind
    assert voice.narrator_engine == "higgs-v3"
    assert voice.sample_rate == 24000
    assert sorted(voice.backends) == ["cuda-linux", "mlx-darwin"]
    for backend in voice.backends:
        assert voice.spec(backend).max_chars == cap


@pytest.mark.parametrize("voice_id", sorted(SHIPPED))
def test_every_shipped_voice_renders_at_the_boson_default(voice_id: str) -> None:
    voice = load_voice(voice_id)
    for backend in voice.backends:
        spec = voice.spec(backend)
        assert spec.sampling == NARRATOR_ENGINE_SAMPLING["higgs-v3"]
        assert spec.sampling_reason is None


@pytest.mark.parametrize("voice_id", sorted(SHIPPED))
def test_no_shipped_voice_claims_a_measured_estimate(voice_id: str) -> None:
    voice = load_voice(voice_id)
    for backend in voice.backends:
        spec = voice.spec(backend)
        assert spec.estimate_basis == "declared"
        assert spec.estimate_note


@pytest.mark.parametrize("voice_id", sorted(SHIPPED))
def test_the_five_fine_tunes_declare_a_second_rung_and_nothing_else_does(
    voice_id: str,
) -> None:
    voice = load_voice(voice_id)
    kind = SHIPPED[voice_id][0]
    if kind != "checkpoint":
        assert len(voice.takes) == 1
        assert voice.take(0).overrides == {}
        return
    assert len(voice.takes) == 2
    assert voice.take(0).overrides == {}
    assert voice.take(0).reason is None
    assert voice.take(1).overrides == {"temperature": 0.7}
    reason = voice.take(1).reason
    assert reason is not None
    assert "measured 2026-09-11" in reason
    assert "88 chunks" in reason
    assert "a DIFFERENT one" in reason


def test_the_zeroshot_voice_takes_its_clips_from_the_request() -> None:
    voice = load_voice("zeroshot")
    for backend in voice.backends:
        assert voice.spec(backend).clips_from_request is True


def test_the_shipped_manifests_are_the_directory_beside_the_package() -> None:
    assert voices_dir() == Path(__file__).resolve().parent.parent / "crucible" / "voices"


def a_voice_file(into: Path, voice_id: str, display: str | None = None) -> Path:
    import re

    into.mkdir(parents=True, exist_ok=True)
    raw = GOOD.replace('id = "probe"', f'id = "{voice_id}"', 1)
    if display is not None:
        raw = re.sub(r'display\s*=\s*"[^"]*"', f'display = "{display}"', raw, count=1)
    path = into / f"{voice_id}.toml"
    path.write_text(raw, encoding="utf-8")
    return path


def test_a_voice_dropped_into_the_home_is_served(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CRUCIBLE_HOME", str(tmp_path))
    monkeypatch.delenv(VOICES_DIR_ENV, raising=False)
    configure_box(tmp_path)
    a_voice_file(tmp_path / "voices", "tonights-finetune")
    assert "tonights-finetune" in load_all_voices()
    assert load_voice("tonights-finetune").id == "tonights-finetune"


def test_the_shipped_voices_are_still_there_beside_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CRUCIBLE_HOME", str(tmp_path))
    monkeypatch.delenv(VOICES_DIR_ENV, raising=False)
    configure_box(tmp_path)
    a_voice_file(tmp_path / "voices", "tonights-finetune")
    served = load_all_voices()
    for shipped in ("deathstalker", "mistborn", "owen", "zeroshot"):
        assert shipped in served, f"the overlay hid the packaged {shipped}"
    assert list(served) == sorted(served)


def test_a_home_manifest_overrides_a_shipped_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CRUCIBLE_HOME", str(tmp_path))
    monkeypatch.delenv(VOICES_DIR_ENV, raising=False)
    configure_box(tmp_path)
    path = a_voice_file(tmp_path / "voices", "mistborn", display="Mistborn (tonight)")
    assert load_all_voices()["mistborn"].display == "Mistborn (tonight)"
    assert load_voice("mistborn").display == "Mistborn (tonight)"
    path.unlink()
    assert load_all_voices()["mistborn"].display != "Mistborn (tonight)"


def test_the_full_override_still_replaces_everything(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CRUCIBLE_HOME", str(tmp_path / "home"))
    configure_box(tmp_path / "home")
    only = tmp_path / "only"
    a_voice_file(only, "just-this-one")
    a_voice_file(tmp_path / "home" / "voices", "ignored-because-overridden")
    monkeypatch.setenv(VOICES_DIR_ENV, str(only))
    assert sorted(load_all_voices()) == ["just-this-one"]


PIN = (
    'hf_repo = "owenmorgan/probe-higgs-v3"\n'
    'revision = "0123456789abcdef0123456789abcdef01234567"'
)
LOCAL_SOURCE = (
    'path = "/home/telltale/higgs_v3_merged/mb_ha_rvcbed1_5368"\n'
    'identity = "mb_ha_rvcbed1@5368"'
)


def test_a_local_block_loads_and_says_what_it_is() -> None:
    voice = parse(swap(PIN, LOCAL_SOURCE))
    spec = voice.spec("cuda-linux")
    assert spec.source == "local"
    assert spec.hf_repo is None and spec.revision is None
    assert spec.path == "/home/telltale/higgs_v3_merged/mb_ha_rvcbed1_5368"
    assert spec.identity == "mb_ha_rvcbed1@5368"
    assert spec.identity_basis == "asserted"
    assert str(spec.local_path).replace("\\", "/").endswith("mb_ha_rvcbed1_5368")


def test_a_pinned_block_is_still_verified() -> None:
    spec = parse(GOOD).spec("cuda-linux")
    assert spec.source == "pinned"
    assert spec.identity_basis == "verified"
    assert spec.path is None and spec.identity is None
    assert spec.local_path is None


def test_the_fingerprint_is_the_identity_either_way() -> None:
    assert parse(GOOD).fingerprint("cuda-linux") == (
        "probe@0123456789abcdef0123456789abcdef01234567"
    )
    assert parse(swap(PIN, LOCAL_SOURCE)).fingerprint("cuda-linux") == (
        "probe@mb_ha_rvcbed1@5368"
    )


def test_two_sources_are_refused_as_two_sources() -> None:
    message = refused(swap(PIN, PIN + "\n" + LOCAL_SOURCE))
    assert "declares both hf_repo" in message
    assert "names ONE source" in message


def test_no_source_at_all_is_refused_as_none() -> None:
    message = refused(swap(PIN + "\n", ""))
    assert "names no weights" in message
    assert "hf_repo + revision" in message and "path + identity" in message


def test_a_local_block_may_not_also_carry_a_pin_field() -> None:
    message = refused(
        swap(PIN, LOCAL_SOURCE + '\nhf_repo = "owenmorgan/probe-higgs-v3"')
    )
    assert "declares both" in message


def test_a_pinned_block_may_not_carry_an_identity() -> None:
    message = refused(swap(PIN, PIN + '\nidentity = "mb_ha_rvcbed1@5368"'))
    assert "is a pinned block and also carries identity" in message
    assert "VERIFIED" in message


def test_a_local_path_must_be_absolute() -> None:
    message = refused(swap(PIN, LOCAL_SOURCE.replace("/home/telltale", "merged")))
    assert "is not absolute" in message
    assert "whatever directory that process happens to have been started in" in message


def test_a_windows_path_is_absolute_too() -> None:
    voice = parse(
        swap(PIN, 'path = "C:/merged/mb_5368"\nidentity = "mb_ha_rvcbed1@5368"')
    )
    assert voice.spec("cuda-linux").source == "local"


def test_a_local_block_without_an_identity_is_refused() -> None:
    message = refused(swap(PIN, LOCAL_SOURCE.split("\n")[0]))
    assert "and no identity" in message
    assert "no client could tell two of them apart" in message


def test_an_empty_path_is_no_source_rather_than_a_bad_one() -> None:
    message = refused(swap(PIN, 'path = ""\nidentity = "x"'))
    assert "names no weights" in message


def test_a_pin_without_a_revision_says_which_door_may_omit_one() -> None:
    message = refused(swap(PIN, PIN.split("\n")[0]))
    assert "and no revision" in message
    assert "PUT /v1/voices" in message


