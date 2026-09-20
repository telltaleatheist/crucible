"""Voice manifests: every refusal, and the real catalog this build ships.

`crucible/voices.py` is `manifests.py`'s strictness applied to a file where a
wrong number is a whole book rendered wrong and reported as success — a cap that
cuts chunks mid-sentence, a pace band that re-rolls every healthy take, a
reference clip cloned from an absent transcript. So every refusal has a test, and
the shipped manifests are checked against the numbers they were translated from.
"""

from __future__ import annotations

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

#: A manifest that loads. Every refusal test below is this document with one
#: thing wrong, so a failure names the one thing rather than the whole schema.
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


def parse(text: str, voice_id: str = "probe"):
    return parse_voice(text, Path(f"{voice_id}.toml"), voice_id)


def refused(text: str, voice_id: str = "probe") -> str:
    with pytest.raises(VoiceError) as caught:
        parse(text, voice_id)
    return str(caught.value)


def swap(old: str, new: str) -> str:
    assert old in GOOD, f"the good manifest does not contain {old!r}"
    return GOOD.replace(old, new)


#: The good manifest with the `[voice.serving]` table removed. `max_num_seqs`
#: is a HIGGS_* variable, so a manifest naming an engine that reads none of
#: them must drop the table with it — and the fixture exists so a test about
#: the missing table is not also a test about the engine.
SERVING_TABLE = GOOD[GOOD.index("[voice.serving]"):GOOD.index("[voice.backends")]


# ------------------------------------------------------------------ the base


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
    """Two merges of one run are two sets of weights under one name."""
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


# ------------------------------------------------------------------- [voice]


def test_an_unknown_top_level_table_is_refused() -> None:
    assert "unknown top-level table(s) ['serving']" in refused(
        GOOD + '\n[serving]\nport = 8095\n'
    )


def test_a_manifest_with_no_voice_table_is_refused() -> None:
    assert "missing the [voice] table" in refused("")


def test_an_unknown_key_in_voice_is_refused() -> None:
    """A typo must not load with the field silently absent."""
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


# -------------------------------------------------------------- [voice.pace]


def test_a_voice_may_omit_the_pace_table_entirely() -> None:
    """THE TABLE ITSELF, not only its rates (PHASE18 section 4.1, 2026-09-19).

    `missing the [voice.pace] table` was the refusal here, and with the three
    rates optional as a group since 2026-09-18 it was the last thing making a
    screening checkpoint inexpressible: the only way past it was an empty table
    written to satisfy a parser. Absent and empty now mean the same thing, which
    is "nothing was measured", and neither is a band of zeros or a predecessor's
    numbers.
    """
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
    """Omissible is not the same as "anything goes": a `pace` that is a number
    is a file somebody got wrong, and reading it as absent would pass it."""
    without_pace = (
        GOOD[: GOOD.index("[voice.pace]")] + GOOD[GOOD.index("[voice.serving]"):]
    )
    assert "[voice.pace] must be a table" in refused(
        without_pace.replace("sample_rate = 24000", "sample_rate = 24000\npace = 16.0")
    )


def test_half_a_band_is_refused() -> None:
    """narrator's rule: the band is a triple, write all three or none."""
    message = refused(swap("min_chars_per_sec = 12.3\n", ""))
    assert "declares only part of its rate band" in message
    assert "['min_chars_per_sec']" in message


def test_a_voice_may_state_no_rates_at_all() -> None:
    """THE OTHER HALF OF "all three or none": none is a legal statement.

    A voice nobody ran a ladder on has no pace to state, and narrator already
    knows what to do with one — given no band it uses its engine's own default
    and derives the centre as the geometric mean of the two edges
    (`engine/higgs/truncation.tracker_for`). The alternative, which the loader
    obliged until 2026-09-18, is a manifest made to invent three numbers.
    """
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
    # The packing shape is a different fact in the same table and is untouched.
    assert voice.pace.safe_min_chars == 600
    assert voice.pace.safe_max_chars == 800


def test_a_pace_outside_its_own_edges_is_refused() -> None:
    message = refused(swap("pace_chars_per_sec = 16.0", "pace_chars_per_sec = 24.0"))
    assert "out of order" in message
    assert "the band is min < pace < max" in message


#: THE TRIPLE THAT CAUSED THE RULING — narrator's Higgs v3 defaults spliced
#: together, `HiggsDefaults.CHARS_PER_SEC` 15.0 between `HiggsV3Defaults`'
#: 20.0/14.5 edges. It satisfies `min < pace < max` and is still wrong: the two
#: halves were written around different centres, so the band is 1.333 long and
#: 1.034 short, and narrator — which keeps only the RATIOS — judged healthy
#: chunks run-ons and re-rolled them to MAX_DEPTH.
SPLICED = swap(
    "pace_chars_per_sec = 16.0\nmax_chars_per_sec = 20.8\nmin_chars_per_sec = 12.3",
    "pace_chars_per_sec = 15.0\nmax_chars_per_sec = 20.0\nmin_chars_per_sec = 14.5",
)


def test_a_lopsided_triple_is_refused_naming_both_ratios() -> None:
    """A BAND IS SYMMETRIC UNLESS IT SAYS OTHERWISE (ruled 2026-09-18).

    `min < pace < max` passes this triple, which is how it shipped. The defect
    is one level up: the edges were not derived from the pace at all.
    """
    message = refused(SPLICED)
    assert "1.333" in message
    assert "1.034" in message
    assert "edges" in message


def test_a_percentile_band_may_be_lopsided() -> None:
    """The escape hatch, and it is a STATEMENT rather than a tolerance.

    A band read off a distribution's percentiles is lopsided because the
    distribution is, and there is nothing to refuse. Saying so in the manifest
    is what separates that voice from one whose edges were spliced.
    """
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
    """A typo in the escape hatch must not read as "not percentile, so check"."""
    message = refused(
        SPLICED.replace(
            "min_chars_per_sec = 14.5",
            'min_chars_per_sec = 14.5\nedges = "percentiles"',
        )
    )
    assert "edges" in message
    assert "'percentiles'" in message


def test_edges_without_a_band_is_refused() -> None:
    """A key that describes edges the manifest does not state is a leftover."""
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
    """Floor == cap is how a 400-character chunk ships alone (2026-09-09)."""
    message = refused(swap("safe_min_chars = 600", "safe_min_chars = 800"))
    assert "is not below safe_max_chars" in message


def test_a_voice_declaring_neither_packs_to_the_backend_cap() -> None:
    """What `higgs-default` does, and what BookForge does with such a voice."""
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


# ---------------------------------------------------------- backend blocks


def test_a_voice_with_no_backend_block_is_refused() -> None:
    assert "missing every [voice.backends.<kind>] table" in refused(
        GOOD[: GOOD.index("[voice.backends")]
    )


def test_an_empty_backends_table_is_refused() -> None:
    """A voice nothing can serve is not a voice."""
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
    """PHASE18 section 4, 2026-09-19. `max_chars` was `_BACKEND_REQUIRED` and
    narratorvoices called it "the one field narrator refuses a checkpoint voice
    without"; both moved on the same day.

    A cap is a RESULT — the longest chunk a sweep on these weights on this arm
    came back whole from — so a checkpoint being screened has none, and
    requiring it made a screening voice inexpressible. `None` means NOT
    MEASURED and nothing substitutes: not the other arm's number, not a
    sibling's, not the engine's.
    """
    voice = parse(
        swap("safe_min_chars = 600\nsafe_max_chars = 800\n", "").replace(
            "max_chars = 800\n", ""
        )
    )
    assert voice.spec("cuda-linux").max_chars is None
    assert voice.spec("cuda-linux").to_dict()["max_chars"] is None


def test_a_packing_band_above_a_cap_is_still_refused_when_a_cap_is_stated() -> None:
    """The relaxation is "may be absent", not "is no longer checked": a stated
    cap keeps every rule it had, so the band-above-the-cap refusal below is not
    reachable only by accident."""
    assert "The band may never exceed the arm's cap" in refused(
        swap("safe_max_chars = 800", "safe_max_chars = 900")
    )


# ---------------------------------------------------------- estimate_basis


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
    """A note beside a measured number reads as an excuse for it."""
    message = refused(
        swap(
            'estimate_basis = "measured"',
            'estimate_basis = "measured"\nestimate_note = "about right"',
        )
    )
    assert "estimate_basis is 'measured' and it also carries an estimate_note" in message


# ----------------------------------------------------------------- sampling


def test_the_boson_default_needs_no_reason() -> None:
    assert parse(GOOD).spec("cuda-linux").sampling_reason is None


def test_a_deviation_without_a_reason_is_refused() -> None:
    """Owen's rule: one engine-level number, and a deviation owes a reason."""
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
    """On SGLang an unfilled top_k samples the untruncated codebook tail."""
    message = refused(
        swap(
            "sampling = { temperature = 0.8, top_p = 0.95, top_k = 50 }",
            "sampling = { temperature = 0.8 }",
        )
    )
    assert "missing required key(s) ['top_k', 'top_p']" in message


# --------------------------------------------------------- [voice.serving]


def test_the_serving_width_is_read() -> None:
    voice = parse(GOOD)
    assert voice.serving is not None
    assert voice.serving.max_num_seqs == 16
    assert "stage-0" in voice.serving.max_num_seqs_note
    assert voice.to_dict()["serving"]["max_num_seqs"] == 16


def test_a_higgs_voice_without_a_serving_table_is_refused() -> None:
    """narrator refuses HIGGS_MAX_NUM_SEQS by name — it is stage 0's admission
    width AND the width of narrator's own batch — so a manifest that does not
    state it cannot start a server."""
    message = refused(GOOD.replace(SERVING_TABLE, ""))
    assert "[voice.serving]" in message
    assert "max_num_seqs" in message


def test_a_serving_width_below_one_is_refused() -> None:
    assert "at least 1" in refused(swap("max_num_seqs = 16", "max_num_seqs = 0"))


def test_a_serving_width_with_no_note_is_refused() -> None:
    """The same contract `estimate_note` has, for the same reason: 16 is
    contested by a live certificate that ran at 64."""
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


#: The two levers added on 2026-09-19, as a screening voice states them.
SCREENING_SERVING = """mem_fraction = 0.48
mem_fraction_note = "0.48 + width 4 is ~20 GB; 0.55 measured 24.0-24.1 GB on this 24 GB card and WDDM then pages to host RAM"
context_length = 8192
context_length_note = "the Third Reich bank tops at 2,008 chars and 4096 tokens holds ~2,000, so the top rungs truncate on the context"
"""


def test_a_voice_may_state_a_mem_fraction_and_a_context_length() -> None:
    """Owen's ruling of 2026-09-19, which decides PHASE18 section 11.

    Both OPTIONAL — every manifest this build ships states neither and must
    keep loading — and both reach narrator at engine start on BOTH arms.
    """
    voice = parse(swap("max_num_seqs = 16\n", "max_num_seqs = 16\n" + SCREENING_SERVING))
    assert voice.serving.mem_fraction == 0.48
    assert voice.serving.context_length == 8192
    assert "24 GB card" in voice.serving.mem_fraction_note
    assert "2,008" in voice.serving.context_length_note


def test_a_voice_stating_neither_reports_them_as_null_on_its_row() -> None:
    """Absent means narrator's own launcher default — 0.60
    (`serve_higgs_sgl.sh:59`) and the Higgs builder's 4096 — which is a number
    in a file with an owner. The KEYS are still on the row, because an absent
    key would say "this build has no such field"."""
    row = parse(GOOD).serving.to_dict()
    assert row["mem_fraction"] is None and row["mem_fraction_note"] is None
    assert row["context_length"] is None and row["context_length_note"] is None


@pytest.mark.parametrize("key", ["mem_fraction", "context_length"])
def test_a_serving_lever_with_no_note_is_refused(key: str) -> None:
    """`max_num_seqs`'s contract, applied to both: a number that reconfigures
    the server narrator starts owes the measurement that chose it."""
    stated = {"mem_fraction": "0.48", "context_length": "8192"}[key]
    message = refused(
        swap("max_num_seqs = 16\n", f"max_num_seqs = 16\n{key} = {stated}\n")
    )
    assert f"{key} carries no note" in message


@pytest.mark.parametrize("key", ["mem_fraction", "context_length"])
def test_a_serving_note_with_no_number_is_refused_as_the_leftover_it_is(
    key: str,
) -> None:
    """The other half. A note describing a number that is not there reads as a
    lever this loader checked and passed — `[voice.pace]`'s `edges` key and the
    repo schema's `max_chars_basis` are refused with the same sentence."""
    message = refused(
        swap("max_num_seqs = 16\n", f'max_num_seqs = 16\n{key}_note = "x"\n')
    )
    assert f"states {key}_note and no {key}" in message


def test_a_mem_fraction_outside_zero_to_one_is_refused() -> None:
    """narrator's launcher refuses anything else by name
    (`serve_higgs_sgl.sh:129-131`), so the manifest refuses it first — a
    fraction of 1.2 would otherwise be a worker that exits 4 after a load."""
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
    """Owen, 2026-09-19: *"we're going to want to configure darwin to work the
    same way. context limits and such."* So these are NOT cuda-linux only, and
    a voice that serves on MLX may state both. What narrator's MLX backend has
    no knob for is narrator's refusal to make, by name, at load."""
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
    """What the job result pins (2026-09-19). The rung's override alone says
    nothing about the top-p and top-k a run actually used, and those are what a
    ladder's comparison rests on."""
    voice = parse(GOOD + LADDER)
    assert voice.applied_sampling("cuda-linux", 0) == {
        "temperature": 0.8, "top_p": 0.95, "top_k": 50,
    }
    assert voice.applied_sampling("cuda-linux", 1) == {
        "temperature": 0.7, "top_p": 0.95, "top_k": 50,
    }
    # And past the ladder it is take 0's, which is what a seed lane means.
    assert voice.applied_sampling("cuda-linux", 5) == {
        "temperature": 0.8, "top_p": 0.95, "top_k": 50,
    }


def test_every_shipped_higgs_voice_declares_one() -> None:
    """Not a fixture: the real manifests. A voice that loads but cannot be
    started is a row on /v1/voices that fails at the spawn."""
    for voice in load_all_voices().values():
        if voice.narrator_engine == "higgs-v3":
            assert voice.serving is not None, voice.id
            assert voice.serving.max_num_seqs >= 1, voice.id
            assert voice.serving.max_num_seqs_note.strip(), voice.id


# -------------------------------------------------------------------- clips


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
    """narrator refuses it too: a clone on an absent transcript is a whole book
    in a subtly wrong voice, reported as success."""
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


# -------------------------------------------------------------------- takes


def test_a_voice_with_no_ladder_still_has_take_zero() -> None:
    voice = parse(GOOD)
    assert len(voice.takes) == 1
    assert voice.take(0).overrides == {}
    assert voice.take(0).reason is None


def test_a_take_past_the_end_is_a_seed_lane_at_take_zeros_sampling() -> None:
    """PHASE18 section 5, 2026-09-19. It used to be `unknown_take`.

    A screening sweep names takes 0..N on a voice that declares no ladder at
    all and needs every one of them to be the SAME sampling — the matched-cell
    property its Wilson-bound comparison rests on — so "past the end" has to
    mean the voice's own numbers. narrator still moves the draw, because `take`
    rides on every item and seeds `base + index + stride * take`.
    """
    rung = parse(GOOD).take(3)
    assert rung.index == 3
    assert rung.overrides == {}
    assert rung.reason is None
    # And the ladder is still what the manifest declares: a lane is not a rung.
    assert len(parse(GOOD).takes) == 1


def test_a_take_past_a_DECLARED_ladder_is_not_the_last_rungs_numbers() -> None:
    """Not a clamp, and this is the assertion that says so. A clamp would hand
    back take 1's `temperature = 0.7` under take 4's name, which is the silent
    substitution the old refusal existed to prevent — and the relaxation keeps
    preventing it by answering with take 0's sampling instead."""
    rung = parse(GOOD + LADDER).take(4)
    assert rung.overrides == {}
    assert parse(GOOD + LADDER).take(1).overrides == {"temperature": 0.7}


def test_a_negative_take_is_still_refused() -> None:
    """The one thing `take` still refuses. No door can reach it — both carry
    `Field(ge=0)` — but a lane below 0 is a bug in a caller, not a draw."""
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
    """PHASE18 section 5, 2026-09-19. This was `take 1 changes nothing`.

    The refusal was true when written — a rung could not move the seed, so a
    numberless one rendered take 0 byte for byte. Since 2026-09-15 narrator
    draws `base + index + REROLL_SEED_STRIDE * (TAKE_REROLL_LANES * take +
    attempt)`, so the same numbers in another lane is a different draw, which
    is the screening ladder's entire unit of work. narrator's own
    `item_sampling.py` says so and leaves the ruling to Crucible.
    """
    voice = parse(GOOD + "\n[[voice.takes]]\n\n[[voice.takes]]\n")
    assert len(voice.takes) == 2
    assert voice.take(1).overrides == {}
    assert voice.take(1).reason is None


def test_take_zero_still_may_not_deviate_and_a_deviation_still_owes_a_reason() -> None:
    """The two rules the relaxation does NOT touch, pinned together so a future
    edit cannot take all three out as one."""
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


# ------------------------------------------------------------------ loading


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
    """`zeroshot` sorts before `zeroshot-x` as ids and after it as filenames,
    because '-' is 0x2D and '.' is 0x2E. This build has that exact pair."""
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


# ------------------------------------------------------- the shipped catalog


SHIPPED = {
    # id: (kind, caps per backend, safe band or target)
    "deathstalker": ("checkpoint", 800),
    "mistborn": ("checkpoint", 800),
    "owen": ("checkpoint", 800),
    "thirdreich": ("checkpoint", 1000),
    "sigma": ("checkpoint", 1100),
    "higgs-default": ("token", 600),
    "zeroshot": ("zeroshot", 600),
}


#: The two voices in this build that are the BASE WEIGHTS rather than a
#: fine-tune, and therefore the two nobody has run a length ladder on. They are
#: named here because "which voices have no measured pace" is a fact about the
#: catalog, and a third one appearing must be a deliberate edit to this line
#: rather than a manifest quietly shipping without a band.
UNMEASURED_PACE = ("higgs-default", "zeroshot")


def test_this_build_ships_the_voices_it_says_it_does() -> None:
    assert sorted(load_all_voices()) == sorted(SHIPPED)


@pytest.mark.parametrize("voice_id", sorted(SHIPPED))
def test_a_manifest_states_the_pace_it_measured_and_no_other(voice_id: str) -> None:
    """A MANIFEST STATES WHAT WAS MEASURED (ruling of 2026-09-18).

    `higgs-default` and `zeroshot` are the base weights, no ladder has been run
    on either, and until 2026-09-18 both satisfied a required-triple rule by
    copying narrator's Higgs v3 defaults out of its source — `CHARS_PER_SEC`
    15.0, which is the DIVISOR `cap_frames()` sizes the frame cap against and
    not a rate anything was measured speaking at, between edges written around
    a book pace nearer 17.2. narrator keeps a band's RATIOS, and those ratios
    are 1.333 short against 1.034 long, so it judged healthy chunks run-ons and
    re-rolled them to MAX_DEPTH. Stating nothing puts the derivation back where
    it has one owner: narrator centres its own default band on the geometric
    mean of the edges.

    The five fine-tunes DO state all three, each from its own ladder, and this
    test is the pair of those two statements.
    """
    pace = load_voice(voice_id).pace
    rates = (pace.pace_chars_per_sec, pace.max_chars_per_sec, pace.min_chars_per_sec)
    if voice_id in UNMEASURED_PACE:
        assert rates == (None, None, None)
        # Dropping the rates did not take the packing shape with them: they are
        # two different facts in one table, and zeroshot still packs to the
        # catalog's single 600-character target.
        if voice_id == "zeroshot":
            assert pace.target_chars == 600
        return
    assert all(rate is not None for rate in rates), voice_id
    assert pace.min_chars_per_sec < pace.pace_chars_per_sec < pace.max_chars_per_sec


@pytest.mark.parametrize("voice_id", sorted(set(SHIPPED) - set(UNMEASURED_PACE)))
def test_every_measured_band_in_this_catalog_is_symmetric(voice_id: str) -> None:
    """EVERY LADDER IN THIS BUILD WROTE max = pace x 1.3 AND min = pace / 1.3.

    Not a rule the loader imposes — `edges = "percentile"` exists for a band
    read off a distribution instead — but a fact about the five manifests that
    are here, and the reason the symmetry check can be the default. A sixth
    voice arriving with genuinely lopsided edges must say so in its manifest,
    and this test is where that shows up.
    """
    pace = load_voice(voice_id).pace
    long_side = pace.max_chars_per_sec / pace.pace_chars_per_sec
    short_side = pace.pace_chars_per_sec / pace.min_chars_per_sec
    assert round(long_side, 2) == 1.3, voice_id
    assert round(short_side, 2) == 1.3, voice_id


@pytest.mark.parametrize("voice_id", sorted(SHIPPED))
def test_each_shipped_voice_carries_the_catalog_numbers(voice_id: str) -> None:
    """The caps are 600 / 800 / 1000 / 1100, from higgs-models.json."""
    kind, cap = SHIPPED[voice_id]
    voice = load_voice(voice_id)
    assert voice.kind == kind
    assert voice.narrator_engine == "higgs-v3"
    assert voice.sample_rate == 24000
    # Every Higgs voice is staged on both arms in the catalog, so every manifest
    # carries both blocks — and the cap is stated per backend even though the two
    # agree today, because a cap is produced by rendering and the two arms sample
    # through different implementations.
    assert sorted(voice.backends) == ["cuda-linux", "mlx-darwin"]
    for backend in voice.backends:
        assert voice.spec(backend).max_chars == cap


@pytest.mark.parametrize("voice_id", sorted(SHIPPED))
def test_every_shipped_voice_renders_at_the_boson_default(voice_id: str) -> None:
    """One engine-level number for every Higgs voice on both arms; no voice in
    this build has a measured reason to deviate, so none does."""
    voice = load_voice(voice_id)
    for backend in voice.backends:
        spec = voice.spec(backend)
        assert spec.sampling == NARRATOR_ENGINE_SAMPLING["higgs-v3"]
        assert spec.sampling_reason is None


@pytest.mark.parametrize("voice_id", sorted(SHIPPED))
def test_no_shipped_voice_claims_a_measured_estimate(voice_id: str) -> None:
    """Neither of Owen's cards was free; PHASE3-TTS.md section 10 leaves the
    measurement owed, and the row must say so rather than imply otherwise."""
    voice = load_voice(voice_id)
    for backend in voice.backends:
        spec = voice.spec(backend)
        assert spec.estimate_basis == "declared"
        assert spec.estimate_note


@pytest.mark.parametrize("voice_id", sorted(SHIPPED))
def test_the_five_fine_tunes_declare_a_second_rung_and_nothing_else_does(
    voice_id: str,
) -> None:
    """The ladder as shipped on 2026-09-14, after Owen's ruling that a retake
    must not reuse the settings that produced the problem.

    Two rungs on every `checkpoint` voice — take 0 the boson default, take 1
    the measured 0.7 — and ONE on the base-weights pair. `higgs-default` and
    `zeroshot` are left alone deliberately: the 0.7 measurement is 88 chunks
    of a fine-tune's output, a zero-shot voice's spread depends on a clip
    nobody has measured against, and a rung that is not measured is a number
    somebody will later mistake for one. They still have take 0, which every
    voice has whether or not its file says so.
    """
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
    # The rung owes the measurement that chose it, in writing, and the reason
    # says what the measurement was rather than that 0.7 is better — it is
    # not: it fired the guard twice as often. It is DIFFERENT, which is the
    # whole property a retake needs.
    reason = voice.take(1).reason
    assert reason is not None
    assert "measured 2026-09-11" in reason
    assert "88 chunks" in reason
    assert "a DIFFERENT one" in reason


def test_the_zeroshot_voice_takes_its_clips_from_the_request() -> None:
    """Its reference wavs live in a userData directory nothing can pull from."""
    voice = load_voice("zeroshot")
    for backend in voice.backends:
        assert voice.spec(backend).clips_from_request is True


def test_the_shipped_manifests_are_the_directory_beside_the_package() -> None:
    assert voices_dir() == Path(__file__).resolve().parent.parent / "crucible" / "voices"


# ------------------------------------------- deploying a voice without a release
#
# Owen, 2026-09-16: *"we dont have to cut a new release every time we deploy a
# model do we? if thats the case, we should simplify it so it just reads the
# model manifest and i can upload new models at will. i train models all the
# time. nearly every night."*
#
# Before this, voice manifests were package data and the answer was yes: a voice
# could not be served until a version was tagged, its packs rebuilt on CI and the
# result installed on three machines. These four tests are the answer being no.


def a_voice_file(into: Path, voice_id: str, display: str | None = None) -> Path:
    """A real manifest, copied from a shipped one and re-identified."""
    import re

    into.mkdir(parents=True, exist_ok=True)
    raw = (voices_dir() / "mistborn.toml").read_text(encoding="utf-8")
    raw = raw.replace('id = "mistborn"', f'id = "{voice_id}"', 1)
    if display is not None:
        raw = re.sub(r'display\s*=\s*"[^"]*"', f'display = "{display}"', raw, count=1)
    path = into / f"{voice_id}.toml"
    path.write_text(raw, encoding="utf-8")
    return path


def test_a_voice_dropped_into_the_home_is_served(tmp_path, monkeypatch) -> None:
    """THE WHOLE POINT: a file appears, the voice exists, nothing was released."""
    monkeypatch.setenv("CRUCIBLE_HOME", str(tmp_path))
    monkeypatch.delenv(VOICES_DIR_ENV, raising=False)
    a_voice_file(tmp_path / "voices", "tonights-finetune")
    assert "tonights-finetune" in load_all_voices()
    assert load_voice("tonights-finetune").id == "tonights-finetune"


def test_the_shipped_voices_are_still_there_beside_it(tmp_path, monkeypatch) -> None:
    """AN OVERLAY, NOT A REPLACEMENT — the distinction the old env var got wrong.

    `CRUCIBLE_VOICES_DIR` replaces the set, so adding one voice through it meant
    copying all seven shipped manifests somewhere and maintaining them by hand
    forever. Adding must not cost that.
    """
    monkeypatch.setenv("CRUCIBLE_HOME", str(tmp_path))
    monkeypatch.delenv(VOICES_DIR_ENV, raising=False)
    a_voice_file(tmp_path / "voices", "tonights-finetune")
    served = load_all_voices()
    for shipped in ("deathstalker", "mistborn", "owen", "zeroshot"):
        assert shipped in served, f"the overlay hid the packaged {shipped}"
    # And the order is by ID, not by directory: a home voice belongs where its
    # name puts it, because this dict's order is what `/v1/voices` lists in.
    assert list(served) == sorted(served)


def test_a_home_manifest_overrides_a_shipped_id(tmp_path, monkeypatch) -> None:
    """Retuning a shipped voice is the same gesture, and is REVERSIBLE.

    Deleting the file restores the packaged manifest, which is what makes
    trying a new pace on deathstalker a safe thing to do on a Tuesday.
    """
    monkeypatch.setenv("CRUCIBLE_HOME", str(tmp_path))
    monkeypatch.delenv(VOICES_DIR_ENV, raising=False)
    path = a_voice_file(tmp_path / "voices", "mistborn", display="Mistborn (tonight)")
    assert load_all_voices()["mistborn"].display == "Mistborn (tonight)"
    assert load_voice("mistborn").display == "Mistborn (tonight)"
    path.unlink()
    assert load_all_voices()["mistborn"].display != "Mistborn (tonight)"


def test_the_full_override_still_replaces_everything(tmp_path, monkeypatch) -> None:
    """The escape hatch keeps its meaning: run THIS set and nothing else."""
    monkeypatch.setenv("CRUCIBLE_HOME", str(tmp_path / "home"))
    only = tmp_path / "only"
    a_voice_file(only, "just-this-one")
    a_voice_file(tmp_path / "home" / "voices", "ignored-because-overridden")
    monkeypatch.setenv(VOICES_DIR_ENV, str(only))
    assert sorted(load_all_voices()) == ["just-this-one"]


# ------------------------------------------------- the source axis (PHASE18)
#
# A backend block names ONE source: a pin Crucible fetches and owns, or a
# directory somebody else put on the serving machine
# (PHASE18-UNCERTIFIED.md section 3). Both halves of "exactly one" are refused
# by their own name, because a block with two sources and a block with none are
# different mistakes.

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
    # ASSERTED, not verified: nothing checked that directory against anything.
    assert spec.identity_basis == "asserted"
    assert str(spec.local_path).replace("\\", "/").endswith("mb_ha_rvcbed1_5368")


def test_a_pinned_block_is_still_verified() -> None:
    spec = parse(GOOD).spec("cuda-linux")
    assert spec.source == "pinned"
    assert spec.identity_basis == "verified"
    assert spec.path is None and spec.identity is None
    assert spec.local_path is None


def test_the_fingerprint_is_the_identity_either_way() -> None:
    # One shape, so a client comparing two renders never parses before it can
    # compare. How much the answer is worth is `identity_basis` on the row.
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
    # The loader may be running on a different OS than the one that will serve
    # the voice, so "absolute" is asked of the PATH and not of this host.
    voice = parse(
        swap(PIN, 'path = "C:/merged/mb_5368"\nidentity = "mb_ha_rvcbed1@5368"')
    )
    assert voice.spec("cuda-linux").source == "local"


def test_a_local_block_without_an_identity_is_refused() -> None:
    message = refused(swap(PIN, LOCAL_SOURCE.split("\n")[0]))
    assert "and no identity" in message
    assert "no client could tell two of them apart" in message


def test_an_empty_path_is_no_source_rather_than_a_bad_one() -> None:
    # `path = ""` reads as a declaration to `in`, and would otherwise be
    # refused for not being absolute — a second-order message about a block
    # that really declared no source at all.
    message = refused(swap(PIN, 'path = ""\nidentity = "x"'))
    assert "names no weights" in message


def test_a_pin_without_a_revision_says_which_door_may_omit_one() -> None:
    message = refused(swap(PIN, PIN.split("\n")[0]))
    assert "and no revision" in message
    assert "PUT /v1/voices" in message


# The pin's own two checks — the `<owner>/<name>` shape and the 40-character
# sha — moved into `_check_source` with everything else and are NOT re-tested
# here: `test_a_branch_name_is_not_a_pin` and `test_a_bare_repo_name_is_refused`
# above run through the moved code and are the owners of those two facts.
