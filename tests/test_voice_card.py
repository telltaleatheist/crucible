from __future__ import annotations

import os
from pathlib import Path

import pytest

from crucible.voicecard import (
    CardError,
    export_manifest,
    read_frontmatter,
    render_card,
    render_limits,
)
from crucible.voicecatalog import load_voice
from crucible.voicerepo import REPO_MANIFEST_NAME, parse_repo_manifest
from crucible.voices import VoiceError

from .conftest import configure_box

GOOD = """
schema = 1

[voice]
display         = "Mistborn"
kind            = "checkpoint"
narrator_engine = "higgs-v3"
language        = "en"
sample_rate     = 24000

[voice.pace]
basis              = "measured"
pace_chars_per_sec = 13.76
max_chars_per_sec  = 17.89
min_chars_per_sec  = 10.58
safe_min_chars     = 500
safe_max_chars     = 800
measured_from      = "mb_hp_rvcbed1 ckpt-4257, n=51 in the 500-800 band"

[voice.arms.cuda-linux]
max_chars       = 800
max_chars_basis = "measured"
sampling        = { temperature = 0.8, top_p = 0.95, top_k = 50 }

[voice.arms.mlx-darwin]
max_chars       = 900
max_chars_basis = "placeholder"
sampling        = { temperature = 0.8, top_p = 0.95, top_k = 50 }
"""

CARD = """---
license: other
license_name: research-and-non-commercial
tags:
- text-to-speech
higgs_max_chars_served: 1623
higgs_pace_chars_per_sec: 99.9
---
# Mistborn (Higgs v3)

A LoRA fine-tune (training run `mb_hp_rvcbed1`, checkpoint 4257).

## Measured limits (on the shipped weights, 2026-09-01)

- **Pace:** 99.9 characters per second, which nobody measured.

## NOTICE

This is a Derivative Work.
"""


@pytest.fixture(autouse=True)
def a_configured_box() -> None:
    configure_box(Path(os.environ["CRUCIBLE_HOME"]))


def repo():
    return parse_repo_manifest(GOOD, Path(REPO_MANIFEST_NAME))


def test_what_the_loader_reads_is_what_the_card_says() -> None:
    rendered, added = render_card(repo(), CARD)
    assert added is False
    front = read_frontmatter(rendered)
    manifest = repo()
    assert front["higgs_pace_chars_per_sec"] == str(
        manifest.pace["pace_chars_per_sec"]
    )
    assert front["higgs_pace_basis"] == manifest.pace_basis
    assert front["higgs_safe_min_chars"] == str(manifest.pace["safe_min_chars"])
    assert front["higgs_safe_max_chars"] == str(manifest.pace["safe_max_chars"])
    assert front["higgs_max_chars_served"] == str(
        manifest.arms["cuda-linux"]["max_chars"]
    )
    assert front["higgs_max_chars_mlx"] == str(
        manifest.arms["mlx-darwin"]["max_chars"]
    )
    assert front["higgs_max_chars_served_basis"] == "measured"
    assert front["higgs_max_chars_mlx_basis"] == "placeholder"


def test_a_stale_number_is_replaced_rather_than_added_beside(
) -> None:
    rendered, _ = render_card(repo(), CARD)
    assert rendered.count("higgs_max_chars_served:") == 1
    assert "1623" not in rendered
    assert "99.9" not in rendered


def test_every_line_the_renderer_does_not_own_is_byte_identical() -> None:
    rendered, _ = render_card(repo(), CARD)
    for line in (
        "license: other",
        "license_name: research-and-non-commercial",
        "- text-to-speech",
        "# Mistborn (Higgs v3)",
        "A LoRA fine-tune (training run `mb_hp_rvcbed1`, checkpoint 4257).",
        "## NOTICE",
        "This is a Derivative Work.",
    ):
        assert line in rendered.split("\n"), line


def test_the_notice_survives_because_it_is_the_deploy_s(
) -> None:
    rendered, _ = render_card(repo(), CARD)
    assert rendered.endswith("## NOTICE\n\nThis is a Derivative Work.\n")


def test_a_card_with_no_limits_section_gets_one_and_says_so() -> None:
    bare = CARD[: CARD.index("## Measured limits")] + "## NOTICE\n\nx\n"
    rendered, added = render_card(repo(), bare)
    assert added is True
    assert "## Measured limits" in rendered
    assert "500-800 characters" in rendered


def test_a_card_with_no_frontmatter_is_refused_rather_than_guessed_at() -> None:
    with pytest.raises(CardError) as caught:
        render_card(repo(), "# Mistborn\n\nnothing else\n")
    assert "no YAML frontmatter" in str(caught.value)


def test_the_limits_section_reproduces_the_manifest_s_own_prose() -> None:
    section = render_limits(repo())
    assert "13.76 characters of text per second" in section
    assert "(measured)" in section
    assert "mb_hp_rvcbed1 ckpt-4257, n=51 in the 500-800 band" in section
    assert "**Per-chunk cap, mlx-darwin:** 900 characters (placeholder)." in section


def test_an_uncertified_voice_s_card_says_the_pace_is_not_measured() -> None:
    text = GOOD[: GOOD.index("[voice.pace]")] + GOOD[GOOD.index("[voice.arms.cuda-linux]"):]
    section = render_limits(parse_repo_manifest(text, Path(REPO_MANIFEST_NAME)))
    assert "not measured on these weights" in section
    rendered, _ = render_card(parse_repo_manifest(text, Path(REPO_MANIFEST_NAME)), CARD)
    assert "higgs_pace_chars_per_sec" not in read_frontmatter(rendered)


def test_an_export_refuses_to_invent_the_two_bases() -> None:
    manifest = load_voice("mistborn")
    with pytest.raises(VoiceError) as caught:
        export_manifest(
            manifest,
            pace_basis=None,
            measured_from=None,
        inherited_from=None,
            max_chars_basis="measured",
            uncertified=False,
        )
    assert "--pace-basis" in str(caught.value)
    assert "16.64" in str(caught.value)

    with pytest.raises(VoiceError) as caught:
        export_manifest(
            manifest,
            pace_basis="measured",
            measured_from="ckpt-5368's own ladder, n=51",
        inherited_from=None,
            max_chars_basis=None,
            uncertified=False,
        )
    assert "--max-chars-basis" in str(caught.value)


def test_a_measured_pace_owes_its_prose_on_the_way_out() -> None:
    with pytest.raises(VoiceError) as caught:
        export_manifest(
            load_voice("mistborn"),
            pace_basis="measured",
            measured_from=" ",
        inherited_from=None,
            max_chars_basis="measured",
            uncertified=False,
        )
    assert "--measured-from" in str(caught.value)


def test_an_exported_manifest_parses_as_a_repo_manifest() -> None:
    manifest = load_voice("mistborn")
    text, dropped = export_manifest(
        manifest,
        pace_basis="measured",
        measured_from="mb_ha_rvcbed1 ckpt-5368's own ladder, n=51 in the band",
        inherited_from=None,
        max_chars_basis="measured",
        uncertified=False,
    )
    repo_manifest = parse_repo_manifest(text, Path(REPO_MANIFEST_NAME))
    assert repo_manifest.voice["display"] == manifest.display
    assert repo_manifest.voice["kind"] == manifest.kind
    assert repo_manifest.pace["pace_chars_per_sec"] == (
        manifest.pace.pace_chars_per_sec
    )
    assert repo_manifest.pace["safe_min_chars"] == manifest.pace.safe_min_chars
    assert sorted(repo_manifest.arms) == sorted(manifest.backends)
    for arm in repo_manifest.arms:
        assert repo_manifest.arms[arm]["max_chars"] == manifest.spec(arm).max_chars
    assert len(repo_manifest.takes) == len(manifest.takes)


def test_the_machine_rows_it_drops_are_returned_so_nothing_is_lost_silently(
) -> None:
    manifest = load_voice("mistborn")
    _text, dropped = export_manifest(
        manifest,
        pace_basis="measured",
        measured_from="ckpt-5368's own ladder",
        inherited_from=None,
        max_chars_basis="measured",
        uncertified=False,
    )
    joined = "\n".join(dropped)
    assert "max_num_seqs = 16" in joined
    assert "memory_bytes_estimate = 19000000000" in joined
    assert "pins.toml [mistborn]" in joined
    assert manifest.spec("cuda-linux").hf_repo in joined


def test_an_exported_file_carries_no_machine_fact() -> None:
    text, _dropped = export_manifest(
        load_voice("mistborn"),
        pace_basis="measured",
        measured_from="ckpt-5368's own ladder",
        inherited_from=None,
        max_chars_basis="measured",
        uncertified=False,
    )
    import tomllib

    voice = tomllib.loads(text)["voice"]
    for forbidden in (
        "id",
        "hf_repo",
        "revision",
        "path",
        "identity",
        "memory_bytes_estimate",
        "estimate_basis",
        "estimate_note",
        "serving",
        "backends",
    ):
        assert forbidden not in voice, forbidden
    for arm in voice["arms"].values():
        for forbidden in ("memory_bytes_estimate", "estimate_basis", "estimate_note"):
            assert forbidden not in arm, forbidden
    parse_repo_manifest(text, Path(REPO_MANIFEST_NAME))


def test_an_uncertified_export_omits_the_pace_table_in_whole() -> None:
    from dataclasses import replace

    from crucible.voices import Pace

    manifest = load_voice("mistborn")
    blank = replace(
        manifest,
        pace=Pace(None, None, None, None, None, None),
    )
    with pytest.raises(VoiceError) as caught:
        export_manifest(
            blank,
            pace_basis=None,
            measured_from=None,
        inherited_from=None,
            max_chars_basis="placeholder",
            uncertified=False,
        )
    assert "--uncertified" in str(caught.value)
    text, _dropped = export_manifest(
        blank,
        pace_basis=None,
        measured_from=None,
        inherited_from=None,
        max_chars_basis="placeholder",
        uncertified=True,
    )
    import tomllib

    assert "pace" not in tomllib.loads(text)["voice"]
    repo_manifest = parse_repo_manifest(text, Path(REPO_MANIFEST_NAME))
    assert repo_manifest.pace is None


INHERITED_TOML = GOOD.replace('basis              = "measured"', 'basis = "inherited"').replace(
    'measured_from      = "mb_hp_rvcbed1 ckpt-4257, n=51 in the 500-800 band"',
    'inherited_from = "ow_v8_rvcbed1 ckpt-966; these weights have no ladder yet"',
)


def test_the_card_names_the_weights_an_inherited_pace_came_from() -> None:
    section = render_limits(parse_repo_manifest(INHERITED_TOML, Path(REPO_MANIFEST_NAME)))
    assert "(inherited)" in section
    assert "Inherited from ow_v8_rvcbed1 ckpt-966" in section
    rendered, _ = render_card(
        parse_repo_manifest(INHERITED_TOML, Path(REPO_MANIFEST_NAME)), CARD
    )
    assert read_frontmatter(rendered)["higgs_pace_basis"] == "inherited"


def test_an_export_refuses_to_invent_the_inherited_sentence() -> None:
    with pytest.raises(VoiceError) as caught:
        export_manifest(
            load_voice("mistborn"),
            pace_basis="inherited",
            measured_from=None,
            inherited_from=None,
            max_chars_basis="measured",
            uncertified=False,
        )
    assert "--pace-basis inherited owes --inherited-from" in str(caught.value)
    assert "16.64" in str(caught.value)


def test_an_export_refuses_the_sentence_the_other_basis_owes() -> None:
    with pytest.raises(VoiceError) as caught:
        export_manifest(
            load_voice("mistborn"),
            pace_basis="measured",
            measured_from="ckpt-5368's own ladder",
            inherited_from="ow_v8_rvcbed1 ckpt-966",
            max_chars_basis="measured",
            uncertified=False,
        )
    assert "was given --inherited-from as well" in str(caught.value)


def test_an_inherited_export_round_trips() -> None:
    text, _dropped = export_manifest(
        load_voice("mistborn"),
        pace_basis="inherited",
        measured_from=None,
        inherited_from="mb_full_rvc1 ckpt-5947; these weights have no ladder yet",
        max_chars_basis="measured",
        uncertified=False,
    )
    repo_manifest = parse_repo_manifest(text, Path(REPO_MANIFEST_NAME))
    assert repo_manifest.pace_basis == "inherited"
    assert repo_manifest.inherited_from.startswith("mb_full_rvc1 ckpt-5947")
    assert repo_manifest.measured_from is None
