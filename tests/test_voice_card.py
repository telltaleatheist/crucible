"""`crucible voices card` and `crucible voices export` — PHASE21 sections 2.5 and 4.

The card used to be written by a regex in a campaign script generated fresh per
deploy, and nothing ever read it back: thirdreich's card carried
`higgs_target_chars` ten days after that field was retired, and its
`higgs_max_chars_served: 1623` is a training-row length rather than a served
sweep. One renderer, in the repo with the loader, tested as *what the loader
reads is what the card says*.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.voicecard import (
    CardError,
    export_manifest,
    read_frontmatter,
    render_card,
    render_limits,
)
from crucible.voicerepo import REPO_MANIFEST_NAME, parse_repo_manifest
from crucible.voices import VoiceError, load_voice

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


def repo():
    return parse_repo_manifest(GOOD, Path(REPO_MANIFEST_NAME))


def test_what_the_loader_reads_is_what_the_card_says() -> None:
    """RENDER -> PARSE THE FRONTMATTER BACK -> EQUALS THE TOML.

    The one assertion this whole command exists for. Every number on the card is
    read back out of the card and compared with the manifest it was rendered
    from, so the two cannot drift by a typo or a stale regex.
    """
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
    # THE TWO ARMS ARE NOT ALWAYS EQUAL and one of them may be a number nobody
    # measured — thirdreich's 900 on mlx. The card says so now.
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


def test_the_notice_survives_because_it_is_the_deploy_s(  # noqa: N802
) -> None:
    """Section 7: the card is Crucible's; the NOTICE stays the deploy's. It is a
    licence notice, not a voice fact."""
    rendered, _ = render_card(repo(), CARD)
    assert rendered.endswith("## NOTICE\n\nThis is a Derivative Work.\n")


def test_a_retired_key_is_refused_by_name() -> None:
    with pytest.raises(CardError) as caught:
        render_card(repo(), CARD.replace("higgs_max_chars_served: 1623",
                                         "higgs_target_chars: 1000"))
    message = str(caught.value)
    assert "higgs_target_chars" in message
    assert "retired" in message
    assert "not dropped silently" in message


def test_a_card_with_no_limits_section_gets_one_and_says_so() -> None:
    """thirdreich's and sigma's carry no safe band at all today (section 1)."""
    bare = CARD[: CARD.index("## Measured limits")] + "## NOTICE\n\nx\n"
    rendered, added = render_card(repo(), bare)
    assert added is True
    assert "## Measured limits" in rendered
    assert "500-800 characters" in rendered


def test_a_card_with_no_frontmatter_is_refused_rather_than_guessed_at() -> None:
    with pytest.raises(CardError) as caught:
        render_card(repo(), "# Mistborn\n\nnothing else\n")
    assert "no YAML frontmatter" in str(caught.value)


def test_the_limits_section_reproduces_the_manifest_s_own_prose() -> None:  # noqa: N802
    section = render_limits(repo())
    assert "13.76 characters of text per second" in section
    assert "(measured)" in section
    assert "mb_hp_rvcbed1 ckpt-4257, n=51 in the 500-800 band" in section
    assert "**Per-chunk cap, mlx-darwin:** 900 characters (placeholder)." in section


def test_an_uncertified_voice_s_card_says_the_pace_is_not_measured() -> None:  # noqa: N802
    text = GOOD[: GOOD.index("[voice.pace]")] + GOOD[GOOD.index("[voice.arms.cuda-linux]"):]
    section = render_limits(parse_repo_manifest(text, Path(REPO_MANIFEST_NAME)))
    assert "not measured on these weights" in section
    rendered, _ = render_card(parse_repo_manifest(text, Path(REPO_MANIFEST_NAME)), CARD)
    # A key with nothing to say is not written at all: an empty
    # `higgs_pace_chars_per_sec:` reads as zero to an audit script.
    assert "higgs_pace_chars_per_sec" not in read_frontmatter(rendered)


# ------------------------------------------------------------------- export


def test_an_export_refuses_to_invent_the_two_bases() -> None:
    """The whole of section 4's honesty. The packaged schema cannot say how a
    pace or a cap was got, and a default here would launder exactly the two
    defects the fields exist to expose."""
    manifest = load_voice("mistborn")
    with pytest.raises(VoiceError) as caught:
        export_manifest(
            manifest,
            pace_basis=None,
            measured_from=None,
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
            max_chars_basis="measured",
            uncertified=False,
        )
    assert "--measured-from" in str(caught.value)


def test_an_exported_manifest_parses_as_a_repo_manifest() -> None:
    """The round trip that makes `export` a bridge rather than a draft."""
    manifest = load_voice("mistborn")
    text, dropped = export_manifest(
        manifest,
        pace_basis="measured",
        measured_from="mb_ha_rvcbed1 ckpt-5368's own ladder, n=51 in the band",
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
        max_chars_basis="measured",
        uncertified=False,
    )
    joined = "\n".join(dropped)
    assert "max_num_seqs = 16" in joined
    assert "memory_bytes_estimate = 19000000000" in joined
    assert "pins.toml [mistborn]" in joined
    assert manifest.spec("cuda-linux").hf_repo in joined


def test_an_exported_file_carries_no_machine_fact() -> None:
    """A converted file cannot make a claim about a box it has never run on —
    which is enforced by the repo parser, so this asserts it there."""
    text, _dropped = export_manifest(
        load_voice("mistborn"),
        pace_basis="measured",
        measured_from="ckpt-5368's own ladder",
        max_chars_basis="measured",
        uncertified=False,
    )
    # ASKED OF THE PARSED DOCUMENT, not of the raw text: the file opens with a
    # comment saying where the machine facts went, and a substring search would
    # be a test of that sentence rather than of the keys.
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
    # And the proof it is not merely absent: the parser refuses each by name.
    parse_repo_manifest(text, Path(REPO_MANIFEST_NAME))


def test_an_uncertified_export_omits_the_pace_table_in_whole() -> None:
    """PHASE18 4.1: absent, never zeroed, never partial."""
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
            max_chars_basis="placeholder",
            uncertified=False,
        )
    assert "--uncertified" in str(caught.value)
    text, _dropped = export_manifest(
        blank,
        pace_basis=None,
        measured_from=None,
        max_chars_basis="placeholder",
        uncertified=True,
    )
    import tomllib

    assert "pace" not in tomllib.loads(text)["voice"]
    repo_manifest = parse_repo_manifest(text, Path(REPO_MANIFEST_NAME))
    assert repo_manifest.pace is None
