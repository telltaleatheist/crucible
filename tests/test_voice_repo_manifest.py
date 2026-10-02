from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crucible.voicecatalog import load_all_voices, load_voice
from crucible.voicerepo import (
    REPO_MANIFEST_NAME,
    Pin,
    load_pins,
    packaged_pins_path,
    parse_repo_manifest,
    remove_home_pin,
    write_home_pin,
)
from crucible.voices import VOICES_DIR_ENV, VoiceError, voice_document

SHA = "a" * 40
OTHER_SHA = "b" * 40
REPO = "owenmorgan/mistborn-higgs-v3"

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
max_chars       = 800
max_chars_basis = "placeholder"
sampling        = { temperature = 0.8, top_p = 0.95, top_k = 50 }

[[voice.takes]]

[[voice.takes]]
temperature = 0.7
reason = "a retake must not reuse the settings that produced the problem"
"""

CONFIG = """
[tts.higgs-v3]
memory_bytes_estimate = 19_000_000_000
estimate_basis = "declared"
estimate_note = "SGLang-Omni's configured reservation on this arm, 2026-09-05."
max_num_seqs = 16
max_num_seqs_note = "vllm-omni's own stage-0 value, measured at 0.35 + 0.10."
"""


MEASURED_LINE = (
    'measured_from      = "mb_hp_rvcbed1 ckpt-4257, n=51 in the 500-800 band"'
)
INHERITED = (
    "ow_v8_rvcbed1 ckpt-966; these weights have no ladder yet, re-measurement owed"
)


def parse(text: str):
    return parse_repo_manifest(text, Path(REPO_MANIFEST_NAME))


def refused(text: str) -> str:
    with pytest.raises(VoiceError) as caught:
        parse(text)
    return str(caught.value)


def swap(old: str, new: str) -> str:
    assert old in GOOD, f"the good manifest does not contain {old!r}"
    return GOOD.replace(old, new)


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CRUCIBLE_HOME", str(tmp_path))
    monkeypatch.delenv(VOICES_DIR_ENV, raising=False)
    (tmp_path / "config.toml").write_text(CONFIG, encoding="utf-8")
    return tmp_path


PINNED_ID = "nightingale"


def a_pin(home: Path, voice_id: str = PINNED_ID, revision: str = SHA) -> None:
    (home / "voices").mkdir(parents=True, exist_ok=True)
    (home / "voices" / "pins.toml").write_text(
        f'[{voice_id}]\nhf_repo = "{REPO}"\nrevision = "{revision}"\n',
        encoding="utf-8",
    )


def a_cached_manifest(home: Path, text: str = GOOD, revision: str = SHA) -> Path:
    path = (
        home
        / "voice-manifests"
        / REPO.replace("/", "--")
        / revision
        / REPO_MANIFEST_NAME
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_the_good_repo_manifest_parses() -> None:
    repo = parse(GOOD)
    assert repo.schema == 1
    assert repo.voice["display"] == "Mistborn"
    assert repo.pace_basis == "measured"
    assert repo.measured_from.startswith("mb_hp_rvcbed1")
    assert sorted(repo.arms) == ["cuda-linux", "mlx-darwin"]
    assert repo.max_chars_basis == {
        "cuda-linux": "measured",
        "mlx-darwin": "placeholder",
    }
    assert "basis" not in repo.pace and "measured_from" not in repo.pace


def test_a_manifest_with_no_schema_is_refused_by_name() -> None:
    message = refused(GOOD.replace("schema = 1\n", ""))
    assert "voice_manifest_schema" in message
    assert "states no `schema`" in message


def test_an_unknown_schema_is_refused_rather_than_half_read() -> None:
    message = refused(swap("schema = 1", "schema = 2"))
    assert "voice_manifest_schema 2" in message
    assert "will not read half of a newer one" in message


def test_a_schema_that_is_not_a_number_is_refused() -> None:
    assert "voice_manifest_schema" in refused(swap("schema = 1", 'schema = "1"'))


@pytest.mark.parametrize(
    "line, key",
    [
        ('id = "mistborn"', "id"),
        (f'hf_repo = "{REPO}"', "hf_repo"),
        (f'revision = "{SHA}"', "revision"),
        ("memory_bytes_estimate = 19_000_000_000", "memory_bytes_estimate"),
        ('estimate_basis = "declared"', "estimate_basis"),
        ('estimate_note = "somebody watched a card"', "estimate_note"),
    ],
)
def test_a_machine_fact_in_the_repo_schema_is_refused_by_name(
    line: str, key: str
) -> None:
    message = refused(swap('display         = "Mistborn"', f'display = "M"\n{line}'))
    assert f"carries {key}" in message
    assert "may not state" in message


def test_the_serving_table_is_refused_and_says_where_it_went() -> None:
    message = refused(
        swap("[voice.arms.cuda-linux]", "[voice.serving]\nmax_num_seqs = 16\n\n[voice.arms.cuda-linux]")
    )
    assert "carries serving" in message
    assert "[tts.<engine>]" in message


def test_the_word_backends_is_refused_so_the_two_schemas_cannot_be_confused() -> None:
    message = refused(swap("[voice.arms.cuda-linux]", "[voice.backends]\nx = 1\n\n[voice.arms.cuda-linux]"))
    assert "carries backends" in message
    assert "[voice.arms.<backend>]" in message


def test_a_pace_table_without_a_basis_is_refused() -> None:
    message = refused(swap('basis              = "measured"\n', ""))
    assert "states no basis" in message
    assert "16.64" in message and "15.91" in message


def test_an_unknown_pace_basis_is_refused() -> None:
    message = refused(swap('basis              = "measured"', 'basis = "vibes"'))
    assert "basis 'vibes' is not one of" in message


def test_a_measured_pace_owes_its_prose() -> None:
    message = refused(
        swap('measured_from      = "mb_hp_rvcbed1 ckpt-4257, n=51 in the 500-800 band"\n', "")
    )
    assert "is 'measured' and there is no measured_from" in message


def test_an_inherited_pace_owes_its_own_sentence() -> None:
    repo = parse(
        swap('basis              = "measured"', 'basis = "inherited"').replace(
            MEASURED_LINE, f'inherited_from = "{INHERITED}"'
        )
    )
    assert repo.pace_basis == "inherited"
    assert repo.inherited_from == INHERITED
    assert repo.measured_from is None


def test_an_inherited_pace_with_no_inherited_from_is_refused() -> None:
    message = refused(
        swap('basis              = "measured"', 'basis = "inherited"').replace(
            MEASURED_LINE + "\n", ""
        )
    )
    assert "basis is 'inherited' and there is no inherited_from" in message
    assert "16.64 onto weights that measured 15.91" in message


def test_inherited_from_on_a_measured_pace_is_refused() -> None:
    message = refused(
        swap(MEASURED_LINE, MEASURED_LINE + f'\ninherited_from = "{INHERITED}"')
    )
    assert "basis is 'measured' and it also carries inherited_from" in message
    assert "owes exactly its own sentence" in message


def test_measured_from_on_an_inherited_pace_is_refused() -> None:
    message = refused(
        swap('basis              = "measured"', 'basis = "inherited"').replace(
            MEASURED_LINE, MEASURED_LINE + f'\ninherited_from = "{INHERITED}"'
        )
    )
    assert "basis is 'inherited' and it also carries measured_from" in message


def test_an_inherited_pace_rides_on_the_row(host: Path) -> None:
    a_pin(host)
    a_cached_manifest(
        host,
        swap('basis              = "measured"', 'basis = "inherited"').replace(
            MEASURED_LINE, f'inherited_from = "{INHERITED}"'
        ),
    )
    voice = load_all_voices()[PINNED_ID]
    assert voice.pace_basis == "inherited"
    assert voice.inherited_from == INHERITED
    assert voice.to_dict()["inherited_from"] == INHERITED


def test_a_measured_pace_reports_a_null_inherited_from(host: Path) -> None:
    a_pin(host)
    a_cached_manifest(host)
    voice = load_all_voices()[PINNED_ID]
    assert voice.pace_basis == "measured"
    assert voice.inherited_from is None


def test_a_measured_pace_keeps_where_it_was_measured_and_saving_it_home_says_so(
    host: Path,
) -> None:
    a_pin(host)
    a_cached_manifest(host)
    voice = load_all_voices()[PINNED_ID]
    measured = "mb_hp_rvcbed1 ckpt-4257, n=51 in the 500-800 band"
    assert voice.measured_from == measured
    assert voice.to_dict()["measured_from"] == measured
    _, not_carried = voice_document(voice)
    assert f"measured_from = {measured!r}" in not_carried


def test_a_voice_may_omit_its_pace_table_in_whole(host: Path) -> None:
    text = GOOD[: GOOD.index("[voice.pace]")] + GOOD[GOOD.index("[voice.arms.cuda-linux]"):]
    repo = parse(text)
    assert repo.pace is None and repo.pace_basis is None
    a_pin(host)
    a_cached_manifest(host, text)
    voice = load_all_voices()[PINNED_ID]
    assert voice.pace.pace_chars_per_sec is None
    assert voice.pace_basis is None


def test_a_cap_without_a_basis_is_refused() -> None:
    message = refused(swap('max_chars_basis = "measured"\n', ""))
    assert "states max_chars 800 and no max_chars_basis" in message


def test_a_basis_without_a_cap_is_refused_as_the_leftover_it_is() -> None:
    message = refused(
        GOOD.replace(
            'max_chars       = 800\nmax_chars_basis = "measured"',
            'max_chars_basis = "measured"',
        )
    )
    assert "states max_chars_basis 'measured' and no max_chars" in message


UNCAPPED_ARMS = (
    GOOD.replace('max_chars       = 800\nmax_chars_basis = "measured"\n', "")
    .replace('max_chars       = 800\nmax_chars_basis = "placeholder"\n', "")
)


def test_an_arm_may_state_no_cap_at_all(host: Path) -> None:
    repo = parse(UNCAPPED_ARMS)
    assert repo.max_chars_basis == {"cuda-linux": None, "mlx-darwin": None}
    a_pin(host)
    a_cached_manifest(host, UNCAPPED_ARMS)
    voice = load_all_voices()[PINNED_ID]
    for arm in ("cuda-linux", "mlx-darwin"):
        assert voice.spec(arm).max_chars is None, arm
        assert voice.spec(arm).max_chars_basis is None, arm


def test_an_unknown_max_chars_basis_is_refused() -> None:
    message = refused(swap('max_chars_basis = "measured"', 'max_chars_basis = "ok"'))
    assert "max_chars_basis 'ok' is not one of" in message
    assert "higgs_max_chars_mlx" in message


def test_the_existing_pace_rules_apply_verbatim(host: Path) -> None:
    a_pin(host)
    a_cached_manifest(host, swap("min_chars_per_sec  = 10.58", "min_chars_per_sec = 13.31"))
    assert PINNED_ID not in load_all_voices()
    with pytest.raises(VoiceError) as caught:
        load_voice(PINNED_ID)
    assert "not symmetric" in str(caught.value)


def test_a_band_above_the_arms_cap_is_refused_verbatim(host: Path) -> None:
    a_pin(host)
    a_cached_manifest(host, swap("safe_max_chars     = 800", "safe_max_chars = 900"))
    assert PINNED_ID not in load_all_voices()
    with pytest.raises(VoiceError) as caught:
        load_voice(PINNED_ID)
    assert "may never exceed the arm's cap" in str(caught.value)


def test_a_deviating_sampling_still_owes_a_reason(host: Path) -> None:
    a_pin(host)
    a_cached_manifest(
        host,
        swap(
            "[voice.arms.cuda-linux]\nmax_chars       = 800\nmax_chars_basis = \"measured\"\n"
            "sampling        = { temperature = 0.8, top_p = 0.95, top_k = 50 }",
            "[voice.arms.cuda-linux]\nmax_chars = 800\nmax_chars_basis = \"measured\"\n"
            "sampling = { temperature = 0.6, top_p = 0.95, top_k = 50 }",
        ),
    )
    assert PINNED_ID not in load_all_voices()
    with pytest.raises(VoiceError) as caught:
        load_voice(PINNED_ID)
    assert "carries no sampling_reason" in str(caught.value)


def test_a_pin_is_read_and_the_voice_is_served(host: Path) -> None:
    a_pin(host)
    a_cached_manifest(host)
    voice = load_all_voices()[PINNED_ID]
    assert voice.manifest_source == "repo"
    assert voice.display == "Mistborn"
    assert voice.pace_basis == "measured"
    assert voice.pace.safe_min_chars == 500
    for arm in ("cuda-linux", "mlx-darwin"):
        spec = voice.spec(arm)
        assert spec.hf_repo == REPO
        assert spec.revision == SHA
        assert spec.max_chars == 800
    assert voice.spec("cuda-linux").max_chars_basis == "measured"
    assert voice.spec("mlx-darwin").max_chars_basis == "placeholder"


def test_the_machine_table_fills_the_facts_the_manifest_may_not_state(
    host: Path,
) -> None:
    a_pin(host)
    a_cached_manifest(host)
    voice = load_all_voices()[PINNED_ID]
    assert voice.serving.max_num_seqs == 16
    for arm in voice.backends:
        spec = voice.spec(arm)
        assert spec.memory_bytes_estimate == 19_000_000_000
        assert spec.estimate_basis == "declared"
        assert spec.estimate_note


def test_a_box_with_no_footprint_refuses_the_voice_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_HOME", str(tmp_path))
    monkeypatch.delenv(VOICES_DIR_ENV, raising=False)
    (tmp_path / "config.toml").write_text("[server]\nname = 'x'\n", encoding="utf-8")
    a_pin(tmp_path)
    a_cached_manifest(tmp_path)
    assert PINNED_ID not in load_all_voices()
    with pytest.raises(VoiceError) as caught:
        load_voice(PINNED_ID)
    message = str(caught.value)
    assert "engine_footprint_unset" in message
    assert "[tts.higgs-v3]" in message
    assert "nothing here is defaulted" in message


def test_a_revision_with_no_manifest_is_refused_by_name_and_not_served(
    host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    a_pin(host)

    def missing(**_kwargs: Any) -> str:
        raise EntryNotFoundError("404")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    assert PINNED_ID not in load_all_voices()
    with pytest.raises(VoiceError) as caught:
        load_voice(PINNED_ID)
    message = str(caught.value)
    assert "voice_manifest_missing" in message
    assert "not read from any other source" in message


def test_a_hub_that_is_down_is_not_reported_as_a_missing_manifest(
    host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    a_pin(host)

    def broken(**_kwargs: Any) -> str:
        raise TimeoutError("the hub did not answer")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", broken)
    assert PINNED_ID not in load_all_voices()
    with pytest.raises(VoiceError) as caught:
        load_voice(PINNED_ID)
    message = str(caught.value)
    assert "voice_manifest_unreadable" in message
    assert "voice_manifest_missing" not in message


def test_the_manifest_is_fetched_alone_so_an_uninstalled_voice_still_lists(
    host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    asked: list[tuple[str, str, str]] = []

    def download(*, repo_id: str, filename: str, revision: str, local_dir: str, **_k: Any) -> str:
        asked.append((repo_id, filename, revision))
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(GOOD, encoding="utf-8")
        return str(target)

    a_pin(host)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert load_all_voices()[PINNED_ID].display == "Mistborn"
    for_this_pin = [question for question in asked if question[2] == SHA]
    assert for_this_pin == [(REPO, REPO_MANIFEST_NAME, SHA)]
    before = len(asked)
    assert load_all_voices()[PINNED_ID].display == "Mistborn"
    assert len(asked) == before


def test_the_pulled_snapshot_answers_before_the_cache(host: Path) -> None:
    snapshot = host / "voices" / PINNED_ID / "cuda-linux"
    snapshot.mkdir(parents=True)
    (snapshot / "crucible-pull.json").write_text(
        json.dumps({"hf_repo": REPO, "revision": SHA}), encoding="utf-8"
    )
    (snapshot / REPO_MANIFEST_NAME).write_text(
        GOOD.replace('display         = "Mistborn"', 'display = "From the snapshot"'),
        encoding="utf-8",
    )
    a_pin(host)
    a_cached_manifest(host)
    assert load_all_voices()[PINNED_ID].display == "From the snapshot"


def test_a_snapshot_at_the_old_revision_is_not_read(host: Path) -> None:
    snapshot = host / "voices" / PINNED_ID / "cuda-linux"
    snapshot.mkdir(parents=True)
    (snapshot / "crucible-pull.json").write_text(
        json.dumps({"hf_repo": REPO, "revision": OTHER_SHA}), encoding="utf-8"
    )
    (snapshot / REPO_MANIFEST_NAME).write_text(
        GOOD.replace('display         = "Mistborn"', 'display = "Stale"'),
        encoding="utf-8",
    )
    a_pin(host)
    a_cached_manifest(host)
    assert load_all_voices()[PINNED_ID].display == "Mistborn"


def test_pulled_weights_with_no_manifest_beside_them_are_refused(host: Path) -> None:
    snapshot = host / "voices" / PINNED_ID / "cuda-linux"
    snapshot.mkdir(parents=True)
    (snapshot / "crucible-pull.json").write_text(
        json.dumps({"hf_repo": REPO, "revision": SHA}), encoding="utf-8"
    )
    a_pin(host)
    a_cached_manifest(host)
    assert PINNED_ID not in load_all_voices()
    with pytest.raises(VoiceError) as caught:
        load_voice(PINNED_ID)
    assert "voice_manifest_missing" in str(caught.value)


def test_a_home_pin_wins_over_the_packaged_one(host: Path) -> None:
    packaged = load_pins()
    assert packaged["mistborn"].path == packaged_pins_path()
    assert packaged["mistborn"].revision != SHA
    write_home_pin("mistborn", REPO, SHA)
    assert load_pins()["mistborn"].revision == SHA
    assert load_pins()["mistborn"].path == host / "voices" / "pins.toml"


def test_a_pin_row_needs_both_halves(host: Path) -> None:
    (host / "voices").mkdir(parents=True)
    (host / "voices" / "pins.toml").write_text(
        f'[mistborn]\nhf_repo = "{REPO}"\n', encoding="utf-8"
    )
    with pytest.raises(VoiceError) as caught:
        load_pins()
    assert "states neither revision nor ref" in str(caught.value)


def test_a_branch_name_is_not_a_pin(host: Path) -> None:
    (host / "voices").mkdir(parents=True)
    (host / "voices" / "pins.toml").write_text(
        f'[mistborn]\nhf_repo = "{REPO}"\nrevision = "main"\n', encoding="utf-8"
    )
    with pytest.raises(VoiceError) as caught:
        load_pins()
    assert "must be a full 40-character commit sha" in str(caught.value)


def test_an_unknown_key_in_a_pin_row_is_refused(host: Path) -> None:
    (host / "voices").mkdir(parents=True)
    (host / "voices" / "pins.toml").write_text(
        f'[mistborn]\nhf_repo = "{REPO}"\nrevision = "{SHA}"\npace = 13.3\n',
        encoding="utf-8",
    )
    with pytest.raises(VoiceError) as caught:
        load_pins()
    assert "unknown key(s) ['pace']" in str(caught.value)


def test_removing_a_home_pin_says_whether_one_went(host: Path) -> None:
    assert remove_home_pin("mistborn") is False
    write_home_pin("mistborn", REPO, SHA)
    assert remove_home_pin("mistborn") is True
    assert load_pins()["mistborn"].path == packaged_pins_path()


def test_pins_toml_is_not_read_as_a_voice_called_pins(host: Path) -> None:
    write_home_pin(PINNED_ID, REPO, SHA)
    a_cached_manifest(host)
    assert "pins" not in load_all_voices()


CATALOG: dict[str, tuple[str, int, tuple[int, int] | None]] = {
    "deathstalker": ("checkpoint", 800, (400, 700)),
    "mistborn": ("checkpoint", 800, (300, 800)),
    "owen": ("checkpoint", 800, (600, 800)),
    "thirdreich": ("checkpoint", 1000, (500, 700)),
    "sigma": ("checkpoint", 1000, (400, 800)),
}


@pytest.mark.parametrize("voice_id", sorted(CATALOG))
def test_a_fake_repo_carries_the_catalog_numbers_through(
    host: Path, voice_id: str
) -> None:
    kind, cap, band = CATALOG[voice_id]
    packaged = load_voice(voice_id)
    text = GOOD
    text = text.replace('kind            = "checkpoint"', f'kind = "{kind}"')
    text = text.replace("max_chars       = 800", f"max_chars = {cap}")
    text = text.replace("safe_min_chars     = 500", f"safe_min_chars = {band[0]}")
    text = text.replace("safe_max_chars     = 800", f"safe_max_chars = {band[1]}")
    text = text.replace(
        "pace_chars_per_sec = 13.76",
        f"pace_chars_per_sec = {packaged.pace.pace_chars_per_sec}",
    )
    text = text.replace(
        "max_chars_per_sec  = 17.89",
        f"max_chars_per_sec = {packaged.pace.max_chars_per_sec}",
    )
    text = text.replace(
        "min_chars_per_sec  = 10.58",
        f"min_chars_per_sec = {packaged.pace.min_chars_per_sec}",
    )
    a_pin(host, voice_id="from-the-repo")
    path = (
        host / "voice-manifests" / REPO.replace("/", "--") / SHA / REPO_MANIFEST_NAME
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

    served = load_all_voices()["from-the-repo"]
    assert served.kind == kind
    assert served.narrator_engine == "higgs-v3"
    assert served.sample_rate == 24000
    assert sorted(served.backends) == ["cuda-linux", "mlx-darwin"]
    for arm in served.backends:
        assert served.spec(arm).max_chars == cap
        assert served.spec(arm).sampling == {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
        }
    assert (served.pace.safe_min_chars, served.pace.safe_max_chars) == band
    assert served.pace.pace_chars_per_sec == packaged.pace.pace_chars_per_sec
    assert len(served.takes) == 2


def test_the_engine_rows_report_themselves_as_the_engines(host: Path) -> None:
    served = load_all_voices()
    for voice_id in ("higgs-default", "zeroshot"):
        assert served[voice_id].manifest_source == "engine"
        assert served[voice_id].pace.pace_chars_per_sec is None
        assert served[voice_id].pace_basis is None


def a_config(home: Path, body: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.toml").write_text(body, encoding="utf-8")


def footprint_refused(home: Path, body: str) -> str:
    from crucible.config import ConfigError, tts_engine_footprints

    a_config(home, body)
    with pytest.raises(ConfigError) as caught:
        tts_engine_footprints(home)
    return str(caught.value)


def test_the_machine_table_is_read(tmp_path: Path) -> None:
    from crucible.config import tts_engine_footprints

    a_config(tmp_path, CONFIG)
    found = tts_engine_footprints(tmp_path)["higgs-v3"]
    assert found.memory_bytes_estimate == 19_000_000_000
    assert found.estimate_basis == "declared"
    assert found.max_num_seqs == 16


def test_a_footprint_for_an_engine_nothing_serves_is_refused(tmp_path: Path) -> None:
    message = footprint_refused(tmp_path, CONFIG.replace("higgs-v3", "orpheus"))
    assert "'orpheus' is not one of narrator's engines" in message


def test_a_declared_footprint_owes_its_note(tmp_path: Path) -> None:
    message = footprint_refused(
        tmp_path,
        CONFIG.replace(
            'estimate_note = "SGLang-Omni\'s configured reservation on this arm, 2026-09-05."\n',
            "",
        ),
    )
    assert "estimate_basis is 'declared' and there is no estimate_note" in message


def test_a_measured_footprint_may_not_carry_a_note(tmp_path: Path) -> None:
    message = footprint_refused(
        tmp_path, CONFIG.replace('estimate_basis = "declared"', 'estimate_basis = "measured"')
    )
    assert "is 'measured' and it also carries an estimate_note" in message


def test_a_serving_width_with_no_note_is_refused(tmp_path: Path) -> None:
    message = footprint_refused(
        tmp_path, CONFIG.replace('max_num_seqs_note = "vllm-omni\'s own stage-0 value, measured at 0.35 + 0.10."', 'max_num_seqs_note = "  "')
    )
    assert "max_num_seqs carries no note" in message
    assert "measured at 64" in message


MACHINE_LEVERS = """mem_fraction = 0.48
mem_fraction_note = "0.48 + width 4 is ~20 GB; 0.55 measured 24.0-24.1 GB here and WDDM then pages to host RAM"
context_length = 8192
context_length_note = "4096 holds ~2,000 chars and this bank tops at 2,008, so the top rungs truncate on the context"
"""


def test_the_machine_table_may_state_the_two_serving_levers(tmp_path: Path) -> None:
    from crucible.config import tts_engine_footprints

    a_config(tmp_path, CONFIG + MACHINE_LEVERS)
    found = tts_engine_footprints(tmp_path)["higgs-v3"]
    assert found.mem_fraction == 0.48
    assert found.context_length == 8192
    assert "24 GB card" in found.mem_fraction_note or "24.0-24.1 GB" in found.mem_fraction_note
    assert "2,008" in found.context_length_note


def test_a_repo_voice_inherits_the_machines_serving_levers(
    host: Path, tmp_path: Path
) -> None:
    a_pin(host)
    a_cached_manifest(host, GOOD)
    a_config(host, CONFIG + MACHINE_LEVERS)
    voice = load_all_voices()[PINNED_ID]
    assert voice.serving.mem_fraction == 0.48
    assert voice.serving.context_length == 8192


def test_a_machine_lever_with_no_note_is_refused(tmp_path: Path) -> None:
    message = footprint_refused(
        tmp_path, CONFIG + "mem_fraction = 0.48\n"
    )
    assert "mem_fraction carries no note" in message


def test_a_machine_note_with_no_number_is_refused(tmp_path: Path) -> None:
    message = footprint_refused(
        tmp_path, CONFIG + 'context_length_note = "x"\n'
    )
    assert "states context_length_note and no context_length" in message


def test_a_machine_mem_fraction_outside_zero_to_one_is_refused(tmp_path: Path) -> None:
    message = footprint_refused(
        tmp_path, CONFIG + 'mem_fraction = 1.2\nmem_fraction_note = "x"\n'
    )
    assert "must be a fraction in (0, 1)" in message


def test_an_unknown_key_in_the_machine_table_is_refused(tmp_path: Path) -> None:
    message = footprint_refused(tmp_path, CONFIG + "\nmax_chars = 800\n")
    assert "unknown key(s) ['max_chars']" in message


def test_init_writes_this_box_s_declared_numbers(tmp_path: Path) -> None:
    from crucible.config import load_config, write_config
    from crucible.narratorengines import declared_tts_footprints

    for backend_kind, estimate in (
        ("cuda-linux", 19_000_000_000),
        ("mlx-darwin", 12_133_000_000),
    ):
        home = tmp_path / backend_kind
        write_config(
            home,
            name="crucible@test",
            host="127.0.0.1",
            port=7100,
            token="t",
            backend_kind=backend_kind,
            enable_echo=True,
            enable_llm=False,
            enable_asr=False,
            enable_tts=True,
            enable_align=False,
            enable_rvc=False,
            desktop_allowance_bytes=3 * 1024 ** 3,
            enable_denoise=False,
            retention_days=7,
            desktop_allowance_basis="stated",
            desktop_allowance_note="",
            tts_engines=declared_tts_footprints(backend_kind),
        )
        found = load_config(home).engine_footprint("higgs-v3")
        assert found is not None
        assert found.memory_bytes_estimate == estimate
        assert found.estimate_basis == "declared"
        assert found.max_num_seqs == 16
        assert "2026-09-05" in found.estimate_note
        assert "vllm-omni" in found.max_num_seqs_note


def test_a_backend_that_serves_no_narrator_engine_writes_no_table() -> None:
    from crucible.narratorengines import declared_tts_footprints

    assert declared_tts_footprints("llama-windows") == ()


@pytest.mark.parametrize("voice_id", sorted(CATALOG))
def test_a_served_voice_exported_and_merged_is_the_same_voice(
    host: Path, voice_id: str
) -> None:
    from crucible.config import tts_engine_footprints
    from crucible.voicecard import export_manifest
    from crucible.voicerepo import merge, parse_repo_manifest

    packaged = load_voice(voice_id)
    text, dropped = export_manifest(
        packaged,
        pace_basis="measured",
        measured_from=f"{voice_id}'s own length ladder, per its packaged manifest",
        inherited_from=None,
        max_chars_basis="measured",
        uncertified=False,
    )
    repo = parse_repo_manifest(text, Path(REPO_MANIFEST_NAME))
    footprint = tts_engine_footprints(host)["higgs-v3"]
    merged = merge(
        repo,
        Pin(id=voice_id, hf_repo=REPO, revision=SHA, path=host / "voices" / "pins.toml"),
        footprint,
    )

    assert merged.id == packaged.id
    assert merged.display == packaged.display
    assert merged.kind == packaged.kind
    assert merged.narrator_engine == packaged.narrator_engine
    assert merged.language == packaged.language
    assert merged.sample_rate == packaged.sample_rate
    assert merged.pace.to_dict() == packaged.pace.to_dict()
    assert [t.to_dict() for t in merged.takes] == [
        t.to_dict() for t in packaged.takes
    ]
    assert sorted(merged.backends) == sorted(packaged.backends)
    for arm in packaged.backends:
        was, now = packaged.spec(arm), merged.spec(arm)
        assert now.max_chars == was.max_chars
        assert now.sampling == was.sampling
        assert now.sampling_reason == was.sampling_reason
        assert now.clips == was.clips
        assert (now.hf_repo, now.revision) == (REPO, SHA)

    assert merged.serving.max_num_seqs == packaged.serving.max_num_seqs
    for arm in merged.backends:
        assert merged.spec(arm).memory_bytes_estimate == footprint.memory_bytes_estimate
        assert merged.spec(arm).estimate_note == footprint.estimate_note
    assert any("config.toml [tts.higgs-v3]" in line for line in dropped)

    assert packaged.manifest_source == "repo"
    assert merged.manifest_source == "repo"
    assert merged.pace_basis == "measured"
    for arm in merged.backends:
        assert merged.spec(arm).max_chars_basis == "measured"


def test_a_missing_config_is_a_config_error_and_not_an_empty_footprint_table(
    tmp_path: Path,
) -> None:
    from crucible.config import ConfigError, tts_engine_footprints

    with pytest.raises(ConfigError) as caught:
        tts_engine_footprints(tmp_path / "nowhere")
    assert "`crucible init`" in str(caught.value)


def test_a_pin_with_no_config_is_one_unserved_voice_with_the_config_s_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crucible import voicerepo

    monkeypatch.setenv("CRUCIBLE_HOME", str(tmp_path))
    monkeypatch.delenv(VOICES_DIR_ENV, raising=False)
    a_pin(tmp_path)
    a_cached_manifest(tmp_path)
    served, refused = voicerepo.load_pinned()
    assert PINNED_ID not in served
    _pin, why = refused[PINNED_ID]
    assert why.startswith("config_unreadable:")
    assert "`crucible init`" in why


def test_the_footprint_advice_names_the_command_that_writes_the_table(
    host: Path,
) -> None:
    from crucible.voicerepo import footprint_unset

    said = footprint_unset("higgs-v3")
    assert "`crucible init --force`" in said
    assert "by hand" not in said
    other = footprint_unset("some-new-engine")
    assert str(host / "config.toml") in other
    assert "memory_bytes_estimate" in other


def test_an_unserved_pin_s_reason_is_logged_once_and_carried(
    host: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    from crucible import voicerepo

    a_pin(host)

    def down(_pin: Pin) -> None:
        raise VoiceError(
            "voice_manifest_unreadable: could not fetch crucible-voice.toml from "
            f"{REPO}@{SHA[:12]}: ConnectionError: hub is down\nsecond line"
        )

    monkeypatch.setattr(voicerepo, "voice_for_pin", down)
    voicerepo._REFUSALS_SAID.clear()
    with caplog.at_level(logging.WARNING, logger="crucible.voicerepo"):
        served, refused = voicerepo.load_pinned()
        voicerepo.load_pinned()
    assert served == {}
    assert refused[PINNED_ID][1].startswith("voice_manifest_unreadable: could not fetch")
    said = [
        r.getMessage()
        for r in caplog.records
        if r.name == "crucible.voicerepo" and f"voice {PINNED_ID!r}" in r.getMessage()
    ]
    assert len(said) == 1, said
    assert "hub is down second line" in said[0]
    assert REPO in said[0]
