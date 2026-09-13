"""The `asr/<id>.toml` loader, and the six manifests this build actually ships.

Two halves. The first asserts that the loader refuses every way a manifest can be
wrong, because an ASR manifest that loads with a key missing is a guard working
from nothing. The second asserts facts about the shipped files themselves — the
pins are full shas, the ids match the filenames, and none of them claims a
backend faster-whisper cannot run on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.asrmodels import (
    ASR_BACKEND_ENGINES,
    AsrManifestError,
    asr_manifests_dir,
    load_all_asr_manifests,
    load_asr_manifest,
    parse_asr_manifest,
)

MODELS = [
    "faster-whisper-base",
    "faster-whisper-distil-large-v3",
    "faster-whisper-large-v3",
    "faster-whisper-medium",
    "faster-whisper-small",
    "faster-whisper-tiny",
]

GOOD = """
[model]
id = "faster-whisper-tiny"
family = "faster-whisper"
parameters_m = 39

[backends.cuda-linux]
engine = "faster-whisper"
hf_repo = "Systran/faster-whisper-tiny"
revision = "d90ca5fe260221311c53c58e660288d3deb8d356"
memory_bytes_estimate = 1686151006
"""


def parse(text: str, model_id: str = "faster-whisper-tiny"):
    return parse_asr_manifest(text, Path(f"{model_id}.toml"), model_id)


# ------------------------------------------------------------ the shipped six


def test_this_build_ships_exactly_six_asr_models() -> None:
    assert sorted(load_all_asr_manifests()) == MODELS


def test_every_shipped_pin_is_a_full_commit_sha() -> None:
    """A branch name is not a pin, and a short sha is not reproducible."""
    for manifest in load_all_asr_manifests().values():
        for spec in manifest.backends.values():
            assert len(spec.revision) == 40
            assert spec.revision == spec.revision.lower()
            int(spec.revision, 16)  # raises if it is not hex


def test_every_shipped_model_is_cuda_linux_only() -> None:
    """CTranslate2 has no Metal backend; there is no mlx-darwin ASR block."""
    for manifest in load_all_asr_manifests().values():
        assert sorted(manifest.backends) == ["cuda-linux"]
        assert manifest.spec("cuda-linux").engine == "faster-whisper"
    assert sorted(ASR_BACKEND_ENGINES) == ["cuda-linux"]


def test_every_shipped_estimate_covers_the_weights() -> None:
    """The estimate is weights plus runtime, so it is never below the weights.

    The weights figures are the `model.bin` sizes the HuggingFace tree API
    reported for these exact revisions on 2026-09-13; the estimates are those
    plus a declared 1.5 GiB. None of it is measured yet and every manifest says
    so — this test only holds the arithmetic to being the arithmetic it claims.
    """
    weights_bytes = {
        "faster-whisper-tiny": 75_538_270,
        "faster-whisper-base": 145_217_532,
        "faster-whisper-small": 483_546_902,
        "faster-whisper-medium": 1_527_906_378,
        "faster-whisper-large-v3": 3_087_284_237,
        "faster-whisper-distil-large-v3": 1_512_927_867,
    }
    runtime = 1024 ** 3 + 512 * 1024 ** 2
    for model_id, size in weights_bytes.items():
        spec = load_asr_manifest(model_id).spec("cuda-linux")
        assert spec.memory_bytes_estimate == size + runtime


def test_the_manifest_directory_is_beside_the_package() -> None:
    assert asr_manifests_dir().name == "asr"
    assert sorted(p.stem for p in asr_manifests_dir().glob("*.toml")) == MODELS


def test_an_unknown_id_names_what_is_shipped() -> None:
    with pytest.raises(AsrManifestError) as caught:
        load_asr_manifest("faster-whisper-enormous")
    assert "faster-whisper-large-v3" in str(caught.value)


def test_the_override_must_be_a_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRUCIBLE_ASR_DIR", "/no/such/place")
    with pytest.raises(AsrManifestError) as caught:
        asr_manifests_dir()
    assert "is not a directory" in str(caught.value)


# ---------------------------------------------------------------- refusals


def test_a_good_manifest_parses() -> None:
    manifest = parse(GOOD)
    assert manifest.id == "faster-whisper-tiny"
    assert manifest.parameters_m == 39
    assert manifest.supports("cuda-linux")
    assert not manifest.supports("mlx-darwin")


def test_a_misspelled_key_is_refused_not_ignored() -> None:
    """The whole reason the loader is strict: a typo must not mean "no estimate"."""
    with pytest.raises(AsrManifestError) as caught:
        parse(GOOD.replace("memory_bytes_estimate", "memory_bytes_estimat"))
    assert "unknown key(s)" in str(caught.value)
    assert "memory_bytes_estimat" in str(caught.value)


def test_a_missing_key_names_itself() -> None:
    with pytest.raises(AsrManifestError) as caught:
        parse(GOOD.replace('family = "faster-whisper"\n', ""))
    assert "missing required key(s) ['family']" in str(caught.value)


def test_an_mlx_block_is_refused_with_the_reason() -> None:
    text = GOOD + """
[backends.mlx-darwin]
engine = "faster-whisper"
hf_repo = "Systran/faster-whisper-tiny"
revision = "d90ca5fe260221311c53c58e660288d3deb8d356"
memory_bytes_estimate = 1686151006
"""
    with pytest.raises(AsrManifestError) as caught:
        parse(text)
    message = str(caught.value)
    assert "not an asr backend" in message
    assert "no Metal backend" in message


def test_a_branch_name_is_not_a_pin() -> None:
    with pytest.raises(AsrManifestError) as caught:
        parse(GOOD.replace("d90ca5fe260221311c53c58e660288d3deb8d356", "main"))
    assert "40-character commit sha" in str(caught.value)


def test_the_id_and_the_filename_are_the_same_thing() -> None:
    with pytest.raises(AsrManifestError) as caught:
        parse_asr_manifest(GOOD, Path("something-else.toml"), "something-else")
    assert "the id and the filename are the same thing" in str(caught.value)


def test_the_wrong_engine_for_the_backend_is_refused() -> None:
    with pytest.raises(AsrManifestError) as caught:
        parse(GOOD.replace('engine = "faster-whisper"', 'engine = "vllm"'))
    assert "does not run asr on cuda-linux" in str(caught.value)


def test_a_model_with_no_backend_block_is_not_a_model() -> None:
    with pytest.raises(AsrManifestError) as caught:
        parse(GOOD.split("[backends.cuda-linux]")[0] + "[backends]\n")
    assert "no backend blocks" in str(caught.value)


def test_a_bool_where_an_int_is_wanted_is_still_wrong() -> None:
    with pytest.raises(AsrManifestError) as caught:
        parse(GOOD.replace("parameters_m = 39", "parameters_m = true"))
    assert "must be int" in str(caught.value)


def test_an_unknown_top_level_table_is_refused() -> None:
    with pytest.raises(AsrManifestError) as caught:
        parse(GOOD + '\n[tuning]\nbeam_size = 5\n')
    assert "unknown top-level table(s)" in str(caught.value)


def test_a_zero_estimate_is_refused() -> None:
    with pytest.raises(AsrManifestError) as caught:
        parse(GOOD.replace("memory_bytes_estimate = 1686151006", "memory_bytes_estimate = 0"))
    assert "must be positive" in str(caught.value)


def test_spec_for_an_absent_backend_names_what_is_declared() -> None:
    manifest = parse(GOOD)
    with pytest.raises(AsrManifestError) as caught:
        manifest.spec("mlx-darwin")
    assert "declares ['cuda-linux']" in str(caught.value)
