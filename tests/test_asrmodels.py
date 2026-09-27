from __future__ import annotations

from pathlib import Path

import pytest

from crucible.asrmodels import (
    ASR_BACKEND_ENGINES,
    ASR_LINEUP,
    AsrManifestError,
    asr_manifests_dir,
    load_all_asr_manifests,
    load_asr_manifest,
    parse_asr_manifest,
)

WHISPER_MODELS = ["whisper-large-v3-turbo", "whisper-tiny"]
QWEN_MODELS = ["qwen3-asr-1.7b"]
MODELS = sorted(WHISPER_MODELS + QWEN_MODELS)

GOOD = """
[model]
id = "whisper-tiny"
family = "whisper"
parameters_m = 39

[backends.cuda-linux]
engine = "faster-whisper"
hf_repo = "Systran/faster-whisper-tiny"
revision = "d90ca5fe260221311c53c58e660288d3deb8d356"
memory_bytes_estimate = 1686151006
"""


MAC_BLOCK = """
[backends.mlx-darwin]
engine = "mlx-whisper"
hf_repo = "mlx-community/whisper-tiny-mlx"
revision = "6caf9c55601caafbe6508a8b0d216bdf4783c4e8"
memory_bytes_estimate = 549418642
"""


def parse(text: str, model_id: str = "whisper-tiny"):
    return parse_asr_manifest(text, Path(f"{model_id}.toml"), model_id)


def test_this_build_ships_exactly_the_three_owen_ruled() -> None:
    assert sorted(load_all_asr_manifests()) == MODELS
    assert set(MODELS) == ASR_LINEUP


def test_every_shipped_pin_is_a_full_commit_sha() -> None:
    for manifest in load_all_asr_manifests().values():
        for spec in manifest.backends.values():
            assert len(spec.revision) == 40
            assert spec.revision == spec.revision.lower()
            int(spec.revision, 16)


def test_every_model_is_one_id_on_both_backends_with_each_backends_engine() -> None:
    assert ASR_BACKEND_ENGINES == {
        "cuda-linux": frozenset({"faster-whisper", "vllm"}),
        "mlx-darwin": frozenset({"mlx-whisper", "mlx-audio"}),
    }
    for model_id in WHISPER_MODELS:
        manifest = load_asr_manifest(model_id)
        assert manifest.family == "whisper"
        assert sorted(manifest.backends) == ["cuda-linux", "mlx-darwin"]
        assert manifest.spec("cuda-linux").engine == "faster-whisper"
        assert manifest.spec("mlx-darwin").engine == "mlx-whisper"
    qwen = load_asr_manifest("qwen3-asr-1.7b")
    assert sorted(qwen.backends) == ["cuda-linux", "mlx-darwin"]
    assert qwen.spec("cuda-linux").engine == "vllm"
    assert qwen.spec("mlx-darwin").engine == "mlx-audio"


def test_the_merged_whispers_kept_every_pin_they_had() -> None:
    expected = {
        ("whisper-large-v3-turbo", "cuda-linux"): (
            "dropbox-dash/faster-whisper-large-v3-turbo",
            "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
        ),
        ("whisper-large-v3-turbo", "mlx-darwin"): (
            "mlx-community/whisper-large-v3-turbo",
            "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb",
        ),
        ("whisper-tiny", "cuda-linux"): (
            "Systran/faster-whisper-tiny",
            "d90ca5fe260221311c53c58e660288d3deb8d356",
        ),
        ("whisper-tiny", "mlx-darwin"): (
            "mlx-community/whisper-tiny-mlx",
            "6caf9c55601caafbe6508a8b0d216bdf4783c4e8",
        ),
    }
    for (model_id, backend), pin in expected.items():
        spec = load_asr_manifest(model_id).spec(backend)
        assert (spec.hf_repo, spec.revision) == pin


def test_the_mac_estimates_are_measured_and_none_is_the_cuda_arithmetic() -> None:
    measured = {
        "whisper-tiny": 549_418_642,
        "whisper-large-v3-turbo": 2_654_916_970,
    }
    repo_bytes = {
        "whisper-tiny": 74_420_620,
        "whisper-large-v3-turbo": 1_613_979_758,
    }
    runtime = 1024 ** 3 + 512 * 1024 ** 2
    for model_id, peak in measured.items():
        manifest = load_asr_manifest(model_id)
        assert manifest.spec("mlx-darwin").memory_bytes_estimate == peak
        assert peak % runtime != 0
        assert peak > repo_bytes[model_id]
        text = manifest.path.read_text(encoding="utf-8")
        assert "MEASURED, on the machine it is for" in text
        assert "mx.get_peak_memory()" in text


def test_every_cuda_whisper_estimate_is_the_weights_plus_the_allowance() -> None:
    weights_bytes = {
        "whisper-tiny": 75_538_270,
        "whisper-large-v3-turbo": 1_617_884_929,
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
        load_asr_manifest("whisper-enormous")
    assert "whisper-large-v3-turbo" in str(caught.value)


def test_the_override_must_be_a_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRUCIBLE_ASR_DIR", "/no/such/place")
    with pytest.raises(AsrManifestError) as caught:
        asr_manifests_dir()
    assert "is not a directory" in str(caught.value)


def test_a_good_manifest_parses() -> None:
    manifest = parse(GOOD)
    assert manifest.id == "whisper-tiny"
    assert manifest.parameters_m == 39
    assert manifest.supports("cuda-linux")
    assert not manifest.supports("mlx-darwin")


def test_a_misspelled_key_is_refused_not_ignored() -> None:
    with pytest.raises(AsrManifestError) as caught:
        parse(GOOD.replace("memory_bytes_estimate", "memory_bytes_estimat"))
    assert "unknown key(s)" in str(caught.value)
    assert "memory_bytes_estimat" in str(caught.value)


def test_a_missing_key_names_itself() -> None:
    with pytest.raises(AsrManifestError) as caught:
        parse(GOOD.replace('family = "whisper"\n', ""))
    assert "missing required key(s) ['family']" in str(caught.value)


def test_faster_whisper_on_the_mac_is_refused_by_name() -> None:
    text = GOOD + MAC_BLOCK.replace('engine = "mlx-whisper"', 'engine = "faster-whisper"')
    with pytest.raises(AsrManifestError) as caught:
        parse(text)
    message = str(caught.value)
    assert "does not run asr on mlx-darwin" in message
    assert "no Metal backend" in message


def test_a_whisper_id_carries_both_backends_with_different_conversions() -> None:
    manifest = parse(GOOD + MAC_BLOCK)
    assert sorted(manifest.backends) == ["cuda-linux", "mlx-darwin"]
    assert manifest.spec("mlx-darwin").engine == "mlx-whisper"
    assert (
        manifest.spec("cuda-linux").hf_repo != manifest.spec("mlx-darwin").hf_repo
    )


def test_an_engine_of_another_family_is_refused() -> None:
    qwen_mac = """
[backends.mlx-darwin]
engine = "mlx-audio"
hf_repo = "Qwen/Qwen3-ASR-1.7B"
revision = "7278e1e70fe206f11671096ffdd38061171dd6e5"
memory_bytes_estimate = 7726809000
dtype = "bfloat16"
aligner = "qwen3-aligner"
max_batch = 1
max_new_tokens = 4096
"""
    with pytest.raises(AsrManifestError) as caught:
        parse(GOOD + qwen_mac)
    message = str(caught.value)
    assert "runs the 'qwen3-asr' family" in message
    assert "family is 'whisper'" in message


def test_a_family_no_engine_runs_is_refused() -> None:
    text = GOOD.replace('family = "whisper"', 'family = "faster-whisper"')
    with pytest.raises(AsrManifestError) as caught:
        parse(text)
    assert "runs the 'whisper' family" in str(caught.value)


def test_an_id_that_does_not_name_its_family_is_refused() -> None:
    text = GOOD.replace('id = "whisper-tiny"', 'id = "faster-whisper-tiny"')
    with pytest.raises(AsrManifestError) as caught:
        parse(text, "faster-whisper-tiny")
    assert "must begin with its family ('whisper')" in str(caught.value)


def test_a_windows_block_is_refused() -> None:
    with pytest.raises(AsrManifestError) as caught:
        parse(GOOD.replace("[backends.cuda-linux]", "[backends.llama-windows]"))
    assert "not an asr backend" in str(caught.value)
    assert "Windows is never a backend" in str(caught.value)


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
        parse(GOOD.replace('engine = "faster-whisper"', 'engine = "mlx-whisper"'))
    assert "does not run asr on cuda-linux" in str(caught.value)
    assert "not two recipes for one thing" in str(caught.value)


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
