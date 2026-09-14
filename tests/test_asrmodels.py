"""The `asr/<id>.toml` loader, and the thirteen manifests this build ships.

Two halves. The first asserts that the loader refuses every way a manifest can be
wrong, because an ASR manifest that loads with a key missing is a guard working
from nothing. The second asserts facts about the shipped files themselves — the
pins are full shas, the ids match the filenames, and neither engine's ids ever
name the other engine's backend.

**Two engines, thirteen ids, and no id shared.** faster-whisper is CTranslate2
and has no Metal backend; mlx-whisper is MLX and has no CUDA one. They convert
the same original whisper checkpoints to different bytes at different
quantisations and they will disagree about a hard passage, so the loader
enforces the id prefix rather than trusting a reviewer to notice — a transcript
records the id and nothing else about what produced it.
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

CUDA_MODELS = [
    "faster-whisper-base",
    "faster-whisper-distil-large-v3",
    "faster-whisper-large-v3",
    "faster-whisper-medium",
    "faster-whisper-small",
    "faster-whisper-tiny",
]
MAC_MODELS = [
    "mlx-whisper-base",
    "mlx-whisper-distil-large-v3",
    "mlx-whisper-large-v3",
    "mlx-whisper-large-v3-turbo",
    "mlx-whisper-medium",
    "mlx-whisper-small",
    "mlx-whisper-tiny",
]
MODELS = sorted(CUDA_MODELS + MAC_MODELS)

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


MAC_GOOD = """
[model]
id = "mlx-whisper-tiny"
family = "mlx-whisper"
parameters_m = 39

[backends.mlx-darwin]
engine = "mlx-whisper"
hf_repo = "mlx-community/whisper-tiny-mlx"
revision = "6caf9c55601caafbe6508a8b0d216bdf4783c4e8"
memory_bytes_estimate = 549418642
"""


def parse(text: str, model_id: str = "faster-whisper-tiny"):
    return parse_asr_manifest(text, Path(f"{model_id}.toml"), model_id)


# ------------------------------------------------------------ the shipped six


def test_this_build_ships_thirteen_asr_models_across_two_engines() -> None:
    assert sorted(load_all_asr_manifests()) == MODELS
    assert len(CUDA_MODELS) == 6 and len(MAC_MODELS) == 7


def test_every_shipped_pin_is_a_full_commit_sha() -> None:
    """A branch name is not a pin, and a short sha is not reproducible."""
    for manifest in load_all_asr_manifests().values():
        for spec in manifest.backends.values():
            assert len(spec.revision) == 40
            assert spec.revision == spec.revision.lower()
            int(spec.revision, 16)  # raises if it is not hex


def test_each_model_serves_one_backend_with_that_backends_engine() -> None:
    """No model spans both, because no two sets of these weights are the same."""
    assert ASR_BACKEND_ENGINES == {
        "cuda-linux": "faster-whisper",
        "mlx-darwin": "mlx-whisper",
    }
    for model_id in CUDA_MODELS:
        manifest = load_asr_manifest(model_id)
        assert sorted(manifest.backends) == ["cuda-linux"]
        assert manifest.spec("cuda-linux").engine == "faster-whisper"
    for model_id in MAC_MODELS:
        manifest = load_asr_manifest(model_id)
        assert sorted(manifest.backends) == ["mlx-darwin"]
        assert manifest.spec("mlx-darwin").engine == "mlx-whisper"


def test_the_mac_estimates_are_measured_and_none_is_the_cuda_arithmetic() -> None:
    """Every mlx figure is `mx.get_peak_memory()` over ONE 900-second window on
    the M1 Ultra on 2026-09-14, recorded in each manifest with its method — not
    "weights plus a declared 1.5 GiB", which was written for a CUDA context and
    cuBLAS/cuDNN workspaces that do not exist on this backend."""
    measured = {
        "mlx-whisper-tiny": 549_418_642,
        "mlx-whisper-base": 877_017_662,
        "mlx-whisper-small": 1_540_273_318,
        "mlx-whisper-medium": 2_607_243_002,
        "mlx-whisper-large-v3": 4_153_379_610,
        "mlx-whisper-large-v3-turbo": 2_654_916_970,
        "mlx-whisper-distil-large-v3": 2_549_972_298,
    }
    assert sorted(measured) == sorted(MAC_MODELS)
    runtime = 1024 ** 3 + 512 * 1024 ** 2
    for model_id, peak in measured.items():
        manifest = load_asr_manifest(model_id)
        assert manifest.spec("mlx-darwin").memory_bytes_estimate == peak
        # A watched number, so it is not weights plus a round constant.
        assert peak % runtime != 0
        text = manifest.path.read_text(encoding="utf-8")
        assert "MEASURED, on the machine it is for" in text
        assert "mx.get_peak_memory()" in text


def test_every_mac_weight_is_smaller_than_its_estimate() -> None:
    """The peak has to cover the weights it loaded; the repo totals are the hub
    tree API's for those exact revisions, read on 2026-09-14."""
    repo_bytes = {
        "mlx-whisper-tiny": 74_420_620,
        "mlx-whisper-base": 143_726_326,
        "mlx-whisper-small": 481_309_720,
        "mlx-whisper-medium": 1_524_927_044,
        "mlx-whisper-large-v3": 3_083_522_487,
        "mlx-whisper-large-v3-turbo": 1_613_979_758,
        "mlx-whisper-distil-large-v3": 1_509_132_231,
    }
    for model_id, size in repo_bytes.items():
        spec = load_asr_manifest(model_id).spec("mlx-darwin")
        assert spec.memory_bytes_estimate > size


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
    assert sorted(weights_bytes) == sorted(CUDA_MODELS)
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


def test_faster_whisper_on_the_mac_is_refused_by_name() -> None:
    """A Mac block on a CTranslate2 model: the pairing that cannot be."""
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
    assert "does not run asr on mlx-darwin" in message
    assert "no Metal backend" in message


def test_an_id_that_does_not_name_its_engine_is_refused() -> None:
    """THE RULE THAT KEEPS A TRANSCRIPT HONEST. mlx-whisper weights filed under
    a `faster-whisper-` id would make two conversions one id, and
    `transcript.json` records the id and nothing else about the bytes."""
    text = MAC_GOOD.replace('id = "mlx-whisper-tiny"', 'id = "faster-whisper-tiny"')
    with pytest.raises(AsrManifestError) as caught:
        parse_asr_manifest(
            text, Path("faster-whisper-tiny.toml"), "faster-whisper-tiny"
        )
    message = str(caught.value)
    assert "requires an id beginning 'mlx-whisper-'" in message
    assert "a transcript records the id" in message


def test_a_good_mac_manifest_parses() -> None:
    manifest = parse_asr_manifest(
        MAC_GOOD, Path("mlx-whisper-tiny.toml"), "mlx-whisper-tiny"
    )
    assert manifest.supports("mlx-darwin")
    assert not manifest.supports("cuda-linux")
    assert manifest.spec("mlx-darwin").engine == "mlx-whisper"


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
        parse(GOOD.replace('engine = "faster-whisper"', 'engine = "vllm"'))
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
