from __future__ import annotations

from pathlib import Path

import pytest

from crucible.alignmodels import (
    ALIGN_BACKEND_ENGINES,
    AlignManifestError,
    align_manifests_dir,
    load_align_manifest,
    load_all_align_manifests,
    parse_align_manifest,
)
from crucible.backend import CUDA_LINUX, MLX_DARWIN

GOOD = """
[model]
id = "qwen3-aligner"
family = "qwen3-forced-aligner"
parameters_m = 600

[backends.cuda-linux]
engine = "qwen3-forced-aligner"
hf_repo = "Qwen/Qwen3-ForcedAligner-0.6B"
revision = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
dtype = "bfloat16"
memory_bytes_estimate = 3446157280
"""


def parse(text: str, model_id: str = "qwen3-aligner"):
    return parse_align_manifest(text, Path(f"{model_id}.toml"), model_id)


def test_the_build_ships_exactly_one_aligner() -> None:
    manifests = load_all_align_manifests()
    assert sorted(manifests) == ["qwen3-aligner"]


def test_the_shipped_manifest_pins_a_commit_and_a_dtype() -> None:
    spec = load_align_manifest("qwen3-aligner").spec(CUDA_LINUX)
    assert spec.hf_repo == "Qwen/Qwen3-ForcedAligner-0.6B"
    assert len(spec.revision) == 40
    assert spec.dtype == "bfloat16"
    assert spec.memory_bytes_estimate == 1_835_544_544 + 1_610_612_736


def test_the_aligner_pulls_into_the_models_tree() -> None:
    assert load_align_manifest("qwen3-aligner").weights_family == "models"


def test_both_backends_run_the_same_engine_on_the_same_weights() -> None:
    assert sorted(ALIGN_BACKEND_ENGINES) == [CUDA_LINUX, MLX_DARWIN]
    assert set(ALIGN_BACKEND_ENGINES.values()) == {"qwen3-forced-aligner"}
    manifest = load_align_manifest("qwen3-aligner")
    cuda, mac = manifest.spec(CUDA_LINUX), manifest.spec(MLX_DARWIN)
    assert mac.hf_repo == cuda.hf_repo
    assert mac.revision == cuda.revision
    assert mac.dtype == cuda.dtype == "bfloat16"


def test_the_mac_estimate_is_measured_and_not_the_cuda_one() -> None:
    manifest = load_align_manifest("qwen3-aligner")
    mac = manifest.spec(MLX_DARWIN).memory_bytes_estimate
    cuda = manifest.spec(CUDA_LINUX).memory_bytes_estimate
    assert mac == 5_885_296_640
    assert mac != cuda
    assert mac > cuda
    text = manifest.path.read_text(encoding="utf-8")
    assert "MEASURED, on the machine it is for" in text
    assert "driver_allocated_memory" in text


def test_the_mac_recipe_is_there_and_its_note_says_what_is_still_owed() -> None:
    envs = align_manifests_dir().parent / "envs" / "align"
    recipe = envs / "mlx-darwin.txt"
    assert recipe.is_file()
    lines = [
        line
        for line in recipe.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert "qwen-asr==0.0.6" in lines
    assert "torch==2.14.0" in lines
    assert not [line for line in lines if line.startswith(("nvidia-", "cuda-"))]
    assert not [line for line in lines if line.startswith("triton==")]
    note = (envs / "mlx-darwin.md").read_text(encoding="utf-8")
    assert "Compare the timestamps" in note
    assert "97x realtime" in note


def test_a_good_manifest_parses() -> None:
    manifest = parse(GOOD)
    assert manifest.id == "qwen3-aligner"
    assert manifest.parameters_m == 600
    assert manifest.spec(CUDA_LINUX).dtype == "bfloat16"


def test_an_unknown_key_is_refused_not_ignored() -> None:
    with pytest.raises(AlignManifestError) as caught:
        parse(GOOD.replace("parameters_m = 600", "parameters_m = 600\nparams_b = 1"))
    assert "unknown key(s) ['params_b']" in str(caught.value)


def test_a_misspelled_estimate_is_refused() -> None:
    with pytest.raises(AlignManifestError) as caught:
        parse(GOOD.replace("memory_bytes_estimate", "memory_bytes_estimat"))
    message = str(caught.value)
    assert "memory_bytes_estimat" in message
    assert "memory_bytes_estimate" in message


def test_a_branch_name_is_not_a_pin() -> None:
    with pytest.raises(AlignManifestError) as caught:
        parse(GOOD.replace('"c7cbfc2048c462b0d63a45797104fc9db3ad62b7"', '"main"'))
    assert "branch names are not pins" in str(caught.value)


def test_a_dtype_torch_does_not_have_is_refused() -> None:
    with pytest.raises(AlignManifestError) as caught:
        parse(GOOD.replace('dtype = "bfloat16"', 'dtype = "bf16"'))
    assert "AttributeError one model load later" in str(caught.value)


def test_a_mac_block_parses_and_a_windows_one_does_not() -> None:
    mac = GOOD + """
[backends.mlx-darwin]
engine = "qwen3-forced-aligner"
hf_repo = "Qwen/Qwen3-ForcedAligner-0.6B"
revision = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
dtype = "bfloat16"
memory_bytes_estimate = 3434819552
"""
    assert parse(mac).supports(MLX_DARWIN)

    with pytest.raises(AlignManifestError) as caught:
        parse(mac.replace("[backends.mlx-darwin]", "[backends.llama-windows]"))
    message = str(caught.value)
    assert "not an align backend" in message
    assert "Windows is never a backend" in message


def test_the_id_and_the_filename_are_the_same_thing() -> None:
    with pytest.raises(AlignManifestError) as caught:
        parse_align_manifest(GOOD, Path("aligner.toml"), "aligner")
    assert "the id and the filename are the same thing" in str(caught.value)


def test_a_manifest_nothing_can_run_is_refused() -> None:
    with pytest.raises(AlignManifestError) as caught:
        parse(GOOD.split("[backends.cuda-linux]")[0] + "[backends]\n")
    assert "an aligner nothing can run is not an aligner" in str(caught.value)


def test_a_missing_manifest_lists_what_ships(tmp_path: Path) -> None:
    with pytest.raises(AlignManifestError) as caught:
        load_align_manifest("whisperx", tmp_path)
    assert "this build ships []" in str(caught.value)
