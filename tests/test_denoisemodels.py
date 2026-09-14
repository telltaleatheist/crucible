"""Denoise manifests: the one this build ships, and what the loader refuses.

Strict for `crucible/asrmodels.py`'s reason. A separator run with the wrong
checkpoint produces audio that sounds nearly right, and the manifest is the only
place that says which weights made it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.denoisemodels import (
    DenoiseManifestError,
    load_all_denoise_manifests,
    load_denoise_manifest,
    parse_denoise_manifest,
)

SHA = "7c1c39191edc34e942ca7f2346ce6b6c0e1208a5f76349ffce6f696bd12910de"
OTHER_SHA = "5d7d83b2e9d232da60941b717b0abdc345155d45cff3f79715cdb2790ba18c36"

GOOD = f"""
[model]
id = "demo-denoise"
display = "Demo Denoiser"
model_filename = "demo_denoise.ckpt"
config_filename = "demo_denoise_config.yaml"
primary_stem = "dry"
sample_rate = 44100

[backends.cuda-linux]
engine = "audio-separator"
hf_repo = "someone/Resources"
revision = "0123456789abcdef0123456789abcdef01234567"
model_path = "models/demo_denoise.ckpt"
model_sha256 = "{SHA}"
model_bytes = 913097300
config_path = "models/config_demo.yaml"
config_sha256 = "{OTHER_SHA}"
memory_bytes_estimate = 2523710036
"""


def parse(text: str, name: str = "demo-denoise"):
    return parse_denoise_manifest(text, Path(f"{name}.toml"), name)


def test_a_complete_manifest_parses() -> None:
    manifest = parse(GOOD)
    assert manifest.id == "demo-denoise"
    assert manifest.primary_stem == "dry"
    assert manifest.sample_rate == 44100
    spec = manifest.spec("cuda-linux")
    assert spec.model_sha256 == SHA
    assert spec.model_bytes == 913097300
    assert manifest.supports("cuda-linux")
    assert not manifest.supports("mlx-darwin")


def test_the_shipped_manifest_names_both_filenames_and_a_real_pin() -> None:
    """The names audio-separator resolves by are NOT the paths they come from,
    and the manifest carries both halves."""
    manifest = load_denoise_manifest("denoise-roformer")
    assert manifest.model_filename.endswith(".ckpt")
    assert manifest.config_filename.endswith("_config.yaml")
    for kind in ("cuda-linux", "mlx-darwin"):
        spec = manifest.spec(kind)
        assert spec.model_path != manifest.model_filename
        assert spec.config_path.rsplit("/", 1)[-1] != manifest.config_filename
        assert len(spec.revision) == 40
        assert len(spec.model_sha256) == 64


def test_every_shipped_manifest_loads() -> None:
    manifests = load_all_denoise_manifests()
    assert "denoise-roformer" in manifests


@pytest.mark.parametrize(
    "bad,fragment",
    [
        (GOOD.replace("primary_stem", "primary_stemm"), "primary_stemm"),
        (GOOD.replace('revision = "0123456789abcdef0123456789abcdef01234567"',
                      'revision = "main"'), "40-character"),
        (GOOD.replace(f'model_sha256 = "{SHA}"', 'model_sha256 = "abc"'),
         "sha256"),
        (GOOD.replace("model_bytes = 913097300", "model_bytes = 0"),
         "model_bytes"),
        (GOOD.replace("sample_rate = 44100", "sample_rate = 0"), "sample_rate"),
        (GOOD.replace('engine = "audio-separator"', 'engine = "vllm"'),
         "does not denoise"),
        (GOOD.replace("[backends.cuda-linux]", "[backends.windows]"),
         "not a denoise backend"),
    ],
)
def test_what_the_loader_refuses(bad: str, fragment: str) -> None:
    with pytest.raises(DenoiseManifestError) as caught:
        parse(bad)
    assert fragment in str(caught.value)


@pytest.mark.parametrize(
    "value",
    ["../../etc/passwd", "sub/dir/model.ckpt", ".hidden.ckpt"],
)
def test_a_filename_that_is_a_path_is_refused(value: str) -> None:
    """These are written into one flat directory an engine then reads BY NAME."""
    with pytest.raises(DenoiseManifestError) as caught:
        parse(GOOD.replace('model_filename = "demo_denoise.ckpt"',
                           f'model_filename = "{value}"'))
    assert "plain filename" in str(caught.value)


def test_an_id_that_disagrees_with_its_filename_is_refused() -> None:
    with pytest.raises(DenoiseManifestError) as caught:
        parse(GOOD, name="something-else")
    assert "the same thing" in str(caught.value)
