from __future__ import annotations

from pathlib import Path

import pytest

from crucible.imagemodels import (
    ImageManifestError,
    load_all_image_manifests,
    parse_image_manifest,
)

MLX_BLOCK = """
[backends.mlx-darwin]
engine = "mflux"
hf_repo = "Qwen/Qwen-Image-2.1"
revision = "790c92633540aa0cb11d9abf19eb46d861714758"
dtype = "bfloat16"
memory_bytes_estimate = 17000000000
memory_basis = "measured"
memory_note = "measured on the Studio"
size_multiple = 16
max_side = 2048
max_pixels = 1048576
image_to_image = true
mlx_cache_limit_bytes = 4000000000
"""

HEAD = """
[model]
id = "qwen-image-2.1"
family = "qwen-image"
display = "Qwen-Image 2.1"
"""


def parse(text: str):
    return parse_image_manifest(text, Path("qwen-image-2.1.toml"), "qwen-image-2.1")


def test_the_shipped_manifest_pins_both_arms_to_one_revision() -> None:
    manifest = load_all_image_manifests()["qwen-image-2.1"]
    mac, pc = manifest.spec("mlx-darwin"), manifest.spec("cuda-linux")
    assert (mac.engine, pc.engine) == ("mflux", "diffusers")
    assert mac.revision == pc.revision
    assert (mac.size_multiple, pc.size_multiple) == (16, 32)
    assert mac.image_to_image and not pc.image_to_image
    assert mac.mlx_cache_limit_bytes and pc.mlx_cache_limit_bytes is None
    assert manifest.pull_command == "crucible models pull qwen-image-2.1"


def test_a_well_formed_block_parses() -> None:
    spec = parse(HEAD + MLX_BLOCK).spec("mlx-darwin")
    assert spec.memory_bytes_estimate == 17_000_000_000
    assert spec.files == ()


@pytest.mark.parametrize(
    ("change", "words"),
    [
        (('engine = "mflux"', 'engine = "diffusers"'), "does not generate images on mlx-darwin"),
        (("mlx_cache_limit_bytes = 4000000000\n", ""), "mlx_cache_limit_bytes is required"),
        (('memory_basis = "measured"', 'memory_basis = "guessed"'), "memory_basis 'guessed'"),
        (('memory_note = "measured on the Studio"', 'memory_note = "  "'), "memory_note is empty"),
        (("size_multiple = 16", "size_multiple = 8"), "size_multiple 8"),
        (("max_side = 2048", "max_side = 2050"), "multiple of size_multiple"),
        (('revision = "790c92633540aa0cb11d9abf19eb46d861714758"', 'revision = "main"'), "40-character"),
        (("image_to_image = true", "image_to_image = 1"), "image_to_image must be bool"),
        (("max_pixels = 1048576", "max_pixels = 1048576\nquantize = 8"), "unknown key(s) ['quantize']"),
    ],
)
def test_a_bad_block_is_refused_by_name(change: tuple[str, str], words: str) -> None:
    with pytest.raises(ImageManifestError) as refused:
        parse(HEAD + MLX_BLOCK.replace(*change))
    assert words in str(refused.value)


def test_the_cache_limit_is_refused_off_the_mac() -> None:
    block = MLX_BLOCK.replace("mlx-darwin", "cuda-linux").replace('"mflux"', '"diffusers"')
    with pytest.raises(ImageManifestError) as refused:
        parse(HEAD + block)
    assert "cuda-linux does not run MLX" in str(refused.value)


def test_windows_is_never_an_image_backend() -> None:
    with pytest.raises(ImageManifestError) as refused:
        parse(HEAD + MLX_BLOCK.replace("mlx-darwin", "llama-windows"))
    assert "not an image backend" in str(refused.value)
