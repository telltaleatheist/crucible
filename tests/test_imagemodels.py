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
pipeline = "qwen-image-2.1"
hf_repo = "Qwen/Qwen-Image-2.1"
revision = "790c92633540aa0cb11d9abf19eb46d861714758"
dtype = "bfloat16"
memory_bytes_estimate = 17000000000
memory_basis = "measured"
memory_note = "measured on the Studio"
size_multiple = 16
max_side = 2048
max_pixels = 1048576
native_width = 1024
native_height = 1024
quality_min_side = 256
size_note = "where the memory was measured"
default_steps = 40
default_guidance = 1.0
image_to_image = true
inpaint = true
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
    assert mac.image_to_image and pc.image_to_image
    assert mac.inpaint and pc.inpaint
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
        (("inpaint = true\n", ""), "missing required key(s) ['inpaint']"),
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


def test_the_small_rung_is_sdxl_on_cuda_only_with_its_own_defaults() -> None:
    manifest = load_all_image_manifests()["dreamshaper-xl-lightning"]
    assert sorted(manifest.backends) == ["cuda-linux"]
    spec = manifest.spec("cuda-linux")
    assert (spec.engine, spec.pipeline, spec.dtype, spec.storage_dtype) == (
        "diffusers", "sdxl", "float16", "float8_e4m3fn",
    )
    assert (spec.variant, spec.scheduler) == ("fp16", "dpmpp-sde-karras")
    assert (spec.native_width, spec.native_height, spec.quality_min_side) == (1024, 1024, 768)
    assert (spec.default_steps, spec.default_guidance) == (6, 2.0)
    assert "text" in spec.default_negative_prompt
    assert not spec.image_to_image and not spec.inpaint
    # fits an 8 GiB card with Victoria's 1 GiB desktop allowance (7 GiB for a model)
    assert spec.memory_bytes_estimate < 7 * 2**30
    # only the fp16 checkpoint, no fp32 file and not the repo's own (fp16-unsafe) VAE
    assert all("fp16" in name for name in spec.files if name.endswith(".safetensors"))
    assert not any(name.startswith("vae/") for name in spec.files)
    vae = spec.vae
    assert vae is not None and vae.hf_repo == "madebyollin/sdxl-vae-fp16-fix"
    assert [entry.target for entry in vae.files] == ["config.json", "diffusion_pytorch_model.safetensors"]
    assert spec.negative_prompt == {
        "read_when_guidance_above": 1.0,
        "costs_extra_pass": False,
        "default": spec.default_negative_prompt,
    }


def test_qwen_states_its_sizes_and_that_a_negative_prompt_costs_a_pass() -> None:
    for spec in load_all_image_manifests()["qwen-image-2.1"].backends.values():
        assert spec.pipeline == "qwen-image-2.1"
        assert (spec.native_width, spec.native_height, spec.quality_min_side) == (1024, 1024, 256)
        assert (spec.default_steps, spec.default_guidance) == (40, 1.0)
        assert spec.negative_prompt["costs_extra_pass"] is True
        assert spec.negative_prompt["default"] is None
        assert spec.facts()["native_width"] == 1024


SDXL_KEYS = 'scheduler = "dpmpp-sde-karras"\n'


@pytest.mark.parametrize(
    ("change", "words"),
    [
        (('pipeline = "qwen-image-2.1"', 'pipeline = "sdxl"'), "is not one mflux runs"),
        (("default_steps = 40", "default_steps = 40\n" + SDXL_KEYS), "read only by the 'sdxl' pipeline"),
        (("native_width = 1024", "native_width = 1000"), "native_width 1000"),
        (("native_height = 1024", "native_height = 2048"), "over max_pixels"),
        (("quality_min_side = 256", "quality_min_side = 128"), "quality_min_side 128"),
        (("default_steps = 40", "default_steps = 0"), "default_steps must be 1 to 100"),
        (("default_guidance = 1.0", "default_guidance = 1"), "default_guidance must be float"),
        (
            ("default_guidance = 1.0", 'default_guidance = 1.0\ndefault_negative_prompt = "text"'),
            "needs default_guidance above 1.0",
        ),
        (("max_pixels = 1048576", 'max_pixels = 1048576\nfiles = ["unet/*"]'), "no patterns"),
    ],
)
def test_a_bad_size_or_default_is_refused_by_name(change: tuple[str, str], words: str) -> None:
    with pytest.raises(ImageManifestError) as refused:
        parse(HEAD + MLX_BLOCK.replace(*change))
    assert words in str(refused.value)
