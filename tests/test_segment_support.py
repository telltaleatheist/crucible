from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from crucible import catalog, installonsubmit, jobenv, verdict, weights
from crucible.capabilityclasses import BY_NAME
from crucible.config import load_config
from crucible.desktop_app.screens import JOB_TYPE_WORDS
from crucible.jobs import segment as segment_job
from crucible.jobs.segment.picture import picture_size
from crucible.memorybudget import GIB
from crucible.segmentmodels import (
    SegmentManifestError,
    load_all_segment_manifests,
    load_segment_manifest,
    parse_segment_manifest,
)

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND, configure_box

BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
SAM_REVISION = "665f8e2ad61cf5f53d65644ff27c8ee525124610"


def test_the_two_models_are_declared_with_their_class_revision_and_licence() -> None:
    manifests = load_all_segment_manifests()
    assert {m.id: m.kind for m in manifests.values()} == {
        "birefnet": "cutout",
        "sam2.1-hiera-large": "select",
    }
    birefnet, sam = manifests["birefnet"], manifests["sam2.1-hiera-large"]
    assert birefnet.licence.startswith("MIT") and sam.licence.startswith("Apache-2.0")
    for manifest, repo, revision in (
        (birefnet, "ZhengPeng7/BiRefNet", BIREFNET_REVISION),
        (sam, "facebook/sam2.1-hiera-large", SAM_REVISION),
    ):
        assert sorted(manifest.backends) == ["cuda-linux", "mlx-darwin"]
        for spec in manifest.backends.values():
            assert (spec.hf_repo, spec.revision) == (repo, revision)
            assert spec.memory_basis == "declared" and "not measured" in spec.memory_note
            assert "model.safetensors" in spec.files
        assert manifest.spec("mlx-darwin").device == "mps"
        assert manifest.spec("cuda-linux").device == "cuda"
    assert birefnet.spec("cuda-linux").dtype == "float16"
    assert birefnet.spec("mlx-darwin").dtype == "float32"
    assert "sam2.1_hiera_large.pt" not in sam.spec("cuda-linux").files
    assert set(birefnet.spec("cuda-linux").files) >= {"birefnet.py", "BiRefNet_config.py", "config.json"}


def test_both_models_fit_the_pc_card_beside_its_desktop_allowance() -> None:
    for manifest in load_all_segment_manifests().values():
        assert manifest.spec("cuda-linux").memory_bytes_estimate <= 24 * GIB - 3 * GIB


BASE = """
[model]
id = "x"
family = "f"
display = "X"
kind = "cutout"
licence = "l"
licence_url = "https://example.invalid"
commercial_use = "yes"

[backends.cuda-linux]
engine = "birefnet"
hf_repo = "o/x"
revision = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
dtype = "float16"
memory_bytes_estimate = 1
memory_basis = "declared"
memory_note = "n"
files = ["model.safetensors"]
working_side = 1024
max_pixels = 40000000
"""


@pytest.mark.parametrize(
    ("change", "words"),
    [
        (('kind = "cutout"', 'kind = "depth"'), "is not one of"),
        (('engine = "birefnet"', 'engine = "sam2"'), "makes 'select' masks"),
        (('engine = "birefnet"', 'engine = "rembg"'), "does not run on cuda-linux"),
        (('revision = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"', 'revision = "main"'), "branch names are not pins"),
        (('memory_basis = "declared"', 'memory_basis = "guessed"'), "memory_basis"),
        (("files = [\"model.safetensors\"]", "files = []"), "non-empty list"),
        (("max_pixels = 40000000", "max_pixels = 40000000\nthreshold = 0.5"), "unknown key(s) ['threshold']"),
        (("[backends.cuda-linux]", "[backends.llama-windows]"), "is not a segment backend"),
    ],
)
def test_a_manifest_that_says_the_wrong_thing_is_refused_by_name(
    change: tuple[str, str], words: str
) -> None:
    with pytest.raises(SegmentManifestError) as caught:
        parse_segment_manifest(BASE.replace(*change), Path("x.toml"), "x")
    assert words in str(caught.value)


def _encoded(fmt: str, size: tuple[int, int], mode: str = "RGB", **options: Any) -> bytes:
    image_module = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    image_module.new(mode, size).save(buffer, format=fmt, **options)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("fmt", "mode", "options"),
    [
        ("PNG", "RGB", {}),
        ("PNG", "RGBA", {}),
        ("PNG", "P", {}),
        ("JPEG", "RGB", {}),
        ("JPEG", "RGB", {"progressive": True}),
        ("JPEG", "L", {"exif": b"Exif\x00\x00" + b"\x00" * 80}),
        ("WEBP", "RGB", {}),
        ("WEBP", "RGB", {"lossless": True}),
        ("WEBP", "RGBA", {}),
    ],
)
def test_the_size_is_read_from_the_header_as_pillow_reads_it(
    tmp_path: Path, fmt: str, mode: str, options: dict
) -> None:
    for size in ((1, 1), (641, 479), (3000, 17)):
        path = tmp_path / f"p.{fmt.lower()}"
        path.write_bytes(_encoded(fmt, size, mode, **options))
        assert picture_size(path) == size, (fmt, mode, options, size)


def test_a_header_that_does_not_say_its_size_reads_as_none(tmp_path: Path) -> None:
    path = tmp_path / "cut.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
    assert picture_size(path) is None
    path.write_bytes(b"RIFF\x00\x00\x00\x00WEBPVP8 ")
    assert picture_size(path) is None


def test_the_segment_env_is_one_worker_env_on_each_backend() -> None:
    assert "segment" in jobenv.WORKER_JOB_TYPES
    assert jobenv.JOB_TYPES_SERVED_BY_ENV["segment"] == ("segment",)
    for backend_kind in ("cuda-linux", "mlx-darwin"):
        spec = jobenv.worker_env("segment", backend_kind)
        recipe = jobenv.recipe_for(spec)
        assert recipe.name == f"{backend_kind}.txt" and recipe.parent.name == "segment"
        assert jobenv.recipe_archive_bytes(recipe) > 0
        pins = jobenv.recipe_pins(recipe)
        for package in ("torch", "torchvision", "transformers", "timm", "kornia", "einops", "pillow"):
            assert package in pins, (backend_kind, package)
        assert pins["transformers"] == "5.17.0"
        assert "timm" in jobenv.SMOKE_IMPORT["segment"][backend_kind]
    cuda = jobenv.recipe_pins(jobenv.recipe_for(jobenv.worker_env("segment", "cuda-linux")))
    mac = jobenv.recipe_pins(jobenv.recipe_for(jobenv.worker_env("segment", "mlx-darwin")))
    assert "triton" in cuda and "triton" not in mac
    assert {k: v for k, v in cuda.items() if k in mac} == mac


def test_install_on_submit_names_the_segment_installer_and_its_recipe() -> None:
    assert installonsubmit._installer_of("segment", "/home/x/envs/segment") == "segment"
    recipe = jobenv.recipe_for(jobenv.worker_env("segment", "cuda-linux"))
    assert installonsubmit._env_bytes("segment", None, "cuda-linux") == jobenv.recipe_archive_bytes(recipe)


@pytest.mark.parametrize(
    ("name", "backend_kind", "summary"),
    [
        ("cutout", "cuda-linux", "can cut out a picture's subject, using birefnet"),
        ("select", "cuda-linux", "can select what is pointed at in a picture, using sam2.1-hiera-large"),
        ("cutout", "mlx-darwin", "can cut out a picture's subject, using birefnet"),
        ("select", "mlx-darwin", "can select what is pointed at in a picture, using sam2.1-hiera-large"),
    ],
)
def test_the_capability_rows_say_what_this_host_can_select_and_with_what(
    name: str, backend_kind: str, summary: str
) -> None:
    total = 24 * GIB if backend_kind == "cuda-linux" else 64 * GIB
    decided = verdict.decide(
        BY_NAME[name], backend_kind, total_bytes=total, desktop_allowance_bytes=3 * GIB,
        gpu_vendor="nvidia" if backend_kind == "cuda-linux" else "apple", chosen=None,
    )
    assert decided.enabled is True, decided.reason
    assert decided.summary.startswith(summary), decided.summary


def test_a_card_too_small_for_either_model_says_so() -> None:
    decided = verdict.decide(
        BY_NAME["cutout"], "cuda-linux", total_bytes=6 * GIB, desktop_allowance_bytes=3 * GIB,
        gpu_vendor="nvidia", chosen=None,
    )
    assert decided.enabled is False and decided.shortfall_bytes > 0
    assert decided.summary.startswith("cannot cut out a picture's subject")


def test_segment_runs_in_wsl_and_never_on_windows() -> None:
    assert "segment" in verdict.WSL_ONLY_JOB_TYPES


def test_the_catalog_lists_segment_models_as_models_of_the_segment_job(home: Path) -> None:
    configure_box(home, enable_segment=True)
    config = load_config(home)
    for backend in (FAKE_BACKEND, FAKE_MAC_BACKEND):
        rows = {s.id: s for s in catalog.subjects(config, backend) if s.job_type == "segment"}
        assert sorted(rows) == ["birefnet", "sam2.1-hiera-large"]
        assert all(s.kind == "model" and s.installed() is None for s in rows.values())
        assert rows["birefnet"].pull_command == "crucible models pull birefnet"
    assert catalog.backends_declaring("model", "sam2.1-hiera-large") == ["cuda-linux", "mlx-darwin"]
    assert "birefnet" in catalog.declared_ids()["model"]


def test_a_pull_fetches_only_the_files_the_arm_names(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configure_box(home, enable_segment=True)
    config = load_config(home)
    manifest = load_segment_manifest("sam2.1-hiera-large")
    spec = manifest.spec("cuda-linux")
    asked: list[Any] = []

    def snapshot(config: Any, name: str, spec: Any, target: Path, *, patterns: Any, **_: Any) -> None:
        asked.append(list(patterns))
        target.mkdir(parents=True, exist_ok=True)
        for file in patterns:
            (target / file).write_bytes(b"w")

    monkeypatch.setattr(weights, "_snapshot", snapshot)
    found = weights.pull(config, manifest, spec)
    assert asked == [list(spec.files)]
    assert found.revision == SAM_REVISION
    assert weights.installed(config, manifest, spec) is not None


def test_the_worker_env_on_the_mac_lets_torch_fall_back_to_the_cpu(tmp_path: Path) -> None:
    python = tmp_path / "envs" / "segment" / "bin" / "python"
    mac = segment_job.worker_environment(python, "mlx-darwin")
    pc = segment_job.worker_environment(python, "cuda-linux")
    assert mac["PYTORCH_ENABLE_MPS_FALLBACK"] == "1" and "PYTORCH_ENABLE_MPS_FALLBACK" not in pc
    for environment in (mac, pc):
        assert environment["HF_HUB_OFFLINE"] == "1"


def test_the_desktop_packages_screen_has_words_for_segment() -> None:
    assert JOB_TYPE_WORDS["segment"][0] == "Cutouts and selections"
