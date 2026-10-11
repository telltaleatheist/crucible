from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audiomodels import Companion, CompanionFile
from .backend import CUDA_LINUX, MLX_DARWIN
from .errors import CrucibleError
from .manifests import MODELS_PULL_COMMAND
from .tomltable import (
    HF_REPO_PATTERN,
    MODEL_ID_PATTERN,
    REVISION_PATTERN,
    SHA256_PATTERN,
    check_table,
)

IMAGE_DIR_ENV = "CRUCIBLE_IMAGE_DIR"

MFLUX = "mflux"
DIFFUSERS = "diffusers"

IMAGE_BACKEND_ENGINES: dict[str, str] = {
    CUDA_LINUX: DIFFUSERS,
    MLX_DARWIN: MFLUX,
}

# Which model a block runs: the pipeline the worker builds (crucible/jobs/image/worker.py).
QWEN_IMAGE_21 = "qwen-image-2.1"
SDXL = "sdxl"

# The pipelines each engine runs. mflux runs Qwen-Image 2.1 only; diffusers runs both.
ENGINE_PIPELINES: dict[str, frozenset[str]] = {
    MFLUX: frozenset({QWEN_IMAGE_21}),
    DIFFUSERS: frozenset({QWEN_IMAGE_21, SDXL}),
}

# The SDXL arm's sampler, by the name a manifest gives it: DPM++ SDE with Karras sigmas
# (diffusers' DPMSolverMultistepScheduler, algorithm_type "sde-dpmsolver++",
# use_karras_sigmas), the one DreamShaper XL Lightning is made for.
DPMPP_SDE_KARRAS = "dpmpp-sde-karras"
SCHEDULERS = frozenset({DPMPP_SDE_KARRAS})

# How the denoiser's weights are STORED on the card when that differs from the compute
# dtype (diffusers layerwise casting: each layer is upcast to `dtype` as it runs). Not a
# quantized model: the fp16 checkpoint's values are rounded to fp8 for storage only,
# with no calibration and no scales
# (docs/internals/image.md, "fp8 storage is not quantization").
STORAGE_DTYPES = frozenset({"float8_e4m3fn"})

# The one companion an image block may carry: a VAE that replaces the checkpoint's own.
VAE_COMPANION = "vae"

# The sides and settings a request may ask for at all (crucible/jobs/image ImageParams);
# a block's native size, quality floor and defaults are checked against them.
MIN_SIDE = 256
MAX_STEPS = 100
MAX_GUIDANCE = 10.0

MEMORY_BASES = frozenset({"measured", "declared"})

DTYPES = frozenset({"bfloat16", "float16", "float32"})

SIZE_MULTIPLES = frozenset({16, 32})

_MODEL_REQUIRED: dict[str, type] = {
    "id": str,
    "family": str,
    "display": str,
}
_BACKEND_REQUIRED: dict[str, type] = {
    "engine": str,
    "pipeline": str,
    "hf_repo": str,
    "revision": str,
    "dtype": str,
    "memory_bytes_estimate": int,
    "memory_basis": str,
    "memory_note": str,
    "size_multiple": int,
    "max_side": int,
    "max_pixels": int,
    "native_width": int,
    "native_height": int,
    "quality_min_side": int,
    "size_note": str,
    "default_steps": int,
    "default_guidance": float,
    "image_to_image": bool,
    "inpaint": bool,
}
_BACKEND_OPTIONAL: dict[str, type] = {
    "mlx_cache_limit_bytes": int,
    "default_negative_prompt": str,
    "variant": str,
    "storage_dtype": str,
    "scheduler": str,
    "files": list,
    "companions": list,
}
_COMPANION_REQUIRED: dict[str, type] = {
    "name": str,
    "hf_repo": str,
    "revision": str,
    "files": list,
}
_COMPANION_FILE_REQUIRED: dict[str, type] = {
    "source": str,
    "target": str,
    "sha256": str,
    "bytes": int,
}


class ImageManifestError(CrucibleError):
    ...


@dataclass(frozen=True)
class ImageBackendSpec:

    backend: str
    engine: str
    pipeline: str
    hf_repo: str
    revision: str
    dtype: str
    memory_bytes_estimate: int
    memory_basis: str
    memory_note: str
    size_multiple: int
    max_side: int
    max_pixels: int
    native_width: int
    native_height: int
    quality_min_side: int
    size_note: str
    default_steps: int
    default_guidance: float
    image_to_image: bool
    inpaint: bool
    mlx_cache_limit_bytes: int | None = None
    default_negative_prompt: str | None = None
    variant: str | None = None
    storage_dtype: str | None = None
    scheduler: str | None = None
    # Exact paths of the repo to pull; empty pulls the whole repo at the revision.
    files: tuple[str, ...] = ()
    companions: tuple[Companion, ...] = ()
    # Read by the companion weights layer (crucible/companionweights.py): no image
    # model is gated.
    gated: bool = False

    @property
    def vae(self) -> Companion | None:
        """The companion VAE that replaces the checkpoint's own, when the block has one."""
        return next((c for c in self.companions if c.name == VAE_COMPANION), None)

    @property
    def negative_prompt(self) -> dict[str, Any]:
        """What this model does with `negative_prompt`, for a client to read before it
        sends one (docs/IMAGE.md, "Negative prompts"). One rule for every image model:
        a negative prompt is read only when guidance is above 1.0, and guidance above 1.0
        needs one (sent, or the model's default). So a model whose default guidance is
        above 1.0 already runs both passes and a negative prompt costs it nothing more;
        one whose default is 1.0 needs guidance raised for it, a second pass per step."""
        return {
            "read_when_guidance_above": 1.0,
            "costs_extra_pass": self.default_guidance <= 1.0,
            "default": self.default_negative_prompt,
        }

    def facts(self) -> dict[str, Any]:
        """What a client needs to ask this model for a picture without knowing it: its
        native size, the side below which it degrades, its defaults and what it does with
        a negative prompt. Published on its playground page (crucible/playground.py)."""
        return {
            "native_width": self.native_width,
            "native_height": self.native_height,
            "quality_min_side": self.quality_min_side,
            "size_note": self.size_note,
            "size_multiple": self.size_multiple,
            "max_side": self.max_side,
            "max_pixels": self.max_pixels,
            "default_steps": self.default_steps,
            "default_guidance": self.default_guidance,
            "negative_prompt": self.negative_prompt,
            "image_to_image": self.image_to_image,
            "inpaint": self.inpaint,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "engine": self.engine,
            "pipeline": self.pipeline,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "dtype": self.dtype,
            "storage_dtype": self.storage_dtype,
            "variant": self.variant,
            "scheduler": self.scheduler,
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "memory_basis": self.memory_basis,
            "memory_note": self.memory_note,
            "mlx_cache_limit_bytes": self.mlx_cache_limit_bytes,
            "files": list(self.files),
            "companions": [companion.to_dict() for companion in self.companions],
            **self.facts(),
        }


@dataclass(frozen=True)
class ImageManifest:
    weights_family = "models"

    id: str
    family: str
    display: str
    backends: dict[str, ImageBackendSpec]
    path: Path

    @property
    def pull_command(self) -> str:
        return f"{MODELS_PULL_COMMAND} {self.id}"

    def aliases(self) -> "tuple[ImageManifest, ...]":
        return ()

    def supports(self, backend_kind: str) -> bool:
        return backend_kind in self.backends

    def spec(self, backend_kind: str) -> ImageBackendSpec:
        found = self.backends.get(backend_kind)
        if found is None:
            raise ImageManifestError(
                f"image model {self.id!r} has no {backend_kind} block; "
                f"{self.path.name} declares {sorted(self.backends)}"
            )
        return found

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "display": self.display,
            "backends": {k: v.to_dict() for k, v in sorted(self.backends.items())},
        }


def image_manifests_dir() -> Path:
    override = os.environ.get(IMAGE_DIR_ENV)
    if override is not None and override != "":
        path = Path(override).expanduser()
        if not path.is_dir():
            raise ImageManifestError(f"{IMAGE_DIR_ENV}={override!r} is not a directory")
        return path
    path = Path(__file__).resolve().parent / "image"
    if not path.is_dir():
        raise ImageManifestError(
            f"no image manifests at {path}; they are package data and this "
            f"install has lost them, or ${IMAGE_DIR_ENV} must point at them"
        )
    return path


def _check_document(document: dict[str, Any], path: Path) -> dict[str, Any]:
    unknown = sorted(set(document) - {"model", "backends"})
    if unknown:
        raise ImageManifestError(
            f"{path.name}: unknown top-level table(s) {unknown}; an image manifest "
            "has exactly [model] and [backends.<kind>]"
        )
    for table in ("model", "backends"):
        if not isinstance(document.get(table), dict):
            raise ImageManifestError(f"{path.name}: missing the [{table}] table")
    if not document["backends"]:
        raise ImageManifestError(
            f"{path.name}: no backend blocks; an image model nothing can run is "
            "not an image model"
        )
    return document


def _parse_model(model: dict[str, Any], path: Path, expected_id: str) -> str:
    check_table(f"{path.name} [model]", model, _MODEL_REQUIRED, error=ImageManifestError)
    model_id = model["id"]
    if not MODEL_ID_PATTERN.match(model_id):
        raise ImageManifestError(
            f"{path.name}: model.id {model_id!r} must be lower-case and start with "
            "a letter or digit ([a-z0-9][a-z0-9._-]*)"
        )
    if model_id != expected_id:
        raise ImageManifestError(
            f"{path.name}: model.id is {model_id!r} but the file is named "
            f"{expected_id!r}; the id and the filename are the same thing"
        )
    return model_id


def _check_pin(where: str, hf_repo: str, revision: str) -> None:
    if not HF_REPO_PATTERN.match(hf_repo):
        raise ImageManifestError(
            f"{where}: hf_repo {hf_repo!r} is not an <owner>/<name> HuggingFace repo id"
        )
    if not REVISION_PATTERN.match(revision):
        raise ImageManifestError(
            f"{where}: revision {revision!r} must be a full 40-character "
            "commit sha, so a pull is reproducible; branch names are not pins"
        )


def _check_engine_and_pins(where: str, kind: str, block: dict[str, Any]) -> None:
    if kind not in IMAGE_BACKEND_ENGINES:
        raise ImageManifestError(
            f"{where}: {kind!r} is not an image backend; the image backends are "
            f"{sorted(IMAGE_BACKEND_ENGINES)}. Windows is never one "
            "(docs/internals/image.md, \"Backends\")"
        )
    if block["engine"] != IMAGE_BACKEND_ENGINES[kind]:
        raise ImageManifestError(
            f"{where}: engine {block['engine']!r} does not generate images on "
            f"{kind}; that backend's image engine is {IMAGE_BACKEND_ENGINES[kind]!r}"
        )
    _check_pin(where, block["hf_repo"], block["revision"])
    if block["dtype"] not in DTYPES:
        raise ImageManifestError(
            f"{where}: dtype {block['dtype']!r} is not one of {sorted(DTYPES)}"
        )


def _check_memory(where: str, block: dict[str, Any]) -> None:
    if block["memory_bytes_estimate"] <= 0:
        raise ImageManifestError(
            f"{where}: memory_bytes_estimate must be positive, got "
            f"{block['memory_bytes_estimate']}"
        )
    if block["memory_basis"] not in MEMORY_BASES:
        raise ImageManifestError(
            f"{where}: memory_basis {block['memory_basis']!r} is not one of "
            f"{sorted(MEMORY_BASES)}"
        )
    if not block["memory_note"].strip():
        raise ImageManifestError(
            f"{where}: memory_note is empty; it says where the number came from "
            "and at what image size it holds"
        )


def _check_sizes(where: str, block: dict[str, Any]) -> None:
    multiple = block["size_multiple"]
    if multiple not in SIZE_MULTIPLES:
        raise ImageManifestError(
            f"{where}: size_multiple {multiple} is not one of {sorted(SIZE_MULTIPLES)}"
        )
    if block["max_side"] <= 0 or block["max_side"] % multiple:
        raise ImageManifestError(
            f"{where}: max_side {block['max_side']} must be a positive multiple of "
            f"size_multiple {multiple}"
        )
    if block["max_pixels"] < multiple * multiple:
        raise ImageManifestError(
            f"{where}: max_pixels {block['max_pixels']} admits no image at all"
        )


def _check_cache_limit(where: str, kind: str, block: dict[str, Any]) -> None:
    stated = "mlx_cache_limit_bytes" in block
    if kind == MLX_DARWIN and not stated:
        raise ImageManifestError(
            f"{where}: mlx_cache_limit_bytes is required on {MLX_DARWIN}; without "
            "it MLX keeps every freed buffer and the process grows past its estimate"
        )
    if kind != MLX_DARWIN and stated:
        raise ImageManifestError(
            f"{where}: mlx_cache_limit_bytes is an MLX setting and {kind} does not "
            "run MLX"
        )
    if stated and block["mlx_cache_limit_bytes"] <= 0:
        raise ImageManifestError(f"{where}: mlx_cache_limit_bytes must be positive")


def _check_pipeline(where: str, block: dict[str, Any]) -> None:
    engine, pipeline = block["engine"], block["pipeline"]
    runs = ENGINE_PIPELINES[engine]
    if pipeline not in runs:
        raise ImageManifestError(
            f"{where}: pipeline {pipeline!r} is not one {engine} runs; it runs {sorted(runs)}"
        )
    sdxl_only = [
        key
        for key in ("scheduler", "storage_dtype", "variant", "companions")
        if key in block and pipeline != SDXL
    ]
    if sdxl_only:
        raise ImageManifestError(
            f"{where}: {sdxl_only} are read only by the {SDXL!r} pipeline, and this "
            f"block runs {pipeline!r}"
        )
    if "scheduler" in block and block["scheduler"] not in SCHEDULERS:
        raise ImageManifestError(
            f"{where}: scheduler {block['scheduler']!r} is not one of {sorted(SCHEDULERS)}"
        )
    if "storage_dtype" in block and block["storage_dtype"] not in STORAGE_DTYPES:
        raise ImageManifestError(
            f"{where}: storage_dtype {block['storage_dtype']!r} is not one of "
            f"{sorted(STORAGE_DTYPES)}"
        )


def _check_native_size(where: str, block: dict[str, Any]) -> None:
    multiple = block["size_multiple"]
    width, height = block["native_width"], block["native_height"]
    for name, side in (("native_width", width), ("native_height", height)):
        if side < MIN_SIDE or side > block["max_side"] or side % multiple:
            raise ImageManifestError(
                f"{where}: {name} {side} must be a multiple of size_multiple {multiple} "
                f"from {MIN_SIDE} to max_side {block['max_side']}"
            )
    if width * height > block["max_pixels"]:
        raise ImageManifestError(
            f"{where}: the native size {width}x{height} is over max_pixels "
            f"{block['max_pixels']}, so a request that sends no size would be refused"
        )
    floor = block["quality_min_side"]
    if not MIN_SIDE <= floor <= min(width, height):
        raise ImageManifestError(
            f"{where}: quality_min_side {floor} must be from {MIN_SIDE} to the native "
            f"size's shorter side, {min(width, height)}"
        )
    if not block["size_note"].strip():
        raise ImageManifestError(
            f"{where}: size_note is empty; it says where the native size and the quality "
            "floor came from"
        )


def _check_defaults(where: str, block: dict[str, Any]) -> None:
    if not 1 <= block["default_steps"] <= MAX_STEPS:
        raise ImageManifestError(f"{where}: default_steps must be 1 to {MAX_STEPS}")
    if not 1.0 <= block["default_guidance"] <= MAX_GUIDANCE:
        raise ImageManifestError(
            f"{where}: default_guidance must be 1.0 to {MAX_GUIDANCE}"
        )
    negative = block.get("default_negative_prompt")
    if negative is not None and (not negative.strip() or block["default_guidance"] <= 1.0):
        raise ImageManifestError(
            f"{where}: default_negative_prompt must say something, and a negative prompt "
            "is read only when guidance is above 1.0, so it needs default_guidance above 1.0"
        )


def _companion_file(where: str, entry: Any) -> CompanionFile:
    if not isinstance(entry, dict):
        raise ImageManifestError(f"{where}: each companion file is a table")
    check_table(where, entry, _COMPANION_FILE_REQUIRED, error=ImageManifestError)
    if not SHA256_PATTERN.match(entry["sha256"]):
        raise ImageManifestError(f"{where}: sha256 must be 64 lower-case hex digits")
    return CompanionFile(**entry)


def _companion(where: str, entry: Any) -> Companion:
    if not isinstance(entry, dict):
        raise ImageManifestError(f"{where}: each companion is a table")
    check_table(where, entry, _COMPANION_REQUIRED, error=ImageManifestError)
    if entry["name"] != VAE_COMPANION:
        raise ImageManifestError(
            f"{where}: companion {entry['name']!r} is not one an image block takes; it "
            f"takes {VAE_COMPANION!r}, a VAE that replaces the checkpoint's own"
        )
    _check_pin(where, entry["hf_repo"], entry["revision"])
    files = tuple(
        _companion_file(f"{where} files[{index}]", item)
        for index, item in enumerate(entry["files"])
    )
    if not files:
        raise ImageManifestError(f"{where}: a companion with no files pulls nothing")
    return Companion(entry["name"], entry["hf_repo"], entry["revision"], files)


def _companions(where: str, block: dict[str, Any]) -> tuple[Companion, ...]:
    found = tuple(
        _companion(f"{where} companions[{index}]", entry)
        for index, entry in enumerate(block.get("companions", []))
    )
    if len({companion.name for companion in found}) != len(found):
        raise ImageManifestError(f"{where}: two companions share the name {VAE_COMPANION!r}")
    return found


def _files(where: str, block: dict[str, Any]) -> tuple[str, ...]:
    if "files" not in block:
        return ()
    files = block["files"]
    if not files:
        raise ImageManifestError(
            f"{where}: files is empty; leave it out to pull the whole repo"
        )
    if not all(isinstance(name, str) and name.strip() and "*" not in name for name in files):
        raise ImageManifestError(
            f"{where}: files names exact paths in the repo, no patterns, so a pull can "
            "check that each one arrived"
        )
    return tuple(files)


def _parse_backend(path: Path, kind: str, block: Any) -> ImageBackendSpec:
    where = f"{path.name} [backends.{kind}]"
    if not isinstance(block, dict):
        raise ImageManifestError(f"{where}: must be a table")
    check_table(
        where, block, _BACKEND_REQUIRED, _BACKEND_OPTIONAL, error=ImageManifestError
    )
    _check_engine_and_pins(where, kind, block)
    _check_pipeline(where, block)
    _check_memory(where, block)
    _check_sizes(where, block)
    _check_native_size(where, block)
    _check_defaults(where, block)
    _check_cache_limit(where, kind, block)
    return ImageBackendSpec(
        backend=kind,
        **{**block, "files": _files(where, block), "companions": _companions(where, block)},
    )


def _parse(document: dict[str, Any], path: Path, expected_id: str) -> ImageManifest:
    _check_document(document, path)
    model = document["model"]
    model_id = _parse_model(model, path, expected_id)
    backends = {
        kind: _parse_backend(path, kind, block)
        for kind, block in document["backends"].items()
    }
    return ImageManifest(
        id=model_id,
        family=model["family"],
        display=model["display"],
        backends=backends,
        path=path,
    )


def parse_image_manifest(text: str, path: Path, expected_id: str) -> ImageManifest:
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ImageManifestError(f"{path.name}: not valid TOML: {exc}") from exc
    return _parse(document, path, expected_id)


def load_image_manifest(model_id: str, directory: Path | None = None) -> ImageManifest:
    root = directory if directory is not None else image_manifests_dir()
    path = root / f"{model_id}.toml"
    if not path.is_file():
        known = sorted(p.stem for p in root.glob("*.toml"))
        raise ImageManifestError(
            f"no image manifest for {model_id!r} at {path}; this build ships {known}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ImageManifestError(f"could not read {path}: {exc}") from exc
    return parse_image_manifest(text, path, model_id)


def load_all_image_manifests(
    directory: Path | None = None,
) -> dict[str, ImageManifest]:
    root = directory if directory is not None else image_manifests_dir()
    return {
        path.stem: load_image_manifest(path.stem, root)
        for path in sorted(root.glob("*.toml"), key=lambda p: p.stem)
    }


__all__ = [
    "DIFFUSERS",
    "DPMPP_SDE_KARRAS",
    "QWEN_IMAGE_21",
    "SDXL",
    "IMAGE_BACKEND_ENGINES",
    "IMAGE_DIR_ENV",
    "ImageBackendSpec",
    "ImageManifest",
    "ImageManifestError",
    "MFLUX",
    "image_manifests_dir",
    "load_all_image_manifests",
    "load_image_manifest",
    "parse_image_manifest",
]
