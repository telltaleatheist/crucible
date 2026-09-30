from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backend import CUDA_LINUX, MLX_DARWIN
from .errors import CrucibleError
from .manifests import MODELS_PULL_COMMAND
from .tomltable import HF_REPO_PATTERN, MODEL_ID_PATTERN, REVISION_PATTERN, check_table

IMAGE_DIR_ENV = "CRUCIBLE_IMAGE_DIR"

MFLUX = "mflux"
DIFFUSERS = "diffusers"

IMAGE_BACKEND_ENGINES: dict[str, str] = {
    CUDA_LINUX: DIFFUSERS,
    MLX_DARWIN: MFLUX,
}

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
    "hf_repo": str,
    "revision": str,
    "dtype": str,
    "memory_bytes_estimate": int,
    "memory_basis": str,
    "memory_note": str,
    "size_multiple": int,
    "max_side": int,
    "max_pixels": int,
    "image_to_image": bool,
    "inpaint": bool,
}
_BACKEND_OPTIONAL: dict[str, type] = {
    "mlx_cache_limit_bytes": int,
}


class ImageManifestError(CrucibleError):
    ...


@dataclass(frozen=True)
class ImageBackendSpec:

    backend: str
    engine: str
    hf_repo: str
    revision: str
    dtype: str
    memory_bytes_estimate: int
    memory_basis: str
    memory_note: str
    size_multiple: int
    max_side: int
    max_pixels: int
    image_to_image: bool
    inpaint: bool
    mlx_cache_limit_bytes: int | None = None

    @property
    def files(self) -> tuple[str, ...]:
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "engine": self.engine,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "dtype": self.dtype,
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "memory_basis": self.memory_basis,
            "memory_note": self.memory_note,
            "size_multiple": self.size_multiple,
            "max_side": self.max_side,
            "max_pixels": self.max_pixels,
            "image_to_image": self.image_to_image,
            "inpaint": self.inpaint,
            "mlx_cache_limit_bytes": self.mlx_cache_limit_bytes,
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
    if not HF_REPO_PATTERN.match(block["hf_repo"]):
        raise ImageManifestError(
            f"{where}: hf_repo {block['hf_repo']!r} is not an <owner>/<name> "
            "HuggingFace repo id"
        )
    if not REVISION_PATTERN.match(block["revision"]):
        raise ImageManifestError(
            f"{where}: revision {block['revision']!r} must be a full 40-character "
            "commit sha, so a pull is reproducible; branch names are not pins"
        )
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


def _parse_backend(path: Path, kind: str, block: Any) -> ImageBackendSpec:
    where = f"{path.name} [backends.{kind}]"
    if not isinstance(block, dict):
        raise ImageManifestError(f"{where}: must be a table")
    check_table(
        where, block, _BACKEND_REQUIRED, _BACKEND_OPTIONAL, error=ImageManifestError
    )
    _check_engine_and_pins(where, kind, block)
    _check_memory(where, block)
    _check_sizes(where, block)
    _check_cache_limit(where, kind, block)
    return ImageBackendSpec(backend=kind, **block)


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
