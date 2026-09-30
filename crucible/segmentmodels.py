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

SEGMENT_DIR_ENV = "CRUCIBLE_SEGMENT_DIR"

BIREFNET = "birefnet"
SAM2 = "sam2"

ENGINES: tuple[str, ...] = (BIREFNET, SAM2)

SEGMENT_BACKEND_ENGINES: dict[str, frozenset[str]] = {
    CUDA_LINUX: frozenset(ENGINES),
    MLX_DARWIN: frozenset(ENGINES),
}

DEVICE_FOR_BACKEND: dict[str, str] = {
    CUDA_LINUX: "cuda",
    MLX_DARWIN: "mps",
}

CUTOUT = "cutout"
SELECT = "select"

KINDS: tuple[str, ...] = (CUTOUT, SELECT)

KIND_WORDS: dict[str, str] = {
    CUTOUT: "the main subject's mask, found by itself (background removal)",
    SELECT: "the mask of the object the caller points at",
}

ENGINE_KIND: dict[str, str] = {BIREFNET: CUTOUT, SAM2: SELECT}

PROMPTED_KINDS: frozenset[str] = frozenset({SELECT})

MEMORY_BASES = frozenset({"measured", "declared"})

DTYPES = frozenset({"bfloat16", "float16", "float32"})

_MODEL_REQUIRED: dict[str, type] = {
    "id": str,
    "family": str,
    "display": str,
    "kind": str,
    "licence": str,
    "licence_url": str,
    "commercial_use": str,
}
_BACKEND_REQUIRED: dict[str, Any] = {
    "engine": str,
    "hf_repo": str,
    "revision": str,
    "dtype": str,
    "memory_bytes_estimate": int,
    "memory_basis": str,
    "memory_note": str,
    "files": list,
    "working_side": int,
    "max_pixels": int,
}


class SegmentManifestError(CrucibleError):
    ...


@dataclass(frozen=True)
class SegmentBackendSpec:

    backend: str
    engine: str
    hf_repo: str
    revision: str
    dtype: str
    memory_bytes_estimate: int
    memory_basis: str
    memory_note: str
    files: tuple[str, ...]
    working_side: int
    max_pixels: int

    @property
    def device(self) -> str:
        return DEVICE_FOR_BACKEND[self.backend]

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
            "files": list(self.files),
            "working_side": self.working_side,
            "max_pixels": self.max_pixels,
        }


@dataclass(frozen=True)
class SegmentManifest:
    weights_family = "models"

    id: str
    family: str
    display: str
    kind: str
    licence: str
    licence_url: str
    commercial_use: str
    backends: dict[str, SegmentBackendSpec]
    path: Path

    @property
    def pull_command(self) -> str:
        return f"{MODELS_PULL_COMMAND} {self.id}"

    @property
    def prompted(self) -> bool:
        return self.kind in PROMPTED_KINDS

    def aliases(self) -> "tuple[SegmentManifest, ...]":
        return ()

    def supports(self, backend_kind: str) -> bool:
        return backend_kind in self.backends

    def spec(self, backend_kind: str) -> SegmentBackendSpec:
        found = self.backends.get(backend_kind)
        if found is None:
            raise SegmentManifestError(
                f"segment model {self.id!r} has no {backend_kind} block; "
                f"{self.path.name} declares {sorted(self.backends)}"
            )
        return found

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "display": self.display,
            "kind": self.kind,
            "licence": self.licence,
            "licence_url": self.licence_url,
            "commercial_use": self.commercial_use,
            "backends": {k: v.to_dict() for k, v in sorted(self.backends.items())},
        }


def segment_manifests_dir() -> Path:
    override = os.environ.get(SEGMENT_DIR_ENV)
    if override is not None and override != "":
        path = Path(override).expanduser()
        if not path.is_dir():
            raise SegmentManifestError(f"{SEGMENT_DIR_ENV}={override!r} is not a directory")
        return path
    path = Path(__file__).resolve().parent / "segment"
    if not path.is_dir():
        raise SegmentManifestError(
            f"no segment manifests at {path}; they are package data and this "
            f"install has lost them, or ${SEGMENT_DIR_ENV} must point at them"
        )
    return path


def _check_document(document: dict[str, Any], path: Path) -> None:
    unknown = sorted(set(document) - {"model", "backends"})
    if unknown:
        raise SegmentManifestError(
            f"{path.name}: unknown top-level table(s) {unknown}; a segment manifest "
            "has exactly [model] and [backends.<kind>]"
        )
    for table in ("model", "backends"):
        if not isinstance(document.get(table), dict):
            raise SegmentManifestError(f"{path.name}: missing the [{table}] table")
    if not document["backends"]:
        raise SegmentManifestError(
            f"{path.name}: no backend blocks; a segment model nothing can run is "
            "not a segment model"
        )


def _parse_model(model: dict[str, Any], path: Path, expected_id: str) -> str:
    check_table(f"{path.name} [model]", model, _MODEL_REQUIRED, error=SegmentManifestError)
    model_id = model["id"]
    if not MODEL_ID_PATTERN.match(model_id):
        raise SegmentManifestError(
            f"{path.name}: model.id {model_id!r} must be lower-case and start with "
            "a letter or digit ([a-z0-9][a-z0-9._-]*)"
        )
    if model_id != expected_id:
        raise SegmentManifestError(
            f"{path.name}: model.id is {model_id!r} but the file is named "
            f"{expected_id!r}; the id and the filename are the same thing"
        )
    if model["kind"] not in KINDS:
        raise SegmentManifestError(
            f"{path.name}: kind {model['kind']!r} is not one of {list(KINDS)}"
        )
    return model_id


def _check_engine(where: str, backend_kind: str, model_kind: str, block: dict[str, Any]) -> None:
    if backend_kind not in SEGMENT_BACKEND_ENGINES:
        raise SegmentManifestError(
            f"{where}: {backend_kind!r} is not a segment backend; the segment backends "
            f"are {sorted(SEGMENT_BACKEND_ENGINES)}. Windows is never one "
            "(docs/internals/segment.md, \"Backends and engines\")"
        )
    engine = block["engine"]
    if engine not in SEGMENT_BACKEND_ENGINES[backend_kind]:
        raise SegmentManifestError(
            f"{where}: engine {engine!r} does not run on {backend_kind}; that backend's "
            f"segment engines are {sorted(SEGMENT_BACKEND_ENGINES[backend_kind])}"
        )
    if ENGINE_KIND[engine] != model_kind:
        raise SegmentManifestError(
            f"{where}: engine {engine!r} makes {ENGINE_KIND[engine]!r} masks, and the "
            f"model's kind is {model_kind!r}"
        )
    if not HF_REPO_PATTERN.match(block["hf_repo"]):
        raise SegmentManifestError(
            f"{where}: hf_repo {block['hf_repo']!r} is not an <owner>/<name> "
            "HuggingFace repo id"
        )
    if not REVISION_PATTERN.match(block["revision"]):
        raise SegmentManifestError(
            f"{where}: revision {block['revision']!r} must be a full 40-character "
            "commit sha, so a pull is reproducible; branch names are not pins"
        )
    if block["dtype"] not in DTYPES:
        raise SegmentManifestError(
            f"{where}: dtype {block['dtype']!r} is not one of {sorted(DTYPES)}"
        )


def _check_memory(where: str, block: dict[str, Any]) -> None:
    if block["memory_bytes_estimate"] <= 0:
        raise SegmentManifestError(f"{where}: memory_bytes_estimate must be positive")
    if block["memory_basis"] not in MEMORY_BASES:
        raise SegmentManifestError(
            f"{where}: memory_basis {block['memory_basis']!r} is not one of "
            f"{sorted(MEMORY_BASES)}"
        )
    if not block["memory_note"].strip():
        raise SegmentManifestError(
            f"{where}: memory_note is empty; it says where the number came from"
        )


def _files(where: str, values: list[Any]) -> tuple[str, ...]:
    if not values or not all(isinstance(v, str) and v.strip() for v in values):
        raise SegmentManifestError(f"{where}: files must be a non-empty list of strings")
    return tuple(values)


def _check_sizes(where: str, block: dict[str, Any]) -> None:
    if block["working_side"] <= 0:
        raise SegmentManifestError(f"{where}: working_side must be positive")
    if block["max_pixels"] < block["working_side"]:
        raise SegmentManifestError(
            f"{where}: max_pixels {block['max_pixels']} admits almost no picture"
        )


def _parse_backend(path: Path, backend_kind: str, model_kind: str, block: Any) -> SegmentBackendSpec:
    where = f"{path.name} [backends.{backend_kind}]"
    if not isinstance(block, dict):
        raise SegmentManifestError(f"{where}: must be a table")
    check_table(where, block, _BACKEND_REQUIRED, error=SegmentManifestError)
    _check_engine(where, backend_kind, model_kind, block)
    _check_memory(where, block)
    _check_sizes(where, block)
    return SegmentBackendSpec(
        backend=backend_kind,
        engine=block["engine"],
        hf_repo=block["hf_repo"],
        revision=block["revision"],
        dtype=block["dtype"],
        memory_bytes_estimate=block["memory_bytes_estimate"],
        memory_basis=block["memory_basis"],
        memory_note=block["memory_note"],
        files=_files(where, block["files"]),
        working_side=block["working_side"],
        max_pixels=block["max_pixels"],
    )


def _parse(document: dict[str, Any], path: Path, expected_id: str) -> SegmentManifest:
    _check_document(document, path)
    model = document["model"]
    model_id = _parse_model(model, path, expected_id)
    backends = {
        backend_kind: _parse_backend(path, backend_kind, model["kind"], block)
        for backend_kind, block in document["backends"].items()
    }
    return SegmentManifest(
        id=model_id,
        family=model["family"],
        display=model["display"],
        kind=model["kind"],
        licence=model["licence"],
        licence_url=model["licence_url"],
        commercial_use=model["commercial_use"],
        backends=backends,
        path=path,
    )


def parse_segment_manifest(text: str, path: Path, expected_id: str) -> SegmentManifest:
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise SegmentManifestError(f"{path.name}: not valid TOML: {exc}") from exc
    return _parse(document, path, expected_id)


def load_segment_manifest(model_id: str, directory: Path | None = None) -> SegmentManifest:
    root = directory if directory is not None else segment_manifests_dir()
    path = root / f"{model_id}.toml"
    if not path.is_file():
        known = sorted(p.stem for p in root.glob("*.toml"))
        raise SegmentManifestError(
            f"no segment manifest for {model_id!r} at {path}; this build ships {known}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SegmentManifestError(f"could not read {path}: {exc}") from exc
    return parse_segment_manifest(text, path, model_id)


def load_all_segment_manifests(directory: Path | None = None) -> dict[str, SegmentManifest]:
    root = directory if directory is not None else segment_manifests_dir()
    return {
        path.stem: load_segment_manifest(path.stem, root)
        for path in sorted(root.glob("*.toml"), key=lambda p: p.stem)
    }


def _of_kind(kind: str, directory: Path | None) -> dict[str, SegmentManifest]:
    return {
        model_id: manifest
        for model_id, manifest in load_all_segment_manifests(directory).items()
        if manifest.kind == kind
    }


def load_cutout_manifests(directory: Path | None = None) -> dict[str, SegmentManifest]:
    return _of_kind(CUTOUT, directory)


def load_select_manifests(directory: Path | None = None) -> dict[str, SegmentManifest]:
    return _of_kind(SELECT, directory)


__all__ = [
    "BIREFNET",
    "CUTOUT",
    "DEVICE_FOR_BACKEND",
    "ENGINES",
    "ENGINE_KIND",
    "KINDS",
    "KIND_WORDS",
    "PROMPTED_KINDS",
    "SAM2",
    "SEGMENT_BACKEND_ENGINES",
    "SEGMENT_DIR_ENV",
    "SELECT",
    "SegmentBackendSpec",
    "SegmentManifest",
    "SegmentManifestError",
    "load_all_segment_manifests",
    "load_cutout_manifests",
    "load_segment_manifest",
    "load_select_manifests",
    "parse_segment_manifest",
    "segment_manifests_dir",
]
