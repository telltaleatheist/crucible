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

ALIGN_DIR_ENV = "CRUCIBLE_ALIGN_DIR"

ALIGN_BACKEND_ENGINES: dict[str, str] = {
    CUDA_LINUX: "qwen3-forced-aligner",
    MLX_DARWIN: "qwen3-forced-aligner",
}

_MODEL_REQUIRED: dict[str, type] = {
    "id": str,
    "family": str,
    "parameters_m": int,
}
_BACKEND_REQUIRED: dict[str, type] = {
    "engine": str,
    "hf_repo": str,
    "revision": str,
    "memory_bytes_estimate": int,
    "dtype": str,
}

DTYPES = frozenset({"bfloat16", "float16", "float32"})


class AlignManifestError(CrucibleError):
    ...


@dataclass(frozen=True)
class AlignBackendSpec:

    backend: str
    engine: str
    hf_repo: str
    revision: str
    memory_bytes_estimate: int
    dtype: str

    @property
    def files(self) -> tuple[str, ...]:
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "engine": self.engine,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class AlignManifest:
    weights_family = "models"

    id: str
    family: str
    parameters_m: int
    backends: dict[str, AlignBackendSpec]
    path: Path

    @property
    def pull_command(self) -> str:
        return f"{MODELS_PULL_COMMAND} {self.id}"

    def aliases(self) -> "tuple[AlignManifest, ...]":
        return ()

    def supports(self, backend_kind: str) -> bool:
        return backend_kind in self.backends

    def spec(self, backend_kind: str) -> AlignBackendSpec:
        found = self.backends.get(backend_kind)
        if found is None:
            raise AlignManifestError(
                f"aligner {self.id!r} has no {backend_kind} block; "
                f"{self.path.name} declares {sorted(self.backends)}"
            )
        return found

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "parameters_m": self.parameters_m,
            "backends": {k: v.to_dict() for k, v in sorted(self.backends.items())},
        }


def align_manifests_dir() -> Path:
    override = os.environ.get(ALIGN_DIR_ENV)
    if override is not None and override != "":
        path = Path(override).expanduser()
        if not path.is_dir():
            raise AlignManifestError(f"{ALIGN_DIR_ENV}={override!r} is not a directory")
        return path
    path = Path(__file__).resolve().parent / "align"
    if not path.is_dir():
        raise AlignManifestError(
            f"no align manifests at {path}; they are package data and this "
            f"install has lost them, or ${ALIGN_DIR_ENV} must point at them"
        )
    return path


def _parse(document: dict[str, Any], path: Path, expected_id: str) -> AlignManifest:
    unknown = sorted(set(document) - {"model", "backends"})
    if unknown:
        raise AlignManifestError(
            f"{path.name}: unknown top-level table(s) {unknown}; an align manifest "
            "has exactly [model] and [backends.<kind>]"
        )
    if "model" not in document:
        raise AlignManifestError(f"{path.name}: missing the [model] table")
    if "backends" not in document:
        raise AlignManifestError(f"{path.name}: missing every [backends.<kind>] table")

    model = document["model"]
    if not isinstance(model, dict):
        raise AlignManifestError(f"{path.name}: [model] must be a table")
    check_table(f"{path.name} [model]", model, _MODEL_REQUIRED, error=AlignManifestError)

    model_id = model["id"]
    if not MODEL_ID_PATTERN.match(model_id):
        raise AlignManifestError(
            f"{path.name}: model.id {model_id!r} must be lower-case and start with "
            "a letter or digit ([a-z0-9][a-z0-9._-]*)"
        )
    if model_id != expected_id:
        raise AlignManifestError(
            f"{path.name}: model.id is {model_id!r} but the file is named "
            f"{expected_id!r}; the id and the filename are the same thing"
        )
    if model["parameters_m"] <= 0:
        raise AlignManifestError(
            f"{path.name}: model.parameters_m must be positive, got "
            f"{model['parameters_m']}"
        )

    backends_table = document["backends"]
    if not isinstance(backends_table, dict):
        raise AlignManifestError(
            f"{path.name}: [backends] must hold one table per backend"
        )
    if not backends_table:
        raise AlignManifestError(
            f"{path.name}: no backend blocks; an aligner nothing can run is not an "
            "aligner"
        )

    backends: dict[str, AlignBackendSpec] = {}
    for kind, block in backends_table.items():
        where = f"{path.name} [backends.{kind}]"
        if kind not in ALIGN_BACKEND_ENGINES:
            raise AlignManifestError(
                f"{where}: {kind!r} is not an align backend; the align backends are "
                f"{sorted(ALIGN_BACKEND_ENGINES)}, and both run "
                "'qwen3-forced-aligner' on the same weights. Windows is never a "
                "backend (docs/PHASE15-HOST.md)"
            )
        if not isinstance(block, dict):
            raise AlignManifestError(f"{where}: must be a table")
        check_table(where, block, _BACKEND_REQUIRED, error=AlignManifestError)

        engine = block["engine"]
        if engine != ALIGN_BACKEND_ENGINES[kind]:
            raise AlignManifestError(
                f"{where}: engine {engine!r} does not align on {kind}; that "
                f"backend's align engine is {ALIGN_BACKEND_ENGINES[kind]!r}"
            )
        if not HF_REPO_PATTERN.match(block["hf_repo"]):
            raise AlignManifestError(
                f"{where}: hf_repo {block['hf_repo']!r} is not an <owner>/<name> "
                "HuggingFace repo id"
            )
        if not REVISION_PATTERN.match(block["revision"]):
            raise AlignManifestError(
                f"{where}: revision {block['revision']!r} must be a full 40-character "
                "commit sha, so a pull is reproducible; branch names are not pins"
            )
        if block["memory_bytes_estimate"] <= 0:
            raise AlignManifestError(
                f"{where}: memory_bytes_estimate must be positive, got "
                f"{block['memory_bytes_estimate']}"
            )
        if block["dtype"] not in DTYPES:
            raise AlignManifestError(
                f"{where}: dtype {block['dtype']!r} is not one of {sorted(DTYPES)}; "
                "it reaches the worker as `torch.<name>` and a name torch does not "
                "have is an AttributeError one model load later"
            )
        backends[kind] = AlignBackendSpec(
            backend=kind,
            engine=engine,
            hf_repo=block["hf_repo"],
            revision=block["revision"],
            memory_bytes_estimate=block["memory_bytes_estimate"],
            dtype=block["dtype"],
        )

    return AlignManifest(
        id=model_id,
        family=model["family"],
        parameters_m=model["parameters_m"],
        backends=backends,
        path=path,
    )


def parse_align_manifest(text: str, path: Path, expected_id: str) -> AlignManifest:
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise AlignManifestError(f"{path.name}: not valid TOML: {exc}") from exc
    return _parse(document, path, expected_id)


def load_align_manifest(model_id: str, directory: Path | None = None) -> AlignManifest:
    root = directory if directory is not None else align_manifests_dir()
    path = root / f"{model_id}.toml"
    if not path.is_file():
        known = sorted(p.stem for p in root.glob("*.toml"))
        raise AlignManifestError(
            f"no align manifest for {model_id!r} at {path}; this build ships {known}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AlignManifestError(f"could not read {path}: {exc}") from exc
    return parse_align_manifest(text, path, model_id)


def load_all_align_manifests(
    directory: Path | None = None,
) -> dict[str, AlignManifest]:
    root = directory if directory is not None else align_manifests_dir()
    manifests: dict[str, AlignManifest] = {}
    for path in sorted(root.glob("*.toml"), key=lambda p: p.stem):
        manifests[path.stem] = load_align_manifest(path.stem, root)
    return manifests
