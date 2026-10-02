from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backend import CUDA_LINUX, MLX_DARWIN
from .errors import CrucibleError
from .tomltable import (
    HF_REPO_PATTERN,
    MODEL_ID_PATTERN,
    REVISION_PATTERN,
    SHA256_PATTERN,
    check_table,
)

RVC_DIR_ENV = "CRUCIBLE_RVC_DIR"

RVC_BACKEND_ENGINES: dict[str, str] = {
    CUDA_LINUX: "ultimate-rvc",
    MLX_DARWIN: "ultimate-rvc",
}

_MODEL_REQUIRED: dict[str, type] = {
    "id": str,
    "display": str,
    "model_name": str,
    "has_index": bool,
}
_BACKEND_REQUIRED: dict[str, type] = {
    "engine": str,
    "hf_repo": str,
    "revision": str,
    "archive": str,
    "archive_sha256": str,
    "archive_bytes": int,
    "memory_bytes_estimate": int,
}

_ARCHIVE = re.compile(r"^(?!/)(?!.*(?:^|/)\.\.(?:/|$))[A-Za-z0-9._/-]+\.tar\.gz$")


PULL_COMMAND = "crucible rvc pull"


class RvcManifestError(CrucibleError):
    ...


@dataclass(frozen=True)
class RvcBackendSpec:
    backend: str
    engine: str
    hf_repo: str
    revision: str
    archive: str
    archive_sha256: str
    archive_bytes: int
    memory_bytes_estimate: int

    @property
    def files(self) -> tuple[str, ...]:
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "engine": self.engine,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "archive": self.archive,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "memory_bytes_estimate": self.memory_bytes_estimate,
        }


@dataclass(frozen=True)
class RvcManifest:
    weights_family = "rvc"

    id: str
    display: str
    model_name: str
    has_index: bool
    backends: dict[str, RvcBackendSpec]
    path: Path

    @property
    def pull_command(self) -> str:
        return f"{PULL_COMMAND} {self.id}"

    def aliases(self) -> "tuple[RvcManifest, ...]":
        return ()

    def supports(self, backend_kind: str) -> bool:
        return backend_kind in self.backends

    def spec(self, backend_kind: str) -> RvcBackendSpec:
        found = self.backends.get(backend_kind)
        if found is None:
            raise RvcManifestError(
                f"RVC model {self.id!r} has no {backend_kind} block; "
                f"{self.path.name} declares {sorted(self.backends)}"
            )
        return found

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display": self.display,
            "model_name": self.model_name,
            "has_index": self.has_index,
            "backends": {k: v.to_dict() for k, v in sorted(self.backends.items())},
        }


def rvc_manifests_dir() -> Path:
    override = os.environ.get(RVC_DIR_ENV)
    if override is not None and override != "":
        path = Path(override).expanduser()
        if not path.is_dir():
            raise RvcManifestError(f"{RVC_DIR_ENV}={override!r} is not a directory")
        return path
    path = Path(__file__).resolve().parent / "rvc"
    if not path.is_dir():
        raise RvcManifestError(
            f"no RVC manifests at {path}; they are package data and this "
            f"install has lost them, or ${RVC_DIR_ENV} must point at them"
        )
    return path


def _parse(document: dict[str, Any], path: Path, expected_id: str) -> RvcManifest:
    unknown = sorted(set(document) - {"model", "backends"})
    if unknown:
        raise RvcManifestError(
            f"{path.name}: unknown top-level table(s) {unknown}; an RVC manifest "
            "has exactly [model] and [backends.<kind>]"
        )
    if "model" not in document:
        raise RvcManifestError(f"{path.name}: missing the [model] table")
    if "backends" not in document:
        raise RvcManifestError(f"{path.name}: missing every [backends.<kind>] table")

    model = document["model"]
    if not isinstance(model, dict):
        raise RvcManifestError(f"{path.name}: [model] must be a table")
    check_table(f"{path.name} [model]", model, _MODEL_REQUIRED, error=RvcManifestError)

    model_id = model["id"]
    if not MODEL_ID_PATTERN.match(model_id):
        raise RvcManifestError(
            f"{path.name}: model.id {model_id!r} must be lower-case and start with "
            "a letter or digit ([a-z0-9][a-z0-9._-]*)"
        )
    if model_id != expected_id:
        raise RvcManifestError(
            f"{path.name}: model.id is {model_id!r} but the file is named "
            f"{expected_id!r}; the id and the filename are the same thing"
        )
    model_name = model["model_name"]
    if model_name.strip() == "" or "/" in model_name or "\\" in model_name:
        raise RvcManifestError(
            f"{path.name}: model.model_name {model_name!r} must be the single "
            "folder name inside the archive — it becomes a path member and a "
            "command-line argument"
        )

    backends_table = document["backends"]
    if not isinstance(backends_table, dict):
        raise RvcManifestError(f"{path.name}: [backends] must hold one table per backend")
    if not backends_table:
        raise RvcManifestError(
            f"{path.name}: no backend blocks; a model nothing can run is not a model"
        )

    backends: dict[str, RvcBackendSpec] = {}
    for kind, block in backends_table.items():
        where = f"{path.name} [backends.{kind}]"
        if kind not in RVC_BACKEND_ENGINES:
            raise RvcManifestError(
                f"{where}: {kind!r} is not an rvc backend; the rvc backends are "
                f"{sorted(RVC_BACKEND_ENGINES)}"
            )
        if not isinstance(block, dict):
            raise RvcManifestError(f"{where}: must be a table")
        check_table(where, block, _BACKEND_REQUIRED, error=RvcManifestError)

        engine = block["engine"]
        if engine != RVC_BACKEND_ENGINES[kind]:
            raise RvcManifestError(
                f"{where}: engine {engine!r} does not convert on {kind}; that "
                f"backend's rvc engine is {RVC_BACKEND_ENGINES[kind]!r}"
            )
        if not HF_REPO_PATTERN.match(block["hf_repo"]):
            raise RvcManifestError(
                f"{where}: hf_repo {block['hf_repo']!r} is not an <owner>/<name> "
                "HuggingFace repo id"
            )
        if not REVISION_PATTERN.match(block["revision"]):
            raise RvcManifestError(
                f"{where}: revision {block['revision']!r} must be a full 40-character "
                "commit sha, so a pull is reproducible; branch names are not pins"
            )
        if not _ARCHIVE.match(block["archive"]):
            raise RvcManifestError(
                f"{where}: archive {block['archive']!r} must be a repo-relative "
                "`.tar.gz` path with no leading slash and no `..`; it is fetched by "
                "name and unpacked onto this host's disk"
            )
        if not SHA256_PATTERN.match(block["archive_sha256"]):
            raise RvcManifestError(
                f"{where}: archive_sha256 {block['archive_sha256']!r} must be 64 "
                "lower-case hex characters"
            )
        if block["archive_bytes"] <= 0:
            raise RvcManifestError(
                f"{where}: archive_bytes must be positive, got "
                f"{block['archive_bytes']}"
            )
        if block["memory_bytes_estimate"] <= 0:
            raise RvcManifestError(
                f"{where}: memory_bytes_estimate must be positive, got "
                f"{block['memory_bytes_estimate']}"
            )
        backends[kind] = RvcBackendSpec(
            backend=kind,
            engine=engine,
            hf_repo=block["hf_repo"],
            revision=block["revision"],
            archive=block["archive"],
            archive_sha256=block["archive_sha256"],
            archive_bytes=block["archive_bytes"],
            memory_bytes_estimate=block["memory_bytes_estimate"],
        )

    return RvcManifest(
        id=model_id,
        display=model["display"],
        model_name=model_name,
        has_index=model["has_index"],
        backends=backends,
        path=path,
    )


def parse_rvc_manifest(text: str, path: Path, expected_id: str) -> RvcManifest:
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise RvcManifestError(f"{path.name}: not valid TOML: {exc}") from exc
    return _parse(document, path, expected_id)


def load_rvc_manifest(model_id: str, directory: Path | None = None) -> RvcManifest:
    root = directory if directory is not None else rvc_manifests_dir()
    path = root / f"{model_id}.toml"
    if not path.is_file():
        known = sorted(p.stem for p in root.glob("*.toml"))
        raise RvcManifestError(
            f"no RVC manifest for {model_id!r} at {path}; this build ships {known}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RvcManifestError(f"could not read {path}: {exc}") from exc
    return parse_rvc_manifest(text, path, model_id)


def load_all_rvc_manifests(directory: Path | None = None) -> dict[str, RvcManifest]:
    root = directory if directory is not None else rvc_manifests_dir()
    manifests: dict[str, RvcManifest] = {}
    for path in sorted(root.glob("*.toml"), key=lambda p: p.stem):
        manifests[path.stem] = load_rvc_manifest(path.stem, root)
    return manifests
