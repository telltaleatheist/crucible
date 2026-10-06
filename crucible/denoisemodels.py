from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import weights
from .backend import CUDA_LINUX, MLX_DARWIN
from .config import Config
from .errors import CrucibleError
from .tomltable import (
    HF_REPO_PATTERN,
    MODEL_ID_PATTERN,
    REVISION_PATTERN,
    SHA256_PATTERN,
    check_table,
)

DENOISE_DIR_ENV = "CRUCIBLE_DENOISE_DIR"

DENOISE_MODELS_DIRNAME = "denoise-models"

PULL_COMMAND = "crucible denoise pull"

DENOISE_BACKEND_ENGINES: dict[str, str] = {
    CUDA_LINUX: "audio-separator",
    MLX_DARWIN: "audio-separator",
}

_MODEL_REQUIRED: dict[str, type] = {
    "id": str,
    "display": str,
    "model_filename": str,
    "config_filename": str,
    "primary_stem": str,
    "sample_rate": int,
    "overlap": int,
}
_BACKEND_REQUIRED: dict[str, type] = {
    "engine": str,
    "hf_repo": str,
    "revision": str,
    "model_path": str,
    "model_sha256": str,
    "model_bytes": int,
    "config_path": str,
    "config_sha256": str,
    "config_bytes": int,
    "memory_bytes_estimate": int,
}

_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._()+-]*$")


class DenoiseManifestError(CrucibleError):
    ...


@dataclass(frozen=True)
class DenoiseBackendSpec:
    backend: str
    engine: str
    hf_repo: str
    revision: str
    model_path: str
    model_sha256: str
    model_bytes: int
    config_path: str
    config_sha256: str
    config_bytes: int
    memory_bytes_estimate: int

    @property
    def total_bytes(self) -> int:
        return self.model_bytes + self.config_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "engine": self.engine,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "model_path": self.model_path,
            "model_sha256": self.model_sha256,
            "model_bytes": self.model_bytes,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "config_bytes": self.config_bytes,
            "memory_bytes_estimate": self.memory_bytes_estimate,
        }


@dataclass(frozen=True)
class DenoiseManifest:
    weights_family = "denoise"

    id: str
    display: str
    model_filename: str
    config_filename: str
    primary_stem: str
    sample_rate: int
    # audio-separator's MDXC overlap: how many overlapping windows each stretch of audio
    # is separated in and averaged over. Time is about proportional to it; the library's
    # default is 8.
    overlap: int
    backends: dict[str, DenoiseBackendSpec]
    path: Path

    @property
    def pull_command(self) -> str:
        return f"{PULL_COMMAND} {self.id}"

    def aliases(self) -> "tuple[DenoiseManifest, ...]":
        return ()

    def supports(self, backend_kind: str) -> bool:
        return backend_kind in self.backends

    def spec(self, backend_kind: str) -> DenoiseBackendSpec:
        found = self.backends.get(backend_kind)
        if found is None:
            raise DenoiseManifestError(
                f"denoise model {self.id!r} has no {backend_kind} block; "
                f"{self.path.name} declares {sorted(self.backends)}"
            )
        return found

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display": self.display,
            "model_filename": self.model_filename,
            "config_filename": self.config_filename,
            "primary_stem": self.primary_stem,
            "sample_rate": self.sample_rate,
            "overlap": self.overlap,
            "backends": {k: v.to_dict() for k, v in sorted(self.backends.items())},
        }


def denoise_manifests_dir() -> Path:
    override = os.environ.get(DENOISE_DIR_ENV)
    if override is not None and override != "":
        path = Path(override).expanduser()
        if not path.is_dir():
            raise DenoiseManifestError(
                f"{DENOISE_DIR_ENV}={override!r} is not a directory"
            )
        return path
    path = Path(__file__).resolve().parent / "denoise"
    if not path.is_dir():
        raise DenoiseManifestError(
            f"no denoise manifests at {path}; they are package data and this "
            f"install has lost them, or ${DENOISE_DIR_ENV} must point at them"
        )
    return path


def _filename(where: str, key: str, value: str) -> str:
    if not _FILENAME.match(value):
        raise DenoiseManifestError(
            f"{where}: {key} is {value!r}, which is not a plain filename. These "
            "are written into one flat directory an engine then reads BY NAME, "
            "so anything with a separator in it is a path escaping the directory "
            "rather than a model"
        )
    return value


def _parse(document: dict[str, Any], path: Path, expected_id: str) -> DenoiseManifest:
    unknown = sorted(set(document) - {"model", "backends"})
    if unknown:
        raise DenoiseManifestError(
            f"{path.name}: unknown top-level table(s) {unknown}; a denoise "
            "manifest has exactly [model] and [backends.<kind>]"
        )
    if "model" not in document:
        raise DenoiseManifestError(f"{path.name}: missing the [model] table")
    if "backends" not in document:
        raise DenoiseManifestError(f"{path.name}: missing every [backends.<kind>] table")

    model = document["model"]
    if not isinstance(model, dict):
        raise DenoiseManifestError(f"{path.name}: [model] must be a table")
    check_table(f"{path.name} [model]", model, _MODEL_REQUIRED, error=DenoiseManifestError)

    model_id = model["id"]
    if not MODEL_ID_PATTERN.match(model_id):
        raise DenoiseManifestError(
            f"{path.name}: model.id {model_id!r} must be lower-case and start "
            "with a letter or digit ([a-z0-9][a-z0-9._-]*)"
        )
    if model_id != expected_id:
        raise DenoiseManifestError(
            f"{path.name}: model.id is {model_id!r} but the file is named "
            f"{expected_id!r}; the id and the filename are the same thing"
        )
    _filename(f"{path.name} [model]", "model_filename", model["model_filename"])
    _filename(f"{path.name} [model]", "config_filename", model["config_filename"])
    if model["primary_stem"].strip() == "":
        raise DenoiseManifestError(
            f"{path.name}: model.primary_stem is empty; a separator model that "
            "does not say which of its outputs is the answer is a model the job "
            "cannot check"
        )
    if model["sample_rate"] <= 0:
        raise DenoiseManifestError(
            f"{path.name}: model.sample_rate must be positive, got "
            f"{model['sample_rate']}"
        )
    if model["overlap"] < 1:
        raise DenoiseManifestError(
            f"{path.name}: model.overlap must be at least 1, got {model['overlap']}"
        )

    backends_table = document["backends"]
    if not isinstance(backends_table, dict):
        raise DenoiseManifestError(
            f"{path.name}: [backends] must hold one table per backend"
        )
    if not backends_table:
        raise DenoiseManifestError(
            f"{path.name}: no backend blocks; a model nothing can serve is not a model"
        )

    backends: dict[str, DenoiseBackendSpec] = {}
    for kind, block in backends_table.items():
        where = f"{path.name} [backends.{kind}]"
        if kind not in DENOISE_BACKEND_ENGINES:
            raise DenoiseManifestError(
                f"{where}: {kind!r} is not a denoise backend; they are "
                f"{sorted(DENOISE_BACKEND_ENGINES)}"
            )
        if not isinstance(block, dict):
            raise DenoiseManifestError(f"{where}: must be a table")
        check_table(where, block, _BACKEND_REQUIRED, error=DenoiseManifestError)
        if block["engine"] != DENOISE_BACKEND_ENGINES[kind]:
            raise DenoiseManifestError(
                f"{where}: engine {block['engine']!r} does not denoise on {kind}; "
                f"that backend's engine is {DENOISE_BACKEND_ENGINES[kind]!r}"
            )
        if not HF_REPO_PATTERN.match(block["hf_repo"]):
            raise DenoiseManifestError(
                f"{where}: hf_repo {block['hf_repo']!r} is not an <owner>/<name> "
                "HuggingFace repo id"
            )
        if not REVISION_PATTERN.match(block["revision"]):
            raise DenoiseManifestError(
                f"{where}: revision {block['revision']!r} must be a full "
                "40-character commit sha, so a pull is reproducible; branch names "
                "are not pins"
            )
        for key in ("model_sha256", "config_sha256"):
            if not SHA256_PATTERN.match(block[key]):
                raise DenoiseManifestError(
                    f"{where}: {key} {block[key]!r} is not a 64-character sha256. "
                    "A single file fetched by path gets the assurance a snapshot "
                    "download gets from its revision, or it gets none"
                )
        for key in ("model_bytes", "config_bytes"):
            if block[key] <= 0:
                raise DenoiseManifestError(
                    f"{where}: {key} must be positive, got {block[key]}"
                )
        if block["memory_bytes_estimate"] <= 0:
            raise DenoiseManifestError(
                f"{where}: memory_bytes_estimate must be positive, got "
                f"{block['memory_bytes_estimate']}"
            )
        backends[kind] = DenoiseBackendSpec(
            backend=kind,
            engine=block["engine"],
            hf_repo=block["hf_repo"],
            revision=block["revision"],
            model_path=block["model_path"],
            model_sha256=block["model_sha256"],
            model_bytes=block["model_bytes"],
            config_path=block["config_path"],
            config_sha256=block["config_sha256"],
            config_bytes=block["config_bytes"],
            memory_bytes_estimate=block["memory_bytes_estimate"],
        )

    return DenoiseManifest(
        id=model_id,
        display=model["display"],
        model_filename=model["model_filename"],
        config_filename=model["config_filename"],
        primary_stem=model["primary_stem"],
        sample_rate=model["sample_rate"],
        overlap=model["overlap"],
        backends=backends,
        path=path,
    )


def parse_denoise_manifest(
    text: str, path: Path, expected_id: str
) -> DenoiseManifest:
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise DenoiseManifestError(f"{path.name}: not valid TOML: {exc}") from exc
    return _parse(document, path, expected_id)


def load_denoise_manifest(
    model_id: str, directory: Path | None = None
) -> DenoiseManifest:
    root = directory if directory is not None else denoise_manifests_dir()
    path = root / f"{model_id}.toml"
    if not path.is_file():
        known = sorted(p.stem for p in root.glob("*.toml"))
        raise DenoiseManifestError(
            f"no denoise manifest for {model_id!r} at {path}; this build ships "
            f"{known}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DenoiseManifestError(f"could not read {path}: {exc}") from exc
    return parse_denoise_manifest(text, path, model_id)


def load_all_denoise_manifests(
    directory: Path | None = None,
) -> dict[str, DenoiseManifest]:
    root = directory if directory is not None else denoise_manifests_dir()
    manifests: dict[str, DenoiseManifest] = {}
    for path in sorted(root.glob("*.toml"), key=lambda p: p.stem):
        manifests[path.stem] = load_denoise_manifest(path.stem, root)
    return manifests


@dataclass(frozen=True)
class DenoiseFile:
    source: str
    target: str
    sha256: str
    bytes: int
    why: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "why": self.why,
        }


def denoise_models_root(home: Path) -> Path:
    return home / DENOISE_MODELS_DIRNAME


def stamp_name(manifest: DenoiseManifest) -> str:
    return f"crucible-pull-{manifest.id}.json"


def model_files(
    manifest: DenoiseManifest, spec: DenoiseBackendSpec
) -> tuple[DenoiseFile, ...]:
    return (
        DenoiseFile(
            source=spec.model_path,
            target=manifest.model_filename,
            sha256=spec.model_sha256,
            bytes=spec.model_bytes,
            why=f"the separator checkpoint — {manifest.display}",
        ),
        DenoiseFile(
            source=spec.config_path,
            target=manifest.config_filename,
            sha256=spec.config_sha256,
            bytes=spec.config_bytes,
            why=(
                "the architecture YAML audio-separator loads beside the "
                "checkpoint; without it the checkpoint alone is unreadable"
            ),
        ),
    )


def missing(home: Path, manifest: DenoiseManifest) -> list[str]:
    root = denoise_models_root(home)
    return [
        name
        for name in (manifest.model_filename, manifest.config_filename)
        if not (root / name).is_file()
    ]


def installed(
    home: Path, manifest: DenoiseManifest, spec: DenoiseBackendSpec
) -> weights.InstalledWeights | None:
    return weights.files_installed(
        denoise_models_root(home),
        spec.hf_repo,
        spec.revision,
        stamp_name=stamp_name(manifest),
    )


def pull(
    config: Config,
    manifest: DenoiseManifest,
    spec: DenoiseBackendSpec,
    *,
    force: bool = False,
    on_line: Callable[[str], None] | None = None,
    on_progress: weights.ProgressHook | None = None,
) -> weights.InstalledWeights:
    return weights.pull_files(
        config,
        hf_repo=spec.hf_repo,
        revision=spec.revision,
        files=model_files(manifest, spec),
        target_root=denoise_models_root(config.home),
        label=f"the {manifest.id} separator",
        stamp_name=stamp_name(manifest),
        force=force,
        on_line=on_line,
        on_progress=on_progress,
    )


__all__ = [
    "DENOISE_BACKEND_ENGINES",
    "DENOISE_DIR_ENV",
    "DENOISE_MODELS_DIRNAME",
    "PULL_COMMAND",
    "DenoiseBackendSpec",
    "DenoiseFile",
    "DenoiseManifest",
    "DenoiseManifestError",
    "denoise_manifests_dir",
    "denoise_models_root",
    "installed",
    "load_all_denoise_manifests",
    "load_denoise_manifest",
    "missing",
    "model_files",
    "parse_denoise_manifest",
    "pull",
    "stamp_name",
]
