"""The score models: a recorded song in, a YuE2-native ABC score out (docs/SCORE.md).

A manifest per model in crucible/score/, the way segment and audio declare theirs. The one
model is SheetSage2: a 229 MB adapter and decoder (m-a-p/SheetSage2) on the MERT-v2-FullSong
encoder (m-a-p/MERT-v2-FullSong, 2.53 GB of float32 weights). The encoder is a companion
(crucible/companionweights.py): pulled from its own repo at the revision SheetSage2's config
names, every file hash-verified against this manifest, into `<model dir>/<name>/`, which the
worker hands to SheetSage2 as its `base_model_path`. SheetSage2's own loader checks the same
hashes again before it merges its adapters.
"""

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

SCORE_DIR_ENV = "CRUCIBLE_SCORE_DIR"

SHEETSAGE2 = "sheetsage2"

ENGINES: tuple[str, ...] = (SHEETSAGE2,)

SCORE_BACKEND_ENGINES: dict[str, frozenset[str]] = {
    CUDA_LINUX: frozenset(ENGINES),
    MLX_DARWIN: frozenset(ENGINES),
}

DEVICE_FOR_BACKEND: dict[str, str] = {
    CUDA_LINUX: "cuda",
    MLX_DARWIN: "mps",
}

# The companion SheetSage2 is an adapter on: the worker passes its directory as
# `base_model_path`, so the name is a contract between the manifest and the worker.
ENCODER_COMPANION = "mert"

MEMORY_BASES = frozenset({"measured", "declared"})

# What the weights are held in. SheetSage2 merges its adapters in float32 and runs float32
# weights everywhere; on CUDA it computes under a bfloat16 autocast (`autocast`).
DTYPES = frozenset({"float32"})

AUTOCASTS = frozenset({"bfloat16"})

_MODEL_REQUIRED: dict[str, type] = {
    "id": str,
    "family": str,
    "display": str,
    "licence": str,
    "licence_url": str,
    "commercial_use": str,
    "code_licence": str,
}
_BACKEND_REQUIRED: dict[str, Any] = {
    "engine": str,
    "hf_repo": str,
    "revision": str,
    "gated": bool,
    "dtype": str,
    "memory_bytes_estimate": int,
    "memory_basis": str,
    "memory_note": str,
    "files": list,
    "max_audio_s": int,
    "max_audio_note": str,
    "verified": bool,
    "companions": list,
}
_BACKEND_OPTIONAL: dict[str, Any] = {
    "autocast": str,
    "unverified_note": str,
}
_COMPANION_REQUIRED: dict[str, Any] = {
    "name": str,
    "hf_repo": str,
    "revision": str,
    "files": list,
}
_COMPANION_FILE_REQUIRED: dict[str, Any] = {
    "source": str,
    "target": str,
    "sha256": str,
    "bytes": int,
}


class ScoreManifestError(CrucibleError):
    ...


@dataclass(frozen=True)
class ScoreBackendSpec:

    backend: str
    engine: str
    hf_repo: str
    revision: str
    gated: bool
    dtype: str
    autocast: str | None
    memory_bytes_estimate: int
    memory_basis: str
    memory_note: str
    files: tuple[str, ...]
    max_audio_s: int
    max_audio_note: str
    # False until a real transcription has run on this backend through Crucible; the
    # Mac's arm is declared from the code, not from a run (docs/SCORE.md "Unverified").
    verified: bool
    unverified_note: str | None
    # Read by the companion weights layer (crucible/companionweights.py).
    companions: tuple[Companion, ...]

    @property
    def device(self) -> str:
        return DEVICE_FOR_BACKEND[self.backend]

    @property
    def encoder(self) -> Companion:
        return next(c for c in self.companions if c.name == ENCODER_COMPANION)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "engine": self.engine,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "gated": self.gated,
            "dtype": self.dtype,
            "autocast": self.autocast,
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "memory_basis": self.memory_basis,
            "memory_note": self.memory_note,
            "files": list(self.files),
            "max_audio_s": self.max_audio_s,
            "max_audio_note": self.max_audio_note,
            "verified": self.verified,
            "unverified_note": self.unverified_note,
            "companions": [companion.to_dict() for companion in self.companions],
        }


@dataclass(frozen=True)
class ScoreManifest:
    weights_family = "models"

    id: str
    family: str
    display: str
    licence: str
    licence_url: str
    commercial_use: str
    code_licence: str
    backends: dict[str, ScoreBackendSpec]
    path: Path

    @property
    def pull_command(self) -> str:
        return f"{MODELS_PULL_COMMAND} {self.id}"

    def aliases(self) -> "tuple[ScoreManifest, ...]":
        return ()

    def supports(self, backend_kind: str) -> bool:
        return backend_kind in self.backends

    def spec(self, backend_kind: str) -> ScoreBackendSpec:
        found = self.backends.get(backend_kind)
        if found is None:
            raise ScoreManifestError(
                f"score model {self.id!r} has no {backend_kind} block; "
                f"{self.path.name} declares {sorted(self.backends)}"
            )
        return found

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "display": self.display,
            "licence": self.licence,
            "licence_url": self.licence_url,
            "commercial_use": self.commercial_use,
            "code_licence": self.code_licence,
            "backends": {k: v.to_dict() for k, v in sorted(self.backends.items())},
        }


def score_manifests_dir() -> Path:
    override = os.environ.get(SCORE_DIR_ENV)
    if override is not None and override != "":
        path = Path(override).expanduser()
        if not path.is_dir():
            raise ScoreManifestError(f"{SCORE_DIR_ENV}={override!r} is not a directory")
        return path
    path = Path(__file__).resolve().parent / "score"
    if not path.is_dir():
        raise ScoreManifestError(
            f"no score manifests at {path}; they are package data and this "
            f"install has lost them, or ${SCORE_DIR_ENV} must point at them"
        )
    return path


def _check_document(document: dict[str, Any], path: Path) -> None:
    unknown = sorted(set(document) - {"model", "backends"})
    if unknown:
        raise ScoreManifestError(
            f"{path.name}: unknown top-level table(s) {unknown}; a score manifest "
            "has exactly [model] and [backends.<kind>]"
        )
    for table in ("model", "backends"):
        if not isinstance(document.get(table), dict):
            raise ScoreManifestError(f"{path.name}: missing the [{table}] table")
    if not document["backends"]:
        raise ScoreManifestError(
            f"{path.name}: no backend blocks; a score model nothing can run is not "
            "a score model"
        )


def _parse_model(model: dict[str, Any], path: Path, expected_id: str) -> str:
    check_table(f"{path.name} [model]", model, _MODEL_REQUIRED, error=ScoreManifestError)
    model_id = model["id"]
    if not MODEL_ID_PATTERN.match(model_id):
        raise ScoreManifestError(
            f"{path.name}: model.id {model_id!r} must be lower-case and start with "
            "a letter or digit ([a-z0-9][a-z0-9._-]*)"
        )
    if model_id != expected_id:
        raise ScoreManifestError(
            f"{path.name}: model.id is {model_id!r} but the file is named "
            f"{expected_id!r}; the id and the filename are the same thing"
        )
    return model_id


def _check_pin(where: str, hf_repo: str, revision: str) -> None:
    if not HF_REPO_PATTERN.match(hf_repo):
        raise ScoreManifestError(
            f"{where}: hf_repo {hf_repo!r} is not an <owner>/<name> HuggingFace repo id"
        )
    if not REVISION_PATTERN.match(revision):
        raise ScoreManifestError(
            f"{where}: revision {revision!r} must be a full 40-character commit sha, "
            "so a pull is reproducible; branch names are not pins"
        )


def _check_engine(where: str, backend_kind: str, block: dict[str, Any]) -> None:
    if backend_kind not in SCORE_BACKEND_ENGINES:
        raise ScoreManifestError(
            f"{where}: {backend_kind!r} is not a score backend; the score backends "
            f"are {sorted(SCORE_BACKEND_ENGINES)}. Windows is never one (docs/SCORE.md)"
        )
    if block["engine"] not in SCORE_BACKEND_ENGINES[backend_kind]:
        raise ScoreManifestError(
            f"{where}: engine {block['engine']!r} does not run on {backend_kind}; that "
            f"backend's score engines are {sorted(SCORE_BACKEND_ENGINES[backend_kind])}"
        )
    _check_pin(where, block["hf_repo"], block["revision"])
    if block["dtype"] not in DTYPES:
        raise ScoreManifestError(
            f"{where}: dtype {block['dtype']!r} is not one of {sorted(DTYPES)}; "
            "SheetSage2 merges its adapters into float32 weights and holds them so"
        )
    autocast = block.get("autocast")
    if autocast is not None and autocast not in AUTOCASTS:
        raise ScoreManifestError(
            f"{where}: autocast {autocast!r} is not one of {sorted(AUTOCASTS)}"
        )
    if autocast is not None and backend_kind != CUDA_LINUX:
        raise ScoreManifestError(
            f"{where}: autocast is CUDA's alone; SheetSage2 computes in float32 "
            "everywhere else, whatever is asked"
        )


def _check_memory(where: str, block: dict[str, Any]) -> None:
    if block["memory_bytes_estimate"] <= 0:
        raise ScoreManifestError(f"{where}: memory_bytes_estimate must be positive")
    if block["memory_basis"] not in MEMORY_BASES:
        raise ScoreManifestError(
            f"{where}: memory_basis {block['memory_basis']!r} is not one of "
            f"{sorted(MEMORY_BASES)}"
        )
    if not block["memory_note"].strip():
        raise ScoreManifestError(
            f"{where}: memory_note is empty; it says where the number came from"
        )


def _check_ceiling(where: str, block: dict[str, Any]) -> None:
    if block["max_audio_s"] <= 0:
        raise ScoreManifestError(f"{where}: max_audio_s must be positive")
    if not block["max_audio_note"].strip():
        raise ScoreManifestError(f"{where}: max_audio_note is empty; it says why the ceiling is where it is")
    if not block["verified"] and not (block.get("unverified_note") or "").strip():
        raise ScoreManifestError(
            f"{where}: an arm not verified by a real run says what is unverified "
            "(unverified_note)"
        )


def _strings(where: str, values: list[Any]) -> tuple[str, ...]:
    if not values or not all(isinstance(v, str) and v.strip() for v in values):
        raise ScoreManifestError(f"{where}: files must be a non-empty list of strings")
    return tuple(values)


def _companion_file(where: str, entry: Any) -> CompanionFile:
    if not isinstance(entry, dict):
        raise ScoreManifestError(f"{where}: each companion file is a table")
    check_table(where, entry, _COMPANION_FILE_REQUIRED, error=ScoreManifestError)
    if not SHA256_PATTERN.match(entry["sha256"]):
        raise ScoreManifestError(f"{where}: sha256 must be 64 lower-case hex digits")
    return CompanionFile(**entry)


def _companion(where: str, entry: Any) -> Companion:
    if not isinstance(entry, dict):
        raise ScoreManifestError(f"{where}: each companion is a table")
    check_table(where, entry, _COMPANION_REQUIRED, error=ScoreManifestError)
    _check_pin(where, entry["hf_repo"], entry["revision"])
    if entry["name"] != ENCODER_COMPANION:
        raise ScoreManifestError(
            f"{where}: companion {entry['name']!r} is not one a score block takes; it "
            f"takes {ENCODER_COMPANION!r}, the encoder SheetSage2 is an adapter on"
        )
    files = tuple(
        _companion_file(f"{where} files[{index}]", item)
        for index, item in enumerate(entry["files"])
    )
    if not files:
        raise ScoreManifestError(f"{where}: a companion with no files pulls nothing")
    return Companion(entry["name"], entry["hf_repo"], entry["revision"], files)


def _companions(where: str, block: dict[str, Any]) -> tuple[Companion, ...]:
    found = tuple(
        _companion(f"{where} companions[{index}]", entry)
        for index, entry in enumerate(block["companions"])
    )
    if [companion.name for companion in found] != [ENCODER_COMPANION]:
        raise ScoreManifestError(
            f"{where}: a score block has exactly one companion, {ENCODER_COMPANION!r}"
        )
    return found


def _parse_backend(path: Path, backend_kind: str, block: Any) -> ScoreBackendSpec:
    where = f"{path.name} [backends.{backend_kind}]"
    if not isinstance(block, dict):
        raise ScoreManifestError(f"{where}: must be a table")
    check_table(where, block, _BACKEND_REQUIRED, _BACKEND_OPTIONAL, error=ScoreManifestError)
    _check_engine(where, backend_kind, block)
    _check_memory(where, block)
    _check_ceiling(where, block)
    return ScoreBackendSpec(
        backend=backend_kind,
        engine=block["engine"],
        hf_repo=block["hf_repo"],
        revision=block["revision"],
        gated=block["gated"],
        dtype=block["dtype"],
        autocast=block.get("autocast"),
        memory_bytes_estimate=block["memory_bytes_estimate"],
        memory_basis=block["memory_basis"],
        memory_note=block["memory_note"],
        files=_strings(where, block["files"]),
        max_audio_s=block["max_audio_s"],
        max_audio_note=block["max_audio_note"],
        verified=block["verified"],
        unverified_note=block.get("unverified_note"),
        companions=_companions(where, block),
    )


def _parse(document: dict[str, Any], path: Path, expected_id: str) -> ScoreManifest:
    _check_document(document, path)
    model = document["model"]
    model_id = _parse_model(model, path, expected_id)
    backends = {
        backend_kind: _parse_backend(path, backend_kind, block)
        for backend_kind, block in document["backends"].items()
    }
    return ScoreManifest(
        id=model_id,
        family=model["family"],
        display=model["display"],
        licence=model["licence"],
        licence_url=model["licence_url"],
        commercial_use=model["commercial_use"],
        code_licence=model["code_licence"],
        backends=backends,
        path=path,
    )


def parse_score_manifest(text: str, path: Path, expected_id: str) -> ScoreManifest:
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ScoreManifestError(f"{path.name}: not valid TOML: {exc}") from exc
    return _parse(document, path, expected_id)


def load_score_manifest(model_id: str, directory: Path | None = None) -> ScoreManifest:
    root = directory if directory is not None else score_manifests_dir()
    path = root / f"{model_id}.toml"
    if not path.is_file():
        known = sorted(p.stem for p in root.glob("*.toml"))
        raise ScoreManifestError(
            f"no score manifest for {model_id!r} at {path}; this build ships {known}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScoreManifestError(f"could not read {path}: {exc}") from exc
    return parse_score_manifest(text, path, model_id)


def load_all_score_manifests(directory: Path | None = None) -> dict[str, ScoreManifest]:
    root = directory if directory is not None else score_manifests_dir()
    return {
        path.stem: load_score_manifest(path.stem, root)
        for path in sorted(root.glob("*.toml"), key=lambda p: p.stem)
    }


__all__ = [
    "DEVICE_FOR_BACKEND",
    "ENCODER_COMPANION",
    "ENGINES",
    "SCORE_BACKEND_ENGINES",
    "SCORE_DIR_ENV",
    "SHEETSAGE2",
    "ScoreBackendSpec",
    "ScoreManifest",
    "ScoreManifestError",
    "load_all_score_manifests",
    "load_score_manifest",
    "parse_score_manifest",
    "score_manifests_dir",
]
