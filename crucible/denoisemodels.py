"""Denoise model manifests — `denoise/<id>.toml` (PHASE4-AUDIO.md section 4.2).

One file per separator model. The shape is `crucible/rvcmodels.py`'s rather than
`crucible/manifests.py`': what a separator needs is **two named files in one
directory**, not a repo snapshot, so a block names the repo, the revision, and
the path of each file inside it, with a digest for each.

Why the filenames are declared and not derived
----------------------------------------------
audio-separator resolves a model by **filename** inside its `model_file_dir`,
against a registry it fetches from GitHub, and it expects the checkpoint and its
YAML config under exactly the names that registry lists:

    denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt
    denoise_mel_band_roformer_aufr33_sdr_27.9959_config.yaml

The HuggingFace mirror this manifest points at stores the same two files under
*different* paths and, for the config, a different name. So the manifest carries
both halves — where the bytes come from (`model_path`, `config_path`) and what
they must be called when they land (`model_filename`, `config_filename`) — and
nothing in the code invents either. A wrong name here is a model audio-separator
would try to download over the top of.

Validation is strict for `crucible/asrmodels.py`'s reason. A separator run with
the wrong checkpoint produces audio that sounds nearly right, and the manifest is
the only place that says which weights were used.

Why this is a third manifest loader
-----------------------------------
It should not be. This is now the fourth file in this repo that reads a strict
TOML manifest with a `[model]` table and `[backends.<kind>]` blocks
(`manifests.py`, `asrmodels.py`, `alignmodels.py`, `rvcmodels.py` — and this),
and every one of them says the same thing in its own docstring: merging them
into one loader parameterised by (directory, required keys, permitted engines)
is a mechanical follow-up that nobody should do while other builders are in the
tree. The duplication is deliberate and it is getting expensive.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backend import CUDA_LINUX, MLX_DARWIN
from .errors import CrucibleError

DENOISE_DIR_ENV = "CRUCIBLE_DENOISE_DIR"

#: Both backends run the same engine: audio-separator is torch, torch has an MPS
#: backend, and a separator checkpoint is not quantised per platform. The one
#: thing that differs is `use_autocast`, which is CUDA-only and is read off the
#: backend in `crucible/jobs/denoise/__init__.py` rather than declared here.
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
    "memory_bytes_estimate": int,
}

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_HF_REPO = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
#: A filename, not a path. These are written into one flat directory that
#: audio-separator reads by name, so a separator would happily be handed
#: `../../etc/passwd` by a manifest nobody checked.
_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._()+-]*$")


class DenoiseManifestError(CrucibleError):
    """A denoise manifest is missing, unreadable, or does not say what it must."""


@dataclass(frozen=True)
class DenoiseBackendSpec:
    """One `[backends.<kind>]` block.

    `backend`, `hf_repo` and `revision` are named the way
    `crucible/weights.py`'s protocols name them, so whatever eventually pulls
    these two files goes through the one weights module rather than a second
    downloader (see `crucible/jobs/denoise/__init__.py` on what does not exist
    yet).
    """

    backend: str
    engine: str
    hf_repo: str
    revision: str
    model_path: str
    model_sha256: str
    model_bytes: int
    config_path: str
    config_sha256: str
    memory_bytes_estimate: int

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
            "memory_bytes_estimate": self.memory_bytes_estimate,
        }


@dataclass(frozen=True)
class DenoiseManifest:
    #: Its own subtree of `~/.crucible/`, for `rvcmodels.py`'s reason: these are
    #: named files in a flat directory that an engine reads by name, not a repo
    #: snapshot, and the ids are their own namespace.
    weights_family = "denoise"

    id: str
    display: str
    #: What audio-separator must find in `model_file_dir`. See the module
    #: docstring: the names it resolves by are not the paths they come from.
    model_filename: str
    config_filename: str
    #: The stem this model exists to produce — `dry` for the denoiser, which is
    #: "the signal minus the noise". The job refuses a run that did not produce
    #: exactly one output naming it.
    primary_stem: str
    #: The rate the model was trained at and the ONLY rate it may be fed. Its
    #: librosa front-end crashes on others, and a stem that came back at a
    #: different rate has invalidated every offset the client sliced by.
    sample_rate: int
    backends: dict[str, DenoiseBackendSpec]
    path: Path

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
            "backends": {k: v.to_dict() for k, v in sorted(self.backends.items())},
        }


# ------------------------------------------------------------------ locating


def denoise_manifests_dir() -> Path:
    """Where `denoise/*.toml` live on this host. Refuses by name if absent."""
    override = os.environ.get(DENOISE_DIR_ENV)
    if override is not None and override != "":
        path = Path(override).expanduser()
        if not path.is_dir():
            raise DenoiseManifestError(
                f"{DENOISE_DIR_ENV}={override!r} is not a directory"
            )
        return path
    path = Path(__file__).resolve().parent.parent / "denoise"
    if not path.is_dir():
        raise DenoiseManifestError(
            f"no denoise manifests at {path}; crucible must run from a checkout "
            f"(pip install -e .) or ${DENOISE_DIR_ENV} must point at the manifests"
        )
    return path


# ------------------------------------------------------------------ checking


def _check_table(where: str, table: dict[str, Any], required: dict[str, type]) -> None:
    unknown = sorted(set(table) - set(required))
    if unknown:
        raise DenoiseManifestError(
            f"{where}: unknown key(s) {unknown}; this table takes exactly "
            f"{sorted(required)}"
        )
    missing = sorted(set(required) - set(table))
    if missing:
        raise DenoiseManifestError(f"{where}: missing required key(s) {missing}")
    for key, kind in required.items():
        value = table[key]
        wrong = not isinstance(value, kind)
        if kind is int and isinstance(value, bool):
            wrong = True
        if wrong:
            raise DenoiseManifestError(
                f"{where}: {key} must be {kind.__name__}, got {type(value).__name__}"
            )


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
    _check_table(f"{path.name} [model]", model, _MODEL_REQUIRED)

    model_id = model["id"]
    if not _MODEL_ID.match(model_id):
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
        _check_table(where, block, _BACKEND_REQUIRED)
        if block["engine"] != DENOISE_BACKEND_ENGINES[kind]:
            raise DenoiseManifestError(
                f"{where}: engine {block['engine']!r} does not denoise on {kind}; "
                f"that backend's engine is {DENOISE_BACKEND_ENGINES[kind]!r}"
            )
        if not _HF_REPO.match(block["hf_repo"]):
            raise DenoiseManifestError(
                f"{where}: hf_repo {block['hf_repo']!r} is not an <owner>/<name> "
                "HuggingFace repo id"
            )
        if not _REVISION.match(block["revision"]):
            raise DenoiseManifestError(
                f"{where}: revision {block['revision']!r} must be a full "
                "40-character commit sha, so a pull is reproducible; branch names "
                "are not pins"
            )
        for key in ("model_sha256", "config_sha256"):
            if not _SHA256.match(block[key]):
                raise DenoiseManifestError(
                    f"{where}: {key} {block[key]!r} is not a 64-character sha256. "
                    "A single file fetched by path gets the assurance a snapshot "
                    "download gets from its revision, or it gets none"
                )
        if block["model_bytes"] <= 0:
            raise DenoiseManifestError(
                f"{where}: model_bytes must be positive, got {block['model_bytes']}"
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
            memory_bytes_estimate=block["memory_bytes_estimate"],
        )

    return DenoiseManifest(
        id=model_id,
        display=model["display"],
        model_filename=model["model_filename"],
        config_filename=model["config_filename"],
        primary_stem=model["primary_stem"],
        sample_rate=model["sample_rate"],
        backends=backends,
        path=path,
    )


# ------------------------------------------------------------------- loading


def parse_denoise_manifest(
    text: str, path: Path, expected_id: str
) -> DenoiseManifest:
    """Parse and validate one denoise manifest. Raises by name."""
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise DenoiseManifestError(f"{path.name}: not valid TOML: {exc}") from exc
    return _parse(document, path, expected_id)


def load_denoise_manifest(
    model_id: str, directory: Path | None = None
) -> DenoiseManifest:
    """Load `denoise/<model_id>.toml`. Raises if it is not there."""
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
    """Every denoise manifest this build ships, by id, in id order."""
    root = directory if directory is not None else denoise_manifests_dir()
    manifests: dict[str, DenoiseManifest] = {}
    for path in sorted(root.glob("*.toml"), key=lambda p: p.stem):
        manifests[path.stem] = load_denoise_manifest(path.stem, root)
    return manifests


__all__ = [
    "DENOISE_BACKEND_ENGINES",
    "DENOISE_DIR_ENV",
    "DenoiseBackendSpec",
    "DenoiseManifest",
    "DenoiseManifestError",
    "denoise_manifests_dir",
    "load_all_denoise_manifests",
    "load_denoise_manifest",
    "parse_denoise_manifest",
]
