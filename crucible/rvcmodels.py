"""RVC voice-model manifests — `rvc/<id>.toml` (PHASE4-AUDIO.md section 4).

**Model identity becomes a manifest**, which is the whole point of this file.
Today an RVC model is a *folder name* under
`<userData>/runtime/rvc-models/rvc/voice_models/<Name>`, discovered by looking for
a `.pth`, with `forceIndexRate0` derived from the **absence** of a `.index`
(`electron/rvc-models.ts:9`). That is a filesystem convention standing in for an
identity: it cannot be sent over a wire, it cannot be checked, and it does not
survive the trip to another machine.

Three things this schema has that no other manifest in the repo does, each
because of something that is true about how these models are actually published:

- **`archive`.** PHASE4-AUDIO.md section 4 says `owenmorgan/deathstalker_rvc_v1`
  is already published, and that turned out not to be the shape it is published
  in. There is no per-model repo. Every RVC model Owen has published is a
  `.tar.gz` under `rvc/` in ONE repo, `owenmorgan/owen-morgan-bookforge`
  (`electron/data/rvc-voice-assets.json`), alongside the XTTS weights. So a
  manifest names the repo, the revision AND the file, and `weights.pull_archive`
  fetches that one file rather than snapshot-downloading seven tarballs to get at
  one. The tarballs unpack to `rvc/voice_models/<name>/`, which is a whole
  `URVC_MODELS_DIR` root — convenient, and recorded here so the next reader knows
  it was checked rather than assumed.
- **`archive_sha256`.** A snapshot download is verified by the hub client against
  the revision; a single file fetched by path deserves the same assurance, and
  the app's catalog already carries the digest for every one of these, so it is a
  translation rather than an invention.
- **`has_index`.** This is `forceIndexRate0` said out loud. A model with no
  `.index` cannot do feature retrieval at all, so an `index_rate` above zero on
  one is not a preference the engine will ignore — it is a request for something
  that does not exist, and `jobs/rvc` refuses it by name.

Why this is not `crucible/manifests.py`
---------------------------------------
The same answer `asrmodels.py` and `alignmodels.py` give: these four loaders are
one loader parameterised by (directory, required keys, permitted engines), they
share about two hundred identical lines, and the merge is a follow-up rather than
something to do while three builders are in the tree.
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

RVC_DIR_ENV = "CRUCIBLE_RVC_DIR"

#: Which engine each backend is allowed to name. Both, and the same one: unlike
#: `asr`, ultimate-rvc is torch, torch has an MPS backend, and the app runs this
#: on Owen's Mac today — the 96-file recycle in `jobs/rvc` exists *because* of
#: what a 64 GB Mac did without it.
RVC_BACKEND_ENGINES: dict[str, str] = {
    CUDA_LINUX: "ultimate-rvc",
    MLX_DARWIN: "ultimate-rvc",
}

_MODEL_REQUIRED: dict[str, type] = {
    "id": str,
    "display": str,
    #: The folder name inside the archive, which is the name urvc is given on the
    #: command line. It is NOT derivable from the id: the id is Crucible's
    #: (lower-case, hyphenated) and the folder is whatever it was trained as
    #: ("Sigma Male Narrator", "US_Female_1", "deathstalker_rvc_v1").
    "model_name": str,
    #: `forceIndexRate0`, said as a fact about the model rather than inferred
    #: from a missing file. See the module docstring.
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

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_HF_REPO = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
#: A repo-relative path ending in `.tar.gz`. Anchored and traversal-free because
#: it becomes a path on this host's disk: `hf_hub_download` writes it under the
#: target, and `..` in a manifest must not be able to write outside it.
_ARCHIVE = re.compile(r"^(?!/)(?!.*(?:^|/)\.\.(?:/|$))[A-Za-z0-9._/-]+\.tar\.gz$")


class RvcManifestError(CrucibleError):
    """An RVC manifest is missing, unreadable, or does not say what it must."""


@dataclass(frozen=True)
class RvcBackendSpec:
    """One `[backends.<kind>]` block.

    `backend`, `hf_repo` and `revision` are named exactly as
    `crucible.manifests.BackendSpec` names them, so `weights.installed` and
    `weights.require_installed` read this without a branch for the job type. The
    archive keys are what `weights.pull_archive` needs on top, and they are the
    reason this cannot go through plain `weights.pull`.
    """

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
        """Empty, and NOT `(self.archive,)`.

        `crucible/weights.py`'s `WeightsSource` asks every spec which files a
        pull fetches and an `installed` requires, and this one is fetched by
        `pull_archive`: the archive is downloaded, verified, UNPACKED and
        removed, so naming it here would make `installed` look for a tarball
        that is correctly gone. What proves this subject complete is
        `pull_archive`'s own stamp, as it always was.
        """
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
    #: Its own subtree under `~/.crucible/`. Not `models` and not `voices`: an
    #: RVC model shares a namespace with neither, and `voices` in particular
    #: holds narrator checkpoints, which are a different thing with a confusingly
    #: similar name. `crucible/weights.py` explains why one directory holding two
    #: kinds is a way to overwrite 19 GB with 80 MB and leave a stamp that reads
    #: as installed to both.
    weights_family = "rvc"

    id: str
    display: str
    model_name: str
    has_index: bool
    backends: dict[str, RvcBackendSpec]
    path: Path

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


# ------------------------------------------------------------------ locating


def rvc_manifests_dir() -> Path:
    """Where `rvc/*.toml` live on this host. Refuses by name if absent."""
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


# ------------------------------------------------------------------ checking


def _check_table(where: str, table: dict[str, Any], required: dict[str, type]) -> None:
    """Every required key present and correctly typed; no key that is not listed."""
    unknown = sorted(set(table) - set(required))
    if unknown:
        raise RvcManifestError(
            f"{where}: unknown key(s) {unknown}; this table takes exactly "
            f"{sorted(required)}"
        )
    missing = sorted(set(required) - set(table))
    if missing:
        raise RvcManifestError(f"{where}: missing required key(s) {missing}")
    for key, kind in required.items():
        value = table[key]
        wrong = not isinstance(value, kind)
        # bool is a subclass of int; a bool where an int is wanted is still wrong.
        if kind is int and isinstance(value, bool):
            wrong = True
        if wrong:
            raise RvcManifestError(
                f"{where}: {key} must be {kind.__name__}, got {type(value).__name__}"
            )


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
    _check_table(f"{path.name} [model]", model, _MODEL_REQUIRED)

    model_id = model["id"]
    if not _MODEL_ID.match(model_id):
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
        _check_table(where, block, _BACKEND_REQUIRED)

        engine = block["engine"]
        if engine != RVC_BACKEND_ENGINES[kind]:
            raise RvcManifestError(
                f"{where}: engine {engine!r} does not convert on {kind}; that "
                f"backend's rvc engine is {RVC_BACKEND_ENGINES[kind]!r}"
            )
        if not _HF_REPO.match(block["hf_repo"]):
            raise RvcManifestError(
                f"{where}: hf_repo {block['hf_repo']!r} is not an <owner>/<name> "
                "HuggingFace repo id"
            )
        if not _REVISION.match(block["revision"]):
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
        if not _SHA256.match(block["archive_sha256"]):
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


# ------------------------------------------------------------------- loading


def parse_rvc_manifest(text: str, path: Path, expected_id: str) -> RvcManifest:
    """Parse and validate one RVC manifest's text. Raises by name."""
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise RvcManifestError(f"{path.name}: not valid TOML: {exc}") from exc
    return _parse(document, path, expected_id)


def load_rvc_manifest(model_id: str, directory: Path | None = None) -> RvcManifest:
    """Load `rvc/<model_id>.toml`. Raises RvcManifestError if it is not there."""
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
    """Every RVC manifest this build ships, by id, in id order."""
    root = directory if directory is not None else rvc_manifests_dir()
    manifests: dict[str, RvcManifest] = {}
    for path in sorted(root.glob("*.toml"), key=lambda p: p.stem):
        manifests[path.stem] = load_rvc_manifest(path.stem, root)
    return manifests
