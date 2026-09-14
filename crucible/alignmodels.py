"""Forced-aligner manifests — `align/<id>.toml` (PHASE4-AUDIO.md section 2).

One file per Crucible aligner id, the same shape `models/<id>.toml` and
`asr/<id>.toml` have: a stable id, weights per backend, and a full commit sha so
a pull is reproducible.

Why this is a third directory and not a row in `models/`
--------------------------------------------------------
PHASE4-AUDIO.md section 2 says the aligner "gets a manifest like any other model,
with `job_type = "align"`". It does not, and cannot: `crucible/manifests.py`
requires `params_b`, `context_default` and `modalities` on every `[model]` table
and refuses any engine but `vllm` or `mlx-lm`, because those manifests describe
things an OpenAI-compatible engine serves. A forced aligner has no context, no
modalities in that sense and no engine of that kind — it is a checkpoint a worker
imports — so putting it in `models/` would mean either inventing three numbers
that mean nothing or loosening a schema that is strict for good reasons.

There is no `job_type` key anywhere in this repo's manifests, either. **The
directory is the job type**, which is how `asr/` already works, and it is the
better arrangement: nothing can declare `job_type = "llm"` in `align/` and be
half-believed by two loaders.

Both backends run the same engine, and that is the point
---------------------------------------------------------
Qwen3-ForcedAligner is a torch model and torch has an MPS backend, so unlike
`asr` the Mac needs no second engine, no second worker and no second set of
weights: `ALIGN_BACKEND_ENGINES` maps both backends to `qwen3-forced-aligner`
and `align/qwen3-aligner.toml` pins the identical repo and revision on both.
What changes per backend is the DEVICE the worker is told to load onto, which
`crucible/jobs/align/__init__.py` owns.

Until 2026-09-14 this module shipped `cuda-linux` alone, and the reason it gave
was that nobody had measured the aligner on Metal. Half of that is discharged
and half is not, which is why both halves are written down here rather than one
of them being quietly dropped: BookForge measured 97x realtime warm on MPS in
bfloat16 on the M1 Ultra on 2026-09-08 (`electron/components/qwen-align-env.ts`)
in the very env `envs/align/mlx-darwin.txt` is the freeze of — so the recipe is
a real one — while the TIMESTAMP comparison `envs/align/mlx-darwin.md` asks for
has still not been run. `bfloat16` on MPS is a different numerical path from
`bfloat16` on CUDA, and until one chapter is aligned on both machines and the
cues compared, nobody can say the two agree. The block ships because the engine,
the env and the speed are real; the comparison is named as owed in that file and
in `docs/PHASE15-HOST.md` 7c rather than implied to have happened.

Why this is not `crucible/manifests.py`
---------------------------------------
It should be, and so should `asrmodels.py`; the three are one loader
parameterised by (directory, required keys, permitted engines) and they share
about two hundred identical lines. `asrmodels.py` says the same thing at the same
length and for the same reason: phase 4 was built beside phases 2 and 3 in one
tree. The merge is a follow-up, and doing it now would be rewriting two other
builders' files under them.
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

ALIGN_DIR_ENV = "CRUCIBLE_ALIGN_DIR"

#: Which engine each backend is allowed to name. ONE engine for both, which is
#: the whole shape of this job type on the Mac — see the module docstring.
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
    #: `bfloat16` on an accelerator; the manifest says it rather than the code
    #: guessing it, because it is a property of what the bake-off measured.
    "dtype": str,
}

#: The dtypes a backend block may name. Not an open string: a manifest that said
#: `bf16` or `torch.bfloat16` would reach `torch.<name>` as an AttributeError
#: inside the worker, minutes and one model load later.
DTYPES = frozenset({"bfloat16", "float16", "float32"})

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_HF_REPO = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


class AlignManifestError(CrucibleError):
    """An align manifest is missing, unreadable, or does not say what it must."""


@dataclass(frozen=True)
class AlignBackendSpec:
    """One `[backends.<kind>]` block.

    The first four field names are `crucible.manifests.BackendSpec`'s on purpose:
    `crucible/weights.py` reads `spec.hf_repo`, `spec.revision` and
    `spec.backend` and nothing else, so aligner weights are pulled and stamped by
    the one weights module every other model goes through, with no branch in it
    for this job type.
    """

    backend: str
    engine: str
    hf_repo: str
    revision: str
    memory_bytes_estimate: int
    dtype: str

    @property
    def files(self) -> tuple[str, ...]:
        """Empty: this backend fetches the WHOLE repo.

        `crucible/weights.py`'s `WeightsSource` asks every spec this, and the
        empty tuple is a real answer and not a gap — it is what "there is no
        file to choose, the repository IS the weights" reads as. Only
        `llama-windows` names files (one GGUF, and a projector beside it for a
        vision model), because a GGUF repo holds twenty quantizations and
        pulling all of them is hundreds of gigabytes.
        """
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
    #: Which tree under `~/.crucible/` these weights live in — `models`, with the
    #: llm and asr manifests, so that `crucible models list` shows every set of
    #: weights this server can be asked to fetch and there is one command to
    #: learn rather than three. Not a dataclass field: a property of the kind,
    #: not of the file.
    weights_family = "models"

    id: str
    family: str
    parameters_m: int
    backends: dict[str, AlignBackendSpec]
    path: Path

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


# ------------------------------------------------------------------ locating


def align_manifests_dir() -> Path:
    """Where `align/*.toml` live on this host. Refuses by name if absent."""
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


# ------------------------------------------------------------------ checking


def _check_table(where: str, table: dict[str, Any], required: dict[str, type]) -> None:
    """Every required key present and correctly typed; no key that is not listed."""
    unknown = sorted(set(table) - set(required))
    if unknown:
        raise AlignManifestError(
            f"{where}: unknown key(s) {unknown}; this table takes exactly "
            f"{sorted(required)}"
        )
    missing = sorted(set(required) - set(table))
    if missing:
        raise AlignManifestError(f"{where}: missing required key(s) {missing}")
    for key, kind in required.items():
        value = table[key]
        wrong = not isinstance(value, kind)
        # bool is a subclass of int; a bool where an int is wanted is still wrong.
        if kind is int and isinstance(value, bool):
            wrong = True
        if wrong:
            raise AlignManifestError(
                f"{where}: {key} must be {kind.__name__}, got {type(value).__name__}"
            )


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
    _check_table(f"{path.name} [model]", model, _MODEL_REQUIRED)

    model_id = model["id"]
    if not _MODEL_ID.match(model_id):
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
        _check_table(where, block, _BACKEND_REQUIRED)

        engine = block["engine"]
        if engine != ALIGN_BACKEND_ENGINES[kind]:
            raise AlignManifestError(
                f"{where}: engine {engine!r} does not align on {kind}; that "
                f"backend's align engine is {ALIGN_BACKEND_ENGINES[kind]!r}"
            )
        if not _HF_REPO.match(block["hf_repo"]):
            raise AlignManifestError(
                f"{where}: hf_repo {block['hf_repo']!r} is not an <owner>/<name> "
                "HuggingFace repo id"
            )
        if not _REVISION.match(block["revision"]):
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


# ------------------------------------------------------------------- loading


def parse_align_manifest(text: str, path: Path, expected_id: str) -> AlignManifest:
    """Parse and validate one align manifest's text. Raises by name."""
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise AlignManifestError(f"{path.name}: not valid TOML: {exc}") from exc
    return _parse(document, path, expected_id)


def load_align_manifest(model_id: str, directory: Path | None = None) -> AlignManifest:
    """Load `align/<model_id>.toml`. Raises AlignManifestError if it is not there."""
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
    """Every align manifest this build ships, by id, in id order.

    Ordered by `path.stem` and not by path, for the reason
    `crucible.manifests.load_all_manifests` gives: as whole paths a `-` sorts
    before a `.`, so the extension decides the order whenever one id is a prefix
    of another.
    """
    root = directory if directory is not None else align_manifests_dir()
    manifests: dict[str, AlignManifest] = {}
    for path in sorted(root.glob("*.toml"), key=lambda p: p.stem):
        manifests[path.stem] = load_align_manifest(path.stem, root)
    return manifests
