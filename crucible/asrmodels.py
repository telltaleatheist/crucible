"""ASR model manifests — `asr/<id>.toml` (PHASE4-AUDIO.md section 3).

One file per Crucible ASR model id, the same way `models/<id>.toml` works for the
`llm` types: the id is stable, the weights are per backend, and every revision is
a full commit sha so a pull is reproducible.

Validation is strict for the reason PHASE4-AUDIO.md section 3 gives for there
being no default model at all. An ASR pass at the wrong size is a transcript that
looks fine, is worse, and says nothing about it — so a manifest that misspells
`memory_bytes_estimat` must be a refusal rather than a model that quietly loads
with no estimate behind it.

Two backends, two ENGINES, and never two sets of weights at one id
------------------------------------------------------------------
faster-whisper is CTranslate2, and CTranslate2 has **no Metal backend** — on
Apple Silicon it runs on the CPU through Accelerate and nothing else
(SYSTRAN/faster-whisper#515, #911, still true as of 2026-09). Crucible has no CPU
backend, and PHASE4-AUDIO.md section 3 refuses the CPU road on its own terms: a
transcript that quietly ran at `int8` on a CPU is a different transcript.

So the Mac does not get an `[backends.mlx-darwin]` block on the faster-whisper
manifests. It gets `mlx-whisper`: a **second engine** with its own converted
weights (`mlx-community/whisper-*`), its own recipe
(`envs/asr/mlx-darwin.txt`), its own worker
(`crucible/jobs/asr/mlx_worker.py`) and — the part this module enforces — its
own SEVEN MODEL IDS, all prefixed `mlx-whisper-`.

**Different weights at one id would be a lie**, and it is this loader's job to
make that impossible rather than a convention. `transcript.json` records the
model id and nothing else about the bytes; `faster-whisper-large-v3` and
`mlx-whisper-large-v3` are different conversions at a different quantisation
and they will disagree about a hard passage, so an operator comparing two
transcripts has to be able to tell from the id which engine produced each.
A manifest that pairs a backend with the other backend's engine is refused by
name.

One thing DOES cross the two: `vad_filter`. faster-whisper has Silero VAD and
mlx-whisper has none at all, so `crucible/jobs/asr` refuses `vad_filter: true`
on this engine BY NAME rather than transcribing without it — the same argument
as the CPU one, one layer up.

Why this is not `crucible/manifests.py`
---------------------------------------
It should be. This loader and that one are the same TOML shape with a different
required set, a different directory and a different engine table, and they share
about two hundred lines of identical strictness. They are apart because phase 4
was built beside phases 2 and 3 in one tree, and `manifests.py` was another
builder's file while this was written. Merging them into one loader parameterised
by (directory, required keys, permitted engines) is a follow-up and a mechanical
one — not something to do while three builders are in the tree.
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

ASR_DIR_ENV = "CRUCIBLE_ASR_DIR"

#: Which engine each backend is allowed to name. TWO ENGINES, one per backend,
#: because CTranslate2 has no Metal backend — see the module docstring.
ASR_BACKEND_ENGINES: dict[str, str] = {
    CUDA_LINUX: "faster-whisper",
    MLX_DARWIN: "mlx-whisper",
}

#: The id prefix each engine's manifests must carry. Not decoration: it is what
#: stops one id ever standing for two different sets of weights, which is the
#: thing `transcript.json` cannot recover from. Checked by the loader, so a new
#: manifest cannot break the rule by being written carelessly.
ASR_ENGINE_ID_PREFIX: dict[str, str] = {
    "faster-whisper": "faster-whisper-",
    "mlx-whisper": "mlx-whisper-",
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
}

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_HF_REPO = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


class AsrManifestError(CrucibleError):
    """An ASR manifest is missing, unreadable, or does not say what it must say."""


@dataclass(frozen=True)
class AsrBackendSpec:
    """One `[backends.<kind>]` block.

    The field names are `crucible.manifests.BackendSpec`'s on purpose:
    `crucible/weights.py` reads `spec.hf_repo`, `spec.revision` and
    `spec.backend`, and `manifest.id` and `manifest.path.name`, and nothing else.
    Matching the names means ASR weights are pulled and stamped by the one
    weights module every other model goes through, with no branch in it for this
    job type.
    """

    backend: str
    engine: str
    hf_repo: str
    revision: str
    memory_bytes_estimate: int

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
        }


@dataclass(frozen=True)
class AsrManifest:
    #: Which tree under `~/.crucible/` these weights live in, and therefore which
    #: `pull` command a refusal tells the reader to run (`crucible/weights.py`).
    #: `models`, with the llm manifests: an ASR model is a model, `crucible models
    #: list` shows both directories, and giving it a third tree of its own would
    #: mean a third command to learn for no difference anyone can see. Not a
    #: dataclass field — a class attribute, the way `ModelManifest` declares it —
    #: because it is a property of the kind, not of the file.
    weights_family = "models"

    id: str
    family: str
    parameters_m: int
    backends: dict[str, AsrBackendSpec]
    path: Path

    def supports(self, backend_kind: str) -> bool:
        return backend_kind in self.backends

    def spec(self, backend_kind: str) -> AsrBackendSpec:
        found = self.backends.get(backend_kind)
        if found is None:
            raise AsrManifestError(
                f"ASR model {self.id!r} has no {backend_kind} block; "
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


def asr_manifests_dir() -> Path:
    """Where `asr/*.toml` live on this host. Refuses by name if absent."""
    override = os.environ.get(ASR_DIR_ENV)
    if override is not None and override != "":
        path = Path(override).expanduser()
        if not path.is_dir():
            raise AsrManifestError(f"{ASR_DIR_ENV}={override!r} is not a directory")
        return path
    path = Path(__file__).resolve().parent.parent / "asr"
    if not path.is_dir():
        raise AsrManifestError(
            f"no ASR manifests at {path}; crucible must run from a checkout "
            f"(pip install -e .) or ${ASR_DIR_ENV} must point at the manifests"
        )
    return path


# ------------------------------------------------------------------ checking


def _check_table(where: str, table: dict[str, Any], required: dict[str, type]) -> None:
    """Every required key present and correctly typed; no key that is not listed."""
    unknown = sorted(set(table) - set(required))
    if unknown:
        raise AsrManifestError(
            f"{where}: unknown key(s) {unknown}; this table takes exactly "
            f"{sorted(required)}"
        )
    missing = sorted(set(required) - set(table))
    if missing:
        raise AsrManifestError(f"{where}: missing required key(s) {missing}")
    for key, kind in required.items():
        value = table[key]
        wrong = not isinstance(value, kind)
        # bool is a subclass of int; a bool where an int is wanted is still wrong.
        if kind is int and isinstance(value, bool):
            wrong = True
        if wrong:
            raise AsrManifestError(
                f"{where}: {key} must be {kind.__name__}, got {type(value).__name__}"
            )


def _parse(document: dict[str, Any], path: Path, expected_id: str) -> AsrManifest:
    unknown = sorted(set(document) - {"model", "backends"})
    if unknown:
        raise AsrManifestError(
            f"{path.name}: unknown top-level table(s) {unknown}; an ASR manifest "
            "has exactly [model] and [backends.<kind>]"
        )
    if "model" not in document:
        raise AsrManifestError(f"{path.name}: missing the [model] table")
    if "backends" not in document:
        raise AsrManifestError(f"{path.name}: missing every [backends.<kind>] table")

    model = document["model"]
    if not isinstance(model, dict):
        raise AsrManifestError(f"{path.name}: [model] must be a table")
    _check_table(f"{path.name} [model]", model, _MODEL_REQUIRED)

    model_id = model["id"]
    if not _MODEL_ID.match(model_id):
        raise AsrManifestError(
            f"{path.name}: model.id {model_id!r} must be lower-case and start with "
            "a letter or digit ([a-z0-9][a-z0-9._-]*)"
        )
    if model_id != expected_id:
        raise AsrManifestError(
            f"{path.name}: model.id is {model_id!r} but the file is named "
            f"{expected_id!r}; the id and the filename are the same thing"
        )
    if model["parameters_m"] <= 0:
        raise AsrManifestError(
            f"{path.name}: model.parameters_m must be positive, got "
            f"{model['parameters_m']}"
        )

    backends_table = document["backends"]
    if not isinstance(backends_table, dict):
        raise AsrManifestError(
            f"{path.name}: [backends] must hold one table per backend"
        )
    if not backends_table:
        raise AsrManifestError(
            f"{path.name}: no backend blocks; a model nothing can serve is not a model"
        )

    backends: dict[str, AsrBackendSpec] = {}
    for kind, block in backends_table.items():
        where = f"{path.name} [backends.{kind}]"
        if kind not in ASR_BACKEND_ENGINES:
            raise AsrManifestError(
                f"{where}: {kind!r} is not an asr backend; the asr backends are "
                f"{sorted(ASR_BACKEND_ENGINES)}, each with its own engine "
                f"({ASR_BACKEND_ENGINES}). Windows is never a backend "
                "(docs/PHASE15-HOST.md)"
            )
        if not isinstance(block, dict):
            raise AsrManifestError(f"{where}: must be a table")
        _check_table(where, block, _BACKEND_REQUIRED)

        engine = block["engine"]
        if engine != ASR_BACKEND_ENGINES[kind]:
            raise AsrManifestError(
                f"{where}: engine {engine!r} does not run asr on {kind}; that "
                f"backend's asr engine is {ASR_BACKEND_ENGINES[kind]!r}. "
                "faster-whisper is CTranslate2, which has no Metal backend; "
                "mlx-whisper is MLX, which has no CUDA one. They are not two "
                "recipes for one thing"
            )
        prefix = ASR_ENGINE_ID_PREFIX[engine]
        if not model_id.startswith(prefix):
            # THE RULE THAT KEEPS A TRANSCRIPT HONEST. `transcript.json` names
            # the model id and nothing else about the bytes, and the two
            # engines' conversions of "large-v3" are different weights at a
            # different quantisation that will disagree about a hard passage.
            # An id that did not say which engine made it would leave a reader
            # comparing two transcripts with no way to tell them apart.
            raise AsrManifestError(
                f"{where}: engine {engine!r} requires an id beginning "
                f"{prefix!r} and this manifest is {model_id!r}; two engines' "
                "weights must never share an id, because a transcript records "
                "the id and nothing else about what produced it"
            )
        if not _HF_REPO.match(block["hf_repo"]):
            raise AsrManifestError(
                f"{where}: hf_repo {block['hf_repo']!r} is not an <owner>/<name> "
                "HuggingFace repo id"
            )
        if not _REVISION.match(block["revision"]):
            raise AsrManifestError(
                f"{where}: revision {block['revision']!r} must be a full 40-character "
                "commit sha, so a pull is reproducible; branch names are not pins"
            )
        if block["memory_bytes_estimate"] <= 0:
            raise AsrManifestError(
                f"{where}: memory_bytes_estimate must be positive, got "
                f"{block['memory_bytes_estimate']}"
            )
        backends[kind] = AsrBackendSpec(
            backend=kind,
            engine=engine,
            hf_repo=block["hf_repo"],
            revision=block["revision"],
            memory_bytes_estimate=block["memory_bytes_estimate"],
        )

    return AsrManifest(
        id=model_id,
        family=model["family"],
        parameters_m=model["parameters_m"],
        backends=backends,
        path=path,
    )


# ------------------------------------------------------------------- loading


def parse_asr_manifest(text: str, path: Path, expected_id: str) -> AsrManifest:
    """Parse and validate one ASR manifest's text. Raises AsrManifestError by name."""
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise AsrManifestError(f"{path.name}: not valid TOML: {exc}") from exc
    return _parse(document, path, expected_id)


def load_asr_manifest(model_id: str, directory: Path | None = None) -> AsrManifest:
    """Load `asr/<model_id>.toml`. Raises AsrManifestError if it is not there."""
    root = directory if directory is not None else asr_manifests_dir()
    path = root / f"{model_id}.toml"
    if not path.is_file():
        known = sorted(p.stem for p in root.glob("*.toml"))
        raise AsrManifestError(
            f"no ASR manifest for model {model_id!r} at {path}; this build ships "
            f"{known}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AsrManifestError(f"could not read {path}: {exc}") from exc
    return parse_asr_manifest(text, path, model_id)


def load_all_asr_manifests(directory: Path | None = None) -> dict[str, AsrManifest]:
    """Every ASR manifest this build ships, by id, in id order.

    Ordered by `path.stem` and not by path, for the reason
    `crucible.manifests.load_all_manifests` gives: as whole paths a `-` sorts
    before a `.`, so the extension decides the order whenever one id is a prefix
    of another. This function's order is the order `/v1/info` lists in.
    """
    root = directory if directory is not None else asr_manifests_dir()
    manifests: dict[str, AsrManifest] = {}
    for path in sorted(root.glob("*.toml"), key=lambda p: p.stem):
        manifests[path.stem] = load_asr_manifest(path.stem, root)
    return manifests
