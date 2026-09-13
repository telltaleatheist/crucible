"""Model manifests — `models/<id>.toml` (PHASE2-LLM.md section 1).

One file per Crucible model id. The id is stable across backends; the weights
differ per backend, so each manifest carries one `[backends.<kind>]` block per
backend it can be served on.

Validation is strict on purpose. An unknown key is a refusal, not a warning: a
manifest with `memory_bytes_estimat` in it must not quietly load with no estimate
and let the guard wave a 27B onto a 24 GB card.

Where the manifests live
------------------------
`models/` sits beside the `crucible` package in the checkout, exactly as the
contract writes it. `manifests_dir()` resolves it there, and honours
`$CRUCIBLE_MODELS_DIR` so a test can point at a fixture directory. If neither
exists the loader refuses by name; it never falls back to "no models".
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

MODELS_DIR_ENV = "CRUCIBLE_MODELS_DIR"

#: Which engine each backend is allowed to name. A manifest that pairs them any
#: other way is a manifest bug, not a runtime decision.
BACKEND_ENGINES: dict[str, str] = {
    CUDA_LINUX: "vllm",
    MLX_DARWIN: "mlx-lm",
}

_MODEL_REQUIRED: dict[str, type] = {
    "id": str,
    "family": str,
    "params_b": int,
    "context_default": int,
}
_BACKEND_REQUIRED: dict[str, type] = {
    "engine": str,
    "hf_repo": str,
    "revision": str,
    "memory_bytes_estimate": int,
}
_BACKEND_OPTIONAL: dict[str, type] = {
    "engine_args": list,
    # A context this backend can actually hold, when the model's own number is
    # not one it can. `[model] context_default` is what the model is FOR; this is
    # what a particular accelerator has room for, and the two are allowed to
    # disagree — `qwen3.8-27b-4bit` wants Owen's 98304 and gets it on 64 GB of
    # unified memory, while 98304 of its KV is 7.9 GiB the 3090 Ti does not have
    # once the weights are down. Absent means "the model's number"; it is never
    # a silent default.
    "context_default": int,
}

def fingerprint(model_id: str, revision: str) -> str:
    """`qwen3.5-9b@<sha>` — how a model's identity is written down.

    The bare id is not enough to identify bytes. Foundry hashes the served model
    id into its cleanup cache key and BookForge stamps it into a book's OPF
    (CLIENT-SURFACES.md section 6.5), so "what cleaned this book" has to name the
    pin as well as the model, or a manifest that moves to a new revision goes on
    answering from a cache built by the old one.
    """
    return f"{model_id}@{revision}"


_REVISION = re.compile(r"^[0-9a-f]{40}$")
_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_HF_REPO = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


class ManifestError(CrucibleError):
    """A manifest is missing, unreadable, or does not say what it must say."""


@dataclass(frozen=True)
class BackendSpec:
    """One `[backends.<kind>]` block."""

    backend: str
    engine: str
    hf_repo: str
    revision: str
    memory_bytes_estimate: int
    engine_args: tuple[str, ...]
    #: This backend's own context, or None to use the model's.
    context_default: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "engine": self.engine,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "engine_args": list(self.engine_args),
            "context_default": self.context_default,
        }


@dataclass(frozen=True)
class ModelManifest:
    id: str
    family: str
    params_b: int
    context_default: int
    backends: dict[str, BackendSpec]
    path: Path

    def supports(self, backend_kind: str) -> bool:
        return backend_kind in self.backends

    def context_for(self, backend_kind: str) -> int:
        """The context THIS backend serves: its own, or the model's.

        Everything that names a context — vLLM's `--max-model-len`, the resident
        model's row, `/v1/models` — must ask this and not read
        `self.context_default` directly, or a backend's override would be
        reported by one and ignored by the other.
        """
        found = self.backends.get(backend_kind)
        if found is None or found.context_default is None:
            return self.context_default
        return found.context_default

    def fingerprint_for(self, backend_kind: str) -> str | None:
        """`<id>@<revision>` for this backend, or None where there is no block.

        None rather than the bare id: a host with no block for this model has no
        revision to name here, and an unpinned fingerprint would be a worse
        record than no fingerprint — it would look like one.
        """
        found = self.backends.get(backend_kind)
        return None if found is None else fingerprint(self.id, found.revision)

    def spec(self, backend_kind: str) -> BackendSpec:
        """The block for `backend_kind`, or a named refusal."""
        found = self.backends.get(backend_kind)
        if found is None:
            raise ManifestError(
                f"model {self.id!r} has no {backend_kind} block; {self.path.name} "
                f"declares {sorted(self.backends)}"
            )
        return found

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "params_b": self.params_b,
            "context_default": self.context_default,
            "backends": {k: v.to_dict() for k, v in sorted(self.backends.items())},
        }


# ------------------------------------------------------------------ locating


def manifests_dir() -> Path:
    """Where `models/*.toml` live on this host. Refuses by name if absent."""
    override = os.environ.get(MODELS_DIR_ENV)
    if override is not None and override != "":
        path = Path(override).expanduser()
        if not path.is_dir():
            raise ManifestError(
                f"{MODELS_DIR_ENV}={override!r} is not a directory"
            )
        return path
    # models/ sits beside the crucible package in the checkout.
    path = Path(__file__).resolve().parent.parent / "models"
    if not path.is_dir():
        raise ManifestError(
            f"no model manifests at {path}; crucible must run from a checkout "
            f"(pip install -e .) or ${MODELS_DIR_ENV} must point at the manifests"
        )
    return path


# ------------------------------------------------------------------ checking


def _check_table(
    where: str,
    table: dict[str, Any],
    required: dict[str, type],
    optional: dict[str, type],
) -> None:
    """Every required key present and correctly typed; no key that is not listed."""
    allowed = set(required) | set(optional)
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ManifestError(
            f"{where}: unknown key(s) {unknown}; this table takes exactly "
            f"{sorted(allowed)}"
        )
    missing = sorted(set(required) - set(table))
    if missing:
        raise ManifestError(f"{where}: missing required key(s) {missing}")
    for key, kind in {**required, **optional}.items():
        if key not in table:
            continue
        value = table[key]
        wrong = not isinstance(value, kind)
        # bool is a subclass of int; a bool where an int is wanted is still wrong.
        if kind is int and isinstance(value, bool):
            wrong = True
        if wrong:
            raise ManifestError(
                f"{where}: {key} must be {kind.__name__}, got "
                f"{type(value).__name__}"
            )


def _parse(document: dict[str, Any], path: Path, expected_id: str) -> ModelManifest:
    unknown = sorted(set(document) - {"model", "backends"})
    if unknown:
        raise ManifestError(
            f"{path.name}: unknown top-level table(s) {unknown}; a manifest has "
            "exactly [model] and [backends.<kind>]"
        )
    if "model" not in document:
        raise ManifestError(f"{path.name}: missing the [model] table")
    if "backends" not in document:
        raise ManifestError(f"{path.name}: missing every [backends.<kind>] table")

    model = document["model"]
    if not isinstance(model, dict):
        raise ManifestError(f"{path.name}: [model] must be a table")
    _check_table(f"{path.name} [model]", model, _MODEL_REQUIRED, {})

    model_id = model["id"]
    if not _MODEL_ID.match(model_id):
        raise ManifestError(
            f"{path.name}: model.id {model_id!r} must be lower-case and start with "
            "a letter or digit ([a-z0-9][a-z0-9._-]*)"
        )
    if model_id != expected_id:
        raise ManifestError(
            f"{path.name}: model.id is {model_id!r} but the file is named "
            f"{expected_id!r}; the id and the filename are the same thing"
        )
    if model["params_b"] <= 0:
        raise ManifestError(
            f"{path.name}: model.params_b must be positive, got {model['params_b']}"
        )
    if model["context_default"] <= 0:
        raise ManifestError(
            f"{path.name}: model.context_default must be positive, got "
            f"{model['context_default']}"
        )

    backends_table = document["backends"]
    if not isinstance(backends_table, dict):
        raise ManifestError(f"{path.name}: [backends] must hold one table per backend")
    if not backends_table:
        raise ManifestError(
            f"{path.name}: no backend blocks; a model nothing can serve is not a model"
        )

    backends: dict[str, BackendSpec] = {}
    for kind, block in backends_table.items():
        where = f"{path.name} [backends.{kind}]"
        if kind not in BACKEND_ENGINES:
            raise ManifestError(
                f"{where}: {kind!r} is not a Crucible backend; the backends are "
                f"{sorted(BACKEND_ENGINES)}"
            )
        if not isinstance(block, dict):
            raise ManifestError(f"{where}: must be a table")
        _check_table(where, block, _BACKEND_REQUIRED, _BACKEND_OPTIONAL)

        engine = block["engine"]
        if engine != BACKEND_ENGINES[kind]:
            raise ManifestError(
                f"{where}: engine {engine!r} does not run on {kind}; that backend's "
                f"engine is {BACKEND_ENGINES[kind]!r}"
            )
        if not _HF_REPO.match(block["hf_repo"]):
            raise ManifestError(
                f"{where}: hf_repo {block['hf_repo']!r} is not an <owner>/<name> "
                "HuggingFace repo id"
            )
        if not _REVISION.match(block["revision"]):
            raise ManifestError(
                f"{where}: revision {block['revision']!r} must be a full 40-character "
                "commit sha, so a pull is reproducible; branch names are not pins"
            )
        if block["memory_bytes_estimate"] <= 0:
            raise ManifestError(
                f"{where}: memory_bytes_estimate must be positive, got "
                f"{block['memory_bytes_estimate']}"
            )
        engine_args = block.get("engine_args", [])
        for index, argument in enumerate(engine_args):
            if not isinstance(argument, str):
                raise ManifestError(
                    f"{where}: engine_args[{index}] must be a string, got "
                    f"{type(argument).__name__}"
                )
        backend_context = block.get("context_default")
        if backend_context is not None and backend_context <= 0:
            raise ManifestError(
                f"{where}: context_default must be positive, got {backend_context}"
            )
        backends[kind] = BackendSpec(
            backend=kind,
            engine=engine,
            hf_repo=block["hf_repo"],
            revision=block["revision"],
            memory_bytes_estimate=block["memory_bytes_estimate"],
            engine_args=tuple(engine_args),
            context_default=backend_context,
        )

    return ModelManifest(
        id=model_id,
        family=model["family"],
        params_b=model["params_b"],
        context_default=model["context_default"],
        backends=backends,
        path=path,
    )


# ------------------------------------------------------------------- loading


def parse_manifest(text: str, path: Path, expected_id: str) -> ModelManifest:
    """Parse and validate one manifest's text. Raises ManifestError by name."""
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(f"{path.name}: not valid TOML: {exc}") from exc
    return _parse(document, path, expected_id)


def load_manifest(model_id: str, directory: Path | None = None) -> ModelManifest:
    """Load `models/<model_id>.toml`. Raises ManifestError if it is not there."""
    root = directory if directory is not None else manifests_dir()
    path = root / f"{model_id}.toml"
    if not path.is_file():
        known = sorted(p.stem for p in root.glob("*.toml"))
        raise ManifestError(
            f"no manifest for model {model_id!r} at {path}; this build ships {known}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"could not read {path}: {exc}") from exc
    return parse_manifest(text, path, model_id)


def load_all_manifests(directory: Path | None = None) -> dict[str, ModelManifest]:
    """Every manifest this build ships, by id, in id order."""
    root = directory if directory is not None else manifests_dir()
    manifests: dict[str, ModelManifest] = {}
    # By id — `path.stem` — and not by path. The two orders differ whenever one
    # id is a prefix of another, because the extension gets in the way: as whole
    # paths, `qwen3.8-27b-4bit.toml` sorts BEFORE `qwen3.8-27b.toml` ('-' is
    # 0x2D, '.' is 0x2E), while as ids `qwen3.8-27b` comes first. This function's
    # order is what `/v1/models` lists in, so it is the documented one.
    for path in sorted(root.glob("*.toml"), key=lambda p: p.stem):
        manifests[path.stem] = load_manifest(path.stem, root)
    return manifests
