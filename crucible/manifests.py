from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .backend import CUDA_LINUX, LLAMA_WINDOWS, MLX_DARWIN
from .errors import CrucibleError
from .precision import MIN_WEIGHT_BITS, below_floor, gguf_bits, implied_bits
from .tomltable import HF_REPO_PATTERN, MODEL_ID_PATTERN, REVISION_PATTERN, check_table

MODELS_DIR_ENV = "CRUCIBLE_MODELS_DIR"

MODELS_PULL_COMMAND = "crucible models pull"

TEXT_FAMILY = "text"
PAGES_FAMILY = "pages"

BACKEND_ENGINES: dict[str, dict[str, str]] = {
    CUDA_LINUX: {TEXT_FAMILY: "vllm", PAGES_FAMILY: "vllm"},
    MLX_DARWIN: {TEXT_FAMILY: "mlx-lm", PAGES_FAMILY: "mlx-vlm"},
    LLAMA_WINDOWS: {TEXT_FAMILY: "llama-server", PAGES_FAMILY: "llama-server"},
}


def class_family(modalities: "tuple[str, ...] | list[str]") -> str:
    return PAGES_FAMILY if "image" in modalities else TEXT_FAMILY


def engine_for(backend_kind: str, modalities: "tuple[str, ...] | list[str]") -> str:
    engines = BACKEND_ENGINES.get(backend_kind)
    if engines is None:
        raise ManifestError(
            f"{backend_kind!r} is not a Crucible backend; the backends are "
            f"{sorted(BACKEND_ENGINES)}"
        )
    family = class_family(modalities)
    found = engines.get(family)
    if found is None:
        raise ManifestError(
            f"{backend_kind!r} serves no {family!r} engine; it serves "
            f"{sorted(engines)}"
        )
    return found


MODALITIES: frozenset[str] = frozenset({"text", "image"})

SKIP_MM_PROFILING = "--skip-mm-profiling"

LANGUAGE_MODEL_ONLY = "--language-model-only"

DEFAULTS_KEYS: dict[str, type] = {
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "max_tokens": int,
    "repetition_penalty": float,
    "thinking": bool,
}

DEFAULTS_WIRE_KEYS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "repetition_penalty",
)

NUMBER: tuple[type, ...] = (int, float)

_MODEL_REQUIRED: dict[str, Any] = {
    "id": str,
    "family": str,
    "params_b": NUMBER,
    "context_default": int,
    "trained_context": int,
    "modalities": list,
}
_MODEL_OPTIONAL: dict[str, type] = {
    "display": str,
    "description": str,
    "weights_of": str,
}
_BACKEND_REQUIRED: dict[str, type] = {
    "engine": str,
    "hf_repo": str,
    "revision": str,
    "memory_bytes_estimate": int,
}
_BACKEND_OPTIONAL: dict[str, type] = {
    "bits": int,
    "engine_args": list,
    "file": str,
    "mmproj": str,
    "context_default": int,
    "max_context": int,
    "memory": dict,
    "serves": list,
}

_MEMORY_REQUIRED: dict[str, type] = {
    "weights_bytes": int,
    "overhead_bytes": int,
    "kv_bytes_per_token": int,
    "basis": str,
    "measured_at_context": int,
}

MEMORY_BASES: frozenset[str] = frozenset({"measured", "computed", "declared"})

MEMORY_TERMS_TOLERANCE = 0.05

LOCAL_KINDS: frozenset[str] = frozenset({"ollama", "gguf"})

NEEDS_BASES: frozenset[str] = frozenset({"measured", "declared"})

_LOCAL_COMMON_REQUIRED: dict[str, type] = {
    "kind": str,
    "download_bytes": int,
    "needs_bytes": int,
    "needs_basis": str,
}
_LOCAL_KIND_REQUIRED: dict[str, dict[str, type]] = {
    "ollama": {
        "tag": str,
    },
    "gguf": {
        "hf_repo": str,
        "revision": str,
        "file": str,
    },
}
_LOCAL_KIND_OPTIONAL: dict[str, dict[str, type]] = {
    "ollama": {},
    "gguf": {
        "mmproj": str,
    },
}


def fingerprint(model_id: str, revision: str) -> str:
    return f"{model_id}@{revision}"


class ManifestError(CrucibleError):
    ...


@dataclass(frozen=True)
class MemoryTerms:

    weights_bytes: int
    overhead_bytes: int
    kv_bytes_per_token: int
    basis: str
    measured_at_context: int

    @property
    def fixed_bytes(self) -> int:
        return self.weights_bytes + self.overhead_bytes

    def bytes_for(self, *, context: int, concurrency: int) -> int:
        if context <= 0:
            raise ValueError(f"context must be positive, got {context}")
        if concurrency <= 0:
            raise ValueError(f"concurrency must be positive, got {concurrency}")
        return self.fixed_bytes + self.kv_bytes_per_token * context * concurrency

    def max_context(self, *, available_bytes: int, concurrency: int) -> int:
        if concurrency <= 0:
            raise ValueError(f"concurrency must be positive, got {concurrency}")
        room = available_bytes - self.fixed_bytes
        if room <= 0:
            return 0
        return room // (self.kv_bytes_per_token * concurrency)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights_bytes": self.weights_bytes,
            "overhead_bytes": self.overhead_bytes,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "basis": self.basis,
            "measured_at_context": self.measured_at_context,
        }


@dataclass(frozen=True)
class BackendSpec:

    backend: str
    engine: str
    hf_repo: str
    revision: str
    memory_bytes_estimate: int
    engine_args: tuple[str, ...]
    context_default: int | None
    max_context: int | None = None
    memory: "MemoryTerms | None" = None
    file: str | None = None
    mmproj: str | None = None
    serves: tuple[str, ...] = ()
    bits: int | None = None

    @property
    def files(self) -> tuple[str, ...]:
        return tuple(name for name in (self.file, self.mmproj) if name is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "engine": self.engine,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "engine_args": list(self.engine_args),
            "context_default": self.context_default,
            "max_context": self.max_context,
            "memory": None if self.memory is None else self.memory.to_dict(),
            "file": self.file,
            "mmproj": self.mmproj,
            "serves": list(self.serves),
            "bits": self.bits,
        }


@dataclass(frozen=True)
class ModelDefaults:

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None
    thinking: bool | None = None

    def stated(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for key in DEFAULTS_KEYS:
            value = getattr(self, key)
            if value is not None:
                values[key] = value
        return values

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in DEFAULTS_KEYS}


NO_DEFAULTS = ModelDefaults()


@dataclass(frozen=True)
class LocalForm:

    kind: str
    download_bytes: int
    needs_bytes: int
    needs_basis: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "download_bytes": self.download_bytes,
            "needs_bytes": self.needs_bytes,
            "needs_basis": self.needs_basis,
        }


@dataclass(frozen=True)
class OllamaLocal(LocalForm):
    tag: str

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "tag": self.tag}


@dataclass(frozen=True)
class GgufLocal(LocalForm):
    hf_repo: str
    revision: str
    file: str
    mmproj: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "file": self.file,
            "mmproj": self.mmproj,
        }


@dataclass(frozen=True)
class ModelManifest:
    id: str
    family: str
    params_b: int | float
    context_default: int
    trained_context: int
    modalities: tuple[str, ...]
    backends: dict[str, BackendSpec]
    path: Path
    defaults: ModelDefaults = NO_DEFAULTS
    display: str | None = None
    description: str | None = None
    local: LocalForm | None = None
    weights_of: str | None = None
    weights_base: "ModelManifest | None" = field(
        default=None, compare=False, repr=False
    )

    weights_family = "models"

    @property
    def pull_command(self) -> str:
        return f"{MODELS_PULL_COMMAND} {self.id}"

    def aliases(self) -> "tuple[ModelManifest, ...]":
        return aliases_of(self)

    @property
    def store_id(self) -> str:
        return self.id if self.weights_of is None else self.weights_of

    def extra_files(self, backend_kind: str) -> tuple[str, ...]:
        spec = self.spec(backend_kind)
        if self.weights_base is None:
            return ()
        shared = self.weights_base.spec(backend_kind).files
        return tuple(name for name in spec.files if name not in shared)

    def supports(self, backend_kind: str) -> bool:
        return backend_kind in self.backends

    def context_for(self, backend_kind: str) -> int:
        found = self.backends.get(backend_kind)
        if found is None or found.context_default is None:
            return self.context_default
        return found.context_default

    def max_context_for(self, backend_kind: str) -> int:
        found = self.backends.get(backend_kind)
        if found is None or found.max_context is None:
            return self.context_for(backend_kind)
        return found.max_context

    def fingerprint_for(self, backend_kind: str) -> str | None:
        found = self.backends.get(backend_kind)
        return None if found is None else fingerprint(self.id, found.revision)

    def serves(self, backend_kind: str) -> tuple[str, ...]:
        return self.spec(backend_kind).serves

    def spec(self, backend_kind: str) -> BackendSpec:
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
            "trained_context": self.trained_context,
            "modalities": list(self.modalities),
            "display": self.display,
            "description": self.description,
            "defaults": self.defaults.to_dict(),
            "weights_of": self.weights_of,
            "backends": {k: v.to_dict() for k, v in sorted(self.backends.items())},
        }


def manifests_dir() -> Path:
    override = os.environ.get(MODELS_DIR_ENV)
    if override is not None and override != "":
        path = Path(override).expanduser()
        if not path.is_dir():
            raise ManifestError(
                f"{MODELS_DIR_ENV}={override!r} is not a directory"
            )
        return path
    path = Path(__file__).resolve().parent / "models"
    if not path.is_dir():
        raise ManifestError(
            f"no model manifests at {path}; they are package data and this "
            f"install has lost them, or ${MODELS_DIR_ENV} must point at them"
        )
    return path


_DEFAULT_BOUNDS: dict[str, tuple[Any, str]] = {
    "temperature": (lambda v: v >= 0.0, "must be zero or more (0 is greedy)"),
    "top_p": (lambda v: 0.0 < v <= 1.0, "must be above 0 and at most 1"),
    "top_k": (
        lambda v: v >= 1,
        "must be at least 1; to leave top-k alone, omit the key rather than "
        "writing a number that means 'off' on one engine and nothing on the other",
    ),
    "max_tokens": (lambda v: v >= 1, "must be at least 1"),
    "repetition_penalty": (lambda v: v > 0.0, "must be above 0 (1.0 is no penalty)"),
    "thinking": (lambda v: True, ""),
}


def _default_value(where: str, key: str, kind: type, value: Any) -> Any:
    if kind is bool:
        if not isinstance(value, bool):
            raise ManifestError(f"{where}: {key} must be bool, got {type(value).__name__}")
        return value
    if kind is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ManifestError(f"{where}: {key} must be int, got {type(value).__name__}")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(
            f"{where}: {key} must be a number, got {type(value).__name__}"
        )
    return float(value)


def _parse_defaults(table: Any, path: Path) -> ModelDefaults:
    where = f"{path.name} [defaults]"
    if not isinstance(table, dict):
        raise ManifestError(f"{where}: must be a table")
    unknown = sorted(set(table) - set(DEFAULTS_KEYS))
    if unknown:
        raise ManifestError(
            f"{where}: unknown key(s) {unknown}; this table takes exactly "
            f"{sorted(DEFAULTS_KEYS)} — the only knobs both vLLM and mlx-lm "
            "honour. A key no engine reads would be a number in a file that "
            "looks like it is doing something"
        )
    if not table:
        raise ManifestError(
            f"{where}: the table is empty. A model that states no defaults says "
            "so by having no [defaults] table at all; an empty one reads as a "
            "decision somebody made and then forgot to write down"
        )
    values: dict[str, Any] = {}
    for key, kind in DEFAULTS_KEYS.items():
        if key not in table:
            continue
        value = _default_value(where, key, kind, table[key])
        test, why = _DEFAULT_BOUNDS[key]
        if not test(value):
            raise ManifestError(f"{where}: {key} is {value!r} and {why}")
        values[key] = value
    return ModelDefaults(**values)


_GGUF_FILE = ".gguf"


def _gguf_name(where: str, key: str, value: str) -> str:
    if value == "" or value.strip() != value:
        raise ManifestError(f"{where}: {key} must be a file name, got {value!r}")
    if not value.endswith(_GGUF_FILE):
        raise ManifestError(
            f"{where}: {key} {value!r} does not end in {_GGUF_FILE!r}; llama-server "
            "reads nothing else, and a name without the extension is usually a "
            "repo id or a directory rather than the file"
        )
    bits = gguf_bits(value)
    if below_floor(bits):
        raise ManifestError(
            f"{where}: {key} {value!r} is a {bits}-bit quantization, and nothing "
            f"under {MIN_WEIGHT_BITS} bits is ever offered (Owen, 2026-09-26: "
            "\"no less than 4\")"
        )
    return value


def _check_hf_pin(where: str, table: dict[str, Any]) -> None:
    if not HF_REPO_PATTERN.match(table["hf_repo"]):
        raise ManifestError(
            f"{where}: hf_repo {table['hf_repo']!r} is not an <owner>/<name> "
            "HuggingFace repo id"
        )
    if not REVISION_PATTERN.match(table["revision"]):
        raise ManifestError(
            f"{where}: revision {table['revision']!r} must be a full 40-character "
            "commit sha, so a pull is reproducible; branch names are not pins"
        )


def _local_kind(where: str, table: Any) -> str:
    if not isinstance(table, dict):
        raise ManifestError(f"{where}: must be a table")
    if "kind" not in table:
        raise ManifestError(
            f"{where}: missing required key(s) ['kind']; the kinds are "
            f"{sorted(LOCAL_KINDS)}"
        )
    kind = table["kind"]
    if not isinstance(kind, str) or kind not in LOCAL_KINDS:
        raise ManifestError(
            f"{where}: kind {kind!r} is not a local form Crucible knows; the "
            f"kinds are {sorted(LOCAL_KINDS)}"
        )
    return kind


def _check_local_sizes(where: str, table: dict[str, Any]) -> None:
    download = table["download_bytes"]
    needs = table["needs_bytes"]
    if download <= 0:
        raise ManifestError(f"{where}: download_bytes must be positive, got {download}")
    if needs <= 0:
        raise ManifestError(f"{where}: needs_bytes must be positive, got {needs}")
    if needs < download:
        raise ManifestError(
            f"{where}: needs_bytes ({needs}) is less than download_bytes "
            f"({download}); a model cannot run in less memory than its weights "
            "occupy, so one of the two numbers is wrong"
        )
    basis = table["needs_basis"]
    if basis not in NEEDS_BASES:
        raise ManifestError(
            f"{where}: needs_basis {basis!r} must be one of {sorted(NEEDS_BASES)}"
        )


def _ollama_local(
    where: str, table: dict[str, Any], common: dict[str, Any], modalities: tuple[str, ...]
) -> OllamaLocal:
    tag = table["tag"]
    name, colon, version = tag.partition(":")
    if colon == "" or name == "" or version == "" or tag.split() != [tag]:
        raise ManifestError(
            f"{where}: tag {tag!r} must be <name>:<tag>; a bare name is "
            "`:latest`, which is a floating pointer and not a pin"
        )
    return OllamaLocal(**common, tag=tag)


def _local_mmproj(
    where: str, mmproj: str | None, file: str, modalities: tuple[str, ...]
) -> str | None:
    reads_images = "image" in modalities
    if reads_images and mmproj is None:
        raise ManifestError(
            f"{where}: [model] modalities declares 'image' and this block has no "
            "mmproj. llama-server serves a vision model as a text tower plus a "
            "projector; without the projector it loads, answers /v1/models, and "
            "refuses every page. Name the mmproj file"
        )
    if mmproj is not None:
        mmproj = _gguf_name(where, "mmproj", mmproj)
    if mmproj is not None and not reads_images:
        raise ManifestError(
            f"{where}: mmproj {mmproj!r} names a vision projector, but [model] "
            f"modalities is {list(modalities)}. A projector nothing here sends a "
            "page to is a file nobody would load; either offer 'image' or take "
            "it out"
        )
    if mmproj == file:
        raise ManifestError(
            f"{where}: mmproj and file are the same name {file!r}; the projector "
            "is a second file"
        )
    return mmproj


def _gguf_local(
    where: str, table: dict[str, Any], common: dict[str, Any], modalities: tuple[str, ...]
) -> GgufLocal:
    _check_hf_pin(where, table)
    file = _gguf_name(where, "file", table["file"])
    mmproj = _local_mmproj(where, table.get("mmproj"), file, modalities)
    return GgufLocal(
        **common,
        hf_repo=table["hf_repo"],
        revision=table["revision"],
        file=file,
        mmproj=mmproj,
    )


_LOCAL_BUILDERS = {"ollama": _ollama_local, "gguf": _gguf_local}


def _parse_local(
    table: Any, path: Path, modalities: tuple[str, ...]
) -> LocalForm:
    where = f"{path.name} [local]"
    kind = _local_kind(where, table)
    check_table(
        where,
        table,
        {**_LOCAL_COMMON_REQUIRED, **_LOCAL_KIND_REQUIRED[kind]},
        _LOCAL_KIND_OPTIONAL[kind],
        error=ManifestError,
    )
    _check_local_sizes(where, table)
    common = {
        "kind": kind,
        "download_bytes": table["download_bytes"],
        "needs_bytes": table["needs_bytes"],
        "needs_basis": table["needs_basis"],
    }
    return _LOCAL_BUILDERS[kind](where, table, common, modalities)


def _parse_serves(
    where: str, block: dict[str, Any], modalities: "list[str] | tuple[str, ...]"
) -> tuple[str, ...]:
    if "serves" not in block:
        return tuple(modalities)
    served = block["serves"]
    if not served:
        raise ManifestError(
            f"{where}: serves is empty; a backend that serves nothing is not a "
            "backend. Omit the key to serve everything [model] modalities "
            f"declares ({list(modalities)})"
        )
    for index, entry in enumerate(served):
        if not isinstance(entry, str):
            raise ManifestError(
                f"{where}: serves[{index}] must be a string, got "
                f"{type(entry).__name__}"
            )
        if entry not in MODALITIES:
            raise ManifestError(
                f"{where}: serves[{index}] is {entry!r}; Crucible knows "
                f"{sorted(MODALITIES)}"
            )
    if len(set(served)) != len(served):
        raise ManifestError(f"{where}: serves lists a modality twice: {served}")
    beyond = [entry for entry in served if entry not in modalities]
    if beyond:
        raise ManifestError(
            f"{where}: serves_not_subset — serves {served} names {beyond}, which "
            f"[model] modalities {list(modalities)} does not. What the weights "
            "accept is the model's; a backend may serve less of it, never more"
        )
    return tuple(served)


_TOP_LEVEL_TABLES = frozenset({"model", "backends", "defaults", "local"})


def _check_document(document: dict[str, Any], path: Path) -> None:
    unknown = sorted(set(document) - _TOP_LEVEL_TABLES)
    if unknown:
        raise ManifestError(
            f"{path.name}: unknown top-level table(s) {unknown}; a manifest has "
            "exactly [model], [backends.<kind>], an optional [defaults] and an "
            "optional [local]"
        )
    if "model" not in document:
        raise ManifestError(f"{path.name}: missing the [model] table")
    if "backends" not in document:
        raise ManifestError(f"{path.name}: missing every [backends.<kind>] table")


def _check_model_id(path: Path, model_id: str, expected_id: str) -> None:
    if "/" in model_id:
        raise ManifestError(
            f"{path.name}: manifest_model_id_slash — model.id {model_id!r} "
            "contains '/', which is reserved: a model id with a slash is an "
            "UPSTREAM model id (`<upstream>/<model>`), and the chat door tells "
            "the two apart by that one character"
        )
    if not MODEL_ID_PATTERN.match(model_id):
        raise ManifestError(
            f"{path.name}: model.id {model_id!r} must be lower-case and start with "
            "a letter or digit ([a-z0-9][a-z0-9._-]*)"
        )
    if model_id != expected_id:
        raise ManifestError(
            f"{path.name}: model.id is {model_id!r} but the file is named "
            f"{expected_id!r}; the id and the filename are the same thing"
        )


def _check_weights_of(path: Path, model: dict[str, Any], has_local: bool) -> None:
    weights_of = model.get("weights_of")
    if weights_of is None:
        return
    if not MODEL_ID_PATTERN.match(weights_of):
        raise ManifestError(
            f"{path.name}: weights_of_unknown — model.weights_of "
            f"{weights_of!r} is not a model id ([a-z0-9][a-z0-9._-]*)"
        )
    if weights_of == model["id"]:
        raise ManifestError(
            f"{path.name}: weights_of_chain — model.weights_of names this "
            "model itself. A model that shares its own weights shares "
            "nothing; omit the key"
        )
    if has_local:
        raise ManifestError(
            f"{path.name}: weights_of_local — this model shares the weights "
            f"of {weights_of!r} and carries a [local] table. The local form "
            f"belongs to the model that owns the download; take [local] out "
            f"of this file"
        )


def _check_model_sizes(path: Path, model: dict[str, Any]) -> None:
    if model["params_b"] <= 0:
        raise ManifestError(
            f"{path.name}: model.params_b must be positive, got {model['params_b']}"
        )
    if model["trained_context"] <= 0:
        raise ManifestError(
            f"{path.name}: model.trained_context must be positive, got "
            f"{model['trained_context']}"
        )
    if model["context_default"] > model["trained_context"]:
        raise ManifestError(
            f"{path.name}: model.context_default is {model['context_default']} "
            f"and the weights are trained at {model['trained_context']}. A host "
            f"may serve less than the checkpoint supports; it cannot serve more"
        )
    if model["context_default"] <= 0:
        raise ManifestError(
            f"{path.name}: model.context_default must be positive, got "
            f"{model['context_default']}"
        )


def _check_model_labels(path: Path, model: dict[str, Any], has_local: bool) -> None:
    for key in ("display", "description"):
        if key in model and model[key].strip() == "":
            raise ManifestError(
                f"{path.name}: model.{key} is empty; a display fact nobody wrote "
                "is said by omitting the key, not by an empty string a screen "
                "would print as nothing"
            )
    unnamed = sorted(key for key in ("display", "description") if key not in model)
    if has_local and unnamed:
        raise ManifestError(
            f"{path.name}: [local] is present but [model] is missing {unnamed}; "
            "the lineup that table feeds is drawn as a tile, and a tile "
            "needs its label and its sentence from the same file as its "
            "numbers"
        )


def _check_modalities(path: Path, modalities: list[Any]) -> None:
    if not modalities:
        raise ManifestError(
            f"{path.name}: model.modalities is empty; a model that accepts no "
            f"input at all is not a model. It takes one or more of "
            f"{sorted(MODALITIES)}"
        )
    for index, entry in enumerate(modalities):
        if not isinstance(entry, str):
            raise ManifestError(
                f"{path.name}: model.modalities[{index}] must be a string, got "
                f"{type(entry).__name__}"
            )
        if entry not in MODALITIES:
            raise ManifestError(
                f"{path.name}: model.modalities[{index}] is {entry!r}; Crucible "
                f"knows {sorted(MODALITIES)}"
            )
    if len(set(modalities)) != len(modalities):
        raise ManifestError(
            f"{path.name}: model.modalities lists a modality twice: {modalities}"
        )


def _parse_model(document: dict[str, Any], path: Path, expected_id: str) -> dict[str, Any]:
    model = document["model"]
    if not isinstance(model, dict):
        raise ManifestError(f"{path.name}: [model] must be a table")
    check_table(
        f"{path.name} [model]", model, _MODEL_REQUIRED, _MODEL_OPTIONAL,
        error=ManifestError,
    )
    has_local = "local" in document
    _check_model_id(path, model["id"], expected_id)
    _check_weights_of(path, model, has_local)
    _check_model_sizes(path, model)
    _check_model_labels(path, model, has_local)
    _check_modalities(path, model["modalities"])
    return model


def _check_engine(
    where: str, kind: str, engine: str, serves_here: tuple[str, ...], modalities: list[str]
) -> None:
    expected = engine_for(kind, serves_here)
    if engine == expected:
        return
    family = class_family(serves_here)
    raise ManifestError(
        f"{where}: engine {engine!r} does not serve {family!r} models on "
        f"{kind}; that pairing's engine is {expected!r}. The family is "
        f"read off [model] modalities = {list(modalities)} as this block "
        f"serves them ({list(serves_here)}; `serves` narrows, never "
        "widens) and not out of the engine name, so an engine cannot be "
        "chosen by naming it"
    )


def _backend_engine_args(where: str, block: dict[str, Any]) -> tuple[str, ...]:
    engine_args = block.get("engine_args", [])
    for index, argument in enumerate(engine_args):
        if not isinstance(argument, str):
            raise ManifestError(
                f"{where}: engine_args[{index}] must be a string, got "
                f"{type(argument).__name__}"
            )
    return tuple(engine_args)


def _check_llama_files(where: str, block: dict[str, Any], serves_here: tuple[str, ...]) -> None:
    if "file" not in block:
        raise ManifestError(
            f"{where}: llama-windows needs `file`, the one GGUF in "
            f"{block['hf_repo']!r} this row is. A GGUF repo holds every "
            "quantization of a model and this server pulls one"
        )
    if "image" in serves_here and "mmproj" not in block:
        raise ManifestError(
            f"{where}: [model] modalities declares 'image' and this block "
            f"serves it ({list(serves_here)}), and it names no `mmproj`. Half a vision model is a model "
            "that loads and then cannot see; the projector is not "
            "optional (docs/internals/config-envs-weights.md, \"Model manifests\")"
        )
    if "image" not in serves_here and "mmproj" in block:
        raise ManifestError(
            f"{where}: mmproj {block['mmproj']!r} names a vision "
            f"projector, and this block serves {list(serves_here)}. Either "
            "serve 'image' here or take the projector out"
        )
    for key in ("file", "mmproj"):
        name = block.get(key)
        if name is not None and (name != Path(name).name or name.startswith(".")):
            raise ManifestError(
                f"{where}: {key} {name!r} must be a plain file name "
                "inside the repo, not a path"
            )


def _check_backend_files(
    where: str, kind: str, block: dict[str, Any], serves_here: tuple[str, ...]
) -> None:
    if kind == LLAMA_WINDOWS:
        _check_llama_files(where, block, serves_here)
        return
    extra = sorted({"file", "mmproj"} & set(block))
    if extra:
        raise ManifestError(
            f"{where}: {extra} belong to a llama-windows block. On "
            f"{kind} the whole repo is the weights and there is no "
            "file to choose"
        )


def _check_vision_flags(
    where: str, serves_here: tuple[str, ...], engine_args: tuple[str, ...]
) -> None:
    if "image" in serves_here and SKIP_MM_PROFILING in engine_args:
        raise ManifestError(
            f"{where}: engine_args carries {SKIP_MM_PROFILING!r} while "
            f"[model] modalities declares 'image' and this block serves it. "
            f"That flag stops vLLM "
            f"reserving for an image, so it belongs only to a model this "
            f"server serves text-only; take it out and measure "
            f"--gpu-memory-utilization again with the image profiled in "
            f"(docs/internals/config-envs-weights.md, \"Model manifests\")"
        )
    if "image" in serves_here and LANGUAGE_MODEL_ONLY in engine_args:
        raise ManifestError(
            f"{where}: engine_args carries {LANGUAGE_MODEL_ONLY!r} while "
            f"[model] modalities declares 'image' and this block serves it. "
            f"That flag makes vLLM skip "
            f"loading the vision tower, so this engine would answer every "
            f"page from the text alone — a well-formed reading of something "
            f"it was never shown. It belongs only to a model this server "
            f"serves text-only"
        )


def _backend_context(where: str, block: dict[str, Any], trained: int) -> int | None:
    backend_context = block.get("context_default")
    if backend_context is not None and backend_context <= 0:
        raise ManifestError(
            f"{where}: context_default must be positive, got {backend_context}"
        )
    if backend_context is not None and backend_context > trained:
        raise ManifestError(
            f"{where}: context_default is {backend_context} and the weights "
            f"are trained at {trained}. An accelerator with "
            f"room to spare does not give a checkpoint a longer memory"
        )
    return backend_context


def _backend_max_context(
    where: str, block: dict[str, Any], trained: int, served_here: int
) -> int | None:
    block_max = block.get("max_context")
    if block_max is None:
        return None
    if block_max <= 0:
        raise ManifestError(f"{where}: max_context must be positive, got {block_max}")
    if block_max > trained:
        raise ManifestError(
            f"{where}: max_context is {block_max} and the weights are "
            f"trained at {trained}. No load may start "
            "an engine past what the checkpoint's positions reach"
        )
    if block_max < served_here:
        raise ManifestError(
            f"{where}: max_context is {block_max} and this block serves "
            f"{served_here} by default. The largest context a load may "
            "ask for cannot be smaller than the one a load that asks for "
            "nothing gets; lower context_default or raise max_context"
        )
    return block_max


def _check_memory_values(where: str, table: dict[str, Any]) -> None:
    for key in ("weights_bytes", "kv_bytes_per_token"):
        if table[key] <= 0:
            raise ManifestError(f"{where}: {key} must be positive, got {table[key]}")
    if table["overhead_bytes"] < 0:
        raise ManifestError(
            f"{where}: overhead_bytes cannot be negative, got "
            f"{table['overhead_bytes']}"
        )
    if table["basis"] not in MEMORY_BASES:
        raise ManifestError(
            f"{where}: basis {table['basis']!r} is not one "
            f"of {sorted(MEMORY_BASES)}"
        )
    if table["measured_at_context"] <= 0:
        raise ManifestError(
            f"{where}: measured_at_context must be positive, got "
            f"{table['measured_at_context']}"
        )


def _check_memory_agrees(where: str, terms: MemoryTerms, served: int, stated: int) -> None:
    if terms.measured_at_context != served:
        raise ManifestError(
            f"{where}: measured_at_context is "
            f"{terms.measured_at_context} and this block serves "
            f"{served}. An estimate is only true at the context it was "
            f"taken at; state the terms at the context this block runs, "
            f"or move the block's context_default to match what was "
            f"measured (docs/FITS-AND-THE-CARD.md section 0b)"
        )
    from_terms = terms.bytes_for(context=served, concurrency=1)
    drift = abs(from_terms - stated) / stated
    if drift > MEMORY_TERMS_TOLERANCE:
        raise ManifestError(
            f"{where}: the terms come to {from_terms} bytes at "
            f"{served} tokens and memory_bytes_estimate says {stated} — "
            f"{drift:.1%} apart, past the {MEMORY_TERMS_TOLERANCE:.0%} a "
            f"parts-against-whole reading is allowed. One of the two is "
            f"wrong and the manifest does not say which; check for GB "
            f"where GiB was meant, and for a kv_bytes_per_token computed "
            f"from config.json on a backend whose card says otherwise"
        )


def _parse_memory(where: str, block: dict[str, Any], served: int) -> MemoryTerms | None:
    table = block.get("memory")
    if table is None:
        return None
    memory_where = f"{where}.memory"
    if not isinstance(table, dict):
        raise ManifestError(f"{memory_where}: must be a table")
    check_table(memory_where, table, _MEMORY_REQUIRED, error=ManifestError)
    _check_memory_values(memory_where, table)
    terms = MemoryTerms(
        weights_bytes=table["weights_bytes"],
        overhead_bytes=table["overhead_bytes"],
        kv_bytes_per_token=table["kv_bytes_per_token"],
        basis=table["basis"],
        measured_at_context=table["measured_at_context"],
    )
    _check_memory_agrees(memory_where, terms, served, block["memory_bytes_estimate"])
    return terms


MAX_STATED_BITS = 32


def _check_stated_bits(where: str, spec: BackendSpec) -> None:
    """`bits` is the block's weight precision, the second key of the automatic pick
    (docs/VERB-SIZING.md rule 3). Stated, so the pick never reads it out of a name; checked
    against what the block's file, repo and dtype imply, so the two can never disagree."""
    if spec.bits is None:
        return
    if below_floor(spec.bits):
        raise ManifestError(
            f"{where}: bits = {spec.bits}, and nothing under {MIN_WEIGHT_BITS} bits "
            "is ever offered (Owen, 2026-09-26: \"no less than 4\")"
        )
    if spec.bits > MAX_STATED_BITS:
        raise ManifestError(
            f"{where}: bits = {spec.bits}; no weights are wider than "
            f"{MAX_STATED_BITS} bits"
        )
    implied = implied_bits(replace(spec, bits=None))
    if implied is not None and implied != spec.bits:
        raise ManifestError(
            f"{where}: bits = {spec.bits}, but the block's file, repo or dtype says "
            f"{implied}-bit; one of the two is wrong"
        )


def _backend_block(where: str, kind: str, block: Any) -> dict[str, Any]:
    if kind not in BACKEND_ENGINES:
        raise ManifestError(
            f"{where}: {kind!r} is not a Crucible backend; the backends are "
            f"{sorted(BACKEND_ENGINES)}"
        )
    if not isinstance(block, dict):
        raise ManifestError(f"{where}: must be a table")
    check_table(where, block, _BACKEND_REQUIRED, _BACKEND_OPTIONAL, error=ManifestError)
    return block


def _parse_backend(kind: str, block: Any, path: Path, model: dict[str, Any]) -> BackendSpec:
    where = f"{path.name} [backends.{kind}]"
    block = _backend_block(where, kind, block)
    modalities = model["modalities"]
    serves_here = _parse_serves(where, block, modalities)
    _check_engine(where, kind, block["engine"], serves_here, modalities)
    _check_hf_pin(where, block)
    if block["memory_bytes_estimate"] <= 0:
        raise ManifestError(
            f"{where}: memory_bytes_estimate must be positive, got "
            f"{block['memory_bytes_estimate']}"
        )
    engine_args = _backend_engine_args(where, block)
    _check_backend_files(where, kind, block, serves_here)
    _check_vision_flags(where, serves_here, engine_args)
    trained = model["trained_context"]
    backend_context = _backend_context(where, block, trained)
    served = backend_context or model["context_default"]
    block_max = _backend_max_context(where, block, trained, served)
    spec = BackendSpec(
        backend=kind,
        engine=block["engine"],
        hf_repo=block["hf_repo"],
        revision=block["revision"],
        memory_bytes_estimate=block["memory_bytes_estimate"],
        engine_args=engine_args,
        context_default=backend_context,
        max_context=block_max,
        memory=_parse_memory(where, block, served),
        file=block.get("file"),
        mmproj=block.get("mmproj"),
        serves=serves_here,
        bits=block.get("bits"),
    )
    _check_stated_bits(where, spec)
    return spec


def _parse_backends(table: Any, path: Path, model: dict[str, Any]) -> dict[str, BackendSpec]:
    if not isinstance(table, dict):
        raise ManifestError(f"{path.name}: [backends] must hold one table per backend")
    if not table:
        raise ManifestError(
            f"{path.name}: no backend blocks; a model nothing can serve is not a model"
        )
    return {kind: _parse_backend(kind, block, path, model) for kind, block in table.items()}


def _parse(document: dict[str, Any], path: Path, expected_id: str) -> ModelManifest:
    _check_document(document, path)
    model = _parse_model(document, path, expected_id)
    backends = _parse_backends(document["backends"], path, model)
    modalities = tuple(model["modalities"])
    defaults = document.get("defaults")
    local = document.get("local")
    return ModelManifest(
        id=model["id"],
        family=model["family"],
        params_b=model["params_b"],
        context_default=model["context_default"],
        trained_context=model["trained_context"],
        modalities=modalities,
        backends=backends,
        path=path,
        defaults=NO_DEFAULTS if "defaults" not in document else _parse_defaults(defaults, path),
        display=model.get("display"),
        description=model.get("description"),
        weights_of=model.get("weights_of"),
        local=None if "local" not in document else _parse_local(local, path, modalities),
    )


WEIGHTS_OF_SHARED_FACTS: tuple[str, ...] = (
    "family",
    "params_b",
    "trained_context",
    "defaults",
)

WEIGHTS_OF_PIN_FIELDS: tuple[str, ...] = ("hf_repo", "revision", "file")


def _weights_base(manifest: ModelManifest, directory: Path) -> ModelManifest:
    where = manifest.path.name
    base_id = manifest.weights_of
    base_path = directory / f"{base_id}.toml"
    if not base_path.is_file():
        known = sorted(p.stem for p in directory.glob("*.toml"))
        raise ManifestError(
            f"{where}: weights_of_unknown — model.weights_of is {base_id!r} and "
            f"there is no {base_path.name} beside it; this catalog ships {known}"
        )
    base = _load_unresolved(base_id, directory)
    if base.weights_of is not None:
        raise ManifestError(
            f"{where}: weights_of_chain — model.weights_of is {base_id!r}, which "
            f"itself shares the weights of {base.weights_of!r}. An alias names "
            "the model that OWNS the download; a folder has one owner, so an "
            "alias of an alias is refused, and so is aliasing a base to "
            "something else"
        )
    return base


def _differing(here: Any, there: Any, names: tuple[str, ...], there_name: str) -> tuple[list[str], str]:
    differing = [name for name in names if getattr(here, name) != getattr(there, name)]
    detail = "; ".join(
        f"{name}: {getattr(here, name)!r} here, {getattr(there, name)!r} in {there_name}"
        for name in differing
    )
    return differing, detail


def _check_shared_facts(manifest: ModelManifest, base: ModelManifest) -> None:
    differing, detail = _differing(manifest, base, WEIGHTS_OF_SHARED_FACTS, base.path.name)
    if differing:
        raise ManifestError(
            f"{manifest.path.name}: weights_of_fact_mismatch — this model shares the weights "
            f"of {manifest.weights_of!r} and states {differing} differently ({detail}). "
            "These are facts about the weights, and the weights are one set of "
            "bytes"
        )


def _check_shared_pins(manifest: ModelManifest, base: ModelManifest) -> None:
    where = manifest.path.name
    for kind, spec in sorted(manifest.backends.items()):
        base_spec = base.backends.get(kind)
        if base_spec is None:
            raise ManifestError(
                f"{where}: weights_of_backend_missing — [backends.{kind}] is "
                f"declared here and {base.path.name} declares no {kind} block "
                f"(it declares {sorted(base.backends)}). There is no download of "
                f"{manifest.weights_of!r} on {kind} to share"
            )
        pins, detail = _differing(spec, base_spec, WEIGHTS_OF_PIN_FIELDS, base.path.name)
        if pins:
            raise ManifestError(
                f"{where}: weights_of_pin_mismatch — [backends.{kind}] shares "
                f"{manifest.weights_of!r}'s download and pins it differently ({detail}). "
                "One folder holds one pin"
            )


def resolve_weights_of(manifest: ModelManifest, directory: Path) -> ModelManifest:
    if manifest.weights_of is None:
        return manifest
    base = _weights_base(manifest, directory)
    _check_shared_facts(manifest, base)
    _check_shared_pins(manifest, base)
    return replace(manifest, weights_base=base)


def parse_manifest(
    text: str, path: Path, expected_id: str, *, directory: Path | None = None
) -> ModelManifest:
    manifest = _parse_text(text, path, expected_id)
    return resolve_weights_of(
        manifest, directory if directory is not None else path.parent
    )


def _parse_text(text: str, path: Path, expected_id: str) -> ModelManifest:
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(f"{path.name}: not valid TOML: {exc}") from exc
    return _parse(document, path, expected_id)


def _load_unresolved(model_id: str, root: Path) -> ModelManifest:
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
    return _parse_text(text, path, model_id)


def load_manifest(model_id: str, directory: Path | None = None) -> ModelManifest:
    root = directory if directory is not None else manifests_dir()
    return resolve_weights_of(_load_unresolved(model_id, root), root)


def aliases_of(manifest: ModelManifest) -> tuple[ModelManifest, ...]:
    if manifest.weights_of is not None:
        return ()
    found = load_all_manifests(manifest.path.parent)
    return tuple(
        other for other in found.values() if other.weights_of == manifest.id
    )


def load_all_manifests(directory: Path | None = None) -> dict[str, ModelManifest]:
    root = directory if directory is not None else manifests_dir()
    manifests: dict[str, ModelManifest] = {}
    for path in sorted(root.glob("*.toml"), key=lambda p: p.stem):
        manifests[path.stem] = load_manifest(path.stem, root)
    return manifests
