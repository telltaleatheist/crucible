from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .backend import CUDA_LINUX, MLX_DARWIN
from .errors import CrucibleError
from .manifests import MODELS_PULL_COMMAND
from .tomltable import HF_REPO_PATTERN, MODEL_ID_PATTERN, REVISION_PATTERN, check_table

ASR_DIR_ENV = "CRUCIBLE_ASR_DIR"

VLLM_ENGINE = "vllm"
MLX_AUDIO_ENGINE = "mlx-audio"
QWEN_ASR_TORCH_ENGINE = "qwen-asr"
QWEN_ASR_ENGINES: frozenset[str] = frozenset(
    {VLLM_ENGINE, MLX_AUDIO_ENGINE, QWEN_ASR_TORCH_ENGINE}
)

ASR_BACKEND_ENGINES: dict[str, frozenset[str]] = {
    CUDA_LINUX: frozenset({"faster-whisper", VLLM_ENGINE}),
    MLX_DARWIN: frozenset({"mlx-whisper", MLX_AUDIO_ENGINE, QWEN_ASR_TORCH_ENGINE}),
}

ASR_ENGINE_FAMILY: dict[str, str] = {
    "faster-whisper": "whisper",
    "mlx-whisper": "whisper",
    VLLM_ENGINE: "qwen3-asr",
    MLX_AUDIO_ENGINE: "qwen3-asr",
    QWEN_ASR_TORCH_ENGINE: "qwen3-asr",
}

ONE_CHECKPOINT_FAMILIES: frozenset[str] = frozenset({"qwen3-asr"})

ASR_LINEUP: frozenset[str] = frozenset(
    {
        "qwen3-asr-1.7b", "qwen3-asr-1.7b-mlx",
        "qwen3-asr-0.6b", "qwen3-asr-0.6b-mlx",
        "whisper-large-v3-turbo", "whisper-tiny",
    }
)

QWEN_ASR_DTYPES: frozenset[str] = frozenset({"bfloat16"})

QWEN_BACKEND_REQUIRED: dict[str, type] = {
    "dtype": str,
    "aligner": str,
    "max_batch": int,
    "max_new_tokens": int,
}

VLLM_BACKEND_REQUIRED: dict[str, type] = {
    "max_model_len": int,
    "kv_cache_memory_bytes": int,
    "kv_bytes_per_token": int,
}

QWEN_PIECE_MAX_SECONDS = 180

QWEN_AUDIO_TOKENS_PER_SECOND = 13

QWEN_CONTEXT_MAX_TOKENS = 1024

QWEN_PROMPT_SCAFFOLD_TOKENS = 64


def qwen_prompt_ceiling_tokens(max_new_tokens: int) -> int:
    return (
        QWEN_PIECE_MAX_SECONDS * QWEN_AUDIO_TOKENS_PER_SECOND
        + QWEN_CONTEXT_MAX_TOKENS
        + QWEN_PROMPT_SCAFFOLD_TOKENS
        + max_new_tokens
    )

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


class AsrManifestError(CrucibleError):
    ...


@dataclass(frozen=True)
class AsrBackendSpec:

    backend: str
    engine: str
    hf_repo: str
    revision: str
    memory_bytes_estimate: int
    dtype: str | None = None
    aligner: str | None = None
    max_batch: int | None = None
    max_new_tokens: int | None = None
    max_model_len: int | None = None
    kv_cache_memory_bytes: int | None = None
    kv_bytes_per_token: int | None = None

    @property
    def files(self) -> tuple[str, ...]:
        return ()

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "backend": self.backend,
            "engine": self.engine,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "memory_bytes_estimate": self.memory_bytes_estimate,
        }
        for key in (*QWEN_BACKEND_REQUIRED, *VLLM_BACKEND_REQUIRED):
            value = getattr(self, key)
            if value is not None:
                row[key] = value
        return row

    def require(self, key: str) -> Any:
        value = getattr(self, key)
        if value is None:
            raise AsrManifestError(
                f"the {self.backend} block has no {key!r}, and the "
                f"{self.engine!r} engine requires it"
            )
        return value


@dataclass(frozen=True)
class AsrManifest:
    weights_family = "models"

    id: str
    family: str
    parameters_m: int
    backends: dict[str, AsrBackendSpec]
    path: Path
    weights_of: str | None = None
    weights_base: "AsrManifest | None" = field(default=None, compare=False, repr=False)

    @property
    def pull_command(self) -> str:
        return f"{MODELS_PULL_COMMAND} {self.id}"

    def aliases(self) -> "tuple[AsrManifest, ...]":
        return asr_aliases_of(self)

    def extra_files(self, backend_kind: str) -> tuple[str, ...]:
        return ()

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


def asr_manifests_dir() -> Path:
    override = os.environ.get(ASR_DIR_ENV)
    if override is not None and override != "":
        path = Path(override).expanduser()
        if not path.is_dir():
            raise AsrManifestError(f"{ASR_DIR_ENV}={override!r} is not a directory")
        return path
    path = Path(__file__).resolve().parent / "asr"
    if not path.is_dir():
        raise AsrManifestError(
            f"no ASR manifests at {path}; they are package data and this "
            f"install has lost them, or ${ASR_DIR_ENV} must point at them"
        )
    return path


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
    model = dict(model)
    weights_of = model.pop("weights_of", None)
    if weights_of is not None and not (isinstance(weights_of, str) and MODEL_ID_PATTERN.match(weights_of)):
        raise AsrManifestError(f"{path.name}: model.weights_of {weights_of!r} is not a model id")
    check_table(f"{path.name} [model]", model, _MODEL_REQUIRED, error=AsrManifestError)

    model_id = model["id"]
    if not MODEL_ID_PATTERN.match(model_id):
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
        engine = block.get("engine")
        if not isinstance(engine, str):
            check_table(where, block, _BACKEND_REQUIRED, error=AsrManifestError)
            raise AsrManifestError(f"{where}: engine must be str")
        if engine not in ASR_BACKEND_ENGINES[kind]:
            raise AsrManifestError(
                f"{where}: engine {engine!r} does not run asr on {kind}; that "
                f"backend's asr engines are {sorted(ASR_BACKEND_ENGINES[kind])}. "
                "faster-whisper (CTranslate2) and vllm have no Metal backend; "
                "mlx-whisper and mlx-audio are MLX, which has no CUDA one, and "
                "qwen-asr is offered only where the MLX port needed an "
                "alternative (the Mac)"
            )
        required = dict(_BACKEND_REQUIRED)
        if engine in QWEN_ASR_ENGINES:
            required.update(QWEN_BACKEND_REQUIRED)
        if engine == VLLM_ENGINE:
            required.update(VLLM_BACKEND_REQUIRED)
        check_table(where, block, required, error=AsrManifestError)
        engine_family = ASR_ENGINE_FAMILY[engine]
        if model["family"] != engine_family:
            raise AsrManifestError(
                f"{where}: engine {engine!r} runs the {engine_family!r} family "
                f"and this manifest's [model] family is {model['family']!r}; "
                "every block of one asr id runs the same model"
            )
        if not HF_REPO_PATTERN.match(block["hf_repo"]):
            raise AsrManifestError(
                f"{where}: hf_repo {block['hf_repo']!r} is not an <owner>/<name> "
                "HuggingFace repo id"
            )
        if not REVISION_PATTERN.match(block["revision"]):
            raise AsrManifestError(
                f"{where}: revision {block['revision']!r} must be a full 40-character "
                "commit sha, so a pull is reproducible; branch names are not pins"
            )
        if block["memory_bytes_estimate"] <= 0:
            raise AsrManifestError(
                f"{where}: memory_bytes_estimate must be positive, got "
                f"{block['memory_bytes_estimate']}"
            )
        if engine in QWEN_ASR_ENGINES:
            _check_qwen_block(where, engine, block)
        backends[kind] = AsrBackendSpec(
            backend=kind,
            engine=engine,
            hf_repo=block["hf_repo"],
            revision=block["revision"],
            memory_bytes_estimate=block["memory_bytes_estimate"],
            dtype=block.get("dtype"),
            aligner=block.get("aligner"),
            max_batch=block.get("max_batch"),
            max_new_tokens=block.get("max_new_tokens"),
            max_model_len=block.get("max_model_len"),
            kv_cache_memory_bytes=block.get("kv_cache_memory_bytes"),
            kv_bytes_per_token=block.get("kv_bytes_per_token"),
        )

    if not model_id.startswith(f"{model['family']}-"):
        raise AsrManifestError(
            f"{path.name}: model.id {model_id!r} must begin with its family "
            f"({model['family']!r}) and a hyphen"
        )

    pins = sorted({(spec.hf_repo, spec.revision) for spec in backends.values()})
    if model["family"] in ONE_CHECKPOINT_FAMILIES and len(pins) > 1:
        raise AsrManifestError(
            f"{path.name}: its backend blocks pin different weights "
            f"({[f'{repo}@{revision[:12]}' for repo, revision in pins]}); a "
            f"{model['family']!r} id is one checkpoint on every backend, because "
            "both of its engines read the official weights"
        )

    if weights_of == model_id:
        raise AsrManifestError(f"{path.name}: model.weights_of names this model itself")
    return AsrManifest(
        id=model_id,
        family=model["family"],
        parameters_m=model["parameters_m"],
        backends=backends,
        path=path,
        weights_of=weights_of,
    )


def _check_qwen_block(where: str, engine: str, block: dict[str, Any]) -> None:
    if block["dtype"] not in QWEN_ASR_DTYPES:
        raise AsrManifestError(
            f"{where}: dtype {block['dtype']!r} is not one this engine runs; "
            f"Qwen3-ASR runs in {sorted(QWEN_ASR_DTYPES)} on both machines "
            "(Owen, 2026-09-24: full precision, never the 8-bit build)"
        )
    if not MODEL_ID_PATTERN.match(block["aligner"]):
        raise AsrManifestError(
            f"{where}: aligner {block['aligner']!r} is not an align model id"
        )
    positive = ["max_batch", "max_new_tokens"]
    if engine == VLLM_ENGINE:
        positive += list(VLLM_BACKEND_REQUIRED)
    for key in positive:
        if block[key] <= 0:
            raise AsrManifestError(f"{where}: {key} must be positive, got {block[key]}")
    if engine in (MLX_AUDIO_ENGINE, QWEN_ASR_TORCH_ENGINE) and block["max_batch"] != 1:
        raise AsrManifestError(
            f"{where}: max_batch is {block['max_batch']}, and {engine} decodes "
            "one piece per call here (the loop guard reads each piece's own "
            "token count), so the only true value is 1"
        )
    if engine == VLLM_ENGINE:
        ceiling = qwen_prompt_ceiling_tokens(block["max_new_tokens"])
        if block["max_model_len"] < ceiling:
            raise AsrManifestError(
                f"{where}: max_model_len {block['max_model_len']} cannot hold the "
                f"longest piece: {QWEN_PIECE_MAX_SECONDS} s of audio is "
                f"{QWEN_PIECE_MAX_SECONDS * QWEN_AUDIO_TOKENS_PER_SECOND} tokens, "
                f"and with a {QWEN_CONTEXT_MAX_TOKENS}-token context, the "
                f"scaffolding and {block['max_new_tokens']} new tokens that is "
                f"{ceiling}"
            )


def parse_asr_manifest(text: str, path: Path, expected_id: str) -> AsrManifest:
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise AsrManifestError(f"{path.name}: not valid TOML: {exc}") from exc
    return _parse(document, path, expected_id)


def load_asr_manifest(model_id: str, directory: Path | None = None) -> AsrManifest:
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
    return _resolve_weights_of(parse_asr_manifest(text, path, model_id), root)


def _resolve_weights_of(manifest: AsrManifest, root: Path) -> AsrManifest:
    if manifest.weights_of is None:
        return manifest
    where = manifest.path.name
    base_path = root / f"{manifest.weights_of}.toml"
    if not base_path.is_file():
        raise AsrManifestError(
            f"{where}: weights_of_unknown: {manifest.weights_of!r} has no manifest beside it"
        )
    base = parse_asr_manifest(base_path.read_text(encoding="utf-8"), base_path, manifest.weights_of)
    if base.weights_of is not None:
        raise AsrManifestError(
            f"{where}: weights_of_chain: {base.id!r} itself shares {base.weights_of!r}"
        )
    if base.family != manifest.family:
        raise AsrManifestError(
            f"{where}: weights_of family {manifest.family!r} differs from {base.id!r}'s {base.family!r}"
        )
    for kind, spec in sorted(manifest.backends.items()):
        base_spec = base.backends.get(kind)
        if base_spec is None:
            raise AsrManifestError(
                f"{where}: weights_of_backend_missing: {base.id!r} has no {kind} block to share"
            )
        if (spec.hf_repo, spec.revision) != (base_spec.hf_repo, base_spec.revision):
            raise AsrManifestError(
                f"{where}: weights_of_pin_mismatch on {kind}: {spec.hf_repo}@{spec.revision[:12]} "
                f"here, {base_spec.hf_repo}@{base_spec.revision[:12]} in {base.path.name}"
            )
    return replace(manifest, weights_base=base)


def asr_aliases_of(manifest: AsrManifest) -> tuple[AsrManifest, ...]:
    if manifest.weights_of is not None:
        return ()
    found = load_all_asr_manifests(manifest.path.parent)
    return tuple(other for other in found.values() if other.weights_of == manifest.id)


def load_all_asr_manifests(directory: Path | None = None) -> dict[str, AsrManifest]:
    root = directory if directory is not None else asr_manifests_dir()
    manifests: dict[str, AsrManifest] = {}
    for path in sorted(root.glob("*.toml"), key=lambda p: p.stem):
        manifests[path.stem] = load_asr_manifest(path.stem, root)
    return manifests
