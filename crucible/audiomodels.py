from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

AUDIO_DIR_ENV = "CRUCIBLE_AUDIO_DIR"

STABLE_AUDIO_3 = "stable-audio-3"
YUE2 = "yue2"

AUDIO_BACKEND_ENGINES: dict[str, frozenset[str]] = {
    CUDA_LINUX: frozenset({STABLE_AUDIO_3, YUE2}),
    MLX_DARWIN: frozenset({STABLE_AUDIO_3, YUE2}),
}

ENGINE_DEVICE: dict[tuple[str, str], str] = {
    (STABLE_AUDIO_3, CUDA_LINUX): "cuda",
    (STABLE_AUDIO_3, MLX_DARWIN): "mps",
    (YUE2, CUDA_LINUX): "cuda",
    (YUE2, MLX_DARWIN): "mps",
}

SFX = "sfx"
MUSIC = "music"
SONG = "song"

KINDS: tuple[str, ...] = (SFX, MUSIC, SONG)

KIND_WORDS: dict[str, str] = {
    SFX: "sound effects",
    MUSIC: "instrumental music",
    SONG: "songs with sung lyrics",
}

TEXT_PARAM: dict[str, str] = {SFX: "prompt", MUSIC: "prompt", SONG: "tags"}

LYRICS_KINDS: frozenset[str] = frozenset({SONG})

OPTIONAL_PARAMS: tuple[str, ...] = ("negative_prompt", "duration_s", "steps", "cfg", "instrumental")

MEMORY_BASES = frozenset({"measured", "declared"})

DTYPES = frozenset({"bfloat16", "float16", "float32"})

CHANNELS = frozenset({1, 2})

_MODEL_REQUIRED: dict[str, type] = {
    "id": str,
    "family": str,
    "display": str,
    "kind": str,
    "licence": str,
    "licence_url": str,
    "commercial_use": str,
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
    "sample_rate": int,
    "channels": int,
    "max_duration_s": int,
    "takes": list,
}
_BACKEND_OPTIONAL: dict[str, Any] = {
    "default_duration_s": int,
    "default_steps": int,
    "max_steps": int,
    "default_cfg": (int, float),
    "max_cfg": (int, float),
    "not_taken": dict,
    "companions": list,
    "low_vram_memory_bytes_estimate": int,
    "low_vram_memory_note": str,
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
_TAKEN_DEFAULTS: dict[str, tuple[str, ...]] = {
    "duration_s": ("default_duration_s",),
    "steps": ("default_steps", "max_steps"),
    "cfg": ("default_cfg", "max_cfg"),
}


class AudioManifestError(CrucibleError):
    ...


@dataclass(frozen=True)
class CompanionFile:
    source: str
    target: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class Companion:
    name: str
    hf_repo: str
    revision: str
    files: tuple[CompanionFile, ...]

    @property
    def bytes(self) -> int:
        return sum(entry.bytes for entry in self.files)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "files": [entry.target for entry in self.files],
            "bytes": self.bytes,
        }


# The host setting that holds one part of a splittable audio model on the card at a time,
# named as an operator writes it in config.toml.
LOW_VRAM_SETTING = "[audio] low_vram"


@dataclass(frozen=True)
class HeldNeed:
    """What one audio model needs on this host, and whether `[audio] low_vram` is why."""

    bytes: int
    low_vram: bool


def held_need(whole_bytes: int, low_vram_bytes: int | None, host_low_vram: bool) -> HeldNeed:
    """The one rule for which need a host uses for an audio model: the manifest's low-VRAM
    figure when this host's `[audio] low_vram` is on and the model declares one, else the
    whole figure. The audio job admits against it and the capability verdict weighs it, so
    the two can never name different figures for one model on one host."""
    if host_low_vram and low_vram_bytes is not None:
        return HeldNeed(low_vram_bytes, True)
    return HeldNeed(whole_bytes, False)


@dataclass(frozen=True)
class AudioBackendSpec:

    backend: str
    engine: str
    hf_repo: str
    revision: str
    gated: bool
    dtype: str
    memory_bytes_estimate: int
    memory_basis: str
    memory_note: str
    files: tuple[str, ...]
    sample_rate: int
    channels: int
    max_duration_s: int
    takes: tuple[str, ...]
    default_duration_s: int | None = None
    default_steps: int | None = None
    max_steps: int | None = None
    default_cfg: float | None = None
    max_cfg: float | None = None
    not_taken: tuple[tuple[str, str], ...] = ()
    companions: tuple[Companion, ...] = ()
    # What the model needs when a host's `[audio] low_vram` holds only the part a stage
    # uses on the card; None for a model that cannot be split.
    low_vram_memory_bytes_estimate: int | None = None
    low_vram_memory_note: str | None = None

    @property
    def device(self) -> str:
        return ENGINE_DEVICE[(self.engine, self.backend)]

    def need_on(self, host_low_vram: bool) -> HeldNeed:
        return held_need(
            self.memory_bytes_estimate, self.low_vram_memory_bytes_estimate, host_low_vram
        )

    def why_not(self, param: str) -> str | None:
        return dict(self.not_taken).get(param)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "engine": self.engine,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "gated": self.gated,
            "dtype": self.dtype,
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "memory_basis": self.memory_basis,
            "memory_note": self.memory_note,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "max_duration_s": self.max_duration_s,
            "takes": list(self.takes),
            "default_duration_s": self.default_duration_s,
            "default_steps": self.default_steps,
            "max_steps": self.max_steps,
            "default_cfg": self.default_cfg,
            "max_cfg": self.max_cfg,
            "companions": [companion.to_dict() for companion in self.companions],
            "low_vram_memory_bytes_estimate": self.low_vram_memory_bytes_estimate,
            "low_vram_memory_note": self.low_vram_memory_note,
        }


@dataclass(frozen=True)
class AudioManifest:
    weights_family = "models"

    id: str
    family: str
    display: str
    kind: str
    licence: str
    licence_url: str
    commercial_use: str
    backends: dict[str, AudioBackendSpec]
    path: Path

    @property
    def pull_command(self) -> str:
        return f"{MODELS_PULL_COMMAND} {self.id}"

    @property
    def text_param(self) -> str:
        return TEXT_PARAM[self.kind]

    @property
    def takes_lyrics(self) -> bool:
        return self.kind in LYRICS_KINDS

    def aliases(self) -> "tuple[AudioManifest, ...]":
        return ()

    def supports(self, backend_kind: str) -> bool:
        return backend_kind in self.backends

    def spec(self, backend_kind: str) -> AudioBackendSpec:
        found = self.backends.get(backend_kind)
        if found is None:
            raise AudioManifestError(
                f"audio model {self.id!r} has no {backend_kind} block; "
                f"{self.path.name} declares {sorted(self.backends)}"
            )
        return found

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "display": self.display,
            "kind": self.kind,
            "licence": self.licence,
            "licence_url": self.licence_url,
            "commercial_use": self.commercial_use,
            "backends": {k: v.to_dict() for k, v in sorted(self.backends.items())},
        }


def audio_manifests_dir() -> Path:
    override = os.environ.get(AUDIO_DIR_ENV)
    if override is not None and override != "":
        path = Path(override).expanduser()
        if not path.is_dir():
            raise AudioManifestError(f"{AUDIO_DIR_ENV}={override!r} is not a directory")
        return path
    path = Path(__file__).resolve().parent / "audio"
    if not path.is_dir():
        raise AudioManifestError(
            f"no audio manifests at {path}; they are package data and this "
            f"install has lost them, or ${AUDIO_DIR_ENV} must point at them"
        )
    return path


def _check_document(document: dict[str, Any], path: Path) -> None:
    unknown = sorted(set(document) - {"model", "backends"})
    if unknown:
        raise AudioManifestError(
            f"{path.name}: unknown top-level table(s) {unknown}; an audio manifest "
            "has exactly [model] and [backends.<kind>]"
        )
    for table in ("model", "backends"):
        if not isinstance(document.get(table), dict):
            raise AudioManifestError(f"{path.name}: missing the [{table}] table")
    if not document["backends"]:
        raise AudioManifestError(
            f"{path.name}: no backend blocks; an audio model nothing can run is "
            "not an audio model"
        )


def _parse_model(model: dict[str, Any], path: Path, expected_id: str) -> str:
    check_table(f"{path.name} [model]", model, _MODEL_REQUIRED, error=AudioManifestError)
    model_id = model["id"]
    if not MODEL_ID_PATTERN.match(model_id):
        raise AudioManifestError(
            f"{path.name}: model.id {model_id!r} must be lower-case and start with "
            "a letter or digit ([a-z0-9][a-z0-9._-]*)"
        )
    if model_id != expected_id:
        raise AudioManifestError(
            f"{path.name}: model.id is {model_id!r} but the file is named "
            f"{expected_id!r}; the id and the filename are the same thing"
        )
    if model["kind"] not in KINDS:
        raise AudioManifestError(
            f"{path.name}: kind {model['kind']!r} is not one of {list(KINDS)}"
        )
    return model_id


def _check_pin(where: str, hf_repo: Any, revision: Any) -> None:
    if not HF_REPO_PATTERN.match(hf_repo):
        raise AudioManifestError(
            f"{where}: hf_repo {hf_repo!r} is not an <owner>/<name> HuggingFace repo id"
        )
    if not REVISION_PATTERN.match(revision):
        raise AudioManifestError(
            f"{where}: revision {revision!r} must be a full 40-character commit "
            "sha, so a pull is reproducible; branch names are not pins"
        )


def _check_engine(where: str, kind: str, block: dict[str, Any]) -> None:
    if kind not in AUDIO_BACKEND_ENGINES:
        raise AudioManifestError(
            f"{where}: {kind!r} is not an audio backend; the audio backends are "
            f"{sorted(AUDIO_BACKEND_ENGINES)}. Windows is never one "
            "(docs/internals/audio.md, \"Backends\")"
        )
    if block["engine"] not in AUDIO_BACKEND_ENGINES[kind]:
        raise AudioManifestError(
            f"{where}: engine {block['engine']!r} does not run on {kind}; that "
            f"backend's audio engines are {sorted(AUDIO_BACKEND_ENGINES[kind])}"
        )
    _check_pin(where, block["hf_repo"], block["revision"])
    if block["dtype"] not in DTYPES:
        raise AudioManifestError(
            f"{where}: dtype {block['dtype']!r} is not one of {sorted(DTYPES)}"
        )


def _check_low_vram(where: str, block: dict[str, Any]) -> None:
    keys = ("low_vram_memory_bytes_estimate", "low_vram_memory_note")
    present = [key in block for key in keys]
    if not any(present):
        return
    if not all(present):
        raise AudioManifestError(
            f"{where}: {keys[0]} and {keys[1]} go together; a low-VRAM figure says what "
            "it was measured on"
        )
    need = block["low_vram_memory_bytes_estimate"]
    if not 0 < need < block["memory_bytes_estimate"]:
        raise AudioManifestError(
            f"{where}: low_vram_memory_bytes_estimate must be positive and below "
            f"memory_bytes_estimate ({block['memory_bytes_estimate']}), got {need}"
        )
    if not block["low_vram_memory_note"].strip():
        raise AudioManifestError(f"{where}: low_vram_memory_note is empty")


def _check_memory(where: str, block: dict[str, Any]) -> None:
    _check_low_vram(where, block)
    if block["memory_bytes_estimate"] <= 0:
        raise AudioManifestError(f"{where}: memory_bytes_estimate must be positive")
    if block["memory_basis"] not in MEMORY_BASES:
        raise AudioManifestError(
            f"{where}: memory_basis {block['memory_basis']!r} is not one of "
            f"{sorted(MEMORY_BASES)}"
        )
    if not block["memory_note"].strip():
        raise AudioManifestError(
            f"{where}: memory_note is empty; it says where the number came from "
            "and at what duration it holds"
        )


def _strings(where: str, key: str, values: list[Any]) -> tuple[str, ...]:
    if not values or not all(isinstance(v, str) and v.strip() for v in values):
        raise AudioManifestError(f"{where}: {key} must be a non-empty list of strings")
    return tuple(values)


def _check_takes(where: str, block: dict[str, Any]) -> tuple[str, ...]:
    takes = tuple(block["takes"])
    unknown = sorted(set(takes) - set(OPTIONAL_PARAMS))
    if unknown or not all(isinstance(v, str) for v in takes):
        raise AudioManifestError(
            f"{where}: takes names {unknown or takes}; the optional audio params "
            f"are {list(OPTIONAL_PARAMS)}"
        )
    for param, keys in _TAKEN_DEFAULTS.items():
        for key in keys:
            if (param in takes) != (key in block):
                raise AudioManifestError(
                    f"{where}: {key} goes with {param!r} in takes, and only with it"
                )
    return takes


def _check_audio_shape(where: str, block: dict[str, Any]) -> None:
    if block["sample_rate"] <= 0 or block["channels"] not in CHANNELS:
        raise AudioManifestError(
            f"{where}: sample_rate must be positive and channels 1 or 2"
        )
    if block["max_duration_s"] <= 0:
        raise AudioManifestError(f"{where}: max_duration_s must be positive")
    default = block.get("default_duration_s")
    if default is not None and not 0 < default <= block["max_duration_s"]:
        raise AudioManifestError(
            f"{where}: default_duration_s {default} is outside 1..{block['max_duration_s']}"
        )
    for low, high in (("default_steps", "max_steps"), ("default_cfg", "max_cfg")):
        if low in block and not 0 <= block[low] <= block[high]:
            raise AudioManifestError(f"{where}: {low} must be between 0 and {high}")


def _not_taken(where: str, block: dict[str, Any], takes: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    table = block.get("not_taken", {})
    for param, why in table.items():
        if param not in OPTIONAL_PARAMS or param in takes or not isinstance(why, str):
            raise AudioManifestError(
                f"{where}: not_taken.{param} must name an optional param this arm "
                "does not take, with the reason as a string"
            )
    return tuple(sorted(table.items()))


def _companion_file(where: str, entry: Any) -> CompanionFile:
    if not isinstance(entry, dict):
        raise AudioManifestError(f"{where}: each companion file is a table")
    check_table(where, entry, _COMPANION_FILE_REQUIRED, error=AudioManifestError)
    if not SHA256_PATTERN.match(entry["sha256"]):
        raise AudioManifestError(f"{where}: sha256 must be 64 lower-case hex digits")
    return CompanionFile(**entry)


def _companion(where: str, entry: Any) -> Companion:
    if not isinstance(entry, dict):
        raise AudioManifestError(f"{where}: each companion is a table")
    check_table(where, entry, _COMPANION_REQUIRED, error=AudioManifestError)
    _check_pin(where, entry["hf_repo"], entry["revision"])
    if not MODEL_ID_PATTERN.match(entry["name"]):
        raise AudioManifestError(f"{where}: companion name {entry['name']!r} is not a plain name")
    files = tuple(
        _companion_file(f"{where} files[{index}]", item)
        for index, item in enumerate(entry["files"])
    )
    if not files:
        raise AudioManifestError(f"{where}: a companion with no files pulls nothing")
    return Companion(entry["name"], entry["hf_repo"], entry["revision"], files)


def _parse_backend(path: Path, kind: str, block: Any) -> AudioBackendSpec:
    where = f"{path.name} [backends.{kind}]"
    if not isinstance(block, dict):
        raise AudioManifestError(f"{where}: must be a table")
    check_table(where, block, _BACKEND_REQUIRED, _BACKEND_OPTIONAL, error=AudioManifestError)
    _check_engine(where, kind, block)
    _check_memory(where, block)
    _check_audio_shape(where, block)
    takes = _check_takes(where, block)
    companions = tuple(
        _companion(f"{where} companions[{index}]", entry)
        for index, entry in enumerate(block.get("companions", []))
    )
    return AudioBackendSpec(
        backend=kind,
        engine=block["engine"],
        hf_repo=block["hf_repo"],
        revision=block["revision"],
        gated=block["gated"],
        dtype=block["dtype"],
        memory_bytes_estimate=block["memory_bytes_estimate"],
        memory_basis=block["memory_basis"],
        memory_note=block["memory_note"],
        files=_strings(where, "files", block["files"]),
        sample_rate=block["sample_rate"],
        channels=block["channels"],
        max_duration_s=block["max_duration_s"],
        takes=takes,
        default_duration_s=block.get("default_duration_s"),
        default_steps=block.get("default_steps"),
        max_steps=block.get("max_steps"),
        default_cfg=None if "default_cfg" not in block else float(block["default_cfg"]),
        max_cfg=None if "max_cfg" not in block else float(block["max_cfg"]),
        not_taken=_not_taken(where, block, takes),
        companions=companions,
        low_vram_memory_bytes_estimate=block.get("low_vram_memory_bytes_estimate"),
        low_vram_memory_note=block.get("low_vram_memory_note"),
    )


def _parse(document: dict[str, Any], path: Path, expected_id: str) -> AudioManifest:
    _check_document(document, path)
    model = document["model"]
    model_id = _parse_model(model, path, expected_id)
    backends = {
        kind: _parse_backend(path, kind, block)
        for kind, block in document["backends"].items()
    }
    return AudioManifest(
        id=model_id,
        family=model["family"],
        display=model["display"],
        kind=model["kind"],
        licence=model["licence"],
        licence_url=model["licence_url"],
        commercial_use=model["commercial_use"],
        backends=backends,
        path=path,
    )


def parse_audio_manifest(text: str, path: Path, expected_id: str) -> AudioManifest:
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise AudioManifestError(f"{path.name}: not valid TOML: {exc}") from exc
    return _parse(document, path, expected_id)


def load_audio_manifest(model_id: str, directory: Path | None = None) -> AudioManifest:
    root = directory if directory is not None else audio_manifests_dir()
    path = root / f"{model_id}.toml"
    if not path.is_file():
        known = sorted(p.stem for p in root.glob("*.toml"))
        raise AudioManifestError(
            f"no audio manifest for {model_id!r} at {path}; this build ships {known}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AudioManifestError(f"could not read {path}: {exc}") from exc
    return parse_audio_manifest(text, path, model_id)


def load_all_audio_manifests(directory: Path | None = None) -> dict[str, AudioManifest]:
    root = directory if directory is not None else audio_manifests_dir()
    return {
        path.stem: load_audio_manifest(path.stem, root)
        for path in sorted(root.glob("*.toml"), key=lambda p: p.stem)
    }


def _of_kind(kind: str, directory: Path | None) -> dict[str, AudioManifest]:
    return {
        model_id: manifest
        for model_id, manifest in load_all_audio_manifests(directory).items()
        if manifest.kind == kind
    }


def load_sfx_manifests(directory: Path | None = None) -> dict[str, AudioManifest]:
    return _of_kind(SFX, directory)


def load_music_manifests(directory: Path | None = None) -> dict[str, AudioManifest]:
    return _of_kind(MUSIC, directory)


def load_song_manifests(directory: Path | None = None) -> dict[str, AudioManifest]:
    return _of_kind(SONG, directory)


def engines_on(backend_kind: str) -> tuple[str, ...]:
    return tuple(sorted(AUDIO_BACKEND_ENGINES.get(backend_kind, ())))


__all__ = [
    "AUDIO_BACKEND_ENGINES",
    "AUDIO_DIR_ENV",
    "AudioBackendSpec",
    "AudioManifest",
    "AudioManifestError",
    "Companion",
    "CompanionFile",
    "HeldNeed",
    "KINDS",
    "KIND_WORDS",
    "LOW_VRAM_SETTING",
    "MUSIC",
    "OPTIONAL_PARAMS",
    "SFX",
    "SONG",
    "STABLE_AUDIO_3",
    "YUE2",
    "audio_manifests_dir",
    "engines_on",
    "held_need",
    "load_all_audio_manifests",
    "load_audio_manifest",
    "load_music_manifests",
    "load_sfx_manifests",
    "load_song_manifests",
    "parse_audio_manifest",
]
