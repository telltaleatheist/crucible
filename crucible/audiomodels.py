from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field, replace
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

OPTIONAL_PARAMS: tuple[str, ...] = (
    "negative_prompt",
    "duration_s",
    "steps",
    "cfg",
    "instrumental",
    # A song's length range (docs/AUDIO.md "Song length"): checked against its score
    # before anything is composed.
    "min_duration_s",
    "max_duration_s",
    # A cover (docs/AUDIO.md "Covers"): the input carrying a score to render instead of
    # planning one, and whether its chord symbols are kept.
    "score",
    "chords",
)

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
_MODEL_OPTIONAL: dict[str, type] = {
    "keep_calls_together": bool,
    # The model whose weights these are (FROM, in Ollama's words): its main repo and its
    # companions, pulled once into that model's folder. This model adds only companions
    # of its own beside them (an adapter: [backends.<kind>.adapter]).
    "weights_of": str,
    # The model makes instrumentals only: every request sends `instrumental: true`.
    "instrumental_only": bool,
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
    "host_memory_bytes_estimate": int,
    "host_memory_note": str,
    "adapter": dict,
}
_ADAPTER_REQUIRED: dict[str, Any] = {
    "companion": str,
    "file": str,
    "rank": int,
    "scale": (int, float),
    "layers": int,
    "targets": list,
    "applies_to": list,
    "note": str,
}
# The stages an adapter can be folded around. Only composing is built: YuE2 writes the
# song's tokens and synthesizes from them with one backbone, so "compose" is both passes
# (yue2_worker.YuE2Engine._stages), and the score is always planned on the base weights
# (docs/AUDIO.md "yue2-3b-instrumental").
ADAPTER_STAGES: tuple[str, ...] = ("compose",)
# Where an adapter can be folded: through the CUDA worker's host homes, which hold the
# base weights it is taken out again from (yue2_worker.HostHomes.folded). The Mac's worker
# keeps no host home, so it has no exact way back to the base yet.
ADAPTER_BACKENDS: frozenset[str] = frozenset({CUDA_LINUX})
# One linear inside every layer, named as the adapter file names it (`self_attn.q_proj`);
# the worker finds it at `model.layers.<i>.<target>`.
ADAPTER_TARGET_PATTERN = re.compile(r"^[a-z_]+\.[a-z_]+$")
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


@dataclass(frozen=True)
class Adapter:
    """A low-rank delta the worker folds into the base's linears in memory, around the
    stages the manifest names, and takes out again after them (W += scale * B @ A, never
    on disk: weights_manifest.json's hashes stay the base's). Its file is a companion of
    the arm, pulled and hash-checked like any other part."""

    companion: Companion
    file: CompanionFile
    rank: int
    scale: float
    layers: int
    targets: tuple[str, ...]
    applies_to: tuple[str, ...]
    note: str

    def identity(self) -> dict[str, Any]:
        """What names this adapter's bytes: the record a song carries and the worker is
        held to (its sha256, re-hashed at load)."""
        return {
            "name": self.companion.name,
            "hf_repo": self.companion.hf_repo,
            "revision": self.companion.revision,
            "file": self.file.source,
            "sha256": self.file.sha256,
            "bytes": self.file.bytes,
            "rank": self.rank,
            "scale": self.scale,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity(),
            "layers": self.layers,
            "targets": list(self.targets),
            "applies_to": list(self.applies_to),
            "note": self.note,
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
    # What the worker keeps in host memory while it serves, the card setting either way
    # (doctor weighs it against this machine's memory: crucible/hostmemory.py); None for a
    # model that keeps nothing there worth weighing.
    host_memory_bytes_estimate: int | None = None
    host_memory_note: str | None = None
    # The delta folded into the base around some stages (`[backends.<kind>.adapter]`);
    # None for a model that runs its weights as they are.
    adapter: Adapter | None = None

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
            "host_memory_bytes_estimate": self.host_memory_bytes_estimate,
            "host_memory_note": self.host_memory_note,
            "adapter": None if self.adapter is None else self.adapter.to_dict(),
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
    # A model slow enough to load that its queued calls run back to back before anything
    # that would take it off the card (crucible/keeptogether.py, docs/QUEUE.md).
    keep_calls_together: bool = False
    # The model whose weights these are, and that model (resolved on load); None for a
    # model that owns its download.
    weights_of: str | None = None
    weights_base: "AudioManifest | None" = field(default=None, compare=False, repr=False)
    instrumental_only: bool = False

    @property
    def pull_command(self) -> str:
        return f"{MODELS_PULL_COMMAND} {self.id}"

    @property
    def text_param(self) -> str:
        return TEXT_PARAM[self.kind]

    @property
    def takes_lyrics(self) -> bool:
        return self.kind in LYRICS_KINDS

    @property
    def makes(self) -> str:
        """What the model makes, in words a page shows."""
        if self.instrumental_only:
            return "instrumental songs (nothing is sung)"
        return KIND_WORDS[self.kind]

    def aliases(self) -> "tuple[AudioManifest, ...]":
        """The models that share this one's weights (`weights_of`), so removing it is
        refused while one of them is pulled (weights.refuse_if_shared)."""
        if self.weights_of is not None:
            return ()
        found = load_all_audio_manifests(self.path.parent)
        return tuple(other for other in found.values() if other.weights_of == self.id)

    def extra_files(self, backend_kind: str) -> tuple[str, ...]:
        """Main-repo files this model adds to its base's: none, since an audio model that
        shares weights declares exactly its base's (`_resolve_weights_of`)."""
        return ()

    def own_companions(self, backend_kind: str) -> tuple[Companion, ...]:
        """The companions this model owns: all of them, or for a model that shares its
        base's weights only those its base does not declare (its adapter)."""
        spec = self.spec(backend_kind)
        if self.weights_base is None:
            return spec.companions
        shared = self.weights_base.spec(backend_kind).companions
        return tuple(companion for companion in spec.companions if companion not in shared)

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
            "weights_of": self.weights_of,
            "instrumental_only": self.instrumental_only,
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
    check_table(
        f"{path.name} [model]", model, _MODEL_REQUIRED, _MODEL_OPTIONAL,
        error=AudioManifestError,
    )
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
    weights_of = model.get("weights_of")
    if weights_of is not None and (not MODEL_ID_PATTERN.match(weights_of) or weights_of == model_id):
        raise AudioManifestError(
            f"{path.name}: model.weights_of {weights_of!r} must name another audio model"
        )
    if model.get("instrumental_only") and model["kind"] != SONG:
        raise AudioManifestError(
            f"{path.name}: instrumental_only is for a song model, whose `instrumental` "
            f"switch it fixes; this one makes {KIND_WORDS[model['kind']]}"
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


def _check_host_memory(where: str, block: dict[str, Any]) -> None:
    keys = ("host_memory_bytes_estimate", "host_memory_note")
    present = [key in block for key in keys]
    if not any(present):
        return
    if not all(present):
        raise AudioManifestError(
            f"{where}: {keys[0]} and {keys[1]} go together; a host-memory figure says what "
            "it was measured on"
        )
    if block["host_memory_bytes_estimate"] <= 0:
        raise AudioManifestError(
            f"{where}: host_memory_bytes_estimate must be positive, got "
            f"{block['host_memory_bytes_estimate']}"
        )
    if not block["host_memory_note"].strip():
        raise AudioManifestError(f"{where}: host_memory_note is empty")


def _check_memory(where: str, block: dict[str, Any]) -> None:
    _check_low_vram(where, block)
    _check_host_memory(where, block)
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


def _adapter_file(where: str, table: dict[str, Any], companions: tuple[Companion, ...]) -> tuple[Companion, CompanionFile]:
    carrier = next((c for c in companions if c.name == table["companion"]), None)
    if carrier is None:
        raise AudioManifestError(
            f"{where}: companion {table['companion']!r} is not one of this arm's companions "
            f"{[c.name for c in companions]}; the adapter's file is pulled as a companion"
        )
    found = next((f for f in carrier.files if f.target == table["file"]), None)
    if found is None:
        raise AudioManifestError(
            f"{where}: companion {carrier.name!r} places {[f.target for f in carrier.files]}, "
            f"not {table['file']!r}"
        )
    return carrier, found


def _adapter(where: str, kind: str, block: dict[str, Any], companions: tuple[Companion, ...]) -> Adapter | None:
    table = block.get("adapter")
    if table is None:
        return None
    where = f"{where} adapter"
    check_table(where, table, _ADAPTER_REQUIRED, error=AudioManifestError)
    if kind not in ADAPTER_BACKENDS or block["engine"] != YUE2:
        raise AudioManifestError(
            f"{where}: an adapter is folded through the YuE2 worker's host homes on "
            f"{sorted(ADAPTER_BACKENDS)}, not on {kind} with engine {block['engine']!r}"
        )
    carrier, found = _adapter_file(where, table, companions)
    targets = tuple(table["targets"])
    if not targets or len(set(targets)) != len(targets) or not all(
        isinstance(t, str) and ADAPTER_TARGET_PATTERN.match(t) for t in targets
    ):
        raise AudioManifestError(
            f"{where}: targets must be distinct names such as 'self_attn.q_proj', got {list(targets)}"
        )
    if table["applies_to"] != list(ADAPTER_STAGES):
        raise AudioManifestError(
            f"{where}: applies_to {table['applies_to']} is not built; an adapter is folded "
            f"around {list(ADAPTER_STAGES)} only, and the score is planned on the base"
        )
    if table["rank"] <= 0 or table["layers"] <= 0 or not table["scale"] > 0 or not table["note"].strip():
        raise AudioManifestError(f"{where}: rank, layers and scale must be positive and note said")
    return Adapter(
        companion=carrier, file=found, rank=table["rank"], scale=float(table["scale"]),
        layers=table["layers"], targets=targets, applies_to=tuple(table["applies_to"]),
        note=table["note"],
    )


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
        host_memory_bytes_estimate=block.get("host_memory_bytes_estimate"),
        host_memory_note=block.get("host_memory_note"),
        adapter=_adapter(where, kind, block, companions),
    )


def _check_instrumental_only(path: Path, model: dict[str, Any], backends: dict[str, AudioBackendSpec]) -> None:
    if not model.get("instrumental_only"):
        return
    for kind, spec in sorted(backends.items()):
        if "instrumental" not in spec.takes:
            raise AudioManifestError(
                f"{path.name} [backends.{kind}]: an instrumental_only model takes "
                "`instrumental`, which every request sends as true"
            )


def _parse(document: dict[str, Any], path: Path, expected_id: str) -> AudioManifest:
    _check_document(document, path)
    model = document["model"]
    model_id = _parse_model(model, path, expected_id)
    backends = {
        kind: _parse_backend(path, kind, block)
        for kind, block in document["backends"].items()
    }
    _check_instrumental_only(path, model, backends)
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
        keep_calls_together=model.get("keep_calls_together", False),
        weights_of=model.get("weights_of"),
        instrumental_only=model.get("instrumental_only", False),
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
    return _resolve_weights_of(parse_audio_manifest(text, path, model_id), root)


def _check_shared_arm(where: str, kind: str, spec: AudioBackendSpec, base: "AudioManifest") -> None:
    base_spec = base.backends.get(kind)
    if base_spec is None:
        raise AudioManifestError(f"{where}: {base.id!r} has no {kind} block to share")
    same = ("engine", "hf_repo", "revision", "dtype", "files")
    differs = [key for key in same if getattr(spec, key) != getattr(base_spec, key)]
    if differs:
        raise AudioManifestError(
            f"{where}: on {kind} its {', '.join(differs)} differ from {base.path.name}'s; a "
            "model that shares weights declares the same main download"
        )
    missing = [c.name for c in base_spec.companions if c not in spec.companions]
    if missing:
        raise AudioManifestError(
            f"{where}: on {kind} it lacks {base.id!r}'s companion(s) {missing} as "
            f"{base.path.name} pins them; the shared folder holds them once, at one pin"
        )


def _resolve_weights_of(manifest: AudioManifest, root: Path) -> AudioManifest:
    """FROM a base: the base's manifest is read beside this one, and every arm must pin
    the base's main download and companions exactly, so pulling this model fetches only
    its own companions when the base is there (weights.pull, `_pull_alias`)."""
    if manifest.weights_of is None:
        return manifest
    where = manifest.path.name
    base_path = root / f"{manifest.weights_of}.toml"
    if not base_path.is_file():
        raise AudioManifestError(f"{where}: weights_of {manifest.weights_of!r} has no manifest beside it")
    base = parse_audio_manifest(base_path.read_text(encoding="utf-8"), base_path, manifest.weights_of)
    if base.weights_of is not None or base.family != manifest.family:
        raise AudioManifestError(
            f"{where}: weights_of names {base.id!r}, which must own its weights and be of "
            f"family {manifest.family!r}"
        )
    for kind, spec in sorted(manifest.backends.items()):
        _check_shared_arm(where, kind, spec, base)
    return replace(manifest, weights_base=base)


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
    "ADAPTER_BACKENDS",
    "ADAPTER_STAGES",
    "AUDIO_BACKEND_ENGINES",
    "AUDIO_DIR_ENV",
    "Adapter",
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
