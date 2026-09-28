from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backend import CUDA_LINUX, MLX_DARWIN
from .errors import EngineError
from .narratorengines import DOCUMENT_READERS
from .voicereference import ClipEntry, VoiceReference, place
from .voices import VoiceBackendSpec, VoiceManifest
from .weights import PINNED

DOCUMENT_VARIABLE = "NARRATOR_HIGGS_VOICES"

MLX_MODEL_VARIABLE = "NARRATOR_HIGGS3_MLX_MODEL"

DOCUMENT_NAME = "narrator-higgs-voices.json"

_KIND_ON_THE_WIRE: dict[str, str] = {
    "checkpoint": "checkpoint",
    "zeroshot": "clips",
    "token": "default",
}

_SAMPLING_ON_THE_WIRE: dict[str, str] = {
    "temperature": "temperature",
    "top_p": "topP",
    "top_k": "topK",
}


class NarratorVoicesError(EngineError):
    ...


def document_path(home: Path) -> Path:
    return home / DOCUMENT_NAME


@dataclass(frozen=True)
class VoicesDocument:
    path: Path
    voices: dict[str, dict[str, Any]]
    base_weights: Path | None

    def environment(self) -> dict[str, str]:
        environment = {DOCUMENT_VARIABLE: str(self.path)}
        if self.base_weights is not None:
            environment[MLX_MODEL_VARIABLE] = str(self.base_weights)
        return environment

    def entry(self, voice: str) -> dict[str, Any]:
        found = self.voices.get(voice)
        if found is None:
            raise NarratorVoicesError(
                f"{self.path} carries no voice {voice!r}; it carries "
                f"{sorted(self.voices)}. narrator resolves a Higgs v3 voice by "
                f"name in the {DOCUMENT_VARIABLE} document and would refuse this "
                "load the same way"
            )
        return found

    def weights_for(self, voice: str) -> Path:
        entry = self.entry(voice)
        checkpoint = entry.get("checkpointDir")
        if checkpoint is not None:
            return Path(checkpoint)
        if self.base_weights is not None:
            return self.base_weights
        raise NarratorVoicesError(
            f"{self.path} names no weights for {voice!r}: it has no "
            f"checkpointDir and the document sets no {MLX_MODEL_VARIABLE}"
        )


def _sampling_entry(manifest: VoiceManifest, spec: VoiceBackendSpec) -> dict[str, Any]:
    entry: dict[str, Any] = {}
    for key, value in spec.sampling.items():
        wire = _SAMPLING_ON_THE_WIRE.get(key)
        if wire is None:
            raise NarratorVoicesError(
                f"{manifest.path.name} [voice.backends.{spec.backend}] sampling "
                f"names {key!r}, and narrator's voice document has no such lever; "
                f"it takes {sorted(_SAMPLING_ON_THE_WIRE)} (as "
                f"{sorted(_SAMPLING_ON_THE_WIRE.values())} on the wire)"
            )
        entry[wire] = int(value) if wire == "topK" else float(value)
    return entry


def take_sampling(manifest: VoiceManifest, take: int) -> dict[str, Any] | None:
    rung = manifest.take(take)
    if not rung.overrides:
        return None
    entry: dict[str, Any] = {}
    for key, value in rung.overrides.items():
        wire = _SAMPLING_ON_THE_WIRE.get(key)
        if wire is None:
            raise NarratorVoicesError(
                f"{manifest.path.name} [[voice.takes]][{take}] names {key!r}, and "
                f"narrator's per-item sampling has no such lever; it takes "
                f"{sorted(_SAMPLING_ON_THE_WIRE)}"
            )
        entry[wire] = int(value) if wire == "topK" else float(value)
    return entry


def voice_entry(
    manifest: VoiceManifest,
    spec: VoiceBackendSpec,
    weights_dir: Path,
    clip: ClipEntry | None = None,
) -> dict[str, Any]:
    kind = _entry_kind(manifest, spec, weights_dir, clip)
    entry: dict[str, Any] = {"kind": kind}
    if kind in ("checkpoint", "clips"):
        entry["checkpointDir"] = str(weights_dir)
    if clip is not None:
        entry["clips"] = [clip.to_dict()]
    if spec.max_chars is not None:
        entry["maxChars"] = spec.max_chars
    pace = manifest.pace
    if pace.target_chars is not None:
        entry["targetChars"] = pace.target_chars
    if pace.safe_min_chars is not None:
        entry["safeMinChars"] = pace.safe_min_chars
    if pace.safe_max_chars is not None:
        entry["safeMaxChars"] = pace.safe_max_chars
    entry["sampling"] = _sampling_entry(manifest, spec)
    if pace.pace_chars_per_sec is not None:
        entry["paceCharsPerSec"] = pace.pace_chars_per_sec
        entry["maxCharsPerSec"] = pace.max_chars_per_sec
        entry["minCharsPerSec"] = pace.min_chars_per_sec
    return entry


def _entry_kind(
    manifest: VoiceManifest,
    spec: VoiceBackendSpec,
    weights_dir: Path,
    clip: ClipEntry | None,
) -> str:
    if manifest.narrator_engine not in DOCUMENT_READERS:
        raise NarratorVoicesError(
            f"{manifest.id} names narrator_engine {manifest.narrator_engine!r}, "
            f"which reads no {DOCUMENT_VARIABLE} document; only "
            f"{sorted(DOCUMENT_READERS)} resolve a voice by name in one"
        )
    kind = _KIND_ON_THE_WIRE.get(manifest.kind)
    if kind is None:
        raise NarratorVoicesError(
            f"voice {manifest.id!r} is a {manifest.kind} voice, and this build "
            f"writes no document entry for one; it knows "
            f"{sorted(_KIND_ON_THE_WIRE)}"
        )
    if kind == "clips" and clip is None:
        raise NarratorVoicesError(
            f"voice {manifest.id!r} is a zeroshot voice and no reference clip "
            "was placed for this load. The base weights without a reference are "
            "the model's own voice, which is a DIFFERENT voice — 12 % of the "
            "narrator ceiling — and rendering a book in it under this id would "
            "be reported as success"
        )
    if kind != "clips" and clip is not None:
        raise NarratorVoicesError(
            f"voice {manifest.id!r} is a {manifest.kind} voice and a reference "
            f"clip was placed for it ({clip.path}). A checkpoint's voice is in "
            "its weights and a token voice's is in the engine; narrator would "
            "clone from the clip and ignore the weights this load names"
        )
    if kind == "default" and spec.backend == CUDA_LINUX:
        meant = (
            f"pulled at {spec.hf_repo}@{spec.revision[:12]}"
            if spec.source == PINNED
            else f"named by this voice's {spec.backend} block"
        )
        raise NarratorVoicesError(
            f"voice {manifest.id!r} is a token voice, and narrator's served arm "
            f"on {CUDA_LINUX} serves a default voice from the HuggingFace cache "
            "rather than from a directory Crucible names: its launcher reads "
            "HIGGS_MODEL_DIR for a checkpoint voice only and otherwise picks "
            "whatever base snapshot the cache holds. That is not the directory "
            f"{meant} ({weights_dir}), and "
            "a server started on other bytes would render under this voice's "
            "fingerprint. RULING OWED on narrator's side; until then a token "
            f"voice loads on {MLX_DARWIN} only"
        )
    return kind


def write_document(
    home: Path,
    manifest: VoiceManifest,
    spec: VoiceBackendSpec,
    weights_dir: Path,
    reference: VoiceReference | None = None,
) -> VoicesDocument:
    clip = None if reference is None else place(home, reference)
    entry = voice_entry(manifest, spec, weights_dir, clip)
    path = document_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {manifest.id: entry}
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    base_weights = (
        weights_dir if entry["kind"] == "default" and spec.backend == MLX_DARWIN
        else None
    )
    return VoicesDocument(path=path, voices=document, base_weights=base_weights)


__all__ = [
    "DOCUMENT_NAME",
    "DOCUMENT_VARIABLE",
    "MLX_MODEL_VARIABLE",
    "NarratorVoicesError",
    "VoicesDocument",
    "document_path",
    "take_sampling",
    "voice_entry",
    "write_document",
]
