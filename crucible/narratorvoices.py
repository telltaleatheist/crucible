"""The voices document narrator reads — written by Crucible, from its manifests.

PHASE3-TTS.md section 4. A Higgs v3 voice is not a directory handed over on the
`load` message. narrator refuses `modelDir` BY NAME on both arms
(`HiggsV3Engine.resolve_load_voice`: "the served model is the launch script's
argument, not a per-load field"; `HiggsV3MlxEngine.resolve_load_voice`: "which
weights the MLX backend loads comes from the voice document ... not from a
per-load field") and then looks the voice NAME up in a JSON document whose path
is `NARRATOR_HIGGS_VOICES`. That document is per-engine tuning, not a list of
files: it carries the merged directory the server is started on, the cap the
packer certified, the safe band, the pace triple the length guard is seeded
from and the sampling the engine renders at — every one of them a fact
BookForge measured and wrote into `electron/data/higgs-models.json`, and
BookForge's `electron/higgs-models.ts:higgsVoicesDocument` translates that
catalog into this document per spawn.

Under Crucible the ONE owner of every one of those facts is the voice manifest
(`voices/<id>.toml`, `crucible/voices.py`), and the weights live where
`crucible voices pull` put them, `~/.crucible/voices/<id>/<backend>/`. So this
module is `higgsVoicesDocument` for a server that has never heard of BookForge:
it writes the document narrator reads FROM the manifest and the pulled path,
one file per server under `~/.crucible/`, regenerated at every `load`. Not at
engine start, and not once at install: a Higgs v3 voice change IS a worker
restart, so a load is the one moment the document has to be true, and writing
it there means it can never name a voice whose stamp has since moved.

Crucible's first real render found both halves of this the hard way
(2026-09-14, the launcher agent on `cuda-linux`, then the keeper on the Mac):
the `load` carried `modelDir`, which narrator refused by name, and had it not,
narrator would have looked the voice up in a document Crucible never wrote.

WHAT THE ENTRY CARRIES, and why exactly that. Every key is one narrator's
`engine/higgs/config.py:load_voices` reads, spelled as it reads it (the
catalog's camelCase is the WIRE), and nothing it does not read is written:

    kind             `checkpoint` for a manifest `kind = "checkpoint"`;
                     `default` for `kind = "token"` — narrator's name for the
                     model's own voice, which Crucible calls a token voice
                     (PHASE3-TTS.md section 2's third kind).
    checkpointDir    the pulled directory, checkpoint voices only. narrator
                     checks the directory's required files itself at the load
                     message (`v3_served.checkpoint_serve_target`), before any
                     server starts.
    maxChars         the backend block's `max_chars` — CHARACTERS, and the one
                     field narrator refuses a checkpoint voice without.
    targetChars      `[voice.pace].target_chars`, when declared.
    safeMinChars /   `[voice.pace].safe_min_chars` / `safe_max_chars`, when
    safeMaxChars     declared. `crucible/voices.py` has already refused a band
                     above the cap or a floor at the ceiling, which are the
                     same two refusals narrator's `_safe_band` makes.
    paceCharsPerSec  `[voice.pace]`'s three rates, which narrator's
    maxCharsPerSec   `_length_band` takes as a triple or not at all — and a
    minCharsPerSec   manifest cannot load without all three.
    sampling         the backend block's `sampling`, with narrator's key names
                     (`topP`, `topK`). This is the channel PHASE3-TTS.md
                     section 4 said sampling did not have: `register_voice_caps`
                     speaks narrator's older engine's vocabulary, but the
                     DOCUMENT's `sampling` is
                     read onto the voice by `load_voices` and applied as the
                     engine's override on both arms (`v3_engine
                     .higgs_v3_config_from_worker_kwargs`, `mlx_backend
                     .higgs_v3_mlx_config_from_worker_kwargs`). Every manifest
                     in this build states the boson default, so what the
                     document asks for is what the engine would have rendered
                     at anyway — except on a checkpoint whose own
                     `generation_config.json` says otherwise, where writing it
                     is what makes take 0 the boson default rather than
                     whatever the merge script left in the file.

NOT written, and each one deliberately: `maxCharsSource` (narrator's default
for an absent key is `catalog`, which is what a manifest number is; the
manifest has no such field and inventing `placeholder` for a token voice would
be Crucible labelling a measurement it did not take); `scene` (v2 only);
`allowedControls` / `maxReferenceSeconds` (narrator's engine defaults, which
no manifest overrides); `clips` (a zeroshot voice is refused below, and
PHASE3-TTS.md section 6 says what narrator owes before that changes);
`_overrideNote` (BookForge's own post-mortem marker for a directory that is
not the catalog's, and under Crucible the directory is always the pin's).

TWO KINDS ARE REFUSED HERE, by name, before any engine starts:

- `zeroshot`. The document CAN carry `clips`, and BookForge writes them, but a
  Crucible zeroshot voice's clips are either `from-request` (they arrive with a
  job, not at load) or files in a pulled refs repo that nothing in this build
  has laid out for narrator yet. Writing a clips entry that names files
  Crucible has not checked is a load that dies inside narrator after the
  refusal could have been made here. The render door already refuses the kind
  as `voice_kind_unsupported`; this is the same refusal at the load door.
- `token` on `cuda-linux`. narrator's served arm exports `HIGGS_MODEL_DIR`
  only for a checkpoint voice and UNSETS it otherwise, and its launch script
  then serves "the base snapshot out of the HF cache" (`serve_higgs_v3.sh`:
  `~/.cache/huggingface/hub/models--bosonai--higgs-audio-v3-tts-4b/snapshots/*`).
  That is not the directory Crucible pulled at the manifest's pin, and a
  server that came up on whatever snapshot the cache happened to hold would
  render under a fingerprint that names bytes it never read. RULING OWED
  (narrator's side): a way for the served arm to be told the base directory
  for a `default` voice — until then `higgs-default` loads on `mlx-darwin`
  only, where `NARRATOR_HIGGS3_MLX_MODEL` names the base weights and this
  module sets it to the pulled directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backend import CUDA_LINUX, MLX_DARWIN
from .engines.base import EngineError
from .voices import VoiceBackendSpec, VoiceManifest

#: The variable narrator reads the document's PATH from
#: (`engine/higgs/config.py:VOICES_ENV`). A path and not a value: a transcript
#: is prose with quotes and newlines in it, and BookForge's writer says why a
#: JSON blob in an exported shell variable is one quoting bug from the wrong
#: voice.
DOCUMENT_VARIABLE = "NARRATOR_HIGGS_VOICES"

#: The MLX arm's BASE weights (`mlx_backend.MODEL_ENV`), read only for a voice
#: with no `checkpointDir` — a `default` or `clips` voice — and refused by name
#: when unset ("there is no default and no search"). A checkpoint voice never
#: reads it: `model_dir = checkpoint or model_dir_from_env()`.
MLX_MODEL_VARIABLE = "NARRATOR_HIGGS3_MLX_MODEL"

#: One file per server, under `~/.crucible/`, overwritten at every load.
DOCUMENT_NAME = "narrator-higgs-voices.json"

#: The narrator engines that resolve a voice through the document. A set of
#: one, written as a set so the rule reads as a rule: an engine outside it
#: takes its weights on the `load` message (`modelDir` / `adapterDir` /
#: `baseDir`) and reads no `NARRATOR_HIGGS_*` variable at all. Since Owen's
#: ruling of 2026-09-14 (`voices.NARRATOR_ENGINE_SAMPLING`) `higgs-v3` is the
#: only engine Crucible names, so this set and that table happen to agree —
#: the next engine is what separates them again.
DOCUMENT_READERS: frozenset[str] = frozenset({"higgs-v3"})

#: Crucible's voice kinds -> narrator's `kind` values. `zeroshot` is absent on
#: purpose (refused in `voice_entry`, see the module docstring).
_KIND_ON_THE_WIRE: dict[str, str] = {"checkpoint": "checkpoint", "token": "default"}

#: The manifest's sampling keys -> the document's. The inverse of narrator's
#: `config._SAMPLING_KEYS`, minus `repetitionPenalty`, which no Higgs manifest
#: may state (`NARRATOR_ENGINE_SAMPLING["higgs-v3"]` has three keys and
#: `_check_sampling` replaces the table wholesale). A key outside this map is
#: refused rather than passed through: narrator would refuse it too, but after
#: the process was up.
_SAMPLING_ON_THE_WIRE: dict[str, str] = {
    "temperature": "temperature",
    "top_p": "topP",
    "top_k": "topK",
}


class NarratorVoicesError(EngineError):
    """The document cannot be written truthfully for this voice on this arm.

    An `EngineError` because that is what it is to the caller: narrator cannot
    be started for this voice, and the refusal arrives before the process does.
    """


def document_path(home: Path) -> Path:
    return home / DOCUMENT_NAME


@dataclass(frozen=True)
class VoicesDocument:
    """What was written, where, and what narrator's environment must say."""

    path: Path
    voices: dict[str, dict[str, Any]]
    #: The pulled directory `NARRATOR_HIGGS3_MLX_MODEL` names, or None where
    #: nothing reads it (a checkpoint voice on either arm; anything on
    #: `cuda-linux`).
    base_weights: Path | None

    def environment(self) -> dict[str, str]:
        environment = {DOCUMENT_VARIABLE: str(self.path)}
        if self.base_weights is not None:
            environment[MLX_MODEL_VARIABLE] = str(self.base_weights)
        return environment

    def entry(self, voice: str) -> dict[str, Any]:
        """The entry for `voice`, or a refusal naming the voices there are.

        The same refusal narrator makes (`config.load_voice`: "a load for a
        voice the file does not carry FAILS, naming the file and the voices it
        does carry"), made here, before the message is sent.
        """
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
        """The directory narrator will load `voice`'s weights from.

        A checkpoint's `checkpointDir`, or the base weights for a voice with
        none — which on the MLX arm is `NARRATOR_HIGGS3_MLX_MODEL`. Both are
        this document's, so a caller holding a directory of its own can check
        the two agree rather than assume it.
        """
        entry = self.entry(voice)
        checkpoint = entry.get("checkpointDir")
        if checkpoint is not None:
            return Path(checkpoint)
        if self.base_weights is not None:
            return self.base_weights
        # Unreachable through `write_document`: a voice with no checkpointDir
        # is a token voice, and a token voice is refused on the one arm that
        # names no base directory.
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
        # narrator's `_voice_sampling` requires `topK` to be a whole number and
        # refuses `50.0` by name; the manifest loader has already made every
        # value a float (`_number`), so the count goes back to being one here.
        entry[wire] = int(value) if wire == "topK" else float(value)
    return entry


def voice_entry(
    manifest: VoiceManifest, spec: VoiceBackendSpec, weights_dir: Path
) -> dict[str, Any]:
    """One document entry, from one manifest and the directory its weights are in.

    Pure: nothing is read from disk and nothing is written. `write_document`
    is the side effect, and this is what a test asserts field by field.
    """
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
            "writes no document entry for one: its reference clips are either "
            "carried by a request or in a refs repo nothing has laid out for "
            "narrator, and an entry naming files Crucible has not checked is a "
            "load that dies inside narrator instead of here (PHASE3-TTS.md "
            "section 6)"
        )
    if kind == "default" and spec.backend == CUDA_LINUX:
        raise NarratorVoicesError(
            f"voice {manifest.id!r} is a token voice, and narrator's served arm "
            f"on {CUDA_LINUX} serves a default voice from the HuggingFace cache "
            "rather than from a directory Crucible names: its launcher reads "
            "HIGGS_MODEL_DIR for a checkpoint voice only and otherwise picks "
            "whatever base snapshot the cache holds. That is not the directory "
            f"pulled at {spec.hf_repo}@{spec.revision[:12]} ({weights_dir}), and "
            "a server started on other bytes would render under this voice's "
            "fingerprint. RULING OWED on narrator's side; until then a token "
            f"voice loads on {MLX_DARWIN} only"
        )

    entry: dict[str, Any] = {"kind": kind}
    if kind == "checkpoint":
        entry["checkpointDir"] = str(weights_dir)
    entry["maxChars"] = spec.max_chars
    pace = manifest.pace
    if pace.target_chars is not None:
        entry["targetChars"] = pace.target_chars
    if pace.safe_min_chars is not None:
        entry["safeMinChars"] = pace.safe_min_chars
    if pace.safe_max_chars is not None:
        entry["safeMaxChars"] = pace.safe_max_chars
    entry["sampling"] = _sampling_entry(manifest, spec)
    entry["paceCharsPerSec"] = pace.pace_chars_per_sec
    entry["maxCharsPerSec"] = pace.max_chars_per_sec
    entry["minCharsPerSec"] = pace.min_chars_per_sec
    return entry


def write_document(
    home: Path, manifest: VoiceManifest, spec: VoiceBackendSpec, weights_dir: Path
) -> VoicesDocument:
    """Write the document for THIS load — one voice, the one being loaded.

    One entry and not every installed voice, because narrator serves one voice
    per process (a Higgs v3 voice change is a worker restart) and a document
    listing others would be a list of claims about stamps nobody re-checked at
    this load. Overwritten in place: the previous load's document is exactly
    the stale thing a post-mortem must not find.
    """
    entry = voice_entry(manifest, spec, weights_dir)
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
    "DOCUMENT_READERS",
    "DOCUMENT_VARIABLE",
    "MLX_MODEL_VARIABLE",
    "NarratorVoicesError",
    "VoicesDocument",
    "document_path",
    "voice_entry",
    "write_document",
]
