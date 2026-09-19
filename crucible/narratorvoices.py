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
    checkpointDir    the pulled directory — a checkpoint voice's merged
                     weights, or a zeroshot voice's BASE weights. narrator
                     checks the directory's required files itself at the load
                     message (`v3_served.checkpoint_serve_target`), before any
                     server starts.
    clips            the reference a `load-voice` carried, as narrator's own
                     `[{path, transcript, seconds}]` — zeroshot voices only,
                     and required of them. See `crucible/voicereference.py`.
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
no manifest overrides); `_overrideNote` (BookForge's own post-mortem marker for
a directory that is not the catalog's, and under Crucible the directory is
always the pin's).

**`zeroshot` STOPPED BEING REFUSED HERE on 2026-09-14.** It was, on the
grounds that "a Crucible zeroshot voice's clips are either `from-request` or
files in a refs repo nothing has laid out, and an entry naming files Crucible
has not checked is a load that dies inside narrator". The load door now carries
the clip (`crucible/voicereference.py`, PHASE3-TTS.md section 5), Crucible
writes the file itself, and the entry names a path this process just wrote — so
the reason is gone and so is the refusal. What replaced it is the pair of
refusals above `voice_entry`: a zeroshot voice with no clip, and a clip on a
voice that is not one.

ONE KIND IS STILL REFUSED HERE, by name, before any engine starts:

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

  **A lead on that ruling, found while wiring zeroshot (2026-09-14) and NOT
  acted on.** narrator's document reader passes `checkpointDir` into
  `DefaultVoice` as well as into `ClipsVoice`, and the served arm exports
  whatever `checkpoint_dir` the config ends up with — so writing the pulled
  base directory as a `default` voice's `checkpointDir` would very likely make
  the served arm start on the bytes Crucible pinned, which is the whole of
  what the refusal above is waiting for. It is the same move this module now
  makes for `clips`. It is NOT made for `token` here, because the two cases
  differ in what has been tested and in whose decision it is: a zeroshot load
  is a new door being built to a written plan, and re-pointing `higgs-default`
  is a behaviour change to a shipped smoke voice on an arm nobody has run it
  on. Owen's ruling, with this lead in front of him.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backend import CUDA_LINUX, MLX_DARWIN
from .engines.base import EngineError
from .voicereference import ClipEntry, VoiceReference, place
from .voices import VoiceBackendSpec, VoiceManifest
from .weights import PINNED

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

#: Crucible's voice kinds -> narrator's `kind` values. `zeroshot` became
#: `clips` on 2026-09-14, when the load door grew a channel for the clip — see
#: the module docstring.
_KIND_ON_THE_WIRE: dict[str, str] = {
    "checkpoint": "checkpoint",
    "zeroshot": "clips",
    "token": "default",
}

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


def take_sampling(manifest: VoiceManifest, take: int) -> dict[str, Any] | None:
    """The rung's sampling in narrator's PER-ITEM spelling, or None for take 0.

    The document's `sampling` (above) is the voice's take-0 numbers and is
    written per LOAD. A rung is per RENDER, and since 2026-09-14 narrator has a
    channel for it: an item of `generate` / `generate_batch` may carry
    `sampling: {temperature?, topP?, topK?, repetitionPenalty?}`, which the
    engine lays OVER its resolved sampling key by key
    (`narrator/engine/item_sampling.py`). So this returns **only the keys the
    rung declares** — `[[voice.takes]]` take 1 is one line, `temperature =
    0.7`, and it means "take 0, but cooler". Sending the other two back at
    take 0's values would say the same thing, but it would also be Crucible
    restating numbers it was not asked about, and the first partial rung that
    meant something else would be applied wrong.

    **None for take 0, and None is not an empty object.** An item with no
    `sampling` key renders at the voice's loaded default, which IS take 0;
    sending `{}` would be Crucible asking for a rung with nothing in it, which
    narrator refuses as `sampling_malformed` — correctly.

    The translation is `_SAMPLING_ON_THE_WIRE`, the same map the document uses,
    because the per-item channel deliberately took the document's spelling: two
    names for one lever is the shape `docs/ARCHITECTURE.md`'s audit found seven
    times.
    """
    rung = manifest.take(take)
    if not rung.overrides:
        return None
    entry: dict[str, Any] = {}
    for key, value in rung.overrides.items():
        wire = _SAMPLING_ON_THE_WIRE.get(key)
        if wire is None:  # pragma: no cover — `_check_takes` allows no other key
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
    """One document entry, from one manifest and the directory its weights are in.

    Pure: nothing is read from disk and nothing is written. `write_document`
    is the side effect, and this is what a test asserts field by field.

    `clip` is the reference the `load-voice` carried, already written to disk
    by `voicereference.place` — required of a `zeroshot` voice and refused on
    any other kind, the same pair of rules the load door states as
    `reference_required` / `reference_not_allowed`. Stated twice on purpose:
    the door refuses before a job is queued, and this refuses before a process
    is started, and the second is what a caller reaching `write_document`
    another way still gets.
    """
    if manifest.narrator_engine not in DOCUMENT_READERS:
        raise NarratorVoicesError(
            f"{manifest.id} names narrator_engine {manifest.narrator_engine!r}, "
            f"which reads no {DOCUMENT_VARIABLE} document; only "
            f"{sorted(DOCUMENT_READERS)} resolve a voice by name in one"
        )
    kind = _KIND_ON_THE_WIRE.get(manifest.kind)
    if kind is None:  # pragma: no cover — every kind in VOICE_KINDS is mapped
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
        # WHICH BYTES CRUCIBLE MEANT, named in whichever shape the block has.
        # `spec.revision[:12]` subscripted None the moment a token voice
        # declared a `path` instead of a pin (PHASE18-UNCERTIFIED.md section
        # 3), so the voice was still refused — by a TypeError instead of by
        # this sentence.
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

    entry: dict[str, Any] = {"kind": kind}
    if kind in ("checkpoint", "clips"):
        # `checkpointDir` FOR A CLIPS VOICE TOO, and it is not a stretch of the
        # key: both arms load THAT directory, whichever kind named it. For a
        # zero-shot voice the directory is the BASE weights Crucible pulled at
        # the manifest's pin, which is exactly the thing the
        # `token`-on-cuda-linux refusal below exists because it could not name.
        # Written rather than omitted for that reason: omit it and the served
        # arm serves "the base snapshot out of the HF cache", which is a
        # fingerprint naming bytes nobody read.
        #
        # THE `kind` ABOVE IS WHAT SAYS WHICH, and the two travel as one
        # statement (2026-09-15). narrator reads a `clips` voice's directory
        # into `ClipsVoice.base_dir` and a `checkpoint` voice's into
        # `checkpoint_dir`, and asks only the second for a
        # `generation_config.json` — the file a MERGE carries and the published
        # base does not (`bosonai/higgs-tts-3-4b` at 239f63fb: thirteen files,
        # none of them it). Before narrator drew that line the first zero-shot
        # load Crucible ever made was refused for that missing file, with a
        # complete pull on disk. Writing `checkpoint` here would bring the
        # refusal straight back, which is why the test asserts the pair.
        entry["checkpointDir"] = str(weights_dir)
    if clip is not None:
        # narrator's own clip row, verbatim. One clip and not a list of one by
        # accident: vllm-omni takes EXACTLY ONE reference ("multi-shot voice
        # clone is not supported"), so several clips are pre-joined into one
        # wav by whoever cut them, and this wire carries the one.
        entry["clips"] = [clip.to_dict()]
    entry["maxChars"] = spec.max_chars
    pace = manifest.pace
    if pace.target_chars is not None:
        entry["targetChars"] = pace.target_chars
    if pace.safe_min_chars is not None:
        entry["safeMinChars"] = pace.safe_min_chars
    if pace.safe_max_chars is not None:
        entry["safeMaxChars"] = pace.safe_max_chars
    entry["sampling"] = _sampling_entry(manifest, spec)
    # THE RATE BAND ONLY WHEN THE MANIFEST MEASURED ONE, and all three keys
    # together: narrator's `_length_band` (`engine/higgs/config.py`) takes all
    # three or none and refuses a subset by name. Testing one of the three is
    # enough because `_check_pace` refuses a partial triple at the manifest.
    #
    # WHAT ABSENCE BUYS. A voice nobody ran a ladder on has no pace, and the
    # three numbers `higgs-default` and `zeroshot` used to send were narrator's
    # own Higgs v3 defaults read back to it — a pace of 15.0 that is the frame
    # cap's DIVISOR rather than a narration rate, between edges written around
    # a book pace nearer 17.2. narrator keeps a band's RATIOS, so that triple
    # gave it 1.034 tolerance on the long side and it re-rolled healthy chunks
    # to MAX_DEPTH. Sending nothing puts it on the path it already has for an
    # unmeasured voice: its engine's default band, centred on the geometric
    # mean of the edges (`truncation.tracker_for`). Crucible does not derive a
    # centre of its own — that fact has one owner and it is narrator.
    if pace.pace_chars_per_sec is not None:
        entry["paceCharsPerSec"] = pace.pace_chars_per_sec
        entry["maxCharsPerSec"] = pace.max_chars_per_sec
        entry["minCharsPerSec"] = pace.min_chars_per_sec
    return entry


def write_document(
    home: Path,
    manifest: VoiceManifest,
    spec: VoiceBackendSpec,
    weights_dir: Path,
    reference: VoiceReference | None = None,
) -> VoicesDocument:
    """Write the document for THIS load — one voice, the one being loaded.

    One entry and not every installed voice, because narrator serves one voice
    per process (a Higgs v3 voice change is a worker restart) and a document
    listing others would be a list of claims about stamps nobody re-checked at
    this load. Overwritten in place: the previous load's document is exactly
    the stale thing a post-mortem must not find.

    A `reference` is written to disk here, beside the document and in the same
    breath, because narrator calls `os.path.isfile` on every clip path the
    document names: two files, one moment, or the load fails inside the engine
    for a reason that was knowable here.
    """
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
    "DOCUMENT_READERS",
    "DOCUMENT_VARIABLE",
    "MLX_MODEL_VARIABLE",
    "NarratorVoicesError",
    "VoicesDocument",
    "document_path",
    "take_sampling",
    "voice_entry",
    "write_document",
]
