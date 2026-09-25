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
*History: the id half of this section was replaced on 2026-09-24 — see
"Three models, one id each" below. The engine half still stands.*

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
own SEVEN MODEL IDS, all prefixed `mlx-whisper-` (until 2026-09-24).

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

A third and fourth engine, and the first id on BOTH backends (2026-09-24)
------------------------------------------------------------------------
Owen, 2026-09-24: *"we're fully switching over to qwen for transcribing and
aligning. it seems flawless. 1.7b - the biggest one"*, then *"the 41 gb memory
use wont be a problem since we'll be using vllm or sglang instead"*, and full
precision on both machines. So `Qwen/Qwen3-ASR-1.7B` arrives as two more
engines, one per backend, under the one `asr` job type
(docs/PHASE25-QWEN-ASR.md):

- `vllm` on cuda-linux: vLLM 0.29.0, the llm env's own pin, which registers
  `Qwen3ASRForConditionalGeneration` natively
  (`vllm/model_executor/models/registry.py` L565 at tag v0.29.0).
- `mlx-audio` on mlx-darwin: mlx-audio 0.5.5, already pinned in the Mac's llm
  env as mlx-vlm's dependency, whose `stt/models/qwen3_asr` loads the OFFICIAL
  checkpoint and converts it on load (`Qwen3ASRModel.sanitize`: strips
  `thinker.`, drops the tied `lm_head.weight`, transposes the conv kernels).

**This is the first asr id with a block on both backends, and that is allowed
for exactly one reason: both blocks pin the SAME repo at the SAME revision.**
The whisper rule above — different weights must never share an id — was not
relaxed THAT morning; it was satisfied a second way. (It was replaced that
afternoon: the section after this one.) mlx-audio reads the very safetensors
vLLM reads, in bfloat16, so `qwen3-asr-1.7b` names one set of bytes whichever
machine ran it. The loader enforces it: a manifest whose blocks pin different
repos or revisions is refused by name, so a community conversion cannot slip
in under the official id.

The Qwen engines carry more keys than whisper's (`QWEN_BACKEND_REQUIRED`,
`VLLM_BACKEND_REQUIRED`), because what whisper decides inside its library —
batch, output ceiling, KV pool — is a per-backend decision here, and the spec
ContentStudio measured says a library default is what crashed a Mac (batch 32
on MPS, 2026-09-24). A key the job needs is a key the manifest states; nothing
reaches a worker as a library default.

Three models, one id each, across both backends (2026-09-24)
-----------------------------------------------------------
Owen, 2026-09-24, the same day: the asr job offers EXACTLY three models and
the caller picks — Whisper large-v3-turbo, Qwen3-ASR-1.7B and Whisper tiny.
Every other whisper size (base, small, medium, large-v3, distil-large-v3) was
removed, on both engines. And the whispers that stayed became ONE id each
(`whisper-large-v3-turbo`, `whisper-tiny`) with a block per backend, because a
client should not have to know which machine it is talking to before it can
name a transcriber — Qwen already had that shape, and so does every
`models/<id>.toml`.

That REPLACES the rule the two sections above were written to defend, and it
is replaced rather than bent, so here is what took its place:

- **The family, not the id prefix, is what an engine is bound to.** A block's
  engine must belong to the manifest's `[model] family` (`ASR_ENGINE_FAMILY`),
  and the id must begin with that family. So `whisper-*` is whisper on both
  machines and `qwen3-asr-*` is Qwen3-ASR on both, and no id can be whisper on
  one backend and Qwen on the other.
- **"One id, one set of bytes" still holds where it can hold.** Qwen's two
  engines read the same official checkpoint, so a `qwen3-asr` manifest whose
  blocks pin different weights is still refused (`ONE_CHECKPOINT_FAMILIES`).
  Whisper's two engines CANNOT read the same bytes — CTranslate2 and MLX are
  different formats — so a whisper id is two conversions, exactly as
  `qwen3.5-9b` is a GGUF on one machine and an MLX conversion on the other.
- **What produced a transcript is in its RECORD, not its name.** The
  provenance sidecar already named the backend; it now also names the engine,
  the repo and the revision (`jobs/asr/__init__.py`, `model_provenance`), so
  two transcripts made on two machines are told apart the way two llm renders
  always were.
- **The old ids are GONE, not aliases.** A request naming one is refused
  `unknown_model`, naming the three that exist and — for the four that were
  renamed — the id that replaced it (`RENAMED_ASR_IDS`, `RETIRED_ASR_IDS`).
  Weights already pulled under a renamed id are MOVED to the new id's folder
  once, at server start, and only when their stamp matches the new block's pin
  exactly (`jobs/asr`'s `adopt_renamed_asr_weights`); anything left under an
  id nothing declares is reported by `crucible doctor` with its size and path
  (`catalog.stranded_weights`), never deleted behind the operator's back.

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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .backend import CUDA_LINUX, MLX_DARWIN
from .errors import CrucibleError

ASR_DIR_ENV = "CRUCIBLE_ASR_DIR"

#: The Qwen3-ASR engines, one per backend. Named once because three modules ask
#: "is this a Qwen engine": the job type dispatches on it, this loader requires
#: the Qwen keys on it, and the worker table is keyed by it.
VLLM_ENGINE = "vllm"
MLX_AUDIO_ENGINE = "mlx-audio"
#: Qwen's own `qwen_asr` package on torch, which the Mac runs on MPS. Added
#: 2026-09-24 after the first live runs: on identical pieces it hears a few more
#: fillers than the MLX port and is 2.5x slower (asr/qwen3-asr-1.7b.toml, its
#: mlx-darwin block). Owen kept both, the port under its own id.
QWEN_ASR_TORCH_ENGINE = "qwen-asr"
QWEN_ASR_ENGINES: frozenset[str] = frozenset(
    {VLLM_ENGINE, MLX_AUDIO_ENGINE, QWEN_ASR_TORCH_ENGINE}
)

#: Which engines each backend is allowed to name. Whisper is one engine per
#: backend because CTranslate2 has no Metal backend (the module docstring).
#: vLLM has no Metal backend and mlx-audio has no CUDA one. The Mac has TWO
#: Qwen engines by Owen's choice of 2026-09-24: the official package (fidelity)
#: under `qwen3-asr-1.7b` and the MLX port (speed) under `qwen3-asr-1.7b-mlx`.
ASR_BACKEND_ENGINES: dict[str, frozenset[str]] = {
    CUDA_LINUX: frozenset({"faster-whisper", VLLM_ENGINE}),
    MLX_DARWIN: frozenset({"mlx-whisper", MLX_AUDIO_ENGINE, QWEN_ASR_TORCH_ENGINE}),
}

#: The model family each engine runs. A block's engine must belong to its
#: manifest's `[model] family`, and the id must begin `<family>-`, so one id is
#: one MODEL on every backend: whisper on both machines or Qwen3-ASR on both,
#: never one on the PC and the other on the Mac.
#:
#: Until 2026-09-24 this was a table of id PREFIXES per engine
#: (`faster-whisper-`, `mlx-whisper-`), which made every whisper id
#: backend-specific. Owen's ruling of that day made the whispers one id each
#: across both backends (the module docstring); the per-engine prefix went with
#: it, and the family is what is bound now.
ASR_ENGINE_FAMILY: dict[str, str] = {
    "faster-whisper": "whisper",
    "mlx-whisper": "whisper",
    VLLM_ENGINE: "qwen3-asr",
    MLX_AUDIO_ENGINE: "qwen3-asr",
    QWEN_ASR_TORCH_ENGINE: "qwen3-asr",
}

#: The families whose engines read ONE checkpoint on every backend, and whose
#: manifests are therefore refused if their blocks pin different weights.
#: Qwen3-ASR: vLLM and mlx-audio both load the official safetensors, and Owen's
#: full-precision ruling says they must (a community conversion must not slip in
#: under the official id). NOT whisper: CTranslate2 and MLX cannot read each
#: other's format, so a whisper id is necessarily two conversions, and its
#: transcripts are told apart by their provenance sidecar (backend, engine,
#: repo, revision) — the way every `models/<id>.toml` has always worked.
ONE_CHECKPOINT_FAMILIES: frozenset[str] = frozenset({"qwen3-asr"})

#: THE LINEUP, Owen 2026-09-24: these three, and the caller picks, plus the
#: Mac-only fast port of the first (`qwen3-asr-1.7b-mlx`, added that evening:
#: *"if i want speed i can get it via mlx"*). Not read by the loader (the
#: directory is the lineup); stated so a test can hold the directory to it and a
#: reader can see the ruling in one line.
ASR_LINEUP: frozenset[str] = frozenset(
    {
        "qwen3-asr-1.7b", "qwen3-asr-1.7b-mlx",
        # The 0.6B and its port (Owen, 2026-09-24: "lets add the smaller qwen
        # asr model too. not just the 1.7b").
        "qwen3-asr-0.6b", "qwen3-asr-0.6b-mlx",
        "whisper-large-v3-turbo", "whisper-tiny",
    }
)

#: Ids that were RENAMED on 2026-09-24, old -> new. NOT aliases: a request
#: naming an old id is refused `unknown_model` exactly like any other unknown
#: id, and this table only lets the refusal say what replaced it. It also tells
#: `jobs/asr`'s `adopt_renamed_asr_weights` which folders under `~/.crucible/models/` hold
#: bytes the new id pins — the same repo at the same revision, moved rather
#: than downloaded again.
RENAMED_ASR_IDS: dict[str, str] = {
    "faster-whisper-large-v3-turbo": "whisper-large-v3-turbo",
    "mlx-whisper-large-v3-turbo": "whisper-large-v3-turbo",
    "faster-whisper-tiny": "whisper-tiny",
    "mlx-whisper-tiny": "whisper-tiny",
}

#: Ids REMOVED on 2026-09-24 with nothing in their place (Owen: "every other
#: whisper size is removed"). Listed so a refusal can say "retired" rather than
#: leave a client wondering whether it misspelled one.
RETIRED_ASR_IDS: frozenset[str] = frozenset(
    f"{engine}-{size}"
    for engine in ("faster-whisper", "mlx-whisper")
    for size in ("base", "small", "medium", "large-v3", "distil-large-v3")
)


def retired_asr_id_note(model_id: str) -> str | None:
    """A sentence about an id this build removed, or None for any other id.

    For `unknown_model`'s message only. It never resolves anything: the
    request is refused either way.
    """
    renamed = RENAMED_ASR_IDS.get(model_id)
    if renamed is not None:
        return (
            f"{model_id!r} was renamed {renamed!r} on 2026-09-24, one id on "
            "every backend; the old id is not an alias"
        )
    if model_id in RETIRED_ASR_IDS:
        return (
            f"{model_id!r} was retired on 2026-09-24: the asr job offers "
            f"exactly {sorted(ASR_LINEUP)}"
        )
    return None

#: FULL PRECISION, both machines. Owen, 2026-09-24: bf16, unquantised, for
#: Qwen3-ASR-1.7B on the PC and on the Mac; "do not use the 8-bit MLX build".
#: A table of one so a manifest that says `float16` or `int8` is refused by
#: name rather than handed to a worker that would honour it.
QWEN_ASR_DTYPES: frozenset[str] = frozenset({"bfloat16"})

#: What every Qwen block states beyond whisper's four keys.
#:
#:   dtype           the precision the engine loads in (`QWEN_ASR_DTYPES`).
#:   aligner         the `align/<id>.toml` whose model stamps the word times. A
#:                   transcript with word timestamps is TWO models' output, so
#:                   it names both, and the job's memory is both.
#:   max_batch       how many pieces decode at once: vLLM's `max_num_seqs`, and
#:                   1 on mlx-audio, which is given one piece per call
#:                   (`_check_qwen_block` says why). NEVER the library default:
#:                   `qwen_asr`'s is 32, and 32 is what aborted ContentStudio's
#:                   MPS process ("too large for kernel", 2026-09-24).
#:   max_new_tokens  the most one 180-second piece may generate. 4096 is what
#:                   ContentStudio ran; `jobs/asr/loopguard.py` derives the
#:                   per-piece budget under it.
QWEN_BACKEND_REQUIRED: dict[str, type] = {
    "dtype": str,
    "aligner": str,
    "max_batch": int,
    "max_new_tokens": int,
}

#: What vLLM needs on top: the context it is started with and the KV pool it is
#: given. `kv_cache_memory_bytes` is stated so vLLM does not size its own pool
#: from its 0.92-of-the-card default (`vllm/entrypoints/llm.py` L198 at
#: v0.29.0) inside a job that shares the card with the aligner.
VLLM_BACKEND_REQUIRED: dict[str, type] = {
    "max_model_len": int,
    "kv_cache_memory_bytes": int,
}

#: The longest piece of audio one decode is given. 180 s is `qwen_asr` 0.0.6's
#: `MAX_FORCE_ALIGN_INPUT_SECONDS` (`qwen_asr/inference/utils.py` L35): with
#: timestamps on, the official SDK cuts audio into pieces no longer than this at
#: quiet points because the aligner is not trusted past it, and Crucible cuts at
#: the same length for the same reason (`jobs/asr/qwen.py`).
QWEN_PIECE_MAX_SECONDS = 180

#: Audio tokens per second of audio, COMPUTED from vLLM 0.29.0's
#: `_get_feat_extract_output_lengths` (`models/qwen3_asr.py` L175-182): every
#: whole 100 mel frames (one second, at a 160-sample hop at 16 kHz) becomes 13
#: tokens. So a 180-second piece is 2,340 audio tokens, which agrees with
#: ContentStudio's crash report: logits 151936 x 78720 is batch 32 x 2,460, i.e.
#: 2,340 audio tokens plus their prompt.
QWEN_AUDIO_TOKENS_PER_SECOND = 13

#: The longest `context` a job may send, in the model's own tokens. COMPUTED:
#: `max_model_len` less the longest piece's 2,340 audio tokens, less
#: `max_new_tokens`, less the chat scaffolding; at 8192 and 4096 that leaves
#: about 1,700, and 1,024 is the round figure under it. A context is a sentence
#: of instruction and a list of names, not a document.
QWEN_CONTEXT_MAX_TOKENS = 1024

#: The chat scaffolding around the context and the audio, in tokens, rounded UP.
#: `<|im_start|>system` ... `<|im_start|>assistant` + `language English<asr_text>`
#: is about two dozen tokens; 64 covers the longest language name with room.
QWEN_PROMPT_SCAFFOLD_TOKENS = 64


def qwen_prompt_ceiling_tokens(max_new_tokens: int) -> int:
    """The most context one piece can need. What `max_model_len` must hold."""
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
    #: The Qwen engines' keys (`QWEN_BACKEND_REQUIRED`, `VLLM_BACKEND_REQUIRED`).
    #: None on a whisper block, where the loader refuses them as unknown keys,
    #: and never None on a block whose engine requires them: `_parse` refuses
    #: the absence by name.
    dtype: str | None = None
    aligner: str | None = None
    max_batch: int | None = None
    max_new_tokens: int | None = None
    max_model_len: int | None = None
    kv_cache_memory_bytes: int | None = None

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
        row: dict[str, Any] = {
            "backend": self.backend,
            "engine": self.engine,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "memory_bytes_estimate": self.memory_bytes_estimate,
        }
        # Only the keys this engine states. A whisper row carrying six nulls
        # would say "unknown" about things that do not exist for it.
        for key in (*QWEN_BACKEND_REQUIRED, *VLLM_BACKEND_REQUIRED):
            value = getattr(self, key)
            if value is not None:
                row[key] = value
        return row

    def require(self, key: str) -> Any:
        """A Qwen key the job needs, or a refusal naming it. Never a default."""
        value = getattr(self, key)
        if value is None:
            raise AsrManifestError(
                f"the {self.backend} block has no {key!r}, and the "
                f"{self.engine!r} engine requires it"
            )
        return value


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
    #: `[model] weights_of`: the id whose download these weights ARE, or None
    #: for a model that owns its own. The `models/` rule (PHASE22-DECIDE.md
    #: section 2.9, "one copy on disk"), brought to asr on 2026-09-24 when the
    #: `-mlx` ids pulled a second copy of their official sibling's checkpoint
    #: (Owen: "go ahead"). `crucible/weights.py` reads it through `_store_id`.
    weights_of: str | None = None
    #: The base manifest, resolved by `load_asr_manifest`. None exactly where
    #: `weights_of` is.
    weights_base: "AsrManifest | None" = field(default=None, compare=False, repr=False)

    def extra_files(self, backend_kind: str) -> tuple[str, ...]:
        """What an alias owns on disk beyond its base's: nothing. Every asr block
        is a whole-repo download, so an alias shares the base's folder whole."""
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


# ------------------------------------------------------------------ locating


def asr_manifests_dir() -> Path:
    """Where `asr/*.toml` live on this host. Refuses by name if absent."""
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
    model = dict(model)
    weights_of = model.pop("weights_of", None)
    if weights_of is not None and not (isinstance(weights_of, str) and _MODEL_ID.match(weights_of)):
        raise AsrManifestError(f"{path.name}: model.weights_of {weights_of!r} is not a model id")
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
        engine = block.get("engine")
        if not isinstance(engine, str):
            # The engine decides which other keys are required, so it is read
            # first; a block without one is refused in the loader's own words.
            _check_table(where, block, _BACKEND_REQUIRED)
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
        _check_table(where, block, required)
        engine_family = ASR_ENGINE_FAMILY[engine]
        if model["family"] != engine_family:
            # ONE ID IS ONE MODEL ON EVERY BACKEND. Whisper on the PC and Qwen
            # on the Mac under one id would be two different transcribers
            # wearing one name, and no provenance record makes that honest.
            raise AsrManifestError(
                f"{where}: engine {engine!r} runs the {engine_family!r} family "
                f"and this manifest's [model] family is {model['family']!r}; "
                "every block of one asr id runs the same model"
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
        )

    # THE ID NAMES ITS FAMILY. `whisper-tiny`, `qwen3-asr-1.7b`: a reader of a
    # transcript's id knows which model made it before reading anything else.
    if not model_id.startswith(f"{model['family']}-"):
        raise AsrManifestError(
            f"{path.name}: model.id {model_id!r} must begin with its family "
            f"({model['family']!r}) and a hyphen"
        )

    # ONE CHECKPOINT, WHERE THE ENGINES CAN SHARE ONE. `qwen3-asr-1.7b` is
    # the official checkpoint on both machines by Owen's ruling, so blocks
    # that disagree about it are a community conversion slipping in under the
    # official id. Whisper is exempt for a reason, not by oversight:
    # CTranslate2 and MLX cannot read each other's format
    # (`ONE_CHECKPOINT_FAMILIES`).
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
    """The Qwen keys' VALUES, once `_check_table` has proved they are present."""
    if block["dtype"] not in QWEN_ASR_DTYPES:
        raise AsrManifestError(
            f"{where}: dtype {block['dtype']!r} is not one this engine runs; "
            f"Qwen3-ASR runs in {sorted(QWEN_ASR_DTYPES)} on both machines "
            "(Owen, 2026-09-24: full precision, never the 8-bit build)"
        )
    if not _MODEL_ID.match(block["aligner"]):
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
        # Both are handed one piece per call because the loop guard reads each
        # piece's own token count. mlx-audio 0.5.5 batches only the chunks it
        # cut out of ONE input anyway; `qwen_asr` on MPS at a batch of 4 used
        # 41 GB and at its default of 32 aborted the process (ContentStudio,
        # 2026-09-24). Any batch above 1 is a number nothing here would honour.
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
    return _resolve_weights_of(parse_asr_manifest(text, path, model_id), root)


def _resolve_weights_of(manifest: AsrManifest, root: Path) -> AsrManifest:
    """An alias with its base attached, or a refusal naming the broken rule.

    `crucible/manifests.py`'s `resolve_weights_of` rules for asr: the base must
    be a manifest here, must not itself be an alias (a folder has one owner),
    must be the same family, and must pin the SAME repo and revision on every
    backend the alias declares. Otherwise two ids would share a folder holding
    bytes one of them never pinned.
    """
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
    """Every asr manifest beside this one whose `weights_of` names it."""
    if manifest.weights_of is not None:
        return ()
    found = load_all_asr_manifests(manifest.path.parent)
    return tuple(other for other in found.values() if other.weights_of == manifest.id)


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
