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
The whisper rule above — different weights must never share an id — is not
relaxed; it is satisfied a second way. mlx-audio reads the very safetensors
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

#: The Qwen3-ASR engines, one per backend. Named once because three modules ask
#: "is this a Qwen engine": the job type dispatches on it, this loader requires
#: the Qwen keys on it, and the worker table is keyed by it.
VLLM_ENGINE = "vllm"
MLX_AUDIO_ENGINE = "mlx-audio"
QWEN_ASR_ENGINES: frozenset[str] = frozenset({VLLM_ENGINE, MLX_AUDIO_ENGINE})

#: Which engines each backend is allowed to name. Whisper is one engine per
#: backend because CTranslate2 has no Metal backend (the module docstring);
#: Qwen3-ASR is one engine per backend because vLLM has no Metal backend and
#: mlx-audio has no CUDA one. Two per backend, and never each other's.
ASR_BACKEND_ENGINES: dict[str, frozenset[str]] = {
    CUDA_LINUX: frozenset({"faster-whisper", VLLM_ENGINE}),
    MLX_DARWIN: frozenset({"mlx-whisper", MLX_AUDIO_ENGINE}),
}

#: The id prefix each engine's manifests must carry. Not decoration: it is what
#: stops one id ever standing for two different sets of weights, which is the
#: thing `transcript.json` cannot recover from. Checked by the loader, so a new
#: manifest cannot break the rule by being written carelessly.
#:
#: The two Qwen engines share one prefix, and that is the rule working rather
#: than an exception to it: they read the same official checkpoint, and
#: `_parse` refuses a manifest whose blocks pin different bytes.
ASR_ENGINE_ID_PREFIX: dict[str, str] = {
    "faster-whisper": "faster-whisper-",
    "mlx-whisper": "mlx-whisper-",
    VLLM_ENGINE: "qwen3-asr-",
    MLX_AUDIO_ENGINE: "qwen3-asr-",
}

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
                "mlx-whisper and mlx-audio are MLX, which has no CUDA one. They "
                "are not two recipes for one thing"
            )
        required = dict(_BACKEND_REQUIRED)
        if engine in QWEN_ASR_ENGINES:
            required.update(QWEN_BACKEND_REQUIRED)
        if engine == VLLM_ENGINE:
            required.update(VLLM_BACKEND_REQUIRED)
        _check_table(where, block, required)
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

    # ONE ID, ONE SET OF BYTES. The first asr id with more than one block
    # (`qwen3-asr-1.7b`) is honest only because both blocks read the same
    # checkpoint; a manifest whose blocks disagree about that is two models
    # wearing one name, and `transcript.json` could not tell them apart.
    pins = sorted({(spec.hf_repo, spec.revision) for spec in backends.values()})
    if len(pins) > 1:
        raise AsrManifestError(
            f"{path.name}: its backend blocks pin different weights "
            f"({[f'{repo}@{revision[:12]}' for repo, revision in pins]}); one asr "
            "id is one set of bytes, because a transcript records the id and "
            "nothing else about what produced it. Weights that differ per "
            "backend need an id per backend, the way mlx-whisper's do"
        )

    return AsrManifest(
        id=model_id,
        family=model["family"],
        parameters_m=model["parameters_m"],
        backends=backends,
        path=path,
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
    if engine == MLX_AUDIO_ENGINE and block["max_batch"] != 1:
        # mlx-audio 0.5.5 batches only the chunks it cut out of ONE input
        # (`Qwen3ASRModel._generate_chunks_batched`, padded to equal length); a
        # list of inputs is decoded one after another. Crucible hands it one
        # piece per call because the loop guard reads each piece's own token
        # count, so any batch above 1 is a number nothing would honour.
        raise AsrManifestError(
            f"{where}: max_batch is {block['max_batch']}, and mlx-audio decodes "
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
