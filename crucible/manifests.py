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

The local form
--------------
A manifest may carry one `[local]` table: what the SAME model is on a machine
that has no Crucible at all — an Ollama tag, or a GGUF (plus its vision
projector) for llama-server. Owen, 2026-09-13: these manifests are the catalog
of record for Foundry's local lineup too, so that "what can this machine run"
has one owner (PHASE9-CAPABILITY.md, "The local form: one catalog, two doors").
`crucible/lineup.py` reads the table into the JSON Foundry vendors; nothing on
Crucible's own wire reads it, because Crucible never runs Ollama. It is
validated exactly as strictly as every other table here, for the same reason:
a `needs_byte` typo must not quietly become a row with no memory figure that a
picker then lights on a card that cannot hold it.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backend import CUDA_LINUX, LLAMA_WINDOWS, MLX_DARWIN
from .errors import CrucibleError

MODELS_DIR_ENV = "CRUCIBLE_MODELS_DIR"

#: The two CLASS FAMILIES a `models/` manifest can belong to. Not a new key and
#: not a new vocabulary: `pages` is exactly the set of models whose
#: `modalities` carries `image`, which is already the load-bearing fact about
#: what this server OFFERS the model for, and `text` is the rest. Deriving the
#: family instead of declaring it keeps one owner (ARCHITECTURE.md R1) —
#: `qwen3.5-9b` has a vision tower and is served text-only, and it says so
#: once.
TEXT_FAMILY = "text"
PAGES_FAMILY = "pages"
CLASS_FAMILIES: tuple[str, ...] = (TEXT_FAMILY, PAGES_FAMILY)

#: Which engine each backend is allowed to name, PER CLASS FAMILY. A manifest
#: that pairs them any other way is a manifest bug, not a runtime decision.
#:
#: ONE ENGINE PER (BACKEND, CLASS FAMILY), decided in docs/PHASE15-HOST.md 4.6
#: and not one per backend as this table read until 2026-09-14. The old shape
#: was never a rule anybody chose; it was true by accident because `cuda-linux`
#: happens to serve both families with vLLM. `mlx-darwin` cannot: `mlx-lm` is a
#: text server that cannot be handed an image, so a Mac that reads pages needs
#: a second server class beside it, and a table with one slot per backend had
#: nowhere to say which.
#:
#: **The lease and the four facts are per RESIDENT, not per engine**, so nothing
#: about arbitration changes. What changes is that `crucible/residency.py`
#: learns which server class to start from the manifest's own `engine`, which
#: it already read — the block below is what decides whether that name is
#: allowed.
#:
#: `mlx-vlm` appears here as the `pages` engine on `mlx-darwin` and NO manifest
#: names it yet. That is deliberate and it is written down in
#: `models/dots-ocr.toml` with the measurement that stopped it: mlx-vlm's own
#: HTTP server does not put the image into the prompt for dots.ocr, so shipping
#: the block would light `pages: yes` on a Mac that answers every page with a
#: single `Picture`. The engine slot exists so the day that is fixed is a
#: manifest block and nothing else.
BACKEND_ENGINES: dict[str, dict[str, str]] = {
    CUDA_LINUX: {TEXT_FAMILY: "vllm", PAGES_FAMILY: "vllm"},
    MLX_DARWIN: {TEXT_FAMILY: "mlx-lm", PAGES_FAMILY: "mlx-vlm"},
    # PHASE15-HOST.md sections 0 and 3.10. Windows natively, llama.cpp's
    # `llama-server` on GGUF. BOTH families, one name, and that is not a
    # slot left unfilled: llama.cpp serves a text GGUF and a vision GGUF
    # pair from the same binary, with `--mmproj` as the whole difference,
    # so this backend genuinely has one engine for both. A block for it
    # names FILES inside the repo rather than the whole repo, because a
    # GGUF repo holds twenty quantizations and this server pulls one.
    LLAMA_WINDOWS: {TEXT_FAMILY: "llama-server", PAGES_FAMILY: "llama-server"},
}


def class_family(modalities: "tuple[str, ...] | list[str]") -> str:
    """Which family a BACKEND BLOCK belongs to, from what it SERVES.

    `image` in the served modalities makes it a page reader and nothing else
    does. The family is DERIVED rather than declared for the reason the table
    above gives: a `family = "pages"` key would be a second owner of a fact the
    modalities already state.

    PER BLOCK SINCE 2026-09-23 (PHASE22 section 2.9). It was read off the
    model's `[model] modalities`, model-wide, and that was one fact standing in
    for two: what the WEIGHTS accept and what a given ENGINE serves. They came
    apart the first time a small Qwen3.5 was offered for images: vLLM and
    llama-server serve its tower, and on the Mac the text engine is mlx-lm,
    which cannot — while the Mac's `pages` engine is Crucible's own dots-
    specific server. A model-wide `image` would have sent these weights to the
    page server on the Mac. So callers pass a block's `serves`
    (`BackendSpec.serves`), which defaults to the model's modalities and may
    only narrow them.
    """
    return PAGES_FAMILY if "image" in modalities else TEXT_FAMILY


def engine_for(backend_kind: str, modalities: "tuple[str, ...] | list[str]") -> str:
    """The engine this backend serves this family with. Refuses either unknown.

    `modalities` is what the block SERVES (see `class_family`)."""
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


#: What a client may put in a chat request's content parts for this model
#: (PHASE3-VLM.md section 2). It is a statement about what Crucible OFFERS on
#: this server, not about what the checkpoint could theoretically do:
#: `qwen3.5-9b` has a vision tower and is served text-only here, and says
#: `["text"]` because a client that sends it a page gets an engine error rather
#: than a reading.
MODALITIES: frozenset[str] = frozenset({"text", "image"})

#: vLLM's flag for "do not budget for an image while profiling". It does not make
#: an engine text-only; it only stops it RESERVING for one image of the maximum
#: feature size, which on `qwen3.5-9b` was MEASURED at 1.90 GiB of the budget and
#: is the difference between a KV pool and no KV pool at all
#: (models/qwen3.5-9b.toml). That manifest's comment has always carried the
#: condition — "if the `llm` proxy is ever given image input, this line must come
#: out and the utilisation be measured again" — and `modalities` is the first
#: thing in the repo that can say when that day has arrived. So the loader now
#: refuses the pair rather than trusting a reviewer to remember, because the
#: failure it prevents is silent: an image-capable model started with this flag
#: loads, serves, and then meets a real page with no reservation behind it.
SKIP_MM_PROFILING = "--skip-mm-profiling"

#: vLLM's flag for "this model is served text-only", and the SECOND half of the
#: pair above rather than a variation on it.
#:
#: `--skip-mm-profiling` stops the engine RESERVING for an image;
#: `--language-model-only` stops it READING THE VISION TOWER AT ALL. vLLM 0.29
#: implements it by returning 0 from `MultiModalConfig.get_limit_per_prompt` for
#: every modality, which puts the tower's construction inside `no_init_weights`
#: (`model_executor/models/interfaces.py`) so its parameters are never allocated.
#: On the two multimodal text checkpoints this build serves that is 912_020_960 B
#: and 921_460_192 B of weights the `llm` lane can never reach — BookForge's
#: `--limit-mm-per-prompt '{"image":0,"video":0}'` said in vLLM's own vocabulary.
#:
#: IT IS REFUSED BESIDE `image` FOR A WORSE REASON THAN ITS PARTNER IS. A model
#: advertised for images and started with `--skip-mm-profiling` meets a page with
#: nothing reserved and falls over — loudly, eventually. One started
#: `--language-model-only` has no tower to show the page to, so it ANSWERS: a
#: well-formed reading of a page the model never saw, which is the exact failure
#: `engines/mlx_vlm.py` refuses to ship a manifest for. A wrong answer nothing
#: records is the one outcome this loader exists to make impossible.
LANGUAGE_MODEL_ONLY = "--language-model-only"

#: What a `[defaults]` table may state, and the type each key takes.
#:
#: **Only keys an engine actually honours.** Every one of these reaches vLLM's
#: and mlx-lm's OpenAI chat surface under this exact name — `temperature`,
#: `top_p`, `top_k`, `max_tokens` and `repetition_penalty` are sampling fields
#: both accept, and `thinking` is the one that is not a wire field at all: it
#: becomes `chat_template_kwargs: {"enable_thinking": …}`, which PHASE2-LLM.md
#: section 5 already records as read per request by both engines.
#:
#: A key outside this table is a REFUSAL naming it, for the reason every other
#: manifest key is: a `[defaults]` block with `temperture` in it must not load
#: with no temperature and leave a reader believing one was set. And a knob no
#: engine reads would be worse than useless — it would be a number in a file
#: that looks like it is doing something.
#:
#: The float-typed keys accept an int too, because TOML reads `temperature = 0`
#: as an integer and "zero temperature" is exactly the value a page reader
#: wants. They are stored as floats.
DEFAULTS_KEYS: dict[str, type] = {
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "max_tokens": int,
    "repetition_penalty": float,
    "thinking": bool,
}

#: Which of those are sampling fields sent under their own name, in the order a
#: row reports them. `thinking` is deliberately absent: it travels inside
#: `chat_template_kwargs` and is applied by `crucible/sampling.py`.
DEFAULTS_WIRE_KEYS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "repetition_penalty",
)

#: A count of billions of parameters, which is not always whole: Qwen publishes a
#: `Qwen3.5-0.8B`. An int or a float; a bool is neither (`check_table`).
NUMBER: tuple[type, ...] = (int, float)

_MODEL_REQUIRED: dict[str, Any] = {
    "id": str,
    "family": str,
    # The capability classes' floors are read against this
    # (`capability.CatalogCandidates.min_params_b`, 2026-09-23), so it is a
    # load-bearing number and not a label: the 9B floor for clean / translate /
    # simplify / analysis is `params_b >= 9`.
    "params_b": NUMBER,
    "context_default": int,
    # WHAT THE WEIGHTS SUPPORT, which is a different fact from either of the
    # other two contexts in this file and is the only one that belongs to the
    # checkpoint rather than to a machine. `context_default` is what a host
    # CHOOSES to serve; a backend block's override is what an accelerator has
    # ROOM for; this is the wall behind both, `max_position_embeddings` from the
    # repo's own config.json at the pinned revision.
    #
    # Required, and required for a reason that showed up the hour it was missing:
    # the ceiling Crucible derives from the card (`max_context_affordable`) is
    # unbounded above without it, and the Mac's 64 GB of unified memory afforded
    # 1_389_135 tokens of the 9B — five times what the weights can do. Publishing
    # that would be handing out a number nothing behind it can honour, which is
    # exactly what this design refuses Ollama for (docs/FITS-AND-THE-CARD.md
    # section 6.1). A model with no stated wall must not silently get an
    # infinite one.
    "trained_context": int,
    # Required, not defaulted to `["text"]`, for the reason every other key in
    # this table is required: a manifest that forgets it must not quietly load as
    # text-only and have a page reader refused at request time with an error
    # about content parts, several layers away from the file that was wrong.
    "modalities": list,
}
_MODEL_OPTIONAL: dict[str, type] = {
    # The two display facts a catalog row carries: what a picker prints as the
    # model's name, and the sentence under it. Optional on the model, because
    # Crucible's own doors name a model by id and nothing on the wire draws a
    # tile; REQUIRED the moment the manifest carries a `[local]` table, because
    # the lineup that table feeds IS drawn on a screen, and a row with no label
    # is a row somebody downstream would invent a label for. `display` is the
    # name every other catalog in this repo already uses (voices, rvc, denoise).
    "display": str,
    "description": str,
}
_BACKEND_REQUIRED: dict[str, type] = {
    "engine": str,
    "hf_repo": str,
    "revision": str,
    "memory_bytes_estimate": int,
}
_BACKEND_OPTIONAL: dict[str, type] = {
    "engine_args": list,
    # `llama-windows` only, and REQUIRED there — checked below rather than in
    # this table, because the table is per-key and this rule is per-backend.
    # A GGUF repo holds every quantization of a model; `file` says which one
    # this row is, and `mmproj` says which vision projector goes with it.
    # `mmproj` is absent for a text model and MANDATORY for one this server
    # offers images on: half a vision model is a model that loads and then
    # cannot see (section 3.10, fact 2).
    "file": str,
    "mmproj": str,
    # A context this backend can actually hold, when the model's own number is
    # not one it can. `[model] context_default` is what the model is FOR; this is
    # what a particular accelerator has room for, and the two are allowed to
    # disagree — `qwen3.8-27b-4bit` wants Owen's 98304 and gets it on 64 GB of
    # unified memory, while 98304 of its KV is 7.9 GiB the 3090 Ti does not have
    # once the weights are down. Absent means "the model's number"; it is never
    # a silent default.
    "context_default": int,
    # THE SAME NUMBER AS `memory_bytes_estimate`, TAKEN APART — see
    # docs/FITS-AND-THE-CARD.md. The collapsed estimate answers one question
    # ("does this model fit at the context this block serves") and the class that
    # is about to run is not always asking it: translate sends a paragraph at a
    # time, batched and independent, and gets refused for a KV cache it will
    # never fill. Optional, because a block that has not been taken apart is
    # honest as it stands; where it IS present the two must agree, and the parser
    # below checks that rather than trusting it.
    "memory": dict,
    # WHAT THIS BACKEND SERVES, when it is less than what the weights accept
    # (PHASE22-DECIDE.md section 2.9, 2026-09-23). `[model] modalities` is the
    # model's: what the checkpoint can be shown. This is the block's: what the
    # ENGINE this block names will actually be handed. Absent means "everything
    # the model accepts"; present, it must be a non-empty subset of it, and it
    # is what the engine choice (`engine_for`), the image-pairing rules below
    # (`mmproj`, `--skip-mm-profiling`, `--language-model-only`) and the
    # decision door's `model_text_only` all read. The case it exists for: a
    # small Qwen3.5 whose tower vLLM and llama-server serve, on a Mac whose text
    # engine (mlx-lm) cannot see a picture.
    "serves": list,
}

#: What `[backends.<kind>.memory]` must state. Every term is a count of bytes
#: except `basis` and the context the estimate was taken at, and nothing is
#: optional: a term left out would be a term some later reader supplies from
#: somewhere else.
_MEMORY_REQUIRED: dict[str, type] = {
    #: What the weights themselves occupy ON THE CARD, which is not the size of
    #: the files: vLLM's own "Model loading took" is larger than the safetensors
    #: sum, and an MLX load is smaller because it maps rather than reads.
    "weights_bytes": int,
    #: Everything the engine holds that is neither weights nor KV — activation
    #: peak, CUDA graphs, the allocator's own slack. It is a real term and it is
    #: the one nobody remembers: on the 27B-4bit it is 1.44 GiB, larger than the
    #: whole KV pool that block runs with.
    "overhead_bytes": int,
    #: The slope. MEASURED where a card has answered, COMPUTED from config.json
    #: where none has — and the difference is not academic: the 27B-4bit's
    #: computed 65_536 was 24% under its measured 86_251, because vLLM pads the
    #: attention page up to the linear layers' recurrent state. A computed rate
    #: is a FLOOR.
    "kv_bytes_per_token": int,
    #: `measured` or `computed`, and it travels to the screen for the reason
    #: `[local] needs_basis` does: a picker is entitled to know which of the two
    #: kinds of number it is about to refuse somebody with.
    "basis": str,
    #: The context the numbers above were taken at. Present so the consistency
    #: check below has something to check AGAINST, and because an estimate is
    #: only true at the context it was measured at — the defect that put a Mac
    #: context of 98304 on a Windows block with a 1.5 GB KV allowance
    #: (docs/FITS-AND-THE-CARD.md section 0b).
    "measured_at_context": int,
}

#: Where the terms came from, weakest last. `measured` is watched on a card;
#: `computed` is derived from the checkpoint's own config.json — and is a FLOOR,
#: because the 27B-4bit's computed 65_536 B/token turned out to be 24% under the
#: 86_251 its card actually spent. `declared` is neither: it is a stated
#: allowance nobody has watched or worked out, which is what every
#: `llama-windows` block has today (Foundry's 1.5 GB `OVERHEAD_GB`), and it
#: travels to the screen for the reason `[local] needs_basis` does — a picker is
#: entitled to know which of the three it is about to refuse somebody with.
MEMORY_BASES: frozenset[str] = frozenset({"measured", "computed", "declared"})

#: How far the terms may sit from the collapsed estimate before the manifest is
#: refused. They are measurements of PARTS against a measurement of the WHOLE at
#: a peak, so bit-exact agreement would be a lie rather than a standard; 5% is
#: wide enough for that and narrow enough to catch the mistakes that actually
#: happen — a term typed in GB where the estimate is GiB (7.4% adrift), or a KV
#: rate computed from config.json where the card says otherwise (24% on the
#: 27B-4bit).
MEMORY_TERMS_TOLERANCE = 0.05

#: The two shapes a model takes on a machine with no Crucible. `ollama` is a
#: tag in Ollama's library; `gguf` is a file (and, for a model that reads
#: images, a projector) in a HuggingFace repo, served by llama-server. There is
#: no third kind: an MLX conversion is Crucible's own `mlx-darwin` block, not a
#: local form, because on the Mac Crucible IS the local route.
LOCAL_KINDS: frozenset[str] = frozenset({"ollama", "gguf"})

#: Where `needs_bytes` came from — the same discipline the voice manifests keep
#: as `estimate_basis`. `measured` is a number watched on a card; `declared` is
#: arithmetic (a download plus a stated overhead), and a row that is declared
#: says so all the way to the screen, so a picker can err on the side it wants.
NEEDS_BASES: frozenset[str] = frozenset({"measured", "declared"})

_LOCAL_COMMON_REQUIRED: dict[str, type] = {
    "kind": str,
    #: The bytes a pull fetches, so a picker can say what it is about to ask for.
    "download_bytes": int,
    #: The memory the model wants to RUN — weights resident plus its working room.
    "needs_bytes": int,
    "needs_basis": str,
}
_LOCAL_COMMON_OPTIONAL: dict[str, type] = {
    #: Owen's tile rule: the classes this model is the FLOOR for — the smallest
    #: model the class may run on at all. A machine that cannot hold a model
    #: named here does not light that class's tile. Each entry is a name in
    #: `capability.CLASSES`; the manifest is not allowed to invent a class.
    "minimum_for": list,
}
_LOCAL_KIND_REQUIRED: dict[str, dict[str, type]] = {
    "ollama": {
        #: The exact tag — what `ollama pull` is given and what `--model` is.
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
        #: The vision projector llama-server is handed as `--mmproj`. Required
        #: for a model whose `modalities` carries `image` and refused for one
        #: whose does not: a server started without it loads, answers
        #: `/v1/models`, and then refuses every page — which looks exactly like
        #: a broken page rather than a missing file.
        "mmproj": str,
    },
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
class MemoryTerms:
    """`memory_bytes_estimate`, taken apart into the terms it is made of.

        engine_total = weights + overhead + kv_bytes_per_token x context x concurrency

    The first two belong to the model and the engine; the last two belong to the
    WORK, and that is the whole point of the split (docs/FITS-AND-THE-CARD.md
    section 1). Collapsed into one number keyed to one context, a 27B is refused
    on a 24 GB card for a 98304-token working context that translate — which
    sends a paragraph at a time, batched, each block independent of the last —
    is never going to ask for.
    """

    weights_bytes: int
    overhead_bytes: int
    kv_bytes_per_token: int
    basis: str
    measured_at_context: int

    @property
    def fixed_bytes(self) -> int:
        """What is on the card before a single token of KV: the intercept."""
        return self.weights_bytes + self.overhead_bytes

    def bytes_for(self, *, context: int, concurrency: int) -> int:
        """What the engine holds serving `concurrency` requests of `context`."""
        if context <= 0:
            raise ValueError(f"context must be positive, got {context}")
        if concurrency <= 0:
            raise ValueError(f"concurrency must be positive, got {concurrency}")
        return self.fixed_bytes + self.kv_bytes_per_token * context * concurrency

    def max_context(self, *, available_bytes: int, concurrency: int) -> int:
        """The tallest context this many bytes affords, at this concurrency.

        THE ANSWER TO "how much can I send", and it is derived rather than typed
        — the number a client should read instead of guessing a chunk size and
        discovering the ceiling as a 400 (docs/FITS-AND-THE-CARD.md section 6.3).
        Zero means the weights and overhead alone do not fit, which is a
        different refusal from "your request is too long" and must not be
        rounded into one.
        """
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
    """One `[backends.<kind>]` block."""

    backend: str
    engine: str
    hf_repo: str
    revision: str
    memory_bytes_estimate: int
    engine_args: tuple[str, ...]
    #: This backend's own context, or None to use the model's.
    context_default: int | None
    #: This block's estimate taken apart, or None where nobody has taken it
    #: apart yet. `memory_bytes_estimate` above stays the answer at this block's
    #: own context either way; these terms are what lets a CLASS ask a different
    #: question (docs/FITS-AND-THE-CARD.md).
    memory: "MemoryTerms | None" = None
    #: `llama-windows`: the one GGUF in `hf_repo` this row IS, and the vision
    #: projector beside it. None on every other backend, where the whole repo
    #: is the weights and there is no file to choose.
    file: str | None = None
    mmproj: str | None = None
    #: What this block's engine is handed: the block's `serves`, or the model's
    #: `modalities` when the block does not narrow them. Always resolved by the
    #: parser, never empty; `()` only on a spec built by hand in a test that
    #: does not care, which the parser never produces.
    serves: tuple[str, ...] = ()

    @property
    def files(self) -> tuple[str, ...]:
        """Every file this spec names, in pull order. Empty = the whole repo.

        The one place "which files does this backend fetch" is answered, so
        the puller, the catalog's `installed` and the engine's `-m` cannot
        come to disagree about whether a subject is complete (section 3.5's
        last bullet: the catalog's `installed` list is the input to the host's
        weights migration, so it has to be exact).
        """
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
            "memory": None if self.memory is None else self.memory.to_dict(),
            "file": self.file,
            "mmproj": self.mmproj,
            "serves": list(self.serves),
        }


@dataclass(frozen=True)
class ModelDefaults:
    """`[defaults]` — what this model is answered with when a request is silent.

    PHASE2-LLM.md section 9. Every field is `None` when the manifest did not
    state it, and `None` means **the engine's own default**, never a value
    Crucible picked: a server that filled one in would be substituting its guess
    for the engine's measurement, and nothing in the answer would say so.

    The precedence is one line and it is the whole contract: **a field the
    request STATES wins; a field the request omits takes the manifest's; a field
    neither states is the engine's.** `crucible/sampling.py` is the one place
    that is applied, and it reports which of the three each effective value came
    from, because a default that cannot be seen is a default nobody can debug.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None
    thinking: bool | None = None

    def stated(self) -> dict[str, Any]:
        """Only the keys this manifest actually states, in `DEFAULTS_KEYS` order.

        The distinction this method exists for: an absent key and a key set to
        the engine's own value are different manifests, and only the second one
        is a decision somebody made.
        """
        values: dict[str, Any] = {}
        for key in DEFAULTS_KEYS:
            value = getattr(self, key)
            if value is not None:
                values[key] = value
        return values

    def to_dict(self) -> dict[str, Any]:
        """All six keys, `null` where this manifest states nothing.

        Never the `stated()` subset: a row whose keys come and go would make
        "this model has no default temperature" and "this build predates the
        field" read the same, which is the mistake `[jobs] enable_*` had to be
        rescued from in `crucible/config.py`.
        """
        return {key: getattr(self, key) for key in DEFAULTS_KEYS}


#: A manifest that states nothing. Shared rather than rebuilt, and it is what
#: every manifest without a `[defaults]` table carries — so the application code
#: has no "is there a defaults table" branch at all.
NO_DEFAULTS = ModelDefaults()


@dataclass(frozen=True)
class LocalForm:
    """`[local]` — what this model is on a machine with no Crucible.

    The two concrete shapes are `OllamaLocal` and `GgufLocal`; this is what they
    share. A manifest without the table has `local = None` on its `ModelManifest`,
    and that is a statement — "this model has no local form; a machine without
    Crucible cannot run it" — which `crucible/lineup.py` reports rather than
    fills in.
    """

    kind: str
    download_bytes: int
    needs_bytes: int
    needs_basis: str
    #: In the order the manifest wrote them. Empty when the model is the floor
    #: for nothing, which is most models.
    minimum_for: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "download_bytes": self.download_bytes,
            "needs_bytes": self.needs_bytes,
            "needs_basis": self.needs_basis,
            "minimum_for": list(self.minimum_for),
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
    #: None on a text-only model, and only there.
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
    #: Billions of parameters — an int where the model is whole (9, 27), a float
    #: where it is not (0.8). Capability floors compare against it.
    params_b: int | float
    context_default: int
    #: What `max_position_embeddings` says at the pinned revision — the wall
    #: behind every host's choice.
    trained_context: int
    #: In the order the manifest wrote them, so a row reads the way the file does.
    modalities: tuple[str, ...]
    backends: dict[str, BackendSpec]
    path: Path
    #: `[defaults]`, or `NO_DEFAULTS` when the manifest has no such table. A
    #: MODEL-level fact and not a per-backend one: the reason a model wants
    #: `thinking = false` is what the model does with a prompt, which is the
    #: same on both cards. If a backend ever needs its own, it needs its own
    #: argument first.
    defaults: ModelDefaults = NO_DEFAULTS
    #: `[model] display` and `[model] description`, or None where the manifest
    #: states neither. Both are present whenever `local` is, by validation.
    display: str | None = None
    description: str | None = None
    #: `[local]`, or None: no local form, which the lineup reports by name.
    local: LocalForm | None = None

    #: Which subtree of `~/.crucible/` this thing's weights live under. A model id
    #: and a voice id are separate namespaces and must not be able to collide on
    #: disk — see `crucible/weights.py`.
    weights_family = "models"

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

    def serves(self, backend_kind: str) -> tuple[str, ...]:
        """What this model is SERVED for on `backend_kind`, or a named refusal.

        Not `self.modalities`: that is what the weights accept, and a backend may
        serve less (PHASE22-DECIDE.md section 2.9). Every question of the form
        "may I send this model an image HERE" asks this.
        """
        return self.spec(backend_kind).serves

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
            "trained_context": self.trained_context,
            "modalities": list(self.modalities),
            # Always present, null when unstated, for the reason `defaults` is:
            # a row whose keys come and go cannot tell "no display name" from
            # "a build that predates the field".
            "display": self.display,
            "description": self.description,
            "defaults": self.defaults.to_dict(),
            "backends": {k: v.to_dict() for k, v in sorted(self.backends.items())},
            # `local` is deliberately NOT here. It is what a machine WITHOUT
            # Crucible runs, and this dict is what a Crucible tells its clients
            # about itself; its one door is the lineup file (crucible/lineup.py).
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
    path = Path(__file__).resolve().parent / "models"
    if not path.is_dir():
        raise ManifestError(
            f"no model manifests at {path}; they are package data and this "
            f"install has lost them, or ${MODELS_DIR_ENV} must point at them"
        )
    return path


# ------------------------------------------------------------------ checking


def check_table(
    where: str,
    table: dict[str, Any],
    required: dict[str, type],
    optional: dict[str, type],
    *,
    error: type[CrucibleError] = ManifestError,
) -> None:
    """Every required key present and correctly typed; no key that is not listed.

    `error` is the exception class to refuse with, because the voice manifests
    (`crucible/voices.py`) are the same kind of file held to the same strictness
    and must refuse in their own vocabulary — a reader told "manifest" about a
    voice file goes looking in `models/`.

    A kind may be a tuple of types (`NUMBER`), named in a refusal as
    "int or float".
    """
    allowed = set(required) | set(optional)
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise error(
            f"{where}: unknown key(s) {unknown}; this table takes exactly "
            f"{sorted(allowed)}"
        )
    missing = sorted(set(required) - set(table))
    if missing:
        raise error(f"{where}: missing required key(s) {missing}")
    for key, kind in {**required, **optional}.items():
        if key not in table:
            continue
        value = table[key]
        kinds = kind if isinstance(kind, tuple) else (kind,)
        wrong = not isinstance(value, kinds)
        # bool is a subclass of int; a bool where an int is wanted is still wrong.
        if int in kinds and isinstance(value, bool):
            wrong = True
        if wrong:
            named = " or ".join(k.__name__ for k in kinds)
            raise error(
                f"{where}: {key} must be {named}, got "
                f"{type(value).__name__}"
            )


#: The bounds each `[defaults]` key is checked against, as (test, why). They are
#: the engines' own ranges, and a value outside one is refused here rather than
#: at the first request: a manifest is read at startup and a bad number in it
#: should not become a 400 on somebody's book.
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


def _parse_defaults(table: Any, path: Path) -> ModelDefaults:
    """`[defaults]`, validated as strictly as every other table in this file.

    Unknown key: refusal naming it. Wrong type: refusal naming both. Out of
    range: refusal naming the bound. An absent table is `NO_DEFAULTS`, which is
    a statement — "this model states none" — and not an unknown.
    """
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
        value = table[key]
        if kind is bool:
            if not isinstance(value, bool):
                raise ManifestError(
                    f"{where}: {key} must be bool, got {type(value).__name__}"
                )
        elif kind is int:
            # bool is a subclass of int; a bool where an int is wanted is wrong.
            if not isinstance(value, int) or isinstance(value, bool):
                raise ManifestError(
                    f"{where}: {key} must be int, got {type(value).__name__}"
                )
        else:
            # TOML reads `temperature = 0` as an int, and zero temperature is
            # exactly what a page reader wants, so an int is accepted and
            # stored as a float rather than refused on a technicality.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ManifestError(
                    f"{where}: {key} must be a number, got {type(value).__name__}"
                )
            value = float(value)
        test, why = _DEFAULT_BOUNDS[key]
        if not test(value):
            raise ManifestError(f"{where}: {key} is {value!r} and {why}")
        values[key] = value
    return ModelDefaults(**values)


_GGUF_FILE = ".gguf"


def _gguf_name(where: str, key: str, value: str) -> str:
    """A GGUF file name as the hub's tree lists it: non-empty, ending `.gguf`."""
    if value == "" or value.strip() != value:
        raise ManifestError(f"{where}: {key} must be a file name, got {value!r}")
    if not value.endswith(_GGUF_FILE):
        raise ManifestError(
            f"{where}: {key} {value!r} does not end in {_GGUF_FILE!r}; llama-server "
            "reads nothing else, and a name without the extension is usually a "
            "repo id or a directory rather than the file"
        )
    return value


def _parse_local(
    table: Any, path: Path, modalities: tuple[str, ...]
) -> LocalForm:
    """`[local]`, validated as strictly as every other table in this file.

    Unknown key: refusal naming it, and a key that belongs to the OTHER kind is
    an unknown key — an ollama block with `hf_repo` in it is a block somebody
    half-converted. Wrong type: refusal naming both. A pin that is not a pin
    (a bare Ollama name, a branch name for a revision): refused, because the
    lineup this feeds is vendored into another app and compared by content, and
    a floating pointer would make two vendorings of one file disagree.
    """
    where = f"{path.name} [local]"
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
    check_table(
        where,
        table,
        {**_LOCAL_COMMON_REQUIRED, **_LOCAL_KIND_REQUIRED[kind]},
        {**_LOCAL_COMMON_OPTIONAL, **_LOCAL_KIND_OPTIONAL[kind]},
    )

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

    minimum_for = table.get("minimum_for", [])
    if "minimum_for" in table and not minimum_for:
        raise ManifestError(
            f"{where}: minimum_for is empty. A model that is the floor for no "
            "class says so by omitting the key; an empty list reads as a "
            "decision somebody made and then forgot to write down"
        )
    if minimum_for:
        # Deferred, not top-level: `crucible/capability.py` imports this module
        # to read the catalog, and the class table is the one owner of which
        # classes exist (ARCHITECTURE.md R1) — a second list of their names here
        # would be the drift this check exists to refuse. Importing it at call
        # time costs nothing and closes the cycle in the only direction it can.
        from .capability import BY_NAME

        for index, entry in enumerate(minimum_for):
            if not isinstance(entry, str):
                raise ManifestError(
                    f"{where}: minimum_for[{index}] must be a string, got "
                    f"{type(entry).__name__}"
                )
            if entry not in BY_NAME:
                raise ManifestError(
                    f"{where}: minimum_for[{index}] is {entry!r}, which is not a "
                    f"capability class; this build knows {sorted(BY_NAME)}"
                )
        if len(set(minimum_for)) != len(minimum_for):
            raise ManifestError(
                f"{where}: minimum_for lists a class twice: {minimum_for}"
            )

    common = {
        "kind": kind,
        "download_bytes": download,
        "needs_bytes": needs,
        "needs_basis": basis,
        "minimum_for": tuple(minimum_for),
    }

    if kind == "ollama":
        tag = table["tag"]
        name, colon, version = tag.partition(":")
        if colon == "" or name == "" or version == "" or tag.split() != [tag]:
            raise ManifestError(
                f"{where}: tag {tag!r} must be <name>:<tag>; a bare name is "
                "`:latest`, which is a floating pointer and not a pin"
            )
        return OllamaLocal(**common, tag=tag)

    if not _HF_REPO.match(table["hf_repo"]):
        raise ManifestError(
            f"{where}: hf_repo {table['hf_repo']!r} is not an <owner>/<name> "
            "HuggingFace repo id"
        )
    if not _REVISION.match(table["revision"]):
        raise ManifestError(
            f"{where}: revision {table['revision']!r} must be a full 40-character "
            "commit sha, so a pull is reproducible; branch names are not pins"
        )
    file = _gguf_name(where, "file", table["file"])
    mmproj = table.get("mmproj")
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
        if not reads_images:
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
    return GgufLocal(
        **common,
        hf_repo=table["hf_repo"],
        revision=table["revision"],
        file=file,
        mmproj=mmproj,
    )


def _parse_serves(
    where: str, block: dict[str, Any], modalities: "list[str] | tuple[str, ...]"
) -> tuple[str, ...]:
    """A block's `serves`, or the model's modalities where it states none.

    A subset and never a superset, refused by name otherwise: a backend cannot
    serve a modality the weights do not accept, and a block that claimed one
    would be a promise with no tower behind it. Empty is refused for the reason
    an empty `[model] modalities` is — a backend that serves nothing is not a
    backend — and a duplicate is refused because it is a typo that reads like
    a decision.
    """
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


def _parse(document: dict[str, Any], path: Path, expected_id: str) -> ModelManifest:
    unknown = sorted(set(document) - {"model", "backends", "defaults", "local"})
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

    model = document["model"]
    if not isinstance(model, dict):
        raise ManifestError(f"{path.name}: [model] must be a table")
    check_table(f"{path.name} [model]", model, _MODEL_REQUIRED, _MODEL_OPTIONAL)

    model_id = model["id"]
    if "/" in model_id:
        # manifest_model_id_slash — PHASE15-HOST.md sections 1 and 3.4. The
        # slash is how the chat door tells `anthropic/claude-sonnet-5` from a
        # model on this card, and it can only do that while no local id has
        # one. `_MODEL_ID` below already excludes it as a side effect of its
        # character class; this says it by NAME, first, because the rule is now
        # load-bearing on another door and a reader who broke it deserves to be
        # told which rule they broke rather than shown a regex.
        raise ManifestError(
            f"{path.name}: manifest_model_id_slash — model.id {model_id!r} "
            "contains '/', which is reserved: a model id with a slash is an "
            "UPSTREAM model id (`<upstream>/<model>`), and the chat door tells "
            "the two apart by that one character"
        )
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
    for key in _MODEL_OPTIONAL:
        if key in model and model[key].strip() == "":
            raise ManifestError(
                f"{path.name}: model.{key} is empty; a display fact nobody wrote "
                "is said by omitting the key, not by an empty string a screen "
                "would print as nothing"
            )
    if "local" in document:
        # The one rule that crosses [model] and [local]: a local form is drawn
        # on a screen, and a row with no name is a row somebody would name for
        # it. Both are required here and nowhere else.
        unnamed = sorted(key for key in _MODEL_OPTIONAL if key not in model)
        if unnamed:
            raise ManifestError(
                f"{path.name}: [local] is present but [model] is missing {unnamed}; "
                "the lineup that table feeds is drawn as a tile, and a tile "
                "needs its label and its sentence from the same file as its "
                "numbers"
            )

    modalities = model["modalities"]
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
        check_table(where, block, _BACKEND_REQUIRED, _BACKEND_OPTIONAL)

        # ------------------------------------------- what this block serves
        # (PHASE22-DECIDE.md section 2.9.) Resolved FIRST, because the engine
        # and every image rule below read it rather than [model] modalities.
        serves_here = _parse_serves(where, block, modalities)

        engine = block["engine"]
        # PER (BACKEND, CLASS FAMILY), and the family comes off what this block
        # SERVES — `modalities` above unless the block narrows it — rather than
        # out of the engine name, so a manifest cannot claim a page-reading
        # engine for a text model by naming one.
        expected = engine_for(kind, serves_here)
        if engine != expected:
            family = class_family(serves_here)
            raise ManifestError(
                f"{where}: engine {engine!r} does not serve {family!r} models on "
                f"{kind}; that pairing's engine is {expected!r}. The family is "
                f"read off [model] modalities = {list(modalities)} as this block "
                f"serves them ({list(serves_here)}; `serves` narrows, never "
                "widens) and not out of the engine name, so an engine cannot be "
                "chosen by naming it"
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
        # WHICH FILE, and it is per-backend rather than per-key. `file` is
        # meaningless on a backend that pulls a whole repo and mandatory on one
        # that pulls one GGUF out of twenty, so the table above cannot express
        # it and this does (section 3.10, fact 2).
        if kind == LLAMA_WINDOWS:
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
                    "optional (PHASE15-HOST.md section 3.10, fact 2)"
                )
            if "image" not in serves_here and "mmproj" in block:
                # The mirror of the rule above, and the one `[local]` has always
                # had: a projector beside an engine that is never sent a page is
                # a file pulled for nothing, and a reader of the block would
                # take it as a promise the server does not keep.
                raise ManifestError(
                    f"{where}: mmproj {block['mmproj']!r} names a vision "
                    f"projector, and this block serves {list(serves_here)}. Either "
                    "serve 'image' here or take the projector out"
                )
            for key in ("file", "mmproj"):
                name = block.get(key)
                if name is None:
                    continue
                if name != Path(name).name or name.startswith("."):
                    raise ManifestError(
                        f"{where}: {key} {name!r} must be a plain file name "
                        "inside the repo, not a path"
                    )
        else:
            extra = sorted({"file", "mmproj"} & set(block))
            if extra:
                raise ManifestError(
                    f"{where}: {extra} belong to a llama-windows block. On "
                    f"{kind} the whole repo is the weights and there is no "
                    "file to choose"
                )
        if "image" in serves_here and SKIP_MM_PROFILING in engine_args:
            # The one rule that crosses the two tables, and it crosses them
            # because the fact and the flag live apart: what a model is offered
            # for is `[model]`, how its engine is started is `[backends.<kind>]`.
            # A model advertised for images whose engine was told not to profile
            # for one starts, serves, and then meets a real page with nothing
            # reserved behind it — the kind of failure that arrives as an OOM in
            # the middle of somebody's book rather than at load.
            raise ManifestError(
                f"{where}: engine_args carries {SKIP_MM_PROFILING!r} while "
                f"[model] modalities declares 'image' and this block serves it. "
                f"That flag stops vLLM "
                f"reserving for an image, so it belongs only to a model this "
                f"server serves text-only; take it out and measure "
                f"--gpu-memory-utilization again with the image profiled in "
                f"(PHASE3-VLM.md section 3)"
            )
        if "image" in serves_here and LANGUAGE_MODEL_ONLY in engine_args:
            # The same crossing of the two tables, and the sharper of the two.
            # This flag does not shrink a budget, it deletes the vision tower:
            # the engine starts, `/v1/models` answers, a page goes in and a
            # reading comes back that the model produced without ever seeing it.
            raise ManifestError(
                f"{where}: engine_args carries {LANGUAGE_MODEL_ONLY!r} while "
                f"[model] modalities declares 'image' and this block serves it. "
                f"That flag makes vLLM skip "
                f"loading the vision tower, so this engine would answer every "
                f"page from the text alone — a well-formed reading of something "
                f"it was never shown. It belongs only to a model this server "
                f"serves text-only"
            )
        backend_context = block.get("context_default")
        if backend_context is not None and backend_context <= 0:
            raise ManifestError(
                f"{where}: context_default must be positive, got {backend_context}"
            )
        if backend_context is not None and backend_context > model["trained_context"]:
            raise ManifestError(
                f"{where}: context_default is {backend_context} and the weights "
                f"are trained at {model['trained_context']}. An accelerator with "
                f"room to spare does not give a checkpoint a longer memory"
            )
        # ------------------------------------------------- the terms, if stated
        terms: MemoryTerms | None = None
        memory_table = block.get("memory")
        if memory_table is not None:
            memory_where = f"{where}.memory"
            if not isinstance(memory_table, dict):
                raise ManifestError(f"{memory_where}: must be a table")
            check_table(memory_where, memory_table, _MEMORY_REQUIRED, {})
            for key in ("weights_bytes", "kv_bytes_per_token"):
                if memory_table[key] <= 0:
                    raise ManifestError(
                        f"{memory_where}: {key} must be positive, got "
                        f"{memory_table[key]}"
                    )
            # OVERHEAD MAY BE ZERO, and zero is a statement rather than a gap:
            # it says this estimate never accounted for the engine holding
            # anything but weights and KV. Both `qwen3.8-27b` blocks are exactly
            # that, and their own header says so — "weights at bfloat16 as they
            # sit on disk plus the KV cache at context_default" — and warns that
            # the same arithmetic ran 6% under on the 9B where a card could
            # check it. Refusing zero here would force somebody to invent a
            # number to get past the parser, which is the opposite of the point.
            if memory_table["overhead_bytes"] < 0:
                raise ManifestError(
                    f"{memory_where}: overhead_bytes cannot be negative, got "
                    f"{memory_table['overhead_bytes']}"
                )
            if memory_table["basis"] not in MEMORY_BASES:
                raise ManifestError(
                    f"{memory_where}: basis {memory_table['basis']!r} is not one "
                    f"of {sorted(MEMORY_BASES)}"
                )
            if memory_table["measured_at_context"] <= 0:
                raise ManifestError(
                    f"{memory_where}: measured_at_context must be positive, got "
                    f"{memory_table['measured_at_context']}"
                )
            terms = MemoryTerms(
                weights_bytes=memory_table["weights_bytes"],
                overhead_bytes=memory_table["overhead_bytes"],
                kv_bytes_per_token=memory_table["kv_bytes_per_token"],
                basis=memory_table["basis"],
                measured_at_context=memory_table["measured_at_context"],
            )
            # THE TWO OWNERS ARE COMPARED, which is the only reason it is safe to
            # have two (ARCHITECTURE.md R1: the chaos is always a fact with two
            # owners and nothing checking them against each other). The terms and
            # the collapsed estimate are two readings of one thing — parts and
            # whole — so they are allowed to differ by a measurement's worth and
            # not by a mistake's worth.
            served = backend_context or model["context_default"]
            if terms.measured_at_context != served:
                raise ManifestError(
                    f"{memory_where}: measured_at_context is "
                    f"{terms.measured_at_context} and this block serves "
                    f"{served}. An estimate is only true at the context it was "
                    f"taken at; state the terms at the context this block runs, "
                    f"or move the block's context_default to match what was "
                    f"measured (docs/FITS-AND-THE-CARD.md section 0b)"
                )
            from_terms = terms.bytes_for(context=served, concurrency=1)
            stated = block["memory_bytes_estimate"]
            drift = abs(from_terms - stated) / stated
            if drift > MEMORY_TERMS_TOLERANCE:
                raise ManifestError(
                    f"{memory_where}: the terms come to {from_terms} bytes at "
                    f"{served} tokens and memory_bytes_estimate says {stated} — "
                    f"{drift:.1%} apart, past the {MEMORY_TERMS_TOLERANCE:.0%} a "
                    f"parts-against-whole reading is allowed. One of the two is "
                    f"wrong and the manifest does not say which; check for GB "
                    f"where GiB was meant, and for a kv_bytes_per_token computed "
                    f"from config.json on a backend whose card says otherwise"
                )
        backends[kind] = BackendSpec(
            backend=kind,
            engine=engine,
            hf_repo=block["hf_repo"],
            revision=block["revision"],
            memory_bytes_estimate=block["memory_bytes_estimate"],
            engine_args=tuple(engine_args),
            context_default=backend_context,
            memory=terms,
            file=block.get("file"),
            mmproj=block.get("mmproj"),
            serves=serves_here,
        )

    return ModelManifest(
        id=model_id,
        family=model["family"],
        params_b=model["params_b"],
        context_default=model["context_default"],
        trained_context=model["trained_context"],
        modalities=tuple(modalities),
        backends=backends,
        path=path,
        defaults=(
            NO_DEFAULTS
            if "defaults" not in document
            else _parse_defaults(document["defaults"], path)
        ),
        display=model.get("display"),
        description=model.get("description"),
        local=(
            None
            if "local" not in document
            else _parse_local(document["local"], path, tuple(modalities))
        ),
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
