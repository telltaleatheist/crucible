"""What this server can actually do, decided from the card it is installed on.

PHASE9-CAPABILITY.md is the ruling. Owen, 2026-09-13: *"the extreme/moderate/fast
settings are irrelevant now. it uses what is available on the system. the user
doesnt set those. crucible does."* This module is the thing that does.

The shape of the decision
-------------------------
A client asks for a **capability class** — "a translate-class model", "a voice" —
and this server answers with the concrete thing it has, or refuses with the number
that stopped it. Section 1 of the phase doc splits the ownership and it is the
reason this file exists at all:

    which FAMILY a task needs        the client   a quality requirement about the work
    which QUANTIZATION runs here     Crucible     how the engine realizes the model
    none of them fits                Crucible     the class is disabled, with the number

So every class is "binary in its family, graded in its quantization". `tts` looks
binary only because Higgs has exactly one quantization — Owen: *"higgs is tied to a
certain size. we cant (or wont) quantize that. if higgs doesnt fit in a card that
crucible is installed on, it's disabled on that gpu."* It is the same rule with a
family of one, and it needs no special case here.

Four decisions this file makes, and why each one is the way it is
-----------------------------------------------------------------
**1. Candidates are ordered by DECLARED SIZE, descending, and the first that fits
wins.** Not by a `precision` field in the manifests, because there is no such
field and adding one would give "how heavily is this quantized" a SECOND owner
alongside `memory_bytes_estimate`, which already answers it (ARCHITECTURE.md R1).
Within one family on one backend the estimate is dominated by the weights, so
bigger *is* less quantized: `qwen3.8-27b` at 56.4 GB against `qwen3.8-27b-4bit` at
21.6 GB on cuda-linux, 55.5 against 33.9 on mlx-darwin. The order comes out right
on both without anybody maintaining a table.

Best-first, never smallest-that-fits. A 4-bit translation is a worse translation
than a bf16 one, and this rule is about the best output the host can hold, not the
most that can be crammed onto it.

**2. The bar is TOTAL memory, not free memory.** A capability is a fact about the
host; free VRAM is a fact about this second. `crucible/accelerator.py` already owns
"is there room right now" and refuses `insufficient_memory` at load time with the
measured figure. If this module also selected on `free_bytes`, a browser open
during `crucible install` would permanently disable TTS on a 24 GB card — a
capability decided by a transient. Total comes from `backend.detect_backend()`,
which is the one owner of "how big is this card" (`nvidia-smi memory.total` on
cuda-linux, `hw.memsize` on mlx-darwin — on Apple Silicon the pool and the machine
are the same thing).

**3. There is no `margin` term, and its absence is the ruling rather than an
omission.** Section 1.3 of the phase doc writes the test as
`estimate + margin <= total - desktop_allowance`, and then section 1.2 answers what
`margin` is: it is `desktop_allowance_bytes`. Owen, 2026-09-13: *"I don't think we
need to measure estimates. I've been using this system the way it is for months and
it works fine. Use the current settings for each."* Adding a second reserve on top
of the first would be a number nobody has measured, invented to feel safe, and it
would disable translate on the 3090 Ti that has been translating for months —
exactly the failure section 1.1 records, where a rule nobody had run against a
known-good answer disagreed with the operator and the rule was the thing that was
wrong.

**4. The two known-good answers this rule is checked against.** Both live in
`tests/test_capability.py` and both predate the rule:

    3090 Ti, 24.0 GiB, 3.0 GiB reserve   -> translate ENABLED on qwen3.8-27b-4bit
                                            (20.1 GiB needed, 21.0 GiB available)
    64 GiB Studio, 25% reserve           -> translate ENABLED on qwen3.8-27b-4bit,
                                            bf16 REFUSED (51.7 GiB against 48.0)

Owen runs both of those today. A selection rule that has never been run against an
answer somebody already knows is a rule nobody has tested.

Units
-----
Every figure this module prints is GiB, because its messages sit beside
`crucible/accelerator.py`'s refusals about the same bytes and a reader comparing
"needs 19.0 GB" with "needs 17.7 GiB" for one number has been given a puzzle
instead of an answer. The phase doc's tables are in GB; the bytes are identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .alignmodels import load_all_align_manifests
from .asrmodels import load_all_asr_manifests
from .backend import CUDA_LINUX, LLAMA_WINDOWS, MLX_DARWIN
from .config import CapabilityRecord, CapabilityRow
from .decide import UNSTATED_ENGINE_CONCURRENCY
from .denoisemodels import load_all_denoise_manifests
from .errors import ApiError
from .manifests import BACKEND_ENGINES, MemoryTerms, load_all_manifests
from .pages import PAGE_CONCURRENCY
from .rvcmodels import load_all_rvc_manifests
from .voices import load_all_voices

GIB = 1024 ** 3


def _gib(value: int) -> str:
    return f"{value / GIB:.1f} GiB"


#: What the pool is called on each backend, because "card" is a lie on a Mac: the
#: model, the compositor and every open app allocate from the same unified memory.
POOL_NAME: dict[str, str] = {
    CUDA_LINUX: "card",
    MLX_DARWIN: "unified memory",
    LLAMA_WINDOWS: "card",
}

#: What `llama-windows` is measuring when there is no card: the machine's RAM,
#: which is where a GGUF on the CPU allocates from. Keyed on the GPU VENDOR
#: rather than on the backend, because one backend kind serves both machines
#: and the pool is the only thing that differs (PHASE15-HOST.md section 3.10:
#: *"no NVIDIA → enabled: true with the reason 'cpu build — slow'"*).
CPU_VENDOR = "cpu"
CPU_POOL_NAME = "system memory"

#: The job types that need the WSL2 engine and will never run natively on
#: Windows: they are PYTHON environments (narrator, whisper, the aligner, urvc,
#: the separator), not llama.cpp. PHASE15-HOST.md section 3.3 gives all five
#: ONE sentence, so an app shows it once instead of five times.
WSL_ONLY_JOB_TYPES: frozenset[str] = frozenset(
    {"tts", "asr", "align", "rvc", "denoise"}
)

NEEDS_WSL_REASON = (
    "this job type needs the WSL2 engine (vLLM/SGLang); install it from the "
    "console"
)

#: What a class answers on a Windows box with no NVIDIA card. Owen: *"a
#: crucible server will run on absolutely anything."* Nothing refuses it — the
#: sentence is the warning, and the row stays enabled.
CPU_BUILD_REASON = (
    "cpu build — slow; the model runs on this machine's CPU"
)

#: Said in front of a class's LOCAL sentence when the operator has routed it
#: upstream. The local answer is kept whole after it (section 3.3), so routing
#: back loses nothing and a reader can see what this host would do on its own.
LOCAL_ANSWER_PREFIX = "the local answer would be: "

#: Said at the end of a REFUSAL for a class that could have been routed
#: (docs/MODEL-CHOICE.md section 5). Owen, 2026-09-16: *"if nothing fits their
#: card, it should give them the option of using api keys for claude or
#: openai."*
#:
#: It goes on the refusal rather than in an app's own copy for the reason every
#: other number in these sentences is here: the server is the thing that knows
#: this class CAN be routed, and an app that hard-coded the offer would show it
#: beside `pages` — which is deliberately not routable, because sending page
#: images to Anthropic is a different feature with a different body that nobody
#: has asked for. A class that grew or lost `routable` would then be wrong in
#: two repos at once.
UPSTREAM_OFFER = (
    " This class can run somewhere else instead: add an API key for Anthropic or "
    "OpenAI in settings and this host will route it rather than refuse it."
)


@dataclass(frozen=True)
class WorkingContext:
    """How much context a CLASS actually uses, and how much of it at once.

    docs/FITS-AND-THE-CARD.md section 3. The free variable in

        engine_total = weights + overhead + kv_bytes_per_token x context x concurrency

    is the WORK, and the work belongs to the class rather than to the model. A
    model's `context_default` is what the weights are FOR; it is not what
    translate sends, and using it to decide whether translate can run is how a
    27B gets refused on a 24 GB card for a 98304-token KV cache that a
    paragraph-at-a-time act was never going to fill.

    `source` is required and is prose: a context declared here with no stated
    origin is the 12288 that turned out to be BookForge's 32B tier reaching a 9B
    and took a day to disprove.
    """

    tokens: int
    concurrency: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "concurrency": self.concurrency,
            "source": self.source,
        }


@dataclass(frozen=True)
class Candidate:
    """One concrete thing that could satisfy a class on one backend."""

    id: str
    memory_bytes_estimate: int
    #: This candidate's estimate taken apart, where its backend block has been
    #: taken apart. None means the collapsed number is all there is, and
    #: `need_bytes` below then answers with it whatever the class asks — which
    #: is the behaviour every class had before the split, kept exactly, so a
    #: block without terms decides today what it decided yesterday.
    memory: "MemoryTerms | None" = None
    #: The LARGEST context an engine serving this candidate may be started
    #: with on this backend: `ModelManifest.max_context_for(backend)` — the
    #: block's `max_context` (2026-09-23), or its `context_default` where it
    #: states none. A `load-model` with `params.context` may start the engine
    #: anywhere up to it (vLLM's `--max-model-len`, llama-server's `-c`, the
    #: resident row's `max_model_len`); a load that states none starts at
    #: `context_for`. Read here rather than restated, so the ceiling below, the
    #: load check and `/v1/models` cannot disagree. None for a candidate from a
    #: catalog that is not measured in tokens (voices, aligners, RVC, ASR,
    #: denoise).
    served_context: int | None = None

    @classmethod
    def of(cls, manifest: Any, backend_kind: str) -> "Candidate":
        """One manifest's candidate on one backend — the ONE construction.

        Used by every class's walk (`CatalogCandidates`) and by the
        `load-model` context check (`check_load_context`), so the ceiling a
        client reads on `GET /v1/capability` and the ceiling a load is refused
        against are computed from the same object by the same method.
        """
        return cls(
            id=manifest.id,
            memory_bytes_estimate=manifest.spec(backend_kind).memory_bytes_estimate,
            # ONLY THE MODEL CATALOG HAS TERMS, and that is not a gap to be
            # filled later: the classes also read the voice, RVC, aligner, ASR
            # and denoise catalogs, whose specs have no KV term because their
            # work is not measured in tokens. A voice is one engine holding one
            # reservation whatever the sentence is. `getattr` here is asking
            # WHICH CATALOG this is, not papering over a missing attribute — the
            # classes that read those catalogs declare no `work` either, so
            # `need_bytes` answers with the collapsed estimate and the two facts
            # agree.
            memory=getattr(manifest.spec(backend_kind), "memory", None),
            # The same question of the same catalog: only a MODEL manifest
            # serves a context, and it answers through `max_context_for`, the
            # one owner of how far a load may raise it.
            served_context=(
                manifest.max_context_for(backend_kind)
                if hasattr(manifest, "max_context_for")
                else None
            ),
        )

    def context_ceiling(
        self, available_bytes: int, concurrency: int
    ) -> "ContextCeiling | None":
        """The longest request this candidate can serve on this host, or None.

        The SMALLER of two halves Crucible already owns, and nothing typed:

            served   the most an engine is ever started with here
                     (`served_context`: the block's `max_context`, a number
                     computed so it will not page, thrash or OOM on the
                     reference host; its `context_default` where it states
                     none)
            memory   what this host's memory affords at this concurrency
                     (`MemoryTerms.max_context`, docs/FITS-AND-THE-CARD.md 6.3)

        On a Mac the memory half is large and the served half usually binds; on
        a 24 GB card it can go either way, and the answer says which one did.
        None for a candidate that is not token-shaped at all.

        THE ONE CEILING FUNCTION. `GET /v1/capability`'s `context_ceilings`,
        its `context_over_limit`, and the `load-model` job's check of
        `params.context` (`check_load_context`, at one in flight) all call this
        and nothing else computes a ceiling. The budget is `available_bytes`
        (the pool less the desktop allowance), the same one every fit in this
        module uses. The memory half counts KV once only if `overhead_bytes`
        holds none, and the parser CANNOT tell: `qwen3.8-27b-8bit`'s Mac
        overhead was a measured peak that already contained a 98_220-token
        run's KV, its estimate carried the same double count, so the two agreed
        and the ceiling read ~64k until 2026-09-23. An overhead taken from a
        peak must have that peak's KV subtracted where it is written.
        """
        if self.served_context is None:
            return None
        memory = (
            None
            if self.memory is None
            else self.memory.max_context(
                available_bytes=available_bytes, concurrency=concurrency
            )
        )
        if memory is None or self.served_context <= memory:
            ceiling, bound_by = self.served_context, "served"
        else:
            ceiling, bound_by = memory, "memory"
        return ContextCeiling(
            model=self.id,
            tokens=ceiling,
            bound_by=bound_by,
            served_context=self.served_context,
            memory_context=memory,
            concurrency=concurrency,
        )

    def need_bytes(self, work: "WorkingContext | None") -> int:
        """What this candidate costs doing THAT work.

        The one place the reframe is spent. Everything else — the walk, the
        refusals, the ordering — asks this and does not know whether the answer
        came from arithmetic or from a stored number.
        """
        if work is None or self.memory is None:
            return self.memory_bytes_estimate
        return self.memory.bytes_for(
            context=work.tokens, concurrency=work.concurrency
        )

    def max_context(self, available_bytes: int, work: "WorkingContext | None") -> int | None:
        """The tallest context this candidate affords here, or None if unknown.

        None rather than a guess: a block with no terms cannot say what it would
        cost at another length, and answering with `context_default` would be
        this server stating a ceiling it has no arithmetic for — the thing
        section 6.2 refuses Ollama for doing.
        """
        if self.memory is None:
            return None
        concurrency = 1 if work is None else work.concurrency
        return self.memory.max_context(
            available_bytes=available_bytes, concurrency=concurrency
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "memory": None if self.memory is None else self.memory.to_dict(),
            "served_context": self.served_context,
        }


@dataclass(frozen=True)
class ContextCeiling:
    """One candidate's context ceiling on one host, with both halves shown.

    Published so an app can read the ceiling BEFORE it asks, and so the refusal
    for asking above it (`context_over_limit`) can name where each half came
    from. `memory_context` is None where the backend block has not been taken
    apart into terms — the served half is then all that is known, and the row
    says so rather than inventing a memory figure.
    """

    model: str
    tokens: int
    #: "served" or "memory": which half is the smaller one.
    bound_by: str
    served_context: int
    memory_context: int | None
    concurrency: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tokens": self.tokens,
            "bound_by": self.bound_by,
            "served_context": self.served_context,
            "served_context_source": (
                "the most this backend ever starts an engine with: the model "
                "manifest's max_context for this backend, or its "
                "context_default where it states none "
                "(manifest.max_context_for; a load-model's params.context may "
                "raise --max-model-len / -c / max_model_len up to it)"
            ),
            "memory_context": self.memory_context,
            "memory_context_source": (
                None
                if self.memory_context is None
                else (
                    "this host's available bytes less the model's weights and "
                    f"overhead, over its KV bytes per token x {self.concurrency} "
                    "in flight (MemoryTerms.max_context)"
                )
            ),
            "concurrency": self.concurrency,
        }


@dataclass(frozen=True)
class CatalogCandidates:
    """One catalog, read into candidates for a backend, best-first.

    `families` filters `models/` down to the model families a class may run —
    the `[model] family` key already in the manifests (`qwen3.5`, `qwen3.8`,
    `dots`), so the class table below names a fact the repo already states rather
    than listing model ids that would go stale the day a variant is added.

    SEVERAL rather than one since 2026-09-16 (docs/MODEL-CHOICE.md). Translate
    used to read `qwen3.8` alone, which made the 27B a floor — Owen's ruling of
    2026-09-13, *"if 27b doesnt fit on the card then it cant translate"* —
    and he has withdrawn it: *"they cant pick smaller than 9b… i think 9b could
    do an ok job at translation."* A class's floor is now the smallest family it
    lists, and nothing but this tuple has to change to move it.

    A class with fields rather than a closure, because `classes_for_model` below
    has to ask a class WHICH catalog it reads — the lineup Foundry vendors lists
    model manifests only, and must not load the voice catalog to find out that
    `tts` never names a model. A closure cannot be asked; a dataclass can.
    """

    load: Callable[[], dict[str, Any]]
    families: tuple[str, ...] | None = None
    #: THE SIZE FLOOR, read against each manifest's `[model] params_b`, or None
    #: for a class with none. Explicit since 2026-09-23 (docs/MODEL-CHOICE.md,
    #: addendum). Until then the floor was IMPLICIT in `families`: "the smallest
    #: family a class lists" was the 9B because the 9B was the smallest model the
    #: catalog shipped. The day `qwen3.5-4b` and `qwen3.5-0.8b` joined it for
    #: the decision door, a family filter alone would have put a 4B under
    #: `clean` and under `translate` — against Owen's *"they cant pick smaller
    #: than 9b"* — without anybody deciding it. A floor that is a side effect of
    #: what happens to be in `models/` is a floor nothing owns.
    min_params_b: float | None = None
    #: WHETHER A MODEL THAT SHARES ANOTHER'S WEIGHTS IS A CANDIDATE HERE
    #: (`[model] weights_of`, PHASE22-DECIDE.md section 2.9). An alias is the
    #: same weights served another way — `qwen3.5-9b-vl` is `qwen3.5-9b` with
    #: its vision tower loaded and an image reserved for — so it is always the
    #: DEARER of the two, and a best-first walk would put it ahead of its base
    #: for work that never sends an image: clean on the 9B-vl, paying a tower
    #: and a 1.90 GiB reserve for nothing. Off unless a class says it wants the
    #: served form an alias exists for; `decide` does, because a decision may
    #: carry images (section 2.7).
    aliases: bool = False

    def __call__(self, backend_kind: str) -> tuple[Candidate, ...]:
        found: list[Candidate] = []
        for manifest in self.load().values():
            if self.families is not None and manifest.family not in self.families:
                continue
            if self.min_params_b is not None and manifest.params_b < self.min_params_b:
                continue
            # `getattr` asks which catalog this is: only `models/` manifests
            # can be aliases, and the voice catalog this class also reads has
            # no such key.
            if not self.aliases and getattr(manifest, "weights_of", None) is not None:
                continue
            if not manifest.supports(backend_kind):
                continue
            found.append(Candidate.of(manifest, backend_kind))
        # Descending by size, then by id. The id is not decoration: every voice in
        # the catalog declares the SAME estimate (Higgs is one engine holding one
        # reservation whatever weights it was started on), and every RVC model
        # declares the same 2.5 GiB, so without a second key the "selected" id
        # would change with dict ordering and two runs on one host would disagree.
        found.sort(key=lambda c: (-c.memory_bytes_estimate, c.id))
        return tuple(found)


def _from_catalog(
    load: Callable[[], dict[str, Any]],
    *families: str,
    min_params_b: float | None = None,
    aliases: bool = False,
) -> CatalogCandidates:
    return CatalogCandidates(load, families or None, min_params_b, aliases)


@dataclass(frozen=True)
class CapabilityClass:
    """One thing a client can ask this server for, and what satisfies it."""

    #: The name a client asks by, and the key this class is recorded under.
    name: str
    #: Which `[jobs] enable_*` flag this class contributes to. A job type is
    #: enabled when ANY of its classes is, which is why `llm` needs three classes:
    #: a host that can clean and cannot translate is a real and common host.
    job_type: str
    #: What the class is for, in a sentence, for `crucible capability`'s output.
    purpose: str
    #: THE SAME THING, TO SOMEBODY WHO IS NOT AN OPERATOR: a bare verb phrase,
    #: no jargon, no door names, no backend names. `purpose` is the operator's
    #: half and keeps its internals — `crucible doctor` and the operator page
    #: SHOULD say "(the VLM door)" and "with a mlx-darwin block", because an
    #: operator is the person who can act on those.
    #:
    #: Added 2026-09-20 after a `pages` refusal reached a user as *"…cannot
    #: pages: disabled: reading page images (the VLM door) needs page readers,
    #: and this build ships none with a mlx-darwin block"*. Owen: *"it looks
    #: like an error."* It was a correct answer written for the wrong reader.
    #: Rather than strip the diagnosis — which an operator needs — the decision
    #: now carries BOTH, and each reader takes its own.
    plainly: str
    #: The noun for the things that satisfy it, so a refusal reads like English.
    noun: str
    #: Every candidate on a backend, best-first. None for a class that never
    #: touches the accelerator.
    candidates: Callable[[str], tuple[Candidate, ...]] | None
    #: Appended to a refusal when nothing fits. This is where a BINARY ruling gets
    #: said out loud, so the operator is not left looking for a smaller variant
    #: that was never going to exist.
    binary_note: str = ""
    #: May this class's work run somewhere other than this card
    #: (PHASE15-HOST.md section 1)? True for exactly the five chat-shaped `llm`
    #: classes. **Declared here rather than derived from `job_type == "llm"`**,
    #: because `pages` is an `llm` job type and is NOT one of the four: it sends
    #: page IMAGES to a vision model, and "forward it to Anthropic" is a
    #: different feature with a different body that nobody has asked for (and
    #: `decide` is not either: no upstream returns the logprobs it reads). A
    #: derivation would have made the two indistinguishable and routed the VLM
    #: the first time somebody typed the wrong class name.
    routable: bool = False
    #: The context this class's work actually uses, and how much of it at once.
    #: None for a class whose candidates are not context-shaped at all — a voice,
    #: an aligner, an RVC model — where there is no KV term to scale and the
    #: collapsed estimate IS the answer.
    work: "WorkingContext | None" = None
    #: MAY A CLIENT STATE THIS CLASS'S WORKING CONTEXT (`GET /v1/capability`'s
    #: `?context_tokens=` and `?concurrency=`)? True for `generate` alone, and
    #: declared rather than true of every class, for the reason `work` has a
    #: `source`: every other class's context is a RULING about its act —
    #: translate's paragraph at a time is Owen's, pages' 32768 is the page
    #: reader's — and a client that could restate it would be overruling that
    #: ruling for everybody who reads the same row. `generate` has no act of its
    #: own to rule on: its request sizes are the app's, and only the app knows
    #: them (Owen, 2026-09-23: *"make it one class and give it the ability to
    #: set the context limit"*). A class that is not client-sized refuses the
    #: parameters by name rather than ignoring them.
    #:
    #: A client-sized class is also the one whose fit checks the SERVED context
    #: (`manifest.max_context_for`: the most a load may start the engine's
    #: `--max-model-len` / `-c` at) as well as the memory: a client may ask for
    #: more than any engine here is ever started with, and a fit that said yes
    #: to a request no engine can take would be a lie.
    #: The other classes' contexts are rulings sized under what their models
    #: serve, and keep the memory-only fit they have always had.
    client_sized: bool = False

    @property
    def min_params_b(self) -> float | None:
        """This class's size floor in billions of parameters, or None.

        Read off the candidate source, which is the thing that applies it, so
        the class cannot state one floor and select on another (ARCHITECTURE.md
        R1). None for a class with no floor — `decide`, whose whole point is
        that a 0.8B can answer it — and for every class that reads a catalog
        other than `models/`.
        """
        if isinstance(self.candidates, CatalogCandidates):
            return self.candidates.min_params_b
        return None


#: THE 9B FLOOR, in one place. docs/MODEL-CHOICE.md section 1, Owen 2026-09-16:
#: *"they cant pick smaller than 9b"* — said of translate and simplify, and
#: carried to analysis (the same acts, "/etc") and to clean (the 9B was always
#: its model; "9B-class" is its purpose). Compared against `[model] params_b`.
NINE_B_FLOOR = 9

#: What a DECISION holds on the card while it is answered (PHASE22-DECIDE.md
#: section 2.9). Its state is a group of blocks or a page, not a book — Foundry's
#: Categorize tile sends about 24 blocks with 12 of context each side — and 8192
#: is the context the 0.8B was measured serving decisions at on 2026-09-23
#: (`max_model_len 8192`, PHASE22 section 8a).
DECIDE_STATE_TOKENS = 8192

#: `generate`'s working context when the client states none. Owen, 2026-09-23:
#: *"context limit can be set to 8k tokens by default, and it can request
#: higher."* It is the DEFAULT, not a ceiling: the ceiling is per host and per
#: model (`Candidate.context_ceiling`).
GENERATE_DEFAULT_TOKENS = 8192


#: Every capability class this build knows, in report order.
#:
#: `llm` is three classes and not one because the ruling is about the work, not
#: the job type. Owen, 2026-09-13: *"translation is binary per server as well. it
#: should use a 27b to translate. if 27b doesnt fit on the card then it cant
#: translate."* A 12 GB card that cleans happily and cannot translate has to be
#: able to SAY that, and `enable_llm = true` cannot say it.
CLASSES: tuple[CapabilityClass, ...] = (
    CapabilityClass(
        name="echo",
        job_type="echo",
        purpose="the test job type; it never touches the accelerator",
        plainly="run the test job",
        noun="engines",
        candidates=None,
    ),
    CapabilityClass(
        name="clean",
        job_type="llm",
        routable=True,
        # A LONGER RUN OF TEXT THAN TRANSLATE, AND STILL NOT A BOOK. Cleanup
        # reads a passage and rewrites it, so it needs enough context to keep a
        # paragraph's neighbours in view, and the two apps that run it already
        # size their requests: Foundry's `CTX_MAX` is 16384 and it sends
        # `max_model_len - (ceil(chars/2.5) + 256)`, so a full-width request is
        # the whole of the context. 8192 with two in flight is that budget spent
        # the way the lane actually spends it — Foundry keeps twelve requests
        # moving but the cleanup pass is the narrow one — and it comes to the
        # same bytes as one 16384 request, which is what this block was sized
        # against before any of this arithmetic existed.
        work=WorkingContext(
            tokens=8192,
            concurrency=2,
            source=(
                "Foundry clean/runner.ts CTX_MAX 16384 spent as two in flight; "
                "PLACEHOLDER until a cleanup run is watched"
            ),
        ),
        purpose="cleanup and the other 9B-class text work",
        plainly="clean up text",
        noun="qwen3.5 variants",
        candidates=_from_catalog(load_all_manifests, "qwen3.5", min_params_b=NINE_B_FLOOR),
        binary_note=(
            "This build ships no 4-bit 9B, and the 4B and 0.8B it does ship are "
            "below cleanup's 9B floor, so there is nothing smaller to fall back "
            "to (PHASE9-CAPABILITY.md section 1.1)."
        ),
    ),
    CapabilityClass(
        name="translate",
        job_type="llm",
        routable=True,
        # THE RULING THIS WHOLE SPLIT CAME OUT OF. A paragraph at a time, batched,
        # each block independent of the last — so the KV this act needs is
        # thousands of tokens, not the model's 98304. Four in flight because the
        # blocks are independent, which is the property that makes batching safe
        # here and does not hold for cleanup.
        work=WorkingContext(
            tokens=4096,
            concurrency=4,
            source=(
                "Owen 2026-09-16: \"translate/simplify/etc dont actually need "
                "that much kv cache because it's batched with small blocks. it "
                "isnt sending in the entire book to be translated, its only "
                "sending it in one block (roughly a paragraph) at a time. and "
                "its batched, so each block doesnt depend on the context of the "
                "one that came before it\""
            ),
        ),
        purpose="translation, which needs a 27B-class model",
        plainly="translate",
        noun="qwen3.8 and qwen3.5 variants",
        candidates=_from_catalog(
            load_all_manifests, "qwen3.8", "qwen3.5", min_params_b=NINE_B_FLOOR
        ),
        binary_note=(
            "The floor for translation is the 9B, not the 27B — so a host that "
            "cannot translate cannot hold a 9B either, and nothing smaller is "
            "coming (docs/MODEL-CHOICE.md section 1)."
        ),
    ),
    # THREE ACTS, ONE MODEL, THREE CLASSES. `simplify` and `analysis` select the
    # same 27B `translate` does and will answer identically on every card this
    # build knows — and they are still separate classes, on Owen's ruling of
    # 2026-09-13: *"they can't lie to the user and say a translate job is running
    # when it's actually a simplify job. It must accurately represent the job
    # that's running. Previously, before crucible, everything ran under
    # translate."*
    #
    # That is the naming half, and it is the half that decides this table.
    # Folding them into one class would have made a client ask about `translate`
    # in order to learn whether it may simplify, which puts the old lie back at
    # the API boundary — and naming an umbrella after one of its members is
    # exactly how "everything ran under translate" happened in the first place.
    #
    # The usual objection to a duplicated axis — that a field nothing selects on
    # differently is a field that will drift — does not hold here, twice over.
    # These are three DIFFERENT facts that share an answer today, not one fact
    # with three owners; nothing can disagree with anything. And they can already
    # be seen to diverge: `analysis` needs guided decoding (a `response_format`
    # schema) and `pages` needs vision, neither of which is a memory question, so
    # the day a backend serves the 27B without guided decoding this table has
    # somewhere to say so.
    CapabilityClass(
        name="simplify",
        job_type="llm",
        routable=True,
        # Translate's ruling names this act in the same breath, so it carries the
        # same working context rather than one reasoned separately.
        work=WorkingContext(
            tokens=4096,
            concurrency=4,
            source=(
                "Owen 2026-09-16: \"translate/simplify/etc dont actually need "
                "that much kv cache because it's batched with small blocks. it "
                "isnt sending in the entire book to be translated, its only "
                "sending it in one block (roughly a paragraph) at a time. and "
                "its batched, so each block doesnt depend on the context of the "
                "one that came before it\""
            ),
        ),
        purpose="simplification, which runs on the same 27B translation needs",
        plainly="simplify text",
        noun="qwen3.8 and qwen3.5 variants",
        candidates=_from_catalog(
            load_all_manifests, "qwen3.8", "qwen3.5", min_params_b=NINE_B_FLOOR
        ),
        binary_note=(
            "The floor for simplification is the 9B, for translation's reason: "
            "a host that cannot hold a 9B cannot do this work at all."
        ),
    ),
    CapabilityClass(
        name="analysis",
        job_type="llm",
        routable=True,
        # Translate's ruling ends "/etc", and analysis is one of the acts it
        # covers: a structured answer about a passage, not about a book.
        work=WorkingContext(
            tokens=4096,
            concurrency=4,
            source=(
                "Owen 2026-09-16: \"translate/simplify/etc dont actually need "
                "that much kv cache because it's batched with small blocks. it "
                "isnt sending in the entire book to be translated, its only "
                "sending it in one block (roughly a paragraph) at a time. and "
                "its batched, so each block doesnt depend on the context of the "
                "one that came before it\""
            ),
        ),
        purpose="structured analysis answers, on the same 27B",
        plainly="analyse text",
        noun="qwen3.8 and qwen3.5 variants",
        candidates=_from_catalog(
            load_all_manifests, "qwen3.8", "qwen3.5", min_params_b=NINE_B_FLOOR
        ),
        binary_note=(
            "The floor for analysis is the 9B, for translation's reason: a host "
            "that cannot hold a 9B cannot do this work at all."
        ),
    ),
    # THE GENERIC CHAT-SHAPED ACT. Crucible is a GPU orchestrator and knows no
    # task: it adds a verb of its own only where a task needs HANDLING of its
    # own, and everything around the call — the prompt, the fields, the
    # post-processing — belongs to the app. `generate` is open-ended text
    # generation for any app whose work is none of the acts above.
    #
    # ONE CLASS, NOT SEVERAL, on Owen's ruling of 2026-09-23, made when
    # ContentStudio's work was first proposed as a `write` and a `rewrite`:
    # *"if the only difference is the context limit then make it one class and
    # give it the ability to set the context limit"*. Two classes selecting the
    # same candidates on the same floor and differing only in how much context
    # they reserve would be a context limit written down as a vocabulary. So
    # the limit is the CLIENT'S to state (`client_sized`, and `sized_work`
    # below), and this table keeps one default for a client that states none.
    #
    # Why it is not one of the acts above, which the naming ruling needs it to
    # be able to say: `clean` is book cleanup of a passage, sized by Foundry;
    # `simplify` changes a book's reading level; `translate` changes its
    # language; `analysis` is a structured answer about a passage. A video
    # description or a set of chapter titles is none of those, and reporting it
    # under one would be "everything ran under translate" again.
    #
    # THE DEFAULT IS SMALL AND A CLIENT ASKS FOR MORE. Owen, the same day:
    # *"context limit can be set to 8k tokens by default, and it can request
    # higher … requesting higher than that throws an error back to the app
    # thats making the call"* — "that" being the host's own ceiling for the
    # model, which `context_ceiling` below computes and nothing here types.
    # ContentStudio (YouTube metadata tooling), the first measured user, is an
    # example of asking for more: it sends the whole video transcript on every
    # call, sizes one context per run in 4096-token buckets up to its
    # LOCAL_FIELD_CTX_MAX of 40960, and runs its calls serially. Its titles,
    # descriptions, chapters, summaries, scrub pass and "Soften" are examples
    # of this class, not its definition — which is why no app is named in
    # `purpose` or `plainly`.
    CapabilityClass(
        name="generate",
        job_type="llm",
        routable=True,
        client_sized=True,
        work=WorkingContext(
            tokens=GENERATE_DEFAULT_TOKENS,
            concurrency=1,
            source=(
                "Owen 2026-09-23: \"context limit can be set to 8k tokens by "
                "default, and it can request higher\"; one in flight, as its "
                "first measured user (ContentStudio, whose calls are serial) "
                "sends. A client states its own with ?context_tokens= and "
                "?concurrency= (ContentStudio asks for up to its "
                "LOCAL_FIELD_CTX_MAX, 40960)"
            ),
        ),
        purpose="open-ended text generation, on the 9B-and-up text models",
        plainly="generate text",
        noun="qwen3.8 and qwen3.5 variants",
        candidates=_from_catalog(
            load_all_manifests, "qwen3.8", "qwen3.5", min_params_b=NINE_B_FLOOR
        ),
        binary_note=(
            "The floor for generation is the 9B, for translation's reason: a "
            "host that cannot hold a 9B cannot do this work at all."
        ),
    ),
    # ONE FORWARD PASS PER QUESTION (PHASE22-DECIDE.md section 2.9). Its own
    # class rather than a ride on `analysis` (section 7.2): the act is different
    # — a distribution read off the next token, not a structured answer decoded
    # to the end — and so is the model it wants. It is the one text act a 0.8B
    # can do (measured 2026-09-23 on `Qwen/Qwen3.5-0.8B`: the worked example
    # answered sensibly, section 8a), so it has NO size floor, and every qwen3.8
    # and qwen3.5 manifest is a candidate — best-first, as every class walks,
    # so a card that holds the 27B decides on the 27B and a laptop still decides.
    CapabilityClass(
        name="decide",
        job_type="llm",
        # NOT ROUTABLE, and that is the door's own contract rather than a
        # preference: a decision reads the next-token distribution at the
        # resident model, and `POST /v1/decide` refuses an upstream id
        # `400 decide_needs_logprobs` because no upstream returns one (section
        # 2.1). A routable class would let an operator send this act to
        # Anthropic and then have every decision refused, and its refusals would
        # carry `UPSTREAM_OFFER` — advice to add an API key that cannot help.
        # `pages` is not routable for the same kind of reason.
        routable=False,
        # THE STATE PLUS THE FAN-OUT'S TAILS, NOT SIXTEEN STATES. A decision
        # sends its state once as a prime and then up to
        # `decide.UNSTATED_ENGINE_CONCURRENCY` questions that each EXTEND it
        # (section 2.5), so the questions share the state's KV through the
        # prefix cache and each adds only its own tail. On vLLM a tail costs at
        # least one attention block, measured at 544 tokens on this hybrid
        # family (section 8a): 16 x 544 = 8704 tokens, about one more state. So
        # two states' worth. Sizing it as 16 independent states would refuse the
        # 9B on the 3090 Ti's cuda-linux (24.8 GB against 22.5 GB) — the card
        # and model snap measured decisions on.
        work=WorkingContext(
            tokens=DECIDE_STATE_TOKENS,
            concurrency=2,
            source=(
                f"one {DECIDE_STATE_TOKENS}-token state (Foundry's Categorize "
                "tile: ~24 blocks with 12 of context each side) shared through "
                f"the prefix cache by up to {UNSTATED_ENGINE_CONCURRENCY} "
                "questions (decide.UNSTATED_ENGINE_CONCURRENCY), whose tails at "
                "one 544-token vLLM block each (PHASE22 section 8a) come to "
                "about one more state"
            ),
        ),
        purpose="one-forward-pass decisions (the decision door, PHASE22)",
        plainly="decide",
        noun="qwen3.8 and qwen3.5 variants",
        # ALIASES INCLUDED (section 2.9): the `-vl` forms are the only way a
        # 9B or a 27B answers a decision about an image, and they fall into
        # this list by family like every other tier. Best-first, so a card
        # that fits the vision form of a model is offered it ahead of the text
        # form — and a card that does not skips it by the same arithmetic.
        candidates=_from_catalog(
            load_all_manifests, "qwen3.8", "qwen3.5", aliases=True
        ),
    ),
    CapabilityClass(
        name="pages",
        job_type="llm",
        # THE ONE CLASS THAT REALLY WANTS THE HEIGHT, and the reason a flat
        # per-model context was never going to serve all five. A page at the
        # app's 200 dpi is about 3_450 image tokens plus the prompt plus up to
        # 8192 of answer (`models/dots-ocr.toml`), and `dots-ocr` declares 32768
        # on cuda-linux for it.
        #
        # THE WIDTH IS NOT DECLARED HERE (ledger N3, Owen's ruling 2026-09-18).
        # It used to be `concurrency=1`, on the strength of a comment saying
        # "one page at a time is what the guard and the lease already assume" —
        # and the comment was the stale copy: both apps send twelve, which is
        # the number the manifest's own KV note is written against, so the
        # arithmetic that decides whether `dots-ocr` FITS was sized for a
        # twelfth of the work that arrives. `crucible/pages.py` publishes the
        # number to clients as `pages_engine.request.concurrency`, so it is the
        # owner and this reads it; there is no second literal to drift.
        work=WorkingContext(
            tokens=32768,
            concurrency=PAGE_CONCURRENCY,
            source=(
                "models/dots-ocr.toml: context_default 32768 on cuda-linux, and "
                "its own note that one page is ~3450 image tokens plus up to "
                f"8192 of answer, at the {PAGE_CONCURRENCY} pages in flight "
                "crucible/pages.py publishes to clients"
            ),
        ),
        purpose="reading page images (the VLM door)",
        plainly="read pages",
        noun="page readers",
        candidates=_from_catalog(load_all_manifests, "dots"),
    ),
    CapabilityClass(
        name="tts",
        job_type="tts",
        purpose="narration",
        plainly="narrate",
        noun="voices",
        candidates=_from_catalog(load_all_voices),
        binary_note=(
            "Higgs v3 is not quantized and will not be, so this is not a tuning "
            "choice — the engine is disabled on this accelerator."
        ),
    ),
    CapabilityClass(
        name="asr",
        job_type="asr",
        purpose="transcription",
        plainly="transcribe",
        noun="transcribers",
        # THE THREE, BEST-FIRST, ON BOTH BACKENDS (Owen, 2026-09-24): the asr
        # job offers exactly `qwen3-asr-1.7b`, `whisper-large-v3-turbo` and
        # `whisper-tiny`, each one id across cuda-linux and mlx-darwin, and the
        # caller picks. When it does not, this walk picks, and the order is
        # QWEN, TURBO, TINY: Qwen first by Owen's "we're fully switching over
        # to qwen for transcribing", turbo as the whisper that keeps large-v3's
        # ear, tiny last as the rough pass. That is the catalog's own
        # size-descending order on both machines (PC 11.0 / 3.2 / 1.7 GB, Mac
        # 7.7 / 2.7 / 0.5 GB), so no second ordering is declared here to
        # drift from it; `tests/test_asr_lineup.py` holds the order, and a
        # measurement that ever reordered the sizes turns that test red rather
        # than quietly promoting a whisper.
        candidates=_from_catalog(load_all_asr_manifests),
    ),
    CapabilityClass(
        name="align",
        job_type="align",
        purpose="forced alignment",
        plainly="align audio to text",
        noun="aligners",
        candidates=_from_catalog(load_all_align_manifests),
    ),
    CapabilityClass(
        name="rvc",
        job_type="rvc",
        purpose="voice conversion",
        plainly="convert a voice",
        noun="RVC models",
        candidates=_from_catalog(load_all_rvc_manifests),
    ),
    # Its own class even though it shares the rvc ENV, because a class is about
    # what the CARD can hold and the two hold different things: a 913 MB
    # separator and a 2.5 GiB urvc stack are different arithmetic, and a host
    # that can denoise and cannot convert is a host that should say so.
    CapabilityClass(
        name="denoise",
        job_type="denoise",
        purpose="noise removal and stem separation",
        plainly="remove noise or split stems",
        noun="separator models",
        candidates=_from_catalog(load_all_denoise_manifests),
    ),
)

#: By name, for a lookup that refuses rather than returns None on a typo.
BY_NAME: dict[str, CapabilityClass] = {entry.name: entry for entry in CLASSES}

#: The classes a route may name, in report order. Read off the table's own
#: `routable` field, so `[routes]`, `PUT /v1/settings` and the operator page
#: all ask ONE thing which classes those are (ARCHITECTURE.md R1). The day
#: `generate` became routable (the fifth), the flag moved and every door
#: followed.
ROUTABLE_CLASSES: tuple[str, ...] = tuple(
    entry.name for entry in CLASSES if entry.routable
)

#: The classes an APP may choose a local model for, in report order: every one
#: with candidates to choose between. Read off the same table as
#: `ROUTABLE_CLASSES` and for the same reason — `[local_models]`,
#: `PUT /v1/settings` and the settings document must ask ONE thing which
#: classes those are (ARCHITECTURE.md R1). `echo` is absent because it has no
#: candidates, not because it was left out by hand.
SELECTABLE_CLASSES: tuple[str, ...] = tuple(
    entry.name for entry in CLASSES if entry.candidates is not None
)


def classes_for_job_type(job_type: str) -> tuple[CapabilityClass, ...]:
    """Every class whose verdict feeds one `enable_*` flag."""
    return tuple(entry for entry in CLASSES if entry.job_type == job_type)


def classes_for_model(model_id: str) -> tuple[str, ...]:
    """Every class a MODEL manifest can satisfy, on any backend, in report order.

    Read off the class table and not off the manifest: which classes a model
    serves is decided by `CLASSES` — its family filter, its catalog — and nowhere
    else, so the lineup Foundry vendors (`crucible/lineup.py`) asks here rather
    than carrying a `classes = [...]` key in the manifest that would be a second
    owner of the same fact (ARCHITECTURE.md R1).

    The union over backends, because a class is about what a model is FOR:
    `dots-ocr` reads pages whether or not this host is the one with its
    cuda-linux block, and the machine the lineup describes has no Crucible at all.
    Only the classes that read `models/` are walked — the voice, whisper, aligner
    and RVC catalogs are different namespaces and a model id can never appear in
    them. An id the catalog does not hold is refused by name rather than answered
    with an empty tuple that reads as "a model with no class".
    """
    catalog = load_all_manifests()
    if model_id not in catalog:
        raise ValueError(
            f"{model_id!r} is not a model in this build's catalog; it ships "
            f"{sorted(catalog)}"
        )
    names: list[str] = []
    for entry in CLASSES:
        source = entry.candidates
        if not isinstance(source, CatalogCandidates):
            continue
        if source.load is not load_all_manifests:
            continue
        served = {c.id for kind in BACKEND_ENGINES for c in source(kind)}
        if model_id in served:
            names.append(entry.name)
    return tuple(names)


@dataclass(frozen=True)
class Decision:
    """What the walk decided about one class on one host, and the numbers it used."""

    capability: str
    job_type: str
    enabled: bool
    #: The candidate that won, or "" when none did. Never None on the wire: the
    #: config's TOML has no null, and an absent key would read as "not recorded".
    selected: str
    reason: str
    #: THE SAME VERDICT FOR A PERSON WHO IS NOT AN OPERATOR, and `reason` keeps
    #: every internal it has.
    #:
    #: A BARE PHRASE WITH NO SUBJECT, beginning with the verb: *"cannot read
    #: pages — no page reader runs on this machine's hardware"*. The caller
    #: supplies the subject, because the caller is the only one who knows what
    #: to call this server: it asked a particular server this question and has
    #: its name, while the walk that decides has neither. So BookForge writes
    #: `${serverName} ${summary}` and gets *"crucible@owens-mac-studio cannot
    #: read pages — …"*, and an operator page can print it alone.
    #:
    #: WHY BOTH EXIST. On 2026-09-20 a `pages` refusal reached a user as
    #: "…cannot pages: disabled: reading page images (the VLM door) needs page
    #: readers, and this build ships none with a mlx-darwin block" — a correct
    #: sentence written for the wrong reader. Stripping it would have cost the
    #: operator the only line that says WHICH backend's block is missing, so
    #: neither reader was asked to give way: `reason` diagnoses, `summary`
    #: tells somebody what they can do, and no client has to parse one to
    #: produce the other.
    summary: str
    #: How much more memory the SMALLEST candidate would have needed, or 0. This
    #: is the number that turned the class off, and it is stored as a number and
    #: not only inside the sentence, because a sentence is not load-bearing
    #: (ARCHITECTURE.md R4).
    shortfall_bytes: int
    available_bytes: int
    candidates: tuple[Candidate, ...]
    fit_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "job_type": self.job_type,
            "enabled": self.enabled,
            "selected": self.selected,
            "reason": self.reason,
            "summary": self.summary,
            "shortfall_bytes": self.shortfall_bytes,
            "available_bytes": self.available_bytes,
            "fit_count": self.fit_count,
            "candidates": [c.to_dict() for c in self.candidates],
        }

    def row(self) -> CapabilityRow:
        return CapabilityRow(
            capability=self.capability,
            enabled=self.enabled,
            selected=self.selected,
            reason=self.reason,
            summary=self.summary,
            shortfall_bytes=self.shortfall_bytes,
        )


def available_bytes(total_bytes: int, desktop_allowance_bytes: int) -> int:
    """What a job may have: the pool, less the host's own reserve.

    Never below zero. An allowance larger than the pool is a misconfiguration and
    the arithmetic would otherwise report a NEGATIVE budget, which reads as a
    number rather than as the nonsense it is — `accelerator.unattributed_bytes`
    clamps the mirror image of this for the same reason.
    """
    return max(0, total_bytes - desktop_allowance_bytes)


def pool_name(backend_kind: str, gpu_vendor: str) -> str:
    """What the pool this class is measured against is CALLED.

    Two inputs because one backend serves two machines: `llama-windows` on a
    box with an NVIDIA card is measuring the card, and on a box without one it
    is measuring the machine's RAM, which is where a GGUF on the CPU really
    allocates from. Calling both "card" would put a lie in every reason string
    on a CPU-only host, the way "card" was already a lie on a Mac.
    """
    if gpu_vendor == CPU_VENDOR:
        return CPU_POOL_NAME
    pool = POOL_NAME.get(backend_kind)
    if pool is None:
        raise ValueError(
            f"{backend_kind!r} is not a Crucible backend; the backends are "
            f"{sorted(POOL_NAME)}"
        )
    return pool


def spell_out(candidate: Candidate, work: "WorkingContext | None") -> str:
    """A candidate's need, with the terms it is made of, in one clause.

    docs/FITS-AND-THE-CARD.md section 3: `fits` stops being a stored boolean and
    becomes arithmetic that can STATE ITSELF. A refusal that says only "it needs
    19.4 GiB" leaves the reader to guess which of the four terms is the one they
    could do something about; this one names all four, and the two that belong to
    the WORK are the two an app can change.

    Falls back to the bare figure where the backend block has not been taken
    apart — the sentence every class had before, unchanged, rather than a
    breakdown invented to fill the shape.
    """
    need = candidate.need_bytes(work)
    if work is None or candidate.memory is None:
        return _gib(need)
    terms = candidate.memory
    kv = terms.kv_bytes_per_token * work.tokens * work.concurrency
    return (
        f"{_gib(need)} — {_gib(terms.weights_bytes)} weights + "
        f"{_gib(terms.overhead_bytes)} overhead + {_gib(kv)} KV for "
        f"{work.tokens} tokens x {work.concurrency} in flight"
    )


def _over_served(
    entry: CapabilityClass, candidate: Candidate, work: "WorkingContext | None"
) -> bool:
    """Does this client-sized work ask for more than this backend ever serves?

    `served_context` is the block's `max_context` (`max_context_for`): the most
    a `load-model` may start the engine with here. A resident engine started
    lower is the app's to reload with `params.context`; that is not a reason
    this host cannot do the work.

    Only a `client_sized` class is checked (see the field for why), and only a
    token-shaped candidate can be: a voice has no served context.
    """
    return (
        entry.client_sized
        and work is not None
        and candidate.served_context is not None
        and work.tokens > candidate.served_context
    )


def _fits(
    entry: CapabilityClass,
    candidate: Candidate,
    work: "WorkingContext | None",
    budget: int,
) -> bool:
    """Whether one candidate can do this work on this host: memory, and for a
    client-sized class, the context its engine serves."""
    return candidate.need_bytes(work) <= budget and not _over_served(
        entry, candidate, work
    )


def decide(
    entry: CapabilityClass,
    backend_kind: str,
    *,
    total_bytes: int,
    desktop_allowance_bytes: int,
    gpu_vendor: str,
    chosen: str | None,
    work: "WorkingContext | None" = None,
) -> Decision:
    """Walk one class's candidates best-first and take the first that fits.

    `work` is a CLIENT-STATED working context (`sized_work`), and may only be
    given for a `client_sized` class; omitted, the class's own `work` is the
    work. The omission is the class's declared default, not a guess: every
    class states one, with its source.

    `gpu_vendor` is REQUIRED and has no default, because on `llama-windows` it
    is the difference between two true answers and there is no safe guess: a
    machine with no card told it has one would report a 24 GiB pool that does
    not exist, and one with a card told it has none would say "slow" about a
    4090. Every caller has a `Backend` in hand.
    """
    if work is None:
        work = entry.work
    elif not entry.client_sized:
        raise ValueError(
            f"{entry.name} is not client-sized; its working context is its own "
            f"ruling ({entry.work.source if entry.work else 'none'}), and a "
            "caller may not restate it"
        )
    if backend_kind == LLAMA_WINDOWS and entry.job_type in WSL_ONLY_JOB_TYPES:
        # THE FIVE PYTHON JOB TYPES, and they are off for a reason that has
        # nothing to do with the card: narrator, whisper, the aligner, urvc and
        # the separator are Python environments, and this backend is
        # llama.cpp. One sentence for all five (section 3.3), so an app shows
        # it once rather than printing five variations of "install WSL".
        return Decision(
            capability=entry.name,
            job_type=entry.job_type,
            enabled=False,
            selected="",
            reason=NEEDS_WSL_REASON,
            summary=(
                f"cannot {entry.plainly} — this machine has no Linux engine "
                "installed yet. Finish setting it up, or use another server"
            ),
            shortfall_bytes=0,
            available_bytes=available_bytes(total_bytes, desktop_allowance_bytes),
            candidates=(),
            fit_count=0,
        )
    budget = available_bytes(total_bytes, desktop_allowance_bytes)
    pool = pool_name(backend_kind, gpu_vendor)
    arithmetic = (
        f"{_gib(budget)} available ({_gib(total_bytes)} {pool} less a "
        f"{_gib(desktop_allowance_bytes)} desktop allowance)"
    )

    if entry.candidates is None:
        return Decision(
            capability=entry.name,
            job_type=entry.job_type,
            enabled=True,
            selected="",
            reason=f"always available: {entry.purpose}",
            summary=f"can {entry.plainly}",
            shortfall_bytes=0,
            available_bytes=budget,
            candidates=(),
            fit_count=0,
        )

    found = entry.candidates(backend_kind)
    if not found:
        return Decision(
            capability=entry.name,
            job_type=entry.job_type,
            enabled=False,
            selected="",
            reason=(
                f"disabled: {entry.purpose} needs {entry.noun}, and this build "
                f"ships none with a {backend_kind} block"
            ),
            summary=(
                f"cannot {entry.plainly} — nothing that can do it runs on this "
                "machine's hardware. Another server has to take this work"
            ),
            shortfall_bytes=0,
            available_bytes=budget,
            candidates=(),
            fit_count=0,
        )

    # ON A CARDLESS WINDOWS BOX THE ROW STILL LIGHTS, and says what it will
    # cost. Owen: *"a crucible server will run on absolutely anything."* The
    # sentence is the warning; the arithmetic is unchanged, because RAM is a
    # real limit and a 27B in 8 GB does not run slowly, it thrashes.
    cpu_note = (
        f" {CPU_BUILD_REASON}."
        if backend_kind == LLAMA_WINDOWS and gpu_vendor == CPU_VENDOR
        else ""
    )
    # THE ONE LINE THE REFRAME CHANGES. `need_bytes` answers the question this
    # class is actually asking — its own working context, its own concurrency —
    # instead of the question the model's `context_default` asks. On a block with
    # no terms it answers with the collapsed estimate, so nothing moves.
    fitting = [c for c in found if _fits(entry, c, work, budget)]

    if chosen is not None:
        # AN APP'S OWN CHOICE, and the reason the best-first walk below is not
        # the only way a class gets its model. INTENT.md gives the app the
        # choice of its models; what stays HERE is the arithmetic, because
        # whether a model fits is a fact about this card and no app can know it
        # from where it sits. A choice is honoured or it is refused with the
        # numbers — it is never quietly replaced by a different model, which
        # would make the settings document a suggestion.
        picked = next((c for c in found if c.id == chosen), None)
        if picked is None:
            return Decision(
                capability=entry.name,
                job_type=entry.job_type,
                enabled=False,
                selected="",
                reason=(
                    f"disabled: {chosen} was chosen for {entry.name}, and it is "
                    f"not among the {len(found)} {entry.noun} this build ships "
                    f"with a {backend_kind} block"
                ),
                summary=(
                    f"cannot {entry.plainly} — it is set to use {chosen}, which "
                    "this machine cannot run. Choose another in Settings"
                ),
                shortfall_bytes=0,
                available_bytes=budget,
                candidates=found,
                fit_count=len(fitting),
            )
        if _over_served(entry, picked, work):
            # A CLIENT-SIZED class asked for more than this model's engine is
            # started with. Memory is not the question; the engine would refuse
            # every request of this length, so the choice cannot serve it.
            return Decision(
                capability=entry.name,
                job_type=entry.job_type,
                enabled=False,
                selected="",
                reason=(
                    f"disabled: {picked.id} was chosen for {entry.name} and is "
                    f"never served past {picked.served_context} tokens on "
                    f"{backend_kind} (its manifest's max_context), which is "
                    f"less than the {work.tokens} tokens this work asks for"
                ),
                summary=(
                    f"cannot {entry.plainly} — {picked.id} cannot take requests "
                    f"of {work.tokens} tokens on this machine"
                ),
                shortfall_bytes=0,
                available_bytes=budget,
                candidates=found,
                fit_count=len(fitting),
            )
        if picked.need_bytes(work) > budget:
            # The settings door refuses a choice that does not fit, so reaching
            # here means the MACHINE changed under a choice that did fit when it
            # was made — a config carried to a smaller card, or a desktop
            # allowance raised since. Say that, rather than silently demoting to
            # something that fits and leaving an app to wonder why its model
            # never runs.
            shortfall = picked.need_bytes(work) - budget
            return Decision(
                capability=entry.name,
                job_type=entry.job_type,
                enabled=False,
                selected="",
                reason=(
                    f"disabled: {picked.id} was chosen for {entry.name} and needs "
                    f"{spell_out(picked, work)}, and there is only "
                    f"{arithmetic} — short by {_gib(shortfall)}. This choice fit "
                    f"the machine it was made on{cpu_note}."
                    + (UPSTREAM_OFFER if entry.routable else "")
                ),
                summary=(
                    f"cannot {entry.plainly} — {picked.id} needs "
                    f"{_gib(shortfall)} more memory than this machine has free. "
                    "A smaller choice, or another server"
                ),
                shortfall_bytes=shortfall,
                available_bytes=budget,
                candidates=found,
                fit_count=len(fitting),
            )
        return Decision(
            capability=entry.name,
            job_type=entry.job_type,
            enabled=True,
            selected=picked.id,
            reason=(
                f"{picked.id} was chosen for {entry.name}: it needs "
                f"{spell_out(picked, work)} and there is {arithmetic}; "
                f"{len(fitting)} of {len(found)} {entry.noun} fit{cpu_note}"
            ),
            summary=f"can {entry.plainly}, using {picked.id}",
            shortfall_bytes=0,
            available_bytes=budget,
            candidates=found,
            fit_count=len(fitting),
        )

    if fitting:
        best = fitting[0]
        return Decision(
            capability=entry.name,
            job_type=entry.job_type,
            enabled=True,
            selected=best.id,
            reason=(
                f"{best.id} fits: it needs {spell_out(best, work)} and "
                f"there is {arithmetic}; {len(fitting)} of {len(found)} "
                f"{entry.noun} fit{cpu_note}"
            ),
            summary=f"can {entry.plainly}, using {best.id}",
            shortfall_bytes=0,
            available_bytes=budget,
            candidates=found,
            fit_count=len(fitting),
        )

    smallest = found[-1]
    shortfall = smallest.need_bytes(work) - budget
    note = f" {entry.binary_note}" if entry.binary_note else ""
    return Decision(
        capability=entry.name,
        job_type=entry.job_type,
        enabled=False,
        selected="",
        reason=(
            f"disabled: the smallest of {len(found)} {entry.noun} is {smallest.id} "
            f"at {spell_out(smallest, work)} and there is only "
            f"{arithmetic} — short by {_gib(shortfall)}.{note}"
            + (UPSTREAM_OFFER if entry.routable else "")
        ),
        summary=(
            f"cannot {entry.plainly} — the smallest option needs "
            f"{_gib(shortfall)} more memory than this machine has free"
        ),
        shortfall_bytes=shortfall,
        available_bytes=budget,
        candidates=found,
        fit_count=0,
    )


def decide_all(
    backend_kind: str,
    *,
    total_bytes: int,
    desktop_allowance_bytes: int,
    gpu_vendor: str,
    chosen: Mapping[str, str],
) -> tuple[Decision, ...]:
    """Every class, decided on one host. The order of `CLASSES`.

    `chosen` is the app selections — class name to model id — and a class
    absent from it is decided best-first. It is REQUIRED and has no default
    for the reason `gpu_vendor` is: a caller that forgot it would silently
    un-choose every model an app had picked, while the config went on saying
    otherwise, and the two would disagree with nothing comparing them.
    """
    return tuple(
        decide(
            entry,
            backend_kind,
            total_bytes=total_bytes,
            desktop_allowance_bytes=desktop_allowance_bytes,
            gpu_vendor=gpu_vendor,
            chosen=chosen.get(entry.name),
        )
        for entry in CLASSES
    )


#: Where a row's working context came from, as `GET /v1/capability` echoes it.
#: Two words, so a reading is never ambiguous about whose numbers it is.
WORK_FROM_DEFAULT = "default"
WORK_FROM_REQUEST = "request"

#: The query parameters that size a client-sized class, by the name the wire
#: uses. One spelling, here, so the refusals and the route cannot disagree.
CONTEXT_TOKENS_PARAM = "context_tokens"
CONCURRENCY_PARAM = "concurrency"


def _positive_int(name: str, raw: str) -> int:
    """A query value as a positive integer, or a refusal naming it."""
    text = raw.strip()
    if not text.isdigit() or int(text) < 1:
        unit = "tokens" if name == CONTEXT_TOKENS_PARAM else "requests in flight"
        raise ApiError(
            400,
            "invalid_working_context",
            f"{name} is {raw!r}; it must be a positive whole number of {unit}",
            {"field": name, "value": raw},
        )
    return int(text)


def sized_work(
    entry: CapabilityClass,
    *,
    context_tokens: str | None,
    concurrency: str | None,
) -> "WorkingContext | None":
    """A client's stated working context for one class, or None if it stated none.

    Refused BY NAME, never ignored: `capability_not_client_sized` for a class
    whose context is a ruling rather than the app's (`CapabilityClass.
    client_sized` says why), and `invalid_working_context` for a value that is
    not a positive whole number. There is no upper bound HERE because the upper
    bound is a fact about the host and the model, not about the number: it is
    `check_ceiling`'s, and it is refused there with both halves named.

    Stating one half keeps the class's default for the other — the class's
    declared default, and the `source` says which half came from where.
    """
    if context_tokens is None and concurrency is None:
        return None
    if not entry.client_sized:
        client_sized = [c.name for c in CLASSES if c.client_sized]
        ruling = entry.work.source if entry.work is not None else "it has none"
        raise ApiError(
            400,
            "capability_not_client_sized",
            f"{entry.name}'s working context is not the client's to state: it "
            f"is a ruling about the act ({ruling}). Only {client_sized} take "
            f"{CONTEXT_TOKENS_PARAM} and {CONCURRENCY_PARAM}",
            {"capability": entry.name, "client_sized": client_sized},
        )
    default = entry.work
    if default is None:  # pragma: no cover - a client-sized class declares one
        raise ValueError(f"{entry.name} is client-sized and declares no default work")
    tokens = (
        default.tokens
        if context_tokens is None
        else _positive_int(CONTEXT_TOKENS_PARAM, context_tokens)
    )
    width = (
        default.concurrency
        if concurrency is None
        else _positive_int(CONCURRENCY_PARAM, concurrency)
    )
    stated = []
    if context_tokens is not None:
        stated.append(f"{CONTEXT_TOKENS_PARAM}={tokens}")
    if concurrency is not None:
        stated.append(f"{CONCURRENCY_PARAM}={width}")
    rest = (
        ""
        if context_tokens is not None and concurrency is not None
        else f"; the rest is the class default ({default.source})"
    )
    return WorkingContext(
        tokens=tokens,
        concurrency=width,
        source="stated by the client: " + ", ".join(stated) + rest,
    )


def context_ceilings(
    entry: CapabilityClass,
    backend_kind: str,
    *,
    available_bytes: int,
    concurrency: int,
) -> tuple[ContextCeiling, ...]:
    """Every candidate's context ceiling for this class on this host, best-first.

    Empty for a class whose candidates are not token-shaped. Published on the
    client-sized rows of `GET /v1/capability` so an app reads the ceiling
    BEFORE it asks rather than discovering it as a refusal.
    """
    if entry.candidates is None:
        return ()
    found = entry.candidates(backend_kind)
    ceilings = (c.context_ceiling(available_bytes, concurrency) for c in found)
    return tuple(ceiling for ceiling in ceilings if ceiling is not None)


def check_ceiling(
    entry: CapabilityClass,
    backend_kind: str,
    *,
    available_bytes: int,
    work: WorkingContext,
    chosen: str | None,
) -> tuple[ContextCeiling, ...]:
    """Refuse a stated context above this host's ceiling, or return the ceilings.

    Owen, 2026-09-23: *"requesting higher than that throws an error back to the
    app thats making the call"*. `context_over_limit`, 400, naming the request,
    the ceiling, the model it was computed for and where each half came from.
    NEVER CLAMPED and never answered with a smaller context: an app that asked
    for 60000 tokens and was quietly sized for 16384 would find out as a
    truncated transcript.

    WHICH MODEL'S CEILING. An app's own choice for this class, when it made
    one, because that is the model that will run; otherwise the HIGHEST
    ceiling among the candidates, because the best-first walk takes any
    candidate that serves the work and a request is only unservable when none
    of them can.

    A HOST THAT CANNOT HOLD THE WEIGHTS AT ALL IS NOT THIS REFUSAL. There the
    ceiling is 0 for every length, which is `MemoryTerms.max_context`'s own
    warning — "a different refusal from 'your request is too long' and must not
    be rounded into one". The class is then disabled on this host, which the
    row says with its shortfall, and no request length is to blame.
    """
    ceilings = context_ceilings(
        entry,
        backend_kind,
        available_bytes=available_bytes,
        concurrency=work.concurrency,
    )
    if not ceilings or entry.candidates is None:
        return ceilings
    by_model = {ceiling.model: ceiling for ceiling in ceilings}
    if chosen is not None:
        if chosen not in by_model:
            # `decide` refuses a choice this build cannot run, by name; that is
            # its sentence, not this one.
            return ceilings
        governing = by_model[chosen]
    else:
        governing = max(ceilings, key=lambda ceiling: ceiling.tokens)
    holds_weights = [
        c
        for c in entry.candidates(backend_kind)
        if c.memory is None or c.memory.fixed_bytes < available_bytes
    ]
    if not holds_weights or work.tokens <= governing.tokens:
        return ceilings
    whose = (
        " (the model chosen for this class)"
        if chosen is not None
        else " (the highest of this class's candidates on this host)"
    )
    raise _over_limit(
        f"{work.tokens} tokens x {work.concurrency} in flight is more than "
        f"{entry.name} can serve here",
        backend_kind,
        work=work,
        governing=governing,
        whose=whose,
        details={"capability": entry.name},
        ceilings=ceilings,
    )


def _over_limit(
    opening: str,
    backend_kind: str,
    *,
    work: WorkingContext,
    governing: ContextCeiling,
    whose: str,
    details: dict[str, Any],
    ceilings: tuple[ContextCeiling, ...],
) -> ApiError:
    """`400 context_over_limit`, the ONE body for a context above a ceiling.

    Built here for both doors that refuse one — `GET /v1/capability?class=`
    (`check_ceiling`) and `load-model`'s `params.context`
    (`check_load_context`) — so a client handles one shape: the request, the
    governing ceiling with both halves and their sources, and every ceiling
    that was considered. `details` adds what names the subject (`capability`,
    or `model` for a load).
    """
    memory_half = (
        f"{governing.memory_context} that this host's memory affords at "
        f"{work.concurrency} in flight"
        if governing.memory_context is not None
        else "no memory figure (this model's block is not taken apart into terms)"
    )
    return ApiError(
        400,
        "context_over_limit",
        f"{opening}: the ceiling is {governing.tokens} tokens, computed for "
        f"{governing.model}{whose} — the smaller of {governing.served_context} "
        f"served (the most its manifest ever starts an engine with on "
        f"{backend_kind}: max_context, or context_default where none is "
        f"stated) and {memory_half}. Ask for {governing.tokens} or fewer; "
        "nothing is clamped",
        {
            **details,
            "requested": {"tokens": work.tokens, "concurrency": work.concurrency},
            "ceiling": governing.to_dict(),
            "ceilings": [ceiling.to_dict() for ceiling in ceilings],
        },
    )


#: The smallest context a `load-model` may ask for. No engine this build runs
#: states a real minimum of its own (vLLM's `--max-model-len` and llama-server's
#: `-c` take any positive length), so this is a stated floor rather than a
#: measured one: below it a model cannot hold a chat template, a system prompt
#: and an answer at once, and a load there is a mistake to refuse rather than an
#: engine to start.
MIN_LOAD_CONTEXT = 2048


def check_load_context(
    manifest: Any,
    backend_kind: str,
    *,
    available_bytes: int,
    context: int,
) -> ContextCeiling:
    """Refuse a `load-model` context above this host's ceiling for that model.

    THE SAME CEILING `GET /v1/capability` publishes — `Candidate.of(...)
    .context_ceiling(...)`, one construction and one function — at ONE in
    flight, because a load's context is the longest single request the engine
    will take (vLLM's `--max-model-len`), and the `KvPlan` the load sizes
    checks exactly that: one full-context request must fit the pool. The
    refusal is the same `400 context_over_limit` body, naming the model.

    A HOST THAT CANNOT HOLD THE WEIGHTS AT ALL is not this refusal, for
    `check_ceiling`'s reason: the memory half is then 0 for every length, and
    the load is refused by the accelerator guard with the bytes instead
    (`insufficient_memory`), not blamed on the length asked for. The ceiling is
    returned either way so the caller can say what it checked.

    The FLOOR (`MIN_LOAD_CONTEXT`) is not checked here: `LoadParams.context`
    carries it as its own bound and refuses below it as `invalid_params`
    before this runs — one owner of each refusal.
    """
    candidate = Candidate.of(manifest, backend_kind)
    ceiling = candidate.context_ceiling(available_bytes, 1)
    if ceiling is None:  # pragma: no cover - every model manifest is token-shaped
        raise ValueError(f"{manifest.id} is not token-shaped")
    holds_weights = (
        candidate.memory is None or candidate.memory.fixed_bytes < available_bytes
    )
    if not holds_weights or context <= ceiling.tokens:
        return ceiling
    raise _over_limit(
        f"a context of {context} tokens is more than {manifest.id} can be "
        "loaded at here",
        backend_kind,
        work=WorkingContext(
            tokens=context, concurrency=1, source="load-model params.context"
        ),
        governing=ceiling,
        whose="",
        details={"model": manifest.id},
        ceilings=(ceiling,),
    )


def routed_row(row: CapabilityRow, model: str) -> CapabilityRow:
    """One class's row, as it reads once the operator has routed it upstream.

    PHASE15-HOST.md section 3.3. `selected` becomes the upstream model id —
    *"which is exactly the `model` an app sends to `/v1/openai/chat/completions`"*
    — and `enabled` becomes true, because `PUT /v1/settings` refuses to store a
    route whose upstream is unconfigured, so a stored route is always servable.

    **The local sentence is kept whole**, after `the local answer would be: `.
    Nothing is lost when the operator routes back: the row is rebuilt from
    `decide()` on the next write, and until then a reader can still see what
    this host would do on its own. `shortfall_bytes` is zeroed with the same
    honesty — nothing is short of anything when the work is not on this card.
    """
    return CapabilityRow(
        capability=row.capability,
        enabled=True,
        selected=model,
        reason=(
            f"routed to {model.partition('/')[0]}; "
            f"{LOCAL_ANSWER_PREFIX}{row.reason}"
        ),
        # The person-facing half says WHERE the work goes, not what this card
        # could not do — a routed class is not a refusal and must not read like
        # one.
        summary=f"sends this work to {model.partition('/')[0]}",
        shortfall_bytes=0,
    )


def record(
    backend_kind: str,
    *,
    total_bytes: int,
    desktop_allowance_bytes: int,
    decisions: tuple[Decision, ...],
    routes: dict[str, str],
) -> CapabilityRecord:  # noqa: D401 - the docstring below is the contract
    """The decisions, in the shape `config.toml` keeps them, routes applied.

    `routes` is REQUIRED and has no default, which is the whole point of it
    being a parameter. A default of `{}` would mean every caller that forgot it
    silently unrouted the server on its next capability write — and the callers
    are `crucible capability --write`, `crucible install` and
    `PUT /v1/settings`, all three of which rewrite the record of a server an
    operator may have routed hours ago (PHASE15-HOST.md section 2: capability
    is recomputed and re-written on every settings write that touches a route).
    """
    rows = []
    for decision in decisions:
        row = decision.row()
        model = routes.get(decision.capability)
        rows.append(row if model is None else routed_row(row, model))
    return CapabilityRecord(
        backend_kind=backend_kind,
        total_bytes=total_bytes,
        desktop_allowance_bytes=desktop_allowance_bytes,
        rows=tuple(rows),
    )


def served_rows(
    record: CapabilityRecord,
    *,
    gpu_vendor: str,
    chosen: Mapping[str, str],
    routes: Mapping[str, str],
    capability_class: str | None,
    context_tokens: str | None,
    concurrency: str | None,
) -> list[dict[str, Any]]:
    """`GET /v1/capability`'s rows: the record, with each row's WORK stated.

    Every row carries `work` — the working context its fit was computed for,
    and `from`, which is `"default"` (the class's own, `CLASSES`) or
    `"request"` (the client's `?context_tokens=` / `?concurrency=`) — or null
    for a class whose candidates are not token-shaped. A reading is then never
    ambiguous about whose numbers decided it. Every row also carries
    `context_ceilings`: on a client-sized row, every candidate's ceiling on
    this host at that row's concurrency, so an app reads the limit before it
    asks; null on every other row.

    THE STORED RECORD IS UNCHANGED BY A REQUEST. A sized row is decided LIVE,
    from the record's own card numbers (`backend_kind`, `total_bytes`,
    `desktop_allowance_bytes` — the inputs the record was decided on, as
    `settings.recomputed_capability` also uses them), the live selections and
    the live routes, and is answered to this caller alone. Writing it back
    would let one app's request size re-decide the class for every other app.

    Refusals, all 400 and all by name, before anything is decided:
    `capability_class_required` (a size with no `class` to apply it to),
    `unknown_capability`, `capability_not_client_sized`,
    `invalid_working_context`, and `context_over_limit` (`check_ceiling`).
    A class this RECORD predates is 503 `capability_undecided`, the route's
    own word for a record that has decided nothing about it.
    """
    sizing = context_tokens is not None or concurrency is not None
    if sizing and capability_class is None:
        raise ApiError(
            400,
            "capability_class_required",
            f"{CONTEXT_TOKENS_PARAM} and {CONCURRENCY_PARAM} size ONE class; "
            "name it with ?class=. Only "
            f"{[c.name for c in CLASSES if c.client_sized]} may be sized",
        )
    entry: CapabilityClass | None = None
    if capability_class is not None:
        entry = BY_NAME.get(capability_class)
        if entry is None:
            raise ApiError(
                400,
                "unknown_capability",
                f"{capability_class!r} is not a capability class; this build "
                f"knows {sorted(BY_NAME)}",
                {"capability": capability_class, "known": sorted(BY_NAME)},
            )
    requested = (
        None
        if entry is None
        else sized_work(entry, context_tokens=context_tokens, concurrency=concurrency)
    )
    budget = available_bytes(record.total_bytes, record.desktop_allowance_bytes)
    if entry is not None and requested is not None:
        if not any(row.capability == entry.name for row in record.rows):
            raise ApiError(
                503,
                "capability_undecided",
                f"this server's capability record predates the {entry.name!r} "
                "class and has decided nothing about it. Run `crucible "
                "capability --write` to decide it",
            )
        # Routed upstream, the work does not run on this card and this card's
        # ceiling is not the limit — the upstream's is, and it is the
        # upstream's to refuse. The local answer is still decided (below) so
        # the row keeps "the local answer would be" whole.
        if routes.get(entry.name) is None:
            check_ceiling(
                entry,
                record.backend_kind,
                available_bytes=budget,
                work=requested,
                chosen=chosen.get(entry.name),
            )

    rows: list[dict[str, Any]] = []
    for stored in record.rows:
        row = stored.to_dict()
        found = BY_NAME.get(stored.capability)
        if found is None:
            # A class a newer build wrote and this one does not know. The row
            # is the record's and is served as the record says; there is no
            # class here to state its work.
            row["work"] = None
            row["context_ceilings"] = None
            rows.append(row)
            continue
        work = found.work
        basis = WORK_FROM_DEFAULT
        if entry is not None and found.name == entry.name and requested is not None:
            work, basis = requested, WORK_FROM_REQUEST
            decision = decide(
                found,
                record.backend_kind,
                total_bytes=record.total_bytes,
                desktop_allowance_bytes=record.desktop_allowance_bytes,
                gpu_vendor=gpu_vendor,
                chosen=chosen.get(found.name),
                work=requested,
            )
            fresh = decision.row()
            model = routes.get(found.name)
            row = (fresh if model is None else routed_row(fresh, model)).to_dict()
        row["work"] = None if work is None else {**work.to_dict(), "from": basis}
        # On EVERY row, null where the class is not client-sized: one shape,
        # so a reader never has to tell "absent" from "none".
        row["context_ceilings"] = None
        if found.client_sized and work is not None:
            row["context_ceilings"] = [
                ceiling.to_dict()
                for ceiling in context_ceilings(
                    found,
                    record.backend_kind,
                    available_bytes=budget,
                    concurrency=work.concurrency,
                )
            ]
        rows.append(row)
    return rows


def job_type_enabled(job_type: str, decisions: tuple[Decision, ...]) -> bool:
    """Can this host run ANY of the classes behind one `enable_*` flag?

    The disjunction is the whole reason `llm` has three classes: a 12 GB card that
    cleans and cannot translate is an `llm` server, and says which half it has in
    the per-class rows rather than in prose.
    """
    mine = [d for d in decisions if d.job_type == job_type]
    if not mine:
        raise ValueError(
            f"no capability class feeds {job_type!r}; CLASSES covers "
            f"{sorted({entry.job_type for entry in CLASSES})}"
        )
    return any(d.enabled for d in mine)


__all__ = [
    "BY_NAME",
    "CLASSES",
    "CPU_BUILD_REASON",
    "CPU_POOL_NAME",
    "CPU_VENDOR",
    "LOCAL_ANSWER_PREFIX",
    "UPSTREAM_OFFER",
    "NEEDS_WSL_REASON",
    "WSL_ONLY_JOB_TYPES",
    "pool_name",
    "ROUTABLE_CLASSES",
    "SELECTABLE_CLASSES",
    "routed_row",
    "Candidate",
    "CapabilityClass",
    "CatalogCandidates",
    "CONCURRENCY_PARAM",
    "CONTEXT_TOKENS_PARAM",
    "ContextCeiling",
    "Decision",
    "GENERATE_DEFAULT_TOKENS",
    "WORK_FROM_DEFAULT",
    "WORK_FROM_REQUEST",
    "check_ceiling",
    "check_load_context",
    "MIN_LOAD_CONTEXT",
    "context_ceilings",
    "served_rows",
    "sized_work",
    "available_bytes",
    "classes_for_job_type",
    "classes_for_model",
    "decide",
    "decide_all",
    "job_type_enabled",
    "record",
]
