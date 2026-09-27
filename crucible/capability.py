from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .alignmodels import load_all_align_manifests
from .asrmodels import load_all_asr_manifests
from .backend import (
    BF16,
    CUDA_LINUX,
    FEATURE_FLOORS,
    LLAMA_WINDOWS,
    MEASURED_FEATURES,
    MLX_DARWIN,
    TENSOR_CORES,
    CardFacts,
    feature_floor,
    sm_name,
)
from .config import CapabilityRecord, CapabilityRow, desktop_reserve_words
from .decide import UNSTATED_ENGINE_CONCURRENCY
from .denoisemodels import load_all_denoise_manifests
from .engines import vllm as vllm_engine
from .engines.vllm import bf16_fallback, card_needs
from .errors import ApiError
from .manifests import BACKEND_ENGINES, MemoryTerms, load_all_manifests
from .pages import PAGE_CONCURRENCY
from .precision import below_floor, weight_bits
from . import asrplan, ttsplan
from .ttsplan import ServingVariant
from .precision import label as precision_label
from .rvcmodels import load_all_rvc_manifests
from .voices import load_all_voices

GIB = 1024 ** 3


def _gib(value: int) -> str:
    return f"{value / GIB:.1f} GiB"


POOL_NAME: dict[str, str] = {
    CUDA_LINUX: "card",
    MLX_DARWIN: "unified memory",
    LLAMA_WINDOWS: "card",
}

CPU_VENDOR = "cpu"
CPU_POOL_NAME = "system memory"

WSL_ONLY_JOB_TYPES: frozenset[str] = frozenset(
    {"tts", "asr", "align", "rvc", "denoise"}
)

NEEDS_WSL_REASON = (
    "this job type needs the WSL2 engine (vLLM/SGLang); install it from the "
    "console"
)

CPU_BUILD_REASON = (
    "cpu build — slow; the model runs on this machine's CPU"
)

LOCAL_ANSWER_PREFIX = "the local answer would be: "

UPSTREAM_OFFER = (
    " This class can run somewhere else instead: add an API key for Anthropic or "
    "OpenAI in settings and this host will route it rather than refuse it."
)


@dataclass(frozen=True)
class WorkingContext:
    tokens: int
    concurrency: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "concurrency": self.concurrency,
            "source": self.source,
        }


def _stated_dtype(spec: Any) -> str | None:
    if getattr(spec, "engine", None) == vllm_engine.ENGINE_NAME:
        stated = vllm_engine.stated_dtype(spec)
        return None if stated == vllm_engine.AUTO_DTYPE else stated
    return getattr(spec, "dtype", None)


@dataclass(frozen=True)
class Candidate:
    id: str
    memory_bytes_estimate: int
    memory: "MemoryTerms | None" = None
    served_context: int | None = None
    needs: tuple[str, ...] = ()
    bits: int | None = None
    dtype: str | None = None
    bf16_fallback: str | None = None
    serving: "tuple[ServingVariant, ...] | None" = None

    @classmethod
    def of(cls, manifest: Any, backend_kind: str) -> "Candidate":
        return cls(
            id=manifest.id,
            memory_bytes_estimate=manifest.spec(backend_kind).memory_bytes_estimate,
            memory=getattr(manifest.spec(backend_kind), "memory", None),
            served_context=(
                manifest.max_context_for(backend_kind)
                if hasattr(manifest, "max_context_for")
                else None
            ),
            needs=card_needs(manifest.spec(backend_kind)),
            bits=weight_bits(manifest.spec(backend_kind)),
            dtype=_stated_dtype(manifest.spec(backend_kind)),
            bf16_fallback=bf16_fallback(manifest.spec(backend_kind)),
            serving=(
                ttsplan.ladder_for(manifest, manifest.spec(backend_kind), backend_kind)
                or asrplan.ladder_for(
                    manifest, manifest.spec(backend_kind), backend_kind
                )
            ),
        )

    def serving_on(self, budget: int) -> "ServingVariant | None":
        if self.serving is None:
            return None
        return ttsplan.choose(self.serving, budget)

    def holds(self, work: "WorkingContext | None", budget: int) -> bool:
        if self.serving is not None:
            return self.serving_on(budget) is not None
        return self.need_bytes(work) <= budget

    def floor_bytes(self, work: "WorkingContext | None") -> int:
        if self.serving is not None:
            return min(v.need_bytes for v in self.serving if v.available)
        return self.need_bytes(work)

    def lacks(self, card: "CardFacts | None") -> tuple[str, ...]:
        if card is None:
            return ()
        return tuple(need for need in self.needs if card.has(need) is False)

    def run_dtype(self, card: "CardFacts | None") -> str | None:
        if (
            self.bf16_fallback is not None
            and card is not None
            and card.has(BF16) is False
        ):
            return self.bf16_fallback
        return self.dtype

    def precision_on(self, card: "CardFacts | None") -> str:
        return precision_label(self.bits, self.run_dtype(card))

    def degraded_on(self, card: "CardFacts | None") -> bool:
        return self.run_dtype(card) != self.dtype

    def context_ceiling(
        self, available_bytes: int, concurrency: int
    ) -> "ContextCeiling | None":
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
        if work is None or self.memory is None:
            return self.memory_bytes_estimate
        return self.memory.bytes_for(
            context=work.tokens, concurrency=work.concurrency
        )

    def max_context(self, available_bytes: int, work: "WorkingContext | None") -> int | None:
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
            "needs": list(self.needs),
            "bits": self.bits,
            "dtype": self.dtype,
            "bf16_fallback": self.bf16_fallback,
        }


@dataclass(frozen=True)
class ContextCeiling:
    model: str
    tokens: int
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
    load: Callable[[], dict[str, Any]]
    families: tuple[str, ...] | None = None
    min_params_b: float | None = None
    aliases: bool = False

    def __call__(self, backend_kind: str) -> tuple[Candidate, ...]:
        found: list[Candidate] = []
        for manifest in self.load().values():
            if self.families is not None and manifest.family not in self.families:
                continue
            if self.min_params_b is not None and manifest.params_b < self.min_params_b:
                continue
            if not self.aliases and getattr(manifest, "weights_of", None) is not None:
                continue
            if not manifest.supports(backend_kind):
                continue
            candidate = Candidate.of(manifest, backend_kind)
            if below_floor(candidate.bits):
                continue
            found.append(candidate)
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
    name: str
    job_type: str
    purpose: str
    plainly: str
    noun: str
    candidates: Callable[[str], tuple[Candidate, ...]] | None
    binary_note: str = ""
    routable: bool = False
    work: "WorkingContext | None" = None
    client_sized: bool = False

    @property
    def min_params_b(self) -> float | None:
        if isinstance(self.candidates, CatalogCandidates):
            return self.candidates.min_params_b
        return None


NINE_B_FLOOR = 9

DECIDE_STATE_TOKENS = 8192

GENERATE_DEFAULT_TOKENS = 8192


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
    CapabilityClass(
        name="simplify",
        job_type="llm",
        routable=True,
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
    CapabilityClass(
        name="decide",
        job_type="llm",
        routable=False,
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
        candidates=_from_catalog(
            load_all_manifests, "qwen3.8", "qwen3.5", aliases=True
        ),
    ),
    CapabilityClass(
        name="pages",
        job_type="llm",
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
    ),
    CapabilityClass(
        name="asr",
        job_type="asr",
        purpose="transcription",
        plainly="transcribe",
        noun="transcribers",
        candidates=_from_catalog(load_all_asr_manifests, aliases=True),
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
    CapabilityClass(
        name="denoise",
        job_type="denoise",
        purpose="noise removal and stem separation",
        plainly="remove noise or split stems",
        noun="separator models",
        candidates=_from_catalog(load_all_denoise_manifests),
    ),
)

BY_NAME: dict[str, CapabilityClass] = {entry.name: entry for entry in CLASSES}

ROUTABLE_CLASSES: tuple[str, ...] = tuple(
    entry.name for entry in CLASSES if entry.routable
)

SELECTABLE_CLASSES: tuple[str, ...] = tuple(
    entry.name for entry in CLASSES if entry.candidates is not None
)


def classes_for_job_type(job_type: str) -> tuple[CapabilityClass, ...]:
    return tuple(entry for entry in CLASSES if entry.job_type == job_type)


def classes_for_model(model_id: str) -> tuple[str, ...]:
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
    capability: str
    job_type: str
    enabled: bool
    selected: str
    reason: str
    summary: str
    shortfall_bytes: int
    available_bytes: int
    candidates: tuple[Candidate, ...]
    fit_count: int
    lacking_features: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "job_type": self.job_type,
            "enabled": self.enabled,
            "selected": self.selected,
            "reason": self.reason,
            "summary": self.summary,
            "shortfall_bytes": self.shortfall_bytes,
            "lacking_features": list(self.lacking_features),
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
    return max(0, total_bytes - desktop_allowance_bytes)


def pool_name(backend_kind: str, gpu_vendor: str) -> str:
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


def _feature_order(features: "set[str] | tuple[str, ...]") -> tuple[str, ...]:
    order = [name for name, _floor, _what in FEATURE_FLOORS] + list(MEASURED_FEATURES)
    return tuple(name for name in order if name in features)


def _card_words(card: "CardFacts | None") -> str:
    if card is None or card.compute_capability is None:
        return "a card whose compute capability could not be read"
    return f"{sm_name(card.compute_capability)} ({card.compute_capability})"


_MEASURED_WORDS: dict[str, str] = {
    "vllm": "vLLM to start on this card",
    "cuda_graphs": "CUDA graphs to capture on this card",
}


def _needs_phrase(features: tuple[str, ...], card: "CardFacts | None") -> str:
    parts: list[str] = []
    for feature in features:
        if feature in MEASURED_FEATURES:
            when = (
                ""
                if card is None or card.measured_at is None
                else f" on {card.measured_at}"
            )
            detail = "" if card is None else card.measured_detail.get(feature, "")
            parts.append(
                f"{_MEASURED_WORDS.get(feature, feature)}, and Crucible's own "
                f"measurement{when} found it does not"
                + (f" ({detail})" if detail else "")
            )
        else:
            parts.append(
                f"{feature} (compute capability {feature_floor(feature)} or "
                f"newer), and this card is {_card_words(card)}"
            )
    return "; ".join(parts)


def _too_old_phrase(features: tuple[str, ...], card: "CardFacts | None") -> str:
    if all(feature in MEASURED_FEATURES for feature in features):
        return (
            "Crucible tested this graphics card and the engine this needs would "
            "not start on it"
        )
    floors = ", ".join(
        f"{feature} needs {feature_floor(feature)} or newer"
        for feature in features
        if feature not in MEASURED_FEATURES
    )
    number = "unknown" if card is None else card.compute_capability
    return f"this graphics card is too old for it ({floors}; this card is {number})"


def _barred_note(barred: tuple[Candidate, ...], card: "CardFacts | None") -> str:
    if not barred:
        return ""
    missing = _feature_order({f for c in barred for f in c.lacks(card)})
    return (
        f" {len(barred)} more — {', '.join(c.id for c in barred)} — "
        f"{'needs' if len(barred) == 1 else 'need'} "
        f"{_needs_phrase(missing, card)}."
    )


def _precision_note(candidate: Candidate, card: "CardFacts | None") -> str:
    if candidate.degraded_on(card):
        return (
            f" On this card {candidate.id} runs in "
            f"{candidate.precision_on(card)}, not "
            f"{precision_label(candidate.bits, candidate.dtype)}: it needs "
            f"{_needs_phrase((BF16,), card)}."
        )
    if (
        candidate.bf16_fallback is not None
        and card is not None
        and card.has(BF16) is None
    ):
        return (
            f" {candidate.id} is started in bf16, and this card's compute "
            "capability could not be read (nvidia-smi --query-gpu=compute_cap), "
            "so whether it needs the fp16 fallback was not checked."
        )
    return ""


def _spell_floor(candidate: Candidate, work: "WorkingContext | None") -> str:
    if candidate.serving is None:
        return spell_out(candidate, work)
    floor = min(
        (v for v in candidate.serving if v.available), key=lambda v: (v.need_bytes, v.width)
    )
    return f"{_gib(floor.need_bytes)} for {floor.label()} (declared)"


def _spell_chosen(
    candidate: Candidate, work: "WorkingContext | None", budget: int
) -> str:
    chosen = candidate.serving_on(budget)
    if chosen is None or chosen == candidate.serving[0]:
        return spell_out(candidate, work)
    return f"{_gib(chosen.need_bytes)} for {chosen.label()} (declared)"


def _serving_note(candidate: Candidate, budget: int) -> str:
    if candidate.serving is None:
        return ""
    chosen = candidate.serving_on(budget)
    if chosen is None or chosen == candidate.serving[0]:
        return ""
    if asrplan.is_ladder(candidate.serving):
        return (
            f" {candidate.id} {asrplan.explain(candidate.serving, chosen, budget)}. "
            f"{asrplan.LOAD_TEST_NOTE}"
        )
    return (
        f" {candidate.id} {ttsplan.explain(candidate.serving, chosen, budget)}. "
        "SGLang-Omni's memory fraction is not rescaled for a narrower width, so "
        "this card's first load is the load test."
    )


def _explain(candidate: Candidate, chosen: ServingVariant, budget: int) -> str:
    plan = asrplan if asrplan.is_ladder(candidate.serving) else ttsplan
    return plan.explain(candidate.serving, chosen, budget)


def _skipped_ladder_note(
    usable: tuple[Candidate, ...], best: Candidate, budget: int
) -> str:
    skipped = []
    for candidate in usable:
        if candidate.id == best.id:
            break
        if asrplan.is_ladder(candidate.serving):
            skipped.append(
                f"{candidate.id} ({_gib(candidate.serving[-1].need_bytes)}, declared)"
            )
    if not skipped:
        return ""
    return (
        f" {' and '.join(skipped)} "
        f"{'does' if len(skipped) == 1 else 'do'} not fit even one piece at a "
        f"time at full precision, so {best.id} is taken: fewer pieces at once "
        "is tried before a smaller or quantized model."
    )


def _serving_summary(candidate: Candidate, budget: int) -> str:
    if candidate.serving is None:
        return ""
    chosen = candidate.serving_on(budget)
    if chosen is None or chosen == candidate.serving[0]:
        return ""
    return " — " + _explain(candidate, chosen, budget)


def _serving_refusal_summary(
    entry: CapabilityClass, candidate: Candidate, budget: int
) -> str:
    if candidate.serving is None or asrplan.is_ladder(candidate.serving):
        return ""
    would = ttsplan.first_fitting(candidate.serving, budget)
    if would is not None and not would.available:
        return (
            f"cannot {entry.plainly} — full quality does not fit this card even "
            f"one passage at a time, and the {would.bits}-bit version that would "
            "fit has not been made yet"
        )
    return (
        f"cannot {entry.plainly} — even one passage at a time at "
        f"{candidate.serving[-1].bits}-bit needs more memory than this card has"
    )


def _serving_refusal_note(candidate: Candidate, budget: int) -> str:
    if candidate.serving is None or asrplan.is_ladder(candidate.serving):
        return ""
    would = ttsplan.first_fitting(candidate.serving, budget)
    if would is not None and not would.available:
        return (
            f" {'An' if would.bits == 8 else 'A'} {would.bits}-bit version, one passage at a time "
            f"(~{would.need_bytes / 1e9:.1f} GB, declared), would fit; "
            f"{ttsplan.PENDING_NOTE}."
        )
    lowest = candidate.serving[-1]
    return (
        f" Even the {lowest.bits}-bit version one passage at a time "
        f"(~{lowest.need_bytes / 1e9:.1f} GB, declared; {ttsplan.PENDING_NOTE}) "
        "needs more than this card gives a job, and nothing under 4-bit is "
        "ever offered."
    )


def _with_notes(text: str, *notes: str) -> str:
    said = "".join(notes)
    if not said:
        return text
    return text.rstrip(".") + "." + said


def _over_served(
    entry: CapabilityClass, candidate: Candidate, work: "WorkingContext | None"
) -> bool:
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
    return candidate.holds(work, budget) and not _over_served(entry, candidate, work)


def decide(
    entry: CapabilityClass,
    backend_kind: str,
    *,
    total_bytes: int,
    desktop_allowance_bytes: int,
    gpu_vendor: str,
    chosen: str | None,
    work: "WorkingContext | None" = None,
    card: "CardFacts | None" = None,
) -> Decision:
    if work is None:
        work = entry.work
    elif not entry.client_sized:
        raise ValueError(
            f"{entry.name} is not client-sized; its working context is its own "
            f"ruling ({entry.work.source if entry.work else 'none'}), and a "
            "caller may not restate it"
        )
    if backend_kind == LLAMA_WINDOWS and entry.job_type in WSL_ONLY_JOB_TYPES:
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

    cpu_note = (
        f" {CPU_BUILD_REASON}."
        if backend_kind == LLAMA_WINDOWS and gpu_vendor == CPU_VENDOR
        else ""
    )
    usable = tuple(c for c in found if not c.lacks(card))
    barred = tuple(c for c in found if c.lacks(card))
    barred_note = _barred_note(barred, card)
    fitting = [c for c in usable if _fits(entry, c, work, budget)]

    if chosen is not None:
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
        missing = picked.lacks(card)
        if missing:
            return Decision(
                capability=entry.name,
                job_type=entry.job_type,
                enabled=False,
                selected="",
                reason=(
                    f"disabled: {picked.id} was chosen for {entry.name}; its "
                    f"engine needs {_needs_phrase(missing, card)}. "
                    "It refuses to start on this card whatever the memory."
                    + (UPSTREAM_OFFER if entry.routable else "")
                ),
                summary=(
                    f"cannot {entry.plainly} — it is set to use {picked.id}, and "
                    f"{_too_old_phrase(missing, card)}. Choose "
                    "another in Settings"
                ),
                shortfall_bytes=0,
                available_bytes=budget,
                candidates=found,
                fit_count=len(fitting),
                lacking_features=missing,
            )
        if _over_served(entry, picked, work):
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
        if not picked.holds(work, budget):
            shortfall = picked.floor_bytes(work) - budget
            return Decision(
                capability=entry.name,
                job_type=entry.job_type,
                enabled=False,
                selected="",
                reason=(
                    f"disabled: {picked.id} was chosen for {entry.name} and needs "
                    f"{_spell_floor(picked, work)}, and there is only "
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
            reason=_with_notes(
                f"{picked.id} was chosen for {entry.name}: it needs "
                f"{_spell_chosen(picked, work, budget)} and there is {arithmetic}; "
                f"{len(fitting)} of {len(found)} {entry.noun} fit{cpu_note}",
                _precision_note(picked, card),
                _serving_note(picked, budget),
            ),
            summary=f"can {entry.plainly}, using {picked.id}" + _serving_summary(picked, budget),
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
            reason=_with_notes(
                f"{best.id} fits: it needs {_spell_chosen(best, work, budget)} and "
                f"there is {arithmetic}; {len(fitting)} of {len(found)} "
                f"{entry.noun} fit{cpu_note}",
                barred_note,
                _skipped_ladder_note(usable, best, budget),
                _precision_note(best, card),
                _serving_note(best, budget),
            ),
            summary=f"can {entry.plainly}, using {best.id}" + _serving_summary(best, budget),
            shortfall_bytes=0,
            available_bytes=budget,
            candidates=found,
            fit_count=len(fitting),
        )

    if not usable:
        missing = _feature_order(
            {f for c in barred for f in c.lacks(card)}
        )
        return Decision(
            capability=entry.name,
            job_type=entry.job_type,
            enabled=False,
            selected="",
            reason=(
                "disabled: "
                + (
                    "the only candidate"
                    if len(found) == 1
                    else f"all {len(found)} candidates"
                )
                + f" this build ships for {entry.name} on {backend_kind} — "
                f"{', '.join(c.id for c in found)} — "
                f"{'needs' if len(found) == 1 else 'need'} "
                f"{_needs_phrase(missing, card)}. The engine "
                "refuses to start on this card whatever the memory, so this "
                "is not a sizing choice."
                + (UPSTREAM_OFFER if entry.routable else "")
            ),
            summary=(
                f"cannot {entry.plainly} — "
                f"{_too_old_phrase(missing, card)}"
            ),
            shortfall_bytes=0,
            available_bytes=budget,
            candidates=found,
            fit_count=0,
            lacking_features=missing,
        )

    smallest = usable[-1]
    shortfall = smallest.floor_bytes(work) - budget
    note = f" {entry.binary_note}" if entry.binary_note else ""
    note = _serving_refusal_note(smallest, budget) + note
    of_these = (
        f"{len(found)} {entry.noun}"
        if not barred
        else f"{len(usable)} {entry.noun} this card can start"
    )
    return Decision(
        capability=entry.name,
        job_type=entry.job_type,
        enabled=False,
        selected="",
        reason=(
            f"disabled: the smallest of {of_these} is {smallest.id} "
            f"at {_spell_floor(smallest, work)} and there is only "
            f"{arithmetic} — short by {_gib(shortfall)}.{barred_note}{note}"
            + (UPSTREAM_OFFER if entry.routable else "")
        ),
        summary=(
            _serving_refusal_summary(entry, smallest, budget)
            or (
                f"cannot {entry.plainly} — the smallest option needs "
                f"{_gib(shortfall)} more memory than this machine has free"
            )
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
    card: "CardFacts | None" = None,
) -> tuple[Decision, ...]:
    return tuple(
        decide(
            entry,
            backend_kind,
            total_bytes=total_bytes,
            desktop_allowance_bytes=desktop_allowance_bytes,
            gpu_vendor=gpu_vendor,
            chosen=chosen.get(entry.name),
            card=card,
        )
        for entry in CLASSES
    )


WORK_FROM_DEFAULT = "default"
WORK_FROM_REQUEST = "request"

CONTEXT_TOKENS_PARAM = "context_tokens"
CONCURRENCY_PARAM = "concurrency"


def _positive_int(name: str, raw: str) -> int:
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
    if default is None:
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
    card: "CardFacts | None" = None,
) -> tuple[ContextCeiling, ...]:
    if entry.candidates is None:
        return ()
    found = tuple(
        c
        for c in entry.candidates(backend_kind)
        if not c.lacks(card)
    )
    ceilings = (c.context_ceiling(available_bytes, concurrency) for c in found)
    return tuple(ceiling for ceiling in ceilings if ceiling is not None)


def check_ceiling(
    entry: CapabilityClass,
    backend_kind: str,
    *,
    available_bytes: int,
    work: WorkingContext,
    chosen: str | None,
    card: "CardFacts | None" = None,
) -> tuple[ContextCeiling, ...]:
    ceilings = context_ceilings(
        entry,
        backend_kind,
        available_bytes=available_bytes,
        concurrency=work.concurrency,
        card=card,
    )
    if not ceilings or entry.candidates is None:
        return ceilings
    by_model = {ceiling.model: ceiling for ceiling in ceilings}
    if chosen is not None:
        if chosen not in by_model:
            return ceilings
        governing = by_model[chosen]
    else:
        governing = max(ceilings, key=lambda ceiling: ceiling.tokens)
    holds_weights = [
        c
        for c in entry.candidates(backend_kind)
        if not c.lacks(card)
        and (c.memory is None or c.memory.fixed_bytes < available_bytes)
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


MIN_LOAD_CONTEXT = 2048


def check_load_context(
    manifest: Any,
    backend_kind: str,
    *,
    available_bytes: int,
    context: int,
) -> ContextCeiling:
    candidate = Candidate.of(manifest, backend_kind)
    ceiling = candidate.context_ceiling(available_bytes, 1)
    if ceiling is None:
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
    return CapabilityRow(
        capability=row.capability,
        enabled=True,
        selected=model,
        reason=(
            f"routed to {model.partition('/')[0]}; "
            f"{LOCAL_ANSWER_PREFIX}{row.reason}"
        ),
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
) -> CapabilityRecord:
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
    card: "CardFacts | None" = None,
) -> list[dict[str, Any]]:
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
        if routes.get(entry.name) is None:
            check_ceiling(
                entry,
                record.backend_kind,
                available_bytes=budget,
                work=requested,
                chosen=chosen.get(entry.name),
                card=card,
            )

    rows: list[dict[str, Any]] = []
    for stored in record.rows:
        row = stored.to_dict()
        found = BY_NAME.get(stored.capability)
        if found is None:
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
                card=card,
            )
            fresh = decision.row()
            model = routes.get(found.name)
            row = (fresh if model is None else routed_row(fresh, model)).to_dict()
        row["work"] = None if work is None else {**work.to_dict(), "from": basis}
        row["context_ceilings"] = None
        if found.client_sized and work is not None:
            row["context_ceilings"] = [
                ceiling.to_dict()
                for ceiling in context_ceilings(
                    found,
                    record.backend_kind,
                    available_bytes=budget,
                    concurrency=work.concurrency,
                    card=card,
                )
            ]
        rows.append(row)
    return rows


def describe_card(card: "CardFacts | None", total_bytes: int, pool: str) -> str:
    name = "this machine" if card is None else card.name
    parts = [name, f"{total_bytes / GIB:.1f} GiB {pool}"]
    if card is not None:
        if card.has(BF16) is False:
            parts.append("no bf16")
        if card.has(TENSOR_CORES) is False:
            parts.append("no tensor cores")
    return ", ".join(parts)


def _shown_precision(candidate: Candidate, card: "CardFacts | None") -> str:
    shown = candidate.precision_on(card)
    if candidate.degraded_on(card):
        return (
            f" in {shown} instead of "
            f"{precision_label(candidate.bits, candidate.dtype)} (this card has "
            f"no bf16: it needs compute capability {feature_floor(BF16)}, and "
            f"this card is {'unknown' if card is None else card.compute_capability})"
        )
    if candidate.bits is not None and candidate.bits < 16:
        return f" in {shown}"
    return ""


def _why_not_best(
    decision: Decision,
    best: Candidate,
    work: "WorkingContext | None",
    card: "CardFacts | None",
) -> str:
    missing = best.lacks(card)
    if missing:
        return f" The best, {best.id}, cannot start here: it needs {_needs_phrase(missing, card)}."
    if not best.holds(work, decision.available_bytes):
        fewer = (
            " Fewer pieces at once was tried first; a smaller model is taken "
            "only when even one at a time does not fit."
            if asrplan.is_ladder(best.serving)
            else ""
        )
        return (
            f" The best, {best.id}{_shown_precision(best, card) or ''}, needs "
            f"{_spell_floor(best, work)} and this card gives a job "
            f"{_gib(decision.available_bytes)}.{fewer}"
        )
    return f" {decision.selected} is the one chosen in Settings; {best.id} would also fit."


def _class_line(
    entry: CapabilityClass, decision: Decision, card: "CardFacts | None"
) -> str:
    if not decision.enabled:
        return decision.summary[0].upper() + decision.summary[1:] + "."
    if not decision.candidates:
        return f"Can {entry.plainly}."
    by_id = {c.id: c for c in decision.candidates}
    picked = by_id.get(decision.selected)
    if picked is None:
        return f"Can {entry.plainly}, using {decision.selected}."
    line = (
        f"Will {entry.plainly} with {picked.id}{_shown_precision(picked, card)}"
        f"{_serving_summary(picked, decision.available_bytes)}."
    )
    best = decision.candidates[0]
    if best.id != picked.id:
        line += _why_not_best(decision, best, entry.work, card)
    return line


def install_plan(
    job_type: str,
    decisions: tuple[Decision, ...],
    *,
    card: "CardFacts | None",
    total_bytes: int,
    pool: str,
    desktop_allowance_bytes: int | None = None,
    desktop_basis: str | None = None,
) -> dict[str, Any]:
    entries = classes_for_job_type(job_type)
    if not entries:
        raise ApiError(
            400,
            "unknown_job_type",
            f"{job_type!r} has no capability classes; this build knows "
            f"{sorted({entry.job_type for entry in CLASSES})}",
        )
    by_name = {d.capability: d for d in decisions}
    rows: list[dict[str, Any]] = []
    for entry in entries:
        decision = by_name.get(entry.name)
        if decision is None:
            continue
        picked = next(
            (c for c in decision.candidates if c.id == decision.selected), None
        )
        rows.append(
            {
                "capability": entry.name,
                "enabled": decision.enabled,
                "selected": decision.selected,
                "precision": None if picked is None else picked.precision_on(card),
                "reduced_precision": False if picked is None else picked.degraded_on(card),
                "best": None if not decision.candidates else decision.candidates[0].id,
                "lacking_features": list(decision.lacking_features),
                "line": _class_line(entry, decision, card),
            }
        )
    usable = any(row["enabled"] for row in rows)
    card_words = describe_card(card, total_bytes, pool)
    reserve_words = (
        None
        if desktop_allowance_bytes is None or desktop_basis is None
        else desktop_reserve_words(desktop_allowance_bytes, desktop_basis)
    )
    reserve_line = ""
    if reserve_words is not None:
        budget = available_bytes(total_bytes, desktop_allowance_bytes or 0)
        reserve_line = (
            f"{reserve_words[0].upper()}{reserve_words[1:]}, so a job gets "
            f"{_gib(budget)}.\n"
        )
    lines = "\n".join(f"- {row['line']}" for row in rows)
    closing = (
        "Install it?"
        if usable
        else (
            "Nothing it offers can run on this card. Install it anyway? It stays "
            "off until this server has a card that can run it."
        )
    )
    return {
        "job_type": job_type,
        "card": None if card is None else card.to_dict(),
        "card_words": card_words,
        "desktop_reserve": reserve_words,
        "desktop_allowance_basis": desktop_basis,
        "usable": usable,
        "classes": rows,
        "confirm": (
            f"Install {job_type}.\n\nYour card ({card_words}):\n{reserve_line}"
            f"{lines}\n\n{closing}"
        ),
    }


def subject_plan(
    subject_id: str,
    decisions: tuple[Decision, ...],
    *,
    card: "CardFacts | None",
    total_bytes: int,
    pool: str,
) -> dict[str, Any]:
    card_words = describe_card(card, total_bytes, pool)
    lines: list[str] = []
    runs = False
    for decision in decisions:
        entry = BY_NAME.get(decision.capability)
        candidate = next((c for c in decision.candidates if c.id == subject_id), None)
        if entry is None or candidate is None:
            continue
        missing = candidate.lacks(card)
        if missing:
            lines.append(
                f"Cannot {entry.plainly} with it: it needs {_needs_phrase(missing, card)}."
            )
        elif not candidate.holds(entry.work, decision.available_bytes):
            lines.append(
                f"Cannot {entry.plainly} with it: it needs "
                f"{_spell_floor(candidate, entry.work)} and this card gives a job "
                f"{_gib(decision.available_bytes)}."
                + _serving_refusal_note(candidate, decision.available_bytes)
            )
        elif decision.selected == subject_id:
            runs = True
            lines.append(
                f"Will {entry.plainly} with it{_shown_precision(candidate, card)}"
                f"{_serving_summary(candidate, decision.available_bytes)}."
            )
        else:
            runs = True
            lines.append(
                f"Can {entry.plainly} with it{_shown_precision(candidate, card)}; "
                f"{decision.selected or 'nothing'} is what this server uses for "
                "that unless it is chosen in Settings."
            )
    if not lines:
        raise ApiError(
            404,
            "unknown_subject",
            f"{subject_id!r} is not offered by any capability class on this backend",
        )
    closing = "Download it?" if runs else "It cannot run on this card. Download it anyway?"
    body = "\n".join(f"- {line}" for line in lines)
    return {
        "subject": subject_id,
        "card": None if card is None else card.to_dict(),
        "card_words": card_words,
        "usable": runs,
        "lines": lines,
        "confirm": f"Download {subject_id}.\n\nYour card ({card_words}):\n{body}\n\n{closing}",
    }


def job_type_enabled(job_type: str, decisions: tuple[Decision, ...]) -> bool:
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
    "describe_card",
    "install_plan",
    "subject_plan",
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
