from __future__ import annotations

from typing import TYPE_CHECKING

from . import asrplan, ttsplan
from .backend import (
    BF16,
    FEATURE_FLOORS,
    MEASURED_FEATURES,
    TENSOR_CORES,
    CardFacts,
    feature_floor,
    sm_name,
)
from .fit import Candidate, WorkingContext
from .memorybudget import gib_text
from .precision import label as precision_label
from .servingplan import ServingVariant
from .upstreamrecord import UPSTREAM_DISPLAY, UPSTREAM_FIELD, UPSTREAM_NAMES

if TYPE_CHECKING:
    from .capabilityclasses import CapabilityClass


NEEDS_WSL_REASON = (
    "this job type needs the WSL2 engine (vLLM/SGLang); install it from the "
    "console"
)

CPU_BUILD_REASON = (
    "cpu build — slow; the model runs on this machine's CPU"
)

LOCAL_ANSWER_PREFIX = "the local answer would be: "

TTS_WIDTH_LOAD_TEST_NOTE = (
    "SGLang-Omni's memory fraction is not rescaled for a narrower width, so "
    "this card's first load is the load test."
)

MEASURED_WORDS: dict[str, str] = {
    "vllm": "vLLM to start on this card",
    "cuda_graphs": "CUDA graphs to capture on this card",
}


def either(names: list[str]) -> str:
    if len(names) < 2:
        return "".join(names)
    return ", ".join(names[:-1]) + " or " + names[-1]


def upstream_offer() -> str:
    keyed = [
        UPSTREAM_DISPLAY[name] for name in UPSTREAM_NAMES if UPSTREAM_FIELD[name] == "key"
    ]
    addressed = [
        UPSTREAM_DISPLAY[name] for name in UPSTREAM_NAMES if UPSTREAM_FIELD[name] == "url"
    ]
    ways = []
    if keyed:
        ways.append(f"an API key for {either(keyed)}")
    if addressed:
        ways.append(f"the address of a server running {either(addressed)}")
    return (
        f" This class can run somewhere else instead: in settings, add "
        f"{', or '.join(ways)}, and this host will route it rather than refuse it."
    )


UPSTREAM_OFFER = upstream_offer()


def spell_out(candidate: Candidate, work: "WorkingContext | None") -> str:
    need = candidate.need_bytes(work)
    if work is None or candidate.memory is None:
        return gib_text(need)
    terms = candidate.memory
    kv = terms.kv_bytes_per_token * work.tokens * work.concurrency
    return (
        f"{gib_text(need)} — {gib_text(terms.weights_bytes)} weights + "
        f"{gib_text(terms.overhead_bytes)} overhead + {gib_text(kv)} KV for "
        f"{work.tokens} tokens x {work.concurrency} in flight"
    )


def feature_order(features: "set[str] | tuple[str, ...]") -> tuple[str, ...]:
    order = [name for name, _floor, _what in FEATURE_FLOORS] + list(MEASURED_FEATURES)
    return tuple(name for name in order if name in features)


def card_words(card: "CardFacts | None") -> str:
    if card is None or card.compute_capability is None:
        return "a card whose compute capability could not be read"
    return f"{sm_name(card.compute_capability)} ({card.compute_capability})"


def needs_phrase(features: tuple[str, ...], card: "CardFacts | None") -> str:
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
                f"{MEASURED_WORDS.get(feature, feature)}, and Crucible's own "
                f"measurement{when} found it does not"
                + (f" ({detail})" if detail else "")
            )
        else:
            parts.append(
                f"{feature} (compute capability {feature_floor(feature)} or "
                f"newer), and this card is {card_words(card)}"
            )
    return "; ".join(parts)


def too_old_phrase(features: tuple[str, ...], card: "CardFacts | None") -> str:
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


def barred_note(barred: tuple[Candidate, ...], card: "CardFacts | None") -> str:
    if not barred:
        return ""
    missing = feature_order({f for c in barred for f in c.lacks(card)})
    return (
        f" {len(barred)} more — {', '.join(c.id for c in barred)} — "
        f"{'needs' if len(barred) == 1 else 'need'} "
        f"{needs_phrase(missing, card)}."
    )


def precision_note(candidate: Candidate, card: "CardFacts | None") -> str:
    if candidate.degraded_on(card):
        return (
            f" On this card {candidate.id} runs in "
            f"{candidate.precision_on(card)}, not "
            f"{precision_label(candidate.bits, candidate.dtype)}: it needs "
            f"{needs_phrase((BF16,), card)}."
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


def spell_floor(candidate: Candidate, work: "WorkingContext | None") -> str:
    if candidate.serving is None:
        return spell_out(candidate, work)
    floor = min(
        (v for v in candidate.serving if v.available), key=lambda v: (v.need_bytes, v.width)
    )
    return f"{gib_text(floor.need_bytes)} for {floor.label()} (declared)"


def spell_chosen(
    candidate: Candidate, work: "WorkingContext | None", budget: int
) -> str:
    chosen = candidate.serving_on(budget)
    if chosen is None or chosen == candidate.serving[0]:
        return spell_out(candidate, work)
    return f"{gib_text(chosen.need_bytes)} for {chosen.label()} (declared)"


def serving_explained(candidate: Candidate, chosen: ServingVariant, budget: int) -> str:
    plan = asrplan if asrplan.is_ladder(candidate.serving) else ttsplan
    return plan.explain(candidate.serving, chosen, budget)


def _narrower(candidate: Candidate, budget: int) -> "ServingVariant | None":
    if candidate.serving is None:
        return None
    chosen = candidate.serving_on(budget)
    if chosen is None or chosen == candidate.serving[0]:
        return None
    return chosen


def serving_note(candidate: Candidate, budget: int) -> str:
    chosen = _narrower(candidate, budget)
    if chosen is None:
        return ""
    if asrplan.is_ladder(candidate.serving):
        return (
            f" {candidate.id} {asrplan.explain(candidate.serving, chosen, budget)}. "
            f"{asrplan.LOAD_TEST_NOTE}"
        )
    return (
        f" {candidate.id} {ttsplan.explain(candidate.serving, chosen, budget)}. "
        + TTS_WIDTH_LOAD_TEST_NOTE
    )


def serving_summary(candidate: Candidate, budget: int) -> str:
    chosen = _narrower(candidate, budget)
    if chosen is None:
        return ""
    return " — " + serving_explained(candidate, chosen, budget)


def skipped_ladder_note(
    usable: tuple[Candidate, ...], best: Candidate, budget: int
) -> str:
    skipped = []
    for candidate in usable:
        if candidate.id == best.id:
            break
        if asrplan.is_ladder(candidate.serving):
            skipped.append(
                f"{candidate.id} ({gib_text(candidate.serving[-1].need_bytes)}, declared)"
            )
    if not skipped:
        return ""
    return (
        f" {' and '.join(skipped)} "
        f"{'does' if len(skipped) == 1 else 'do'} not fit even one piece at a "
        f"time at full precision, so {best.id} is taken: fewer pieces at once "
        "is tried before a smaller or quantized model."
    )


def _passage_ladder(candidate: Candidate) -> bool:
    return candidate.serving is not None and not asrplan.is_ladder(candidate.serving)


def serving_refusal_summary(
    entry: "CapabilityClass", candidate: Candidate, budget: int
) -> str:
    if not _passage_ladder(candidate):
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


def serving_refusal_note(candidate: Candidate, budget: int) -> str:
    if not _passage_ladder(candidate):
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


def with_notes(text: str, *notes: str) -> str:
    said = "".join(notes)
    if not said:
        return text
    return text.rstrip(".") + "." + said


def shown_precision(candidate: Candidate, card: "CardFacts | None") -> str:
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


def describe_card(card: "CardFacts | None", total_bytes: int, pool: str) -> str:
    name = "this machine" if card is None else card.name
    parts = [name, f"{gib_text(total_bytes)} {pool}"]
    if card is not None:
        if card.has(BF16) is False:
            parts.append("no bf16")
        if card.has(TENSOR_CORES) is False:
            parts.append("no tensor cores")
    return ", ".join(parts)


__all__ = [
    "CPU_BUILD_REASON",
    "LOCAL_ANSWER_PREFIX",
    "MEASURED_WORDS",
    "NEEDS_WSL_REASON",
    "TTS_WIDTH_LOAD_TEST_NOTE",
    "UPSTREAM_OFFER",
    "barred_note",
    "card_words",
    "describe_card",
    "either",
    "feature_order",
    "needs_phrase",
    "precision_note",
    "serving_explained",
    "serving_note",
    "serving_refusal_note",
    "serving_refusal_summary",
    "serving_summary",
    "shown_precision",
    "skipped_ladder_note",
    "spell_chosen",
    "spell_floor",
    "spell_out",
    "too_old_phrase",
    "upstream_offer",
    "with_notes",
]
