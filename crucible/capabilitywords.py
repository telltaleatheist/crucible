from __future__ import annotations

from typing import TYPE_CHECKING

from . import asrplan, ttsplan
from .audiomodels import LOW_VRAM_SETTING
from .backend import (
    BF16,
    FEATURE_FLOORS,
    MEASURED_FEATURES,
    TENSOR_CORES,
    CardFacts,
    feature_floor,
    sm_name,
)
from .capabilityclasses import API_KEY_ADVICE_BELOW_PARAMS_B
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

# Where a person gives this server an upstream: the Settings page's accounts block (the
# operator page's own heading), or the same tables in config.toml.
UPSTREAM_SETTINGS_BLOCK = "Accounts this engine may spend"


def api_key_advice(entry: "CapabilityClass") -> str:
    """The one sentence a small automatic pick adds (CapabilityClass.advises_api_key),
    without its closing stop: what to do for better results, and exactly where."""
    keyed = [
        UPSTREAM_DISPLAY[name] for name in UPSTREAM_NAMES if UPSTREAM_FIELD[name] == "key"
    ]
    return (
        f"models under {API_KEY_ADVICE_BELOW_PARAMS_B:g}B give weaker results, so for "
        f"better ones add an API key for {either(keyed)} in Settings, under "
        f"\"{UPSTREAM_SETTINGS_BLOCK}\", and send {entry.name} to it in the same page "
        "(or set [upstreams] and [routes] in config.toml)"
    )


# Crucible turns the setting on by itself where the card needs it (crucible/lowvram.py),
# so a refusal that names it is a host where a person turned it off.
LOW_VRAM_STEPS = (
    "`crucible audio low-vram on` turns it on (`crucible guest audio low-vram on` on a "
    "Windows PC), or `crucible audio low-vram auto` lets Crucible decide it from this "
    "card; Settings has the same switch"
)


def spell_out(candidate: Candidate, work: "WorkingContext | None") -> str:
    need = candidate.need_bytes(work)
    if candidate.whole_bytes is not None:
        return (
            f"{gib_text(need)} with {LOW_VRAM_SETTING}, holding only the part each stage "
            f"uses on the card ({gib_text(candidate.whole_bytes)} whole)"
        )
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


def low_vram_summary(candidate: Candidate) -> str:
    if candidate.whole_bytes is None:
        return ""
    return f" — with {LOW_VRAM_SETTING}, one part of it on the card at a time"


def low_vram_offer(candidates: "tuple[Candidate, ...]", budget: int) -> "Candidate | None":
    """The first of these that this host does not hold under `[audio] low_vram` and would
    fit if it did."""
    return next((c for c in candidates if c.would_fit_low_vram(budget)), None)


def low_vram_refusal_note(candidate: "Candidate | None") -> str:
    if candidate is None:
        return ""
    assert candidate.low_vram_bytes is not None
    return (
        f" {LOW_VRAM_SETTING} is off on this host. With it on, {candidate.id} holds only "
        f"the part each stage uses on the card and needs {gib_text(candidate.low_vram_bytes)}, "
        f"which fits: {LOW_VRAM_STEPS}."
    )


def low_vram_refusal_summary(entry: "CapabilityClass", candidate: Candidate) -> str:
    return (
        f"cannot {entry.plainly} — {candidate.id} fits this machine only with "
        f"{LOW_VRAM_SETTING} on, and it is off: {LOW_VRAM_STEPS}"
    )


def goal_phrase(
    entry: "CapabilityClass",
    picked: Candidate,
    ranked: "tuple[Candidate, ...]",
    work: "WorkingContext | None",
    budget: int,
    card: "CardFacts | None",
) -> str:
    """What a goal class's automatic pick took, against its goal (docs/VERB-SIZING.md
    rule 3): "goal 9B; bf16 fits with 2.1 GiB to spare", or below the goal, "goal 27B;
    the largest that fits this card". `ranked` is the class's pick order on this host."""
    goal = entry.goal
    assert goal is not None and picked.params_b is not None and picked.bits is not None
    if picked.params_b < goal.params_b:
        return f"goal {goal.words}; the largest that fits this card"
    spare = gib_text(budget - picked.need_bytes(work))
    precision = picked.precision_on(card)
    finer = any(
        c.params_b == picked.params_b and c.bits is not None and c.bits > picked.bits
        for c in ranked
    )
    if finer:
        return (
            f"goal {goal.words}; {precision}, the highest precision of it that fits, "
            f"with {spare} to spare"
        )
    return f"goal {goal.words}; {precision} fits with {spare} to spare"


def above_goal_note(entry: "CapabilityClass", fitting: "tuple[Candidate, ...]") -> str:
    """The fitting candidates a goal class's automatic pick passed over for being above
    its goal: a person may still choose one in Settings."""
    goal = entry.goal
    if goal is None:
        return ""
    above = [c.id for c in fitting if c.params_b is not None and c.params_b > goal.params_b]
    if not above:
        return ""
    return (
        f" {', '.join(above)} also {'fits' if len(above) == 1 else 'fit'}, and "
        f"{'is' if len(above) == 1 else 'are'} above the {goal.words} goal, which the "
        f"automatic pick never exceeds; Settings can still choose "
        f"{'it' if len(above) == 1 else 'one'}."
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
    "UPSTREAM_SETTINGS_BLOCK",
    "api_key_advice",
    "above_goal_note",
    "goal_phrase",
    "LOCAL_ANSWER_PREFIX",
    "LOW_VRAM_STEPS",
    "MEASURED_WORDS",
    "NEEDS_WSL_REASON",
    "TTS_WIDTH_LOAD_TEST_NOTE",
    "UPSTREAM_OFFER",
    "barred_note",
    "card_words",
    "describe_card",
    "either",
    "feature_order",
    "low_vram_offer",
    "low_vram_refusal_note",
    "low_vram_refusal_summary",
    "low_vram_summary",
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
