from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .backend import CUDA_LINUX
from .memorybudget import available_bytes
from .narratorengines import HIGGS_V3 as HIGGS_ENGINE
from .servingplan import ServingVariant

BF16_ONE_AT_A_TIME_BYTES = 10_000_000_000

KV_BYTES_PER_TOKEN = 147_456
TOKENS_PER_PASSAGE = 2_000
BYTES_PER_EXTRA_PASSAGE = KV_BYTES_PER_TOKEN * TOKENS_PER_PASSAGE

QUANTIZED_ONE_AT_A_TIME_BYTES: tuple[tuple[int, int], ...] = (
    (8, 5_800_000_000),
    (4, 4_200_000_000),
)

PENDING_NOTE = (
    "no quantized Higgs voice has been made yet — Owen and the trainer are to "
    "make and listen-test one — and whether SGLang-Omni serves a quantized "
    "checkpoint of this architecture is unverified"
)

BASIS = (
    "declared: the owens-pc session's arithmetic of 2026-09-26 (Qwen3-4B-Base "
    "backbone, 8.49 GB bf16 + 0.81 GB codec, ~144 KiB/token KV, ~2,000 tokens a "
    "passage); unmeasured"
)


def bf16_need(width: int) -> int:
    return BF16_ONE_AT_A_TIME_BYTES + (width - 1) * BYTES_PER_EXTRA_PASSAGE


def variants(declared_width: int, declared_need_bytes: int) -> tuple[ServingVariant, ...]:
    rows = [
        ServingVariant(
            bits=16,
            width=declared_width,
            need_bytes=declared_need_bytes,
            available=True,
            basis="the voice's own estimate at its declared width",
        )
    ]
    for width in range(declared_width - 1, 0, -1):
        rows.append(
            ServingVariant(bits=16, width=width, need_bytes=bf16_need(width), available=True, basis=BASIS)
        )
    for bits, need in QUANTIZED_ONE_AT_A_TIME_BYTES:
        rows.append(ServingVariant(bits=bits, width=1, need_bytes=need, available=False, basis=BASIS))
    return tuple(rows)


def ladder_for(manifest: Any, spec: Any, backend_kind: str) -> tuple[ServingVariant, ...] | None:
    if backend_kind != CUDA_LINUX:
        return None
    if getattr(manifest, "narrator_engine", None) != HIGGS_ENGINE:
        return None
    serving = getattr(manifest, "serving", None)
    if serving is None:
        return None
    return variants(serving.max_num_seqs, spec.memory_bytes_estimate)


def choose(ladder: tuple[ServingVariant, ...], budget_bytes: int) -> ServingVariant | None:
    for variant in ladder:
        if variant.available and variant.need_bytes <= budget_bytes:
            return variant
    return None


def first_fitting(ladder: tuple[ServingVariant, ...], budget_bytes: int) -> ServingVariant | None:
    for variant in ladder:
        if variant.need_bytes <= budget_bytes:
            return variant
    return None


def explain(ladder: tuple[ServingVariant, ...], chosen: ServingVariant, budget_bytes: int) -> str:
    top = ladder[0]
    if chosen == top:
        return f"renders {chosen.width} passages at a time at full quality"
    if chosen.full_precision:
        if chosen.width == 1:
            return (
                "renders one passage at a time at full quality: this card cannot "
                "hold more at once (slower, and it sounds the same)"
            )
        return (
            f"renders {chosen.width} passages at a time at full quality instead of "
            f"{top.width}: this card cannot hold more at once"
        )
    return (
        f"{chosen.bits}-bit: full quality does not fit even one passage at a time "
        f"(~{bf16_need(1) / 1e9:.1f} GB needed)"
    )


@dataclass(frozen=True)
class LoadPlan:
    need_bytes: int
    width: int | None
    floor_bytes: int
    variant: ServingVariant | None


def load_plan(
    manifest: Any,
    spec: Any,
    backend_kind: str,
    *,
    total_bytes: int,
    desktop_allowance_bytes: int,
) -> LoadPlan:
    ladder = ladder_for(manifest, spec, backend_kind)
    if ladder is None:
        estimate = spec.memory_bytes_estimate
        return LoadPlan(estimate, None, estimate, None)
    floor = min(v.need_bytes for v in ladder if v.available)
    budget = available_bytes(total_bytes, desktop_allowance_bytes)
    chosen = choose(ladder, budget)
    if chosen is None:
        return LoadPlan(floor, None, floor, None)
    width = None if chosen.width == ladder[0].width else chosen.width
    return LoadPlan(chosen.need_bytes, width, floor, chosen)


__all__ = [
    "LoadPlan",
    "load_plan",
    "BF16_ONE_AT_A_TIME_BYTES",
    "HIGGS_ENGINE",
    "PENDING_NOTE",
    "ServingVariant",
    "bf16_need",
    "choose",
    "explain",
    "first_fitting",
    "ladder_for",
    "variants",
]
