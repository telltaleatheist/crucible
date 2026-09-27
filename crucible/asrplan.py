from __future__ import annotations

from typing import Any

from .backend import CUDA_LINUX
from .enginespec import VLLM_ENGINE
from .servingplan import ServingVariant

LADDER_BACKEND = CUDA_LINUX
LADDER_ENGINE = VLLM_ENGINE
LADDER_FAMILY = "qwen3-asr"

UNIT = "piece"

BASIS = (
    "declared: the manifest's own figures, with the KV pool scaled to the "
    "width and the weights, overhead and audio-tower reserve held at their "
    "declared values; unmeasured"
)

LOAD_TEST_NOTE = (
    "The narrower width's need is the manifest's declared arithmetic, not a "
    "measurement, so this card's first transcription is the load test."
)


def kv_pool(spec: Any, width: int) -> int:
    per_piece = spec.kv_cache_memory_bytes // spec.max_batch
    one_sequence = spec.max_model_len * spec.kv_bytes_per_token
    return max(width * per_piece, one_sequence)


def need(spec: Any, width: int) -> int:
    if width >= spec.max_batch:
        return spec.memory_bytes_estimate
    return spec.memory_bytes_estimate - spec.kv_cache_memory_bytes + kv_pool(spec, width)


def ladder_for(manifest: Any, spec: Any, backend_kind: str) -> tuple[ServingVariant, ...] | None:
    if backend_kind != LADDER_BACKEND:
        return None
    if getattr(manifest, "family", None) != LADDER_FAMILY:
        return None
    if getattr(spec, "engine", None) != LADDER_ENGINE:
        return None
    for key in ("max_batch", "kv_cache_memory_bytes", "max_model_len", "kv_bytes_per_token"):
        if getattr(spec, key, None) is None:
            return None
    rows = [
        ServingVariant(
            bits=16,
            width=spec.max_batch,
            need_bytes=spec.memory_bytes_estimate,
            available=True,
            basis="the model's own estimate at its declared width",
            unit=UNIT,
        )
    ]
    for width in range(spec.max_batch - 1, 0, -1):
        rows.append(
            ServingVariant(
                bits=16,
                width=width,
                need_bytes=need(spec, width),
                available=True,
                basis=BASIS,
                unit=UNIT,
            )
        )
    return tuple(rows)


def is_ladder(ladder: tuple[ServingVariant, ...] | None) -> bool:
    return bool(ladder) and ladder[0].unit == UNIT


def explain(ladder: tuple[ServingVariant, ...], chosen: ServingVariant, budget_bytes: int) -> str:
    top = ladder[0]
    if chosen == top:
        return f"transcribes {chosen.width} pieces at a time at full precision"
    if chosen.width == 1:
        return (
            "transcribes one piece at a time at full precision: this card cannot "
            f"hold {top.width} at once, and fewer at once is tried before a "
            "smaller or quantized model (slower, same precision)"
        )
    return (
        f"transcribes {chosen.width} pieces at a time at full precision instead "
        f"of {top.width}: this card cannot hold more at once, and fewer at once "
        "is tried before a smaller or quantized model (slower, same precision)"
    )


def width_for(
    spec: Any, backend_kind: str, manifest: Any, budget_bytes: int, extra_bytes: int = 0
) -> int | None:
    ladder = ladder_for(manifest, spec, backend_kind)
    if ladder is None:
        return None
    for variant in ladder:
        if variant.need_bytes + extra_bytes <= budget_bytes:
            return variant.width
    return None


__all__ = [
    "BASIS",
    "LOAD_TEST_NOTE",
    "UNIT",
    "explain",
    "is_ladder",
    "kv_pool",
    "ladder_for",
    "need",
    "width_for",
]
