"""How a Qwen3-ASR model is served on a card that cannot hold its full width.

Owen, 2026-09-26: *"yes, fewer at once before quantizing for asr too"* — the
order `ttsplan` gives Higgs (*"we should drop batches to 1 at a time before we
quantize. id rather it go slow than sound worse"*), applied to transcription.
So on a card that cannot hold a Qwen3-ASR model at its declared width at full
precision, the capability walk offers, in this order:

    1. full precision at the declared width (`max_batch`, 8 on cuda-linux)
    2. full precision at the widest narrower width that fits, down to ONE
    3. only then the next candidate: a smaller model (the 0.6B after the
       1.7B, then the whispers), quantized or not, and never one under 4 bits
       (`precision.MIN_WEIGHT_BITS`, `capability.CatalogCandidates`)

Full precision is bf16, or fp16 on a card without bf16 (`engines.vllm.
run_dtype`): the same two bytes a parameter, so that fallback is not
quantizing and every figure below holds for it. No quantized Qwen3-ASR is
pinned, so step 3 reaches a smaller model rather than fewer bits of this one.

WHAT "WIDTH" IS HERE. The manifest's `max_batch`, passed to vLLM as
`max_num_seqs`: how many pieces (<= 180 s of audio each) decode at once. The
job's pieces are cut before decoding and fed to vLLM together, so the width is
pieces in flight, and it is the only term of the estimate that scales with it:

    need(width) = memory_bytes_estimate - kv_cache_memory_bytes + kv_pool(width)

    kv_pool(width) = max(width x kv_cache_memory_bytes / max_batch,
                         max_model_len x kv_bytes_per_token)

EVERY FIGURE IS THE MANIFEST'S, DECLARED, NOT MEASURED (the Qwen3-ASR blocks
say COMPUTED for each term). The weights, the non-KV overhead and the audio
tower's 1 GiB activation reserve are held FIXED at their declared values: the
tower's is declared for 8 pieces in one prefill and is not scaled down with the
width, because no per-piece figure for it is declared and a guessed one would
be an invented constant. That makes a narrow width's need an upper bound. The
KV pool is the manifest's own `kv_cache_memory_bytes`, which is `max_batch`
pieces of 4,096 tokens (the block's comment), scaled to the width.

The floor under the pool is vLLM's, not a choice: vLLM v1 refuses to start when
its KV pool cannot hold ONE sequence of `max_model_len` tokens
(`check_enough_kv_cache_memory`; not re-read at the 0.29.0 pin, and the load
test on a small card is the check). At the shipped numbers that is 8,192 x
114,688 B = 896 MiB, two pieces' worth, so one piece at a time needs exactly
what two do. The ladder still says "down to one" because that is the ruling; a
width of two is simply taken first when both fit.
"""

from __future__ import annotations

from typing import Any

from .ttsplan import ServingVariant

#: The only Qwen3-ASR arm with a width to narrow: vLLM on cuda-linux. Both
#: Mac engines take one piece per call already (`asrmodels._check_qwen_block`).
LADDER_BACKEND = "cuda-linux"
LADDER_ENGINE = "vllm"
LADDER_FAMILY = "qwen3-asr"

#: What a piece is called in a person's sentence.
UNIT = "piece"

BASIS = (
    "declared: the manifest's own figures, with the KV pool scaled to the "
    "width and the weights, overhead and audio-tower reserve held at their "
    "declared values; unmeasured"
)

#: Said wherever a narrower width is the answer.
LOAD_TEST_NOTE = (
    "The narrower width's need is the manifest's declared arithmetic, not a "
    "measurement, so this card's first transcription is the load test."
)


def kv_pool(spec: Any, width: int) -> int:
    """vLLM's KV pool for `width` pieces at once, never under one full sequence."""
    per_piece = spec.kv_cache_memory_bytes // spec.max_batch
    one_sequence = spec.max_model_len * spec.kv_bytes_per_token
    return max(width * per_piece, one_sequence)


def need(spec: Any, width: int) -> int:
    """What the ASR engine alone needs at `width` pieces at once."""
    if width >= spec.max_batch:
        return spec.memory_bytes_estimate
    return spec.memory_bytes_estimate - spec.kv_cache_memory_bytes + kv_pool(spec, width)


def ladder_for(manifest: Any, spec: Any, backend_kind: str) -> tuple[ServingVariant, ...] | None:
    """The width ladder for one Qwen3-ASR model on one backend, or None.

    None for everything else: whisper, the Mac's engines, and any block that
    does not state the four figures the arithmetic reads.
    """
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
    """Is this ladder an ASR one (and not Higgs's)?"""
    return bool(ladder) and ladder[0].unit == UNIT


def explain(ladder: tuple[ServingVariant, ...], chosen: ServingVariant, budget_bytes: int) -> str:
    """Why this width, in the person's words."""
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
    """The widest width whose need, plus `extra_bytes`, fits `budget_bytes`.

    None when the model has no ladder (keep its own `max_batch`) or when not
    even one piece fits (the guard then refuses at the declared width's need,
    by name). `extra_bytes` is what else the JOB holds beside the engine: the
    aligner, for a word-timestamped job. The capability walk asks with 0,
    because its verdict is about the transcriber alone, as the manifest's
    estimate is; a job with word timestamps may therefore narrow further than
    the verdict said, and its warming line says so.
    """
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
