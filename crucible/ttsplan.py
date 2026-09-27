"""How a Higgs v3 voice is served on a card that cannot hold its full width.

Two rulings of Owen's, 2026-09-26, and their order is the whole module:

> *"no less than 4 covers higgs as well"* — superseding 2026-09-13's *"higgs is
> tied to a certain size. we cant (or wont) quantize that"*.
>
> *"we should drop batches to 1 at a time before we quantize. id rather it go
> slow than sound worse"*.

So a card that cannot hold the served width at full precision is offered, in
this order, and the first that fits is taken:

    1. bf16 at the voice's declared width (`[voice.serving] max_num_seqs`, 16)
    2. bf16 at the widest narrower width that fits, down to ONE at a time
    3. 8-bit, one at a time
    4. 4-bit, one at a time — and never below (`precision.MIN_WEIGHT_BITS`)

PRECISION BEATS PARALLELISM: a slower render at full quality is taken before a
faster one that sounds worse, which is why 8-bit is not tried at any width
until bf16 has been tried at every one.

EVERY FIGURE HERE IS DECLARED, NOT MEASURED, and each says whose it is:

* width 16, bf16: the voice's own `memory_bytes_estimate` — SGLang-Omni's
  `--mem-fraction-static 0.60` reservation at 16 in flight on the 3090 Ti
  (engines/higgs-v3/base.toml's `estimate_note`), not the model's need. It is
  read off the candidate, not restated here.
* one at a time, bf16: ~10 GB; 8-bit ~5.8 GB; 4-bit ~4.2 GB — the owens-pc
  session's arithmetic of 2026-09-26 from the checkpoint's facts (below),
  unmeasured.
* each extra passage in flight: the KV of one ~60 s chunk, ~2,000 tokens at
  ~144 KiB/token at 16 bits (Qwen3-4B-Base backbone: 36 layers x 8 KV heads x
  head_dim 128 x 2 (K,V) x 2 B = 147,456 B/token).

The checkpoint facts behind them, as the owens-pc session read them: the
mistborn-higgs-v3 backbone is Qwen3-4B-Base (36 layers, 8 KV heads, head_dim
128) at 8.49 GB of bf16 safetensors, beside the 0.81 GB
`bosonai/higgs-audio-v2-tokenizer` codec. The Mac's one measurement is 11.3 GiB
peak at a 900-character chunk under MLX (base.toml's mlx-darwin note), which
is a different engine and is not used here.

NO QUANTIZED HIGGS EXISTS YET. Owen and the trainer will make one and listen to
it. The 8-bit and 4-bit rows are therefore `pending`: capability says when one
WOULD fit and that it has not been made, and never offers it as something to
pull. Whether SGLang-Omni can serve a quantized checkpoint of this
architecture is also unverified; the pending note says so.

WHAT A NARROWER WIDTH CHANGES AT LOAD, AND WHAT IT DOES NOT. The width goes to
narrator as `HIGGS_MAX_NUM_SEQS` (`Residency.load_voice(serving_width=)`), the
lever it already reads. SGLang-Omni's `--mem-fraction-static` is left as the
voice or `[tts.<engine>]` states it: scaling it to a smaller card is a number
nobody has measured on one (its stages split the card between them), so a
narrow-width plan on a card smaller than the 3090 Ti is a DECLARED fit with a
load test owed, and its reason says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The one narrator engine this ladder is for. Any other engine keeps the
#: single estimate its manifest states.
HIGGS_ENGINE = "higgs-v3"

#: What one passage at a time needs at full precision (bf16). Declared: the
#: owens-pc session, 2026-09-26, from the backbone and codec above; unmeasured.
BF16_ONE_AT_A_TIME_BYTES = 10_000_000_000

#: The KV one more passage in flight holds: ~2,000 tokens of a ~60 s chunk at
#: 147,456 B/token (36 x 8 x 128 x 2 x 2 B). Declared, as above.
KV_BYTES_PER_TOKEN = 147_456
TOKENS_PER_PASSAGE = 2_000
BYTES_PER_EXTRA_PASSAGE = KV_BYTES_PER_TOKEN * TOKENS_PER_PASSAGE

#: The quantized rows, one at a time. Declared; no artifact exists.
QUANTIZED_ONE_AT_A_TIME_BYTES: tuple[tuple[int, int], ...] = (
    (8, 5_800_000_000),
    (4, 4_200_000_000),
)

#: Said wherever a pending row is the reason for an answer.
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


@dataclass(frozen=True)
class ServingVariant:
    """One way to serve a voice: a precision and a width, and what it needs."""

    bits: int
    width: int
    need_bytes: int
    #: False for a variant whose weights do not exist yet (`PENDING_NOTE`).
    available: bool
    basis: str

    @property
    def full_precision(self) -> bool:
        return self.bits >= 16

    def label(self) -> str:
        precision = "full quality (bf16)" if self.full_precision else f"{self.bits}-bit"
        pace = "one passage at a time" if self.width == 1 else f"{self.width} passages at a time"
        return f"{precision}, {pace}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "width": self.width,
            "need_bytes": self.need_bytes,
            "available": self.available,
            "basis": self.basis,
        }


def bf16_need(width: int) -> int:
    """What `width` passages at a time need at bf16, below the declared width."""
    return BF16_ONE_AT_A_TIME_BYTES + (width - 1) * BYTES_PER_EXTRA_PASSAGE


def variants(declared_width: int, declared_need_bytes: int) -> tuple[ServingVariant, ...]:
    """Every way to serve a Higgs voice, best first (this module's docstring)."""
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
    """The serving ladder for one voice on one backend, or None where it has none.

    Only a Higgs v3 voice on `cuda-linux`, the arm SGLang-Omni serves with a
    width. The MLX arm sizes its own batch from a measured tier table
    (`engines.narrator.MLX_TIERS`) and keeps its single estimate.
    """
    if backend_kind != "cuda-linux":
        return None
    if getattr(manifest, "narrator_engine", None) != HIGGS_ENGINE:
        return None
    serving = getattr(manifest, "serving", None)
    if serving is None:
        return None
    return variants(serving.max_num_seqs, spec.memory_bytes_estimate)


def choose(ladder: tuple[ServingVariant, ...], budget_bytes: int) -> ServingVariant | None:
    """The first AVAILABLE variant that fits, or None."""
    for variant in ladder:
        if variant.available and variant.need_bytes <= budget_bytes:
            return variant
    return None


def first_fitting(ladder: tuple[ServingVariant, ...], budget_bytes: int) -> ServingVariant | None:
    """The first variant that fits whether or not it exists yet: what WOULD work."""
    for variant in ladder:
        if variant.need_bytes <= budget_bytes:
            return variant
    return None


def explain(ladder: tuple[ServingVariant, ...], chosen: ServingVariant, budget_bytes: int) -> str:
    """Why this variant, in the person's words."""
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
    """What a voice load asks of the card, and the width it starts narrator at.

    The load's half of the capability walk's answer, from the SAME `choose`
    over the same ladder and the same budget, so the verdict a person read and
    the load that follows cannot disagree.
    """

    #: What the accelerator guard is asked for: the chosen variant's need, or
    #: the least that exists when none fits (the guard then refuses by name).
    need_bytes: int
    #: `HIGGS_MAX_NUM_SEQS` for narrator, or None to keep the voice's own.
    width: int | None
    #: What `refuse_if_larger_than_host` checks: the least this voice can be
    #: served in among what exists. "Never on this host" is only true below it.
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
    """The load's plan for one voice on this card. The budget is the capability
    walk's (`capability.available_bytes`: total less the allowance)."""
    ladder = ladder_for(manifest, spec, backend_kind)
    if ladder is None:
        estimate = spec.memory_bytes_estimate
        return LoadPlan(estimate, None, estimate, None)
    floor = min(v.need_bytes for v in ladder if v.available)
    budget = max(0, total_bytes - desktop_allowance_bytes)
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
