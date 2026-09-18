"""How many bytes a vLLM engine may have on THIS card, RIGHT NOW.

This module exists because of one measured fact (docs/MEASUREMENTS.md,
2026-09-18): **inside WSL2, CUDA cannot see the Windows desktop.**

    nvidia-smi --query-gpu=memory.free   21_254 / 21_352 / 21_336 MiB
    torch.cuda.mem_get_info()            23_285 / 23_285 / 23_285 MiB

Three samples seconds apart. nvidia-smi's figure moves with the desktop; CUDA's
does not move at all, because 24_564 − 23_285 is a fixed driver reservation and
not a reading. Windows' own nvidia-smi agrees with WSL's, so this is not a
Windows-versus-Linux disagreement — the guest's CUDA runtime is simply blind to
the host compositor.

Every number vLLM sizes an engine with comes from `mem_get_info`:

    requested      = total x gpu_memory_utilization      (worker/utils.py)
    total_consumed = free_at_init − free_after_profile   (utils/mem_utils.py)
    non_kv         = total_consumed + transient_peak_headroom
    available_kv   = requested − non_kv − cudagraph_estimate

So the engine CANNOT discover the desktop for itself, and its startup gate
(`free >= requested`) can never refuse on account of it. Crucible reads the card
with nvidia-smi — `accelerator.py`, which has always been right about this — and
has to tell the engine what it found.

WHY THE FRACTION WAS NEVER THE PROBLEM
--------------------------------------
`qwen3.5-9b` failed to load on 2026-09-17 at its manifest's own
`--gpu-memory-utilization 0.84` (`Available KV cache memory: -0.19 GiB`), and
the identical argv succeeded the next night at `+1.94 GiB` with the desktop the
same size. The term that moved was `total_consumed`, which is a WHOLE-CARD delta
across the profiling window: the longer the load, the more of somebody else's
allocation is charged to our KV pool. Last night's load was a cold
`torch.compile` — *"Initial profiling/warmup run took 65.00 s"* — and about
2.1 GiB of desktop growth landed inside it.

Raising the fraction would buy a bigger pool on a quiet machine and fail again on
a busy one, because the term that moves is not in the fraction at all.

WHAT THIS MODULE DOES INSTEAD
-----------------------------
vLLM 0.29 takes `--kv-cache-memory-bytes`, and its own config doc says it
*"(when not-None) ignores gpu_memory_utilization"*; `gpu_worker.py:527` skips
memory profiling entirely when it is set. That removes the mechanism rather than
tuning it: no fraction of a shared card, and no profiling window for anyone
else's allocation to land in.

The cost is that Crucible must know the intercept and the slope itself — which
is exactly what `[backends.<kind>.memory]` holds and what the two-point
calibration measures (docs/FITS-AND-THE-CARD.md section 4). The two halves were
built for each other.

WHAT THIS MODULE IS NOT
-----------------------
**It is not capability selection, and it must never be used for it.**
`capability.py`'s ruling 2 is explicit: *"The bar is TOTAL memory, not free
memory. A capability is a fact about the host; free VRAM is a fact about this
second."* A browser open during `crucible install` must not permanently disable
TTS on a 24 GB card. `capability.decide()` therefore stays on `total_bytes` and
does not import anything from here.

This module answers a different question, at a different time, for a different
consumer: not *can this host run this model* but *how many bytes do I hand this
engine in the next second*. That one is a fact about this second, and free is
the right reading for it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .accelerator import AcceleratorState
from .manifests import BackendSpec, MemoryTerms

GIB = 1024**3


def _gib(value: int) -> str:
    return f"{value / GIB:.2f} GiB"


def engine_budget_bytes(
    total_bytes: int, desktop_allowance_bytes: int, free_bytes: int
) -> int:
    """What the engine may hold in total: the smaller of the two true answers.

    Both terms are consulted and the SMALLER wins:

      * the **measurement**, because the desktop may be over its allowance and a
        budget larger than the card has is a load that fails;
      * the **allowance**, because the desktop may be UNDER it right now and grow
        afterwards. `total − allowance` holds that room open.

    There is no third term. `capability.py`'s ruling 3 settled that: *"There is
    no `margin` term, and its absence is the ruling rather than an omission…
    Adding a second reserve on top of the first would be a number nobody has
    measured, invented to feel safe."* The allowance IS the margin, and it is
    the growth reserve as well.

    Measured, Owen's PC, 2026-09-18: total 24_564 MiB, allowance 3_072 MiB, free
    21_300 MiB, so the budget is 21_300 MiB — the measurement, because the
    desktop was over its allowance. On 2026-09-17, with 21_234 MiB free, it would
    have been 21_234 where the allowance alone answered 21_492: 258 MiB more than
    the card had.
    """
    return max(0, min(total_bytes - desktop_allowance_bytes, free_bytes))


def max_num_seqs(spec: BackendSpec) -> int | None:
    """`--max-num-seqs` off this block's own args, or None if it says nothing.

    Read rather than assumed, because it is the ceiling on how much KV the engine
    could ever USE: reserving beyond `context x max_num_seqs` would hold bytes
    nothing can reach. `qwen3.5-9b` sets 16 and says why in place — the number of
    CUDA graphs captured follows it, and 16 is what BookForge's own text server
    runs on this card.

    None is not a default of 1. It means this block does not state the number, so
    there is no ceiling to apply and the engine may have the whole budget.
    """
    args = list(spec.engine_args)
    for index, arg in enumerate(args):
        if arg == "--max-num-seqs" and index + 1 < len(args):
            return int(args[index + 1])
        if arg.startswith("--max-num-seqs="):
            return int(arg.split("=", 1)[1])
    return None


@dataclass(frozen=True)
class KvPlan:
    """The KV pool Crucible intends to give one engine, with every term shown.

    Every refusal and every log line about memory is built from this object, so
    the numbers a client is told are the numbers that were used.
    """

    model_id: str
    total_bytes: int
    free_bytes: int
    desktop_allowance_bytes: int
    budget_bytes: int
    fixed_bytes: int
    kv_bytes_per_token: int
    basis: str
    context: int
    concurrency: int | None
    pool_bytes: int

    @property
    def fits(self) -> bool:
        """Can the engine serve even ONE request at its own context?

        This is vLLM's `_check_enough_kv_cache_memory` condition, asked before
        the engine starts instead of two minutes into a load. A pool that cannot
        hold one full-context request is the same refusal either way; the
        difference is who says it, when, and whether the sentence names numbers.
        """
        return self.pool_bytes >= self.kv_bytes_per_token * self.context

    @property
    def affordable_context(self) -> int:
        """The tallest context this pool affords at one request in flight."""
        return self.pool_bytes // self.kv_bytes_per_token

    def sentence(self) -> str:
        """Why the card cannot serve this model at this context, in full.

        Named terms, in the order they were computed, because a refusal that
        says only "not enough memory" costs the next reader a measurement.
        """
        return (
            f"cannot give {self.model_id} a KV cache at {self.context} tokens: "
            f"the card has {_gib(self.free_bytes)} free of "
            f"{_gib(self.total_bytes)} and the desktop's allowance is "
            f"{_gib(self.desktop_allowance_bytes)}, so the engine's budget is "
            f"{_gib(self.budget_bytes)}; {_gib(self.fixed_bytes)} of that is "
            f"weights and overhead, leaving {_gib(self.pool_bytes)} for KV where "
            f"one request of {self.context} tokens needs "
            f"{_gib(self.kv_bytes_per_token * self.context)} at "
            f"{self.kv_bytes_per_token:_} B/token ({self.basis}). "
            f"This card affords {self.affordable_context:_} tokens"
        )

    def detail(self) -> str:
        """The one-line version for the engine log and the warming progress."""
        held = f" x {self.concurrency} in flight" if self.concurrency else ""
        return (
            f"{self.model_id}: KV pool {_gib(self.pool_bytes)} of a "
            f"{_gib(self.budget_bytes)} budget ({_gib(self.free_bytes)} free, "
            f"{_gib(self.fixed_bytes)} weights+overhead), which is "
            f"{self.affordable_context:_} tokens at {self.kv_bytes_per_token:_} "
            f"B/token ({self.basis}){held}"
        )

    def flags(self) -> list[str]:
        """What to put on the engine's command line.

        BOTH flags, and they do different jobs. `--kv-cache-memory-bytes` is the
        pool and ignores the utilisation. `--gpu-memory-utilization` is still
        read by `request_memory()` in `init_device`, which raises if
        `free < total x util` — so it remains a GATE, derived from the same
        budget, and stops being the sizing knob it was never able to be.
        """
        return [
            "--kv-cache-memory-bytes",
            str(self.pool_bytes),
            "--gpu-memory-utilization",
            f"{self.budget_bytes / self.total_bytes:.4f}",
        ]


def plan_vllm_memory(
    *,
    model_id: str,
    spec: BackendSpec,
    context: int,
    card: AcceleratorState,
    desktop_allowance_bytes: int,
) -> KvPlan | None:
    """Size this engine's KV pool against the card, or None to leave it alone.

    **None is a real answer, not a failure.** Two blocks in this catalog cannot
    be planned and each says why in its own manifest:

      * `dots-ocr`'s cuda-linux estimate IS a budget — `0.5 x` the card, chosen
        so page reading can share the machine — rather than a sum of terms, so
        there is nothing to take apart and its fraction is a DECISION that must
        be left alone.
      * every `llama-windows` block, because llama-server is not vLLM.

    A block with no `[memory]` table keeps the args its manifest states, exactly
    as before this module existed. Nothing is invented for it.
    """
    if spec.engine != "vllm":
        return None
    terms: MemoryTerms | None = spec.memory
    if terms is None:
        return None

    budget = engine_budget_bytes(
        card.total_bytes, desktop_allowance_bytes, card.free_bytes
    )
    room = max(0, budget - terms.fixed_bytes)
    concurrency = max_num_seqs(spec)
    if concurrency is not None:
        # Never reserve more than the engine could reach: past
        # `context x max_num_seqs` the pool holds bytes no request can use, and
        # on a shared card those are bytes the desktop wanted.
        room = min(room, terms.kv_bytes_per_token * context * concurrency)

    return KvPlan(
        model_id=model_id,
        total_bytes=card.total_bytes,
        free_bytes=card.free_bytes,
        desktop_allowance_bytes=desktop_allowance_bytes,
        budget_bytes=budget,
        fixed_bytes=terms.fixed_bytes,
        kv_bytes_per_token=terms.kv_bytes_per_token,
        basis=terms.basis,
        context=context,
        concurrency=concurrency,
        pool_bytes=room,
    )
