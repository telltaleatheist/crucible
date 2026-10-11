from __future__ import annotations

from dataclasses import dataclass

from .accelerator import AcceleratorState
from .enginespec import VLLM_ENGINE, flag_value
from .manifests import MODELS_DIR_ENV, BackendSpec, ManifestError
from .memorybudget import engine_budget_bytes, gib_text

MAX_NUM_SEQS_FLAG = "--max-num-seqs"

# What vLLM's own process holds on the card before it checks that the budget is free:
# its CUDA context. vLLM refuses to start when gpu_memory_utilization x total exceeds the
# free memory IT sees, which is the probe's free less this. Measured on Victoria's RTX
# 3070 Laptop (WSL2, 2026-10-09): the probe saw 7.83 GiB free, vLLM 6.95 GiB, so 0.88
# GiB; rounded up to the next 64 MiB. A budget of the whole 7.0 GiB the desktop
# allowance leaves was refused there; on a 24 GiB card the allowance is always the
# smaller term, which is why it never showed.
VLLM_CUDA_CONTEXT_BYTES = 960 * 1024**2

# What vLLM needs per in-flight request beyond context x bytes-per-token, in tokens of
# KV. On Qwen3.5 (hybrid attention) vLLM 0.29 pads attention to 528-token blocks to
# match the mamba page, and holds each request's linear-attention state inside the KV
# pool. Victoria's RTX 3070, 2026-10-09, qwen3.5-4b-bside-4bit at 8192 tokens x 1 with a
# pool of exactly 8192 x 40,337 B: "0.35 GiB KV cache is needed, which is larger than the
# available KV cache memory (0.29 GiB)". One block of padding (528) and the state
# (0.019 GiB, about 430 tokens at 46,581 B/token) fit in 1024.
VLLM_SEQUENCE_SLACK_TOKENS = 1024


def _gib(value: int) -> str:
    return gib_text(value, 2)


def max_num_seqs(spec: BackendSpec, model_id: str | None = None) -> int | None:
    stated = flag_value(list(spec.engine_args), MAX_NUM_SEQS_FLAG)
    if stated is None:
        return None
    try:
        return int(stated)
    except ValueError:
        whose = "this model" if model_id is None else model_id
        raise ManifestError(
            f"{whose}'s engine_args give {MAX_NUM_SEQS_FLAG} {stated!r}, which is "
            "not a whole number of requests; set it to one (for example "
            f"{MAX_NUM_SEQS_FLAG} 16) in {whose}.toml in the model manifests "
            f"(crucible/models, or ${MODELS_DIR_ENV} where it is set)"
        ) from None


@dataclass(frozen=True)
class KvPlan:
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
        return self.pool_bytes >= self.kv_bytes_per_token * (
            self.context + VLLM_SEQUENCE_SLACK_TOKENS
        )

    @property
    def affordable_context(self) -> int:
        return max(0, self.pool_bytes // self.kv_bytes_per_token - VLLM_SEQUENCE_SLACK_TOKENS)

    def sentence(self) -> str:
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
        held = f" x {self.concurrency} in flight" if self.concurrency else ""
        return (
            f"{self.model_id}: KV pool {_gib(self.pool_bytes)} of a "
            f"{_gib(self.budget_bytes)} budget ({_gib(self.free_bytes)} free, "
            f"{_gib(self.fixed_bytes)} weights+overhead), which is "
            f"{self.affordable_context:_} tokens at {self.kv_bytes_per_token:_} "
            f"B/token ({self.basis}){held}"
        )

    def flags(self) -> list[str]:
        return [
            "--kv-cache-memory-bytes",
            str(self.pool_bytes),
            "--gpu-memory-utilization",
            f"{self.budget_bytes / self.total_bytes:.4f}",
        ]


def plans_memory(spec: BackendSpec) -> bool:
    return spec.engine == VLLM_ENGINE and spec.memory is not None


def plan_vllm_memory(
    *,
    model_id: str,
    spec: BackendSpec,
    context: int,
    card: AcceleratorState,
    desktop_allowance_bytes: int,
    reclaimable_bytes: int,
) -> KvPlan | None:
    """The engine's budget and KV pool on ``card``.

    The plan vLLM is STARTED with is made from a card read after the eviction, with
    ``reclaimable_bytes=0`` (crucible/jobs/llm ``plan_on_the_card``): vLLM refuses to start
    when its budget exceeds the free memory it sees, so the free term must be what the card
    has, not a forecast. ``reclaimable_bytes`` is for the forecast a load makes before it
    evicts (whether the model can have a KV cache at this context at all): it is the
    resident's estimate or load-time measurement, not what it holds now, so free plus it can
    exceed what the card will have. Victoria's RTX 3070, 2026-10-11: qwen3.5-4b-8bit planned
    with the evicted resident's figure added got the whole 7.0 GiB the allowance leaves
    (0.875) and vLLM, seeing 6.95 GiB, refused to start."""
    if not plans_memory(spec):
        return None
    terms = spec.memory
    assert terms is not None

    # The resident cannot hand back more than the card has in use: a reclaimable figure
    # past that (an estimate larger than what a staged engine holds) is not memory.
    free_after_eviction = card.free_bytes + min(reclaimable_bytes, card.used_bytes)
    budget = engine_budget_bytes(
        card.total_bytes,
        desktop_allowance_bytes,
        free_after_eviction - VLLM_CUDA_CONTEXT_BYTES,
    )
    room = max(0, budget - terms.fixed_bytes)
    concurrency = max_num_seqs(spec, model_id)
    if concurrency is not None:
        room = min(
            room,
            terms.kv_bytes_per_token
            * (context + VLLM_SEQUENCE_SLACK_TOKENS)
            * concurrency,
        )

    return KvPlan(
        model_id=model_id,
        total_bytes=card.total_bytes,
        free_bytes=free_after_eviction,
        desktop_allowance_bytes=desktop_allowance_bytes,
        budget_bytes=budget,
        fixed_bytes=terms.fixed_bytes,
        kv_bytes_per_token=terms.kv_bytes_per_token,
        basis=terms.basis,
        context=context,
        concurrency=concurrency,
        pool_bytes=room,
    )
