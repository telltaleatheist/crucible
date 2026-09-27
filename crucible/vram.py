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
    return max(0, min(total_bytes - desktop_allowance_bytes, free_bytes))


def max_num_seqs(spec: BackendSpec) -> int | None:
    args = list(spec.engine_args)
    for index, arg in enumerate(args):
        if arg == "--max-num-seqs" and index + 1 < len(args):
            return int(args[index + 1])
        if arg.startswith("--max-num-seqs="):
            return int(arg.split("=", 1)[1])
    return None


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
        return self.pool_bytes >= self.kv_bytes_per_token * self.context

    @property
    def affordable_context(self) -> int:
        return self.pool_bytes // self.kv_bytes_per_token

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


def plan_vllm_memory(
    *,
    model_id: str,
    spec: BackendSpec,
    context: int,
    card: AcceleratorState,
    desktop_allowance_bytes: int,
    reclaimable_bytes: int,
) -> KvPlan | None:
    if spec.engine != "vllm":
        return None
    terms: MemoryTerms | None = spec.memory
    if terms is None:
        return None

    free_after_eviction = card.free_bytes + reclaimable_bytes
    budget = engine_budget_bytes(
        card.total_bytes, desktop_allowance_bytes, free_after_eviction
    )
    room = max(0, budget - terms.fixed_bytes)
    concurrency = max_num_seqs(spec)
    if concurrency is not None:
        room = min(room, terms.kv_bytes_per_token * context * concurrency)

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
