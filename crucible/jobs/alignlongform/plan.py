from __future__ import annotations

from dataclasses import dataclass, field

PAD_HEAD = 4.0
PAD_TAIL = 20.0


@dataclass(frozen=True)
class Chunk:

    index: int
    sentences: list[int]
    start: float
    end: float

    @property
    def span(self) -> float:
        return self.end - self.start


@dataclass
class ChunkPlan:
    chunks: list[Chunk] = field(default_factory=list)
    capped_ranges: list[tuple[float, float]] = field(default_factory=list)

    @property
    def capped(self) -> int:
        return len(self.capped_ranges)


def plan_chunks(
    rough: list[float | None],
    first_index: int,
    last_index: int,
    duration: float,
    chunk_s: float,
) -> ChunkPlan:
    narrated = [
        i for i in range(first_index, last_index) if rough[i] is not None
    ]
    plan = ChunkPlan()
    if not narrated:
        return plan

    cur = 0
    base = rough[narrated[0]]
    for x in range(1, len(narrated) + 1):
        if x == len(narrated) or (rough[narrated[x]] - base) >= chunk_s:
            idxs = narrated[cur:x]
            a = max(0.0, rough[idxs[0]] - PAD_HEAD)
            b = min(
                duration,
                (rough[narrated[x]] + PAD_TAIL) if x < len(narrated) else duration,
            )
            if b - a > 2 * chunk_s:
                plan.capped_ranges.append((a, b))
                b = a + 2 * chunk_s
            plan.chunks.append(Chunk(len(plan.chunks), idxs, a, b))
            if x < len(narrated):
                cur = x
                base = rough[narrated[x]]
    return plan


def capped_warning(plan: ChunkPlan, chunk_s: float) -> str | None:
    if not plan.capped_ranges:
        return None
    from .cues import timestamp

    ranges = ", ".join(
        f"[{timestamp(a)}-{timestamp(b)}]" for a, b in plan.capped_ranges
    )
    return (
        f"{plan.capped} chunk(s) exceeded the {2 * chunk_s:.0f}s span cap and "
        f"were truncated — coarse alignment is likely off here: {ranges}"
    )
