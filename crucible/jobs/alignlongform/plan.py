"""Turning coarse times into the chunks the `align` worker is given.

The glue between stage 2 and stage 3, ported from bookforge
`electron/scripts/align_audiobook.py` (the chunk loop around line 1854).

This is where `align-longform` meets the `align` job type that already exists:
each chunk here becomes one `{audio, text}` entry on that worker's wire, and its
span is the seconds of audio sliced out for it. Crucible already holds
Qwen3-ForcedAligner resident for exactly this shape of request, so the aligner
does not have to be built again — only fed.

The span cap is a safety net, not a tuning knob
------------------------------------------------
wav2vec2 memory is QUADRATIC in audio span, so a coarse regression that puts two
adjacent sentences implausibly far apart in audio would otherwise produce a
memory-bomb chunk. `2 * chunk_s` caps it, and the capped ranges are reported by
time so a person can seek straight to the suspect stretch — a cap firing means
the coarse alignment is wrong there, and the number to look at is the audio
range, not the chunk index.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Seconds of audio kept before the first sentence of a chunk and after the last,
#: VERBATIM from the original (`align_audiobook.py:217`, `PAD_HEAD, PAD_TAIL =
#: 4.0, 20.0`). They are not symmetric and they are not small: the head needs
#: only enough room to place the first phone, while the TAIL has to cover
#: however long the last sentence of the window actually runs, because the chunk
#: boundary is drawn at the NEXT sentence's rough start and that start is an
#: estimate.
#:
#: Written down rather than chosen: this file first carried 0.30 / 0.60, which
#: were invented here and would have cut every window short of the audio its
#: last sentence needs. A test caught it. The rule this file already states for
#: `coarse.py` applies to two constants just as much as to an algorithm — port
#: them, do not re-derive them.
PAD_HEAD = 4.0
PAD_TAIL = 20.0


@dataclass(frozen=True)
class Chunk:
    """One window of audio and the sentences believed to be inside it."""

    index: int
    #: Sentence indexes, in reading order.
    sentences: list[int]
    start: float
    end: float

    @property
    def span(self) -> float:
        return self.end - self.start


@dataclass
class ChunkPlan:
    chunks: list[Chunk] = field(default_factory=list)
    #: `(start, end)` of chunks whose ORIGINAL span exceeded the cap, before
    #: truncation — the audio ranges to go and listen to.
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
    """Group the narrated sentences into aligner-sized windows.

    Only sentences with a rough time inside `[first_index, last_index)` are
    chunked — a `None` is text the narrator never read, and including it would
    put unspoken words inside a window and drag the alignment (the Well of
    Ascension shape, `coarse.py`).
    """
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
                # The ORIGINAL span is recorded, then the chunk is truncated.
                plan.capped_ranges.append((a, b))
                b = a + 2 * chunk_s
            plan.chunks.append(Chunk(len(plan.chunks), idxs, a, b))
            if x < len(narrated):
                cur = x
                base = rough[narrated[x]]
    return plan


def capped_warning(plan: ChunkPlan, chunk_s: float) -> str | None:
    """The sentence a person can act on, or None when nothing was capped.

    Names the audio TIME RANGES rather than chunk indexes, deliberately: a cap
    means the coarse anchors are wrong somewhere, and the only way to check is
    to go and listen to that stretch.
    """
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
