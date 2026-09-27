from __future__ import annotations

import bisect
from dataclasses import dataclass, field


def timestamp(t: float) -> str:
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"


@dataclass
class SnapStats:
    considered: int = 0
    snapped: int = 0
    moved_seconds: list[float] = field(default_factory=list)


def snap_boundaries(
    starts: list[float],
    ends: list[float],
    silences: list[tuple[float, float]],
    window: float,
    min_gap: float = 0.05,
) -> tuple[list[float], list[float], SnapStats]:
    n = len(starts)
    if n == 0 or not silences or window <= 0:
        return starts, ends, SnapStats()
    sil_starts = [a for a, _ in silences]
    ns, ne = list(starts), list(ends)
    moved: list[float] = []
    considered = 0
    for i in range(n - 1):
        b = ne[i]
        if abs(starts[i + 1] - ends[i]) > 1e-6:
            continue
        considered += 1
        lo, hi = b - window, b + window
        lo = max(lo, ns[i] + min_gap)
        hi = min(hi, (ends[i + 1] if i + 1 < n else b) - min_gap)
        if hi <= lo:
            continue
        j = bisect.bisect_left(sil_starts, b)
        best: tuple[float, float] | None = None
        for k in range(max(0, j - 2), min(len(silences), j + 2)):
            a, z = silences[k]
            oa, oz = max(a, lo), min(z, hi)
            if oz <= oa:
                continue
            mid = 0.5 * (oa + oz)
            d = abs(mid - b)
            if best is None or d < best[0]:
                best = (d, mid)
        if best is None or best[0] < 1e-3:
            continue
        ne[i] = best[1]
        ns[i + 1] = best[1]
        moved.append(round(best[1] - b, 3))
    return ns, ne, SnapStats(considered, len(moved), moved)


class NoCues(Exception):
    ...


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str
    kind: str = "prose"
    note: str | None = None


def write_vtt(cues: list[Cue]) -> str:
    if not cues:
        raise NoCues(
            "alignment produced 0 cues — no sentence could be matched to the "
            "audio. A bare WEBVTT is not a transcript, so it is not written: a "
            "two-line file here would be indistinguishable from success"
        )
    lines = ["WEBVTT", ""]
    for n, cue in enumerate(sorted(cues, key=lambda c: c.start), start=1):
        if cue.kind == "heading":
            lines += ["NOTE heading", ""]
        if cue.note:
            lines += [cue.note, ""]
        lines += [str(n), f"{timestamp(cue.start)} --> {timestamp(cue.end)}", cue.text, ""]
    return "\n".join(lines)
