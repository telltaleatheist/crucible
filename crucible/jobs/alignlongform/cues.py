"""`write`: cue seams onto silence, and the VTT that comes out.

Ported from bookforge `electron/scripts/align_audiobook.py` — `snap_boundaries`,
`ts` and the VTT emission — under the same discipline as `coarse.py`: verbatim,
with the reasoning carried across rather than summarised, and checked against the
original by `tests/test_align_longform_port_fidelity.py`.

The one refusal that matters
----------------------------
A run that places no cue must FAIL, not write a two-line file. The original says
it plainly — *"A bare WEBVTT is not a transcript — refuse to write it and claim
success"* — and it is the same shape as every other refusal in this job type: a
mis-aligned or empty transcript reports success, so the only place to catch it is
before the artifact exists.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field


def timestamp(t: float) -> str:
    """`HH:MM:SS.mmm`, the original's `ts` verbatim."""
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
    """Pull each cue SEAM onto the middle of a nearby silence.

    CONTIGUOUS MODE ONLY. In the default build each cue's two edges are placed
    independently inside the pause, which is strictly better — a shared seam has
    to be one compromise time for two cues, and the pause belongs to neither
    sentence. In contiguous mode cue i ends exactly where cue i+1 begins, so a
    boundary is ONE time shared by two cues, and it is precisely the seam a
    training-corpus cutter cuts on.

    Forced alignment puts that seam at the CTC frame where the model thinks the
    last phone ended, which routinely lands a couple hundred ms early (clipping
    the word's tail) or late (leaking the next word's onset). The narrator's
    actual pause is a silence, and its MIDDLE is the safest place to cut:
    maximum margin on both sides, so neither clip loses a phone or gains half a
    breath.

    Rules, all of them conservative:
      * only silences OVERLAPPING `[B-window, B+window]` are candidates, so a
        snap can never move a boundary further than `window` and cannot create
        drift;
      * the target is the midpoint of the candidate CLIPPED to that window, so a
        long silence (a chapter gap) pulls the seam to the window edge rather
        than to its own distant centre;
      * the nearest candidate wins;
      * monotonicity is enforced against the already-placed previous boundary
        and the following raw one, leaving `min_gap` so no cue collapses to zero.

    Mutates nothing.
    """
    n = len(starts)
    if n == 0 or not silences or window <= 0:
        return starts, ends, SnapStats()
    sil_starts = [a for a, _ in silences]
    ns, ne = list(starts), list(ends)
    moved: list[float] = []
    considered = 0
    for i in range(n - 1):
        b = ne[i]
        # Cues are only "seamed" when the next starts where this one ends; a gap
        # (whisper-fallback retraction, a cue-length cap) is left alone.
        if abs(starts[i + 1] - ends[i]) > 1e-6:
            continue
        considered += 1
        lo, hi = b - window, b + window
        # Bound the move by the neighbours so ordering and non-empty cues survive.
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
    """Raised instead of writing a WEBVTT with nothing in it."""


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str
    #: `heading` marks a cue the extractor already tagged; carried, never inferred.
    kind: str = "prose"
    #: A provenance line written above the cue, as the original's align NOTE.
    note: str | None = None


def write_vtt(cues: list[Cue]) -> str:
    """The VTT text, cues in time order, NOTEs preserved.

    Refuses an empty result by name. The original's reason is the one that
    matters: a bare `WEBVTT` file is a successful-looking run that aligned
    nothing, and the caller cannot tell it from a book with no speech in it.
    """
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
