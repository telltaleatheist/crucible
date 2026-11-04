"""The interior-pause cap: no rendered chunk carries a stall into the book.

Owen, 2026-10-02 (through training-pc): BookForge should cut pauses longer than
1.5 s in rendered Higgs audio, "for protection". A Mistborn fine-tune trained
without a bed rendered Mutineers' Moon ch. 8 "full of long 10 second pauses",
and its pause screen found interior silences up to 92 s. Higgs returns bare
speech and nothing on the render path looked inside a chunk, so a stall went
straight into the audiobook.

WHAT IS CUT, AND WHAT NEVER IS. Only INTERIOR silence of ONE rendered chunk:
silence that has speech on both sides of it inside the chunk the engine
returned. The silence BETWEEN chunks is the assembler's (`gaps.json`; Owen,
2026-09-18: "whoever assembles them is who owns the gap"), and neither it, a
chapter gap, nor an empty sentence's placeholder ever passes through here as
interior - the placeholder is all zeros, has no peak, and so has no runs.

A PAUSE MEANS WHAT IT MEANS ON THE SCREEN. The definition is orpheus-finetune's
`pipeline/untreated/pause_lib.py`, which `higgs_pause_screen.py` measures with,
so a cut here and a pause there are the same event:

  - non-overlapping 20 ms frames from t = 0 (`HOP_S`); a trailing part-frame
    is not a frame;
  - a frame is silent when its RMS is below the clip's own sample peak less
    40 dB (`SIL_DB`);
  - a run is interior when it starts after 1.5 hops and ends before the last
    1.5 hops (`_pauses`' `edge`).

THE CUT keeps both edges of the pause - the first half and the last half of the
allowed length - so a word's decay and the breath before the next phrase
survive, removes the middle, and joins the two halves with a raised-cosine
crossfade (`assemble/edges.py`'s window, for its reason: no slope corner where
the splice meets). The result is EXACTLY the allowed length, sample for sample.

ONE SETTING. `NARRATOR_MAX_INTERIOR_PAUSE_S`, default 1.5 s; `0` switches the
cap off. It is read per row through narrator's one env reader (`narrator/env.py`),
so a value that is not a number is refused by name - a typo is a
misconfiguration, not a reason to cap at some other length. If it ever becomes
a property of a VOICE, it belongs in that voice's `crucible-voice.toml`, where
every voice fact lives (2026-09-28), and not here.

THE TRADE, said once: real narrators sometimes pause longer than 1.5 s
(training-pc, 2026-10-02: Third Reich's corpus 7.2 pauses >= 2 s per 10k chars,
max 3.7 s; Mistborn 1.1, max 4.4 s). At 1.5 s some genuine dramatic pauses are
shortened too. Every cut is reported, so that is visible rather than silent.
"""

from __future__ import annotations

import numpy as np

from ..env import env_number

#: pause_lib.SIL_DB - a frame is silent this far below the clip's own peak.
SIL_DB = -40.0
#: pause_lib.HOP_S - the frame, 20 ms, non-overlapping.
HOP_S = 0.02
#: higgs_pause_screen._pauses' edge: a run must start after, and end before, this.
EDGE_S = HOP_S * 1.5

#: The default longest interior pause, in seconds (Owen, 2026-10-02).
DEFAULT_MAX_PAUSE_S = 1.5
#: The crossfade at the splice. Inside silence either side of it, so short.
CROSSFADE_MS = 15.0
#: The one setting.
ENV_MAX_PAUSE = 'NARRATOR_MAX_INTERIOR_PAUSE_S'


def max_pause_seconds() -> float | None:
    """The configured cap in seconds, or None when it is switched off (0)."""
    value = env_number(ENV_MAX_PAUSE, DEFAULT_MAX_PAUSE_S, float, 0,
                       'the longest interior pause a rendered chunk may keep, in '
                       'seconds (0 switches the cap off)')
    return None if value == 0 else value


def silence_runs(audio: np.ndarray, rate: int) -> list[tuple[int, int]]:
    """pause_lib.silence_runs, in SAMPLES: [(start, end)) of each silent run."""
    n = max(1, int(round(HOP_S * rate)))
    frames = len(audio) // n
    if frames < 1:
        return []
    peak = float(np.abs(audio).max())
    if peak <= 0:
        return []
    rms = np.sqrt((audio[:frames * n].astype(np.float64).reshape(frames, n) ** 2).mean(1))
    quiet = rms < peak * (10 ** (SIL_DB / 20.0))
    runs: list[tuple[int, int]] = []
    k = 0
    while k < frames:
        if not quiet[k]:
            k += 1
            continue
        s = k
        while k < frames and quiet[k]:
            k += 1
        runs.append((s * n, k * n))
    return runs


def interior_pauses(audio: np.ndarray, rate: int) -> list[tuple[int, int]]:
    """The silent runs with speech on both sides, as higgs_pause_screen counts them."""
    edge = EDGE_S * rate
    total = len(audio)
    return [(s, e) for s, e in silence_runs(audio, rate) if s > edge and e < total - edge]


def _crossfade(out_of: np.ndarray, into: np.ndarray) -> np.ndarray:
    """Raised-cosine crossfade of two equal-length pieces (assemble/edges.py's window)."""
    n = len(out_of)
    t = (np.arange(n, dtype=np.float64) + 1.0) / (n + 1.0)
    rise = 0.5 - 0.5 * np.cos(np.pi * t)
    return (out_of * (1.0 - rise) + into * rise).astype(out_of.dtype, copy=False)


def cap_interior_pauses(audio: np.ndarray, rate: int,
                        max_pause_s: float | None = DEFAULT_MAX_PAUSE_S,
                        crossfade_ms: float = CROSSFADE_MS) -> tuple[np.ndarray, list[dict]]:
    """Shorten every interior pause longer than `max_pause_s` to exactly that.

    Returns the audio (the input itself when nothing was cut) and one record per
    cut, in the chunk's order: `atS` (where the pause started in the audio as
    rendered), `fromS` (its length as rendered) and `toS` (its length now).
    """
    if max_pause_s is None:
        return audio, []
    a = np.asarray(audio)
    target = int(round(max_pause_s * rate))
    fade = min(int(round(crossfade_ms * rate / 1000.0)), target)
    long_runs = [(s, e) for s, e in interior_pauses(a, rate) if e - s > target]
    if not long_runs:
        return audio, []
    pieces: list[np.ndarray] = []
    cuts: list[dict] = []
    at = 0
    for s, e in long_runs:
        # Keep head + tail = target + fade, overlapped by `fade`: exactly `target`.
        head = (target + fade) // 2
        tail = target + fade - head
        pieces.append(a[at:s + head - fade])
        if fade > 0:
            pieces.append(_crossfade(a[s + head - fade:s + head], a[e - tail:e - tail + fade]))
        at = e - tail + fade
        cuts.append({
            'atS': round(s / rate, 3),
            'fromS': round((e - s) / rate, 3),
            'toS': round(target / rate, 3),
        })
    pieces.append(a[at:])
    return np.concatenate(pieces), cuts
