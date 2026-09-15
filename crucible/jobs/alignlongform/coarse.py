"""`coarse-align`: sentence -> rough audio time, drift-proof at book scale.

A FAITHFUL PORT of bookforge `electron/scripts/align_audiobook.py:coarse_align`,
and the word faithful is doing work. Every constant and every test in here was
put there by a book that went wrong, and the comments naming those books are
carried across verbatim rather than summarised — the 8.6 % match rate, the
GraphicAudio recap that measured 212 tok/s, the Well of Ascension front matter
that dragged chapter 1 eighty-five seconds late.

This session already paid for the alternative. The MLX backend rendered seven
times slower than it should have because a width was re-derived on the Crucible
side instead of carried over from the app that had measured it; the fix was to
transcribe BookForge's table rather than tidy it. Same discipline here: this is a
transcription, not a redesign. Where the original is odd, the oddness is kept.

Why this is the piece worth having on the server
------------------------------------------------
`align-longform`'s four stages are `transcribe` -> `coarse-align` -> `align` ->
`write`, and two of them already exist as Crucible job types: `asr` runs
faster-whisper with word timestamps, and `align` holds Qwen3-ForcedAligner
resident. So the stack §B7 worried about ("it needs BOTH stacks on the server")
is already here, in two envs that are already built. What was missing is exactly
this: the pure logic between them, which needs no card and no env at all.

What it takes and what it gives
-------------------------------
`sents` are the book's sentences in reading order; `words` is the rough
transcript as `(word, time)` pairs, which is what `asr` returns with
`word_timestamps`. It answers a rough time per sentence — `None` where the
sentence is not in the audio at all, which is a real answer and the point of
half the code below.
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass

#: Local search window and the confirm span, verbatim from the original.
BACK, FWD, SPAN = 8, 60, 14

#: A trigram appearing more often than this is too common to anchor on.
MAX_ANCHOR_OCCURRENCES = 50

#: The fallback narration rate, tokens/second, and the band a measured one is
#: clamped into. Outside it the measurement is not believed — see `rate`.
DEFAULT_RATE = 2.5
RATE_MIN, RATE_MAX = 0.8, 8.0


def _norm(word: str) -> str:
    """Fold a word to its comparable form.

    Kept deliberately simple and matching the original's `_norm`/`toks` pair:
    strip accents, lowercase, drop everything that is not a letter or digit.
    """
    decomposed = unicodedata.normalize("NFKD", word)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^0-9a-z]+", "", stripped.lower())


def toks(text: str) -> list[str]:
    """The original's tokeniser: normalised words, empties dropped."""
    return [t for t in (_norm(w) for w in text.split()) if t]


@dataclass(frozen=True)
class CoarseResult:
    """What the stage answers.

    `direct` is not decoration. True means the sentence's OWN opening was found
    in the transcript word stream, so `rough` is a real spoken start (~±0.5 s of
    audio truth) rather than a token-weighted interpolation — and the `align`
    stage trusts a direct time over the forced aligner when the two disagree.
    """

    rough: list[float | None]
    first_index: int
    last_index: int
    dropped: int
    rate: float
    direct: list[bool]


def coarse_align(
    sents: list[str],
    words: list[tuple[str, float]],
    failed_ranges: tuple[tuple[float, float], ...] = (),
) -> CoarseResult:
    """PASS 1 global anchors, PASS 2 local fill. See the module docstring.

    PASS 1 indexes the transcript's 3-grams, confirms each sentence's opening
    trigram candidates with an ordered-hit check (tolerating one miss for
    transcription errors), then keeps the LONGEST INCREASING SUBSEQUENCE over
    (sentence asc, word position asc) — spurious matches die structurally
    instead of derailing a running pointer.

    PASS 2 runs the small-window walk BETWEEN consecutive anchors, constrained
    to their word range, so dead-reckoning drift is bounded by anchor spacing
    rather than by the whole book. That is the failure mode that flatlined a
    10k-sentence run at an 8.6 % match rate.
    """
    word_time = [t for _, t in words]
    word_norm = [w for w, _ in words]
    total_words = len(word_norm)
    n = len(sents)
    tk_all = [toks(s) for s in sents]

    rough: list[float | None] = [None] * n
    #: The word-stream index behind each matched rough time.
    roughj: list[int | None] = [None] * n
    direct = [False] * n

    def hits(j: int, tk: list[str], need: int) -> int:
        """Ordered token hits within SPAN words starting at j."""
        k = j
        m = 0
        while k < min(total_words, j + SPAN) and m < need:
            if word_norm[k] == tk[m]:
                m += 1
            k += 1
        return m

    # ── PASS 1 — global anchors ────────────────────────────────────────────
    tri: dict[tuple[str, str, str], list[int]] = {}
    for j in range(total_words - 2):
        tri.setdefault(
            (word_norm[j], word_norm[j + 1], word_norm[j + 2]), []
        ).append(j)

    cands: list[tuple[int, int]] = []
    for si in range(n):
        tk = tk_all[si]
        if len(tk) < 4:
            continue
        pos = tri.get((tk[0], tk[1], tk[2]))
        if not pos or len(pos) > MAX_ANCHOR_OCCURRENCES:
            continue
        need = min(len(tk), 6)
        for j in pos:
            if hits(j, tk, need) >= max(3, need - 1):
                cands.append((si, j))

    # LIS: sort (si asc, j desc), patience over strictly-increasing j — the desc
    # tie-break means a chain can keep at most one candidate per sentence.
    cands.sort(key=lambda c: (c[0], -c[1]))
    tails: list[int] = []
    tidx: list[int] = []
    parent = [-1] * len(cands)
    for i, (_si, j) in enumerate(cands):
        p = bisect.bisect_left(tails, j)
        if p == len(tails):
            tails.append(j)
            tidx.append(i)
        else:
            tails[p] = j
            tidx[p] = i
        parent[i] = tidx[p - 1] if p > 0 else -1

    anchors: list[tuple[int, int]] = []
    i = tidx[-1] if tidx else -1
    while i != -1:
        anchors.append(cands[i])
        i = parent[i]
    anchors.reverse()
    for si, j in anchors:
        rough[si] = word_time[j]
        roughj[si] = j
        direct[si] = True

    # ── PASS 2 — local fill between anchors ────────────────────────────────
    def walk(s_lo: int, s_hi: int, j_lo: int, j_hi: int, wi: int) -> None:
        for si in range(s_lo, s_hi):
            tk = tk_all[si]
            if len(tk) < 2:
                wi += 1
                continue
            need = min(len(tk), 5)
            best = None
            lo = max(j_lo, wi - BACK)
            hi = min(j_hi, wi + FWD)
            for j in range(lo, hi):
                if word_norm[j] != tk[0]:
                    continue
                if hits(j, tk, need) >= max(3, need - 1):
                    best = j
                    break
            if best is not None:
                rough[si] = word_time[best]
                roughj[si] = best
                direct[si] = True
                wi = best + len(tk)
            else:
                # Keep tracking the rate through misses.
                wi += len(tk)

    if anchors:
        for (sa, ja), (sb, jb) in zip(anchors, anchors[1:]):
            if sb > sa + 1:
                walk(sa + 1, sb, ja, jb, ja + len(tk_all[sa]))
        s0, j0 = anchors[0]
        walk(0, s0, 0, j0, max(0, j0 - sum(len(t) for t in tk_all[:s0])))
        sl, jl = anchors[-1]
        walk(sl + 1, n, jl, total_words, jl + len(tk_all[sl]))
    else:
        # No anchors (tiny or odd input): the old full-range behaviour.
        walk(0, n, 0, total_words, 0)

    matched = [i for i in range(n) if rough[i] is not None]
    if not matched:
        return CoarseResult(rough, 0, n, 0, DEFAULT_RATE, direct)
    first_index, last_index = matched[0], matched[-1] + 1

    measured = _rate(matched, rough, tk_all)

    dropped = _fill_interior_runs(
        matched, rough, roughj, tk_all, tri, word_time, word_norm,
        hits, measured, failed_ranges, direct,
    )

    # Monotonic: a later sentence never starts before an earlier one.
    prev: float | None = None
    for idx in range(n):
        value = rough[idx]
        if value is None:
            continue
        if prev is not None and value < prev:
            rough[idx] = prev
        prev = rough[idx]

    return CoarseResult(rough, first_index, last_index, dropped, measured, direct)


def _rate(
    matched: list[int], rough: list[float | None], tk_all: list[list[str]]
) -> float:
    """Narration rate (tokens/sec) from CLOSELY-SPACED matched pairs.

    Only pairs ADJACENT in sentence space (`b - a <= 3`) may contribute, and
    that restriction is the whole point. In a recap or montage the matched
    quotes are thousands of epub tokens apart but seconds apart in audio, and
    one such pair poisons the estimate: a GraphicAudio "story so far" measured
    212 tok/s — 85x reality — which then let a never-narrated 39-sentence run
    pass the fit test below and smear itself ten seconds over a music bridge.
    """
    tok_sum = 0
    t_sum = 0.0
    for a_i, b_i in zip(matched, matched[1:]):
        dt = rough[b_i] - rough[a_i]  # type: ignore[operator]
        if 0 < dt <= 30 and b_i - a_i <= 3:
            tok_sum += sum(len(tk_all[k]) for k in range(a_i, b_i))
            t_sum += dt
    rate = (tok_sum / t_sum) if t_sum > 0 and tok_sum > 0 else DEFAULT_RATE
    if not (RATE_MIN <= rate <= RATE_MAX):
        # An implausible measurement is clamped rather than trusted.
        rate = min(RATE_MAX, max(RATE_MIN, rate))
    return rate


def _fill_interior_runs(
    matched, rough, roughj, tk_all, tri, word_time, word_norm,
    hits, rate, failed_ranges, direct,
) -> int:
    """Interior unmatched runs: interpolate the narrated, drop the unnarrated.

    A SHORT gap is a transcription miss of narrated text and gets token-weighted
    interpolation between its matched neighbours. A run whose spoken duration
    could never fit the audio gap is text the narrator SKIPPED — copyright page,
    TOC, acknowledgments, footnote bodies — and is kept `None` so it is excluded
    from chunking and from the VTT, instead of smeared over real audio. That is
    the Well of Ascension failure: ~90 unspoken front-matter sentences dragged
    chapter 1's cues ~85 s late for the first ~5 minutes.

    TWO INDEPENDENT FIT TESTS, and the run is judged non-narrated when EITHER
    says the text cannot be in the gap:
      * time test — spoken duration at the measured rate vs the audio gap;
      * word test — text tokens vs words the transcriber actually HEARD there.
        Immune to rate poisoning and to dead air: a music bridge makes the time
        gap look roomy while the word count says nobody spoke.

    The word test is only trusted where the transcriber actually RAN. A failed
    transcribe slice leaves a wordless stretch of real narration, so any gap
    touching a failed slice's time range falls back to the time test alone.
    """
    total_words = len(word_norm)
    dropped = 0
    rescued = 0
    for a_i, b_i in zip(matched, matched[1:]):
        if b_i == a_i + 1:
            continue
        gap = rough[b_i] - rough[a_i]
        gap_words = max(0, roughj[b_i] - (roughj[a_i] + len(tk_all[a_i])))
        run_tok = sum(len(tk_all[k]) for k in range(a_i + 1, b_i))
        words_trusted = not any(
            lo < rough[b_i] and rough[a_i] < hi for lo, hi in failed_ranges
        )
        if run_tok >= 12 and (
            (run_tok / rate > 2.0 * gap + 10.0)
            or (words_trusted and run_tok > 2.0 * gap_words + 25)
        ):
            # Non-narrated run — but rescue any sentence inside it that still
            # confirms on an INTERIOR trigram within the gap's transcript
            # window. Narrated sentences land in dropped runs when the
            # transcriber misheard their opening (PASS 1 anchors on openings
            # only): "King Elend" -> "King Ellen", or a heading glued onto real
            # prose. The match time is back-extrapolated to the sentence start
            # by o/rate.
            last_t = rough[a_i]
            for k in range(a_i + 1, b_i):
                tk = tk_all[k]
                for o in range(0, len(tk) - 2):
                    need = min(len(tk) - o, 6)
                    hit = None
                    for j in tri.get((tk[o], tk[o + 1], tk[o + 2])) or []:
                        if not (last_t < word_time[j] < rough[b_i]):
                            continue
                        if hits(j, tk[o:], need) >= max(3, need - 1):
                            hit = j
                            break
                    if hit is not None:
                        rough[k] = max(last_t, word_time[hit] - o / rate)
                        last_t = word_time[hit]
                        rescued += 1
                        direct[k] = True
                        break
                if rough[k] is None:
                    dropped += 1
            continue
        # Narrated run: distribute its sentences over the WORDS the transcriber
        # heard in the gap, not linearly over wall-clock time — a music bridge
        # or SFX pause contributes zero words, so interpolated sentences snap to
        # actual speech instead of being smeared into the silence (the
        # uniform-rate assumption put cues ~10 s late across one 16 s bridge).
        # Falls back to time-linear when the gap has too few words to carry the
        # distribution (failed transcribe slice, ASR that heard almost nothing).
        j_a = roughj[a_i] + len(tk_all[a_i])
        j_b = roughj[b_i]
        n_words = j_b - j_a
        use_words = (
            words_trusted and run_tok > 0 and n_words >= max(10, 0.2 * run_tok)
        )
        total = (run_tok + len(tk_all[a_i])) or 1
        cum = len(tk_all[a_i])
        cum_run = 0
        for k in range(a_i + 1, b_i):
            if use_words:
                jk = j_a + int(n_words * (cum_run / run_tok))
                rough[k] = word_time[min(max(jk, 0), total_words - 1)]
            else:
                rough[k] = rough[a_i] + gap * (cum / total)
            cum += len(tk_all[k])
            cum_run += len(tk_all[k])
    return dropped
