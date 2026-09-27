from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass
from functools import partial
from typing import Callable

BACK, FWD, SPAN = 8, 60, 14

MAX_ANCHOR_OCCURRENCES = 50

DEFAULT_RATE = 2.5
RATE_MIN, RATE_MAX = 0.8, 8.0


def _norm(word: str) -> str:
    decomposed = unicodedata.normalize("NFKD", word)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^0-9a-z]+", "", stripped.lower())


def toks(text: str) -> list[str]:
    return [t for t in (_norm(w) for w in text.split()) if t]


@dataclass(frozen=True)
class CoarseResult:

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
    word_time = [t for _, t in words]
    word_norm = [w for w, _ in words]
    tk_all = [toks(s) for s in sents]
    n = len(sents)
    state = _Placement(
        tk_all, word_norm, word_time, partial(_count_hits, word_norm),
        [None] * n, [None] * n, [False] * n,
    )
    tri = _trigram_index(word_norm)
    anchors = _anchor_chain(_anchor_candidates(tk_all, tri, state.hits))
    for si, j in anchors:
        state.place(si, j)
    _walk_around_anchors(state, anchors)

    rough = state.rough
    matched = [i for i in range(n) if rough[i] is not None]
    if not matched:
        return CoarseResult(rough, 0, n, 0, DEFAULT_RATE, state.direct)
    first_index, last_index = matched[0], matched[-1] + 1
    measured = _rate(matched, rough, tk_all)
    dropped = _fill_interior_runs(
        matched, rough, state.roughj, tk_all, tri, word_time, word_norm,
        state.hits, measured, failed_ranges, state.direct,
    )
    _never_backwards(rough)
    return CoarseResult(rough, first_index, last_index, dropped, measured, state.direct)


@dataclass
class _Placement:

    tk_all: list[list[str]]
    word_norm: list[str]
    word_time: list[float]
    hits: Callable[[int, list[str], int], int]
    rough: list[float | None]
    roughj: list[int | None]
    direct: list[bool]

    def place(self, si: int, j: int) -> None:
        self.rough[si] = self.word_time[j]
        self.roughj[si] = j
        self.direct[si] = True


def _count_hits(word_norm: list[str], j: int, tk: list[str], need: int) -> int:
    end = min(len(word_norm), j + SPAN)
    k = j
    m = 0
    while k < end and m < need:
        if word_norm[k] == tk[m]:
            m += 1
        k += 1
    return m


def _enough(got: int, need: int) -> bool:
    return got >= max(3, need - 1)


def _trigram_index(word_norm: list[str]) -> dict[tuple[str, str, str], list[int]]:
    tri: dict[tuple[str, str, str], list[int]] = {}
    for j in range(len(word_norm) - 2):
        tri.setdefault(
            (word_norm[j], word_norm[j + 1], word_norm[j + 2]), []
        ).append(j)
    return tri


def _anchor_candidates(tk_all, tri, hits) -> list[tuple[int, int]]:
    cands: list[tuple[int, int]] = []
    for si, tk in enumerate(tk_all):
        if len(tk) < 4:
            continue
        pos = tri.get((tk[0], tk[1], tk[2]))
        if not pos or len(pos) > MAX_ANCHOR_OCCURRENCES:
            continue
        need = min(len(tk), 6)
        cands.extend((si, j) for j in pos if _enough(hits(j, tk, need), need))
    return cands


def _anchor_chain(cands: list[tuple[int, int]]) -> list[tuple[int, int]]:
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
    return anchors


def _walk_around_anchors(state: _Placement, anchors: list[tuple[int, int]]) -> None:
    tk_all = state.tk_all
    n = len(tk_all)
    total_words = len(state.word_norm)
    if not anchors:
        _walk(state, 0, n, 0, total_words, 0)
        return
    for (sa, ja), (sb, jb) in zip(anchors, anchors[1:]):
        if sb > sa + 1:
            _walk(state, sa + 1, sb, ja, jb, ja + len(tk_all[sa]))
    s0, j0 = anchors[0]
    _walk(state, 0, s0, 0, j0, max(0, j0 - sum(len(t) for t in tk_all[:s0])))
    sl, jl = anchors[-1]
    _walk(state, sl + 1, n, jl, total_words, jl + len(tk_all[sl]))


def _walk(state: _Placement, s_lo: int, s_hi: int, j_lo: int, j_hi: int, wi: int) -> None:
    for si in range(s_lo, s_hi):
        tk = state.tk_all[si]
        if len(tk) < 2:
            wi += 1
            continue
        best = _first_match(state, tk, max(j_lo, wi - BACK), min(j_hi, wi + FWD))
        if best is not None:
            state.place(si, best)
            wi = best + len(tk)
        else:
            wi += len(tk)


def _first_match(state: _Placement, tk: list[str], lo: int, hi: int) -> int | None:
    need = min(len(tk), 5)
    for j in range(lo, hi):
        if state.word_norm[j] == tk[0] and _enough(state.hits(j, tk, need), need):
            return j
    return None


def _never_backwards(rough: list[float | None]) -> None:
    prev: float | None = None
    for idx, value in enumerate(rough):
        if value is None:
            continue
        if prev is not None and value < prev:
            rough[idx] = prev
        prev = rough[idx]


def _rate(
    matched: list[int], rough: list[float | None], tk_all: list[list[str]]
) -> float:
    tok_sum = 0
    t_sum = 0.0
    for a_i, b_i in zip(matched, matched[1:]):
        dt = rough[b_i] - rough[a_i]
        if 0 < dt <= 30 and b_i - a_i <= 3:
            tok_sum += sum(len(tk_all[k]) for k in range(a_i, b_i))
            t_sum += dt
    rate = (tok_sum / t_sum) if t_sum > 0 and tok_sum > 0 else DEFAULT_RATE
    if not (RATE_MIN <= rate <= RATE_MAX):
        rate = min(RATE_MAX, max(RATE_MIN, rate))
    return rate


def _fill_interior_runs(
    matched, rough, roughj, tk_all, tri, word_time, word_norm,
    hits, rate, failed_ranges, direct,
) -> int:
    dropped = 0
    for a_i, b_i in zip(matched, matched[1:]):
        if b_i == a_i + 1:
            continue
        run = _Run(a_i, b_i, rough, roughj, tk_all, failed_ranges)
        if run.overfull(rate):
            dropped += _rescue_run(run, rough, tk_all, tri, word_time, hits, rate, direct)
        else:
            _spread_run(run, rough, word_time, len(word_norm))
    return dropped


class _Run:

    def __init__(self, a_i, b_i, rough, roughj, tk_all, failed_ranges) -> None:
        self.a_i = a_i
        self.b_i = b_i
        self.gap = rough[b_i] - rough[a_i]
        self.lead = len(tk_all[a_i])
        self.j_a = roughj[a_i] + self.lead
        self.j_b = roughj[b_i]
        self.lengths = [(k, len(tk_all[k])) for k in range(a_i + 1, b_i)]
        self.run_tok = sum(length for _, length in self.lengths)
        self.words_trusted = not any(
            lo < rough[b_i] and rough[a_i] < hi for lo, hi in failed_ranges
        )

    def overfull(self, rate: float) -> bool:
        gap_words = max(0, self.j_b - self.j_a)
        return self.run_tok >= 12 and (
            (self.run_tok / rate > 2.0 * self.gap + 10.0)
            or (self.words_trusted and self.run_tok > 2.0 * gap_words + 25)
        )


def _rescue_run(run: _Run, rough, tk_all, tri, word_time, hits, rate, direct) -> int:
    dropped = 0
    last_t = rough[run.a_i]
    upper = rough[run.b_i]
    for k in range(run.a_i + 1, run.b_i):
        tk = tk_all[k]
        for o in range(0, len(tk) - 2):
            hit = _trigram_hit(tk[o:], tri, word_time, hits, last_t, upper)
            if hit is not None:
                rough[k] = max(last_t, word_time[hit] - o / rate)
                last_t = word_time[hit]
                direct[k] = True
                break
        if rough[k] is None:
            dropped += 1
    return dropped


def _trigram_hit(tail, tri, word_time, hits, after: float, before: float) -> int | None:
    need = min(len(tail), 6)
    for j in tri.get((tail[0], tail[1], tail[2])) or []:
        if after < word_time[j] < before and _enough(hits(j, tail, need), need):
            return j
    return None


def _spread_run(run: _Run, rough, word_time, total_words: int) -> None:
    n_words = run.j_b - run.j_a
    run_tok = run.run_tok
    use_words = (
        run.words_trusted and run_tok > 0 and n_words >= max(10, 0.2 * run_tok)
    )
    total = (run_tok + run.lead) or 1
    cum = run.lead
    cum_run = 0
    for k, length in run.lengths:
        if use_words:
            jk = run.j_a + int(n_words * (cum_run / run_tok))
            rough[k] = word_time[min(max(jk, 0), total_words - 1)]
        else:
            rough[k] = rough[run.a_i] + run.gap * (cum / total)
        cum += length
        cum_run += length
