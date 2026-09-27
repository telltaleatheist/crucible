from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass

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
    total_words = len(word_norm)
    n = len(sents)
    tk_all = [toks(s) for s in sents]

    rough: list[float | None] = [None] * n
    roughj: list[int | None] = [None] * n
    direct = [False] * n

    def hits(j: int, tk: list[str], need: int) -> int:
        k = j
        m = 0
        while k < min(total_words, j + SPAN) and m < need:
            if word_norm[k] == tk[m]:
                m += 1
            k += 1
        return m

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
