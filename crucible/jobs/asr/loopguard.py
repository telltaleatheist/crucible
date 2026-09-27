from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

WINDOW_LADDER_RATIO = 0.5
WINDOW_LADDER_RUNGS = 3


def window_ladder(piece_s: float) -> tuple[float, ...]:
    return tuple(piece_s * WINDOW_LADDER_RATIO**rung for rung in range(WINDOW_LADDER_RUNGS))

WORDS_PER_SECOND_CEILING = 8.0
RATE_MIN_WORDS = 24

REPEAT_MIN_WORDS = 4
REPEAT_MAX_WORDS = 40
REPEAT_MIN_COUNT = 8

COLLAPSE_RUN_ITEMS = 12
ZERO_SPAN_SECONDS = 0.0005

BUDGET_TOKENS_PER_SECOND = 4096 / 180
BUDGET_FLOOR_TOKENS = 256

_WORD_EDGES = re.compile(r"^\W+|\W+$", re.UNICODE)


@dataclass(frozen=True)
class LoopSignal:

    kind: str
    detail: str


def token_budget(duration_s: float, max_new_tokens: int) -> int:
    scaled = math.ceil(duration_s * BUDGET_TOKENS_PER_SECOND)
    return min(max_new_tokens, max(BUDGET_FLOOR_TOKENS, scaled))


def words_of(text: str) -> list[str]:
    words = []
    for raw in text.split():
        word = _WORD_EDGES.sub("", raw).lower()
        if word:
            words.append(word)
    return words


def repeated_phrase(words: Sequence[str]) -> tuple[int, int, int] | None:
    total = len(words)
    for size in range(REPEAT_MIN_WORDS, REPEAT_MAX_WORDS + 1):
        if size * REPEAT_MIN_COUNT > total:
            break
        start = 0
        while start + size * REPEAT_MIN_COUNT <= total:
            phrase = words[start : start + size]
            copies = 1
            while words[start + copies * size : start + (copies + 1) * size] == phrase:
                copies += 1
            if copies >= REPEAT_MIN_COUNT:
                return size, copies, start
            start += 1
    return None


def text_signal(
    text: str, duration_s: float, hit_token_limit: bool, budget: int
) -> LoopSignal | None:
    if hit_token_limit:
        return LoopSignal(
            "token_limit",
            f"the decode was still writing when it reached its {budget}-token "
            f"budget for {duration_s:.1f}s of audio; speech ends on its own",
        )
    words = words_of(text)
    if len(words) >= RATE_MIN_WORDS and duration_s > 0:
        rate = len(words) / duration_s
        if rate > WORDS_PER_SECOND_CEILING:
            return LoopSignal(
                "words_per_second",
                f"{len(words)} words in {duration_s:.1f}s of audio is "
                f"{rate:.1f} a second, past the {WORDS_PER_SECOND_CEILING:.0f} "
                "no sustained speech reaches",
            )
    repeat = repeated_phrase(words)
    if repeat is not None:
        size, copies, first = repeat
        phrase = " ".join(words[first : first + size])
        return LoopSignal(
            "repeated_phrase",
            f"the {size}-word phrase {phrase!r} repeats {copies} times back to "
            "back",
        )
    return None


def alignment_signal(items: Sequence[dict[str, Any]]) -> LoopSignal | None:
    run = longest = 0
    for item in items:
        if float(item["end"]) - float(item["start"]) <= ZERO_SPAN_SECONDS:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    if longest >= COLLAPSE_RUN_ITEMS:
        return LoopSignal(
            "aligner_collapse",
            f"the aligner stamped {longest} consecutive words at a single "
            "instant (zero-length spans): text with no audio under it",
        )
    return None


def next_window(ladder: Sequence[float], level: int) -> float | None:
    following = level + 1
    if following < len(ladder):
        return ladder[following]
    return None


def clock(seconds: float) -> str:
    whole = int(seconds)
    tenths = int(round((seconds - whole) * 10))
    if tenths == 10:
        whole, tenths = whole + 1, 0
    return f"{whole // 3600}:{whole % 3600 // 60:02d}:{whole % 60:02d}.{tenths}"
