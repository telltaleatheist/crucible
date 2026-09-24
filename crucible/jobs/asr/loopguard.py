"""The Qwen3-ASR repetition-loop guard: what counts as a loop, and the budget.

docs/PHASE25-QWEN-ASR.md section 5. Pure functions over what a decode and an
alignment returned, so every threshold here is tested without a model.

THE FAILURE THIS EXISTS FOR (ContentStudio, Mac Studio, 2026-09-24). Once in 78
pieces, a 180-second piece came back as ONE LINE REPEATED ABOUT 60 TIMES — 2,388
words — until the decoder ran out of `max_new_tokens`. The aligner then did what
a forced aligner does with text that is not in the audio: it stamped every one of
those words at one instant, zero-length spans, and the real three minutes of
speech were simply gone. Nothing in the output said so. A transcript with a hole
in it looks exactly like a transcript without one, which is the whole argument
`crucible/jobs/asr/__init__.py` makes about a failed window, and it gets the same
answer here: never accept a collapsed span silently.

A LOOP IS WEATHER, SO IT GETS A BUDGET AND THEN A NAME. Greedy decoding is
deterministic, so re-decoding the same audio the same way reproduces the loop;
what changes the outcome is different INPUT. So the budget is a ladder of
smaller windows over the same audio (`WINDOW_LADDER_SECONDS`): a piece that
loops at 180 s is re-cut at quiet points into pieces of at most 60 s and
decoded again, and a 60 s piece that still loops is re-cut to 20 s. A piece
that loops at 20 s fails the job by name — `asr_decode_loop`, with its time
range — because at that length there is nothing smaller to try that is still a
transcript.

WHY NOT A REPETITION PENALTY. It was considered and refused. vLLM's
`repetition_penalty` and mlx-audio's both scale down EVERY token already in the
prompt or the output, and the prompt is the context, which is exactly where
ContentStudio's "um, uh, ah, er, hmm" live. A penalty is a thumb on the scale
against the very fillers this model was chosen to keep, and against the verbatim
repeats Owen wants kept ("repeats kept verbatim"). A smaller window changes
nothing about what the model is asked to write down. `no_repeat_ngram_size` is
not offered by vLLM's `SamplingParams` at 0.29.0 at all, and would forbid a
real repeated phrase outright.

The four signals, and where each threshold comes from
-----------------------------------------------------
Every threshold is a DECISION, set from the one loop anyone has seen and from
ordinary speech rates, not measured over a corpus. The first live run should
count each signal over a long stream; docs/PHASE25-QWEN-ASR.md section 8 says
how.

1. **The decode hit its token budget** (`hit_token_limit`). A piece of speech
   ends on its own; one that is still writing when its budget runs out is not
   transcribing. The budget per piece is `token_budget`: 4096 tokens per 180 s,
   ContentStudio's own operating point, scaled to the piece and floored.
2. **Words per second of audio above `WORDS_PER_SECOND_CEILING` = 8.0.** Brisk
   conversational English is about 3 words a second (180 wpm); 8 a second is
   480 wpm, past any sustained human speech a livestream carries. The loop
   ContentStudio saw was 2,388 words in 180 s = 13.3 a second. Only applied
   past `RATE_MIN_WORDS`, so a short padded piece with four quick words in it is
   not a loop.
3. **A phrase of `REPEAT_MIN_WORDS`..`REPEAT_MAX_WORDS` words repeated
   back-to-back at least `REPEAT_MIN_COUNT` = 8 times.** A verbatim repeat
   ("I, I, I think") is one or two words a few times; eight consecutive copies
   of a four-word phrase is not something people say. ContentStudio's loop was
   one line about 60 times.
4. **The aligner collapsed: `COLLAPSE_RUN_ITEMS` = 12 consecutive items with a
   zero-length span.** The aligner's resolution is 80 ms (`timestamp_segment_time`
   in Qwen3-ForcedAligner's config); a real word occupies at least one step. One
   or two zero-length items happen at a piece's edges; twelve in a row is text
   with no audio under it — exactly the signature of the loop above.

The text signals (1-3) are counted on whitespace-separated words, so they are
calibrated for languages written with spaces. On Chinese, Japanese and
Cantonese the word count undercounts and rules 2-3 under-fire; rules 1 and 4
still hold there, and a missed loop in those languages ends in rule 4's named
failure rather than in a silent hole.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

#: The window each rung of the budget decodes at, largest first. The first rung
#: is `QWEN_PIECE_MAX_SECONDS` (the aligner's own 180 s limit, asrmodels.py);
#: each later one is a third of the one before, which is small enough to change
#: what the decoder hears and large enough to still hold whole sentences.
WINDOW_LADDER_SECONDS: tuple[int, ...] = (180, 60, 20)

WORDS_PER_SECOND_CEILING = 8.0
RATE_MIN_WORDS = 24

REPEAT_MIN_WORDS = 4
REPEAT_MAX_WORDS = 40
REPEAT_MIN_COUNT = 8

COLLAPSE_RUN_ITEMS = 12
#: A span at most this long is zero-length. The aligner reports seconds to
#: three decimals (`round(..., 3)` in qwen_asr's `_offset_align_result`), so
#: anything under a millisecond is the same instant.
ZERO_SPAN_SECONDS = 0.0005

#: The per-piece token budget, as a rate: 4096 new tokens for 180 s of audio,
#: ContentStudio's own settings, i.e. about 22.8 tokens a second — roughly
#: seven times what 3 words a second of English costs. Floored so a short piece
#: still has room for a fast speaker.
BUDGET_TOKENS_PER_SECOND = 4096 / 180
BUDGET_FLOOR_TOKENS = 256

_WORD_EDGES = re.compile(r"^\W+|\W+$", re.UNICODE)


@dataclass(frozen=True)
class LoopSignal:
    """Why a piece is taken to be a loop. `kind` is stable; `detail` is prose."""

    kind: str
    detail: str


def token_budget(duration_s: float, max_new_tokens: int) -> int:
    """How many new tokens one piece of `duration_s` may generate."""
    scaled = math.ceil(duration_s * BUDGET_TOKENS_PER_SECOND)
    return min(max_new_tokens, max(BUDGET_FLOOR_TOKENS, scaled))


def words_of(text: str) -> list[str]:
    """The words the text rules count and compare: lower-cased, edges stripped.

    Punctuation at a word's edges is dropped so that "um," and "um" compare
    equal; a word that was only punctuation is not a word.
    """
    words = []
    for raw in text.split():
        word = _WORD_EDGES.sub("", raw).lower()
        if word:
            words.append(word)
    return words


def repeated_phrase(words: Sequence[str]) -> tuple[int, int, int] | None:
    """`(phrase_words, copies, first_index)` of a back-to-back repeat, or None.

    The shortest qualifying phrase is reported, at its first position.
    """
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
    """Rules 1-3 over one decoded piece. None means it reads like speech."""
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
    """Rule 4 over one piece's aligned items. None means the text found audio."""
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


def next_window(level: int) -> int | None:
    """The window the next rung decodes at, or None when the budget is spent."""
    following = level + 1
    if following < len(WINDOW_LADDER_SECONDS):
        return WINDOW_LADDER_SECONDS[following]
    return None


def clock(seconds: float) -> str:
    """`h:mm:ss.s`, for naming a piece's place in a stream a person can find."""
    whole = int(seconds)
    tenths = int(round((seconds - whole) * 10))
    if tenths == 10:
        whole, tenths = whole + 1, 0
    return f"{whole // 3600}:{whole % 3600 // 60:02d}:{whole % 60:02d}.{tenths}"
