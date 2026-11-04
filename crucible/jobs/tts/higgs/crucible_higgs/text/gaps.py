"""THE GAP CLASSIFIER: how much silence sits either side of one chunk.

In Crucible this is the Listen door's `gapSec` (`serve/worker.py:row_gap_sec`).
Moved from narrator@dc05e7e7 `text/gaps.py` VERBATIM, Orpheus history and the
ORPHEUS_SENTENCE_GAP fallback name included: that variable is read only when
NARRATOR_SENTENCE_GAP is unset, Crucible sets neither, and the function is kept
byte for byte so a gap here is the gap narrator stated. The history below is
narrator's.

Moved here from `engine/orpheus/prompt.py` on 2026-09-04, BODY UNCHANGED, because
two callers now need it and they may not import each other:

  `engine/orpheus/prompt.py`  bakes the answer into every Orpheus FLAC
                              (`_classify_gap` -> `_save_audio`), which is why an
                              Orpheus session needs no gap file at all.
  `text/prep.py`              WRITES the answer to
                              `chapters/sentences/gaps.json` for a pads=False
                              engine, whose audio carries no silence of its own.

WHY IT MOVED INTO `text/` RATHER THAN BEING IMPORTED OUT OF `engine/`. The
dependency has to point one way, and `text/` is the lower layer: it is pure
stdlib, it must import on a machine with no torch, and `narrator.engine` pulls
in `narrator.engine.orpheus` at package import. `assemble/` already refuses to
import an engine for exactly this reason (`assemble/engine_profiles.py`: "assembly
runs without importing an engine"). So the pure function lives at the bottom and
the Orpheus mixin calls down to it; `PromptMixin._classify_gap` is now a
one-line delegate and every `self._classify_gap(...)` call site is untouched.

THE FUNCTION BELOW IS VERBATIM. It was lifted mechanically (dedent one level,
drop `self`) rather than retyped, so its comments, its numbers and its 2026-07-17
ruling are exactly what the engine has been running.

WHAT THIS MEANS FOR gaps.json, and it is worth saying plainly because it is not
what a reader expects: at 9daab0ba there are NO per-kind tiers left. A heading,
a paragraph `[break]`, an `[item]`, a `[silence]` and a bare sentence end ALL
return `(0.0, sentence_gap)`. The paragraph and section tiers were removed on
2026-07-17 - they were measured as purely additive dead air on top of a model
that already reproduces the narrator's own pausing - and the only thing that
still moves the answer is an EXPLICIT `[pause:X]`. So a gaps.json is mostly one
repeated value, and that is correct, not a bug in the writer.
"""
import os
import re


def classify_gap(sentence: str):
    """Inter-clip silence for a chunk. Returns (lead_gap_sec, trail_gap_sec).

    2026-07-17 - AUTO PARAGRAPH/SECTION GAPS REMOVED FOR ORPHEUS (deliberately).
    Root cause of the long-standing "dialogue has huge pauses" complaint: e2a's
    SHARED text prep rewrites EVERY blank line to a valueless [pause] token, and
    prose puts EVERY dialogue turn in its own blank-line-separated paragraph. The
    old three-tier logic here then stamped a deterministic SECTION gap (1.0-1.6s
    lead) on top of each turn's sentence-gap tail. Measured with auto-editor on
    the mistborn 0.6s-cap model: dialogue-turn gaps were 2.0-2.5s (13 of them, one
    per paragraph break) vs the narrator's own ~0.6-1.0s; a STRAIGHT-NARRATION
    render from the same model measured median 0.57s / max 1.17s with NO outliers
    - i.e. the Orpheus model, trained on clips that keep their natural
    inter-sentence pauses (only the clip EDGES trimmed), already reproduces the
    narrator's pausing itself. The paragraph/section insertion was therefore
    PURELY additive dead air.

    So for Orpheus we now keep ONLY:
      - the sentence-gap FLOOR on every chunk's tail (each chunk is a separate
        generation whose trailing silence is trimmed, so without a small floor the
        chunks butt together), and
      - an EXPLICIT [pause:X] (intentional, markup-specified beat) - still honored,
        because that's a deliberate pause, not the auto blank-line noise.
    The auto valueless [pause] (blank line) and [break]/[silence] (<p>/<br>) tiers
    are GONE. The tokens themselves still drive chunk boundaries in the packer and
    are stripped by _clean_sentence_for_tts before TTS - only their deterministic
    GAP is removed here.

    If a real scene/section break ever under-pauses as a result, re-introduce a
    section tier HERE (or emit an explicit [pause:X] at that break) - do NOT
    restore the blanket blank-line gap that caused the dialogue problem.

    Env override: NARRATOR_SENTENCE_GAP, else ORPHEUS_SENTENCE_GAP (the floor; 0 disables).
    """
    raw = (sentence or '').strip()
    lowered = raw.lower()

    def _env(name):
        v = os.environ.get(name)
        return float(v) if v is not None else None

    # Sentence-gap floor: the minimum tail every chunk gets so a chunk-to-chunk
    # Sentence-gap floor: the minimum tail every chunk gets so a chunk-to-chunk
    # join is never bare.
    #
    # TWO OVERRIDES, AND THE NEUTRAL ONE WINS. NARRATOR_SENTENCE_GAP is engine-agnostic and is
    # what a pads=False engine sets from its own per-voice measurement; ORPHEUS_SENTENCE_GAP is
    # the original Orpheus-only name, kept so every existing caller and doc stays correct.
    #
    # WHY THIS MATTERED (2026-09-11): for a pads=False engine text/prep.py stamps this value
    # into gaps.json for EVERY chunk and the assembler realizes it as real silence, so until
    # now every Higgs voice shipped the same 0.6 s join whatever that voice actually does.
    # A join is (the model's own trailing silence + this), and the tail is voice-specific -
    # measured 0.22 s on sigma. So the number a voice declares here is the INJECT, already net
    # of its own tail; putting a TARGET here instead lands every join long by the tail. That
    # exact confusion is recorded in orpheus-models.json as how thirdreich shipped 0.24 s long
    # on every join.
    #
    # Explicit None checks rather than `or`: 0 is MEANINGFUL here - it disables the floor so
    # chunks butt together on the model's own pauses - and falsiness would swallow it.
    sentence_gap = _env('NARRATOR_SENTENCE_GAP')
    if sentence_gap is None:
        sentence_gap = _env('ORPHEUS_SENTENCE_GAP')
    if sentence_gap is None:
        sentence_gap = 0.6   # ear-approved on rohan (2026-07-12)

    # Honor ONLY an EXPLICIT [pause:X] - a deliberate, markup-specified beat.
    # (Auto valueless [pause] from a blank line is intentionally NOT matched: that
    # is the dialogue-pause noise removed 2026-07-17; see the docstring.)
    m = re.search(r'\[pause:([0-9.]+)\]', raw, flags=re.IGNORECASE)
    if m:
        token_gap = float(m.group(1))
        if lowered.startswith('[pause:'):
            return token_gap, sentence_gap               # explicit beat as lead + normal tail
        return 0.0, max(token_gap, sentence_gap)

    # Everything else - plain sentence end, auto [break]/[silence], auto valueless
    # [pause] - collapses to the sentence-gap floor. Orpheus supplies the real
    # inter-sentence pausing itself (learned from natural-pause training clips).
    return 0.0, sentence_gap


def classify_gap_seconds(sentence: str) -> tuple:
    """`classify_gap` as a plain `(before, after)` tuple of floats.

    The only difference is the return TYPE being pinned: `gaps.json` is JSON and
    a numpy float or an int would round-trip differently, so the one caller that
    serializes the answer goes through here.
    """
    before, after = classify_gap(sentence)
    return float(before), float(after)
