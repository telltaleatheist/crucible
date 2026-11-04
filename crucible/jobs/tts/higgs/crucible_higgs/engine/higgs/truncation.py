"""The Higgs LENGTH guard: watch every chunk's audio against its text, and
re-render the ones that stopped early OR ran on - a seed change first, then the
chunk split at sentence boundaries. Shared by BOTH arms (served and MLX).

Owen, 2026-09-06: "im seeing the occasional truncation. we should watch for
truncations from higgs the same way we do in orpheus and re-render the split
sentences if one appears." And, the same evening, after the whole-book coverage
report: "deterministic solutions are not the right shape for this. i think we
need to detect the expected length and if it steps outside those bounds then
we split at sentence boundaries and re-render. same as orpheus."

MEASURED, the cases this exists for (Working Towards The Fuhrer, PC, SGLang,
deathstalker ckpt-1080, 2026-09-06 15:26, 49 chunks at 1,200; session pace
5.83 s per 100 characters = 17.2 chars/s; every clean chunk sat between 0.94x
and 1.10x of that expectation):

  TOO SHORT  chunk 19: 1,127 characters, 3.0 s of audio (375 chars/s) - it
             stopped at the end of its FIRST sentence and the render reported
             success. Training's live validation of the same book had flagged
             the same chunk; narrator seeds chunk i at `seed + i`, so an early
             stop REPRODUCES at its seed, and a retake at the same seed is not
             a retake. Chunk 32: the last 25 words never spoken, 0.84x.
  TOO LONG   chunks 48, 35, 26, 41: the text was read (aligned ratios 0.878,
             0.954, 1.000, 0.994) and then the model KEPT TALKING - 27.3 s,
             26.0 s, 8.8 s and 6.8 s of speech past the end of the text, at
             1.54x, 1.42x, 1.24x and 1.11x of expectation - and that audio
             shipped in the m4b. Training (field notes 4n.28): late-chunk EOS
             failures at temperature 1.0 - once the EOS draw is lost with no
             text left, the model is off-distribution and continues.

So on that render the run-on class (4) outnumbered the early-stop class (3),
and both are LENGTH defects: audio that is not the length its text should be.
`StopPolicy.max_chars_per_sec` (20.0, the short side) had named a threshold
since the engine landed and nothing enforced it; `min_chars_per_sec` (the long
side) is new with this module.

THE LADDER, per chunk, in order, for either side:

  take 0   the engine's own render (its seed rule: `seed + index`).
           Inside the band -> shipped, no event.
  take 1   RE-ROLL at a different seed (`reroll_seed`). Orpheus's finding
           (memory: orpheus-short-chunk-repeat) was that the re-roll backstop,
           not the EOS boost, is what fixes an early stop; the same holds for
           a run-on, because both are one lost sampling draw.
  split    the chunk cut into two halves at the sentence boundary nearest its
           middle, each half rendered through this same ladder (a half that
           misbehaves gets its own re-roll and its own split), joined with
           `RESPLIT_JOIN_SECONDS` of silence - the pause a sentence boundary
           inside one paragraph gets. This is Orpheus's `_generate_audio_vllm_safe`
           shape, and Owen's stated practice.
  accept   at `MAX_DEPTH`, or when the text cannot be split, the take whose
           length is CLOSEST to expectation is shipped and the event says
           `accepted-off-length` - the render never refuses, because assembly
           is built on the expected text and the real audio length (Owen's
           2026-09-05 ruling), and the coverage audit names the chunk.

THE RUNGS ABOVE ARE THE POLICY; WHEN their renders happen is the driver's, and
there are THREE, all over the same `GuardPlan`:

  serial   `render_guarded` - one chunk, depth-first, taking the plan's next
           request every time. The single-chunk paths (Listen, `convert`).
  rounds   every waiting chunk's next take gathered and rendered as ONE batch
           (`round()`): the MLX arm, where a batch of 32 costs about the wall
           time of one row and a solo re-roll therefore costs a whole batch.
  pool     one request per free slot (`next_request()`), a slot freed the moment
           a take lands: the served arm's `convert_many`, where the batch is N
           concurrent requests against a continuously-batching server.

Owen, 2026-09-08: "it should always be batched. even if it re-rolls, it should
be batched. there are no situations in which serializing is superior." Same
rungs, same events, same accept rule under all three. See "THE LADDER: ONE
POLICY, THREE DRIVERS" below.

WHAT THIS CANNOT SEE, stated: a chunk that repeats a section AND drops the rest
lands near the expected length and passes. Only the ASR coverage audit
(`align/`) sees inside a chunk; this guard is the cheap sensor for the common
shapes (Owen: "this solution will, at least, manage the most common problems").
DURATION IS NOT A COVERAGE PROXY for a dropped span inside a chunk that still
runs long (a v3 render measured ratio 0.99 while dropping 22 % of its text).

THE BAND is two numbers, characters of text per second of audio: above
`max_chars_per_sec` the take is too short, below `min_chars_per_sec` it is too
long. Until 2026-09-08 they were FIXED for the whole book, derived by the
catalog from the voice's ladder pace with 15 % headroom on the ladder's own
tails (p99 x 1.15 / p05 / 1.15). MEASURED on the first full book rendered under
that band (Shift, mistborn, 1,313 chunks, 16.4 h), it was the wrong shape:

  THE BOOK PACES ITSELF. The ladder bank is uniform nonfiction prose; Shift is
  fiction with dialogue, and its clean chunks ran at a median of 14.09 chars/s
  against the bank's 15.12 - 7 % slower. So the fixed band's edges sat at the
  wrong distance from THIS book: the long edge (11.97) fired on 340 healthy
  chunks and re-rolled or split every one of them, while the short edge (19.27)
  sat at 1.37x the book's own pace and let takes at 0.76x of their expected
  length ship (998 chars in 53.6 s), which is the truncation Owen heard.
  TINY CHUNKS SAY NOTHING: a heading is 4-18 characters over a second of
  audio with silence around it (Shift: 274 headings, median 8.3 chars/s), so
  every one of them read as a run-on and the ladder chewed them for nothing.

So (Owen, 2026-09-08: "change the guard to re-render if it deviates too far
from the calculated and recorded characters per second") the band now
FOLLOWS THE BOOK - `PaceTracker`:

  - the catalog still writes the voice's recorded pace and a seed band
    (`paceCharsPerSec`, `maxCharsPerSec` = pace x 1.3, `minCharsPerSec` = pace / 1.3
    - BookForge's PACE_GUARD_SHORT_FACTOR and PACE_GUARD_LONG_FACTOR. The short
    edge was 1.2x until 2026-09-08, when Owen's live MLX Shift run showed it
    re-rolling healthy brisk prose: the tracked pace settled at 13.1 (the slow
    opening chunks seed it, 7 % under the 14.09 shipped median measured below),
    so the edge sat at 15.7 and re-rolled takes at 15.75-16.15 chars/s that came
    back 3-8 % longer with the same words. Every real truncation sat at >= 17.3
    absolute; 1.3 x 13.1 = 17.0. IF THE TRACKER IS EVER RE-SEEDED from the
    shipped median, revisit: 14.09 x 1.3 = 18.3 sits inside the 17.3-18.6
    truncation band. The long side is looser because clean dialogue ran down to
    0.79x. The RATIOS of that seed band are the
    deviation the guard tolerates, and they are the only thing kept from it;
  - the guard's reference is the recorded pace until `PACE_WARMUP_CHUNKS`
    guarded takes have shipped, then the running MEDIAN of the shipped takes'
    own chars/s - the book's calculated pace, which a chunk is judged against;
  - a chunk under `MIN_GUARD_CHARS` is judged on the SHORT side only (a 100-
    character chunk cut in half is still twice too fast) and feeds nothing
    into the running pace.

EVERY FIRE IS REPORTED: a log line a person can read and one parseable line,
`[HIGGS3][HIGGS_GUARD_EVENT] {json}`, the shape and prefix discipline of
Orpheus's `[ORPHEUS][ORPHEUS_GUARD_EVENT]` so the bridge's analytics count them
the same way. Every event carries the TAKE-0 verdict (chars, seconds,
chars_per_second, both thresholds) whatever rung it fired on, so the model's
raw off-length rate per voice is readable from the counts without re-rendering
(training's ask).
"""
from __future__ import annotations

import json
import math
import os
import statistics
import threading
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

import numpy as np

from ..log import log
from .codec import FrameMeasure

#: Silence between the halves of a re-split chunk. A sentence boundary inside a
#: paragraph; the same length `v3_served.REFERENCE_JOIN_SECONDS` uses between
#: joined reference clips, chosen for the same reason (a natural pause, not a
#: gap).
RESPLIT_JOIN_SECONDS = 0.35

#: How deep the split ladder goes: 3 halvings turn a 1,200-character chunk into
#: ~150-character parts, well inside any Higgs voice's measured safe zone.
#: Orpheus's ladder uses the same depth.
MAX_DEPTH = 3

#: A part shorter than this is not split further - two fragments of a
#: sentence are worse than one short take. Orpheus's ladder uses 80.
MIN_SPLIT_CHARS = 80

#: The re-roll's seed offset. Large and odd so `seed + index + STRIDE` never
#: collides with another chunk's own seed inside any book.
#:
#: IT IS ALSO THE BOOK'S INDEX BOUND, stated here because `TAKE_SEED_STRIDE`
#: below rests on it: the sentence above is only true while a book has fewer
#: than 100,003 chunks. Measured headroom - The Final Empire, 24.6 h, packs to
#: ~1,750 chunks at Higgs's 600-1,100-char caps, so the bound is ~50x the
#: longest book anyone has rendered.
REROLL_SEED_STRIDE = 100_003

#: How many re-roll lanes ONE TAKE reserves inside the seed space.
#:
#: The ladder spends two of them: `_LadderTask.request` asks for attempt 1 and
#: only attempt 1, so a chunk draws at `attempt = 0` (its plain take) and at
#: `attempt = 1` (the re-roll). Sixteen is headroom for a deeper ladder and
#: costs nothing but arithmetic - the lanes are not allocated, they are a gap
#: in the integers.
TAKE_REROLL_LANES = 16

#: THE TAKE'S SEED OFFSET - the other half of Owen's retake ruling
#: (2026-09-14): *a retake must not reuse the settings that produced the
#: problem.* A SEED IS A SETTING. Until 2026-09-15 the whole ladder was
#: seeded `seed + index` whatever the take, so take 0 and take 7 of chunk i
#: were byte-identical renders whenever their sampling matched - and two
#: take-0 re-rolls always were.
#:
#: WHY THE TWO FAMILIES CANNOT COLLIDE. Every seed narrator draws is
#:
#:     base + index + REROLL_SEED_STRIDE * (TAKE_REROLL_LANES * take + attempt)
#:
#: - `in_take_lane` contributes `TAKE_SEED_STRIDE * take` and `reroll_seed`
#: contributes `REROLL_SEED_STRIDE * attempt`, and `TAKE_SEED_STRIDE` is
#: `REROLL_SEED_STRIDE * TAKE_REROLL_LANES` exactly so the two add into ONE
#: multiple of `REROLL_SEED_STRIDE`. With `0 <= attempt < TAKE_REROLL_LANES`
#: the quantity `TAKE_REROLL_LANES * take + attempt` is distinct for every
#: (take, attempt) pair, and with `0 <= index < REROLL_SEED_STRIDE` the index
#: can never carry a seed into the next multiple. So (take, attempt, index) ->
#: seed is INJECTIVE: no take's draw is any other take's draw, and no take's
#: draw is any re-roll's.
#:
#: The one deliberate coincidence is `take = 0, attempt = 0`, which is `base +
#: index` - the seed rule this file has always had. Take 0 is unchanged, which
#: is what makes this additive.
#:
#: RANGE. `item_sampling.MAX_TAKE` bounds the take so the arithmetic stays
#: inside a signed 32-bit seed (the wire's `int`): 1234 + 999 * 1_600_048 +
#: 100_003 + 100_003 is about 1.6e9, under 2^31. tests/test_serve_sampling.py
#: pins that.
TAKE_SEED_STRIDE = REROLL_SEED_STRIDE * TAKE_REROLL_LANES

#: The parseable prefix. LOAD-BEARING for the bridge's event parser.
GUARD_EVENT_PREFIX = '[HIGGS3][HIGGS_GUARD_EVENT] '

#: Under this many characters a chunk is judged on the SHORT side only and
#: does not feed the running pace. The ladder's own clean rule (`chars > 150`)
#: for the same reason: a chunk that short is a heading or a line of dialogue
#: whose seconds are mostly the silence around it (Shift: 281 chunks under
#: 150 chars, median 8.5 chars/s, p99 15.2 - not one over the short edge).
MIN_GUARD_CHARS = 150

#: The longest INTERIOR silence a take may carry before it is a hole - a run of
#: the model sitting in silence at a sentence join and taking a while to leave
#: it, which the rate band cannot see: a 7-11 s hole inside a 45 s chunk moves
#: the chunk from ~14 to ~11-12 chars/s, still over the floor, and ships
#: (Fuhrer (5), 2026-09-11: 12 of 88 chunks carried one >=3 s, the worst 11.6 s;
#: one split child shipped its last sentence DROPPED plus 18.5 s of silence).
#: MEASURED ceiling for a real pause: the reader's own max over 24.6 h of
#: The Final Empire is 5.02 s and that is a chapter break; the training corpora
#: (mb7_l, tr3_l) max at 4.0-4.5 s (the slicer's --gap-s 4). So 5.0 fires on
#: nothing a narrator does and on every hole anyone could hear. Not lower: the
#: 4.5-5 s band has not been listened to (bookforge-mac-1, 2026-09-11).
#: Silence = 10 ms frames whose peak is under HOLE_FLOOR (-34 dBFS absolute),
#: the same gate the pause maps that set this number used; leading and trailing
#: silence are NOT holes (the short/long band already prices them).
MAX_HOLE_SECONDS = 5.0
HOLE_FLOOR = 0.02
HOLE_HOP_SECONDS = 0.01

#: Guarded takes the tracker sees before its reference moves from the recorded
#: pace to the book's own median. Ten is one batch on the served arm; a median
#: of fewer is one odd chunk.
PACE_WARMUP_CHUNKS = 10


class PaceTracker:
    """The band, re-centred on the book as it renders.

    Built once per engine from the voice's recorded pace and its seed band;
    `band()` is what `render_guarded` judges a take against and `observe()` is
    what it feeds the shipped take back into. Thread-safe: the served arm
    renders a batch on a pool.
    """

    def __init__(self, seed_pace: float, max_chars_per_sec: float,
                 min_chars_per_sec: float, *, warmup: int = PACE_WARMUP_CHUNKS,
                 min_chars: int = MIN_GUARD_CHARS):
        for label, value in (('seed_pace', seed_pace), ('max_chars_per_sec', max_chars_per_sec),
                             ('min_chars_per_sec', min_chars_per_sec)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f'PaceTracker: {label} must be a positive number; got {value!r}')
        if not (min_chars_per_sec < seed_pace < max_chars_per_sec):
            raise ValueError(
                f'PaceTracker: the seed band is min < pace < max; got min {min_chars_per_sec}, '
                f'pace {seed_pace}, max {max_chars_per_sec}')
        self.seed_pace = float(seed_pace)
        #: The tolerated deviation, as the seed band's own ratios: the short
        #: edge sits at reference x `short_ratio`, the long edge at
        #: reference / `long_ratio`.
        self.short_ratio = float(max_chars_per_sec) / self.seed_pace
        self.long_ratio = self.seed_pace / float(min_chars_per_sec)
        self.warmup = int(warmup)
        self.min_chars = int(min_chars)
        self._observed: List[float] = []
        self._lock = threading.Lock()

    @property
    def observed(self) -> int:
        with self._lock:
            return len(self._observed)

    @property
    def warm(self) -> bool:
        return self.observed >= self.warmup

    @property
    def reference(self) -> float:
        """The pace a take is judged against: the book's own median once warm,
        the recorded pace before that."""
        with self._lock:
            if len(self._observed) >= self.warmup:
                return statistics.median(self._observed)
        return self.seed_pace

    def band(self) -> dict:
        """Keyword-ready for `render_guarded`, and re-read on EVERY take."""
        ref = self.reference
        return {'max_chars_per_sec': round(ref * self.short_ratio, 2),
                'min_chars_per_sec': round(ref / self.long_ratio, 2)}

    def observe(self, chars: int, seconds: float) -> None:
        """A shipped take. Under `min_chars` it is not a pace measurement."""
        if chars < self.min_chars or seconds <= 0:
            return
        with self._lock:
            self._observed.append(chars / seconds)


@dataclass(frozen=True)
class LengthVerdict:
    chars: int
    seconds: float
    chars_per_second: float
    max_chars_per_sec: float   # above this: too SHORT (0 disables)
    min_chars_per_sec: float   # below this: too LONG  (0 disables)
    hole_seconds: float = 0.0      # the longest interior silence in the take
    max_hole_seconds: float = 0.0  # above this: a HOLE (0 disables)

    @property
    def short(self) -> bool:
        return self.max_chars_per_sec > 0 and self.chars_per_second > self.max_chars_per_sec

    @property
    def long(self) -> bool:
        return (self.min_chars_per_sec > 0 and self.chars > 0
                and self.chars_per_second < self.min_chars_per_sec)

    @property
    def hole(self) -> bool:
        return self.max_hole_seconds > 0 and self.hole_seconds > self.max_hole_seconds

    @property
    def off_length(self) -> bool:
        return self.short or self.long or self.hole

    @property
    def side(self) -> Optional[str]:
        if self.short:
            return 'short'
        if self.long:
            return 'long'
        return 'hole' if self.hole else None


def interior_hole_seconds(audio, sample_rate: int) -> float:
    """The longest run of silence strictly INSIDE `audio`, in seconds: frames of
    `HOLE_HOP_SECONDS` whose peak is under `HOLE_FLOOR`, with the leading and
    trailing runs excluded (those are tails, priced by the band). 0.0 for an
    empty, all-silent or hole-free take."""
    if audio is None or len(audio) == 0:
        return 0.0
    hop = max(1, int(round(HOLE_HOP_SECONDS * sample_rate)))
    samples = np.asarray(audio, dtype=np.float32)
    frames = len(samples) // hop
    if frames < 3:
        return 0.0
    peaks = np.abs(samples[:frames * hop]).reshape(frames, hop).max(axis=1)
    quiet = peaks < HOLE_FLOOR
    loud = np.flatnonzero(~quiet)
    if len(loud) < 2:
        return 0.0                      # no interior: silence touches an end
    # Only runs between the first and last loud frame are interior.
    inner = quiet[loud[0]:loud[-1] + 1]
    longest = run = 0
    for q in inner:
        run = run + 1 if q else 0
        if run > longest:
            longest = run
    return longest * hop / float(sample_rate)


def check(text: str, audio, sample_rate: int, max_chars_per_sec: float,
          min_chars_per_sec: float = 0.0,
          max_hole_seconds: float = 0.0) -> LengthVerdict:
    """Is `audio` the wrong length for `text`, or holed? Characters per second
    against the band, and the longest interior silence against
    `max_hole_seconds`; a threshold of 0 disables that side."""
    chars = len((text or '').strip())
    seconds = float(len(audio)) / float(sample_rate) if audio is not None else 0.0
    cps = (chars / seconds) if seconds > 0 else float('inf')
    hole = interior_hole_seconds(audio, sample_rate) if max_hole_seconds > 0 else 0.0
    return LengthVerdict(chars=chars, seconds=round(seconds, 3),
                         chars_per_second=round(cps, 2) if cps != float('inf') else cps,
                         max_chars_per_sec=float(max_chars_per_sec),
                         min_chars_per_sec=float(min_chars_per_sec),
                         hole_seconds=round(hole, 2),
                         max_hole_seconds=float(max_hole_seconds))


#: The band's three numbers, in the spelling the wire and the voices document
#: share. ONE spelling, because the band Crucible puts on a batch is the band
#: BookForge read off the voice row.
BAND_KEYS = ('paceCharsPerSec', 'maxCharsPerSec', 'minCharsPerSec')


def parse_band(value, where: str) -> dict:
    """A band off the wire -> the same three numbers, checked, or a refusal
    that names `where`.

    THREE POSITIVE NUMBERS WITH `min < pace < max`, and nothing filled in: a
    band with two of the three is not a band with a default in the third, it is
    a caller that has not measured one. `PaceTracker.__init__` enforces the
    same two rules - this exists so the refusal lands on the BATCH, before a
    row renders, rather than as one chunk's exception half way through.
    """
    if not isinstance(value, dict):
        raise ValueError(
            f'{where}: the band must be an object with {", ".join(BAND_KEYS)}; '
            f'got {type(value).__name__}.')
    out = {}
    for key in BAND_KEYS:
        if key not in value:
            raise ValueError(
                f'{where}: the band carries no {key!r}. All three of '
                f'{", ".join(BAND_KEYS)} are required - a band is a MEASUREMENT '
                'of one voice at one checkpoint, and two thirds of one is not a '
                'band with the rest defaulted.')
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or number <= 0:
            raise ValueError(
                f'{where}: the band\'s {key} must be a positive number; got '
                f'{number!r}.')
        out[key] = float(number)
    if not (out['minCharsPerSec'] < out['paceCharsPerSec'] < out['maxCharsPerSec']):
        raise ValueError(
            f'{where}: the band is min < pace < max; got min '
            f'{out["minCharsPerSec"]}, pace {out["paceCharsPerSec"]}, max '
            f'{out["maxCharsPerSec"]}.')
    return out


def engine_band(max_chars_per_sec: float, min_chars_per_sec: float) -> dict:
    """THE ENGINE'S OWN band, centred on its geometric mean - the Fuhrer /
    deathstalker whole-book measurements that set `MAX_CHARS_PER_SEC` and
    `MIN_CHARS_PER_SEC`, not a guess.

    It is what narrator's own audiobook path (`convert`, `convert_many`) judges
    against, because there the renderer IS the client and has no other band to
    be given. IT IS NOT A FALLBACK FOR THE SERVE DOOR: a `generate_batch` that
    asks to be judged must carry the band it wants to be judged against, and is
    refused by name when it does not (`retake_without_band`). That fallback WAS
    the defect - a band centred at 15.0 against a book running at 17.2 chars/s
    called healthy chunks run-ons and re-rolled them to MAX_DEPTH.
    """
    return {'paceCharsPerSec': expected_chars_per_sec(float(max_chars_per_sec),
                                                      float(min_chars_per_sec)),
            'maxCharsPerSec': float(max_chars_per_sec),
            'minCharsPerSec': float(min_chars_per_sec)}


def tracker_for(band, where: str = 'tracker_for') -> PaceTracker:
    """A `PaceTracker` on THE BAND IT IS GIVEN, and on nothing else.

    ONE OWNER FOR THE BAND (Owen's ruling, 2026-09-19). Until then this
    function read three keys off the VOICE entry - `paceCharsPerSec` /
    `maxCharsPerSec` / `minCharsPerSec`, whichever the catalog happened to
    carry - and fell back to the engine's default band when it found none. Two
    sources, silently chosen between, and the fallback is the one that fired
    for every voice whose document had not been measured: the guard then judged
    a 17.2 chars/s book against a band centred on 15.0, called healthy chunks
    run-ons, and re-rolled them to MAX_DEPTH.

    So the band is now an ARGUMENT. The serve door gets it from the batch
    (`generate_batch`'s `band`, which is the row BookForge read off the voice);
    narrator's own audiobook path gets it from `engine_band`, the engine's own
    measured pair, and says so at the call site. Nothing reads a voice entry to
    build a tracker any more - the voice document still carries the keys, and
    they still reach `ClipsVoice`, for the CLIENT that packs and sends them.
    """
    checked = parse_band(band, where)
    return PaceTracker(checked['paceCharsPerSec'], checked['maxCharsPerSec'],
                       checked['minCharsPerSec'])


def expected_chars_per_sec(max_chars_per_sec: float, min_chars_per_sec: float) -> float:
    """The band's centre - the geometric mean of its two edges, or whichever
    edge exists. What `accept` measures a take's distance from."""
    if max_chars_per_sec > 0 and min_chars_per_sec > 0:
        return math.sqrt(max_chars_per_sec * min_chars_per_sec)
    return max_chars_per_sec or min_chars_per_sec


def reroll_seed(base_seed: Optional[int], index: int, attempt: int) -> Optional[int]:
    """The seed for re-roll `attempt` (1-based) of chunk `index`. None stays
    None - an unseeded engine samples fresh anyway, which IS the re-roll."""
    if base_seed is None:
        return None
    return int(base_seed) + int(index) + REROLL_SEED_STRIDE * int(attempt)


def in_take_lane(seed: Optional[int], take: int) -> Optional[int]:
    """`seed`, moved into TAKE `take`'s own lane - see `TAKE_SEED_STRIDE` for
    why the lanes cannot collide with the re-roll's.

    THE ONE PLACE A TAKE TOUCHES A SEED, and it is applied to whatever seed the
    ladder already chose rather than only to the engine's `seed + index` rule.
    That is what keeps take N's WHOLE ladder inside take N's lane: take 0's
    seed, the re-roll's `reroll_seed`, and a split half's own all get the same
    shift, so a retake at take 3 re-rolls at a seed take 0 never drew either.

    `take = 0` returns the seed unchanged, which is why this is additive: every
    caller that knows nothing about takes renders exactly what it rendered
    before.

    None stays None, for `reroll_seed`'s reason: an unseeded engine samples
    fresh on every call, so its take N already IS a different draw and there is
    no arithmetic to do.
    """
    if seed is None:
        return None
    return int(seed) + TAKE_SEED_STRIDE * int(take)


def split_halves(text: str) -> List[str]:
    """`text` as two parts cut at the sentence boundary nearest its middle, or
    at the space nearest the middle when it is one sentence. `[]` when it
    cannot be split into two parts of at least `MIN_SPLIT_CHARS` each."""
    from ...text.sentences import split_sentences
    text = (text or '').strip()
    if len(text) < 2 * MIN_SPLIT_CHARS:
        return []
    sentences = [s for s in split_sentences(text) if s.strip()]
    if len(sentences) >= 2:
        best, best_gap = None, None
        total = len(text)
        running = 0
        for i in range(1, len(sentences)):
            running += len(sentences[i - 1]) + 1
            gap = abs(running - total / 2)
            if best_gap is None or gap < best_gap:
                best, best_gap = i, gap
        left = ' '.join(sentences[:best]).strip()
        right = ' '.join(sentences[best:]).strip()
    else:
        mid = len(text) // 2
        cut = text.rfind(' ', 0, mid)
        if cut < 0:
            cut = text.find(' ', mid)
        if cut < 0:
            return []
        left, right = text[:cut].strip(), text[cut:].strip()
    if len(left) < MIN_SPLIT_CHARS or len(right) < MIN_SPLIT_CHARS:
        return []
    return [left, right]


def join_parts(parts: List[np.ndarray], sample_rate: int) -> np.ndarray:
    """The halves, with `RESPLIT_JOIN_SECONDS` of silence between them."""
    gap = np.zeros(int(round(RESPLIT_JOIN_SECONDS * sample_rate)), dtype=np.float32)
    out: List[np.ndarray] = []
    for i, part in enumerate(parts):
        if i:
            out.append(gap)
        out.append(np.asarray(part, dtype=np.float32))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


#: Where a take the guard threw away is kept, set per job by
#: `parallel-tts-bridge.ts` beside `ORPHEUS_REJECT_DIR` (the same directory: one
#: place to look for either engine's evidence). UNSET = keep nothing, which is
#: what a CLI render or a test does.
REJECT_DIR_ENV = 'HIGGS_REJECT_DIR'

#: `events.jsonl` is one shared append-only file and the served arm renders a
#: batch on a pool.
_reject_lock = threading.Lock()


def keep_reject(audio, sample_rate: int, record: dict, text: str) -> None:
    """Preserve a take the guard threw away, and WHY.

    Without this a truncation is only ever a log line: the re-roll overwrites the
    bad audio, so nothing can say WHERE it cut or WHAT it said instead - and
    those are the two facts a fine-tune needs. Owen, 2026-09-08, looking at a
    screen of guard fires on Shift: "we should probably whisper them and see what
    was missing. how much was missing, where it stopped, etc. and pass it to the
    training agent." Orpheus has kept its rejects since 2026-07-28 for the same
    reason (`engine/orpheus/guards.py::_keep_reject`); this is that, for Higgs.

    Writes `<index>_d<depth>_<rung>_<side>.wav` + `.json` and appends the record
    to `events.jsonl`, so a post-mortem is one read rather than a directory walk.
    THE TEXT IS IN THE RECORD: the transcript diff is the whole point, and the
    session that produced the chunk may be gone by the time anyone looks.

    BEST-EFFORT BY DESIGN, and the one place in this module that swallows: this
    is diagnostics, not product, and losing a post-mortem must never take down a
    book. The failure is logged by name.
    """
    directory = os.environ.get(REJECT_DIR_ENV, '').strip()
    if not directory:
        return
    try:
        import json as _json

        import soundfile as sf
        with _reject_lock:
            os.makedirs(directory, exist_ok=True)
            stem = os.path.join(
                directory,
                f"{int(record.get('index', 0)):06d}_d{int(record.get('depth', 0))}"
                f"_{record.get('rung', 'take')}_{record.get('side', 'off')}")
            if audio is not None and len(audio) > 0:
                sf.write(stem + '.wav', np.asarray(audio, dtype=np.float32),
                         int(sample_rate), subtype='PCM_16', format='WAV')
            full = {**record, 'text': text, 'sample_rate': int(sample_rate)}
            with open(stem + '.json', 'w', encoding='utf-8') as handle:
                _json.dump(full, handle, indent=1, ensure_ascii=False)
            with open(os.path.join(directory, 'events.jsonl'), 'a',
                      encoding='utf-8') as handle:
                handle.write(_json.dumps(full, ensure_ascii=False) + '\n')
    except Exception as err:      # noqa: BLE001 - diagnostics never take a book down
        log(f'[HIGGS3] could not keep the rejected take for chunk '
            f"{record.get('index')} ({err})", flush=True)


def emit_event(record: dict) -> None:
    """One human line and one parseable line per guard fire."""
    log(f"[HIGGS3] length guard: chunk {record.get('index')} {record.get('action')} - "
        f"{record.get('chars')} chars in {record.get('seconds')} s "
        f"({record.get('chars_per_second')} chars/s; band {record.get('min_chars_per_sec')}"
        f"-{record.get('max_chars_per_sec')})"
        + (f" depth {record['depth']}" if record.get('depth') else ''), flush=True)
    log(GUARD_EVENT_PREFIX + json.dumps(record, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------------------
# THE LADDER: ONE POLICY, THREE DRIVERS
# ---------------------------------------------------------------------------
# The rungs above are the policy; WHEN their renders happen is the driver's.
# `render_guarded` drives one chunk serially, depth-first - the ladder's own
# order, unchanged, and what every single-chunk path uses. `GuardPlan` drives
# many chunks at once, and a driver takes work from it in whichever shape its
# backend's batch has: `round()` for a slab (MLX), `next_request()` per free slot
# for a pool of concurrent requests (the served arm's `convert_many`).
#
# Owen, 2026-09-08 (through the Mac, after his MLX Shift run): "maybe we should
# batch the re-renders. take note of which ones we need to re-render and add
# them to the task list, and batch them at the end. instead of serializing every
# single one. would that be more efficient". MEASURED reason it is: an MLX batch
# of 32 rows costs about the wall time of ONE row, so a solo re-roll costs a
# whole batch - his log slice had 4 re-rolls in ~28 chunks, which roughly
# doubled the run. The served arm has no such asymmetry (a re-roll there is one
# more concurrent POST against a server that batches continuously), which is why
# it keeps the serial driver. Orpheus solved the same problem the same way, in
# `vllm_backend._render_deferred_resplits`.
#
# THE POLICY IS THE SAME UNDER EITHER DRIVER: same rungs in the same order, same
# events, same accept rule, same tracker discipline. Two things do differ under
# the batch driver, and both are stated rather than hidden:
#   - a re-roll is judged against the band AS IT STANDS WHEN ITS ROUND RUNS, by
#     which time the clean chunks of the same batch have shipped and fed the
#     running median. That is the better reference, not a worse one.
#   - a re-roll's sampling comes from its ROUND's batch seed (`reroll_seed` of
#     the round's first chunk) rather than from its own solo seed, because MLX
#     draws one RNG stream per batch. What the rung needs is a draw the first
#     take did not make; both give that, and a batched re-roll is reproducible
#     as a batch - the same guarantee `_generate_delayed_rows_batch` already
#     makes for take 0. (The POOL driver has neither difference: each request
#     carries its own seed to its own HTTP call.)


@dataclass(frozen=True)
class RenderRequest:
    """One render the ladder wants next, for the driver to satisfy.

    `text` at `seed` - the two arguments the serial `render(text, seed)`
    callable takes, with None meaning the engine's own seed rule for `index`.
    `path` is the ladder position that asked for it, handed straight back to
    `GuardPlan.offer`.
    """
    path: tuple
    index: int
    text: str
    seed: Optional[int]
    rung: str        # 'take' (take 0) | 'reroll' | 'part' (a split half's take 0)
    depth: int


class _LadderTask:
    """One text unit on the ladder: a chunk, or a half of one after a split.

    Advances exactly one rung per take offered to it. At any moment a task is
    WAITING for a render (`request()` says which), waiting for its children, or
    done.
    """

    def __init__(self, plan, path: tuple, index: int, text: str,
                 depth: int, parent=None):
        self.plan = plan
        self.path = path
        self.index = index
        self.text = text
        self.depth = depth
        self.parent = parent
        self.takes: List[tuple] = []
        self.children: List['_LadderTask'] = []
        self.stage = 'take0'      # take0 -> reroll -> children | accept -> done
        self.base: dict = {}
        self.audio = None
        #: The `FrameMeasure` of the take that SHIPS - see `_finish`. None
        #: until this task is decided, and None afterwards for a driver that
        #: measures nothing (every test that offers bare audio).
        self.measure = None
        self.clean = False
        self.done = False

    # -- what it wants next -------------------------------------------------
    def request(self) -> Optional[RenderRequest]:
        if self.stage == 'take0':
            return RenderRequest(self.path, self.index, self.text, None,
                                 'part' if self.depth else 'take', self.depth)
        if self.stage == 'reroll':
            return RenderRequest(self.path, self.index, self.text,
                                 reroll_seed(self.plan.base_seed, self.index, 1),
                                 'reroll', self.depth)
        return None

    # -- one rung -----------------------------------------------------------
    def offer(self, audio, measure=None) -> None:
        if self.stage not in ('take0', 'reroll'):
            raise RuntimeError(
                f'GuardPlan: chunk {self.index} at {self.path} was handed a take '
                f'while it was waiting on {self.stage}.')
        max_edge, min_edge = self.plan.edges()
        chars = len((self.text or '').strip())
        # The short-chunk exemption is for headings and lines of dialogue, whose
        # seconds are mostly the silence around them. A split CHILD is neither:
        # it is half of a chunk that already ran away, and the likeliest text in
        # the book to run away again - Fuhrer (5) chunk 8's 140-char half shipped
        # its last sentence dropped plus 18.5 s of silence through this gap.
        long_edge = min_edge if (chars >= MIN_GUARD_CHARS or self.depth > 0) else 0.0
        verdict = check(self.text, audio, self.plan.sample_rate, max_edge, long_edge,
                        max_hole_seconds=MAX_HOLE_SECONDS)

        if self.stage == 'take0':
            if not verdict.off_length:
                self._finish(audio, True, measure)
                return
            self.takes.append((audio, verdict, measure))
            self.base = {'index': self.index, 'depth': self.depth,
                         'side': verdict.side, **asdict(verdict)}
            if self.plan.tracker is not None:
                # WHAT THE BAND WAS CENTRED ON, so a reader of the log can tell a
                # band still on the recorded pace from one that has moved to the
                # book's.
                self.base['pace'] = round(self.plan.tracker.reference, 2)
                self.base['pace_source'] = ('book' if self.plan.tracker.warm
                                            else 'recorded')
            # Rung 1: the re-roll, at a seed the first take did not use.
            self.plan.on_event({**self.base, 'action': verdict.side, 'rung': 'reroll'})
            # THE DISCARDED TAKE IS THE EVIDENCE. Kept when the run names a
            # directory to keep it in; a no-op otherwise. See `keep_reject`.
            keep_reject(audio, self.plan.sample_rate,
                        {**self.base, 'action': verdict.side, 'rung': 'take0'},
                        self.text)
            self.stage = 'reroll'
            return

        # stage == 'reroll'
        if not verdict.off_length:
            self.plan.on_event({**self.base, 'action': 'rerolled', 'rung': 'reroll',
                                'seconds_after': verdict.seconds})
            self._finish(audio, True, measure)
            return
        self.takes.append((audio, verdict, measure))
        keep_reject(audio, self.plan.sample_rate,
                    {**self.base, 'action': verdict.side, 'rung': 'reroll',
                     'seconds_after': verdict.seconds,
                     'chars_per_second_after': verdict.chars_per_second},
                    self.text)

        # Rung 2: the split, each half through this same ladder.
        parts = split_halves(self.text) if self.depth < MAX_DEPTH else []
        if parts:
            self.plan.on_event({**self.base, 'action': 'resplit', 'rung': 'split',
                                'reroll_side': verdict.side,
                                'parts': [len(p) for p in parts]})
            self.stage = 'children'
            self.children = [self.plan.child(self, position, part)
                             for position, part in enumerate(parts)]
            return
        self._accept(max_edge, min_edge)

    # -- rung 3 -------------------------------------------------------------
    def _accept(self, max_edge: float, min_edge: float) -> None:
        """The take closest to the expected length ships, and the event says so:
        the render never refuses (Owen, 2026-09-05), and the coverage audit
        names the chunk."""
        centre = expected_chars_per_sec(max_edge, min_edge)

        def distance(tv):
            # A holed take loses to any take without one, and between two holed
            # takes the smaller hole wins - otherwise a re-roll that closed the
            # hole but landed 5% long would lose to the original on length alone.
            verdict = tv[1]
            hole = verdict.hole_seconds if verdict.hole else 0.0
            if verdict.chars_per_second == float('inf'):
                return (hole, float('inf'))
            return (hole, abs(math.log(max(verdict.chars_per_second, 1e-9)) - math.log(centre)))
        best, best_verdict, best_measure = min(self.takes, key=distance)
        self.plan.on_event({**self.base, 'action': 'accepted-off-length', 'rung': 'accept',
                            'shipped_side': best_verdict.side,
                            'seconds_shipped': best_verdict.seconds,
                            'why': ('at MAX_DEPTH' if self.depth >= MAX_DEPTH
                                    else 'text cannot be split')})
        self._finish(best, False, best_measure)

    def _finish(self, audio, clean: bool, measure=None) -> None:
        """This task is decided: `audio` ships, and `measure` is what THAT
        take spent. A parent's is its children's, joined - see
        `FrameMeasure.joined`, which is where "one half cut off is a chunk cut
        off" lives."""
        self.audio = audio
        self.measure = measure
        self.clean = bool(clean)
        self.stage = 'done'
        self.done = True
        if self.parent is None:
            self.plan.finish_root(self)
            return
        siblings = self.parent.children
        if all(child.done for child in siblings):
            self.parent._finish(
                join_parts([child.audio for child in siblings], self.plan.sample_rate),
                all(child.clean for child in siblings),
                FrameMeasure.joined([child.measure for child in siblings]))


class GuardPlan:
    """Chunks through the ladder, a ROUND of renders at a time.

    The batch driver's loop:

        plan = GuardPlan(sample_rate=..., base_seed=..., tracker=...)
        for index, text, take0 in batch:       # take 0 the engine already has
            plan.add(index, text, first_take=take0)
        ship(plan.finished())                  # everything take 0 got right
        while True:
            requests = plan.round()            # every retake the guard wants
            if not requests:
                break
            for request, audio in zip(requests, render_as_one_batch(requests)):
                plan.offer(request, audio)
            ship(plan.finished())

    `round()` is ordered by ladder position, so a driver that takes only its
    FIRST request walks the ladder depth-first - which is exactly what
    `render_guarded` does, and why the two drivers cannot drift apart.

    NOT THREAD-SAFE, deliberately: one plan belongs to one batch on one thread.
    The `PaceTracker` it feeds is the shared, locked one.
    """

    def __init__(self, *, sample_rate: int, base_seed: Optional[int],
                 tracker: Optional[PaceTracker] = None,
                 max_chars_per_sec: float = 0.0, min_chars_per_sec: float = 0.0,
                 on_event: Callable[[dict], None] = emit_event):
        self.sample_rate = int(sample_rate)
        self.base_seed = base_seed
        self.tracker = tracker
        self.fixed = (float(max_chars_per_sec), float(min_chars_per_sec))
        #: THE EVENT SINK IS WRAPPED, NOT REPLACED. `_record` collects every
        #: event this plan emits, keyed by chunk, and then passes it to the
        #: caller's sink unchanged - so the `[HIGGS3][HIGGS_GUARD_EVENT]` lines
        #: a driver has always printed keep being printed, byte for byte, while
        #: the same records become readable as DATA through `verdict()`.
        #:
        #: Owen's ruling of 2026-09-13 is why: the model and its inference own
        #: the guard AND the decision, so a driver that ships a chunk over a
        #: wire has to be able to ship the verdict with it. Scraping a log line
        #: works when the renderer and its caller share a stdout; it does not
        #: work over a socket. See crucible/docs/PHASE6-REMOTE-RENDER.md.
        self._sink = on_event
        self.on_event = self._record
        self._records: dict = {}
        self._verdicts: dict = {}
        self._measures: dict = {}
        self._tasks: dict = {}
        self._added = 0
        self._finished: List[tuple] = []
        #: Paths handed to a driver and not yet answered. A POOL driver asks
        #: for one request at a time and must never be given the same one
        #: twice; the round driver takes them all at once, which is the same
        #: rule with a wider hand.
        self._issued: set = set()

    # -- the verdict, as data -----------------------------------------------
    def _record(self, record: dict) -> None:
        """Collect one guard event, then emit it unchanged.

        NOTHING HERE DECIDES ANYTHING. This is the same record the log line is
        built from, kept instead of only printed.
        """
        index = record.get('index')
        if index is not None:
            self._records.setdefault(int(index), []).append(record)
        self._sink(record)

    @staticmethod
    def _leaves(task) -> int:
        """How many text units this chunk was finally rendered as: 1 unless the
        ladder split it, and the product of every split below that."""
        if not task.children:
            return 1
        return sum(GuardPlan._leaves(child) for child in task.children)

    def _build_verdict(self, task) -> dict:
        """What the guard decided about one chunk, for a driver to ship beside
        the audio. PHASE6-REMOTE-RENDER.md section 3.

        `takes` is the event records VERBATIM - the evidence, unsummarised,
        because a client that wants to know why can read it and a client that
        does not can read `verdict` alone. A clean take 0 emits no event at all
        (`_LadderTask.offer` finishes before it records one), so `takes` is
        empty and `verdict` is `clean`: the common case costs one empty list.
        """
        records = self._records.get(task.index, [])
        max_edge, min_edge = self.edges()
        band = {
            'max_chars_per_sec': max_edge,
            'min_chars_per_sec': min_edge,
            'reference': (round(self.tracker.reference, 2)
                          if self.tracker is not None else None),
            'observed': self.tracker.observed if self.tracker is not None else None,
            'warm': self.tracker.warm if self.tracker is not None else None,
        }
        return {
            # The last thing that happened to it, or `clean` when nothing did.
            # Deliberately the ladder's own vocabulary ('rerolled', 'resplit',
            # 'accepted-off-length', a side name) rather than a taxonomy
            # invented here: a word this file does not already emit would be a
            # word that could drift from what the ladder actually did.
            'verdict': records[-1].get('action') if records else 'clean',
            'clean': bool(task.clean),
            'parts': self._leaves(task),
            'band': band,
            'takes': records,
        }

    def verdict(self, index: int) -> Optional[dict]:
        """The verdict for a chunk `finished()` has handed back, once.

        POPS. A book is not a small number of chunks and the records are kept
        only until the driver that asked for them has them.
        """
        return self._verdicts.pop(int(index), None)

    def measure(self, index: int) -> Optional[FrameMeasure]:
        """The `FrameMeasure` of the take this chunk SHIPPED, once.

        POPS, for `verdict`'s reason. None when the driver offered no measure,
        which is honest: the wire reads a missing measurement as "narrator did
        not say" and never as "not capped".
        """
        return self._measures.pop(int(index), None)

    # -- the band -----------------------------------------------------------
    def edges(self) -> tuple:
        """`(max_chars_per_sec, min_chars_per_sec)` for a take being judged NOW:
        the tracker's, read fresh, or the fixed pair when there is no tracker."""
        if self.tracker is None:
            return self.fixed
        band = self.tracker.band()
        return band['max_chars_per_sec'], band['min_chars_per_sec']

    # -- building -----------------------------------------------------------
    def add(self, index: int, text: str, first_take=None,
            first_measure=None) -> None:
        """A chunk onto the ladder. `first_take` is take 0 when the caller
        already has it (every batch path), so it is never rendered twice, and
        `first_measure` is what THAT take spent - the slab arms measure their
        rows as they decode them and would otherwise have nowhere to put it."""
        task = _LadderTask(self, (self._added,), int(index), text, 0)
        self._tasks[task.path] = task
        self._added += 1
        if first_take is not None:
            task.offer(first_take, first_measure)

    def child(self, parent, position: int, text: str):
        """A split half. Its path sorts INSIDE its parent's, so depth-first
        order falls out of sorting the pending tasks."""
        task = _LadderTask(self, parent.path + (position,), parent.index, text,
                           parent.depth + 1, parent=parent)
        self._tasks[task.path] = task
        return task

    # -- the round ----------------------------------------------------------
    def round(self) -> List[RenderRequest]:
        """Every render the ladder is waiting on, in ladder order. Empty once
        every chunk added has been decided."""
        requests = []
        for path in sorted(self._tasks):
            if path in self._issued:
                continue
            request = self._tasks[path].request()
            if request is not None:
                requests.append(request)
                self._issued.add(path)
        return requests

    def next_request(self) -> Optional[RenderRequest]:
        """ONE render the ladder is waiting on, in ladder order, or None.

        For a POOL driver: it takes a request whenever it has a free slot, so a
        chunk's re-roll and a split's two halves are units of work in the same
        pool as every other chunk rather than steps a slot walks through. Taking
        the first waiting request every time is what makes the serial driver
        depth-first (`render_guarded`).
        """
        for path in sorted(self._tasks):
            if path in self._issued:
                continue
            request = self._tasks[path].request()
            if request is not None:
                self._issued.add(path)
                return request
        return None

    def offer(self, request: RenderRequest, audio, measure=None) -> None:
        """The audio for one of `round()`'s requests, and what it SPENT.

        `measure` is that take's `FrameMeasure` - optional because a driver
        that cannot measure (a test offering bare audio) must still be able to
        drive the ladder, and because the measurement is a fact the ladder
        never reads. It is carried, not consulted: no rung, no accept rule and
        no event here looks at it. It rides to whichever take ends up shipping
        and leaves through `GuardPlan.measure`.
        """
        task = self._tasks.get(request.path)
        if task is None:
            raise KeyError(f'GuardPlan.offer: no task at {request.path}')
        self._issued.discard(request.path)
        task.offer(audio, measure)

    def abandon(self, request: RenderRequest) -> int:
        """The driver could not render this request; the CHUNK leaves the ladder.

        A chunk half-way through the ladder has no audio to ship - the take that
        would have been accepted is the one that failed - so the whole tree under
        its root is dropped and the chunk index is returned for the driver to
        report as that row's failure. It never reaches `finished()`: a caller
        that ships what `finished()` gives it can never ship an abandoned chunk.
        """
        task = self._tasks.get(request.path)
        if task is None:
            raise KeyError(f'GuardPlan.abandon: no task at {request.path}')
        root = task
        while root.parent is not None:
            root = root.parent
        for path in [p for p in self._tasks if p[:len(root.path)] == root.path]:
            self._issued.discard(path)
            del self._tasks[path]
        # The chunk failed and will never be finished, so nothing will ever ask
        # for its verdict. Leaving the records would leak one book's worth.
        self._records.pop(root.index, None)
        self._verdicts.pop(root.index, None)
        self._measures.pop(root.index, None)
        return root.index

    def finish_root(self, task) -> None:
        """A whole chunk is decided. THE SHIPPED TAKE FEEDS THE TRACKER ONLY
        WHEN IT IS CLEAN - an `accepted-off-length` take is a defect the ladder
        could not fix, and feeding it back would pull the reference toward the
        defect (Owen's worry, 2026-09-08: a truncation in the earliest chunks).
        """
        if self.tracker is not None and task.clean:
            self.tracker.observe(len((task.text or '').strip()),
                                 float(len(task.audio)) / float(self.sample_rate))
        self._verdicts[task.index] = self._build_verdict(task)
        self._measures[task.index] = task.measure
        self._records.pop(task.index, None)
        self._finished.append((task.index, task.audio, task.clean))

    def finished(self) -> List[tuple]:
        """`(index, audio, clean)` for every chunk decided since the last call.
        Drains, so a driver ships as it goes."""
        out, self._finished = self._finished, []
        return out

    @property
    def pending(self) -> int:
        """Chunks still on the ladder."""
        return sum(1 for task in self._tasks.values()
                   if task.parent is None and not task.done)


class UnjudgedPlan:
    """The ladder's driver surface, with no ladder: one take per chunk, shipped.

    WHY THIS IS A CLASS AND NOT A FLAG ON `GuardPlan`. A batch that carries
    `retake: false` has no band to be judged against — a screening render of a
    fine-tuned checkpoint has no measured pace, because MEASURING IT IS WHAT THE
    RENDER IS FOR (crucible/docs/PHASE18-UNCERTIFIED.md). `GuardPlan(tracker=
    None)` is NOT that: it falls back to `self.fixed`, so it still judges, just
    against a constant. "Judge against a different model's numbers" and "do not
    judge" are different instructions and only one of them is what the wire
    said.

    WHAT IT FIXES, measured on the training PC 2026-09-20. `generate_batch`
    picked the guarded arm `if rows and retake:` and a sequential comprehension
    otherwise, so the two arms differed in TWO ways: the guarded one judged AND
    batched, the bare one did neither. A 128-chunk render at width 4, with
    `HIGGS_MAX_NUM_SEQS=4` exported and sglang demonstrably willing (zero
    rejections across 1,024 renders at concurrency 4, direct), sat at
    `#running-req: 1` on all 684 sampled scheduler lines — 12.9 s/chunk against
    4.16 s/render direct. The client that must send `retake: false` is the
    client rendering a thousand chunks for throughput, so the flag cost exactly
    the caller who could least afford it.

    With this, `render_many` is the ONE driver and the plan is what varies. The
    width is spent by the ENGINE either way — `v3_engine` pools threads,
    `mlx_backend` groups into a BatchGenerator — which is why the fix is here
    and not a thread pool in `serve/worker.py`: a pool there would be right for
    the served arm and wrong for MLX, and the caller cannot tell which engine it
    got.

    THE SURFACE IS `GuardPlan`'s, exactly, because `render_many` drives it and
    a second shape would mean a second driver. What every method does instead is
    the whole of the difference: nothing is re-rolled, nothing is split, nothing
    is measured against anything, and `verdict()` is None for every chunk —
    which is already the wire's contract for an unjudged row (a `guard` key is
    additive and optional; a row with no verdict carries none).
    """

    def __init__(self, *, sample_rate: int, base_seed: Optional[int]) -> None:
        self.sample_rate = int(sample_rate)
        self.base_seed = base_seed
        #: index -> the request issued for it, while it is in flight.
        self._waiting: dict = {}
        #: index -> text, for chunks that have not been issued yet.
        self._queued: list = []
        self._finished: List[tuple] = []
        self._measures: dict = {}
        self._added = 0

    # -- building ------------------------------------------------------------
    def add(self, index: int, text: str, first_take=None,
            first_measure=None) -> None:
        """A chunk to render once. `first_take` is take 0 when the caller
        already has it, exactly as on `GuardPlan` — a slab arm that decoded its
        own rows hands them here and nothing is rendered twice."""
        if first_take is not None:
            self._measures[int(index)] = first_measure
            self._finished.append((int(index), first_take, True))
            return
        self._queued.append((self._added, int(index), text))
        self._added += 1

    # -- the driver's surface ------------------------------------------------
    def next_request(self) -> Optional[RenderRequest]:
        """One render, in the order the chunks were added.

        `seed=None` is the engine's own seed rule for `index` — take 0 of that
        chunk in its own lane. There is no other rung to ask for: a re-roll is a
        judgment, and this plan makes none.
        """
        if not self._queued:
            return None
        order, index, text = self._queued.pop(0)
        request = RenderRequest(path=(order,), index=index, text=text,
                                seed=None, rung='take', depth=0)
        self._waiting[request.path] = request
        return request

    def offer(self, request: RenderRequest, audio, measure=None) -> None:
        """The audio for one request. It ships: there is nothing to decide.

        `clean` is True on the tuple because the row is exactly what was asked
        for and nothing found anything wrong with it — nothing looked. It is not
        a verdict and `verdict()` still answers None.
        """
        self._waiting.pop(request.path, None)
        self._measures[request.index] = measure
        self._finished.append((request.index, audio, True))

    def abandon(self, request: RenderRequest) -> int:
        """The driver could not render this request; the chunk leaves.

        `GuardPlan`'s contract, kept: the index comes back for the driver to
        report as that row's failure, and it never reaches `finished()`.
        """
        self._waiting.pop(request.path, None)
        self._measures.pop(request.index, None)
        return request.index

    def finished(self) -> List[tuple]:
        out, self._finished = self._finished, []
        return out

    def verdict(self, index: int) -> Optional[dict]:
        """None, always. Nothing judged this chunk, so there is no conclusion to
        report — and a verdict here would be narrator claiming to have judged a
        row it was told not to judge."""
        return None

    def measure(self, index: int) -> Optional[FrameMeasure]:
        """What the shipped take SPENT. A measurement, not a judgment: it is
        taken whether or not anything judged the chunk, which is the whole
        reason a screening render is worth running."""
        return self._measures.pop(int(index), None)

    @property
    def pending(self) -> int:
        return len(self._queued) + len(self._waiting)


def render_guarded(render: Callable[[str, Optional[int]], np.ndarray],
                   text: str, index: int, *, sample_rate: int,
                   base_seed: Optional[int],
                   max_chars_per_sec: float = 0.0,
                   min_chars_per_sec: float = 0.0,
                   tracker: Optional[PaceTracker] = None,
                   first_take: Optional[np.ndarray] = None,
                   on_event: Callable[[dict], None] = emit_event) -> np.ndarray:
    """The ladder, ONE chunk, driven serially. `render(text, seed)` is the
    engine's own single render (seed None = the engine's seed rule for `index`);
    `first_take` is take 0 when the caller already has it, so it is never
    rendered twice. Returns the audio to ship.

    THE BAND is the `tracker`'s when one is given - read fresh for every take,
    so a re-roll is judged against the same book pace the take before it was -
    and the fixed `max_chars_per_sec` / `min_chars_per_sec` pair otherwise (the
    tests, and any caller that wants a band that does not move). A chunk under
    `MIN_GUARD_CHARS` is judged on the short side only, either way.

    Taking `round()[0]` every time is what makes this DEPTH-FIRST: a split's
    left half is finished before its right half is begun, which is the order
    the ladder has always rendered and logged in. `GuardPlan` is the same
    policy with every waiting chunk rendered together instead.
    """
    plan = GuardPlan(sample_rate=sample_rate, base_seed=base_seed, tracker=tracker,
                     max_chars_per_sec=max_chars_per_sec,
                     min_chars_per_sec=min_chars_per_sec, on_event=on_event)
    plan.add(index, text, first_take=first_take)
    while True:
        request = plan.next_request()
        if request is None:
            break
        plan.offer(request, render(request.text, request.seed))
    decided = plan.finished()
    if len(decided) != 1:
        raise RuntimeError(
            f'render_guarded: the ladder decided {len(decided)} chunks for one call.')
    return decided[0][1]
