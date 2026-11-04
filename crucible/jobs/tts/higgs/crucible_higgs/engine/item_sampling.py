"""THE PER-ITEM TAKE CHANNEL - one spelling, one validator, one set of refusal
names. A rung is TWO facts, `sampling` and `take`, and they travel together.

WHY THIS MODULE EXISTS (Owen, 2026-09-14, docs/EXTENSION-TO-CRUCIBLE-PLAN.md
section 2): *a retake must not reuse the settings that produced the problem;
the spread IS the take ladder.* Crucible's take ladder
(crucible/docs/PHASE3-TTS.md section 3) is a per-VOICE list of rungs, take 0
being the boson default 0.8 / 0.95 / 50 and take N > 0 a measured alternative
with a written reason. A `tts` job carries `take: N` and nothing else about
sampling: Crucible resolves the rung and sends the NUMBERS; narrator renders
that chunk under them.

Until now it could not. narrator's sampling arrived through the voices
document (`NARRATOR_HIGGS_VOICES`), which is written per LOAD, and
`generate` / `generate_batch` took no sampling at all - so Crucible refused
every rung above 0 by name (`sampling_not_wired`, PHASE3-TTS section 4: "a
rung is per RENDER and narrator's `generate_batch` takes no sampling, so a
ladder has no channel yet"). A reload per take is not a ladder. This is the
channel.

THE SPELLING IS THE VOICES DOCUMENT'S, DELIBERATELY. The document's `sampling`
block is `{temperature, topP, topK}` (+ `repetitionPenalty` on the served arm)
- `engine/higgs/config.py:_SAMPLING_KEYS`, and PHASE3-TTS section 4's table
spells it the same way for the `[voice.backends.<arm>].sampling` it is written
from. A second spelling on the per-item wire would be two names for one fact,
which is the exact shape crucible/docs/ARCHITECTURE.md's audit found seven
times. So the wire key is the document key, and `WIRE_KEYS` below is the one
translation to the engines' snake_case.

A RUNG IS AN OVERLAY, NOT A REPLACEMENT, and that is measured rather than
tidy. `[[voice.takes]]` take 1 in PHASE3-TTS is one line, `temperature = 0.7`;
it says nothing about top_p or top_k because it means "take 0, but cooler". If
a partial rung REPLACED the voice's sampling, that rung would render with
top_k unset - and on SGLang-Omni an unset top_k is the untruncated 1026-way
codebook tail, measured 2026-09-05 as one chunk running to the cap with 80 s
of silence (`engine/higgs/sgl_served.py:_WHY_SAMPLING_IS_REQUIRED`). So each
engine applies an item's keys OVER its resolved sampling, key by key, and a
key the rung omits keeps take 0's value. `apply_over` is the one place that
happens.

THE TWO REFUSALS, both by name and neither ever a clamp:

  sampling_malformed      the item's `sampling` is not an object, is empty,
                          names a key that is not a lever, or carries a value
                          that is not a positive number (top_k: not a whole
                          one). The field is named in the message.
  sampling_not_supported  the value is well formed and this engine cannot
                          honour it - the lever does not exist on this backend
                          (mlx-audio has no repetition penalty), or the engine
                          has no per-item sampling at all (Orpheus, deprecated).

ABSENT IS NOT A REFUSAL AND NOT A FALLBACK. An item with no `sampling` key
renders at the voice's loaded default, which is take 0. That is the documented
meaning of "no rung", not a value substituted for a missing one.

A SEED IS A SETTING, WHICH IS WHY `take` IS ALSO ON THE WIRE (2026-09-15).
The paragraph above was only half of a rung. narrator seeds chunk i at
`config.seed + i` on both Higgs arms, and the take ladder never varied it, so
take 0 and take 7 of one chunk were BYTE-IDENTICAL renders whenever their
sampling matched - and two take-0 re-rolls always were, because take 0's rung
is by definition the voice's own numbers. Owen's ruling of 2026-09-14 - *"if a
sentence/chunk was problematic before, it'll likely be problematic again with
the same settings"* - makes that a defect: a retake must not reuse the settings
that produced the problem, and the seed is one of them.

So an item carries `take: N` (a whole number, absent = 0) BESIDE `sampling`,
and the engine renders the row in take N's own seed lane
(`engine/higgs/truncation.py:in_take_lane`, `TAKE_SEED_STRIDE`). The two facts
are INDEPENDENT on this wire: `take: 3` with no `sampling` at all is a legal
item and renders a draw take 0 never made, which is the whole point and is
something no sampling could express.

(Crucible's own ladder document still refuses a rung above 0 that declares no
numbers - `voices.py:_check_takes`: "a rung that is the same sampling as the
one below it is a different DRAW, which is what a re-roll is for". That rule
was written when narrator had no take seed and a different draw was the one
thing a rung could NOT ask for. This channel makes such a rung expressible;
whether to allow it is Crucible's ruling, not narrator's.)

  take_malformed      the item's `take` is not a whole number >= 0, or is
                      beyond `MAX_TAKE`. Refused, never rounded or clamped.
  take_not_supported  well formed, above 0, and this engine has no take lane
                      at all (Orpheus, deprecated) - so it would render take 0
                      and report take N.
"""

#: The wire's lever names (the voices document's camelCase) -> the engines'.
#:
#: Held EQUAL to `engine/higgs/config.py:_SAMPLING_KEYS` by
#: tests/test_serve_sampling.py, so the per-load channel and the per-item
#: channel cannot drift into two vocabularies for one set of levers.
WIRE_KEYS = {'temperature': 'temperature', 'topP': 'top_p', 'topK': 'top_k',
             'repetitionPenalty': 'repetition_penalty'}

#: The refusal names, as they appear at the head of every message.
MALFORMED = 'sampling_malformed'
NOT_SUPPORTED = 'sampling_not_supported'
TAKE_MALFORMED = 'take_malformed'
TAKE_NOT_SUPPORTED = 'take_not_supported'

#: The highest `take` this wire accepts.
#:
#: A LADDER IS A HANDFUL OF RUNGS - PHASE3-TTS's `[[voice.takes]]` is a short
#: list per voice - so a take in the thousands is a caller error and is refused
#: by name rather than turned into a seed. The number is not arbitrary: the
#: take's seed lane is `TAKE_SEED_STRIDE` (1,600,048) wide, so 999 puts the
#: highest seed narrator can draw at about 1.6e9, inside a signed 32-bit int,
#: which is what `build_request_body` puts on the wire as the request's `seed`.
#: A take of 2,000 would silently exceed that on some servers; refusing is the
#: honest answer, and `engine/higgs/truncation.py:TAKE_SEED_STRIDE` states the
#: arithmetic.
MAX_TAKE = 999


class SamplingMalformed(ValueError):
    """The item's `sampling` is not a sampling. Message starts `sampling_malformed:`."""


class SamplingNotSupported(ValueError):
    """Well formed, and this engine cannot honour it. Message starts
    `sampling_not_supported:`."""


class TakeMalformed(ValueError):
    """The item's `take` is not a take. Message starts `take_malformed:`."""


class TakeNotSupported(ValueError):
    """Well formed, above 0, and this engine has no take lane. Message starts
    `take_not_supported:`."""


def parse_item_sampling(raw, where: str, levers=None) -> dict:
    """One item's `sampling` off the wire -> the engine's snake_case dict, or
    None when the item carried none.

    `where` names the caller in every message (`"HiggsV3Engine item 412"`), so
    a refusal read off a worker's stderr or a `batch_item` message says which
    row and which engine refused.

    `levers`, when given, is the subset of `WIRE_KEYS` this engine can actually
    honour; a lever outside it is `sampling_not_supported` rather than
    `sampling_malformed`, because the difference matters to whoever sent it -
    one is a typo and the other is the wrong backend.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw:
        raise SamplingMalformed(
            f'{MALFORMED}: {where} carries sampling {raw!r}; a rung is a non-empty '
            f'object with any of {", ".join(sorted(WIRE_KEYS))}.')
    allowed = set(WIRE_KEYS) if levers is None else set(levers)
    unknown = sorted(set(raw) - set(WIRE_KEYS))
    if unknown:
        raise SamplingMalformed(
            f'{MALFORMED}: {where} carries sampling key(s) {unknown}; the levers are '
            f'{sorted(WIRE_KEYS)}.')
    unsupported = sorted(set(raw) - allowed)
    if unsupported:
        raise SamplingNotSupported(
            f'{NOT_SUPPORTED}: {where} asks for sampling {unsupported}, which this '
            f'engine has no lever for; it honours {sorted(allowed)}.')
    out = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise SamplingMalformed(
                f'{MALFORMED}: {where} carries sampling {key}={value!r}, which is not '
                'a positive number.')
        if key == 'topK' and int(value) != value:
            raise SamplingMalformed(
                f'{MALFORMED}: {where} carries sampling topK={value!r}; top_k is a '
                'whole number of candidates.')
        out[WIRE_KEYS[key]] = int(value) if key == 'topK' else float(value)
    return out


def parse_item_take(raw, where: str) -> int:
    """One item's `take` off the wire -> a whole number >= 0. ABSENT IS 0.

    `where` names the caller in every message, exactly as
    `parse_item_sampling`'s does, so a refusal read off a worker's stderr says
    which row asked for what.

    0 IS NOT A FALLBACK AND ABSENT IS NOT A REFUSAL. Take 0 is the documented
    bottom rung of the ladder - the voice's own numbers in the seed lane
    narrator has always used - so an item that says nothing about takes renders
    exactly as every item rendered before this channel existed. That is the
    additive guarantee, not a value substituted for a missing one.

    A BOOL IS NOT A TAKE. `isinstance(True, int)` is True in Python, and
    `take: true` from a JSON client that meant `stream: true` would otherwise
    render take 1 and be reported as take 1. It is a malformed take.

    A FLOAT THAT HAPPENS TO BE WHOLE IS STILL REFUSED, for `topK`'s reason: a
    take indexes a rung of a list, a fractional one indexes nothing, and
    `2.0` arriving where `2` was meant is a client whose types are wrong -
    which is worth seeing once rather than rounding away forever.
    """
    if raw is None:
        return 0
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TakeMalformed(
            f'{TAKE_MALFORMED}: {where} carries take {raw!r} '
            f'({type(raw).__name__}); a take is a whole number >= 0 naming a rung '
            'of the voice\'s ladder, and 0 (or no key at all) is take 0.')
    if raw < 0:
        raise TakeMalformed(
            f'{TAKE_MALFORMED}: {where} carries take {raw!r}. A take names a rung '
            'of the ladder and counts up from 0; there is no rung below take 0.')
    if raw > MAX_TAKE:
        raise TakeMalformed(
            f'{TAKE_MALFORMED}: {where} carries take {raw!r}, above MAX_TAKE '
            f'({MAX_TAKE}). A ladder is a handful of rungs, and a take this high '
            'would seed the render outside the 32-bit range the servers take - '
            'see engine/higgs/truncation.py:TAKE_SEED_STRIDE.')
    return int(raw)


def refuse_item_take(raw, where: str, why: str) -> int:
    """An engine with NO take lane at all. Take 0 (and an absent `take`) pass;
    anything above 0 is `take_not_supported`, naming why.

    Orpheus is the only caller, for `refuse_item_sampling`'s reasons: it is
    deprecated, it is not built into Crucible, and it dies with the legacy
    layer. The refusal is what stops a retake rendering at the very seed the
    retake exists to leave and being reported as take N.
    """
    take = parse_item_take(raw, where)
    if take == 0:
        return 0
    raise TakeNotSupported(f'{TAKE_NOT_SUPPORTED}: {where} asks for take {take}. '
                           f'{why}')


def takes_aligned(takes, count: int, where: str) -> list:
    """A `takes` argument aligned to a batch -> a list of `count` whole
    numbers.

    None for the whole argument means "take 0 everywhere", which is every
    caller not climbing a ladder, and is the same documented shape `aligned`
    gives a batch with no rung. A list of the WRONG LENGTH is refused by name:
    a misaligned list would render row i in row j's seed lane and report
    neither.
    """
    if takes is None:
        return [0] * count
    takes = list(takes)
    if len(takes) != count:
        raise TakeMalformed(
            f'{TAKE_MALFORMED}: {where}: {len(takes)} take(s) for {count} rows; '
            'takes must be aligned to the batch or None.')
    return takes


def refuse_item_sampling(raw, where: str, why: str):
    """An engine with NO per-item sampling channel at all. None passes (absent
    is not a rung); anything else is `sampling_not_supported`, naming why.

    Orpheus is the only caller: it is deprecated (Owen, 2026-09-14 - "orpheus
    is deprecated too but hasnt been removed yet. higgs is the frontier"), it
    is not built into Crucible, and it dies with the legacy layer. IGNORING a
    rung there would render the retake at the very settings the retake exists
    to avoid and report it as a take at the asked-for numbers.
    """
    if raw is None:
        return None
    raise SamplingNotSupported(f'{NOT_SUPPORTED}: {where} carries sampling '
                               f'{raw!r}. {why}')


def aligned(samplings, count: int, where: str) -> list:
    """A `samplings` argument aligned to a batch -> a list of `count` rungs,
    each a parsed dict or None.

    None for the whole argument means "no rung anywhere", which is every
    caller that is not climbing a ladder, and is NOT a fallback: it is the
    documented shape of a batch with no take above 0. A list of the WRONG
    LENGTH is refused by name, exactly as `voices` is - a misaligned list
    would render row i at row j's numbers and report neither.
    """
    if samplings is None:
        return [None] * count
    samplings = list(samplings)
    if len(samplings) != count:
        raise SamplingMalformed(
            f'{MALFORMED}: {where}: {len(samplings)} sampling(s) for {count} rows; '
            'samplings must be aligned to the batch or None.')
    return samplings


def group_key(sampling):
    """A hashable, order-independent key for one rung, so a backend that
    cannot MIX sampling inside one batch can split the batch by group instead
    of rendering someone at the wrong numbers (the MLX slab does exactly
    this). None keys itself."""
    if sampling is None:
        return None
    return tuple(sorted((k, float(v)) for k, v in sampling.items()))


def apply_over(base, item) -> dict:
    """The item's rung laid OVER the engine's resolved sampling - see the
    module docstring for why it is an overlay and not a replacement.

    `base` is what this engine would have rendered the chunk at (take 0);
    `item` is the parsed rung or None. Returns a NEW dict; neither argument is
    mutated, because `base` is usually the engine's own long-lived sampling.
    """
    resolved = dict(base or {})
    if item:
        resolved.update(item)
    return resolved
