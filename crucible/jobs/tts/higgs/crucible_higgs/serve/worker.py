#!/usr/bin/env python3
"""
Crucible's Higgs v3 narration worker: `python -m crucible_higgs.serve`.

Moved from narrator@dc05e7e7 `serve/worker.py` (BookForge's `narrator.serve`,
which `crucible/engines/narrator.py` ran until this move). The JSON-lines
protocol, every message type and field, the stdout lock, the reader thread, the
warmup shapes and the env vars the engine adapter passes are UNCHANGED - Crucible
parses this wire format (`crucible/engines/narrator.py`, `jobs/tts/`). What is
gone is everything Orpheus: its three render arms (vLLM, MLX grouping,
transformers), its voice kinds and caps, the ORPHEUS_STREAM_* knobs and the
`orpheus` default for NARRATOR_ENGINE. Higgs v3 (`engine/registry.py`) is the
one engine, and NARRATOR_ENGINE must name it.

It loads the model ONCE and serves chunk requests over stdin, emitting base64
PCM16 (24kHz mono) JSON lines on stdout.

Protocol (one JSON object per line):
  stdin:  {action: 'load', voice, id?, modelDir?, adapterDir?, baseDir?, caps?,
                            warm?: bool}   # warm=False: skip the first-load warmup
                                           # because a speak is already waiting
          {action: 'generate', text, voice?, stream?: bool, sampling?, take?, ...}
          {action: 'generate_batch',
           items: [{i, text, voice?, stream?: bool, sampling?, take?}], ...}
                 # A RUNG IS TWO FACTS, and both ride on the item.
                 #
                 # `sampling` is ONE ITEM's rung NUMBERS:
                 #   {temperature?, topP?, topK?, repetitionPenalty?}
                 # spelled exactly as the NARRATOR_HIGGS_VOICES document
                 # spells it. ABSENT = the voice's loaded sampling, which is
                 # take 0. PRESENT = that item renders under those numbers,
                 # laid OVER take 0 key by key; a batch may mix rungs.
                 # Refused by name, per item: `sampling_malformed` (not a
                 # sampling) and `sampling_not_supported` (this engine or this
                 # backend has no such lever - Orpheus has none at all).
                 #
                 # `take` is WHICH rung, a whole number >= 0, ABSENT = 0, and
                 # it moves the item's SEED into that take's own lane
                 # (engine/higgs/truncation.py:in_take_lane). It is
                 # INDEPENDENT of `sampling`: take N whose rung declares no
                 # override is still a different draw. Refused per item as
                 # `take_malformed` (not a whole number >= 0, or past
                 # MAX_TAKE) and `take_not_supported` (this engine has no
                 # lane - Orpheus).
                 # See engine/item_sampling.py.
          {action: 'cancel' | 'stop' | 'quit'}
                 # 'cancel'/'stop' ABORT a running generate_batch - see the reader
                 # thread in run(). Un-rendered rows come back as ordinary per-item
                 # failures with message 'cancelled', then 'batch_done' as always.
  stdout: {type: 'ready', device, backend?, itemTake: true}
                 # `itemTake` says THIS BUILD parses a per-item rung - BOTH
                 # halves, `sampling` and `take` - on a generate_batch item
                 # (below). A narrator without the key has no such channel and
                 # would drop a rung in silence, so a client driving a take
                 # ladder must refuse rather than render take 0 and call it
                 # take N. ONE key for the two facts because they landed as one
                 # channel and an old build has neither; two keys would be two
                 # owners of one answer. Engine-level support is a separate and
                 # later fact: `sampling_not_supported` / `take_not_supported`,
                 # per row.
          {type: 'status' | 'loaded' | 'error' | 'stopped', ...}
          {type: 'audio', format:'pcm16', data, duration, sampleRate, gapSec}  # batch
          {type: 'chunk', seq, format:'pcm16', data, duration, sampleRate}   # stream
          {type: 'done', duration, chunks, cancelled, gapSec}                # stream end
          {type: 'batch_chunk', i, seq, format:'pcm16', data, duration, sampleRate}
          {type: 'batch_item', i, format:'pcm16', data, duration, sampleRate,
                               gapSec?, guard?: {...}}                       # batch
          {type: 'batch_item', i, streamed: true, duration, chunks, gapSec}  # fast start

`gapSec` IS THE LISTEN DOOR'S GAP, and it rides the row's TERMINAL record only
(Owen, 2026-09-18: "yes, it paces like the book... maybe the browser extension
should handle the gaps for itself"). The audio on this wire is BARE SPEECH; the
silence between two rows is `text/gaps.classify_gap`'s answer for the row - the
same call `text/prep.py` makes to write a book's `gaps.json` - and the PLAYER
inserts it, because on a stream there is no assembler but the player.

NOT ON `chunk`/`batch_chunk`, and that is the design rather than an omission. The
gap follows the row's LAST sample, so nothing can be realized from it until the
row is over; a terminal record is emitted exactly once per row (this worker's own
guarantee) while a row has N chunks, and N copies of one number is the
two-owner shape this change removed. A fast-start player receives the terminal
record immediately after the last chunk - while that chunk is still playing -
so it knows the gap before it needs it.

A RENDER row carries NO `gapSec`: an assembler follows that door and realizes
`gaps.json`, and two statements of one silence is two owners of it.

THE PER-ITEM SAMPLING CHANNEL (Owen, 2026-09-14, the retake spread ruling in
docs/history/EXTENSION-TO-CRUCIBLE-PLAN.md section 2: *a retake must not reuse the
settings that produced the problem; the spread IS the take ladder*). Crucible
resolves a job's `take: N` against the voice's `[[voice.takes]]` ladder and
sends the NUMBERS on the item. Until this existed there was no channel for
them - the voices document is written per LOAD and a reload per take is not a
ladder - and Crucible refused every rung above 0 by name
(`sampling_not_wired`, crucible/docs/PHASE3-TTS.md section 4). A temperature
never reaches this wire from the APP: it is engine config, resolved by the
server, and what the app asks for is a take.

`guard` (Owen's ruling of 2026-09-13, crucible/docs/PHASE6-REMOTE-RENDER.md) is
the verdict the ENGINE'S OWN retake ladder reached for that chunk - `verdict`,
`clean`, `parts`, `band` and the take records - present only on a row an engine
that guards its own batch rendered. It is ADDITIVE and OPTIONAL: a row with no
verdict has no `guard` key, so orpheus-worker-pool.ts and the browser extension
read the field set they always read.

FAST START (Owen's ruling of 2026-09-04). An item carrying `stream: true` is a row
whose audio must reach the client WHILE IT IS STILL GENERATING, in sub-sentence
chunks, so the browser extension can begin playing in about a second instead of
waiting out a whole batch. That is the ONLY thing it changes: the batch is the same
width, scheduled the same way, on the same engine. A generate_batch with no
`stream` flag anywhere takes the pre-existing code path, byte for byte.

A voice is a NAME in the NARRATOR_HIGGS_VOICES document, resolved by the engine
(`resolve_load_voice`). It is servable per request only after a 'load' registered
it against the CURRENT engine - a generate naming anything else is an error,
never a silent render in whatever is loaded.
"""

import base64
import json
import os
import queue
import re
import signal
import sys
import threading

import numpy as np

# THE GAP RULE, from the layer that owns it. `text/gaps` is pure stdlib (os, re)
# and `crucible_higgs.text`'s package init imports nothing, so this pulls no
# engine into a module that must stay empty of one until `ready` has been sent.
from ..text.gaps import classify_gap_seconds

# THE INTERIOR-PAUSE CAP, numpy and the env reader only - no engine.
from .pause_cap import cap_interior_pauses, max_pause_seconds

DEFAULT_SAMPLERATE = 24000

#: THE WARMUP SHAPES `_warmup` renders at load, beside the single-sentence one.
#:
#: narrator read these from ORPHEUS_STREAM_RAMP and ORPHEUS_STREAM_BATCH (and
#: ORPHEUS_STREAM_WARM_MAX), knobs BookForge's Orpheus pool set and Crucible's
#: spawn never did - so every Higgs load Crucible has made warmed at narrator's
#: unset defaults, ramp 8 and full width 16. Those are the numbers here, as
#: constants: the knobs went with the Orpheus grouping they sized, and the warmup
#: a Crucible load pays is exactly what it paid before the move.
WARMUP_RAMP_WIDTH = 8
WARMUP_FULL_WIDTH = 16


def _graceful_exit(signum, frame):
    """Cooperative shutdown: SIGTERM/SIGINT -> SystemExit(143).

    Python's default SIGTERM disposition kills the process WITHOUT running atexit
    hooks, so the engine never releases the GPU - and force-killing a process stuck
    in a WSL dxg GPU wait is what kernel-wedges the whole WSL VM. Raising SystemExit
    instead unwinds the stdin loop, runs atexit (the engine's CUDA cleanup), and
    releases the GPU from inside the process. The stdin 'quit' action remains the
    primary teardown; this covers the pkill path.
    """
    print(f"[narrator.serve] Signal {signum} received - shutting down cleanly (releasing GPU)...",
          file=sys.stderr, flush=True)
    raise SystemExit(143)


signal.signal(signal.SIGTERM, _graceful_exit)
signal.signal(signal.SIGINT, _graceful_exit)


# ---- the fake-engine test door ----------------------------------------------
#
# WHY AN ARGV FLAG AND NOT AN ENVIRONMENT VARIABLE. This module is the
# PRODUCTION entry point, and both spawn arms hand the worker the parent
# environment: the native arm passes `...process.env`, and the WSL arm exports a
# forwarded set into a bash subshell. An env switch is therefore something a
# leaked variable in a developer shell, a stale `.env`, or a CI runner can turn
# on by accident - and the failure it produces is a whole book rendered as
# 220 Hz sine tones, reported as success, with nothing in the protocol saying
# so. An argv flag cannot leak: it is written by whoever wrote the command line.
#
# `argv` is otherwise unused by this worker (its whole interface is the
# JSON-lines protocol on stdin/stdout), so the flag costs nothing and an
# unrecognised argument is a hard error rather than a silent ignore.
_FAKE_ENGINE = False

FAKE_ENGINE_BANNER = """\
=================================================================
[narrator.serve] FAKE ENGINE ACTIVE (--fake-engine)
[narrator.serve] No model is loaded. Every sentence is a 220 Hz
[narrator.serve] sine wave sized from its text length. This is a
[narrator.serve] PROTOCOL TEST MODE - audio from it is not speech.
================================================================="""


def _fake_engine_enabled() -> bool:
    """True only when this process was started with --fake-engine."""
    return _FAKE_ENGINE


def _warn_fake_engine(where: str) -> None:
    """Re-state the banner on stderr wherever the protocol says "I am ready to
    render", so a mode this loud cannot be missed in a log.

    STDERR, not a new JSON field: the wire shape is a contract with
    electron/orpheus-worker-pool.ts and this port keeps it byte-identical. A
    reader of the log sees it; a parser sees nothing new.
    """
    if _FAKE_ENGINE:
        print(FAKE_ENGINE_BANNER, file=sys.stderr, flush=True)
        print(f'[narrator.serve] (at: {where})', file=sys.stderr, flush=True)


def engine_id() -> str:
    """Which engine this worker serves: NARRATOR_ENGINE, REQUIRED.

    narrator defaulted an unset one to 'orpheus'. There is no default here: the
    worker serves Higgs v3 and nothing else, `crucible/engines/narrator.py` sets
    the variable on every spawn, and an unset or unknown value is
    misconfiguration - refused by name at the first use (`main` turns it into a
    non-zero exit before any `ready`), never defaulted, because a book rendered
    by a different model than the one asked for is a silent failure.
    """
    from ..engine import registry
    engine = (os.environ.get('NARRATOR_ENGINE') or '').strip()
    if not engine:
        raise ValueError(
            'NARRATOR_ENGINE is not set. It names the engine this worker serves '
            f"({', '.join(registry.ids())}); crucible/engines/narrator.py sets it "
            'on every spawn, and there is no default to fall back to.')
    if engine not in registry.ids():
        raise ValueError(
            f'NARRATOR_ENGINE={engine}: this worker serves '
            f"{', '.join(registry.ids())} only. narrator's Orpheus and its Higgs v2 "
            'scaffold did not move to Crucible.')
    return engine


def _engine_class():
    """The engine this worker builds, for the id NARRATOR_ENGINE names.

    --fake-engine substitutes a model-free stand-in (serve.fake_engine), which
    produces deterministic sine-wave audio and needs no model, no GPU and no
    torch. It exists so the serve protocol tests can drive THIS file - the real
    reader thread, the real stdout lock, the real batch bookkeeping - and assert
    the exact message sequence Crucible reads. It is a TEST DOOR, never a
    fallback: nothing selects it implicitly.
    """
    if _fake_engine_enabled():
        from .fake_engine import fake_engine_class
        return fake_engine_class(engine_id())
    from ..engine import registry
    return registry.engine_class(engine_id())


def _engine_config(**kwargs):
    """The config object this engine takes, or the fake's stand-in for it.

    The engine's factory decides which of the load message's keywords it
    understands: Higgs refuses model_dir, base_dir and caps by name and
    resolves its voice - reference clips plus transcripts, or a merged
    checkpoint - from the NARRATOR_HIGGS_VOICES document.
    """
    if _fake_engine_enabled():
        from .fake_engine import fake_engine_config
        return fake_engine_config(engine_id(), **kwargs)
    from ..engine import registry
    return registry.engine_config(engine_id(), **kwargs)


# -- Text normalization (numbers/currency/years -> words) ----------------------
# The listen path hands raw page text straight to the model; the audiobook path
# normalizes upstream (BookForge's own model pass over the narration copy).
# Mirror the common cases here so "$5.50", "1995", "50%" read naturally.
# num2words is a DECLARED BASE DEPENDENCY (python/pyproject.toml), imported
# plainly. It used to sit behind `try/except -> _HAS_NUM2WORDS`, and that guard
# was a FALLBACK of the worst kind: with it, a machine missing the module read
# every "$5.50" out as punctuation and every "1995" digit by digit for a whole
# listening session, and said nothing. This path is NOT dead - `normalize_for_tts`
# runs on every `generate` and every streamed batch row, which is the browser
# extension's listen path. (The audiobook path normalizes upstream in BookForge
# and never arrives here.)
#
# A missing module is now an ImportError at worker start: loud, immediate, and
# fixed by installing the dependency the project already declares.
from num2words import num2words as _num2words


def _number_refusal(value, lang, exc):
    """Every num2words failure, refused by name.

    These four helpers each used to end in `except Exception: return str(n)` -
    the same silent passthrough as the import guard, one layer down. A number
    this function cannot say is a defect in the caller's regex or in the
    language, and the listener must not be the one to discover it.
    """
    return ValueError(
        f'num2words could not render {value!r} in {lang!r}: {exc}. The listen '
        'path normalizes numbers before the model sees them; handing the raw '
        'digits on would have the model read them as punctuation.')


def _to_words(n, lang):
    try:
        return _num2words(int(n), lang=lang)
    except Exception as exc:
        raise _number_refusal(n, lang, exc) from exc


def _num_phrase(token, lang):
    token = token.replace(',', '')
    try:
        if '.' in token:
            intpart, frac = token.split('.', 1)
            words = _num2words(int(intpart or '0'), lang=lang)
            digits = ' '.join(_num2words(int(d), lang=lang) for d in frac)
            return f"{words} point {digits}"
        return _num2words(int(token), lang=lang)
    except Exception as exc:
        raise _number_refusal(token, lang, exc) from exc


def _ordinal(n, lang):
    try:
        return _num2words(int(n), lang=lang, to='ordinal')
    except Exception as exc:
        raise _number_refusal(n, lang, exc) from exc


def _year_to_words(y, lang):
    if lang != 'en':
        return _to_words(y, lang)
    try:
        if 2000 <= y <= 2009:
            return f"two thousand {_to_words(y % 100, lang)}" if y % 100 else "two thousand"
        if 1100 <= y <= 1999 or 2010 <= y <= 2099:
            hi, lo = divmod(y, 100)
            if lo == 0:
                return f"{_to_words(hi, lang)} hundred"
            lo_words = _to_words(lo, lang) if lo >= 10 else f"oh {_to_words(lo, lang)}"
            return f"{_to_words(hi, lang)} {lo_words}"
        return _to_words(y, lang)
    except Exception as exc:
        raise _number_refusal(y, lang, exc) from exc


#: Languages num2words has already been asked about, so `check_language` costs
#: one call per language per process rather than one per sentence.
_LANGUAGE_VERDICTS = {}


def resolve_language(language) -> str:
    """The bare language subtag, or a refusal naming what arrived.

    `(language or 'en')` used to sit inside `normalize_for_tts` - a second
    default underneath the protocol's own `request.get('language', 'en')`, and a
    `x or default` on a value the caller is supposed to supply. The wire field
    is optional and its default is documented at the top of this module; a
    request that reaches here with a BLANK one is a caller bug, and narrating an
    unknown language as English is the silent substitution this file spent a
    commit removing.
    """
    lang = (language or '').split('-')[0].strip().lower()
    if not lang:
        raise ValueError(
            f'normalize_for_tts got language={language!r}. The protocol field is '
            "optional and defaults to 'en' at the request boundary; a blank one "
            'arriving here is a caller bug, and reading an unknown language as '
            'English is a silent substitution.')
    return lang


def check_language(language) -> str:
    """Ask num2words ONCE whether it can speak this language. Refuse by name.

    THE REASON THIS IS NOT PER SENTENCE. num2words raises `NotImplementedError`
    for a language it has no module for, and that is a property of the SESSION,
    not of the sentence: without this, every sentence containing a digit failed
    one at a time with the same message for a whole book. One refusal, before
    any audio is rendered, at the point where the language is actually known -
    the `generate` / `generate_batch` request that carries it.
    """
    lang = resolve_language(language)
    if lang not in _LANGUAGE_VERDICTS:
        try:
            _num2words(0, lang=lang)
            _LANGUAGE_VERDICTS[lang] = None
        except Exception as exc:
            _LANGUAGE_VERDICTS[lang] = (
                f'num2words cannot speak {lang!r} ({exc}). The listen path '
                'normalizes numbers before the model sees them, so this whole '
                'request is refused rather than every sentence with a digit in '
                'it failing separately for the same reason.')
    verdict = _LANGUAGE_VERDICTS[lang]
    if verdict:
        raise ValueError(verdict)
    return lang


#: Foundry's ELLIPSIS_RUN (src/clean/tts-punctuation.ts): three or more periods, spaces or tabs between them allowed.
#: Two periods are never an ellipsis.
_ELLIPSIS_RUN = re.compile(r'\.[ \t]*\.[ \t]*\.(?:[ \t]*\.)*')

#: The em dash the Higgs corpora print - unspaced, "word—word" (tc_clean 79 of 79, hoa 1,096 of 1,117).
EM_DASH = '—'

#: A HYPHEN TYPED WHERE AN EM DASH BELONGS, in the shapes that can be nothing else (Owen, 2026-10-02: "emdashes
#: dramatically affect prosody ... where we can, we can do this deterministically. where it isnt trustworthy to do it
#: deterministically, we should hand the rule to the model"). Each is a sentence breaking off or a speaker cut off:
#:   a hyphen between a letter and a closing double quote      Besides-"      (speech broken off)
#:   a hyphen between an opening double quote and a letter     "-we're        (speech resumed)
#:   a hyphen between a letter and ? or !                      What the-?
#:   a hyphen after a letter at the very end of the text       ... but-
#:   a hyphen before a pronoun contraction                     button-he's    (no compound ends in "he's")
#: Every other unspaced hyphen - "wrong-if", "fifth-or sixth-time", "up-to-date", "know-it-all" - needs judgement
#: and is the cleanup MODEL's (Foundry clean-text's dash rule). Single quotes are left alone here: "rock-'n'-roll".
_DASH_SHAPES = (
    re.compile(r'(?<=\w)-(?=["”])'),
    re.compile(r'(?<=["“])-(?=\w)'),
    re.compile(r'(?<=\w)-(?=[?!])'),
    re.compile(r'(?<=[^\W\d])-\s*$'),
    re.compile(r"(?<=\w)-(?=(?:he|she|it|they|we|you|i|that|there|what|who)['’](?:s|re|m|ll|d|ve)\b)", re.I),
)


def restore_dashes(text: str) -> tuple[str, int]:
    """The text with every hyphen in a `_DASH_SHAPES` shape made an em dash, and how many were."""
    count = 0
    for shape in _DASH_SHAPES:
        text, n = shape.subn(EM_DASH, text)
        count += n
    return text, count


def normalize_for_tts(text, language):
    if not text:
        return text
    lang = resolve_language(language)
    # THE ELLIPSIS THE VOICES WERE TRAINED ON (Owen, 2026-10-02: "its pretty clear we need to make sure elipses are
    # printed correctly at a bare minimum"). Mutineer's Moon prints ". . ." 173 times, and the Higgs corpora hold
    # none (tr_v4nb 0 vs "..." 287, mb_v16 0 vs 353, ds_v14 0 vs 121): on "as if waiting to receive . . . or
    # transmit" the Third Reich voice invented words and repeated itself. Every printed form - spaced, four dots,
    # U+2026 - becomes exactly "...", the form the corpora use. Foundry's Clean text does the same to the book file
    # (src/clean/tts-punctuation.ts ELLIPSIS_RUN, the same pattern); this is the engine's input, so a book rendered
    # without that pass is covered too, and a cleaned one passes through unchanged.
    s = _ELLIPSIS_RUN.sub('...', text.replace('…', '...'))
    # AND THE DASHES THAT CAN BE NOTHING ELSE (`_DASH_SHAPES`). Every repair is logged with its sentence, so a
    # book's dashes can be reviewed from the engine log.
    s, dashes = restore_dashes(s)
    if dashes:
        print(f'[narrator.serve] DASH restored x{dashes}: {s[:160]!r}', file=sys.stderr, flush=True)

    def _money(m):
        whole = m.group(1).replace(',', '')
        cents = m.group(2)
        try:
            dollars = int(whole)
            out = f"{_to_words(dollars, lang)} dollar" + ('' if dollars == 1 else 's')
            if cents:
                c = int(cents.ljust(2, '0')[:2])
                if c:
                    out += f" and {_to_words(c, lang)} cent" + ('' if c == 1 else 's')
            return out
        except Exception as exc:
            # The FIFTH site of the same defect the reviewer named three of.
            # `return m.group(0)` hands "$5.50" back verbatim for the model to
            # read as punctuation - silently, for every price in the book.
            raise _number_refusal(m.group(0), lang, exc) from exc
    s = re.sub(r'\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?', _money, s)
    s = re.sub(r'(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?%',
               lambda m: f"{_num_phrase(m.group(1), lang)} percent", s)
    s = re.sub(r'\b(\d+)(?:st|nd|rd|th)\b', lambda m: _ordinal(m.group(1), lang), s)
    s = re.sub(r'(?<![\d,.])(1[1-9]\d{2}|20\d{2})(?![\d,.])',
               lambda m: _year_to_words(int(m.group(1)), lang), s)
    s = re.sub(r'\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?',
               lambda m: _num_phrase(m.group(0), lang), s)
    return s


# -- stdout protocol -----------------------------------------------------------
# One lock around the ONE print. Every message used to be written from the main
# thread (see _read_stdin's note: the reader thread only ever sets a flag), so a
# JSON line could not be interleaved with another. Fast start breaks that
# assumption on MLX: the engine decodes streamed frames on its own decoder thread
# and invokes on_chunk from there, so two threads can reach this function at once
# and `print` is not atomic - a torn line is an unparseable line, and the pool logs
# a JSON parse error and drops it.
#
# A lock rather than a queue-to-the-main-thread hop because the main thread is
# BLOCKED inside generate_batch_stream for the whole batch and has nothing to
# drain with. The ordering guarantees the protocol actually relies on survive it:
# a row's chunks carry `seq` and the client assembles by (i, seq), and 'batch_done'
# is still written by the main thread after generate_batch_stream has returned and
# every callback with it. Uncontended on the classic path, so switch-ON pays
# nothing for it.
_stdout_lock = threading.Lock()


def send_response(response_type: str, data: dict = None):
    msg = {'type': response_type}
    if data:
        msg.update(data)
    line = json.dumps(msg)
    with _stdout_lock:
        print(line, flush=True)


def audio_to_pcm16_base64(audio_array) -> str:
    a = np.asarray(audio_array, dtype=np.float32)
    a = np.clip(a, -1.0, 1.0)
    return base64.b64encode((a * 32767).astype(np.int16).tobytes()).decode('utf-8')


def row_gap_sec(text) -> float:
    """The silence that belongs AFTER this Listen row, in seconds.

    IT PACES LIKE THE BOOK BECAUSE IT IS THE BOOK'S OWN FUNCTION (Owen,
    2026-09-18: "yes, it paces like the book... maybe the browser extension
    should handle the gaps for itself"). `text/gaps.classify_gap` is what
    `text/prep.py` calls to write `gaps.json`, so a sentence heard on Listen
    gets exactly the number it would get inside an audiobook - the 0.6 s floor,
    the voice's inject when `NARRATOR_SENTENCE_GAP` carries one (the same
    variable the prep door sets from the catalog's `chunkGap.injectS`), or an
    explicit `[pause:X]`. This worker states it; the PLAYER realizes it, because
    on the stream there is no assembler but the player.

    THE TEXT IS THE ROW AS IT ARRIVED, before `normalize_for_tts`: that
    normalizer rewrites every number it sees, so `[pause:2.5]` reaches it as
    "[pause:two point five]" and classifies as an ordinary sentence.

    A LEAD GAP IS REFUSED RATHER THAN DROPPED. `classify_gap` returns a non-zero
    lead only for a chunk that STARTS with an explicit `[pause:X]` (a heading
    returns `(0.0, floor)` like everything else - the per-kind tiers were removed
    on 2026-07-17). A lead has to be realized before the row's FIRST sample, and
    a fast-start row's first sample is played long before any terminal record
    reaches the player, so this door has no honest channel for one. Half of
    `classify_gap`'s answer, delivered in silence, is the shape this whole change
    removes.
    """
    lead, trail = classify_gap_seconds(text or '')
    if lead:
        raise ValueError(
            f'this row classifies to a {lead:.3f}s LEAD gap - it opens with an '
            'explicit [pause:X] - and the Listen door carries no lead: the '
            'player learns a row\'s gap when the row retires, which is after its '
            'first sample has already been played. Render it through the book '
            'path, where the assembler realizes gaps.json\'s `before`.')
    return trail


#: The sample rate the LOADED engine produces. DEFAULT_SAMPLERATE until one is
#: loaded. Every duration and every `sampleRate` on the wire reads this, because
#: the wire describes the audio that was actually rendered - not a constant. Both
#: shipping engines are 24 kHz today, so this changes no byte on either path; it
#: is what stops the next engine from mis-timing every cue in a session.
_ACTIVE_SAMPLERATE = DEFAULT_SAMPLERATE
#: Whether the loaded engine BAKES its own silence into a clip (`Engine.pads`).
#: Higgs does not; True is what narrator held before a load, and no render
#: happens before one. Read by finalize_audio - see there.
_ACTIVE_PADS = True


def _can_guard_its_own_batch(engine) -> bool:
    """True when this engine OFFERS the guarded batched driver, `render_many`.

    A CAPABILITY, NOT A CHOICE, since 2026-09-19. It used to be both: an engine
    that could guard was an engine that DID, on every batch, and this predicate
    picked the arm. Owen ruled that the caller decides instead - a batch carries
    `retake`, and `retake: false` renders every row exactly once as it was sent,
    judged by nothing (crucible/docs/PHASE18-UNCERTIFIED.md). The fine-tuning
    ladder is why: screening a checkpoint has no measured pace and no measured
    cap by definition, because MEASURING THEM IS WHAT THE RENDER IS FOR, so a
    guard there judges a model against a band belonging to some other model.
    What is left here is the honest question - can this engine guard if asked -
    and it answers `retake_unsupported` rather than quietly rendering a batch
    that asked to be judged without judging it.

    THE DISCRIMINATOR IS THE CAPABILITY AND NOT THE BACKEND NAME: `backend` is
    a RUNTIME name ('mlx', 'sglang-omni', 'vllm-omni'), so routing on it selects
    a method the object may not have. `render_many` is the method being called;
    asking for it by name is the only test that cannot be wrong.

    WHY THE GUARDED ARM EXISTS AT ALL (Owen, 2026-09-13,
    crucible/docs/PHASE6-REMOTE-RENDER.md sections 0 and 2). narrator had two
    rendering worlds and only the audiobook one was guarded: the serve world
    reached a Higgs engine through a bare `render_audio()` per sentence, with
    no PaceTracker, no re-roll, no split ladder. The model and its inference
    own the guard, so the guard must run wherever the model does; `render_many`
    is `convert_many`'s driver with the file-writing sink removed, so the serve
    world can have the ladder without acquiring a `sentences_dir`.
    """
    return callable(getattr(engine, 'render_many', None))


def _server_log_of(engine):
    """Where this engine's SERVER writes, or None - reported in `loaded`.

    Only a 'served' backend has one (Higgs v3 under vllm-omni); every in-process
    engine answers None, because its logs are this process's own. It is on the
    wire because the log is EVIDENCE, not a debugging convenience: it is the
    stream `v3_served.verify_sentinel_filter` reads to prove the sentinel-filter
    patch is doing its job, so a tool or a ledger entry that wants to re-run
    that proof - or just to say which file it was proved from - must be able to
    find it without reconstructing a path.

    NO FALLBACK and no invention: an engine that declares no `backend_spec` at
    all, or one whose spec names no log, reports None. None means "narrator
    neither started this server nor was told where its log is", which is a
    different and honest answer.
    """
    spec_of = getattr(engine, 'backend_spec', None)
    if not callable(spec_of):
        return None
    return getattr(spec_of(), 'server_log', None)


def _edge_fade_of(engine):
    """The engine's EdgeFade, as the manifest and the pool see it.

    A shape, not a number: Higgs's fade is asymmetric (10 in / 25 out) because a
    chunk ends on a decay the ear does not expect. Matches the manifest's
    `edgeFadeMs: {in, out}` and assemble.engine_profiles.

    NO FALLBACK. An engine with no `edge_fade` used to become EdgeFade(0, 0)
    here, which is Orpheus's answer - so a `pads = False` engine that forgot to
    declare one would report "no fade needed" and the assembler would join its
    chunks bare. That is a book that CLICKS AT EVERY JOIN, shipped as a success.
    `edge_fade` is a member of the Engine protocol; an object without it is not
    an engine.
    """
    fade = getattr(engine, 'edge_fade', None)
    if fade is None:
        raise RuntimeError(
            f"engine '{getattr(engine, 'ENGINE_ID', type(engine).__name__)}' "
            'declares no edge_fade. It is a required member of '
            'narrator.engine.protocol.Engine, and guessing it is EdgeFade(0, 0) '
            "would silently drop a pads=False engine's chunk-edge fades - every "
            'join in the book would click.')
    return fade


def active_samplerate() -> int:
    return _ACTIVE_SAMPLERATE


def set_active_engine_audio(samplerate: int, pads: bool) -> None:
    """Adopt the loaded engine's audio facts. Called once per engine load."""
    global _ACTIVE_SAMPLERATE, _ACTIVE_PADS
    _ACTIVE_SAMPLERATE = int(samplerate)
    _ACTIVE_PADS = bool(pads)


class BatchRefused(ValueError):
    """A whole `generate_batch` is refused, BY NAME, before anything renders.

    `name` is the refusal - the first word of the message, and the word a
    caller matches on. One object rather than a bare string so the door can
    answer every row of the batch with the same sentence (that is this wire's
    refusal shape: one `batch_item` per requested `i`, then `batch_done`) and
    a test can assert the name without matching prose.
    """

    def __init__(self, name: str, detail: str):
        self.name = name
        super().__init__(f'{name}: {detail}')


def parse_retake(value) -> bool:
    """`retake` off the wire. ABSENT IS FALSE.

    WHO DECIDES WHETHER NARRATOR JUDGES (Owen's ruling, 2026-09-19). Before
    it, an engine that COULD guard a batch always did, and the caller had no
    say; the fine-tuning ladder needs the opposite - a screening render of a
    checkpoint has no measured pace and no measured cap, because measuring
    them is what the render produces, so judging it means judging it against
    another model's band. `true` takes the guarded arm (re-roll on truncation,
    runaway or loop), `false` or absent takes the bare one: every row rendered
    exactly once as sent, nothing judged, nothing retaken, nothing split.

    A VALUE THAT IS NOT A BOOLEAN IS REFUSED rather than coerced. `bool("false")`
    is True, and a client that spelled the flag wrong would get the opposite of
    what it asked for on every row of the book - silently, because both answers
    are ordinary audio.
    """
    if value is None:
        return False
    if not isinstance(value, bool):
        raise BatchRefused(
            'retake_malformed',
            f'`retake` must be true or false; got {value!r}. It is not coerced - '
            'a string "false" is truthy in Python, and the batch would be judged '
            'by the ladder its caller was trying to switch off.')
    return value


def batch_tracker(retake: bool, band, engine):
    """The `PaceTracker` for ONE batch, or None when nothing will judge it.

    THE BAND COMES FROM THE BATCH OR THE BATCH IS REFUSED (Owen, 2026-09-19).
    It used to come from the voice entry at load, and from the engine's own
    default band when the entry carried none - and that fallback is the defect
    this refusal exists for: a default band centred at 15.0 chars/s against a
    book actually running near 17.2 judged healthy chunks run-ons and re-rolled
    them to MAX_DEPTH. A band is a MEASUREMENT of one voice at one checkpoint;
    an engine cannot supply one for a checkpoint nobody has measured, and the
    caller that packed the book can.

    A BAND SENT WITH `retake` FALSE IS ACCEPTED AND NOT USED - it was not asked
    for (Owen: "it won't do anything with the number because it wasn't asked
    to"), so it is not read and not checked. Nothing here silently half-honours
    it: the bare arm judges nothing at all.
    """
    if not retake:
        return None
    if band is None:
        raise BatchRefused(
            'retake_without_band',
            'a batch that asks to be judged must carry the band it is judged '
            'against - {"paceCharsPerSec", "maxCharsPerSec", "minCharsPerSec"}. '
            "narrator will not fall back to the engine's own band: that band is "
            'a different measurement of a different model, and using it re-rolled '
            'healthy chunks of a 17.2 chars/s book against a 15.0 centre until the '
            'ladder hit MAX_DEPTH.')
    if not _can_guard_its_own_batch(engine):
        raise BatchRefused(
            'retake_unsupported',
            f"engine '{getattr(engine, 'ENGINE_ID', '?')}' offers no render_many, "
            'so it has no ladder to judge this batch with. The batch is refused '
            'rather than rendered unjudged - a render that silently skipped the '
            'retake it asked for would be reported as a success.')
    # IMPORTED HERE, not at module scope: this worker sends its `ready` line
    # before any engine import. A batch that asks to be judged has a Higgs
    # engine loaded already.
    from ..engine.higgs import truncation
    try:
        return truncation.tracker_for(band, 'generate_batch')
    except ValueError as exc:
        raise BatchRefused('band_malformed', str(exc)) from exc


def serving_width(engine) -> int:
    """The width the ENGINE WAS STARTED WITH - what a batch's `width` is
    measured against.

    `BATCH_SIZE` on every Higgs arm, and it is the process's own
    configuration, not a default: the served arms read `HIGGS_MAX_NUM_SEQS`
    (`v3_served.serve_concurrency`, which refuses to invent one) and the MLX
    arm reads `NARRATOR_HIGGS3_MLX_BATCH`. An engine that declares no
    `BATCH_SIZE` serves one row at a time, which is the honest reading of "no
    width was configured" and not a substituted value.
    """
    return max(1, int(getattr(engine, 'BATCH_SIZE', 1) or 1))


def parse_width(value, engine):
    """`width` off the wire: how many rows of THIS batch narrator may keep in
    flight. None (absent) means the serving width the engine was started with.

    WHY A JOB GETS TO NARROW ITS OWN BATCH (Owen, 2026-09-19, off a
    measurement): SGLang started 16 wide at `mem_fraction_static` 0.60 summed
    to 24.2 GB on a 24 GB card, and WDDM paged the excess to host RAM - 4 to 10
    times slower, with no error and no log line anywhere. The fine-tuning
    ladder renders 4 wide on voices whose manifest says 16, and restarting the
    server to say so is not a thing a job can do. So the width is a per-batch
    CEILING on what narrator asks of a server it does not reconfigure.

    IT CANNOT BE RAISED. A width above the serving width is refused by name
    rather than clamped: a caller that believes it is running 16 wide and is
    quietly given 4 has no way to find out, and the number it reports
    afterwards is wrong. Crucible refuses one above the voice's `max_num_seqs`
    before it reaches here; this is what happens when one arrives anyway.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BatchRefused(
            'width_malformed',
            f'`width` must be a whole number >= 1; got {value!r}. It is the '
            'number of rows this batch may keep in flight, so there is nothing '
            'to round it to.')
    serving = serving_width(engine)
    if value > serving:
        raise BatchRefused(
            'width_over_serving',
            f'this batch asks for width {value}, above the {serving} the engine '
            'was started with. narrator does not restart or reconfigure the '
            'server for a job, and it does not clamp: a caller that believes it '
            f'is running {value} wide while narrator quietly runs {serving} '
            'reports a number nobody rendered at.')
    return value


#: The two contracts `finalize_audio` serves, named because the caller is the
#: only thing that knows which one it is. See that function's docstring.
FOR_STREAM = 'stream'
FOR_RENDER = 'render'


def finalize_audio(audio_np, door, pads=None):
    """Prepare one rendered sentence for the wire, for the door that asked.

    WHOEVER ASSEMBLES THE SENTENCES OWNS THE GAP BETWEEN THEM (Owen,
    2026-09-18: "whoever assembles them is who owns the gap. I think that's
    bookforge."), so `door` is required and has no default: the caller states
    which contract this row is for, and the two are not the same.

      FOR_RENDER  an assembler follows. BookForge's narrator assembler realizes
                  `gaps.json` - 0.6 s, or the voice's inject - for a
                  `pads=False` engine (assemble/engine_profiles.py, higgs-v3).
      FOR_STREAM  a PLAYER follows. The extension's offscreen player and
                  BookForge's reader-audio-store insert the `gapSec` this
                  worker states on the row's terminal record (`row_gap_sec`),
                  which is `classify_gap`'s answer for that row - the same
                  number the assembler would realize for it in a book.

    SO BOTH DOORS EMIT BARE SPEECH, and what `door` still decides is who is
    told the gap: a FOR_STREAM emit carries `gapSec`, a FOR_RENDER emit must
    not, because behind it `gaps.json` already says it and two statements of one
    silence is two owners of it.

    WHY THE ARGUMENT EXISTS. This used to append a flat 0.3 s unconditionally,
    and the docstring justified it with "the audiobook path never goes through
    here". That stopped being true on 2026-09-13, when `_emit_guarded_batch`
    landed and became the door Crucible's `tts` render job drives: every join
    of a remotely rendered book came out model tail + 0.30 baked in here + 0.60
    from the assembler. A caller that does not say which door it is is a caller
    that has not decided, which is exactly how that happened. The 0.3 s itself
    is gone (2026-09-18): Listen paced at half the book's default and ignored
    the voice's inject entirely, because the player's silence was this
    function's constant rather than the book's rule.

    TWO STEPS, AND ONLY THE FIRST IS ENGINE-SPECIFIC.

    1. TRIM, for a `pads=True` engine only. Orpheus bakes its own lead/trail
       silence into every clip (`_classify_gap` + `_save_audio`), and its
       trailing end-pause is long enough to hear as a stall; this cuts back to
       the last sample above 0.01 plus ~150 ms. For a `pads=False` engine -
       Higgs, measured: no pads, no fades, bare speech at both ends - THE SAME
       TRIM WOULD CUT INTO THE WORD. A quiet final consonant sits under 0.01,
       and there is no padding in front of it to absorb the cut. So it is
       skipped, and the audio goes out exactly as decoded, which is also what
       `Engine.edge_fade` assumes (the fades are the assembler's).
    2. Peak-normalize if it clipped. Engine-independent.

    NOTHING IS APPENDED AT EITHER DOOR. The silence BETWEEN two chunks belongs
    to whoever joins them - the assembler for a book, the player on a stream -
    and neither of them is this function.

    `pads` defaults to the loaded engine's (see set_active_engine_audio).
    """
    if door not in (FOR_STREAM, FOR_RENDER):
        raise ValueError(
            f'finalize_audio was asked for door {door!r}, which is neither '
            f'{FOR_STREAM!r} (a player follows and is TOLD the gap, as `gapSec` '
            f'on the row\'s record) nor {FOR_RENDER!r} (an assembler follows '
            'and reads gaps.json). There is no third contract and no default: '
            'the caller is the only thing that knows.')
    if audio_np is None:
        return None
    a = np.asarray(audio_np, dtype=np.float32).flatten()
    if a.size == 0:
        return a
    rate = active_samplerate()
    if _ACTIVE_PADS if pads is None else pads:
        thr = 0.01
        idx = np.where(np.abs(a) > thr)[0]
        if idx.size:
            start = max(0, int(idx[0]) - int(rate * 0.05))
            end = min(a.size, int(idx[-1]) + int(rate * 0.15))
            a = a[start:end]
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    if peak > 1.0:
        a = a / peak * 0.95
    return a


def detect_device() -> str:
    """Informational device label for the 'ready' message. It only feeds
    status, not topology."""
    if _fake_engine_enabled():
        return 'cpu'
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    try:
        from mlx_audio.tts.utils import load_model  # noqa: F401
        return 'mlx'
    except Exception:
        pass
    return 'cpu'


def detect_backend():
    """The backend the engine WILL use in this process ('vllm' | 'mlx' |
    'transformers'), or None if it cannot be determined.

    Reported on the 'ready' line because the pool needs it BEFORE any voice loads:
    only vLLM can render a voice PER REQUEST, and only vLLM can be handed a local
    base dir for stock voices. Everything downstream of that answer is a correctness
    guard, so it is taken from the engine's own detector rather than re-derived here
    from the platform - a second implementation of "which backend" is a second thing
    to drift.

    Costs an import of the engine module (torch), NOT a model load; detect_device()
    above already pays the torch import.

    IT RAISES. It used to swallow every exception and return None, "reporting
    unknown", on the theory that an env with broken imports would degrade to
    pre-adapter behaviour. That theory was wrong, and the Mac validation run
    caught it: importing the engine is also how the engine's MODULE is loaded,
    so an ImportError here means no engine will EVER load in this process. The
    worker then printed `{"type":"ready","device":"mlx"}` with no `backend`, the
    pool saw a healthy handshake, and every generate answered "Model not loaded"
    - a worker that looks alive and can never render. A backend that cannot be
    determined is a dead worker; `main()` exits non-zero with the reason on
    stderr instead of handshaking.

    A backend NAME this worker does not recognise is a different thing and still
    returns None: the engine loaded and answered, we simply have no guard
    calibrated for what it said.
    """
    backend = _engine_class().detect_backend()
    known = ('vllm', 'mlx', 'transformers', 'vllm-omni')
    if backend not in known:
        print(f'[narrator.serve] engine reported backend {backend!r}, which is not '
              f'one of {known}; reporting unknown', file=sys.stderr)
        return None
    return backend


# -- the stream server ---------------------------------------------------------


class StreamServer:
    def __init__(self):
        self.orph = None              # the loaded engine (lazy; narrator's name for it)
        self.current_voice = None
        # Voices this engine can serve RIGHT NOW: name -> adapter dir (always None
        # for Higgs, which has no runtime adapter). Populated by 'load', emptied by
        # teardown - so it can never claim a voice that went away with a previous
        # engine.
        self.engine_voices = {}
        # name -> the CATALOG ID that claimed it (see _check_token_owner). Two ids
        # declaring the same name would otherwise collapse into one slot.
        self.engine_voice_ids = {}
        self.device = None
        self.backend = None           # the engine's detected backend, probed at 'ready'
        # Set by the STDIN READER THREAD the moment a 'cancel'/'stop' arrives, so a
        # generate_batch already running on the main thread can see it between decode
        # steps and abandon what is left. Cleared by the main loop when it dequeues
        # that same cancel and acknowledges it - never anywhere else, so the flag can
        # only ever describe work that was already in flight when the cancel landed.
        self._cancel = threading.Event()

    def _ensure_engine(self, voice: str, model_dir: str = None, caps: dict = None,
                       adapter_dir: str = None, base_dir: str = None,
                       voice_id: str = None, warm: bool = True):
        """Load the engine and make `voice` its voice; on first load WARM the
        generate path before reporting ready - unless the caller asked for a cold
        ready (`warm=False`), see _warmup.

        A Higgs engine is built ONCE per process, for one voice: a served
        engine's voice is baked into the server it launched and the MLX arm
        holds one set of weights. A second load is handed to the engine's
        `set_voice`, which takes the SAME voice as the no-op it is and refuses
        any other by name. `modelDir`, `baseDir` and `adapterDir` are still read
        off the message and handed to the engine, whose `resolve_load_voice` and
        config factory refuse each of them by name - narrator's Orpheus
        pre-checks of the same fields went with Orpheus.
        """
        # THE ENGINE VALIDATES ITS OWN VOICES: an entry in the
        # NARRATOR_HIGGS_VOICES document, with the directory it names checked
        # for its required files - before a server start and before anything
        # holds a GPU.
        try:
            v = _engine_class().resolve_load_voice(
                voice, model_dir=model_dir, adapter_dir=adapter_dir,
                base_dir=base_dir)
        except Exception as e:
            send_response('error', {'message': str(e)})
            return False

        if not self._check_token_owner(v, voice_id):
            return False

        first_load = self.orph is None
        if first_load:
            send_response('status', {'message': f'Loading {engine_id()} model...'})
            # Imported here so 'ready' is sent before the heavy engine import, and
            # so an env without its deps fails on load (surfaced) not at startup.
            engine_cls = _engine_class()
            config = _engine_config(
                voice=v,
                model_dir=model_dir,
                base_dir=base_dir,
                adapter_dir=adapter_dir,
                caps=caps or {},
            )
            self.orph = engine_cls(config)      # __init__ -> load_engine() loads model
            # Every duration, every `sampleRate` on the wire and finalize_audio's
            # trim describe THIS engine instead of a module constant.
            set_active_engine_audio(getattr(self.orph, 'SAMPLE_RATE',
                                            DEFAULT_SAMPLERATE),
                                    bool(getattr(self.orph, 'pads', True)))
            send_response('status', {'message': 'Model loaded'})
        else:
            # Warm engine. NOT EVERY ENGINE CAN SWITCH IN PLACE, and an engine
            # without `set_voice` is refused BY NAME instead of raising a bare
            # AttributeError from a method it does not have.
            if not hasattr(self.orph, 'set_voice'):
                send_response('error', {
                    'message': f"engine '{getattr(self.orph, 'ENGINE_ID', '?')}' "
                               'cannot switch voice on a live worker: it has no '
                               'set_voice(). Quit this worker and start one for '
                               'that voice.'
                })
                return False
            # The same voice is a no-op; any other is refused by name.
            self.orph.set_voice(v, adapter_dir)
            # A refused caps payload must not leave the loaded engine serving
            # requests, so tear the engine down and let the error propagate as a
            # load failure.
            try:
                self._apply_voice_caps(v, caps)
            except Exception:
                self._teardown_engine()
                raise
        self.current_voice = v
        # What this engine can serve RIGHT NOW. On MLX a load REPLACES the voice
        # (the weights ARE the voice); a served engine accumulates, which for a
        # one-voice engine is the same single entry.
        if self.orph.backend == 'mlx':
            self.engine_voices = {v: adapter_dir}
        else:
            self.engine_voices[v] = adapter_dir
        # Only a load that NAMES an id claims the name. A load without one must
        # not overwrite an existing claim with None - that would quietly disarm
        # _check_token_owner for every load after it.
        if voice_id:
            self.engine_voice_ids[v] = voice_id

        # Warm the generate path ONCE per load, so the cold-start cost is paid
        # here and not on the first chunks a job renders - unless the load said
        # warm=False because a request is already waiting.
        if first_load and warm:
            self._warmup()

        send_response('status', {'message': f'Voice loaded: {v}'})
        return True

    def _is_cancelled(self) -> bool:
        """Has a 'cancel'/'stop' arrived since this work started?

        Handed to the engine as `should_stop` and checked around each group, so it
        must stay a plain flag read - no I/O, no locking beyond Event's own.
        """
        return self._cancel.is_set()

    def _check_token_owner(self, token: str, voice_id: str) -> bool:
        """Refuse a load whose voice NAME is already claimed by a different catalog id.

        A caller may identify a voice by a catalog id that differs from the name
        the engine resolves, and nothing in the wire format stops TWO ids from
        naming the SAME voice. If that happened they would collapse into one slot
        here and every request for the first would be served the second, as a
        success.

        A load that carries no id claims nothing and is accepted - it cannot
        create a collision, only fail to detect one.
        """
        if not voice_id:
            return True
        owner = self.engine_voice_ids.get(token)
        if owner is None or owner == voice_id:
            return True
        send_response('error', {
            'message': f"voice id '{voice_id}' wants the voice name '{token}', "
                       f"which is already registered on this engine to '{owner}'. Two voices "
                       'cannot share one name - the second would be rendered as the '
                       'first and reported as a success. Give one of them a distinct '
                       'name in the voices document.'
        })
        return False

    def _teardown_engine(self):
        """Release the current engine so nothing serves on it.

        engine_voices is emptied with it: every entry describes what THIS engine
        can serve, and a surviving entry would let a request for a voice from the
        previous engine be accepted and rendered in whatever is loaded now."""
        # The engine's audio facts go with it - a later load re-adopts.
        set_active_engine_audio(DEFAULT_SAMPLERATE, True)
        if self.orph is not None:
            try:
                self.orph.cleanup()
            except Exception:
                pass
            self.orph = None
        self.current_voice = None
        self.engine_voices = {}
        self.engine_voice_ids = {}

    def _warmup(self):
        """Pay the backend's first-generate cold-start now, at load.

        Output is discarded; only the side effect is wanted. Failures are
        non-fatal - a warmup hiccup must never block the voice from loading.

        THREE shapes, exactly as narrator rendered them for every Crucible load:
        three single chunks of increasing length, then one batch at
        WARMUP_RAMP_WIDTH and one at WARMUP_FULL_WIDTH of the same sentence (see
        those constants for where the numbers came from). A Higgs engine renders
        `_generate_audio_batch` one row at a time, so this is 3 + 8 + 16 renders.
        """
        widths = sorted({w for w in (min(WARMUP_RAMP_WIDTH, WARMUP_FULL_WIDTH),
                                     WARMUP_FULL_WIDTH) if w > 1})
        shapes = ', '.join(str(w) for w in (1, *widths))
        send_response('status', {'message': f'Warming up voice (widths {shapes})...'})
        warm_texts = (
            'Hello.',
            'This is a brief warmup.',
            'Here is a slightly longer warmup sentence to prepare smooth playback.',
        )
        # 1) Single-chunk path, a few increasing lengths.
        for t in warm_texts:
            try:
                # Discarded - the side effect is the warmup - so the door is
                # nominal; FOR_STREAM because nothing assembles a render nobody
                # keeps.
                self._generate_audio(t, door=FOR_STREAM)
            except Exception as e:
                print(f'[narrator.serve] warmup generation failed (non-fatal): {e}',
                      file=sys.stderr)
        # 2) The batch path, at the ramp width and the full width. UNIFORM texts.
        for w in widths:
            try:
                # Discarded too (see above).
                self._generate_audio_batch([warm_texts[1]] * w, door=FOR_STREAM)
            except Exception as e:
                print(f'[narrator.serve] batch warmup (width {w}) failed (non-fatal): {e}',
                      file=sys.stderr)
        send_response('status', {'message': 'Warmup complete'})

    def load_voice(self, voice: str, model_dir: str = None, caps: dict = None,
                   adapter_dir: str = None, base_dir: str = None,
                   voice_id: str = None, warm: bool = True) -> bool:
        try:
            return self._ensure_engine(voice, model_dir, caps, adapter_dir, base_dir,
                                       voice_id, warm)
        except Exception as e:
            # NAME THE ENGINE THAT ACTUALLY FAILED. This said "Failed to load
            # Orpheus" for every engine, so a Higgs load failure on the Mac
            # reported the wrong model by name - which is precisely the moment
            # someone is trying to work out WHICH of two engines is broken.
            # engine_id() cannot raise here: an unservable NARRATOR_ENGINE is
            # refused in main() before any load is accepted.
            send_response('error',
                          {'message': f'Failed to load {engine_id()}: {e}'})
            return False

    def _row_voice(self, voice) -> str:
        """The voice token one generate row must render in.

        Absent/blank -> the server's current default voice, which is what every
        request meant before per-request voices existed. Anything else must have been
        registered against the LIVE engine by a 'load'; a voice this engine cannot
        serve is an ERROR, never a quiet render in whatever happens to be loaded -
        that is the wrong narrator delivered as a success.
        """
        if self.current_voice is None:
            raise RuntimeError('Model not loaded')
        if voice is None or not str(voice).strip():
            return self.current_voice
        v = str(voice).strip().lower()
        if v in self.engine_voices:
            return v
        known = ', '.join(sorted(self.engine_voices)) or '(none)'
        raise ValueError(
            f"voice '{v}' is not loaded on this stream server. Loaded: {known}. "
            'Send a load for it first - a voice is its weights or its encoded '
            'reference, and neither can be conjured for a single request.'
        )

    def _apply_voice_caps(self, voice: str, caps: dict) -> None:
        """A second load's `caps`, handed to the ENGINE's own `_apply_voice_caps`.

        Both Higgs arms define it: an empty payload is the "no tuning" no-op,
        and anything else is refused by name, because the cap names were
        Orpheus's and Higgs implements none of them. narrator called Orpheus's
        `register_voice_caps` here, which no Higgs engine has - so a same-voice
        re-load of a live Higgs worker failed on an AttributeError and tore the
        engine down. Asking the engine is the call that was always meant.
        """
        self.orph._apply_voice_caps(voice, caps or {})

    def _reject_per_request_voice(self, voice: str) -> None:
        """A Higgs engine renders ONE voice - the one it loaded - so a row naming
        another is refused instead of rendered in the wrong one.

        Raised PER ROW, never over a whole batch: one stray voice must not fail
        the ordinary rows beside it."""
        if voice != self.current_voice:
            raise ValueError(
                f"The '{self.orph.backend}' backend cannot render '{voice}' per request - "
                f"it serves one loaded voice at a time (currently '{self.current_voice}'). "
                'Load the voice before generating in it.'
            )

    def _resolve_row(self, item):
        """`(voice token, sampling, take)` one BATCH item must render under, or
        raise with the reason.

        A RUNG IS TWO FACTS and both are resolved here: the numbers
        (`sampling`) and the seed lane (`take`). They are independent - a rung
        that declares no sampling override is still a different draw, because
        the lane moved - so an item may carry either, both or neither.

        The per-item rejections in one place: a voice this engine never loaded
        (_row_voice), a voice other than the loaded one, and a take-ladder rung
        this ENGINE cannot honour. Every one of them fails the one item, not the
        batch - that is the whole reason they live here and not in a loop over
        the whole batch.

        THE ENGINE PARSES ITS OWN RUNG. `accept_item_sampling` and
        `accept_item_take` are members of every Higgs arm, and each arm knows
        its own levers.

        narrator let a row name another voice on Orpheus's vLLM backend, which
        rendered a voice per request; no Higgs backend does, so the check is
        unconditional here - which is what it already was for every Higgs load.
        """
        v = self._row_voice(item.get('voice'))
        self._reject_per_request_voice(v)
        where = f'generate_batch row i={item.get("i")!r}'
        return (v, self._item_rung(item.get('sampling'), where),
                self._item_take(item.get('take'), where))

    def _item_take(self, raw, where: str) -> int:
        """One item's `take` off the wire -> the engine's own answer, or 0 when
        the item asked for none.

        `_item_rung`'s rule, for `_item_rung`'s reasons, and the docstring
        there is the argument. THE ENGINE IS ASKED ONLY WHEN THERE IS SOMETHING
        TO ASK ABOUT: an absent `take` has one meaning for every engine - "take
        0, the seed rule narrator has always had" - so calling the member on
        every row of every batch would turn "this engine object predates the
        member" into "every row fails", including rows that asked for nothing.

        AN EXPLICIT 0 IS ALSO NOTHING TO ASK ABOUT, and that is deliberate
        rather than lazy: Crucible sends `take` on EVERY item, 0 included, so
        that the wire says which take produced the artifact. Take 0 is the
        draw every narrator ever built already makes, so an engine with no lane
        answers it correctly by doing nothing, and only a take ABOVE 0 is a
        question - which is exactly where `take_not_supported` lives.

        A TAKE ARRIVING AT AN ENGINE WITH NO ANSWER IS STILL A REFUSAL, by
        name, never an AttributeError about a method: the caller asked for a
        different draw and needs to know it did not get one.
        """
        from ..engine.item_sampling import parse_item_take
        # THE SHAPE FIRST, and it is not the engine's business: "a whole number
        # >= 0 within MAX_TAKE" is the same sentence on every backend, and
        # `take: "2"` from an engine that has no lane should still be
        # `take_malformed` rather than `take_not_supported` - the two say
        # different things to whoever sent it, one a typo and one a wrong
        # backend.
        if parse_item_take(raw, where) == 0:
            return 0
        accept = getattr(self.orph, 'accept_item_take', None)
        if accept is None:
            raise ValueError(
                f"take_not_supported: {where} carries take {raw!r}, and engine "
                f"'{getattr(self.orph, 'ENGINE_ID', '?')}' implements no "
                '`accept_item_take`. It cannot say what it would do with a take '
                "above 0, so this row is refused rather than rendered in take 0's "
                'own seed lane and reported as the take.')
        return accept(raw, where)

    def _item_rung(self, raw, where: str):
        """One item's `sampling` off the wire -> the engine's own parsed rung,
        or None when the item asked for none.

        THE ENGINE IS ASKED ONLY WHEN THERE IS SOMETHING TO ASK ABOUT, and that
        is the whole of this method. An absent rung has ONE meaning for every
        engine narrator drives - "render this at the voice's loaded sampling,
        which is take 0" - and every implementation of `accept_item_sampling`
        answers None to None. Calling it anyway on every row of every batch is
        ceremony that buys nothing and costs the additive guarantee this
        channel is supposed to have: it turns "this engine object predates the
        member" into "every row of every batch fails", including the rows that
        never asked for anything. Measured 2026-09-14 - a stub engine in
        tests/test_higgs_mlx.py, standing in for a Higgs load on the Mac, had
        every sentence refused with an AttributeError about a field it had not
        sent.

        A RUNG ARRIVING AT AN ENGINE THAT HAS NO ANSWER IS STILL A REFUSAL, by
        name, and not an AttributeError about a method: the caller asked for
        numbers and needs to know they were not applied. `Engine
        .accept_item_sampling` is a declared protocol member and
        tests/test_engine_protocol.py holds every engine to it, so this arm is
        the report of a wiring defect rather than a path anything takes.
        """
        if raw is None:
            return None
        accept = getattr(self.orph, 'accept_item_sampling', None)
        if accept is None:
            raise ValueError(
                f"sampling_not_supported: {where} carries sampling {raw!r}, and "
                f"engine '{getattr(self.orph, 'ENGINE_ID', '?')}' implements no "
                '`accept_item_sampling`. It cannot say what it would do with a '
                'take-ladder rung, so this row is refused rather than rendered at '
                'the voice default and reported as the rung.')
        return accept(raw, where)

    def _generate_audio(self, text: str, voice: str = None, index: int = 0,
                        sampling=None, take: int = 0, *, door):
        """Render one chunk to a float numpy waveform through the engine's
        `render_audio` (or `render_audio_measured`, which also reports what the
        render spent against its frame ceiling).

        `voice` (default: the loaded voice) must be the loaded one - see
        _reject_per_request_voice.

        `sampling` is this item's take-ladder rung, ALREADY PARSED by the
        engine's `accept_item_sampling` (never the raw wire value). None means
        the voice's loaded sampling - take 0.

        `take` is the rung's other half, ALREADY PARSED by the engine's
        `accept_item_take`: which seed lane this render draws from. 0 is take 0.

        Returns `(audio, measure)`."""
        orph = self.orph
        v = self._row_voice(voice)
        clean = orph._clean_sentence_for_tts(text)
        if not clean:
            # An empty chunk is silence, and silence was not generated: there
            # is no cap it could have hit.
            return np.zeros(int(active_samplerate() * 0.05), dtype=np.float32), None
        measure = None
        # Higgs v3 - served by SGLang-Omni or vllm-omni on Linux, in-process
        # through mlx-audio on the Mac. There is no token stream on this side
        # to decode; one call, one waveform.
        self._reject_per_request_voice(v)   # one voice per loaded engine
        render = getattr(orph, 'render_audio', None)
        # `index` seeds the row (seed + i). The sequential path in
        # _generate_audio_batch passes the row's own index, so a batch does
        # not render every sentence from the same draw; a single 'generate'
        # is index 0, which is the whole batch it is.
        if render is None:
            raise RuntimeError(
                f"engine '{getattr(orph, 'ENGINE_ID', '?')}' offers no "
                'render_audio(text). This worker has no way to render one '
                'sentence with it.')
        # MEASURED WHERE THE CAP IS KNOWN. An engine that can say what a
        # render spent against its frame ceiling offers
        # `render_audio_measured` and is asked for both; one that cannot
        # renders and reports nothing, which the wire publishes as absent -
        # "narrator did not say" - and never as `capped: false`.
        measured = getattr(orph, 'render_audio_measured', None)
        if callable(measured):
            audio, measure = measured(clean, index=index, sampling=sampling,
                                      take=take)
        else:
            audio = render(clean, index=index, sampling=sampling, take=take)
        # THE DOOR IS THE CALLER'S, and it is required here for the reason
        # `finalize_audio` requires it: the caller is the only thing that knows
        # whether a player or an assembler follows. It used to be FOR_STREAM on
        # this line, justified by a ROUTING fact - "a Higgs engine is always
        # handed to _emit_guarded_batch, so only Listen reaches here". That
        # stopped being true on 2026-09-19, when `retake: false` made the bare
        # arm the render door's own unguarded path: a Crucible render batch
        # that asks for no judging comes through here, and an assembler
        # follows it.
        return finalize_audio(audio, door), measure

    def _generate_audio_batch(self, texts, voices=None, samplings=None,
                              takes=None, indexes=None, *, door):
        """Render many chunks, ONE AT A TIME through `_generate_audio`. Returns
        a list of `(audio, measure)` aligned to `texts` (a tiny silence for an
        empty chunk; `(None, None)` for a non-empty chunk that failed).

        Reached by `_warmup` and by `generate_batch` for an engine with no
        `render_many` - which no Higgs arm is, so a real batch never comes here.
        narrator's vLLM and MLX batch arms above this one were Orpheus's.

        `voices`, `samplings` and `takes`, when given, are aligned to `texts`:
        each row's voice (None = the loaded one), PARSED rung (None = take 0)
        and seed lane (0 = take 0).

        `door` is required and is passed straight to `finalize_audio`.

        `indexes`, when given, is aligned to `texts` and is THE CALLER'S OWN
        chunk index for each row - what seeds it (`seed + index`). The position
        in a batch is an accident of how the caller packed it, so seeding by
        position would render chunk 412 differently depending on which rows
        travelled with it. Absent (`_warmup`) keeps the position."""
        row_voices = [self._row_voice(voices[i] if voices else None)
                      for i in range(len(texts))]
        row_takes = list(takes) if takes is not None else [0] * len(texts)
        if len(row_takes) != len(texts):
            raise ValueError(
                f'narrator.serve: {len(row_takes)} take(s) for {len(texts)} rows; '
                'takes must be aligned to texts or None.')
        row_sampling = list(samplings) if samplings is not None else [None] * len(texts)
        if len(row_sampling) != len(texts):
            raise ValueError(
                f'narrator.serve: {len(row_sampling)} sampling(s) for {len(texts)} '
                'rows; samplings must be aligned to texts or None.')
        row_indexes = list(indexes) if indexes is not None else list(range(len(texts)))
        if len(row_indexes) != len(texts):
            raise ValueError(
                f'narrator.serve: {len(row_indexes)} index(es) for {len(texts)} '
                'rows; indexes must be aligned to texts or None.')

        return [self._generate_audio(t, row_voices[i], index=row_indexes[i],
                                     sampling=row_sampling[i], take=row_takes[i],
                                     door=door)
                for i, t in enumerate(texts)]

    @staticmethod
    def _emit_batch_item(it, audio, door, guard=None, measure=None):
        """Emit one 'batch_item', keyed by the caller-supplied index `i`. Empty/None
        audio -> the 'No audio generated' message; otherwise the PCM16 payload. This
        is the exact per-item wire shape the non-MLX single-dispatch loop uses, so
        MLX group emission and non-MLX emission are byte-identical per item.

        `door` is `finalize_audio`'s, restated here because it decides one field:
        a FOR_STREAM row carries `gapSec` - the silence the PLAYER must insert
        after it, `classify_gap`'s answer for the row's own text - and a
        FOR_RENDER row carries none, because behind that door the assembler
        realizes `gaps.json` and a second statement would be a second owner. The
        row's raw `text` is classified, never the normalized copy (see
        `row_gap_sec`), and a row whose gap LEADS is refused there by name.

        `guard` is `GuardPlan.verdict()`'s object, verbatim, for a row an engine
        rendered through its own retake ladder - docs/history/PHASE6-REMOTE-RENDER.md section 3.
        STRICTLY ADDITIVE AND OPTIONAL: a row with no verdict carries no `guard`
        key at all, so electron/orpheus-worker-pool.ts and the browser extension
        read exactly the field set they read today and need no change on the day
        this lands. It is the CONCLUSION, not the evidence - `verdict` is what the
        engine decided, `takes` is why - and a client that acts on `takes` instead
        of `verdict` is re-litigating a decision the model has already made.

        It rides only the success shape. A failure item is `{i, message}` and says
        nothing about a ladder: `render_many` hands back `verdict is None` for a
        chunk that never reached a decision (`GuardPlan.abandon` drops its
        records), so there is nothing true to attach.

        `measure` is the render's `FrameMeasure` and is where `capped` (and
        `tokens`) come from. THE MEASUREMENT IS UNCONDITIONAL (Owen,
        2026-09-19): it rides both arms, because whether a generation ran out
        of frame budget is a fact about the render and not a verdict on it, and
        a bare render is exactly the one whose numbers nobody else has. A
        missing measure omits the keys rather than sending `false` - Crucible
        reads absent as "narrator did not say", and "not capped" is the one
        thing a runaway must never be reported as
        (crucible/docs/PHASE18-UNCERTIFIED.md). `tokens` is sent only when the
        count is COUNTED: the served stack returns a WAV and no token count, so
        the frames narrator can see there are a lower bound
        (`FrameMeasure.counted`).
        """
        if audio is None or len(audio) == 0:
            send_response('batch_item', {'i': it.get('i'), 'message': 'No audio generated'})
        else:
            # THE INTERIOR-PAUSE CAP (serve/pause_cap.py; Owen, 2026-10-02), on
            # the render door only: an assembler follows it, and a stall inside a
            # chunk would go into the book. After the ladder has judged the take -
            # the verdict is about the render, the cap is about what ships - and
            # before `duration`, which is measured from what is sent. `pauseCuts`
            # rides EVERY render row, empty when nothing was cut, so a reader can
            # tell "nothing to cut" from a narrator that never looked.
            cuts = None
            if door == FOR_RENDER:
                audio, cuts = cap_interior_pauses(
                    audio, active_samplerate(), max_pause_seconds())
                for cut in cuts:
                    print(f'[narrator.serve] PAUSE CUT row {it.get("i")}: an interior '
                          f'pause of {cut["fromS"]:.2f}s at {cut["atS"]:.2f}s cut to '
                          f'{cut["toS"]:.2f}s', file=sys.stderr, flush=True)
            send_response('batch_item', {
                **({'pauseCuts': cuts} if cuts is not None else {}),
                'i': it.get('i'),
                'format': 'pcm16',
                'data': audio_to_pcm16_base64(audio),
                'duration': len(audio) / active_samplerate(),
                'sampleRate': active_samplerate(),
                **({'gapSec': row_gap_sec(it.get('text'))}
                   if door == FOR_STREAM else {}),
                **({'guard': guard} if guard is not None else {}),
                **({'capped': bool(measure.capped)} if measure is not None else {}),
                **({'tokens': measure.tokens}
                   if measure is not None and measure.tokens is not None else {}),
            })

    def _emit_guarded_batch(self, rows, emitted, tracker, width=None):
        """Drive `rows` through the ENGINE'S OWN guarded driver and emit each chunk
        as the ladder decides it, verdict attached.

        `tracker` is THE BATCH'S `PaceTracker`, built from the band the batch
        carried (`generate_batch`), and there is no other source for it: the
        band has one owner and it is the caller (Owen, 2026-09-19). A batch
        that asks to be judged and names no band never reaches this method -
        `retake_without_band` refuses it whole.

        `width` is how many of this batch's takes the driver may keep in
        flight, already checked against the serving width (`parse_width`).
        None means the engine's own.

        `rows` is generate_batch's `(item, normalized text, voice token,
        sampling, take)` quintuples, already past the per-row voice and rung
        resolution; `emitted` is that method's one-answer-per-item set, added
        to here so its `finally` sweep can still label anything this never
        reached.

        THE RUNG GOES TO THE DRIVER AS TWO MAPS KEYED BY CHUNK INDEX, not on
        the row, because that is the key the ladder itself uses: a re-roll and
        both halves of a split carry their parent's index, and all of them
        must render at the numbers - and in the seed lane - that chunk was
        asked for. Each map names EVERY chunk this call plans (None / 0 for a
        chunk at take 0), so `render_many` can refuse an unnamed one instead of
        quietly rendering it at take 0.

        WHY THIS EXISTS. Until 2026-09-13 the serve world handed a Higgs engine one
        `render_audio()` per sentence - no PaceTracker, no re-roll, no split ladder -
        while the audiobook world ran the same model through all three. Owen ruled
        that the model and its inference own the guard AND the retake decision, so
        the guard runs where the render does and the verdict reaches the wire
        (crucible/docs/PHASE6-REMOTE-RENDER.md sections 0, 2 and 3). Crucible's
        render job drives exactly this door.

        THE LADDER INDEX IS THE CALLER'S `i`, NOT THE POSITION IN THE BATCH. It
        seeds the render (`seed + index`) and names the reject files, so a re-render
        of chunk 412 is reproducible only if 412 is what the ladder was told - and
        Crucible sends the book's own chunk index. The position in a read-ahead
        window is an accident of scheduling and would make the same sentence render
        differently on a resume.

        NOT ONE GUARD DECISION IS MADE HERE. The plan, the bands, the depth-first
        order, MIN_GUARD_CHARS and the split all live inside `render_many`; this
        method starts it, keeps the book-keeping, and ships what it yields.

        RETIREMENT ORDER, NOT READING ORDER. `render_many` yields as the ladder
        decides, so a chunk that re-rolled lands after its neighbours. That is
        already this wire's contract - clients assemble by index - and the three
        guarantees
        the pool depends on are untouched: one batch_item per requested `i`,
        exactly once, then batch_done last, always.

        ONE ARTIFACT PER REQUESTED INDEX, even when the ladder SPLIT. The halves are
        joined (`truncation.join_parts`) before the chunk retires, so a split shows
        up only as `parts: 2` on the verdict. Returning two rows would push the
        ladder's private business into every consumer - PHASE6 section 5.

        CANCELLABLE, BETWEEN CHUNKS (2026-09-15). `self._is_cancelled()` is read
        before the driver is started and again after every chunk it retires; on
        a cancel the generator is CLOSED, which abandons the ladder, and the
        rows never rendered are labelled by generate_batch's own sweep. See the
        check itself for the worst case, for why nothing is plumbed into
        `render_audio`, and for what it cost to have been missing.
        """
        orph = self.orph
        by_index = {}     # ladder index -> the item that asked for it
        plan_rows = []    # (index, cleaned text) - the driver's input
        rungs = {}        # ladder index -> that chunk's sampling (None = take 0)
        takes = {}        # ladder index -> that chunk's take (0 = take 0)

        def fail(it, message):
            send_response('batch_item', {'i': it.get('i'), 'message': message})
            emitted.add(id(it))

        for it, text, voice, sampling, take in rows:
            try:
                # A BACKSTOP, not the gate: _resolve_row already refused a
                # per-request voice on every backend that cannot serve one. It
                # matters here because `render_many` takes NO voice - it renders in
                # whatever the engine has loaded - so a row that slipped through
                # naming another voice would be narrated by the wrong voice and
                # reported as a success.
                self._reject_per_request_voice(voice)
            except Exception as e:
                fail(it, str(e))
                continue

            index = it.get('i')
            try:
                index = int(index)
            except (TypeError, ValueError):
                # The index is the LADDER KEY here, not just a label on the wire,
                # so an unusable one cannot be carried along and sorted out by the
                # client: it would seed the render and key the verdict.
                fail(it, f'generate_batch row carries i={it.get("i")!r}, which is '
                         'not an integer index. A guarded render uses `i` as the '
                         'ladder index - it seeds the take and names the reject '
                         'files - so there is nothing to render this row as.')
                continue
            if index in by_index:
                # GuardPlan keys its records and its verdict by chunk index, so two
                # rows claiming one index would share a verdict and one of the two
                # renders would be thrown away. The pool already drops the second
                # message for an `i` as stale, so this was always a caller bug; on
                # this arm it is refused by name instead of silently half-served.
                fail(it, f'generate_batch sent index {index} twice in one batch. '
                         'The retake ladder is keyed by chunk index, so the second '
                         'row cannot be rendered or reported separately.')
                continue

            clean = orph._clean_sentence_for_tts(text)
            if not clean:
                # "Empty -> tiny silence", the contract every batch path here
                # keeps. It never goes on the ladder: there is no take to judge,
                # and a Higgs render_audio refuses a blank chunk by name.
                self._emit_batch_item(
                    it, np.zeros(int(active_samplerate() * 0.05), dtype=np.float32),
                    FOR_RENDER)
                emitted.add(id(it))
                continue

            by_index[index] = it
            plan_rows.append((index, clean))
            rungs[index] = sampling
            takes[index] = take

        if not plan_rows:
            return

        if self._is_cancelled():
            # Nothing has been asked of the driver yet, so a cancel that landed
            # while the rows were being resolved costs no render at all. The
            # `finally` sweep in generate_batch labels every row 'cancelled'.
            return

        driver = orph.render_many(plan_rows, sampling_by_index=rungs,
                                  take_by_index=takes, tracker=tracker,
                                  width=width)
        for index, audio, verdict, measure in driver:
            it = by_index.pop(index, None)
            if it is None:
                # The driver yielded an index this batch never asked for (or asked
                # for twice). Emitting it would answer a row the pool has no
                # resolver for; swallowing it silently would hide a driver bug.
                print(f'[narrator.serve] guarded driver yielded index {index}, which '
                      'this batch did not request - dropping', file=sys.stderr)
                continue
            emitted.add(id(it))
            # `audio is None` is render_many's failure signal, and `verdict` is then
            # None too - _emit_batch_item turns that into the ordinary
            # 'No audio generated' item, never silence dressed as a success.
            # FOR_RENDER: this is the door Crucible's `tts` render job drives,
            # and behind it BookForge's narrator assembler realizes the
            # manifest's gaps. A gap appended here - or STATED here, as `gapSec`
            # - would be a second one on every join of the book.
            self._emit_batch_item(
                it,
                None if audio is None or len(audio) == 0
                else finalize_audio(audio, FOR_RENDER),
                FOR_RENDER,
                guard=verdict, measure=measure)
            if self._is_cancelled():
                # CANCELLABLE, BETWEEN CHUNKS. `render_many` is a generator, so
                # closing it is the whole of the stop: `break` drops the last
                # reference, GeneratorExit unwinds `_render_many_serial` - whose
                # docstring already names GeneratorExit as one of the three
                # things that must unwind rather than be caught - and no further
                # take is ever issued. Nothing is plumbed into the ladder, which
                # is deliberate: a `should_stop` inside `render_audio` would make
                # a stopped take look like a FAILED one to `GuardPlan`, and the
                # ladder would spend a retake on the very chunk the cancel is
                # abandoning.
                #
                # WORST CASE IS ONE RENDER IN FLIGHT - one chunk's take, ~20s on
                # MLX, and up to one retake rung behind it if the ladder was
                # part-way up a chunk when the flag flipped. Interrupting inside
                # a chunk would mean teaching GuardPlan the difference between a
                # take that failed and a take that was stopped, which PHASE6
                # section 8 says has to be re-measured against the Mistborn
                # pause map. Not worth it for twenty seconds.
                #
                # WHY THIS WAS MISSING (measured on the Mac, 2026-09-15). Every
                # other arm here has had it since the reader thread landed -
                # _generate_batch_streaming hands should_stop to the engine -
                # but this arm was written on 2026-09-13 for the guarded ruling
                # and never grew one. It is the ONLY arm Crucible's render door
                # drives, so a cancelled render of `thirdreich` kept going for
                # eleven minutes after the DELETE, holding narrator's wire, the
                # claim and the card.
                #
                # The rows never rendered are answered 'cancelled' by
                # generate_batch's `finally` sweep, and batch_done still fires
                # last: one answer per requested item, cancel or not.
                #
                # CLOSED EXPLICITLY rather than left to the refcount: the frame
                # holding `driver` lives until this method returns, and "the
                # ladder stops the moment the flag is read" is the claim, not
                # "shortly afterwards, on this interpreter".
                driver.close()
                break

    def _generate_batch_streaming(self, items, language: str):
        """FAST START: render the batch through the engine's generate_batch_stream so
        the rows marked `stream: true` deliver sub-sentence chunks WHILE THEY
        GENERATE.

        Owen's ruling of 2026-09-04: without it the extension waits ~30s before the
        first word, because a sentence only exists once its whole batch retires and
        the client then gates playback on a cushion. With the extension's "Buffer
        before playing" switch OFF, the row being listened to streams instead - 4
        SNAC frames (~0.34s) at a time - and the client starts on about a second of
        audio. Stalls are an accepted cost of that mode; the switch is how you avoid
        them.

        This does NOT change batching or scheduling. Same width, same order, same
        engine, same sampling (the EOS boost/floor apply per request exactly as they
        do on the classic path). The only thing that moves is WHEN audio leaves the
        worker.

        Wire shape, per streamed row i:
            {type:'batch_chunk', i, seq, format:'pcm16', data, duration, sampleRate}
            ... in seq order from 0 ...
            {type:'batch_item',  i, streamed:true, duration, chunks, gapSec}  # no data
        A failed/cancelled row is the ordinary {type:'batch_item', i, message} and
        the client throws away whatever chunks it already had. Non-streamed rows in
        the same batch keep the exact per-item shape _emit_batch_item has always
        sent, so a mixed batch needs nothing new on the client.

        WHAT IS NOT DONE TO STREAMED AUDIO. finalize_audio trims the leading/trailing
        silence and peak-normalizes - both decisions about a whole waveform, and by
        the time the last chunk exists the first has already been PLAYED. So streamed
        chunks go out raw and unretouched. For the same reason a streamed row is
        never re-rendered: the engine logs the truncation guard's verdict and the
        audio stands.

        THE GAP IS NOT AUDIO HERE, WHICH IS WHY THIS ARM NEEDS NO SPECIAL CASE FOR
        IT ANY MORE. It used to be a flat 0.3 s appended by finalize_audio, and the
        only way to honour that on audio already in flight was one final all-silent
        chunk. The player inserts `classify_gap`'s answer for the row now, so the
        number rides the terminal record above and no chunk is invented for it.

        NOTHING HERE IS GUARDED. Higgs's streamed rows go
        through a bare `render_audio` - `v3_engine.render_rows_streaming` and
        `mlx_backend`'s two stream paths - with NO GuardPlan, no PaceTracker, no
        re-roll, no split ladder, no hole check, and no `guard` on the wire. The
        `generate_batch` door routes a `render_many`-capable engine to
        _emit_guarded_batch, but the `stream: true` test above returns before that
        test is ever reached, so nothing streamed is guarded.

        OWEN RULED ON IT (2026-09-13), and the ruling is that this stays: "streaming
        can stay unguarded. it needs speed over all else. i believe it's been
        unguarded this whole time. if it becomes a problem we can add a guard
        later." It has - the serve world has never guarded a Higgs row - so this is
        the status quo affirmed rather than a regression accepted.

        WHAT IT COSTS, written down because the ruling deserves its price next to
        it: every streamed Higgs row ships whatever the model produced. Listen with
        "buffer before playing" off, Crucible's streaming door (which sets
        `stream: true` on EVERY row, so the whole batch lands here), and the browser
        extension's zero-shot path all take takes the audiobook world would have
        caught - the seconds-long render of a thousand-character chunk, and the
        run-on that keeps going. The book path is unaffected: it goes through the
        render door, which sends no `stream` flag anywhere.

        So do not "fix" this by reaching for _emit_guarded_batch here. The absence
        is deliberate and priced. What is NOT acceptable is this docstring claiming
        a guard that does not run - a guard asserted in prose and absent in code is
        worse than an admitted absence, because it stops anyone looking.

        THREADING: on MLX the SNAC decode runs on the engine's decoder thread, so
        on_chunk (and possibly on_row) is called from a thread that is not this one.
        Every stdout write goes through send_response's lock, and the bookkeeping
        below is guarded so a row can be answered exactly once no matter which thread
        gets there first.
        """
        orph = self.orph

        state_lock = threading.Lock()
        emitted = set()          # positions in `items` already answered
        chunk_counts = {}        # position -> chunks emitted for that row so far

        def _claim(pos) -> bool:
            """Take the right to answer `pos`. False when someone already has."""
            with state_lock:
                if pos in emitted:
                    return False
                emitted.add(pos)
                return True

        try:
            # Per-item voice check first, exactly as the other batch paths do: a row
            # this engine/backend cannot serve fails ON ITS OWN and its neighbours
            # render normally.
            positions = []          # positions into `items` that will be rendered
            for pos, it in enumerate(items):
                try:
                    voice, rung, take = self._resolve_row(it)
                    # And the gap this row will state, refused HERE if it leads -
                    # before the engine has rendered anything - so a row this
                    # door cannot carry costs nothing and fails on its own.
                    row_gap_sec(it.get('text'))
                except Exception as e:
                    if _claim(pos):
                        send_response('batch_item', {'i': it.get('i'), 'message': str(e)})
                    continue
                cleaned = orph._clean_sentence_for_tts(
                    normalize_for_tts(it.get('text', ''), language))
                if not cleaned:
                    # Genuinely empty text -> tiny silence, the same contract every
                    # other path here keeps. Never streamed: there is nothing to wait
                    # for, so it is answered immediately and dropped from the batch.
                    if _claim(pos):
                        self._emit_batch_item(
                            it, np.zeros(int(active_samplerate() * 0.05), dtype=np.float32),
                            FOR_STREAM)
                    continue
                positions.append((pos, cleaned, voice, rung, take))

            if not positions:
                return

            texts = [c for _pos, c, _v, _s, _k in positions]
            voices = [v for _pos, _c, v, _s, _k in positions]
            # Each row's take-ladder rung, aligned to `texts`. Streaming is
            # UNGUARDED by ruling (see the docstring), so a rung here is simply
            # the numbers that row renders at - there is no ladder to climb,
            # and the client decided which rung it wanted.
            samplings = [rung for _pos, _c, _v, rung, _k in positions]
            # And its other half: which seed lane the row draws from. Same
            # reasoning - no ladder here, the client chose the take.
            takes = [take for _pos, _c, _v, _s, take in positions]
            # Row index (into `texts`) -> position into `items`, since the engine's
            # callbacks speak in row indices and the wire speaks in the caller's `i`.
            row_to_pos = [pos for pos, _c, _v, _s, _k in positions]
            stream_rows = {row for row, (pos, _c, _v, _s, _k) in enumerate(positions)
                           if items[pos].get('stream') is True}
            print(f'[narrator.serve] fast-start: streaming {len(stream_rows)} of '
                  f'{len(texts)} rows', file=sys.stderr)

            def on_chunk(row, seq, audio):
                """One sub-sentence payload of a streaming row. Called from the
                engine's decoder thread on MLX - see the THREADING note above."""
                pos = row_to_pos[row]
                if audio is None or len(audio) == 0:
                    return
                a = np.asarray(audio, dtype=np.float32).flatten()
                with state_lock:
                    if pos in emitted:
                        # The row was already answered (a failure, or the cancel
                        # sweep). Chunks after that point belong to nothing and the
                        # client has discarded the row - say so rather than emit an
                        # orphan the pool would have to reject.
                        print(f'[narrator.serve] dropping chunk seq={seq} for row '
                              f'[{items[pos].get("i")}] - already answered',
                              file=sys.stderr)
                        return
                    chunk_counts[pos] = chunk_counts.get(pos, 0) + 1
                send_response('batch_chunk', {
                    'i': items[pos].get('i'),
                    'seq': seq,
                    'format': 'pcm16',
                    'data': audio_to_pcm16_base64(a),
                    'duration': len(a) / active_samplerate(),
                    'sampleRate': active_samplerate(),
                })

            def on_row(row, audio):
                """A row retired. Streamed rows are CLOSED here (their audio already
                went out as chunks); every other row is emitted exactly as the
                classic batch path emits it."""
                pos = row_to_pos[row]
                streamed = row in stream_rows
                if not streamed:
                    # FOR_STREAM: fast start is a Listen door end to end - this
                    # is the read-ahead row behind the one being played, and the
                    # player joins it to its neighbours. Both arms state the gap
                    # on their terminal record and neither appends silence, so
                    # the two cannot double one another.
                    if _claim(pos):
                        self._emit_batch_item(
                            items[pos],
                            None if audio is None or len(audio) == 0
                            else finalize_audio(audio, FOR_STREAM),
                            FOR_STREAM)
                    return
                total = 0.0 if audio is None else len(audio) / active_samplerate()
                with state_lock:
                    sent = chunk_counts.get(pos, 0)
                if sent == 0 or total <= 0:
                    # Nothing was ever streamed: a failed row. Report it as a
                    # failure - never as a silent success the client would play as a
                    # missing sentence.
                    if _claim(pos):
                        send_response('batch_item',
                                      {'i': items[pos].get('i'), 'message': 'No audio generated'})
                    return
                # THE GAP IS A NUMBER ON THIS RECORD, not a chunk of silence.
                # It used to ride as one final all-silent chunk, because a flat
                # 0.3 s was the only part of `finalize_audio` that could still be
                # applied to audio already in flight; now the player inserts
                # `classify_gap`'s answer for the row itself, so there is nothing
                # left to send as audio. `duration` therefore measures the SPEECH
                # this row delivered - what the chunks add up to - and the gap
                # that follows it is stated separately.
                if _claim(pos):
                    send_response('batch_item', {
                        'i': items[pos].get('i'),
                        'streamed': True,
                        'duration': total,
                        'chunks': sent,
                        'gapSec': row_gap_sec(items[pos].get('text')),
                    })

            try:
                orph.generate_batch_stream(texts, voices, stream_rows,
                                           on_chunk, on_row, self._is_cancelled,
                                           samplings=samplings, takes=takes)
            except Exception as e:
                import traceback
                traceback.print_exc(file=sys.stderr)
                for pos, _c, _v, _s, _k in positions:
                    if _claim(pos):
                        send_response('batch_item',
                                      {'i': items[pos].get('i'),
                                       'message': f'Batch generation failed: {e}'})
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            for pos, it in enumerate(items):
                if _claim(pos):
                    send_response('batch_item',
                                  {'i': it.get('i'), 'message': f'Batch generation failed: {e}'})
        finally:
            # Same one-answer-per-item guarantee the other two batch paths keep: the
            # pool resolves a batch by index, and a row with no message hangs its
            # sentence until the 180s timeout taints the worker.
            cancelled = self._is_cancelled()
            for pos, it in enumerate(items):
                if _claim(pos):
                    send_response('batch_item', {
                        'i': it.get('i'),
                        'message': 'cancelled' if cancelled else 'No audio generated',
                    })
            send_response('batch_done', {'count': len(items)})

    def _refuse_batch(self, items, message: str) -> None:
        """One refusal, answered on EVERY row of the batch, then `batch_done`.

        The shape this door has always refused a whole batch in ("Model not
        loaded", a language num2words cannot speak): the caller gets a
        `batch_item` for every `i` it sent and then `batch_done`, because that
        is this method's contract and a caller waiting on 400 rows must not be
        left waiting on one.
        """
        for it in items:
            send_response('batch_item', {'i': it.get('i'), 'message': message})
        send_response('batch_done', {'count': len(items)})

    def generate_batch(self, items, language: str = 'en', retake=None,
                       band=None, width=None):
        """Generate a batch of sentences (read-ahead). Emits one 'batch_item' per
        item (keyed by its caller-supplied index `i`) then a 'batch_done'. A failure
        is reported per item so one bad sentence never sinks the batch.

        An item may carry its own `voice`; omitted means the loaded one, and any
        other is refused for that row (`_reject_per_request_voice`).

        An item may also carry its own `sampling` - the take-ladder rung
        Crucible resolved for it. ABSENT is the voice's loaded sampling (take
        0), which is the existing behaviour and the documented meaning of "no
        rung", not a default substituted for a missing value. A BATCH MAY MIX
        RUNGS: every engine here either renders row by row (so mixing is free)
        or splits its slab by sampling group (the MLX arm - one slab samples at
        one temperature, so `_mlx_batch_groups` breaks a group when the rung
        changes). No path renders a row at another row's numbers.

        An item may also carry `stream: true` (fast start, Owen 2026-09-04), which
        routes the WHOLE batch through _generate_batch_streaming so those rows emit
        sub-sentence 'batch_chunk's as they generate. The test is deliberately "does
        ANY item ask for it": a batch mixes the block being listened to with
        read-ahead behind it, and only the former streams, so the streaming path has
        to be able to carry both. With no stream flag anywhere - which is what the
        extension's default "Buffer before playing" produces - nothing below this
        line is reached and the batch takes the code that was already here.

        `retake` and `band` are BATCH-level and decide whether anything JUDGES
        these rows (Owen, 2026-09-19 - see `parse_retake` and `batch_tracker`).
        `retake: true` takes `_emit_guarded_batch`, the engine's own ladder,
        judged against the band THIS BATCH carried, and its items come back
        with `guard` - the verdict the ladder reached. `retake: false` or
        absent takes the sequential dispatch below: one take per row, rendered
        exactly as sent, and no `guard` key at all, because nobody judged it.
        Both arms measure (`capped`).

        THIS IS THE NON-STREAMING DOOR ONLY, which is the one Crucible's render
        job drives; the interactive Listen paths (`generate`,
        `_generate_batch_streaming`) judge nothing and are deliberately left
        alone, because a retake doubles the latency a listener is already
        waiting on and whether they should pay it is Owen's call, not this
        change's.

        `width` is also batch-level: how many rows narrator may keep in flight
        for THIS batch, at most the width the engine was started with
        (`parse_width`). Absent means that serving width."""
        if self.orph is None:
            self._refuse_batch(items, 'Model not loaded')
            return

        # ONE language check for the whole batch, before any audio. Without it a
        # language num2words cannot speak fails every digit-bearing row of the
        # batch separately with the same message - N refusals for one cause.
        # Reported per item, because that is this method's contract: the caller
        # gets a batch_item for every i it sent, then batch_done.
        try:
            check_language(language)
        except ValueError as exc:
            self._refuse_batch(items, str(exc))
            return

        # WHETHER ANYTHING JUDGES THIS BATCH, decided once, before a row
        # renders. A refusal here is the whole batch's: the flag and the band
        # are the batch's own, so answering some rows and refusing others would
        # be two answers to one question.
        streaming = any(it.get('stream') is True for it in items)
        try:
            retake = parse_retake(retake)
            if retake and streaming:
                raise BatchRefused(
                    'retake_with_stream',
                    'a streaming batch cannot be judged: `_generate_batch_streaming` '
                    'emits a row as it generates, so there is nothing left to '
                    're-roll by the time the ladder could decide. Send the rows '
                    'that must be judged without `stream`.')
            tracker = batch_tracker(retake, band, self.orph)
            width = parse_width(width, self.orph)
        except BatchRefused as refused:
            self._refuse_batch(items, str(refused))
            return

        if streaming:
            self._generate_batch_streaming(items, language)
            return

        emitted = set()
        try:
            # Resolve each item's voice FIRST and fail only the items that name one
            # this engine (or this backend) cannot serve. A single unservable voice
            # must not sink the batch: the rest are ordinary sentences in a voice that
            # is right there.
            # WHICH DOOR THIS BATCH IS - who follows the rows, an assembler
            # or a player. It is the ENGINE's answer and NOT `retake`'s: a
            # Higgs batch on this path is Crucible's `tts` render job whether
            # or not it asked to be judged (an assembler follows it, and
            # `gaps.json` owns the silence); an engine with no `render_many`
            # would be a player's (told `gapSec`), and no Higgs arm is one.
            # Judging and assembling are two questions; this is the second one.
            door = FOR_RENDER if _can_guard_its_own_batch(self.orph) else FOR_STREAM
            rows = []   # (item, normalized text, voice token, sampling, take)
            for it in items:
                try:
                    v, rung, take = self._resolve_row(it)
                    if door == FOR_RENDER:
                        # THE INDEX SEEDS THE RENDER behind this door, on both
                        # arms (`seed + i`), so an unusable one cannot be
                        # carried along and sorted out by the client - it would
                        # silently render at some other chunk's draw. The
                        # guarded arm refuses it at its own door too; this
                        # catches it one step earlier and for both arms.
                        try:
                            int(it.get('i'))
                        except (TypeError, ValueError):
                            raise ValueError(
                                f'generate_batch row carries i={it.get("i")!r}, '
                                'which is not an integer index. A render uses `i` '
                                'to seed the chunk, so there is nothing to render '
                                'this row as.') from None
                    if door == FOR_STREAM:
                        # Refused before anything renders: `row_gap_sec` rejects a
                        # row whose gap LEADS, which this door cannot carry, and a
                        # refusal raised at emit time would abandon its neighbours.
                        row_gap_sec(it.get('text'))
                except Exception as e:
                    send_response('batch_item', {'i': it.get('i'), 'message': str(e)})
                    emitted.add(id(it))
                    continue
                rows.append((it, normalize_for_tts(it.get('text', ''), language),
                             v, rung, take))

            if rows and _can_guard_its_own_batch(self.orph):
                # THE ENGINE'S OWN BATCHED DRIVER, judged or not (Owen,
                # 2026-09-13; asked for per batch since 2026-09-19; unjudged
                # batches routed here 2026-09-20).
                #
                # `tracker` carries the whole of the difference. With one, the
                # engine runs its PaceTracker, re-roll and split ladder against
                # the band THIS batch carried and yields each chunk with the
                # verdict it reached. With None - the batch said `retake:
                # false` - it builds a `truncation.UnjudgedPlan` instead: one
                # take per row, exactly as sent, `verdict` None for every chunk,
                # which `_emit_batch_item` already turns into a row with no
                # `guard` key at all.
                #
                # WHY BOTH ARE HERE NOW. Until 2026-09-20 `retake: false` fell
                # through to the sequential comprehension in
                # `_generate_audio_batch`, so ONE flag decided two unrelated
                # things: whether the batch was judged, and whether it was
                # batched at all. The caller that must send `retake: false` is
                # the screening render of a fine-tuned checkpoint - no measured
                # band, because measuring one is what the render is for - which
                # is also the caller rendering a thousand chunks for
                # throughput. Measured on the training PC: `#running-req` pinned
                # at 1 across all 684 sampled scheduler lines at width 4, 12.9
                # s/chunk against 4.16 s/render on the direct path, with sglang
                # demonstrably willing (zero rejections at concurrency 4).
                #
                # The width is spent by the ENGINE either way, which is why the
                # fix is a plan and not a thread pool here: `v3_engine` pools
                # threads, `mlx_backend` groups into a BatchGenerator, and a
                # pool in this method would be right for one and wrong for the
                # other.
                self._emit_guarded_batch(rows, emitted, tracker, width)
            elif rows:
                # AN ENGINE WITH NO BATCHED DRIVER. One take per row, exactly
                # as sent, no verdict - there is no `render_many` to ask. narrator
                # sent Orpheus's Listen batches here; a HIGGS BATCH DOES NOT
                # ARRIVE HERE (both arms offer `render_many`), and this arm is
                # kept so an engine without one is rendered honestly rather than
                # refused for a method it was never asked to have.
                rendered = self._generate_audio_batch(
                    [t for _, t, _, _, _ in rows],
                    [v for _, _, v, _, _ in rows],
                    [g for _, _, _, g, _ in rows],
                    [k for _, _, _, _, k in rows],
                    # THE CALLER'S INDEX SEEDS THE ROW on the render door -
                    # see `_generate_audio_batch`. On the Listen door the `i`
                    # is a label the player resolves by, and the position is
                    # what that door has always rendered at.
                    indexes=([int(it.get('i')) for it, _t, _v, _g, _k in rows]
                             if door == FOR_RENDER else None),
                    door=door)
                for (it, _text, _v, _g, _k), (audio, measure) in zip(rows, rendered):
                    emitted.add(id(it))
                    # Through the ONE emitter, which is also where `gapSec` and
                    # the measurement are attached. This arm used to write the
                    # same five fields out itself - a second copy of the
                    # per-item shape, which is how a field lands on three arms
                    # and not on the fourth.
                    self._emit_batch_item(it, audio, door, measure=measure)
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            # Only items that never got a result - an item already reported (its own
            # voice rejection, or a successful render before the failure) must not be
            # emitted twice; the pool resolves by index and a second message for one
            # index would be dropped as stale.
            for it in items:
                if id(it) not in emitted:
                    # RECORDED, NOT ONLY SENT. Without this the `finally` below
                    # sees the row as unanswered and emits a SECOND batch_item
                    # for it, and the caller refuses the pair by name:
                    # "narrator answered row N twice. One answer per row is
                    # narrator's own guarantee" (crucible jobs/tts/render.py).
                    #
                    # MEASURED 2026-09-15 on Owen's 3090 Ti: a real mid-batch
                    # exception became a `narrator_protocol` failure, and the
                    # duplicate MASKED the true cause — this message carries the
                    # exception, the `finally`'s says only "No audio generated",
                    # and the caller failed on the pair before either was read.
                    # So the defect cost the diagnosis as well as the batch.
                    #
                    # The comment above this block already stated the rule; only
                    # the line that keeps it was missing.
                    emitted.add(id(it))
                    send_response('batch_item', {'i': it.get('i'), 'message': f'Batch generation failed: {e}'})
        finally:
            # Same one-answer-per-item guarantee as the MLX path. vLLM's
            # engine.generate() is one blocking call with no abort hook, so a cancel
            # cannot shorten a vLLM batch - it can only label the rows that never made
            # it. (Nothing here pretends otherwise: the flag is read, not waited on.)
            cancelled = self._is_cancelled()
            for it in items:
                if id(it) in emitted:
                    continue
                emitted.add(id(it))
                send_response('batch_item', {
                    'i': it.get('i'),
                    'message': 'cancelled' if cancelled else 'No audio generated',
                })
            send_response('batch_done', {'count': len(items)})

    def generate(self, text: str, language: str = 'en', stream: bool = False,
                 voice: str = None, sampling=None, take=None, **_ignored):
        """Render ONE sentence. `voice` (optional) must be a voice a 'load'
        registered against the live engine; omitted means the loaded voice.

        `sampling` (optional) is this render's take-ladder rung, in the voices
        document's spelling, and `take` (optional) is which rung - the seed
        lane. Both go through the ENGINE's own `accept_item_sampling` /
        `accept_item_take`, so the refusals are the same ones a batch row gets
        - here they land as the `error` message this door speaks in."""
        if self.orph is None:
            send_response('error', {'message': 'Model not loaded'})
            return
        try:
            check_language(language)
            rung = self._item_rung(sampling, 'generate')
            rung_take = self._item_take(take, 'generate')
            raw_text = text
            text = normalize_for_tts(text, language)
            audio, _measure = self._generate_audio(text, voice, sampling=rung,
                                                   take=rung_take,
                                                   door=FOR_STREAM)
            if audio is None or len(audio) == 0:
                send_response('error', {'message': 'No audio generated'})
                return
            duration = len(audio) / active_samplerate()
            data = audio_to_pcm16_base64(audio)
            # ONE LISTEN SENTENCE, so the Listen contract in full: bare speech,
            # and the gap the player must insert after it stated on the row's
            # TERMINAL record - `done` when this is a stream, `audio` when it is
            # not. Computed from the text as it arrived, before `normalize_for_tts`
            # rewrote its numbers (see `row_gap_sec`).
            gap_sec = row_gap_sec(raw_text)
            if stream:
                # Whole sentence as a single chunk, then the stream terminator -
                # satisfies the scheduler's streaming-first-sentence contract.
                send_response('chunk', {
                    'seq': 0,
                    'format': 'pcm16',
                    'data': data,
                    'duration': duration,
                    'sampleRate': active_samplerate(),
                })
                send_response('done', {
                    'duration': duration,
                    'chunks': 1,
                    'cancelled': False,
                    'gapSec': gap_sec,
                })
            else:
                send_response('audio', {
                    'format': 'pcm16',
                    'data': data,
                    'duration': duration,
                    'sampleRate': active_samplerate(),
                    'gapSec': gap_sec,
                })
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            send_response('error', {'message': f'Generation failed: {e}'})

    def _read_stdin(self, inbox: "queue.Queue"):
        """Read stdin on its OWN thread and hand each request to the main loop.

        The main loop used to BE the stdin loop (`for raw in sys.stdin`), which meant
        a message could only be seen between requests - and a read-ahead batch is
        30-43s long. So a 'cancel' sent the instant the listener switched voice was
        not read until the batch it was meant to abort had already finished, and the
        new voice's load queued behind renders nobody would ever hear. A reader thread
        is the smallest thing that fixes that: it OBSERVES the cancel while the main
        thread is inside the engine, sets the flag the generator polls, and still puts
        the request on the queue so the main loop acknowledges it in arrival order.

        Everything else stays exactly as it was - one request executed at a time, all
        stdout writes from the main thread, so the JSON-lines protocol keeps both of
        its ordering guarantees (a request's messages are contiguous, and 'batch_done'
        is the last of its batch).
        """
        try:
            for raw in sys.stdin:
                line = raw.strip()
                if not line:
                    continue
                try:
                    request = json.loads(line)
                except Exception as e:
                    inbox.put(('bad-json', str(e)))
                    continue
                # THE POINT OF THE THREAD: flip the flag here, not when the main loop
                # gets around to this message. A generate_batch already running sees
                # it at its next decode step.
                if request.get('action') in ('cancel', 'stop'):
                    self._cancel.set()
                inbox.put(('request', request))
        except Exception as e:
            print(f'[narrator.serve] stdin reader failed: {e}', file=sys.stderr, flush=True)
        finally:
            inbox.put(('eof', None))

    def run(self):
        self.device = detect_device()
        # NOT guarded: a failure here means no engine can ever load in this
        # process, and `main()` turns it into a non-zero exit with the reason on
        # stderr. A handshake would be a lie. (See detect_backend.)
        self.backend = detect_backend()
        # `backend` is what the pool gates its per-request-voice guard on, so it is
        # sent even when it could not be determined - as absent, which the pool reads
        # as "unknown" and treats as NOT capable.
        _warn_fake_engine('ready')
        # `itemTake` IS THE CHANNEL SAYING IT EXISTS, and it is a BUILD fact,
        # not an engine one. Crucible resolves a take ladder's rung and sends
        # BOTH its halves on every `generate_batch` item - the numbers as
        # `sampling` and the rung's own number as `take`; a narrator built
        # before `engine/item_sampling.py` has neither channel and DROPS both
        # keys in silence - `_resolve_row` there read only `item['voice']` - so
        # take 1 rendered at take 0's sampling, in take 0's seed lane, and was
        # reported as a successful take 1. That is the shape
        # crucible/docs/ARCHITECTURE.md's audit names: a fact with two owners
        # (the recipe's narrator pin and Crucible's belief about it) and
        # nothing comparing them. Measured 2026-09-15: two renders of one
        # sentence at take 0 and take 1 came back byte-identical because the
        # env's pinned narrator predated the channel. So the handshake carries
        # the fact, and Crucible refuses a rung by name when it is absent.
        #
        # ONE KEY FOR THE TWO FACTS, renamed from `itemSampling` the same day
        # (2026-09-15) the seed half landed. They are one channel: a build has
        # both or neither, they are parsed by one module, and they are refused
        # under one `sampling_not_wired`. Two keys would be two owners of one
        # answer, which is the audit's own shape; `itemSampling` alone would be
        # a name that stopped covering what it announces. Nothing had shipped
        # under the old name - the tts recipes still pin a narrator older than
        # either half - so the rename costs a re-pin nobody has made yet.
        #
        # Sent at 'ready', which is BEFORE any engine loads, so it can only say
        # "this narrator parses a rung on an item" - never "this engine has
        # that lever, or that lane". Those stay where they belong:
        # `accept_item_sampling` / `accept_item_take`, per row, as
        # `sampling_not_supported` / `take_not_supported`.
        send_response('ready', {'device': self.device,
                                **({'backend': self.backend} if self.backend else {}),
                                'itemTake': True})

        inbox: "queue.Queue" = queue.Queue()
        reader = threading.Thread(target=self._read_stdin, args=(inbox,),
                                  name='narrator-stdin', daemon=True)
        reader.start()

        # EVERY WAY OUT RELEASES THE ENGINE: a stdin 'quit', stdin closing, and
        # SIGTERM/SIGINT (`_graceful_exit` raises SystemExit out of the loop).
        # Until 2026-09-27 a 'quit' just broke the loop and the process exited
        # without stopping a SERVED engine's server: only Orpheus has an atexit
        # hook, so a launched Higgs server was left to the setsid watchdog, which
        # TERMed it later and asynchronously while it still held its port
        # (Crucible 1.0.43, narrator 689446).
        try:
            self._serve_inbox(inbox)
        finally:
            self._release_engine_at_exit()

    def _release_engine_at_exit(self):
        """Stop the loaded engine and WAIT for it, as the process ends.

        `cleanup()` is the engine's own teardown: for a served Higgs engine it is
        `server.stop()`, which signals the server's group and waits for its port
        to go quiet. A failure is logged, never raised: the process is ending
        either way, and the exit must not mask whatever ended it."""
        if self.orph is None:
            return
        print(f'[narrator.serve] exiting: stopping the loaded engine '
            f'({type(self.orph).__name__})', file=sys.stderr, flush=True)
        try:
            self.orph.cleanup()
        except Exception as exc:
            print(f'[narrator.serve] WARNING: the engine\'s cleanup failed on exit '
                f'({type(exc).__name__}: {exc}); anything it launched is left to '
                'its watchdog.', file=sys.stderr, flush=True)
        else:
            print('[narrator.serve] engine stopped', file=sys.stderr, flush=True)
        self.orph = None

    def _serve_inbox(self, inbox):
        """The request loop, until 'quit' or stdin closes."""
        while True:
            kind, payload = inbox.get()
            if kind == 'eof':
                break
            if kind == 'bad-json':
                send_response('error', {'message': f'Invalid JSON: {payload}'})
                continue
            request = payload

            action = request.get('action')
            if action == 'load':
                # 'warm' says whether a FIRST load may spend time on discarded
                # warm-up renders. Read explicitly, and ABSENT MEANS TRUE: the app
                # and this worker ship together, but a message from anything older
                # must mean the behaviour that message was written against.
                warm = request.get('warm')
                warm = True if warm is None else bool(warm)
                # No default voice: narrator's was Orpheus's 'leah'. A load
                # without one is refused by the engine, by name.
                if self.load_voice(request.get('voice'), request.get('modelDir'),
                                   request.get('caps'), request.get('adapterDir'),
                                   request.get('baseDir'), request.get('id'), warm):
                    # The CONSTRUCTED engine's backend, not the pre-load probe's - same
                    # value in every normal case, but this one is ground truth, so a
                    # probe that failed at startup is corrected here rather than
                    # leaving the pool's guard permanently armed.
                    self.backend = self.orph.backend
                    _warn_fake_engine(f'loaded voice {self.current_voice!r}')
                    # ADDITIVE fields, so the existing pool reader is unaffected
                    # (it takes `voice` and `backend` and ignores the rest):
                    #   engine        which engine actually loaded
                    #   sampleRate    the rate this engine renders at, so a
                    #                 client does not assume 24 kHz forever
                    #   pads          whether chunk audio already contains its
                    #                 own silence (Orpheus true, Higgs false)
                    #   edgeFadeMs    the fade an ASSEMBLER must apply at each
                    #                 chunk edge before joining (Higgs 25, since
                    #                 a decoded edge sits near -30 dB and clicks)
                    #   serverLog     for a SERVED backend, the file its server's
                    #                 stdout+stderr go to - the stream the Higgs
                    #                 v3 sentinel-filter proof reads. None for
                    #                 every in-process engine.
                    send_response('loaded', {
                        'voice': self.current_voice,
                        'backend': self.backend,
                        'engine': getattr(self.orph, 'ENGINE_ID', None),
                        'sampleRate': active_samplerate(),
                        'pads': bool(getattr(self.orph, 'pads', True)),
                        'edgeFadeMs': _edge_fade_of(self.orph).as_manifest(),
                        'serverLog': _server_log_of(self.orph),
                    })
            elif action == 'generate':
                text = request.get('text', '')
                if not text:
                    send_response('error', {'message': 'No text provided'})
                    continue
                self.generate(
                    text=text,
                    language=request.get('language', 'en'),
                    stream=bool(request.get('stream', False)),
                    voice=request.get('voice'),
                    sampling=request.get('sampling'),
                    take=request.get('take'),
                )
            elif action == 'generate_batch':
                # `retake` and `band` are the BATCH's, not an item's: whether
                # narrator judges these rows, and what it judges them against,
                # are one decision for the whole call. Read with `.get` and
                # None-meaning-absent, then parsed - `parse_retake` refuses a
                # non-boolean rather than coercing it, and `batch_tracker`
                # refuses `retake` with no band by name.
                self.generate_batch(
                    request.get('items', []),
                    language=request.get('language', 'en'),
                    retake=request.get('retake'),
                    band=request.get('band'),
                    width=request.get('width'),
                )
            elif action in ('cancel', 'stop'):
                # The reader thread already SET the flag when this line arrived -
                # whatever was in flight has seen it. Reaching it here means that work
                # is over, so this is where the flag is cleared, and the only place:
                # the queue is FIFO and the reader only ever sets it, so a cancel can
                # never outlive the request it arrived during and can never suppress a
                # batch queued after it (that batch is behind this ack in the queue).
                self._cancel.clear()
                send_response('stopped')
            elif action == 'quit':
                break
            else:
                send_response('error', {'message': f'Unknown action: {action}'})


def main(argv=None):
    """`python -m crucible_higgs.serve [--fake-engine]`.

    The ONLY production writer of _FAKE_ENGINE. An unrecognised argument is a
    hard error: this worker takes no configuration on the command line (its
    interface is the protocol on stdin/stdout and the spawn env
    `crucible/engines/narrator.py` writes), so anything else here is a mistake
    worth surfacing before a model loads.

    Exit codes: 0 clean, 2 bad arguments, 3 the worker could not come up at all
    (and printed no `ready`).
    """
    global _FAKE_ENGINE
    # THIS WORKER'S STDOUT IS THE PROTOCOL. Engine log lines must never land on
    # it: one bare `print` between two JSON messages breaks the client's parse
    # (measured 2026-09-05 - a Higgs load banner arrived where a `loaded` line
    # was expected). stderr is `narrator.engine.log`'s default, and the pool
    # reads it (orpheus-worker-pool.ts logs `[Orpheus Pool stderr]`); this says
    # so explicitly rather than depending on a default staying put.
    # Imported HERE, not at module scope: `narrator.serve` sends its `ready`
    # line before any heavy import, and engine.log costs only `sys` but the
    # rule is worth keeping unbroken.
    from ..engine.log import set_log_stream
    set_log_stream(sys.stderr)
    argv = sys.argv[1:] if argv is None else list(argv)
    unknown = [a for a in argv if a != '--fake-engine']
    if unknown:
        print(f'[narrator.serve] unknown argument(s): {" ".join(unknown)}. '
              'Usage: python -m crucible_higgs.serve [--fake-engine]',
              file=sys.stderr, flush=True)
        return 2
    _FAKE_ENGINE = '--fake-engine' in argv
    if _FAKE_ENGINE:
        print(FAKE_ENGINE_BANNER, file=sys.stderr, flush=True)
    try:
        StreamServer().run()
    except Exception as e:
        # A WORKER THAT CANNOT RENDER MUST NOT HANDSHAKE. The two ways to get
        # here are an unservable NARRATOR_ENGINE and a backend detection that
        # raised - both mean no engine will ever load in this process. Exiting
        # non-zero with the reason on stderr is what the pool can act on; a
        # `ready` line followed by "Model not loaded" on every generate is a
        # worker that looks alive forever (found on the Mac, 2026-09-04).
        print(f'[narrator.serve] FATAL: {type(e).__name__}: {e}',
              file=sys.stderr, flush=True)
        print('[narrator.serve] no engine can load in this process; exiting '
              'without a handshake.', file=sys.stderr, flush=True)
        return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())
