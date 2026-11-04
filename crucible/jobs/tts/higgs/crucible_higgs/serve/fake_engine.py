"""A model-free stand-in for the Higgs v3 engine, so the WIRE PROTOCOL can be tested.

NOT a fallback and not a second implementation of the engine. Crucible parses a
JSON-lines protocol whose exact message sequence and field set are a contract,
and the only other way to exercise that contract is to load a model on a GPU.
This class produces deterministic sine-wave audio whose length is a function of
the text, which is enough to drive serve/worker.py's real reader thread, real
stdout lock, real per-item bookkeeping and real one-answer-per-row guarantees -
and, through the REAL `truncation` module, the real guard, retake ladder and
frame-cap arithmetic.

It is reachable ONLY through the explicit `--fake-engine` argv flag (see
serve.worker._engine_class / main). Deliberately NOT an environment variable: an
env switch is something a leaked variable could turn on by accident - and the
failure it produces is a whole book rendered as sine tones and reported as
success. An argv flag has to be typed by whoever wrote the command line, and the
worker prints a loud stderr banner at startup and at every 'ready'/'loaded' while
it is on.

Moved from narrator@dc05e7e7 `serve/fake_engine.py`, where the Higgs fake was a
subclass of an ORPHEUS-shaped fake built on Orpheus's prompt mixin and SNAC
geometry. That base is gone with Orpheus: this is the Higgs fake alone, its own
class, with Higgs's text cleaning (`engine/higgs/prompt.clean_text`, the call
both real arms make) and its own answers to the lifecycle members the base used
to supply. Everything the Higgs fake itself defined is unchanged.

The Higgs contract, with fake audio. Two things matter to an ASSEMBLER:

  pads = False        Higgs emits bare speech - no silence, no fade at either
                      end - so the manifest's gapBefore/gapAfter are live and
                      the assembler realizes them.
  edge_fade 10/25     the assembler fades each chunk edge (-30 dB -> about
                      -46 dB) before joining, asymmetrically.

It streams the way the real engines do: ONE whole-row chunk at retirement,
because a delay-pattern codec has no sound windowed decode.
"""
import json
import math
import os
import time

import numpy as np

from ..engine.higgs import truncation, v3_served
from ..engine.higgs.codec import FrameMeasure
from ..engine.higgs.prompt import clean_text
from ..engine.log import log
from ..engine.protocol import EdgeFade

SAMPLE_RATE = 24000

#: Enough frames for a chunk of any text to have a duration worth measuring.
MIN_FRAMES = 6

# Wall-clock a streamed row costs, so a 'cancel' written to stdin right after a
# generate_batch has somewhere to LAND. Without it the whole batch retires
# faster than the pipe round-trip and the cancel path is untestable - the fake
# would prove only that a batch nobody interrupted completes. 80 ms is far under
# any timeout in the protocol and only the streaming path pays it.
STREAM_ROW_SECONDS = 0.08


#: What the fake's construction logs. Named so a test can assert on it without
#: copying the string, and unlike any line a real engine writes.
FAKE_LOAD_MARKER = '[FAKE-ENGINE] loaded voice'

# Higgs geometry (engine/higgs/codec.py): 25 frames/s at 24 kHz is 960 samples
# per frame.
HIGGS_SAMPLES_PER_FRAME = 960
HIGGS_FRAMES_PER_CHAR = 0.34      # ~13.6 ms/char


class FakeHiggsEngineConfig:
    """The subset of the real config the worker passes. Kept as its own class so
    the fake never imports the real one (and so a missing field is a TypeError
    here rather than a silent None in the engine)."""

    #: THE REAL ENGINES' SEED, so this fake draws in the same lanes they do.
    #: `HiggsV3Config.seed` and `HiggsV3MlxConfig.seed` are both 1234 and the
    #: rule is `seed + index`, shifted by the take
    #: (`truncation.in_take_lane`). Left at None this fake would report every
    #: seed as null and a test could not tell take 1 from take 0 - which is
    #: exactly the defect the take lane exists to fix.
    seed = 1234

    def __init__(self, voice=None, model_dir=None, base_dir=None, adapter_dir=None,
                 caps=None, **extra):
        self.voice = voice
        self.model_dir = model_dir
        self.base_dir = base_dir
        self.adapter_dir = adapter_dir
        self.caps = caps or {}
        self.extra = extra


class FakeHiggsEngine:
    """Deterministic audio, real text handling, real guard arithmetic, no model."""

    ENGINE_ID = 'higgs-v3'
    SAMPLE_RATE = SAMPLE_RATE
    pads = False
    edge_fade = EdgeFade(10.0, 25.0)

    def __init__(self, config: FakeHiggsEngineConfig):
        self.config = config
        # A label the worker reports and reads; 'transformers' is one its
        # `detect_backend` knows, and it is neither 'mlx' (which REPLACES a voice
        # on a load) nor a serving stack.
        self.backend = 'transformers'
        self.voice = (config.voice or '').strip()
        #: Guard-steering bookkeeping (`_rate_for`). Empty on every path that is
        #: not a guarded batch, which is what makes the multiplier a no-op there.
        self._full_text = {}
        self._attempts = {}
        self.adapter_dir = config.adapter_dir
        # A REAL engine announces its load through the engine log, and that
        # announcement is exactly the thing that once corrupted the worker's JSON
        # stdout. The fake makes the same call through the same helper, so a
        # test of the log routing exercises the real routing.
        log(f'{FAKE_LOAD_MARKER} {self.voice}')

    # ---- lifecycle ----------------------------------------------------------

    @classmethod
    def detect_backend(cls) -> str:
        return 'transformers'

    @classmethod
    def resolve_load_voice(cls, voice, model_dir=None, adapter_dir=None,
                           base_dir=None) -> str:
        """A Higgs voice is a NAME in the voices document.

        The fake does not read the voice document (it has no clips to load), but
        it refuses the two fields Higgs has no meaning for, so a protocol test
        sees the same refusals the real engine gives.
        """
        if model_dir:
            raise ValueError(
                f'Higgs load carried modelDir={model_dir!r}. The served model is '
                "the launch script's argument, not a per-load field.")
        if base_dir:
            raise ValueError(
                f'Higgs load carried baseDir={base_dir!r}. Higgs has no '
                'shared-base + per-voice-adapter split.')
        name = (voice or '').strip()
        if not name:
            raise ValueError(
                'Higgs load has no voice name. The name selects an entry in the '
                'NARRATOR_HIGGS_VOICES document; there is no default.')
        return name

    def cleanup(self):
        self.voice = None

    def set_voice(self, voice: str, adapter_dir: str = None) -> None:
        """Both real arms' rule: the SAME voice is a no-op, any other is refused
        by name - a Higgs engine is one voice for its whole life."""
        want = (voice or '').strip()
        if want == self.voice and not adapter_dir:
            return
        raise ValueError(
            f"FakeHiggsEngine cannot switch voice in place: it holds '{self.voice}', "
            f"and a load for '{want}' needs a new worker.")

    def _apply_voice_caps(self, voice: str, caps: dict) -> None:
        """Both real arms' rule: an empty payload is the no-op it looks like,
        anything else is refused by name."""
        if caps:
            raise ValueError(
                f'Higgs v3 load for {voice!r} carried caps={sorted(caps)}. Those are '
                'ORPHEUS tuning caps and v3 implements none of them.')

    def _clean_sentence_for_tts(self, sentence: str) -> str:
        """The SML strip both real arms make (`engine/higgs/prompt.clean_text`)."""
        return clean_text(sentence)

    # ---- audio --------------------------------------------------------------

    @staticmethod
    def frames_for(text: str) -> int:
        return max(MIN_FRAMES, int(len(text) * HIGGS_FRAMES_PER_CHAR))

    @classmethod
    def audio_for(cls, text: str) -> np.ndarray:
        n = cls.frames_for(text) * HIGGS_SAMPLES_PER_FRAME
        t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
        return (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    #: How the guard is made to FIRE in a test, deterministically.
    #:
    #: `audio_for` is exactly proportional to character count, so every chunk
    #: this fake renders sits at precisely the same chars/sec and the length
    #: guard can never fire - which would make a guarded test prove nothing.
    #: This env var bends one chunk's duration on purpose:
    #:
    #:     NARRATOR_FAKE_HIGGS_RATE='{"3": 0.4}'        chunk 3's take 0 is
    #:                                                  0.4x as long as its text
    #:                                                  implies; its RE-ROLL
    #:                                                  lands, so the ladder
    #:                                                  recovers at rung 2
    #:     NARRATOR_FAKE_HIGGS_RATE='{"3": [0.4, 0.4]}' take 0 AND the re-roll
    #:                                                  are bad, so it splits
    #:
    #: A list is indexed by attempt, with the last value repeating.
    #:
    #: TWO THINGS THIS GETS RIGHT THAT THE OBVIOUS VERSION DOES NOT.
    #:
    #: 1. THE ATTEMPT IS COUNTED, NOT READ OFF THE SEED. The first version of
    #:    this read `seed is None` as "take 0", which is wrong whenever the
    #:    engine is unseeded: `truncation.reroll_seed` returns None when
    #:    `base_seed` is None ("None stays None - an unseeded engine samples
    #:    fresh anyway, which IS the re-roll"), so every retake also looked like
    #:    take 0 and got bent again. Measured: a scalar 0.4 drove chunk 2 all
    #:    the way to `accepted-off-length` at MAX_DEPTH instead of recovering on
    #:    the re-roll it was written to exercise.
    #:
    #: 2. IT APPLIES ONLY TO THE TEXT THAT WAS SUBMITTED. A split half carries
    #:    its parent's `index`, so bending on the index alone would bend every
    #:    child too and no test could ever say "the split halves came back
    #:    fine". The table therefore applies only while the text is the one the
    #:    caller handed in for that index; a half renders at 1.0.
    RATE_ENV = 'NARRATOR_FAKE_HIGGS_RATE'

    #: Milliseconds of REAL time one `render_audio` costs. Default 0, because a
    #: fake that is slow by default makes every suite slow.
    #:
    #: It exists for exactly the reason `STREAM_ROW_SECONDS` above exists, one
    #: arm along: a 'cancel' written to stdin behind a `generate_batch` needs
    #: somewhere to LAND. The guarded arm retires a whole batch of sine waves in
    #: microseconds, so without this a cancel test on that arm would prove only
    #: that a batch nobody interrupted completes - which is what shipped, and
    #: what let `_emit_guarded_batch` reach production with no cancel check at
    #: all (see `serve/worker.py`, 2026-09-15).
    #:
    #: Paid per TAKE and not per chunk, deliberately: a re-roll and a split half
    #: are renders, and the worst case a cancel waits out is one take.
    ROW_MS_ENV = 'NARRATOR_FAKE_HIGGS_ROW_MS'

    #: THE FAKE'S OWN BAND, and it is not the real one - measured, not assumed.
    #:
    #: `audio_for` renders HIGGS_FRAMES_PER_CHAR = 0.34 frames per character at
    #: 960 samples per frame and 24 kHz, which is 73.53 chars/sec. The real band
    #: is 14.5-20.0 (`HiggsV3Defaults`). Seeding this fake with the real band
    #: would put EVERY chunk it renders outside it, so the guard would fire on
    #: all of them and a test asserting "the guard fired" would prove nothing.
    #:
    #: So the band is centred on the fake's own pace and carries the REAL band's
    #: RATIOS (about 1.17 either side), which makes the multiplier above mean
    #: the same thing here as it would on a card. Verified: 1.0 -> 73.53 (clean),
    #: 0.4 -> 183.82 (short), 2.0 -> 36.76 (long). Derived rather than written
    #: out, so it cannot drift from `frames_for`.
    _PACE = float(SAMPLE_RATE) / (HIGGS_FRAMES_PER_CHAR * HIGGS_SAMPLES_PER_FRAME)
    MAX_CHARS_PER_SEC = _PACE * math.sqrt(20.0 / 14.5)
    MIN_CHARS_PER_SEC = _PACE / math.sqrt(20.0 / 14.5)

    def _rate_table(self) -> dict:
        raw = (os.environ.get(self.RATE_ENV) or '').strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except ValueError as exc:
            # Loud, because a typo here silently turns a guard test into a test
            # that asserts the guard never fires - which passes.
            raise ValueError(
                f'{self.RATE_ENV} is not JSON ({exc}); it maps a chunk index to a '
                f'duration multiplier or a list of them, e.g. {{"3": [0.4, 1.0]}}'
            ) from None

    def _rate_for(self, index: int, text: str) -> float:
        """The duration multiplier for this render, and the bookkeeping that
        makes the NEXT one different. 1.0 unless a test said otherwise."""
        entry = self._rate_table().get(str(index))
        if entry is None:
            return 1.0
        if getattr(self, '_full_text', {}).get(index) != text:
            return 1.0           # a split half: never bent - see the note above
        attempt = self._attempts.get(index, 0)
        self._attempts[index] = attempt + 1
        if not isinstance(entry, list):
            return float(entry) if attempt == 0 else 1.0
        return float(entry[min(attempt, len(entry) - 1)])

    #: WHERE THE FAKE WRITES WHAT IT RENDERED EACH ITEM UNDER - BOTH HALVES
    #: OF THE RUNG.
    #:
    #: The serve protocol tests drive a real worker SUBPROCESS, so an in-memory
    #: list is invisible to the assertions. When this variable names a path the
    #: fake appends one JSON object per render - `{"index", "text", "sampling",
    #: "take", "seed"}` - and the test reads the file back. `sampling` is the
    #: ENGINE-shaped dict (snake_case) the render actually ran at, or null for
    #: a chunk at take 0; `seed` is the number the render actually drew at,
    #: which is the only way to SEE a take lane at all, because a rung may move
    #: the seed and change nothing else. The assertion is on what was APPLIED,
    #: never on what was asked for.
    #:
    #: Unset = no file and no cost, which is every other test in the suite.
    RENDER_LOG_ENV = 'NARRATOR_FAKE_RENDER_LOG'

    def _record_render(self, index, text, sampling, take, seed) -> None:
        """One render's whole rung, remembered in this process and - when the
        test asked for a file - appended to it."""
        row = {'index': None if index is None else int(index),
               'text': text, 'sampling': sampling, 'take': int(take),
               'seed': None if seed is None else int(seed),
               # The WIDTH the guarded driver was called with, or null for a
               # render outside it (Listen, the bare arm). See `render_many`.
               'width': getattr(self, '_call_width', None)}
        if not hasattr(self, 'renders_seen'):
            self.renders_seen = []
        self.renders_seen.append(row)
        path = (os.environ.get(self.RENDER_LOG_ENV) or '').strip()
        if not path:
            return
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')

    #: The levers a rung may name here: Higgs v3's MLX arm's set, which is the
    #: narrower of the two real Higgs arms, so a test written against this fake
    #: cannot assert a lever one shipped backend refuses.
    ITEM_SAMPLING_LEVERS = ('temperature', 'topP', 'topK')

    def accept_item_sampling(self, raw, where: str = None):
        """One item's `sampling` off the wire -> the engine-shaped dict, or
        None. The REAL parser (`engine/item_sampling.py`), so the refusals a test
        sees here are the refusals a card would give."""
        from ..engine.item_sampling import parse_item_sampling
        return parse_item_sampling(raw, where or f'FakeHiggsEngine({self.voice!r})',
                                   levers=self.ITEM_SAMPLING_LEVERS)

    def accept_item_take(self, raw, where: str = None) -> int:
        """One item's `take` off the wire. The REAL parser
        (`engine/item_sampling.py:parse_item_take`), so a `take_malformed` seen
        here is the sentence a card would send."""
        from ..engine.item_sampling import parse_item_take
        return parse_item_take(raw, where or f'FakeHiggsEngine({self.voice!r})')

    def _seed_for(self, index: int):
        """The real arms' seed rule, `seed + index` - `HiggsV3Engine._seed_for`."""
        seed = getattr(self.config, 'seed', None)
        return None if seed is None else int(seed) + int(index)

    def _request_seed(self, seed, index: int, take: int):
        """The real arms' `_request_seed`, word for word: the ladder's seed
        when it chose one, `seed + index` when it did not, shifted into take
        `take`'s lane either way."""
        base = self._seed_for(index) if seed is None else seed
        return truncation.in_take_lane(base, take)

    def _sampling_for(self, item_sampling):
        """The rung laid over the voice's own sampling. This fake has no
        generation_config to resolve, so take 0 is None and a rung is itself -
        which is exactly what makes the recorded value readable as "what this
        item rendered under"."""
        return None if item_sampling is None else dict(item_sampling)

    def cap_frames(self, text: str) -> int:
        """The per-chunk frame ceiling, by the REAL formula
        (`v3_served.cap_frames`) over the real text.

        The fake fakes the audio, never the arithmetic: `capped` is a fact
        about a render against its budget, and a fake cap would make every
        assertion about it a test of this file rather than of the engine
        contract. A test bends the AUDIO (`NARRATOR_FAKE_HIGGS_RATE`) and this
        ceiling answers honestly.
        """
        return v3_served.cap_frames(text)

    def measure_of(self, text: str, audio) -> FrameMeasure:
        """What one fake render spent, COUNTED: this engine generates in
        process, so the frames it produced are the frames it produced."""
        frames = 0 if audio is None else int(len(audio)) // HIGGS_SAMPLES_PER_FRAME
        return FrameMeasure.counted_frames(frames, self.cap_frames(text))

    def render_many(self, rows, in_flight=None, sampling_by_index=None,
                    take_by_index=None, *, tracker, width=None):
        """THE GUARDED DRIVER, serial - `(index, audio, verdict, measure)` per
        chunk.

        `tracker` is the band this call judges against, exactly as on both
        real engines (Owen, 2026-09-19): on the serve door it is built from the
        band the BATCH carried, and there is no other source for it. None means
        the batch asked for no judging (`retake: false`) and the rows render
        once, unjudged, through an `UnjudgedPlan` - the same arm the real
        engines take since 2026-09-20. This fake used to build its own from its class constants, which was
        the second owner the ruling removed.

        `width` is how many rows this call may keep in flight. This driver is
        SERIAL, so what it actually keeps in flight is one - inside any width a
        caller can ask for - and the number is written onto every render this
        call makes (`_record_render`'s `width`) rather than pretended about, so
        a test can assert that the door passed the job's width down to the
        driver that would honour it on a real engine.

        The same contract the real engines offer (`HiggsV3Engine.render_many`,
        `HiggsV3MlxEngine.render_many`) and the reason this fake can stand in
        for them in a test of the SERVE path: Owen ruled on 2026-09-13 that the
        model and its inference own the guard and the retake decision, so the
        thing under test is that a guarded chunk reaches the wire WITH its
        verdict. See crucible/docs/PHASE6-REMOTE-RENDER.md.

        ONE PLAN ACROSS THE WHOLE BATCH, not one per chunk, because the pace
        tracker re-centres on the chunks already shipped and a plan per chunk
        would throw that away between every sentence.

        Serial rather than pooled: this fake renders a sine wave, so there is
        nothing to overlap, and `truncation.py`'s header is explicit that the
        drivers differ in WHEN renders happen and never in what the ladder
        decides. `next_request()` one at a time is the depth-first order.
        """
        held = [] if in_flight is None else in_flight
        rungs = None if sampling_by_index is None else dict(sampling_by_index)
        takes = None if take_by_index is None else dict(take_by_index)

        def take_for(index):
            """The chunk's take, by CHUNK index - the rung's other half, and
            refused for an unnamed row exactly as the sampling is."""
            if takes is None:
                return 0
            if int(index) not in takes:
                raise KeyError(
                    f'FakeHiggsEngine.render_many: chunk {index} has no entry in '
                    'take_by_index.')
            return takes[int(index)]

        def rung_for(index):
            """The chunk's rung, by CHUNK index - a re-roll and a split half
            carry their parent's index and must keep its numbers. A row the map
            does not name is refused, never rendered at take 0."""
            if rungs is None:
                return None
            if int(index) not in rungs:
                raise KeyError(
                    f'FakeHiggsEngine.render_many: chunk {index} has no entry in '
                    'sampling_by_index.')
            return rungs[int(index)]

        # RECORDED, not pretended about - see the docstring. It rides on
        # every render this call makes so a test driving the worker as a
        # SUBPROCESS can read it back out of the render log.
        self._call_width = None if width is None else int(width)
        self._full_text = {}
        self._attempts = {}
        # THE SAME CHOICE BOTH REAL ENGINES MAKE (`HiggsV3Engine.render_many`,
        # `HiggsV3MlxEngine.render_many`, 2026-09-20): no tracker means the
        # batch asked for no judging, so it renders through an `UnjudgedPlan` -
        # one take per row, `verdict` None, no `guard` key on the wire. This
        # fake kept building a GuardPlan around a None tracker after that
        # ruling, so the serve tests of `retake: false` judged rows the real
        # engines would have shipped bare.
        base_seed = getattr(self.config, 'seed', None)
        plan = (truncation.UnjudgedPlan(sample_rate=self.SAMPLE_RATE, base_seed=base_seed)
                if tracker is None else
                truncation.GuardPlan(sample_rate=self.SAMPLE_RATE, base_seed=base_seed,
                                     tracker=tracker))
        for index, text in rows:
            self._full_text[int(index)] = text
            rung_for(index)     # refuse an unnamed row at the door, not mid-ladder
            take_for(index)
            plan.add(int(index), text)
            while True:
                request = plan.next_request()
                if request is None:
                    break
                if request.index not in held:
                    held.append(request.index)
                take = self.render_audio(
                    request.text, seed=request.seed, index=request.index,
                    sampling=rung_for(request.index),
                    take=take_for(request.index))
                plan.offer(request, take,
                           self.measure_of(request.text, take))
            for done_index, audio, _clean in plan.finished():
                if done_index in held:
                    held.remove(done_index)
                yield (done_index, audio, plan.verdict(done_index),
                       plan.measure(done_index))

    def render_audio_measured(self, text: str, seed=None, index: int = 0,
                              should_stop=None, sampling=None, take: int = 0):
        """`render_audio` with the `FrameMeasure` it spent, as BOTH real Higgs
        engines offer it: `(audio, measure)`.

        This is the door the worker's BARE arm calls, and it is why the fake
        has one - the measurement rides both arms (Owen, 2026-09-19), and a
        fake that only had the one-value face would let a test of `capped` on
        the unguarded arm pass by reporting nothing at all.
        """
        audio = self.render_audio(text, seed=seed, index=index,
                                  should_stop=should_stop, sampling=sampling,
                                  take=take)
        return audio, self.measure_of(text, audio)

    def render_audio(self, text: str, seed=None, index: int = 0,
                     should_stop=None, sampling=None, take: int = 0) -> np.ndarray:
        """ONE chunk in, one waveform out - the per-chunk entry point both real
        Higgs arms have (`HiggsV3Engine`, `HiggsV3MlxEngine`).

        This is the method under test: every render the worker makes goes
        through it, on both batch arms, the stream arm and `generate`.
        """
        if not (text or '').strip():
            raise ValueError('FakeHiggsEngine.render_audio(): the chunk has no text')
        # RECORDED BEFORE THE RENDER, and recorded for EVERY take: a split half
        # and a re-roll are renders too, and a test asserting "chunk 3 ran at
        # 0.7 in take 2's lane" has to be able to see that all of its takes did.
        self._record_render(index, text, self._sampling_for(sampling), take,
                            self._request_seed(seed, index, take))
        row_ms = (os.environ.get(self.ROW_MS_ENV) or '').strip()
        if row_ms:
            time.sleep(float(row_ms) / 1000.0)
        # THE FIRST TEXT SEEN FOR AN INDEX IS THAT CHUNK'S OWN, whichever arm
        # rendered it. `render_many` records them up front (that is how a split
        # HALF is told from its parent and never bent), but the BARE arm has no
        # ladder to record anything, and without this line a test of the
        # unguarded arm could not bend a duration at all - it would assert
        # against audio the knob never touched.
        if not hasattr(self, '_full_text'):
            self._full_text, self._attempts = {}, {}
        self._full_text.setdefault(index, text)
        audio = self.audio_for(text)
        rate = self._rate_for(index, text)
        if rate == 1.0:
            return audio
        # Fewer samples for the same text = a SHORT take (the model stopped
        # early); more = a LONG one. The waveform is resampled by slicing or
        # tiling rather than regenerated, so the only thing a test changes is
        # the duration the guard measures.
        wanted = max(1, int(round(len(audio) * rate)))
        if wanted <= len(audio):
            return audio[:wanted]
        repeats = int(np.ceil(wanted / len(audio)))
        return np.tile(audio, repeats)[:wanted]

    def generate_batch_stream(self, texts, voices, stream_rows, on_chunk, on_row,
                              should_stop=None, samplings=None, takes=None) -> None:
        """Whole rows, at retirement - the real Higgs cadence.

        `samplings`, when given, is aligned to `texts`: this fake renders row
        by row, so a batch may mix rungs and each row is recorded under its
        own. `takes` is aligned the same way and carries each row's seed
        lane."""
        from ..engine.item_sampling import aligned, takes_aligned
        if not texts:
            return
        if voices is not None and len(voices) != len(texts):
            raise ValueError(
                f'FakeHiggsEngine.generate_batch_stream: {len(voices)} voices for '
                f'{len(texts)} texts; voices must be aligned to texts or None')
        stream_rows = set() if stream_rows is None else set(stream_rows)
        stray = [i for i in stream_rows if not (0 <= i < len(texts))]
        if stray:
            raise ValueError(
                f'FakeHiggsEngine.generate_batch_stream: stream_rows names row(s) '
                f'{stray} outside the batch of {len(texts)}')
        if stream_rows and on_chunk is None:
            raise ValueError(
                'FakeHiggsEngine.generate_batch_stream: stream_rows is non-empty but '
                'no on_chunk was given')
        blank = [i for i, t in enumerate(texts) if not (t or '').strip()]
        if blank:
            raise ValueError(
                f'FakeHiggsEngine.generate_batch_stream: row(s) {blank} have no text '
                'after cleaning.')
        rungs = aligned(samplings, len(texts),
                        'FakeHiggsEngine.generate_batch_stream')
        rows_takes = takes_aligned(takes, len(texts),
                                   'FakeHiggsEngine.generate_batch_stream')

        for i, text in enumerate(texts):
            if should_stop is not None and should_stop():
                return
            time.sleep(STREAM_ROW_SECONDS)
            self._record_render(i, text, self._sampling_for(rungs[i]),
                                rows_takes[i],
                                self._request_seed(None, i, rows_takes[i]))
            audio = self.audio_for(text)
            if i in stream_rows:
                on_chunk(i, 0, audio.copy())
            on_row(i, audio)


# engine id -> (fake engine class, fake config class). Mirrors
# engine/registry.py, so `--fake-engine` exercises the same selection the real
# spawn makes, and refuses an id the registry would refuse.
FAKE_ENGINES = {
    'higgs-v3': (FakeHiggsEngine, FakeHiggsEngineConfig),
}


def _fake_entry(engine_id: str):
    if engine_id not in FAKE_ENGINES:
        raise ValueError(
            f"--fake-engine has no stand-in for engine '{engine_id}'. Known: "
            f"{', '.join(sorted(FAKE_ENGINES))}.")
    return FAKE_ENGINES[engine_id]


def fake_engine_class(engine_id: str):
    """The fake for `engine_id`."""
    return _fake_entry(engine_id)[0]


def fake_engine_config(engine_id: str, **kwargs):
    return _fake_entry(engine_id)[1](**kwargs)
