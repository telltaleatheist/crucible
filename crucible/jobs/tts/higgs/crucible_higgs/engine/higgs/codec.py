"""The Higgs audio codec: 8 codebooks at 25 fps, a delay pattern, 24 kHz.

Geometry, measured 2026-09-04 (`HIGGS_NOTES.md` section A, campaign
`2026-09-01-cod-full-rebuild/higgs`) and IDENTICAL in v2 and v3:

    8 codebooks, codebook_size 1024, audio vocab 1026 (1024 codes + stream
    bos 1024 / eos 1025); 25 frames per second (40 ms per frame); output
    24 000 Hz mono float32. 24000 / 25 = 960 samples per frame.

FROM THE MODEL'S OUTPUT TO A WAVEFORM. `model.generate` returns the WHOLE audio
stream as a (batch, seq, 8) matrix - the reference clips' frames and the newly
generated ones together. The steps below reproduce transformers'
`processing_higgs_audio_v2.batch_decode`, enumerated in HIGGS_NOTES.md, with one
addition marked (BF):

  1. find the LAST row whose 8 entries all equal 1024 (stream BOS): that is where
     generation starts, so everything before it - the references - is dropped.
  2. find the FIRST row after it whose 8 entries all equal 1025 (stream EOS):
     that is the end. No EOS -> the end of the sequence.
  3. slice (bos, eos): THE BOS ROW ITSELF IS DROPPED.
  4. revert the delay pattern: codebook c is delayed by c positions, so audio
     frame t of codebook c sits at stream position t + c. Reverting takes the
     diagonals and leaves `frames - (8 - 1)` = `frames - 7` frames. This is the
     only trim the shipped code does.
  4a. (BF, measured 2026-09-04) DROP THE TRAILING SENTINEL RUN, BY CONTENT. The
     model's ramp-down BOC/EOC sentinels smear over the last 7 frames because of
     the same delay; step 5 then maps them to code 0, WHICH IS A VALID CODE THAT
     DECODES TO SOUND. Heard as "a stray syllable after each sentence" - a
     100-160 ms burst rising from the noise floor to about -30 dB. vllm-omni
     defines `_filter_real_code_frames()` for exactly this and never calls it,
     and trims one frame (40 ms) where ~7 are contaminated. Trimming by CONTENT
     (drop trailing frames holding any out-of-range code) removes exactly the
     contaminated tail and cannot eat real speech the way a blind 7-frame trim
     could. It must run BEFORE step 5, or the sentinels are already 0 and
     invisible.
  5. clip to [0, 1023] so any stray sentinel is a legal codebook index.
  6. hand (1, 8, frames) to the audio tokenizer's decoder -> 1-D float32 at
     24 kHz.

Even after 4a a chunk edge sits near -30 dB and clicks on a join, which is what
`HiggsEngine.edge_fade` - EdgeFade(10 in, 25 out) - is for: the ASSEMBLER fades
it to -45..-48 dB.
No pads, no fades and no trimming are applied at either end here - Higgs emits
bare speech and the gaps are the assembler's (`Engine.pads` is False).

Steps 1-5 are pure numpy and are unit-tested with a FAKE decoder; only step 6
needs the model.
"""
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .prompt import AUDIO_STREAM_BOS_ID, AUDIO_STREAM_EOS_ID, NUM_CODEBOOKS, NUM_REAL_CODES

SAMPLE_RATE = 24000
FRAMES_PER_SECOND = 25.0
SAMPLES_PER_FRAME = int(SAMPLE_RATE / FRAMES_PER_SECOND)   # 960
# The delay pattern's diagonal: reverting it costs num_codebooks - 1 frames.
DELAY_TRIM_FRAMES = NUM_CODEBOOKS - 1                      # 7


@dataclass(frozen=True)
class FrameMeasure:
    """What one render SPENT, against the ceiling it was given.

    THE MEASUREMENT IS UNCONDITIONAL (Owen's ruling, 2026-09-19): a row that
    nobody judged is still a row somebody measured, so this travels on both
    arms of `serve/worker.py`'s `generate_batch` - the guarded one and the bare
    one - and reaches the wire as `capped` (and `tokens`, when the count is a
    count). It is a FACT about a render, never a verdict on it: `capped` says
    the generation stopped because it ran out of budget rather than because the
    model was finished, which is the difference between a chunk that ended and
    a chunk that was cut off. Crucible publishes both and reads `null` as
    "narrator did not say" - never as `false` (crucible/docs/PHASE18-UNCERTIFIED.md).

      frames   the audio frames this measure accounts for.
      cap      the per-chunk frame ceiling the render was sized at, which is
               `HiggsBudget.cap_frames(text)` clamped by the stack
               (`sgl_served.frame_cap`). The engine's own arithmetic over the
               text length and `CHARS_PER_SEC` - not the voice's `maxChars`.
      capped   `frames` reached `cap`.
      counted  whether `frames` is COUNTED from the model's own output (the
               MLX arm, which generates in this process and can see the matrix)
               or INFERRED from the decoded audio (the served arms, where the
               only thing that crosses the HTTP boundary is a WAV). An inferred
               count is a LOWER BOUND - the delay trim and the sentinel run are
               already gone from it - so only a counted one is reported as
               `tokens`.

    WHY AN INFERRED `capped` IS STILL HONEST. A generation that hit the cap
    emitted no EOS, so its tail carries no sentinel run for step 4a to strip
    and the delay trim (`DELAY_TRIM_FRAMES`, 7) is the whole of what was
    removed: the decoded audio is exactly `cap - 7` frames. A generation that
    stopped on EOS is nowhere near - `cap_frames` is 1.8-2.0x the expected
    duration plus 100-150 frames of slack, measured never to be reached across
    the nine-chunk audition. So `frames + DELAY_TRIM_FRAMES >= cap` separates
    the two with frames to spare; `from_audio` is where that rule lives, once.
    """
    frames: int
    cap: int
    capped: bool
    counted: bool

    @property
    def tokens(self) -> Optional[int]:
        """The frame count to REPORT, or None when narrator only has a bound.

        `max_new_tokens` counts frames, so frames is the unit the cap and the
        wire's `tokens` share. A served render's count is short by the trims
        and stating it would be stating a number nobody generated."""
        return int(self.frames) if self.counted else None

    @classmethod
    def counted_frames(cls, frames: int, cap: int) -> 'FrameMeasure':
        """The model's own frame count, against the cap it was given."""
        frames, cap = int(frames), int(cap)
        return cls(frames=frames, cap=cap, capped=frames >= cap, counted=True)

    @classmethod
    def from_audio(cls, audio, cap: int, sample_rate: int = SAMPLE_RATE) -> 'FrameMeasure':
        """The frames a DECODED waveform accounts for - see the class docstring
        for why the cap-hit is still decidable from it."""
        cap = int(cap)
        samples = 0 if audio is None else int(len(audio))
        frames = samples // int(sample_rate / FRAMES_PER_SECOND)
        return cls(frames=frames, cap=cap,
                   capped=frames + DELAY_TRIM_FRAMES >= cap, counted=False)

    @classmethod
    def joined(cls, parts: Sequence['FrameMeasure']) -> Optional['FrameMeasure']:
        """The measure of a chunk the ladder SPLIT and joined back.

        `capped` is ANY part's - one half cut off is a chunk cut off - and the
        frames and the caps both add, because what shipped is every part. None
        when no part was measured, and `counted` only when every part was: a
        sum of one counted and one inferred number is neither.
        """
        parts = [p for p in parts if p is not None]
        if not parts:
            return None
        return cls(frames=sum(int(p.frames) for p in parts),
                   cap=sum(int(p.cap) for p in parts),
                   capped=any(bool(p.capped) for p in parts),
                   counted=all(bool(p.counted) for p in parts))


class HiggsStreamMisaligned(ValueError):
    """The audio-token matrix cannot be sliced into a generated span.

    Named after `narrator.engine.orpheus.errors.TokenStreamMisaligned` and for
    the same reason: a stream that cannot be located is audio nobody can trust,
    so the row fails loudly instead of decoding whatever bytes happen to be
    there.
    """


def _as_matrix(tokens) -> np.ndarray:
    """`tokens` -> a 2-D (frames, 8) int array. Accepts a torch tensor, a numpy
    array or a nested list, with or without a leading batch dimension of 1."""
    if hasattr(tokens, 'detach'):          # a torch tensor, without importing torch
        tokens = tokens.detach().cpu().numpy()
    matrix = np.asarray(tokens)
    if matrix.ndim == 3:
        if matrix.shape[0] != 1:
            raise HiggsStreamMisaligned(
                f'Higgs decode takes one row at a time; got a batch of '
                f'{matrix.shape[0]}. Decode each row separately so a failure names '
                'the row it belongs to.')
        matrix = matrix[0]
    if matrix.ndim != 2:
        raise HiggsStreamMisaligned(
            f'Higgs audio tokens must be (frames, {NUM_CODEBOOKS}); got shape '
            f'{tuple(matrix.shape)}')
    if matrix.shape[1] != NUM_CODEBOOKS:
        raise HiggsStreamMisaligned(
            f'Higgs audio tokens must carry {NUM_CODEBOOKS} codebooks per frame; '
            f'got {matrix.shape[1]}')
    return matrix.astype(np.int64, copy=False)


def generated_span(tokens) -> np.ndarray:
    """Steps 1-3: the frames the model GENERATED, references and BOS removed.

    Returns a (frames, 8) array, possibly empty (a row that emitted EOS
    immediately). Raises when there is no BOS row at all - that is a matrix
    this decoder cannot honestly interpret.
    """
    matrix = _as_matrix(tokens)
    bos_rows = np.nonzero((matrix == AUDIO_STREAM_BOS_ID).all(axis=1))[0]
    if not len(bos_rows):
        raise HiggsStreamMisaligned(
            f'Higgs decode: no audio-stream BOS row (all {NUM_CODEBOOKS} codebooks '
            f'== {AUDIO_STREAM_BOS_ID}) in {matrix.shape[0]} frames, so the start of '
            'generation cannot be located.')
    start = int(bos_rows[-1])              # the LAST one: generation, not a reference
    span = matrix[start + 1:]              # the BOS row itself is dropped
    eos_rows = np.nonzero((span == AUDIO_STREAM_EOS_ID).all(axis=1))[0]
    if len(eos_rows):
        span = span[:int(eos_rows[0])]
    return span


def revert_delay_pattern(span: np.ndarray) -> np.ndarray:
    """Step 4: (frames, 8) delayed -> (8, frames - 7) aligned.

    Codebook c is delayed by c positions, so audio frame t of codebook c sits at
    stream position t + c. Returns the codebook-major (Q, T) layout the audio
    tokenizer's decoder wants.
    """
    frames = int(span.shape[0])
    keep = frames - DELAY_TRIM_FRAMES
    if keep <= 0:
        return np.zeros((NUM_CODEBOOKS, 0), dtype=np.int64)
    return np.stack([span[c:c + keep, c] for c in range(NUM_CODEBOOKS)], axis=0)


def trim_trailing_sentinels(codes_qt: np.ndarray) -> np.ndarray:
    """Step 4a: drop the trailing run of frames holding any out-of-range code.

    Out of range means "not a real codebook entry": >= 1024 (the stream
    sentinels) or < 0. Must run BEFORE the clip in `clip_codes`, which would
    turn those sentinels into the perfectly valid code 0.
    """
    if codes_qt.shape[1] == 0:
        return codes_qt
    bad = ((codes_qt >= NUM_REAL_CODES) | (codes_qt < 0)).any(axis=0)
    end = int(bad.shape[0])
    while end > 0 and bool(bad[end - 1]):
        end -= 1
    return codes_qt[:, :end]


def clip_codes(codes_qt: np.ndarray) -> np.ndarray:
    """Step 5: force anything still out of range into [0, 1023]."""
    return np.clip(codes_qt, 0, NUM_REAL_CODES - 1)


def stream_to_codes(tokens) -> np.ndarray:
    """Steps 1-5: the raw generate() output -> a (8, frames) codebook matrix
    ready for the audio tokenizer. Pure numpy; no model needed."""
    return clip_codes(trim_trailing_sentinels(
        revert_delay_pattern(generated_span(tokens))))


class HiggsCodec:
    """`narrator.engine.protocol.Codec` for Higgs v2/v3.

    `audio_decoder` is a callable taking the (8, frames) int matrix and
    returning a 1-D float32 waveform at 24 kHz - in production
    `processor.audio_tokenizer.decode`, in tests a fake. It is REQUIRED: a codec
    with no decoder that silently returns silence is the failure this whole
    package refuses.
    """

    sample_rate = SAMPLE_RATE
    frames_per_second = FRAMES_PER_SECOND
    tokens_per_frame = NUM_CODEBOOKS          # 8 entries per LM step
    samples_per_frame = SAMPLES_PER_FRAME     # 960
    trim_frames = DELAY_TRIM_FRAMES           # 7

    def __init__(self, audio_decoder):
        if not callable(audio_decoder):
            raise ValueError(
                'HiggsCodec(audio_decoder) needs a callable that turns an (8, frames) '
                'codebook matrix into a float32 waveform '
                "(processor.audio_tokenizer.decode); got "
                f'{type(audio_decoder).__name__}.')
        self._audio_decoder = audio_decoder

    def frames_for_tokens(self, n_tokens: int) -> int:
        """Higgs's autoregressive loop advances ONE step per frame and emits 8
        entries in that step, so a raw count of emitted entries is 8 per
        frame."""
        return int(n_tokens) // self.tokens_per_frame

    def audio_frames(self, generated_frames: int) -> int:
        """`generated_frames - 7`: reverting the delay pattern costs the
        diagonal. Never negative - a generation shorter than the diagonal
        carries no audio at all. The sentinel run (step 4a) takes MORE frames
        than this on top, by content, so this is the CEILING of what a
        generation of that length can yield, not a promise."""
        return max(0, int(generated_frames) - self.trim_frames)

    def samples_for_frames(self, n_frames: int) -> int:
        return int(n_frames) * self.samples_per_frame

    def seconds_for_frames(self, n_frames: int) -> float:
        return int(n_frames) / self.frames_per_second

    def decode(self, tokens) -> np.ndarray:
        """generate()'s (seq, 8) matrix -> 1-D float32 mono at 24 kHz."""
        codes = stream_to_codes(tokens)
        if codes.shape[1] == 0:
            raise HiggsStreamMisaligned(
                'Higgs decode: the generated span holds no audio frames after the '
                'delay-pattern and sentinel trims. The row produced no speech - it '
                'is a failed render, not a silent one.')
        audio = self._audio_decoder(codes)
        if hasattr(audio, 'detach'):
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        expected = self.samples_for_frames(codes.shape[1])
        if audio.size != expected:
            raise HiggsStreamMisaligned(
                f'Higgs decode: {codes.shape[1]} frames should decode to {expected} '
                f'samples at {self.sample_rate} Hz, got {audio.size}. The codec and '
                'this arithmetic disagree - refusing to ship audio of an unknown '
                'length.')
        return audio

    def streaming_decoder(self, decode_frames, label: str = ''):
        """None - Higgs has NO sound windowed decode, so nothing may pretend to
        stream it.

        SNAC's `WindowedFrameEmitter` works because a window can overhang its
        payload by whole frames on both sides and the interior is then exact.
        Higgs's delay pattern makes a window's LAST 7 frames incomplete by
        construction (codebook c of frame t has not been emitted until step
        t + c), and its end-of-stream sentinels - the thing step 4a trims by
        content - only exist once generation has finished, so a mid-row window
        cannot tell a ramp-down from speech. Emitting per row at retirement is
        therefore the honest cadence; see
        `HiggsEngine.generate_batch_stream`. Faking a stream out of independent
        decodes would ship audio the listener has already heard by the time
        anything could notice it was wrong.
        """
        return None
