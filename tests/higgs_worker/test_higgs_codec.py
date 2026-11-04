"""The Higgs decode, arithmetic first: span, delay pattern, sentinel trim.

Every step but the final codec call is pure numpy, so all of it runs here with a
FAKE decoder on an interpreter with no torch and no model.

THE TWO TRIMS ARE THE POINT. `frames - 7` (the delay pattern's diagonal) is the
one the shipped code does. It is not enough: the ramp-down BOC/EOC sentinels
smear across the last seven frames, the shipped code maps them to code 0 - a
VALID code that decodes to sound - and trims exactly one frame, leaving about
240 ms of audible garbage at the end of every chunk. Owen heard it as "a stray
syllable or sound after each sentence". The fix, measured 2026-09-04, is to drop
the trailing sentinel run BY CONTENT, before the substitution. These tests pin
both trims and their ORDER, because doing them the other way round looks
identical in code and silently does nothing.
"""
import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_ROOT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "crucible", "jobs", "tts", "higgs")   # holds crucible_higgs
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)

from crucible_higgs.engine.higgs.codec import (DELAY_TRIM_FRAMES,          # noqa: E402
                                         FrameMeasure,
                                         HiggsCodec, HiggsStreamMisaligned,
                                         SAMPLES_PER_FRAME, clip_codes,
                                         generated_span,
                                         revert_delay_pattern,
                                         stream_to_codes,
                                         trim_trailing_sentinels)
from crucible_higgs.engine.higgs.prompt import (AUDIO_STREAM_BOS_ID,       # noqa: E402
                                          AUDIO_STREAM_EOS_ID,
                                          NUM_CODEBOOKS)

BOS = [AUDIO_STREAM_BOS_ID] * NUM_CODEBOOKS
EOS = [AUDIO_STREAM_EOS_ID] * NUM_CODEBOOKS


def delayed(frames):
    """Build the (T, 8) DELAYED stream whose reverted content is
    `frames[t][c]` - i.e. put codebook c's entry for audio frame t at stream
    position t + c, which is what the model emits."""
    n = len(frames)
    width = n + NUM_CODEBOOKS - 1
    out = np.zeros((width, NUM_CODEBOOKS), dtype=np.int64)
    for t, row in enumerate(frames):
        for c in range(NUM_CODEBOOKS):
            out[t + c, c] = row[c]
    return out


def fake_decoder(samples_per_frame=SAMPLES_PER_FRAME):
    """A stand-in for `processor.audio_tokenizer.decode`: the right LENGTH, and
    a value per frame so a test can see WHICH frames survived."""
    def decode(codes_qt):
        frames = codes_qt.shape[1]
        return np.repeat(codes_qt[0].astype(np.float32) / 1000.0,
                         samples_per_frame)[:frames * samples_per_frame]
    return decode


class GeneratedSpanTest(unittest.TestCase):

    def test_the_last_bos_wins_and_the_bos_row_is_dropped(self):
        """Everything before the last BOS is the REFERENCE clips, which is why
        the whole sequence can be handed back verbatim."""
        matrix = np.array([[1] * 8, BOS, [2] * 8, BOS, [3] * 8, [4] * 8])
        span = generated_span(matrix)
        self.assertEqual(span.tolist(), [[3] * 8, [4] * 8])

    def test_the_first_eos_ends_it(self):
        matrix = np.array([BOS, [3] * 8, EOS, [9] * 8, EOS])
        self.assertEqual(generated_span(matrix).tolist(), [[3] * 8])

    def test_no_eos_decodes_to_the_end(self):
        matrix = np.array([BOS, [3] * 8, [4] * 8])
        self.assertEqual(generated_span(matrix).tolist(), [[3] * 8, [4] * 8])

    def test_a_leading_batch_dimension_of_one_is_accepted(self):
        matrix = np.array([[BOS, [3] * 8]])
        self.assertEqual(generated_span(matrix).tolist(), [[3] * 8])

    def test_a_real_batch_is_refused(self):
        with self.assertRaises(HiggsStreamMisaligned) as caught:
            generated_span(np.zeros((2, 4, 8), dtype=np.int64))
        self.assertIn('one row at a time', str(caught.exception))

    def test_no_bos_at_all_is_refused(self):
        with self.assertRaises(HiggsStreamMisaligned) as caught:
            generated_span(np.array([[3] * 8, [4] * 8]))
        self.assertIn('no audio-stream BOS row', str(caught.exception))

    def test_the_wrong_codebook_count_is_refused(self):
        with self.assertRaises(HiggsStreamMisaligned) as caught:
            generated_span(np.zeros((4, 7), dtype=np.int64))
        self.assertIn('8 codebooks', str(caught.exception))


class DelayPatternTest(unittest.TestCase):

    def test_reverting_recovers_the_frames_exactly(self):
        frames = [[10 * t + c for c in range(NUM_CODEBOOKS)] for t in range(12)]
        reverted = revert_delay_pattern(delayed(frames))
        self.assertEqual(reverted.shape, (NUM_CODEBOOKS, 12))
        # (Q, T) codebook-major, which is what the tokenizer's decoder wants.
        self.assertEqual(reverted.T.tolist(), frames)

    def test_it_costs_exactly_seven_frames(self):
        self.assertEqual(DELAY_TRIM_FRAMES, NUM_CODEBOOKS - 1)
        for length in (8, 20, 100, 1468):
            with self.subTest(length=length):
                span = np.zeros((length, NUM_CODEBOOKS), dtype=np.int64)
                self.assertEqual(revert_delay_pattern(span).shape[1],
                                 length - DELAY_TRIM_FRAMES)

    def test_a_span_shorter_than_the_diagonal_yields_no_audio(self):
        for length in (0, 1, 7):
            with self.subTest(length=length):
                span = np.zeros((length, NUM_CODEBOOKS), dtype=np.int64)
                self.assertEqual(revert_delay_pattern(span).shape, (8, 0))


class SentinelTrimTest(unittest.TestCase):

    def test_the_trailing_sentinel_run_goes_by_content(self):
        codes = np.zeros((NUM_CODEBOOKS, 10), dtype=np.int64)
        codes[:, 6:] = AUDIO_STREAM_EOS_ID       # a 4-frame ramp-down
        self.assertEqual(trim_trailing_sentinels(codes).shape[1], 6)

    def test_a_single_out_of_range_code_condemns_its_frame(self):
        """One codebook of the eight is enough: the frame is contaminated."""
        codes = np.zeros((NUM_CODEBOOKS, 5), dtype=np.int64)
        codes[3, 4] = AUDIO_STREAM_BOS_ID
        self.assertEqual(trim_trailing_sentinels(codes).shape[1], 4)

    def test_it_never_eats_a_clean_tail(self):
        codes = np.full((NUM_CODEBOOKS, 5), 512, dtype=np.int64)
        self.assertEqual(trim_trailing_sentinels(codes).shape[1], 5)

    def test_it_only_trims_the_TRAILING_run(self):
        """A sentinel in the middle is not a ramp-down and must not truncate the
        speech after it."""
        codes = np.zeros((NUM_CODEBOOKS, 8), dtype=np.int64)
        codes[:, 2] = AUDIO_STREAM_EOS_ID
        self.assertEqual(trim_trailing_sentinels(codes).shape[1], 8)

    def test_the_order_matters_and_this_is_why(self):
        """Clipping first turns 1024/1025 into code 0 - a perfectly valid code
        that decodes to sound - and the content trim then finds nothing. This is
        the bug, written down as a test."""
        codes = np.zeros((NUM_CODEBOOKS, 10), dtype=np.int64)
        codes[:, 6:] = AUDIO_STREAM_EOS_ID
        self.assertEqual(trim_trailing_sentinels(clip_codes(codes)).shape[1], 10,
                         'clip-then-trim is a no-op: exactly the defect')
        self.assertEqual(clip_codes(trim_trailing_sentinels(codes)).shape[1], 6,
                         'trim-then-clip is the fix')

    def test_negative_codes_count_as_out_of_range(self):
        """vllm-omni's clone path emits -100 as an audio placeholder; anything
        below zero is not a codebook entry."""
        codes = np.zeros((NUM_CODEBOOKS, 4), dtype=np.int64)
        codes[:, 3] = -100
        self.assertEqual(trim_trailing_sentinels(codes).shape[1], 3)


class EndToEndArithmeticTest(unittest.TestCase):

    def test_the_whole_pipeline_on_a_realistic_stream(self):
        """20 audio frames of speech, then a 7-frame sentinel ramp-down, wrapped
        in the BOS/EOS the model emits."""
        speech = [[100 + t] * NUM_CODEBOOKS for t in range(20)]
        ramp = [[AUDIO_STREAM_EOS_ID] * NUM_CODEBOOKS for _ in range(7)]
        stream = np.concatenate([
            np.array([[7] * NUM_CODEBOOKS] * 3),        # reference frames
            np.array([BOS]),
            delayed(speech + ramp),
            np.array([EOS]),
        ])
        codes = stream_to_codes(stream)
        self.assertEqual(codes.shape[0], NUM_CODEBOOKS)
        self.assertEqual(codes.shape[1], 20,
                         'the 7 ramp frames go by content, the delay diagonal by '
                         'arithmetic, and the 3 reference frames with the BOS')
        self.assertEqual(codes[0].tolist(), [100 + t for t in range(20)])


class CodecTest(unittest.TestCase):

    def setUp(self):
        self.codec = HiggsCodec(fake_decoder())

    def test_the_geometry(self):
        self.assertEqual(self.codec.sample_rate, 24000)
        self.assertEqual(self.codec.frames_per_second, 25.0)
        self.assertEqual(self.codec.tokens_per_frame, 8)
        self.assertEqual(self.codec.samples_per_frame, 960)
        self.assertEqual(self.codec.trim_frames, 7)
        # 25 fps at 24 kHz IS 960 samples, and 40 ms a frame.
        self.assertEqual(self.codec.samples_per_frame * self.codec.frames_per_second,
                         self.codec.sample_rate)

    def test_frames_and_samples(self):
        self.assertEqual(self.codec.frames_for_tokens(80), 10)
        self.assertEqual(self.codec.frames_for_tokens(83), 10)
        self.assertEqual(self.codec.audio_frames(100), 93)
        self.assertEqual(self.codec.audio_frames(3), 0)
        self.assertEqual(self.codec.samples_for_frames(25), 24000)
        self.assertEqual(self.codec.seconds_for_frames(1468), 58.72)

    def test_decode_returns_exactly_the_expected_length(self):
        speech = [[200 + t] * NUM_CODEBOOKS for t in range(12)]
        stream = np.concatenate([np.array([BOS]), delayed(speech), np.array([EOS])])
        audio = self.codec.decode(stream)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.ndim, 1)
        self.assertEqual(audio.size, 12 * SAMPLES_PER_FRAME)

    def test_a_decoder_that_returns_the_wrong_length_is_refused(self):
        """A codec and this arithmetic disagreeing means audio of an unknown
        duration, which the manifest cannot describe."""
        codec = HiggsCodec(lambda codes: np.zeros(7, dtype=np.float32))
        speech = [[1] * NUM_CODEBOOKS for _ in range(12)]
        stream = np.concatenate([np.array([BOS]), delayed(speech), np.array([EOS])])
        with self.assertRaises(HiggsStreamMisaligned) as caught:
            codec.decode(stream)
        self.assertIn('samples', str(caught.exception))

    def test_a_row_with_no_audio_is_a_failure_not_silence(self):
        stream = np.concatenate([np.array([BOS]),
                                 np.zeros((3, NUM_CODEBOOKS), dtype=np.int64),
                                 np.array([EOS])])
        with self.assertRaises(HiggsStreamMisaligned) as caught:
            self.codec.decode(stream)
        self.assertIn('failed render', str(caught.exception))

    def test_a_decoder_is_required(self):
        with self.assertRaises(ValueError):
            HiggsCodec(None)

    def test_there_is_no_windowed_decoder(self):
        self.assertIsNone(self.codec.streaming_decoder(lambda a, b: None))


class FrameMeasureTest(unittest.TestCase):
    """What a render SPENT, against the ceiling it was given.

    THE MEASUREMENT IS UNCONDITIONAL (Owen, 2026-09-19): it rides both arms of
    the serve worker's batch door, because whether a generation ran out of
    frame budget is a fact about the render and not a verdict on it. Crucible
    publishes `capped` and reads a missing one as "narrator did not say" -
    never as `false`, because a runaway reported as "not capped" is exactly the
    failure the field exists to prevent.
    """

    def test_a_counted_frame_count_decides_the_cap_hit_exactly(self):
        """The MLX arm generates in this process, so the frames the model
        produced ARE the rows it handed back."""
        self.assertFalse(FrameMeasure.counted_frames(999, 1000).capped)
        self.assertTrue(FrameMeasure.counted_frames(1000, 1000).capped)
        self.assertTrue(FrameMeasure.counted_frames(1200, 1000).capped)
        self.assertEqual(FrameMeasure.counted_frames(1000, 1000).tokens, 1000)

    def test_an_INFERRED_count_decides_the_cap_hit_and_reports_no_tokens(self):
        """The served stack returns a WAV and nothing else - no token count, no
        stop reason - so the frames narrator can see are a LOWER BOUND and are
        not reported as `tokens`. The cap hit is still decidable: a generation
        that reached the cap emitted no EOS, so no sentinel run was stripped
        and only the delay diagonal (7 frames) was, leaving exactly `cap - 7`.
        """
        cap = 1000
        at_cap = np.zeros((cap - DELAY_TRIM_FRAMES) * SAMPLES_PER_FRAME,
                          dtype=np.float32)
        measure = FrameMeasure.from_audio(at_cap, cap)
        self.assertTrue(measure.capped)
        self.assertFalse(measure.counted)
        self.assertIsNone(measure.tokens,
                          'a lower bound must not be reported as a count')
        short = np.zeros((cap - DELAY_TRIM_FRAMES - 1) * SAMPLES_PER_FRAME,
                         dtype=np.float32)
        self.assertFalse(FrameMeasure.from_audio(short, cap).capped)

    def test_a_SPLIT_chunk_is_capped_when_any_part_was(self):
        """One half cut off is a chunk cut off. `counted` survives only if
        every part was counted - a sum of a count and a bound is neither."""
        counted = FrameMeasure.counted_frames(10, 100)
        hit = FrameMeasure.counted_frames(100, 100)
        joined = FrameMeasure.joined([counted, hit])
        self.assertTrue(joined.capped)
        self.assertEqual(joined.frames, 110)
        self.assertEqual(joined.cap, 200)
        self.assertTrue(joined.counted)
        mixed = FrameMeasure.joined(
            [counted, FrameMeasure(frames=5, cap=100, capped=False, counted=False)])
        self.assertFalse(mixed.counted)
        self.assertIsNone(mixed.tokens)
        self.assertIsNone(FrameMeasure.joined([None, None]),
                          'nothing measured is None, not a zero')


if __name__ == '__main__':
    unittest.main()
