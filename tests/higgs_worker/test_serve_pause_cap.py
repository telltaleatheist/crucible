"""The interior-pause cap (serve/pause_cap.py; Owen, 2026-10-02).

What it must do, from training-pc's spec: a synthetic chunk with a 10 s
interior gap comes out with a 1.5 s gap; a 1.2 s gap is untouched; edge
silences are untouched. And the render door reports every cut while the
Listen door cuts nothing.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from crucible_higgs.serve import pause_cap as P

RATE = 24000


def speech(seconds: float, freq: float = 220.0) -> np.ndarray:
    """A stand-in for a word: a tone well above the silence line."""
    t = np.arange(int(round(seconds * RATE))) / RATE
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def quiet(seconds: float) -> np.ndarray:
    """A pause: room noise far under -40 dB of the speech peak, never exact zeros."""
    rng = np.random.default_rng(int(seconds * 1000))
    return (1e-4 * rng.standard_normal(int(round(seconds * RATE)))).astype(np.float32)


def chunk(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts)


def pause_lengths(audio: np.ndarray) -> list[float]:
    """Interior pauses >= 0.30 s, as higgs_pause_screen measures them."""
    return [round((e - s) / RATE, 2) for s, e in P.interior_pauses(audio, RATE)
            if (e - s) / RATE >= 0.30]


class CapTest(unittest.TestCase):

    def test_a_ten_second_stall_comes_out_one_and_a_half(self):
        audio = chunk(speech(1.0), quiet(10.0), speech(1.0))
        self.assertEqual(pause_lengths(audio), [10.0])
        out, cuts = P.cap_interior_pauses(audio, RATE, 1.5)
        self.assertEqual(pause_lengths(out), [1.5])
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]['fromS'], 10.0)
        self.assertEqual(cuts[0]['toS'], 1.5)
        self.assertAlmostEqual(cuts[0]['atS'], 1.0, places=2)
        # Exactly the removed length, sample for sample.
        self.assertEqual(len(audio) - len(out), int(round(8.5 * RATE)))
        # The speech either side is the speech that was rendered.
        np.testing.assert_array_equal(out[:RATE], audio[:RATE])
        np.testing.assert_array_equal(out[-RATE:], audio[-RATE:])

    def test_a_pause_under_the_cap_is_untouched(self):
        audio = chunk(speech(1.0), quiet(1.2), speech(1.0))
        out, cuts = P.cap_interior_pauses(audio, RATE, 1.5)
        self.assertEqual(cuts, [])
        self.assertIs(out, audio)

    def test_edge_silence_is_never_interior(self):
        audio = chunk(quiet(4.0), speech(1.0), quiet(1.0), speech(1.0), quiet(6.0))
        out, cuts = P.cap_interior_pauses(audio, RATE, 1.5)
        self.assertEqual(cuts, [])
        self.assertIs(out, audio)

    def test_several_stalls_are_each_cut_and_each_reported(self):
        audio = chunk(speech(0.5), quiet(3.0), speech(0.5), quiet(0.8), speech(0.5),
                      quiet(92.0), speech(0.5))
        out, cuts = P.cap_interior_pauses(audio, RATE, 1.5)
        self.assertEqual([c['fromS'] for c in cuts], [3.0, 92.0])
        self.assertEqual(pause_lengths(out), [1.5, 0.8, 1.5])

    def test_the_empty_sentence_placeholder_has_nothing_to_cut(self):
        zeros = np.zeros(int(RATE * 0.05), dtype=np.float32)
        out, cuts = P.cap_interior_pauses(zeros, RATE, 1.5)
        self.assertEqual(cuts, [])
        self.assertIs(out, zeros)

    def test_the_splice_is_continuous(self):
        audio = chunk(speech(1.0), quiet(10.0), speech(1.0))
        out, _cuts = P.cap_interior_pauses(audio, RATE, 1.5)
        # Inside the kept pause, no sample jumps further than the room noise does.
        pause = out[RATE + 600:RATE + int(1.5 * RATE) - 600]
        self.assertLess(float(np.abs(np.diff(pause)).max()), 1e-3)


class SettingTest(unittest.TestCase):

    def setUp(self):
        self.saved = os.environ.pop(P.ENV_MAX_PAUSE, None)

    def tearDown(self):
        os.environ.pop(P.ENV_MAX_PAUSE, None)
        if self.saved is not None:
            os.environ[P.ENV_MAX_PAUSE] = self.saved

    def test_on_by_default_at_one_and_a_half(self):
        self.assertEqual(P.max_pause_seconds(), 1.5)

    def test_one_setting_moves_it_and_zero_switches_it_off(self):
        os.environ[P.ENV_MAX_PAUSE] = '2.5'
        self.assertEqual(P.max_pause_seconds(), 2.5)
        os.environ[P.ENV_MAX_PAUSE] = '0'
        self.assertIsNone(P.max_pause_seconds())

    def test_a_typo_is_refused_by_name(self):
        os.environ[P.ENV_MAX_PAUSE] = '1,5'
        with self.assertRaises(ValueError) as caught:
            P.max_pause_seconds()
        self.assertIn(P.ENV_MAX_PAUSE, str(caught.exception))


class EmitterTest(unittest.TestCase):
    """The render door cuts and reports; the Listen door cuts nothing."""

    def emit(self, audio, door):
        import crucible_higgs.serve.worker as W
        sent = []
        original = W.send_response
        W.send_response = lambda kind, payload=None: sent.append((kind, payload))
        try:
            W.OrpheusStreamServer._emit_batch_item({'i': 7, 'text': 'A line.'}, audio, door)
        finally:
            W.send_response = original
        self.assertEqual(len(sent), 1)
        return sent[0][1]

    def test_a_render_row_is_capped_and_carries_its_cuts(self):
        import crucible_higgs.serve.worker as W
        audio = chunk(speech(1.0), quiet(10.0), speech(1.0))
        row = self.emit(audio, W.FOR_RENDER)
        self.assertEqual([c['fromS'] for c in row['pauseCuts']], [10.0])
        self.assertAlmostEqual(row['duration'], 3.5, places=2)

    def test_a_render_row_with_nothing_to_cut_says_so(self):
        import crucible_higgs.serve.worker as W
        row = self.emit(chunk(speech(1.0), quiet(1.2), speech(1.0)), W.FOR_RENDER)
        self.assertEqual(row['pauseCuts'], [])

    def test_the_listen_door_is_not_cut(self):
        import crucible_higgs.serve.worker as W
        audio = chunk(speech(1.0), quiet(10.0), speech(1.0))
        row = self.emit(audio, W.FOR_STREAM)
        self.assertNotIn('pauseCuts', row)
        self.assertAlmostEqual(row['duration'], 12.0, places=2)


if __name__ == '__main__':
    unittest.main()
