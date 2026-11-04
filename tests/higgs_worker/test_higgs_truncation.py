"""The Higgs truncation ladder (`engine/higgs/truncation.py`) without a GPU.

Measured case (Fuhrer, PC/SGLang, 2026-09-06): chunk 19, 1,127 chars, 3.0 s of
audio - a mid-chunk early stop that reproduces at its seed. The ladder must
re-roll at ANOTHER seed, then split at a sentence boundary, and never refuse.
"""
import json
import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_ROOT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "crucible", "jobs", "tts", "higgs")   # holds crucible_higgs
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)

from crucible_higgs.engine.higgs import truncation, sgl_served, served_common  # noqa: E402
from crucible_higgs.engine.higgs.v3_engine import HiggsV3Config, HiggsV3Engine  # noqa: E402
from tests.higgs_worker.test_higgs_sgl import SglTestCase, state_env  # noqa: E402

RATE = 24000
PACE = 17.0  # chars per second of a normal take on this voice


def audio_for(chars: int, cps: float = PACE) -> np.ndarray:
    return np.zeros(int(chars / cps * RATE), dtype=np.float32)


SENTENCES = ('It would be hard to say this about Nazism. '
             'The goal of national redemption through racial purification and a racial empire was a fantasy, a utopian idea. '
             'The cruelty and destruction that came from trying to achieve this goal are known to everyone. '
             'Its main feature was a lack of system, disorder in administration and government, and the breakdown of clear patterns of governance. ')
TEXT = (SENTENCES * 3).strip()   # ~1,300 chars, 12 sentences


class FakeRender:
    """A render whose behaviour is scripted per (text, seed): `short` says which
    calls stop early (3 s of audio, the chunk-19 shape); `long` says which run
    on (1.5x the expected audio, the chunk-48 shape)."""

    def __init__(self, short, long=lambda t, s, n: False):
        self.short = short
        self.long = long
        self.calls = []

    def __call__(self, text, seed):
        self.calls.append((text, seed))
        if self.short(text, seed, len(self.calls)):
            return audio_for(50)          # ~3 s: stopped after the first sentence
        if self.long(text, seed, len(self.calls)):
            return audio_for(int(len(text) * 1.5))   # a 26 s tail on a 52 s chunk
        return audio_for(len(text))


class GuardTest(unittest.TestCase):

    def test_chunk_19_is_truncated_by_the_policy_number(self):
        verdict = truncation.check('x' * 1127, audio_for(51), RATE, 20.0, 14.5)
        self.assertTrue(verdict.short)
        self.assertEqual(verdict.side, 'short')
        self.assertGreater(verdict.chars_per_second, 300)
        ok = truncation.check('x' * 1127, audio_for(1127), RATE, 20.0, 14.5)
        self.assertFalse(ok.off_length)
        self.assertFalse(truncation.check('x' * 1127, audio_for(51), RATE, 0, 0).off_length,
                         'thresholds of 0 disable the guard')

    def test_the_fuhrer_run_ons_are_long_by_the_policy_number(self):
        """Chunks 48/35/26 ran on at 1.54x/1.42x/1.24x of expectation; the
        1.11x one (chunk 41) is inside the band and left to the audit."""
        pace = 17.2
        for ratio, caught in ((1.54, True), (1.42, True), (1.24, True), (1.11, False), (1.10, False)):
            audio = np.zeros(int(1000 / pace * ratio * RATE), dtype=np.float32)
            verdict = truncation.check('x' * 1000, audio, RATE, 20.0, 14.5)
            self.assertEqual(verdict.long, caught, f'ratio {ratio}: {verdict}')
        # And every clean chunk of the render (0.94x-1.10x) is inside the band.
        for ratio in (0.94, 1.0, 1.10):
            audio = np.zeros(int(1000 / pace * ratio * RATE), dtype=np.float32)
            self.assertFalse(truncation.check('x' * 1000, audio, RATE, 20.0, 14.5).off_length, ratio)

    def test_a_run_on_is_rerolled_and_then_split_like_an_early_stop(self):
        render = FakeRender(lambda t, s, n: False, long=lambda t, s, n: t == TEXT)
        events = []
        out = truncation.render_guarded(render, TEXT, 48, sample_rate=RATE, max_chars_per_sec=20.0,
                                        min_chars_per_sec=14.5, base_seed=1234, on_event=events.append)
        self.assertEqual([e['action'] for e in events], ['long', 'resplit'])
        self.assertEqual(events[0]['side'], 'long')
        halves = truncation.split_halves(TEXT)
        expected = len(audio_for(len(halves[0]))) + len(audio_for(len(halves[1]))) \
            + int(round(truncation.RESPLIT_JOIN_SECONDS * RATE))
        self.assertEqual(len(out), expected)

    def test_at_the_bottom_the_take_nearest_the_expected_length_ships(self):
        # Take 0 far too long (3x), the re-roll far too short; nothing splits
        # (the text is one short sentence) -> the shorter miss ships? No: the
        # one CLOSEST to the band's centre - here the long one is 3x off and the
        # short one is ~7x off, so the long take ships.
        short_text = 'x' * 160    # over MIN_GUARD_CHARS, so the long side is judged
        render = FakeRender(lambda t, s, n: n == 2, long=lambda t, s, n: n == 1)
        render.long = lambda t, s, n: n == 1
        calls = {'n': 0}

        def scripted(text, seed):
            calls['n'] += 1
            render.calls.append((text, seed))
            return audio_for(480) if calls['n'] == 1 else audio_for(24)
        events = []
        out = truncation.render_guarded(scripted, short_text, 5, sample_rate=RATE, max_chars_per_sec=20.0,
                                        min_chars_per_sec=14.5, base_seed=1, on_event=events.append)
        self.assertEqual(events[-1]['action'], 'accepted-off-length')
        self.assertEqual(events[-1]['shipped_side'], 'long')
        self.assertEqual(len(out), len(audio_for(480)))

    def test_the_reroll_seed_differs_from_the_chunks_own_and_none_stays_none(self):
        self.assertNotEqual(truncation.reroll_seed(1234, 19, 1), 1234 + 19)
        self.assertNotEqual(truncation.reroll_seed(1234, 19, 1), truncation.reroll_seed(1234, 19, 2))
        self.assertIsNone(truncation.reroll_seed(None, 19, 1))

    def test_halves_cut_at_the_sentence_boundary_nearest_the_middle(self):
        parts = truncation.split_halves(TEXT)
        self.assertEqual(len(parts), 2)
        self.assertTrue(parts[0].endswith('.'))
        self.assertLess(abs(len(parts[0]) - len(parts[1])), 200)
        self.assertEqual(' '.join(parts), TEXT)
        self.assertEqual(truncation.split_halves('Too short to split.'), [])
        one = 'word ' * 60
        halves = truncation.split_halves(one.strip())
        self.assertEqual(len(halves), 2, 'a single long sentence splits at the middle space')

    def test_a_good_take_ships_untouched_with_no_event(self):
        render = FakeRender(lambda t, s, n: False)
        events = []
        out = truncation.render_guarded(render, TEXT, 19, sample_rate=RATE, max_chars_per_sec=20.0,
                                        base_seed=1234, on_event=events.append)
        self.assertEqual(len(render.calls), 1)
        self.assertEqual(render.calls[0], (TEXT, None), 'take 0 is the engine\'s own seed rule')
        self.assertEqual(events, [])
        self.assertEqual(len(out), len(audio_for(len(TEXT))))

    def test_an_early_stop_is_rerolled_at_another_seed_and_ships_the_reroll(self):
        render = FakeRender(lambda t, s, n: s is None)   # the chunk's own seed stops early, any other seed is fine
        events = []
        out = truncation.render_guarded(render, TEXT, 19, sample_rate=RATE, max_chars_per_sec=20.0,
                                        base_seed=1234, on_event=events.append)
        self.assertEqual([c[1] for c in render.calls], [None, truncation.reroll_seed(1234, 19, 1)])
        self.assertEqual([e['action'] for e in events], ['short', 'rerolled'])
        self.assertEqual(events[0]['index'], 19)
        self.assertEqual(len(out), len(audio_for(len(TEXT))))

    def test_a_stop_that_survives_the_reroll_is_split_and_each_half_rendered(self):
        # Every render of the WHOLE text stops early; halves render fine.
        render = FakeRender(lambda t, s, n: t == TEXT)
        events = []
        out = truncation.render_guarded(render, TEXT, 19, sample_rate=RATE, max_chars_per_sec=20.0,
                                        base_seed=1234, on_event=events.append)
        actions = [e['action'] for e in events]
        self.assertEqual(actions, ['short', 'resplit'])
        halves = truncation.split_halves(TEXT)
        self.assertEqual([c[0] for c in render.calls], [TEXT, TEXT] + halves)
        expected = len(audio_for(len(halves[0]))) + len(audio_for(len(halves[1]))) \
            + int(round(truncation.RESPLIT_JOIN_SECONDS * RATE))
        self.assertEqual(len(out), expected, 'the halves are joined with the sentence pause')

    def test_the_ladder_never_refuses_and_says_when_it_accepts_a_short_take(self):
        render = FakeRender(lambda t, s, n: True)   # nothing ever renders full length
        events = []
        out = truncation.render_guarded(render, TEXT, 19, sample_rate=RATE, max_chars_per_sec=20.0,
                                        base_seed=None, on_event=events.append)
        self.assertGreater(len(out), 0)
        self.assertIn('accepted-off-length', [e['action'] for e in events])
        depths = {e['depth'] for e in events}
        self.assertLessEqual(max(depths), truncation.MAX_DEPTH)
        self.assertTrue(all(c[1] is None for c in render.calls),
                        'an unseeded engine re-rolls by sampling fresh - no invented seed')

    def test_a_first_take_handed_in_is_not_rendered_again(self):
        render = FakeRender(lambda t, s, n: False)
        out = truncation.render_guarded(render, TEXT, 3, sample_rate=RATE, max_chars_per_sec=20.0,
                                        base_seed=1, first_take=audio_for(len(TEXT)))
        self.assertEqual(render.calls, [])
        self.assertEqual(len(out), len(audio_for(len(TEXT))))


class PaceTrackerTest(unittest.TestCase):
    """The band follows the book (Owen, 2026-09-08). Numbers from the Shift
    render: mistborn recorded 15.12 chars/s; the book ran at 14.09."""

    def _tracker(self):
        # The catalog's seed band at F = 1.2: 18.14 / 12.6 around 15.12.
        return truncation.PaceTracker(15.12, 18.14, 12.6, warmup=4)

    def test_the_seed_band_is_the_documents_until_warm(self):
        t = self._tracker()
        self.assertEqual(t.band(), {'max_chars_per_sec': 18.14, 'min_chars_per_sec': 12.6})
        self.assertFalse(t.warm)
        self.assertEqual(t.reference, 15.12)

    def test_the_band_recentres_on_the_shipped_takes_median(self):
        t = self._tracker()
        for cps in (14.0, 14.2, 13.9, 14.3):
            t.observe(1000, 1000 / cps)
        self.assertTrue(t.warm)
        self.assertAlmostEqual(t.reference, 14.1, places=6)
        band = t.band()
        # The seed band's RATIOS, around the book's pace: 18.14/15.12 and 15.12/12.6.
        self.assertAlmostEqual(band['max_chars_per_sec'], round(14.1 * 18.14 / 15.12, 2))
        self.assertAlmostEqual(band['min_chars_per_sec'], round(14.1 / (15.12 / 12.6), 2))
        # Shift's fastest shipped take (998 chars in 53.6 s = 18.6) is now SHORT
        # against the book's pace, where the fixed band (19.27) let it through.
        self.assertTrue(truncation.check('x' * 998, audio_for(998, 18.6), RATE, **band).short)

    def test_small_chunks_feed_nothing_and_are_short_side_only(self):
        t = self._tracker()
        for _ in range(10):
            t.observe(8, 1.0)          # a heading: 8 chars/s, under MIN_GUARD_CHARS
        self.assertEqual(t.observed, 0)
        self.assertFalse(t.warm)
        # Through the ladder: a 40-char heading at 5 chars/s is not 'long'...
        render = FakeRender(lambda t_, s, n: False)
        events = []
        out = truncation.render_guarded(render, 'x' * 40, 3, sample_rate=RATE, tracker=t,
                                        base_seed=1, on_event=events.append)
        self.assertEqual(events, [])
        self.assertEqual(len(render.calls), 1)
        self.assertEqual(len(out), len(audio_for(40)))
        # ...but the same heading at 60 chars/s (cut off) IS short and re-rolled.
        events = []
        calls = {'n': 0}

        def fast_once(text, seed):
            calls['n'] += 1
            return audio_for(40, 60.0) if calls['n'] == 1 else audio_for(40)
        truncation.render_guarded(fast_once, 'x' * 40, 3, sample_rate=RATE, tracker=t,
                                  base_seed=1, on_event=events.append)
        self.assertEqual([e['action'] for e in events], ['short', 'rerolled'])
        self.assertEqual(events[0]['pace_source'], 'recorded')

    def test_the_ladder_feeds_shipped_takes_back_and_not_accepted_misses(self):
        t = self._tracker()
        render = FakeRender(lambda t_, s, n: False)
        for i in range(4):
            truncation.render_guarded(render, TEXT, i, sample_rate=RATE, tracker=t, base_seed=1,
                                      on_event=lambda e: None)
        self.assertTrue(t.warm)
        self.assertAlmostEqual(t.reference, PACE, places=1)
        # A chunk that stays short at every rung is accepted and NOT observed.
        before = t.observed
        always_short = FakeRender(lambda t_, s, n: True)
        events = []
        truncation.render_guarded(always_short, 'x' * 100 + ' ' + 'y' * 100, 9, sample_rate=RATE,
                                  tracker=t, base_seed=1, on_event=events.append)
        self.assertEqual(events[-1]['action'], 'accepted-off-length')
        self.assertEqual(events[-1]['pace_source'], 'book')
        self.assertEqual(t.observed, before)

    def test_a_malformed_seed_is_refused_by_name(self):
        for args, why in (((15.0, 14.0, 12.0), 'min < pace < max'),
                          ((0, 18.0, 12.0), 'positive'),
                          ((15.0, 18.0, -1), 'positive')):
            with self.assertRaises(ValueError) as caught:
                truncation.PaceTracker(*args)
            self.assertIn(why, str(caught.exception))


class VoiceBandTest(unittest.TestCase):
    """The catalog's recorded pace and seed band ride on the voice and seed the guard's tracker."""

    def _load(self, entry):
        import json, tempfile
        from crucible_higgs.engine.higgs.config import load_voices
        d = tempfile.mkdtemp()
        p = os.path.join(d, 'voices.json')
        with open(p, 'w', encoding='utf-8') as h:
            json.dump({'v': {'kind': 'default', **entry}}, h)
        return load_voices(p)['v']

    def test_a_band_in_the_document_lands_on_the_voice_for_the_CLIENT(self):
        """The document's three keys still reach the voice - and NOTHING in
        narrator builds a tracker from them (Owen, 2026-09-19).

        They are parsed and checked because they are what the CLIENT reads off
        the voice row and sends back as a batch's `band`; the band has one
        owner and it is the caller. `tracker_for` takes the band explicitly,
        so the voice object is not even an argument it would accept."""
        v = self._load({'maxCharsPerSec': 21.7, 'minCharsPerSec': 13.6, 'paceCharsPerSec': 17.2})
        self.assertEqual((v.max_chars_per_sec, v.min_chars_per_sec, v.pace_chars_per_sec),
                         (21.7, 13.6, 17.2))
        bare = self._load({})
        self.assertIsNone(bare.max_chars_per_sec)
        self.assertIsNone(bare.pace_chars_per_sec)
        for voice in (v, bare):
            with self.subTest(voice=voice.name):
                with self.assertRaises(ValueError) as caught:
                    truncation.tracker_for(voice, 'VoiceBandTest')
                self.assertIn('must be an object', str(caught.exception))

    def test_the_tracker_is_built_from_the_band_it_is_GIVEN(self):
        """The band is an argument, and the only one. A tracker seeded from a
        band that belongs to a different voice - or to the engine, when the
        voice had none - is the defect this signature exists to make
        impossible: 15.0 against a 17.2 chars/s book re-rolled healthy chunks
        to MAX_DEPTH."""
        tracker = truncation.tracker_for(
            {'paceCharsPerSec': 17.2, 'maxCharsPerSec': 21.7,
             'minCharsPerSec': 13.6})
        self.assertEqual(tracker.seed_pace, 17.2)
        self.assertEqual(tracker.band(),
                         {'max_chars_per_sec': 21.7, 'min_chars_per_sec': 13.6})
        engine = truncation.tracker_for(truncation.engine_band(20.0, 14.5))
        self.assertEqual(engine.band(),
                         {'max_chars_per_sec': 20.0, 'min_chars_per_sec': 14.5})
        self.assertAlmostEqual(engine.seed_pace,
                               truncation.expected_chars_per_sec(20.0, 14.5))

    def test_a_band_off_the_wire_is_refused_by_name(self):
        """`parse_band` is the one door, so the batch door and the engines
        cannot disagree about what a band is."""
        good = {'paceCharsPerSec': 17.2, 'maxCharsPerSec': 21.7,
                'minCharsPerSec': 13.6}
        for band, why in (
                ({k: v for k, v in good.items() if k != 'paceCharsPerSec'},
                 'paceCharsPerSec'),
                ({**good, 'minCharsPerSec': 0}, 'positive'),
                ({**good, 'maxCharsPerSec': True}, 'positive'),
                ({**good, 'paceCharsPerSec': 30.0}, 'min < pace < max'),
                ('17.2', 'must be an object')):
            with self.subTest(band=band):
                with self.assertRaises(ValueError) as caught:
                    truncation.parse_band(band, 'where')
                self.assertIn(why, str(caught.exception))

    def test_a_partial_band_or_an_inverted_one_is_refused_by_name(self):
        for entry, why in (({'maxCharsPerSec': 21.7}, 'all three or none'),
                           ({'maxCharsPerSec': 21.7, 'minCharsPerSec': 13.6}, 'all three or none'),
                           ({'maxCharsPerSec': 10.0, 'minCharsPerSec': 14.5, 'paceCharsPerSec': 12.0},
                            'min < pace < max'),
                           ({'maxCharsPerSec': 21.7, 'minCharsPerSec': 13.6, 'paceCharsPerSec': 30.0},
                            'min < pace < max'),
                           ({'maxCharsPerSec': -1, 'minCharsPerSec': 14.5, 'paceCharsPerSec': 17.0},
                            'positive')):
            with self.assertRaises(ValueError) as caught:
                self._load(entry)
            self.assertIn(why, str(caught.exception))


class ServedEngineTest(SglTestCase):
    """The ladder through the real served engine against the fake SGLang server:
    the server answers a FIXED number of seconds, so a long chunk is 'short' and
    its halves are not - the ladder must split once and write the joined audio."""

    def test_convert_splits_a_chunk_the_server_stops_early_on(self):
        # 20 s for any request: TEXT (~1,300 chars) reads 65 chars/s -> short;
        # a half (~650) reads 32 -> short; a quarter (~325) reads 16 -> fine.
        self.server.httpd.seconds = 20.0
        checkpoint = self.merged_checkpoint()
        state_env(self, served_common.CHECKPOINT_ENV, checkpoint)
        config = HiggsV3Config(voice=self.checkpoint_voice(), base_url=self.server.base_url,
                               checkpoint_dir=checkpoint, sentences_dir=self.dir, seed=7)
        engine = HiggsV3Engine(config)
        self.addCleanup(engine.cleanup)
        engine.load_engine()
        self.assertTrue(engine.convert(19, TEXT))
        texts = [json_body['input'] for json_body in self.server.requests]
        self.assertEqual(texts[0], TEXT)
        self.assertEqual(texts[1], TEXT, 'the re-roll renders the whole text once more')
        self.assertNotEqual(self.server.requests[0].get('seed'), self.server.requests[1].get('seed'),
                            'the re-roll must use a different seed')
        self.assertGreater(len(texts), 4, 'the halves were split again into quarters')
        import soundfile as sf
        audio, rate = sf.read(os.path.join(self.dir, f'19.{config.audio_format}'))
        self.assertEqual(rate, 24000)
        # four quarters at 20 s each plus three joins
        self.assertGreater(len(audio) / rate, 4 * 20.0)


class GuardPlanTest(unittest.TestCase):
    """The BATCH driver (Owen, 2026-09-08: "take note of which ones we need to
    re-render and add them to the task list, and batch them at the end").

    The policy is `render_guarded`'s - what these prove is that the retakes of a
    whole batch arrive in ONE round, and that the two drivers ship the same
    audio and log the same events.
    """

    TEXTS = [f'Chunk {tag}. ' + TEXT for tag in 'ABCDE']

    def plan(self, events, tracker=None):
        return truncation.GuardPlan(sample_rate=RATE, base_seed=1234, tracker=tracker,
                                    max_chars_per_sec=20.0, min_chars_per_sec=14.5,
                                    on_event=events.append)

    @staticmethod
    def drive(plan, render):
        """The batch driver's loop; answers what each round asked for."""
        rounds = []
        shipped = []
        while True:
            requests = plan.round()
            if not requests:
                return rounds, shipped
            rounds.append(list(requests))
            for request in requests:
                plan.offer(request, render(request.text, request.seed))
            shipped.extend(plan.finished())

    def add_all(self, plan, render):
        """Take 0 for every chunk, the way a batch hands it in."""
        for index, text in enumerate(self.TEXTS):
            plan.add(index, text, first_take=render(text, None))
        return plan.finished()

    def test_the_off_length_rows_of_a_batch_all_ask_in_ONE_round(self):
        # Chunks 1 and 3 stop early on their own seed and are fine on any other:
        # serially that is two solo re-rolls, which on MLX costs two batches.
        render = FakeRender(lambda t, s, n: s is None and ('Chunk B' in t or 'Chunk D' in t))
        events = []
        plan = self.plan(events)
        clean = self.add_all(plan, render)
        self.assertEqual([index for index, _audio, _clean in clean], [0, 2, 4],
                         'the rows take 0 got right ship before any retake runs')
        rounds, shipped = self.drive(plan, render)
        self.assertEqual([len(r) for r in rounds], [2], 'one round, both re-rolls in it')
        self.assertEqual([(r.index, r.rung) for r in rounds[0]],
                         [(1, 'reroll'), (3, 'reroll')])
        self.assertEqual([r.seed for r in rounds[0]],
                         [truncation.reroll_seed(1234, 1, 1), truncation.reroll_seed(1234, 3, 1)])
        self.assertEqual([index for index, _audio, _clean in shipped], [1, 3])
        self.assertEqual([e['action'] for e in events], ['short', 'short', 'rerolled', 'rerolled'])

    def test_the_split_halves_of_several_chunks_flatten_into_one_round(self):
        # Nothing fixes the whole text of B and D; their halves are fine (the
        # script fires on length, the shape of a real early stop). Round 1 is
        # the two re-rolls, round 2 is FOUR halves - one batch, not four solos.
        render = FakeRender(
            lambda t, s, n: len(t) > 900 and ('Chunk B' in t or 'Chunk D' in t))
        events = []
        plan = self.plan(events)
        self.add_all(plan, render)
        rounds, shipped = self.drive(plan, render)
        self.assertEqual([len(r) for r in rounds], [2, 4])
        self.assertEqual([(r.index, r.rung) for r in rounds[1]],
                         [(1, 'part'), (1, 'part'), (3, 'part'), (3, 'part')])
        self.assertEqual(sorted(index for index, _audio, _clean in shipped), [1, 3])
        for index, audio, clean in shipped:
            halves = truncation.split_halves(self.TEXTS[index])
            self.assertTrue(clean, 'both halves came out clean, so the chunk did')
            self.assertEqual(len(audio),
                             len(audio_for(len(halves[0]))) + len(audio_for(len(halves[1])))
                             + int(round(truncation.RESPLIT_JOIN_SECONDS * RATE)))

    def test_the_two_drivers_ship_the_same_audio_and_log_the_same_events(self):
        """The drift guard: same scripted renders, one chunk at a time through
        `render_guarded` and all five through a plan."""
        def script(t, s, n):
            return 'Chunk B' in t or ('Chunk D' in t and s is None)

        serial_events, serial_audio = [], []
        for index, text in enumerate(self.TEXTS):
            render = FakeRender(script)
            serial_audio.append(len(truncation.render_guarded(
                render, text, index, sample_rate=RATE, max_chars_per_sec=20.0,
                min_chars_per_sec=14.5, base_seed=1234,
                first_take=render(text, None), on_event=serial_events.append)))

        batched_events = []
        plan = self.plan(batched_events)
        render = FakeRender(script)
        decided = list(self.add_all(plan, render))
        _rounds, shipped = self.drive(plan, render)
        decided.extend(shipped)
        batched_audio = [length for _index, length
                         in sorted((index, len(audio)) for index, audio, _clean in decided)]

        self.assertEqual(batched_audio, serial_audio)
        def by_index(events):
            return {index: [e['action'] for e in events if e['index'] == index]
                    for index in range(len(self.TEXTS))}
        self.assertEqual(by_index(batched_events), by_index(serial_events))

    def test_only_a_clean_take_feeds_the_tracker_through_the_batch_driver(self):
        tracker = truncation.PaceTracker(17.0, 20.0, 14.5, warmup=1)
        # Chunk B can never be fixed: it is accepted off-length and must not
        # pull the running median toward the defect.
        render = FakeRender(lambda t, s, n: 'Chunk B' in t)
        events = []
        plan = self.plan(events, tracker=tracker)
        self.add_all(plan, render)
        self.drive(plan, render)
        self.assertIn('accepted-off-length', [e['action'] for e in events])
        self.assertEqual(tracker.observed, 4, 'the four clean chunks, not the accepted one')
        self.assertAlmostEqual(tracker.reference, PACE, delta=0.05)

    def test_a_take_offered_to_a_chunk_that_is_not_waiting_is_refused(self):
        events = []
        plan = self.plan(events)
        plan.add(7, TEXT, first_take=audio_for(len(TEXT)))    # clean: nothing pending
        self.assertEqual(plan.round(), [])
        self.assertEqual(plan.pending, 0)
        request = truncation.RenderRequest((0,), 7, TEXT, None, 'take', 0)
        with self.assertRaises(RuntimeError):
            plan.offer(request, audio_for(len(TEXT)))
        with self.assertRaises(KeyError):
            plan.offer(truncation.RenderRequest((9,), 9, TEXT, None, 'take', 0),
                       audio_for(len(TEXT)))


class RejectKeepingTest(unittest.TestCase):
    """The take the guard threw away is KEPT when the run names a directory.

    Owen, 2026-09-08: "we should probably whisper them and see what was missing.
    how much was missing, where it stopped, etc." - which needs the audio, and
    the re-roll overwrites it.
    """

    def setUp(self):
        import tempfile
        self.dir = tempfile.mkdtemp(prefix='higgs-rejects-')
        self.addCleanup(lambda: __import__('shutil').rmtree(self.dir, ignore_errors=True))
        self._env = os.environ.get(truncation.REJECT_DIR_ENV)
        self.addCleanup(self._restore)

    def _restore(self):
        if self._env is None:
            os.environ.pop(truncation.REJECT_DIR_ENV, None)
        else:
            os.environ[truncation.REJECT_DIR_ENV] = self._env

    def test_a_discarded_take_lands_as_wav_json_and_one_event_line(self):
        os.environ[truncation.REJECT_DIR_ENV] = self.dir
        # Take 0 stops early, the re-roll is fine: exactly one take is thrown away.
        render = FakeRender(lambda t, s, n: s is None)
        truncation.render_guarded(render, TEXT, 19, sample_rate=RATE,
                                  max_chars_per_sec=20.0, min_chars_per_sec=14.5,
                                  base_seed=1234, on_event=lambda e: None)
        names = sorted(os.listdir(self.dir))
        self.assertEqual(names, ['000019_d0_take0_short.json', '000019_d0_take0_short.wav',
                                 'events.jsonl'])
        with open(os.path.join(self.dir, '000019_d0_take0_short.json'), encoding='utf-8') as handle:
            record = json.load(handle)
        self.assertEqual(record['index'], 19)
        self.assertEqual(record['side'], 'short')
        self.assertEqual(record['text'], TEXT, 'the diff needs the text, and the session may be gone')
        self.assertEqual(record['chars'], len(TEXT))
        self.assertEqual(record['sample_rate'], RATE)
        with open(os.path.join(self.dir, 'events.jsonl'), encoding='utf-8') as handle:
            self.assertEqual(len(handle.read().strip().splitlines()), 1)
        import soundfile as sf
        audio, rate = sf.read(os.path.join(self.dir, '000019_d0_take0_short.wav'))
        self.assertEqual(rate, RATE)
        self.assertEqual(len(audio), len(audio_for(50)), 'the take as it was, not the re-roll')

    def test_both_thrown_away_takes_are_kept_when_the_chunk_is_split(self):
        os.environ[truncation.REJECT_DIR_ENV] = self.dir
        render = FakeRender(lambda t, s, n: t == TEXT)   # whole text always stops early
        truncation.render_guarded(render, TEXT, 7, sample_rate=RATE,
                                  max_chars_per_sec=20.0, min_chars_per_sec=14.5,
                                  base_seed=1234, on_event=lambda e: None)
        stems = sorted(n for n in os.listdir(self.dir) if n.endswith('.wav'))
        self.assertEqual(stems, ['000007_d0_reroll_short.wav', '000007_d0_take0_short.wav'],
                         'take 0 AND the re-roll, each named by its rung')

    def test_no_directory_named_keeps_nothing_and_renders_the_same(self):
        os.environ.pop(truncation.REJECT_DIR_ENV, None)
        render = FakeRender(lambda t, s, n: s is None)
        out = truncation.render_guarded(render, TEXT, 19, sample_rate=RATE,
                                        max_chars_per_sec=20.0, min_chars_per_sec=14.5,
                                        base_seed=1234, on_event=lambda e: None)
        self.assertEqual(len(out), len(audio_for(len(TEXT))))
        self.assertEqual(os.listdir(self.dir), [])

    def test_an_unwritable_directory_is_logged_and_never_raises(self):
        # Diagnostics must never take a book down: the reject dir is a path from
        # the environment and the render has no say in whether it is writable.
        os.environ[truncation.REJECT_DIR_ENV] = os.path.join(self.dir, 'nul-file', 'deeper')
        with open(os.path.join(self.dir, 'nul-file'), 'w', encoding='utf-8') as handle:
            handle.write('not a directory')
        render = FakeRender(lambda t, s, n: s is None)
        out = truncation.render_guarded(render, TEXT, 19, sample_rate=RATE,
                                        max_chars_per_sec=20.0, min_chars_per_sec=14.5,
                                        base_seed=1234, on_event=lambda e: None)
        self.assertEqual(len(out), len(audio_for(len(TEXT))), 'the re-roll still shipped')


if __name__ == '__main__':
    unittest.main()


def _speech(seconds: float) -> np.ndarray:
    """Audio that reads as speech to the hole scan: a flat 0.1 (-20 dBFS) line."""
    return np.full(int(seconds * RATE), 0.1, dtype=np.float32)


def _take_with_hole(chars: int, hole_seconds: float) -> np.ndarray:
    """A take of the NORMAL length for `chars` whose middle is `hole_seconds` of
    silence - the Fuhrer (5) shape (13.flac: 44.5 s, complete text, 11.3 s hole)."""
    total = chars / PACE
    side = max(0.5, (total - hole_seconds) / 2)
    return np.concatenate([_speech(side), np.zeros(int(hole_seconds * RATE), dtype=np.float32),
                           _speech(side)])


class HoleGuardTest(unittest.TestCase):
    """Fix B, 2026-09-11 (pausing-problem channel, finetuning-pc-1 leading): the
    silence loop the rate band cannot see, and the split-child gap it shipped
    through."""

    def test_interior_hole_ignores_tails_and_measures_the_middle(self):
        self.assertEqual(truncation.interior_hole_seconds(np.zeros(RATE * 10, dtype=np.float32), RATE), 0.0)
        tails_only = np.concatenate([np.zeros(RATE * 8, dtype=np.float32), _speech(2), np.zeros(RATE * 8, dtype=np.float32)])
        self.assertEqual(truncation.interior_hole_seconds(tails_only, RATE), 0.0)
        holed = np.concatenate([_speech(3), np.zeros(RATE * 6, dtype=np.float32), _speech(3)])
        self.assertAlmostEqual(truncation.interior_hole_seconds(holed, RATE), 6.0, places=1)
        self.assertEqual(truncation.interior_hole_seconds(np.zeros(0, dtype=np.float32), RATE), 0.0)

    def test_a_hole_over_the_ceiling_fires_and_the_clean_reroll_ships(self):
        text = 'x' * 600
        takes = [_take_with_hole(600, 6.0), audio_for(600)]
        render = lambda t, s: takes.pop(0)   # noqa: E731
        events = []
        out = truncation.render_guarded(render, text, 7, sample_rate=RATE, max_chars_per_sec=20.0,
                                        min_chars_per_sec=14.5, base_seed=1, on_event=events.append)
        self.assertEqual(events[0]['action'], 'hole')
        self.assertEqual(events[0]['side'], 'hole')
        self.assertGreater(events[0]['hole_seconds'], 5.0)
        self.assertEqual(events[-1]['action'], 'rerolled')
        self.assertEqual(len(out), len(audio_for(600)))

    def test_a_hole_under_the_ceiling_is_a_pause_and_passes(self):
        text = 'x' * 600
        events = []
        out = truncation.render_guarded(lambda t, s: _take_with_hole(600, 4.0), text, 7, sample_rate=RATE,
                                        max_chars_per_sec=20.0, min_chars_per_sec=14.5, base_seed=1,
                                        on_event=events.append)
        self.assertEqual(events, [])
        self.assertAlmostEqual(len(out) / RATE, 600 / PACE, delta=0.2)

    def test_between_two_holed_takes_the_smaller_hole_ships(self):
        text = 'x' * 160    # one 'sentence': cannot split, so rung 3 must pick
        takes = [_take_with_hole(160, 7.0), _take_with_hole(160, 5.5)]
        events = []
        out = truncation.render_guarded(lambda t, s: takes.pop(0), text, 7, sample_rate=RATE,
                                        max_chars_per_sec=20.0, min_chars_per_sec=14.5, base_seed=1,
                                        on_event=events.append)
        self.assertEqual(events[-1]['action'], 'accepted-off-length')
        self.assertEqual(events[-1]['shipped_side'], 'hole')
        self.assertAlmostEqual(truncation.interior_hole_seconds(out, RATE), 5.5, places=1)

    def test_a_split_child_under_MIN_GUARD_CHARS_is_still_judged_on_the_long_side(self):
        # Two sentences, ~250 chars: the parent runs long twice and splits into
        # two halves UNDER MIN_GUARD_CHARS (150). Before fix B the halves had no
        # long side, so a child that ran to the cap shipped (Fuhrer (5) chunk 8).
        text = ('The first sentence of this chunk is long enough to matter here, and it keeps going a while. '
                'And the second sentence of this chunk is about the same length as the first one, more or less.')
        self.assertLess(len(text), 2 * truncation.MIN_GUARD_CHARS)
        self.assertGreaterEqual(len(text), truncation.MIN_GUARD_CHARS)
        render = FakeRender(lambda t, s, n: False, long=lambda t, s, n: n <= 3)
        events = []
        out = truncation.render_guarded(render, text, 7, sample_rate=RATE, max_chars_per_sec=20.0,
                                        min_chars_per_sec=14.5, base_seed=1, on_event=events.append)
        child_fires = [e for e in events if e.get('depth') == 1 and e['action'] == 'long']
        self.assertTrue(child_fires, f'no long-side verdict on a split child: {[(e.get("depth"), e["action"]) for e in events]}')
        self.assertLess(len(child_fires[0]['text'] if 'text' in child_fires[0] else 'x' * 149), 150)
        self.assertEqual(events[-1]['action'], 'rerolled')
        self.assertEqual(events[-1]['depth'], 1)
        self.assertGreater(len(out), 0)
