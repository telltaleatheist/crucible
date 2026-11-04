"""THE PER-ITEM TAKE CHANNEL: one item, one rung, and its four refusals.

A RUNG IS TWO FACTS - the sampling DELTAS and the SEED OFFSET - and this file
proves both. The numbers landed on 2026-09-14; the seed landed on 2026-09-15,
when it was measured that narrator seeded chunk i at `config.seed + i` on both
Higgs arms whatever the take, so take 0 and take N of one chunk were
byte-identical whenever their sampling matched, and two take-0 re-rolls always
were. A seed is a setting, and Owen's ruling is that a retake must not reuse
the settings that produced the problem.

WHAT IS UNDER TEST AND WHY IT EXISTS. Owen ruled on 2026-09-14
(docs/EXTENSION-TO-CRUCIBLE-PLAN.md section 2) that *a retake must not reuse
the settings that produced the problem; the spread IS the take ladder*.
Crucible's ladder (crucible/docs/PHASE3-TTS.md section 3) is per VOICE, a job
carries `take: N`, and the server resolves the rung into numbers. It had
nowhere to send them: narrator's sampling arrived through the voices document,
which is written per LOAD, so Crucible refused every take above 0 by name -
`sampling_not_wired`, PHASE3-TTS section 4, "a rung is per RENDER and
narrator's `generate_batch` takes no sampling, so a ladder has no channel yet".
This file is the proof the channel is there.

THE SPELLING IS THE VOICES DOCUMENT'S, and `WireSpellingTest` below pins it to
`engine/higgs/config.py:_SAMPLING_KEYS` so the per-load channel and the
per-item channel cannot drift into two names for one lever.

HOW THE PROTOCOL TESTS SEE WHAT WAS APPLIED. They drive a real worker
SUBPROCESS, so the fake engine writes one JSON line per render to the file
named by `NARRATOR_FAKE_RENDER_LOG` (`FakeHiggsEngine._record_render`) and
the assertions read it back. What is recorded is the sampling the render
actually RAN at, not the sampling that was asked for, which is the only version
of the question worth asking.

THE SEED IS WHAT MAKES A TAKE VISIBLE AT ALL. On this wire a rung may change
the seed and nothing else - `take: 3` with no `sampling` is a legal item - so
the fake records the seed it actually drew at beside the sampling it actually
ran under, and `TakeOnTheWireTest` asserts on that. The lane
arithmetic and its disjointness from the guard's own re-roll seeds are
`TakeSeedLaneTest`, against `engine/higgs/truncation.py:TAKE_SEED_STRIDE`.

NO GPU, NO MODEL: `--fake-engine`, sine tones, CPU only.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_ROOT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "crucible", "jobs", "tts", "higgs")   # holds crucible_higgs
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)

from crucible_higgs.engine import item_sampling as S                       # noqa: E402
from tests.higgs_worker.test_engine_serve_protocol import (         # noqa: E402
    Worker, _WorkerCase, fake_higgs_band)


# ---------------------------------------------------------------------------
# The parser, on its own
# ---------------------------------------------------------------------------


class WireSpellingTest(unittest.TestCase):
    """One vocabulary for both channels."""

    def test_the_item_wire_spells_the_levers_exactly_as_the_document_does(self):
        """A second spelling would be two names for one fact - the shape
        crucible/docs/ARCHITECTURE.md's audit found seven times in one night."""
        from crucible_higgs.engine.higgs.config import _SAMPLING_KEYS
        self.assertEqual(S.WIRE_KEYS, _SAMPLING_KEYS)

    def test_the_refusal_names_are_the_ones_crucible_will_read(self):
        self.assertEqual(S.MALFORMED, 'sampling_malformed')
        self.assertEqual(S.NOT_SUPPORTED, 'sampling_not_supported')


class ParseTest(unittest.TestCase):

    def test_absent_is_none_and_is_not_a_refusal(self):
        """No `sampling` key means take 0, which is a documented meaning and
        not a value substituted for a missing one."""
        self.assertIsNone(S.parse_item_sampling(None, 'row 0'))

    def test_the_document_spelling_becomes_the_engine_spelling(self):
        self.assertEqual(
            S.parse_item_sampling({'temperature': 0.7, 'topP': 0.95, 'topK': 50},
                                  'row 0'),
            {'temperature': 0.7, 'top_p': 0.95, 'top_k': 50})

    def test_top_k_comes_back_a_whole_number(self):
        parsed = S.parse_item_sampling({'topK': 50}, 'row 0')
        self.assertIsInstance(parsed['top_k'], int)

    def test_a_partial_rung_is_legal_because_a_rung_is_an_overlay(self):
        """PHASE3-TTS's take 1 is one line, `temperature = 0.7`. It means
        "take 0, but cooler"."""
        self.assertEqual(S.parse_item_sampling({'temperature': 0.7}, 'row 0'),
                         {'temperature': 0.7})

    def test_a_non_object_is_malformed_by_name(self):
        for bad in (0.7, 'hot', [0.7], True):
            with self.assertRaises(S.SamplingMalformed) as caught:
                S.parse_item_sampling(bad, 'row 4')
            self.assertTrue(str(caught.exception).startswith(S.MALFORMED))
            self.assertIn('row 4', str(caught.exception))

    def test_an_empty_object_is_malformed_by_name(self):
        """`{}` is not "no rung" - `sampling` absent is. An empty object is a
        caller that meant to say something and said nothing."""
        with self.assertRaises(S.SamplingMalformed):
            S.parse_item_sampling({}, 'row 4')

    def test_an_unknown_lever_is_malformed_and_the_field_is_named(self):
        with self.assertRaises(S.SamplingMalformed) as caught:
            S.parse_item_sampling({'top_p': 0.95}, 'row 4')
        self.assertTrue(str(caught.exception).startswith(S.MALFORMED))
        self.assertIn('top_p', str(caught.exception))

    def test_a_non_positive_or_non_numeric_value_is_malformed(self):
        for bad in ({'temperature': 0}, {'temperature': -0.7},
                    {'topP': 'hot'}, {'topK': True}):
            with self.assertRaises(S.SamplingMalformed) as caught:
                S.parse_item_sampling(bad, 'row 4')
            self.assertIn(list(bad)[0], str(caught.exception))

    def test_a_fractional_top_k_is_malformed_by_name(self):
        with self.assertRaises(S.SamplingMalformed) as caught:
            S.parse_item_sampling({'topK': 50.5}, 'row 4')
        self.assertIn('topK', str(caught.exception))

    def test_a_lever_this_engine_has_not_got_is_NOT_SUPPORTED_not_malformed(self):
        """The difference matters to whoever sent it: one is a typo, the other
        is the wrong backend. mlx-audio has no repetition penalty."""
        with self.assertRaises(S.SamplingNotSupported) as caught:
            S.parse_item_sampling({'repetitionPenalty': 1.1}, 'row 4',
                                  levers=('temperature', 'topP', 'topK'))
        self.assertTrue(str(caught.exception).startswith(S.NOT_SUPPORTED))
        self.assertIn('repetitionPenalty', str(caught.exception))

    def test_an_engine_with_no_channel_refuses_by_name_and_says_why(self):
        with self.assertRaises(S.SamplingNotSupported) as caught:
            S.refuse_item_sampling({'temperature': 0.7}, 'Orpheus', 'because.')
        self.assertTrue(str(caught.exception).startswith(S.NOT_SUPPORTED))
        self.assertIn('because.', str(caught.exception))
        # ...and absent is still absent, even there.
        self.assertIsNone(S.refuse_item_sampling(None, 'Orpheus', 'because.'))


class OverlayTest(unittest.TestCase):

    def test_a_rung_overlays_and_does_not_replace(self):
        """THE MEASURED REASON: on SGLang-Omni an unset top_k is the
        untruncated 1026-way codebook tail - one chunk ran to the cap with 80 s
        of silence (2026-09-05). A partial rung that REPLACED the voice's
        sampling would produce exactly that."""
        base = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 50}
        self.assertEqual(S.apply_over(base, {'temperature': 0.7}),
                         {'temperature': 0.7, 'top_p': 0.95, 'top_k': 50})

    def test_no_rung_leaves_the_base_exactly_as_it_was(self):
        base = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 50}
        self.assertEqual(S.apply_over(base, None), base)

    def test_neither_argument_is_mutated(self):
        base = {'temperature': 1.0}
        item = {'temperature': 0.7}
        S.apply_over(base, item)
        self.assertEqual(base, {'temperature': 1.0})
        self.assertEqual(item, {'temperature': 0.7})


class AlignedAndGroupedTest(unittest.TestCase):

    def test_none_means_no_rung_anywhere(self):
        self.assertEqual(S.aligned(None, 3, 'where'), [None, None, None])

    def test_a_misaligned_list_is_refused_by_name(self):
        with self.assertRaises(S.SamplingMalformed) as caught:
            S.aligned([{'temperature': 0.7}], 3, 'where')
        self.assertTrue(str(caught.exception).startswith(S.MALFORMED))

    def test_the_group_key_is_the_numbers_not_the_dict(self):
        self.assertEqual(S.group_key({'temperature': 0.7, 'top_p': 0.95}),
                         S.group_key({'top_p': 0.95, 'temperature': 0.7}))
        self.assertNotEqual(S.group_key({'temperature': 0.7}),
                            S.group_key({'temperature': 0.8}))
        self.assertIsNone(S.group_key(None))


# ---------------------------------------------------------------------------
# The wire, through a real worker subprocess
# ---------------------------------------------------------------------------


class _SamplingWorkerCase(_WorkerCase):
    """A worker whose fake engine writes down what it rendered each row under."""

    ENGINE_ID = 'higgs-v3'
    VOICE = 'deathstalker'

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='narrator-sampling-')
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.log_path = os.path.join(self.root, 'renders.jsonl')
        self.WORKER_ENV = {'NARRATOR_ENGINE': self.ENGINE_ID,
                           'NARRATOR_FAKE_RENDER_LOG': self.log_path}
        super().setUp()

    def rendered(self):
        """Every render the fake made, in order:
        `[{index, text, sampling, take, seed}]`."""
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, encoding='utf-8') as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def sampling_by_index(self):
        """The sampling each chunk index rendered under. Asserts that every
        take of one chunk ran at the SAME numbers - a re-roll or a split half
        that quietly dropped the rung is the bug this channel exists to
        prevent."""
        out = {}
        for row in self.rendered():
            index = row['index']
            if index in out:
                self.assertEqual(out[index], row['sampling'],
                                 f'chunk {index} rendered two takes at different '
                                 'sampling')
            out[index] = row['sampling']
        return out

    #: Whether this case's batches ask to be JUDGED. The guarded class turns
    #: it on; every other class here is about a per-ITEM refusal on the bare
    #: arm, where nothing judges and no band is needed (Owen, 2026-09-19).
    RETAKE = False

    def batch(self, items, voice=None, **extra):
        self._ready()
        self._load(voice or self.VOICE)
        request = dict(extra)
        if self.RETAKE and 'retake' not in request:
            # `retake` and `band` travel together or the batch is refused by
            # name - the band has one owner and it is the caller.
            request['retake'] = True
            request['band'] = fake_higgs_band()
        self.w.send(action='generate_batch', items=items, **request)
        msgs = self.w.read_until('batch_done')
        return self._assert_batch_closed(msgs, [it['i'] for it in items])


class GuardedBatchSamplingTest(_SamplingWorkerCase):
    """The NON-STREAMING door - the one Crucible's render job drives.

    JUDGED, because that is what this class is about: the rung must survive
    every take of a chunk, and a chunk only has more than one take when the
    ladder runs. `RETAKE` puts `retake: true` and the fake's own band on every
    batch below.
    """

    RETAKE = True

    def test_the_worker_really_built_the_higgs_fake(self):
        """If this fails, every other test in the class is testing Orpheus."""
        self._ready()
        loaded = self._load(self.VOICE)[-1]
        self.assertEqual(loaded['type'], 'loaded', loaded)
        self.assertEqual(loaded['engine'], 'higgs-v3')

    def test_absent_sampling_renders_at_the_voices_default(self):
        """Take 0. Not a fallback: the engine was loaded with its sampling and
        an item that names no rung is not asking for a different one."""
        by_i = self.batch([{'i': 0, 'text': 'The first chunk of the chapter.'},
                           {'i': 1, 'text': 'And the second one after it.'}])
        for i in (0, 1):
            self.assertIn('data', by_i[i], by_i[i])
        self.assertEqual(self.sampling_by_index(), {0: None, 1: None})

    def test_a_rung_is_honoured_for_the_item_that_carried_it(self):
        by_i = self.batch([
            {'i': 0, 'text': 'A chunk at the boson default.'},
            {'i': 1, 'text': 'A chunk on rung one, cooler.',
             'sampling': {'temperature': 0.7}},
        ])
        for i in (0, 1):
            self.assertIn('data', by_i[i], by_i[i])
        self.assertEqual(self.sampling_by_index(),
                         {0: None, 1: {'temperature': 0.7}})

    def test_one_batch_may_mix_rungs_and_each_row_gets_its_own(self):
        """The retake spread: Owen's ruling is that candidates are spread
        ACROSS rungs, so a batch carrying three different ones is the normal
        case and not an edge."""
        by_i = self.batch([
            {'i': 10, 'text': 'Row ten, take zero and nothing asked for.'},
            {'i': 11, 'text': 'Row eleven, rung one.',
             'sampling': {'temperature': 0.7}},
            {'i': 12, 'text': 'Row twelve, a whole triple.',
             'sampling': {'temperature': 0.9, 'topP': 0.9, 'topK': 40}},
        ])
        for i in (10, 11, 12):
            self.assertIn('data', by_i[i], by_i[i])
        self.assertEqual(
            self.sampling_by_index(),
            {10: None,
             11: {'temperature': 0.7},
             12: {'temperature': 0.9, 'top_p': 0.9, 'top_k': 40}})

    def test_every_take_of_a_chunk_runs_at_that_chunks_rung(self):
        """A re-roll and a split half carry their parent's index, and the whole
        point of a retake at another temperature is that the retake uses it.
        NARRATOR_FAKE_HIGGS_RATE bends chunk 1's take 0 so the ladder actually
        climbs."""
        self.w.close()
        env = dict(self.WORKER_ENV)
        env['NARRATOR_FAKE_HIGGS_RATE'] = json.dumps({'1': 0.4})
        self.w = Worker(extra_env=env)
        by_i = self.batch([
            {'i': 0, 'text': 'An ordinary opening chunk of the chapter.'},
            {'i': 1, 'text': 'The chunk whose first take comes back far too '
                             'short for its text, so the guard re-rolls it.',
             'sampling': {'temperature': 0.7}},
        ])
        self.assertIn('data', by_i[1], by_i[1])
        rows = [r for r in self.rendered() if r['index'] == 1]
        self.assertGreater(len(rows), 1,
                           'the guard never re-rolled; this test proves nothing')
        # sampling_by_index asserts the equality across takes; state it here too
        # so the failure reads as what it is.
        for row in rows:
            self.assertEqual(row['sampling'], {'temperature': 0.7})

    def test_a_malformed_rung_fails_ONLY_that_row_and_names_the_refusal(self):
        by_i = self.batch([
            {'i': 0, 'text': 'A perfectly ordinary neighbouring sentence.'},
            {'i': 1, 'text': 'The row with a broken rung.', 'sampling': 'hot'},
            {'i': 2, 'text': 'Another ordinary sentence behind it.'},
        ])
        self.assertIn('data', by_i[0])
        self.assertIn('data', by_i[2])
        self.assertNotIn('data', by_i[1])
        self.assertTrue(by_i[1]['message'].startswith(S.MALFORMED), by_i[1])
        self.assertIn('i=1', by_i[1]['message'])
        # The refused row never reached the engine.
        self.assertEqual(sorted(self.sampling_by_index()), [0, 2])

    def test_an_unknown_lever_names_the_field(self):
        by_i = self.batch([{'i': 3, 'text': 'A row.',
                            'sampling': {'temperatur': 0.7}}])
        self.assertTrue(by_i[3]['message'].startswith(S.MALFORMED), by_i[3])
        self.assertIn('temperatur', by_i[3]['message'])

    def test_a_fractional_top_k_names_the_field(self):
        by_i = self.batch([{'i': 3, 'text': 'A row.',
                            'sampling': {'topK': 50.5}}])
        self.assertTrue(by_i[3]['message'].startswith(S.MALFORMED), by_i[3])
        self.assertIn('topK', by_i[3]['message'])

    def test_a_lever_this_backend_cannot_honour_is_refused_by_the_OTHER_name(self):
        """`sampling_not_supported`, not `sampling_malformed`: the value is
        well formed and this engine simply has no such lever."""
        by_i = self.batch([{'i': 3, 'text': 'A row.',
                            'sampling': {'repetitionPenalty': 1.1}}])
        self.assertTrue(by_i[3]['message'].startswith(S.NOT_SUPPORTED), by_i[3])
        self.assertIn('repetitionPenalty', by_i[3]['message'])


class StreamingSamplingTest(_SamplingWorkerCase):
    """The STREAM door. Unguarded by ruling (Owen, 2026-09-13), which changes
    nothing about whose numbers a row renders at."""

    def test_a_streamed_row_renders_at_its_own_rung(self):
        by_i = self.batch([
            {'i': 0, 'text': 'The row being listened to right now.',
             'stream': True, 'sampling': {'temperature': 0.7}},
            {'i': 1, 'text': 'A read-ahead row behind it, at take zero.'},
        ])
        self.assertTrue(by_i[0].get('streamed'), by_i[0])
        self.assertIn('data', by_i[1], by_i[1])
        self.assertEqual(self.sampling_by_index(),
                         {0: {'temperature': 0.7}, 1: None})

    def test_a_malformed_rung_on_a_streamed_batch_fails_only_its_row(self):
        by_i = self.batch([
            {'i': 0, 'text': 'A streamed row that is fine.', 'stream': True},
            {'i': 1, 'text': 'A row with a broken rung.', 'sampling': {}},
        ])
        self.assertTrue(by_i[0].get('streamed'), by_i[0])
        self.assertTrue(by_i[1]['message'].startswith(S.MALFORMED), by_i[1])


class SingleGenerateSamplingTest(_SamplingWorkerCase):
    """`generate` - the one-sentence door. Same channel, same two refusals,
    reported as this door's `error` rather than a `batch_item`."""

    def _generate(self, **request):
        self._ready()
        self._load(self.VOICE)
        self.w.send(action='generate', **request)
        return self.w.read_until('audio', 'error')[-1]

    def test_a_rung_reaches_a_single_render(self):
        msg = self._generate(text='One sentence, on rung one.',
                             sampling={'temperature': 0.7})
        self.assertEqual(msg['type'], 'audio', msg)
        self.assertEqual([r['sampling'] for r in self.rendered()],
                         [{'temperature': 0.7}])

    def test_no_rung_is_the_loaded_sampling(self):
        msg = self._generate(text='One sentence, take zero.')
        self.assertEqual(msg['type'], 'audio', msg)
        self.assertEqual([r['sampling'] for r in self.rendered()], [None])

    def test_a_malformed_rung_is_an_error_by_name_and_renders_nothing(self):
        msg = self._generate(text='One sentence.', sampling={'topK': 0})
        self.assertEqual(msg['type'], 'error', msg)
        self.assertIn(S.MALFORMED, msg['message'])
        self.assertEqual(self.rendered(), [])


class OrpheusRefusesTheChannelTest(_WorkerCase):
    """ORPHEUS IS DEPRECATED AND HAS NO CHANNEL, so it refuses BY NAME rather
    than rendering at its voice caps and calling that the rung.

    Owen, 2026-09-14: *"orpheus is deprecated too but hasnt been removed yet.
    higgs is the frontier"*. It is not a Crucible engine and dies with the
    legacy layer; a rung silently ignored here would be a retake rendered at
    the very settings the retake exists to avoid.
    """

    WORKER_ENV = None       # the Orpheus fake, which is the default

    def test_a_batch_row_with_a_rung_is_refused_and_its_neighbours_render(self):
        self._ready()
        self._load('leah')
        self.w.send(action='generate_batch', items=[
            {'i': 0, 'text': 'An ordinary sentence.'},
            {'i': 1, 'text': 'A sentence asking for a rung.',
             'sampling': {'temperature': 0.7}},
        ])
        msgs = self.w.read_until('batch_done')
        by_i = self._assert_batch_closed(msgs, [0, 1])
        self.assertIn('data', by_i[0], by_i[0])
        self.assertNotIn('data', by_i[1])
        self.assertTrue(by_i[1]['message'].startswith(S.NOT_SUPPORTED), by_i[1])
        self.assertIn('Orpheus', by_i[1]['message'])

    def test_a_row_with_NO_rung_is_untouched(self):
        """The channel is additive: Orpheus's existing wire is unchanged."""
        self._ready()
        self._load('leah')
        self.w.send(action='generate_batch',
                    items=[{'i': 0, 'text': 'An ordinary sentence.'}])
        by_i = self._assert_batch_closed(self.w.read_until('batch_done'), [0])
        self.assertIn('data', by_i[0], by_i[0])

    def test_the_single_generate_door_refuses_it_too(self):
        self._ready()
        self._load('leah')
        self.w.send(action='generate', text='A sentence.',
                    sampling={'temperature': 0.7})
        msg = self.w.read_until('audio', 'error')[-1]
        self.assertEqual(msg['type'], 'error', msg)
        self.assertIn(S.NOT_SUPPORTED, msg['message'])


# ---------------------------------------------------------------------------
# THE RUNG'S OTHER HALF: the take, and the seed lane it moves the render into
# ---------------------------------------------------------------------------


class TakeParserTest(unittest.TestCase):
    """`parse_item_take`, on its own. A SEED IS A SETTING (Owen, 2026-09-14)."""

    def test_absent_is_take_zero_and_is_not_a_fallback(self):
        self.assertEqual(S.parse_item_take(None, 'where'), 0)

    def test_zero_and_the_rungs_above_it_pass_through(self):
        for take in (0, 1, 2, 7, S.MAX_TAKE):
            self.assertEqual(S.parse_item_take(take, 'where'), take)

    def test_a_negative_take_is_refused_by_name(self):
        with self.assertRaises(S.TakeMalformed) as caught:
            S.parse_item_take(-1, 'row 4')
        self.assertTrue(str(caught.exception).startswith(S.TAKE_MALFORMED))
        self.assertIn('row 4', str(caught.exception))

    def test_a_fractional_take_is_refused_and_never_rounded(self):
        """Rounding would render the rung next door and report the one asked
        for - the whole failure this channel exists to prevent."""
        for take in (1.5, 2.0):
            with self.assertRaises(S.TakeMalformed):
                S.parse_item_take(take, 'where')

    def test_a_bool_is_not_a_take(self):
        """`isinstance(True, int)` is True in Python, so `take: true` from a
        client that meant `stream: true` would otherwise render take 1."""
        for take in (True, False):
            with self.assertRaises(S.TakeMalformed):
                S.parse_item_take(take, 'where')

    def test_a_string_is_not_a_take(self):
        with self.assertRaises(S.TakeMalformed):
            S.parse_item_take('1', 'where')

    def test_a_take_past_MAX_TAKE_is_refused_rather_than_seeded(self):
        with self.assertRaises(S.TakeMalformed) as caught:
            S.parse_item_take(S.MAX_TAKE + 1, 'where')
        self.assertIn(str(S.MAX_TAKE), str(caught.exception))

    def test_takes_aligned_is_zeros_for_None_and_refuses_a_misaligned_list(self):
        self.assertEqual(S.takes_aligned(None, 3, 'where'), [0, 0, 0])
        self.assertEqual(S.takes_aligned([0, 2, 1], 3, 'where'), [0, 2, 1])
        with self.assertRaises(S.TakeMalformed) as caught:
            S.takes_aligned([1], 3, 'where')
        self.assertTrue(str(caught.exception).startswith(S.TAKE_MALFORMED))

    def test_an_engine_with_no_lane_refuses_only_above_take_zero(self):
        self.assertEqual(S.refuse_item_take(None, 'where', 'because'), 0)
        self.assertEqual(S.refuse_item_take(0, 'where', 'because'), 0)
        with self.assertRaises(S.TakeNotSupported) as caught:
            S.refuse_item_take(1, 'where', 'because')
        self.assertTrue(str(caught.exception).startswith(S.TAKE_NOT_SUPPORTED))
        self.assertIn('because', str(caught.exception))


class TakeSeedLaneTest(unittest.TestCase):
    """The arithmetic, and the DISJOINTNESS ARGUMENT it rests on.

    Measured 2026-09-15 and the reason any of this exists: narrator seeded
    chunk i at `config.seed + i` on both Higgs arms whatever the take, so take
    0 and take N of one chunk were byte-identical renders whenever their
    sampling matched - and two take-0 re-rolls always were, because take 0's
    rung IS the voice's own numbers.
    """

    def setUp(self):
        from crucible_higgs.engine.higgs import truncation
        self.T = truncation

    def test_take_zero_changes_nothing_which_is_what_makes_this_additive(self):
        for seed in (0, 1234, 999_999):
            self.assertEqual(self.T.in_take_lane(seed, 0), seed)

    def test_take_one_differs_by_exactly_one_stride(self):
        self.assertEqual(self.T.in_take_lane(1234, 1),
                         1234 + self.T.TAKE_SEED_STRIDE)
        self.assertEqual(self.T.in_take_lane(1234, 3),
                         1234 + 3 * self.T.TAKE_SEED_STRIDE)

    def test_an_unseeded_engine_stays_unseeded(self):
        """`reroll_seed`'s rule: an unseeded engine samples fresh on every
        call, so its take N already IS a different draw."""
        self.assertIsNone(self.T.in_take_lane(None, 4))

    def test_the_stride_is_the_reroll_stride_times_the_lanes(self):
        """The disjointness argument in one assertion: the take stride is a
        whole number of re-roll lanes, which is what makes
        `LANES * take + attempt` a distinct multiple for every (take,
        attempt) pair."""
        self.assertEqual(self.T.TAKE_SEED_STRIDE,
                         self.T.REROLL_SEED_STRIDE * self.T.TAKE_REROLL_LANES)
        self.assertGreaterEqual(self.T.TAKE_REROLL_LANES, 2)

    def test_no_take_ever_draws_a_seed_another_take_or_reroll_drew(self):
        """THE PROPERTY, over the realistic space. Every seed narrator draws is
        `base + index + REROLL_SEED_STRIDE * (LANES * take + attempt)`; this
        walks takes, attempts and indices and asserts the map is injective."""
        base = 1234
        seen = {}
        lanes = self.T.TAKE_REROLL_LANES
        indices = list(range(0, 40)) + [1_000, 50_000,
                                        self.T.REROLL_SEED_STRIDE - 1]
        for take in range(0, 12):
            for attempt in range(0, lanes):
                for index in indices:
                    seed = self.T.reroll_seed(
                        self.T.in_take_lane(base, take), index, attempt)
                    key = (take, attempt, index)
                    self.assertNotIn(
                        seed, seen,
                        f'{key} draws the same seed as {seen.get(seed)}')
                    seen[seed] = key

    def test_the_guards_own_reroll_is_inside_its_takes_lane(self):
        """The ladder's re-roll of take 3 must be a seed take 0 never drew -
        not merely a seed take 3's first draw did not."""
        base = 1234
        take0 = {self.T.reroll_seed(base, i, a)
                 for i in range(0, 2000) for a in (0, 1)}
        for index in range(0, 2000):
            for attempt in (0, 1):
                self.assertNotIn(
                    self.T.reroll_seed(self.T.in_take_lane(base, 3), index,
                                       attempt), take0)

    def test_MAX_TAKE_keeps_every_seed_inside_a_signed_32_bit_int(self):
        """`build_request_body` puts the seed on the wire as an int, and a
        server that takes a 32-bit seed would wrap a bigger one in silence."""
        worst = self.T.reroll_seed(
            self.T.in_take_lane(1234, S.MAX_TAKE),
            self.T.REROLL_SEED_STRIDE - 1, self.T.TAKE_REROLL_LANES - 1)
        self.assertLess(worst, 2 ** 31)


class TakeOnTheWireTest(_SamplingWorkerCase):
    """The take through a real worker subprocess, read back off the seed the
    fake actually drew at."""

    def seed_by_index(self):
        """`{chunk index: the seed its take 0 drew at}` - the FIRST render of
        each index, which is that chunk's take 0 on the ladder."""
        out = {}
        for row in self.rendered():
            out.setdefault(row['index'], row['seed'])
        return out

    def test_an_absent_take_is_take_zero_and_seeds_as_it_always_did(self):
        by_i = self.batch([{'i': 0, 'text': 'A chunk with no take named.'},
                           {'i': 5, 'text': 'And another one behind it.'}])
        for i in (0, 5):
            self.assertIn('data', by_i[i], by_i[i])
        self.assertEqual([r['take'] for r in self.rendered()], [0, 0])
        # FakeHiggsEngineConfig.seed is the real arms' 1234 and the rule is
        # `seed + index`. Unchanged by this channel, which is the point.
        self.assertEqual(self.seed_by_index(), {0: 1234, 5: 1239})

    def test_an_explicit_take_zero_is_the_same_render(self):
        """Crucible sends `take` on EVERY item, 0 included. It must mean
        exactly what an absent key means, or the wire would have two take
        zeros."""
        self.batch([{'i': 0, 'text': 'A chunk at an explicit take zero.',
                     'take': 0}])
        self.assertEqual(self.seed_by_index(), {0: 1234})

    def test_take_one_draws_a_different_seed_by_exactly_the_stride(self):
        from crucible_higgs.engine.higgs import truncation
        by_i = self.batch([
            {'i': 7, 'text': 'A chunk asking for the first rung up.',
             'take': 1}])
        self.assertIn('data', by_i[7], by_i[7])
        self.assertEqual(self.seed_by_index(),
                         {7: 1234 + 7 + truncation.TAKE_SEED_STRIDE})

    def test_a_take_with_NO_sampling_override_is_still_a_different_draw(self):
        """The two halves are independent on this wire, and this is why the
        take had to exist at all: an item may ask for a different DRAW and
        nothing else, and until 2026-09-15 there was no way to say it - such a
        row rendered take 0 byte for byte. (Crucible's own ladder document
        still refuses a rung that declares no numbers; that rule was written
        when this was impossible, and relaxing it is Crucible's call.)"""
        self.batch([{'i': 2, 'text': 'A chunk on a rung that changes no '
                                     'numbers at all.', 'take': 4}])
        rows = self.rendered()
        self.assertEqual([r['sampling'] for r in rows], [None])
        self.assertNotEqual(rows[0]['seed'], 1234 + 2)

    def test_one_batch_may_mix_takes_and_each_row_gets_its_own_lane(self):
        from crucible_higgs.engine.higgs import truncation
        stride = truncation.TAKE_SEED_STRIDE
        self.batch([
            {'i': 0, 'text': 'Row zero, the bottom rung.'},
            {'i': 1, 'text': 'Row one, up one rung.', 'take': 1},
            {'i': 2, 'text': 'Row two, up two rungs.', 'take': 2},
        ])
        self.assertEqual(self.seed_by_index(),
                         {0: 1234, 1: 1235 + stride, 2: 1236 + 2 * stride})

    def test_every_take_of_a_chunk_stays_in_that_chunks_lane(self):
        """A re-roll carries its parent's index and must re-roll INSIDE the
        take's lane - otherwise take 3's re-roll is a seed take 0's re-roll
        already used."""
        from crucible_higgs.engine.higgs import truncation
        self.w.close()
        env = dict(self.WORKER_ENV)
        env['NARRATOR_FAKE_HIGGS_RATE'] = json.dumps({'1': 0.4})
        self.w = Worker(extra_env=env)
        # JUDGED - a re-roll only happens on the guarded arm, and that arm
        # is asked for per batch and carries the band it is judged against.
        by_i = self.batch([
            {'i': 0, 'text': 'An ordinary opening chunk of the chapter.'},
            {'i': 1, 'text': 'The chunk whose first take comes back far too '
                             'short for its text, so the guard re-rolls it.',
             'take': 2},
        ], retake=True, band=fake_higgs_band())
        self.assertIn('data', by_i[1], by_i[1])
        seeds = [r['seed'] for r in self.rendered() if r['index'] == 1]
        self.assertGreater(len(seeds), 1,
                           'the guard never re-rolled; this test proves nothing')
        lane = 2 * truncation.TAKE_SEED_STRIDE
        take0 = {truncation.reroll_seed(1234, 1, a) for a in (0, 1)}
        for seed in seeds:
            self.assertGreaterEqual(seed, lane, 'a take fell out of its lane')
            self.assertNotIn(seed, take0)

    def test_a_malformed_take_fails_ONLY_that_row_and_names_the_refusal(self):
        by_i = self.batch([
            {'i': 0, 'text': 'A perfectly ordinary neighbouring sentence.'},
            {'i': 1, 'text': 'The row asking for rung minus one.', 'take': -1},
            {'i': 2, 'text': 'Another ordinary sentence behind it.'},
        ])
        self.assertIn('data', by_i[0])
        self.assertIn('data', by_i[2])
        self.assertNotIn('data', by_i[1])
        self.assertTrue(by_i[1]['message'].startswith(S.TAKE_MALFORMED), by_i[1])
        self.assertIn('i=1', by_i[1]['message'])
        self.assertEqual(sorted(self.seed_by_index()), [0, 2])

    def test_a_fractional_take_is_refused_on_the_wire_too(self):
        by_i = self.batch([{'i': 3, 'text': 'A row.', 'take': 1.5}])
        self.assertTrue(by_i[3]['message'].startswith(S.TAKE_MALFORMED), by_i[3])

    def test_the_single_generate_door_carries_a_take_as_well(self):
        from crucible_higgs.engine.higgs import truncation
        self._ready()
        self._load(self.VOICE)
        self.w.send(action='generate', text='One sentence, one rung up.',
                    take=1)
        msg = self.w.read_until('audio', 'error')[-1]
        self.assertEqual(msg['type'], 'audio', msg)
        # `generate` is index 0 - "the whole batch it is".
        self.assertEqual([r['seed'] for r in self.rendered()],
                         [1234 + truncation.TAKE_SEED_STRIDE])


class OrpheusRefusesTheTakeTest(_WorkerCase):
    """Orpheus has no seed lane, so a take above 0 is refused BY NAME rather
    than drawn exactly as take 0 was drawn and reported as the take."""

    WORKER_ENV = None       # the Orpheus fake, which is the default

    def test_a_batch_row_with_a_take_is_refused_and_its_neighbours_render(self):
        self._ready()
        self._load('leah')
        self.w.send(action='generate_batch', items=[
            {'i': 0, 'text': 'An ordinary sentence.'},
            {'i': 1, 'text': 'A sentence asking for a retake.', 'take': 1},
        ])
        by_i = self._assert_batch_closed(self.w.read_until('batch_done'), [0, 1])
        self.assertIn('data', by_i[0], by_i[0])
        self.assertNotIn('data', by_i[1])
        self.assertTrue(by_i[1]['message'].startswith(S.TAKE_NOT_SUPPORTED),
                        by_i[1])
        self.assertIn('Orpheus', by_i[1]['message'])

    def test_take_zero_is_untouched_on_both_spellings(self):
        """The channel is additive: Orpheus's existing wire is unchanged, and
        an explicit 0 is the same statement as no key at all."""
        self._ready()
        self._load('leah')
        self.w.send(action='generate_batch', items=[
            {'i': 0, 'text': 'An ordinary sentence.'},
            {'i': 1, 'text': 'Another, explicitly at take zero.', 'take': 0},
        ])
        by_i = self._assert_batch_closed(self.w.read_until('batch_done'), [0, 1])
        self.assertIn('data', by_i[0], by_i[0])
        self.assertIn('data', by_i[1], by_i[1])

    def test_the_single_generate_door_refuses_it_too(self):
        self._ready()
        self._load('leah')
        self.w.send(action='generate', text='A sentence.', take=2)
        msg = self.w.read_until('audio', 'error')[-1]
        self.assertEqual(msg['type'], 'error', msg)
        self.assertIn(S.TAKE_NOT_SUPPORTED, msg['message'])


if __name__ == '__main__':
    unittest.main()
