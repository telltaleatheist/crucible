"""Higgs v3 on MLX, WIDER THAN ONE ROW: the ceiling, the budget, the slicing.

WHAT THIS FILE CAN PROVE, AND WHAT IT CANNOT.

Provable here, on any machine, with no model and no GPU:

  * the three environment variables - their defaults, and that garbage is
    REFUSED BY NAME rather than coerced or defaulted past;
  * the memory arithmetic: headroom = budget - weights - the PINNED buffer
    cache, and width = headroom / (depth x MB-per-position-per-row), clamped to
    [1, ceiling];
  * the slicing: consecutive book-order slices, each carrying its own depth
    (prompt positions + that row's frame cap), and an over-deep slice split
    EVENLY rather than into [allowed, remainder];
  * that a ceiling of 1 still takes the serial path, chunk by chunk;
  * the LISTEN ladder's shape: streamed rows solo and first, the read-ahead
    batched behind them, on_chunk only where it was asked for, on_row exactly
    once per completed row, and a failed group raising rather than retrying;
  * that `on_retire` hands a row over the moment it retires, against a fake
    sampler - the row bookkeeping, not the audio;
  * that none of the names this backend reads is an ORPHEUS_ one - the Higgs
    spawn strips those deliberately, and a keeper on the TypeScript side asserts
    none rides along.

NOT provable here: that a batched render sounds like a single-row one. That is a
GPU measurement on the Mac - the left-padded prefill, the per-row retirement and
the cache `filter(keep)` are mirrored from mlx-audio's own `batch_generate`, and
nothing in this file loads a model. Owen measures it.
"""
import os
import unittest
from unittest import mock

import numpy as np

from crucible_higgs.engine.higgs import truncation
from crucible_higgs.engine.higgs.codec import FrameMeasure
from crucible_higgs.engine.higgs import v3_served
from crucible_higgs.engine.higgs.mlx_backend import (BATCH_ENV, CACHE_LIMIT_ENV,
                                               MEM_BUDGET_ENV,
                                               HiggsV3MlxEngine,
                                               mlx_batch_ceiling,
                                               mlx_cache_limit_gb,
                                               mlx_mem_budget_gb)


def _engine(*, ceiling: int, budget: float = 42.0,
            kv_mb: float = 0.140625) -> HiggsV3MlxEngine:
    """An engine object with the batch knobs set and NOTHING loaded.

    `__new__` on purpose: `__init__` loads a model, and every method under test
    reads only these four numbers. That is the point of keeping the arithmetic
    off the model.
    """
    engine = HiggsV3MlxEngine.__new__(HiggsV3MlxEngine)
    engine.BATCH_SIZE = ceiling
    engine.MLX_MEM_BUDGET_GB = budget
    engine.MLX_KV_MB_PER_TOKEN_ROW = kv_mb
    # ...plus the two the truncation ladder reads off the config on every
    # convert: the chunk's seed rule and the chars-per-second guard.
    from types import SimpleNamespace
    engine.config = SimpleNamespace(seed=None, max_chars_per_sec=20.0, min_chars_per_sec=14.5)
    # ...and the voice the guard's `_pace_tracker()` seeds itself from: one with
    # no pace fields, so `tracker_for` falls to the engine's default band above.
    engine.voice_ref = SimpleNamespace()
    # The sampling `mlx_sampling()` resolves once at construction. Set here
    # because `__init__` did not run: every render path lays the item's
    # take-ladder rung OVER this (`_sampling_for`), so a missing one is an
    # AttributeError rather than a silent default.
    engine._sampling = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 50}
    # ...and the stall guard `__init__` reads from HIGGS_STALL_GUARD: off, which
    # is what an unset variable means. Its own tests are test_higgs_stall_guard.
    engine._stall_guard = None
    return engine


class BatchEnvTest(unittest.TestCase):
    """The three variables, their defaults, and the refusals."""

    def test_defaults_are_one_row_forty_two_gb_and_eight_gb_of_cache(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(mlx_batch_ceiling(), 1)
            self.assertEqual(mlx_mem_budget_gb(), 42.0)
            self.assertEqual(mlx_cache_limit_gb(), 8.0)

    def test_the_ceiling_is_read_from_the_environment(self):
        with mock.patch.dict(os.environ, {BATCH_ENV: '16'}, clear=True):
            self.assertEqual(mlx_batch_ceiling(), 16)
        with mock.patch.dict(os.environ, {MEM_BUDGET_ENV: '34'}, clear=True):
            self.assertEqual(mlx_mem_budget_gb(), 34.0)
        with mock.patch.dict(os.environ, {CACHE_LIMIT_ENV: '6'}, clear=True):
            self.assertEqual(mlx_cache_limit_gb(), 6.0)

    def test_blank_is_the_default_not_a_refusal(self):
        # An exported-but-empty variable is what a shell hands over when a
        # setting was cleared; it means "unset", not "zero rows".
        with mock.patch.dict(os.environ, {BATCH_ENV: '   '}, clear=True):
            self.assertEqual(mlx_batch_ceiling(), 1)

    def test_garbage_is_refused_by_name(self):
        for value in ('banana', '8 rows', '1.5'):
            with mock.patch.dict(os.environ, {BATCH_ENV: value}, clear=True):
                with self.assertRaises(ValueError) as caught:
                    mlx_batch_ceiling()
                self.assertIn(BATCH_ENV, str(caught.exception))

    def test_a_ceiling_below_one_is_refused_by_name(self):
        with mock.patch.dict(os.environ, {BATCH_ENV: '0'}, clear=True):
            with self.assertRaises(ValueError) as caught:
                mlx_batch_ceiling()
            self.assertIn(BATCH_ENV, str(caught.exception))
        with mock.patch.dict(os.environ, {MEM_BUDGET_ENV: 'lots'}, clear=True):
            with self.assertRaises(ValueError) as caught:
                mlx_mem_budget_gb()
            self.assertIn(MEM_BUDGET_ENV, str(caught.exception))

    def test_no_variable_this_backend_reads_is_an_orpheus_one(self):
        # The Higgs spawn strips ORPHEUS_* deliberately (tools/test-higgs-engine.js
        # and tools/test-serve-spawn-env.js both assert none rides along), so a
        # Higgs knob spelled ORPHEUS_ would be read by nothing.
        for name in (BATCH_ENV, MEM_BUDGET_ENV, CACHE_LIMIT_ENV):
            self.assertNotIn('ORPHEUS', name)


class WidthMathTest(unittest.TestCase):
    """headroom, and the width it buys at a given depth."""

    def test_headroom_is_budget_minus_weights_minus_the_pinned_cache(self):
        engine = _engine(ceiling=64, budget=42.0)
        with mock.patch.dict(os.environ, {}, clear=True):
            # 42 - 8.5 weights - 8 pinned cache
            self.assertAlmostEqual(engine._mlx_kv_headroom_gb(), 25.5, places=6)

    def test_a_smaller_cache_limit_buys_headroom(self):
        engine = _engine(ceiling=64, budget=42.0)
        with mock.patch.dict(os.environ, {CACHE_LIMIT_ENV: '4'}, clear=True):
            self.assertAlmostEqual(engine._mlx_kv_headroom_gb(), 29.5, places=6)

    def test_an_impossible_budget_is_refused_naming_all_three_knobs(self):
        engine = _engine(ceiling=64, budget=13.0)   # the 'light' tier's budget
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as caught:
                engine._mlx_kv_headroom_gb()
        message = str(caught.exception)
        self.assertIn(MEM_BUDGET_ENV, message)
        self.assertIn(CACHE_LIMIT_ENV, message)
        self.assertIn(BATCH_ENV, message)

    def test_the_worked_example(self):
        # budget 42, weights 8.5, cache 8 -> headroom 25.5 GB.
        # depth 2000 -> 2000 x 0.140625 / 1024 = 0.2747 GB per row.
        # 25.5 / 0.2747 = 92.8 -> 92 rows.
        engine = _engine(ceiling=1000, budget=42.0)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(engine._mlx_width_for_depth(2000), 92)

    def test_the_ceiling_is_never_exceeded(self):
        engine = _engine(ceiling=8, budget=42.0)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(engine._mlx_width_for_depth(10), 8)

    def test_one_row_is_always_attemptable(self):
        # A depth so large that the arithmetic says zero rows: one row is the
        # same work the single-row path would do, so it is never refused.
        engine = _engine(ceiling=64, budget=42.0)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(engine._mlx_width_for_depth(10_000_000), 1)

    def test_a_deeper_backbone_costs_width(self):
        # Same budget, twice the KV per position -> half the rows.
        engine = _engine(ceiling=1000, budget=42.0, kv_mb=0.28125)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(engine._mlx_width_for_depth(2000), 46)


class BatchGroupsTest(unittest.TestCase):
    """Consecutive book-order slices, each with its own depth."""

    @staticmethod
    def _entries(count: int, positions: int = 700, cap: int = 300, sampling=None,
                 take=0):
        # SIX-TUPLES since the per-item TAKE landed beside the rung: the slicer
        # reads entry[4] and entry[5] so it can break a group when either half
        # of the rung changes - a slab samples every row at one temperature AND
        # draws one seed for the whole batch. `(None, 0)` is a chunk at take 0,
        # which is every entry in this file but the two tests that say
        # otherwise.
        return [(i, f'chunk {i}', positions, cap, sampling, take)
                for i in range(count)]

    def test_slices_are_consecutive_and_in_book_order(self):
        engine = _engine(ceiling=4, budget=42.0)
        with mock.patch.dict(os.environ, {}, clear=True):
            groups = engine._mlx_batch_groups(self._entries(10))
        self.assertEqual([[e[0] for e in bucket] for bucket, _d in groups],
                         [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])

    def test_depth_is_the_deepest_prompt_plus_its_own_cap(self):
        engine = _engine(ceiling=4, budget=42.0)
        entries = [(0, 'a', 700, 300, None, 0), (1, 'b', 900, 100, None, 0),
                   (2, 'c', 100, 1500, None, 0)]
        with mock.patch.dict(os.environ, {}, clear=True):
            groups = engine._mlx_batch_groups(entries)
        self.assertEqual(len(groups), 1)
        # max(700+300, 900+100, 100+1500) = 1600, not 900 and not 1500.
        self.assertEqual(groups[0][1], 1600)

    def test_an_over_deep_slice_shrinks_to_what_fits(self):
        """WAS `test_an_over_deep_slice_is_split_evenly_not_into_a_tail`,
        asserting [32, 32] - the ceiling read as a UNIT to divide, so that 64
        rows capped at 46 ran 32+32 rather than 46+18 and no batch carried a
        near-solo tail. Reversed 2026-09-09: the ceiling is a MAXIMUM and the
        book runs on, so a short window has no tail to avoid, and dividing threw
        away every row between the cap and the ceiling. Measured on Owen's
        Streicher render, 62 rows fitted, 64 were asked for, and 32 ran.

        With exactly 64 entries both rules cost two batches, which is what hid
        this; the length test below is the one that shows the difference."""
        engine = _engine(ceiling=64, budget=42.0)
        entries = self._entries(64, positions=1000, cap=3000)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(engine._mlx_width_for_depth(4000), 46)
            groups = engine._mlx_batch_groups(entries)
        sizes = [len(bucket) for bucket, _d in groups]
        self.assertEqual(sizes, [46, 18])
        self.assertEqual(sum(sizes), 64)
        # Still consecutive, still in book order.
        self.assertEqual([e[0] for e in groups[0][0]], list(range(46)))
        self.assertEqual([e[0] for e in groups[1][0]], list(range(46, 64)))

    def test_over_a_whole_book_the_shrink_is_fewer_batches(self):
        """THE ROW THAT WOULD HAVE CAUGHT IT. A batch costs about the wall time
        of its deepest row however many rows ride in it, so batch COUNT is the
        cost. 640 chunks at a cap of 46: shrinking runs ceil(640/46) = 14
        batches; the even split ran 640/32 = 20."""
        engine = _engine(ceiling=64, budget=42.0)
        entries = self._entries(640, positions=1000, cap=3000)
        with mock.patch.dict(os.environ, {}, clear=True):
            groups = engine._mlx_batch_groups(entries)
        sizes = [len(bucket) for bucket, _d in groups]
        self.assertEqual(len(sizes), 14)
        self.assertEqual(sizes[:-1], [46] * 13)
        self.assertEqual(sum(sizes), 640)

    def test_one_deep_row_narrows_only_the_window_it_sits_in(self):
        """A deep row anchors ITS window and no other. Book order is kept - no
        sorting, no bucketing - so a deep row at the head cannot be shrunk out
        of its own slice; what the shrink protects is everything BEHIND it,
        which starts a fresh window and is measured on its own depth."""
        engine = _engine(ceiling=8, budget=42.0)
        entries = ([(0, 'deep', 1000, 30000, None, 0)]
                   + [(i, f'c{i}', 100, 100, None, 0) for i in range(1, 8)])
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(engine._mlx_width_for_depth(31000), 5)
            groups = engine._mlx_batch_groups(entries)
        # 8 asked, 5 afforded at depth 31000 -> a window of 5; the shallow
        # remainder is its own window, measured at depth 200.
        self.assertEqual([len(b) for b, _d in groups], [5, 3])
        self.assertEqual([d for _b, d in groups], [31000, 200])

    def test_an_uneven_split_puts_the_extra_row_first(self):
        engine = _engine(ceiling=5, budget=42.0)
        entries = self._entries(5, positions=1000, cap=3000)
        with mock.patch.dict(os.environ, {'PLACEHOLDER': '1'}, clear=True):
            with mock.patch.object(engine, '_mlx_width_for_depth', return_value=2):
                groups = engine._mlx_batch_groups(entries)
        self.assertEqual([len(bucket) for bucket, _d in groups], [2, 2, 1])

    def test_a_sampling_change_breaks_the_group(self):
        """THE SLAB CANNOT MIX RUNGS. `_generate_delayed_rows_batch` runs one
        `_step_batch_sampler` over every active row with ONE temperature /
        top_p / top_k, so two take-ladder rungs in one bucket would render one
        of them at the other's numbers and report both as asked for. The slicer
        breaks on the rung instead - split, never silently wrong."""
        engine = _engine(ceiling=8, budget=42.0)
        hot = {'temperature': 0.7}
        entries = ([(i, f'c{i}', 100, 100, None, 0) for i in range(3)]
                   + [(i, f'c{i}', 100, 100, hot, 0) for i in range(3, 5)]
                   + [(i, f'c{i}', 100, 100, None, 0) for i in range(5, 7)])
        with mock.patch.dict(os.environ, {}, clear=True):
            groups = engine._mlx_batch_groups(entries)
        self.assertEqual([[e[0] for e in bucket] for bucket, _d in groups],
                         [[0, 1, 2], [3, 4], [5, 6]])

    def test_two_rungs_that_state_the_same_numbers_are_one_group(self):
        """The key is the NUMBERS, not the dict object: two rows Crucible
        resolved to the same rung batch together, which is the common case for
        a retake round where several chunks climbed to the same step."""
        engine = _engine(ceiling=8, budget=42.0)
        entries = [(0, 'a', 100, 100, {'temperature': 0.7, 'top_p': 0.95}, 0),
                   (1, 'b', 100, 100, {'top_p': 0.95, 'temperature': 0.7}, 0)]
        with mock.patch.dict(os.environ, {}, clear=True):
            groups = engine._mlx_batch_groups(entries)
        self.assertEqual([len(bucket) for bucket, _d in groups], [2])

    def test_no_entries_is_no_groups(self):
        engine = _engine(ceiling=8, budget=42.0)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(engine._mlx_batch_groups([]), [])


class LoadAnnouncementTest(unittest.TestCase):
    """What the engine says about batching at load, and what it does about it."""

    def test_a_ceiling_of_one_announces_nothing(self):
        engine = _engine(ceiling=1, budget=42.0)
        with mock.patch('crucible_higgs.engine.higgs.mlx_backend._log') as logged:
            engine._announce_batch_budget()
        logged.assert_not_called()

    def test_a_budget_too_small_for_the_weights_turns_batching_OFF_by_name(self):
        # BookForge hands over the ORPHEUS memory tier's budget, and Orpheus's
        # weights are 6.9 GB against Higgs's 8.5: 'light' (13 GB) can hold this
        # model but not a batch of it. Refusing the LOAD there would break
        # single-row rendering that works fine on that machine.
        engine = _engine(ceiling=24, budget=13.0)
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch('crucible_higgs.engine.higgs.mlx_backend._log') as logged:
                engine._announce_batch_budget()
        self.assertEqual(engine.BATCH_SIZE, 1, 'batching stayed on a budget that cannot hold it')
        said = ' '.join(str(call.args[0]) for call in logged.call_args_list)
        self.assertIn('OFF', said)
        self.assertIn(MEM_BUDGET_ENV, said)

    def test_a_workable_budget_announces_the_width_and_the_certificate(self):
        engine = _engine(ceiling=64, budget=42.0)
        engine.config = mock.Mock(max_chars=900)
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch('crucible_higgs.engine.higgs.mlx_backend._log') as logged:
                engine._announce_batch_budget()
        self.assertEqual(engine.BATCH_SIZE, 64)
        said = ' '.join(str(call.args[0]) for call in logged.call_args_list)
        self.assertIn('batch budget 42 GB', said)
        # The maxChars certificate was measured SINGLE-ROW and the catalog is
        # unchanged; a widened render must say so rather than inherit it.
        self.assertIn('UNCERTIFIED', said)
        self.assertIn(BATCH_ENV, said)


class ConvertBatchRoutingTest(unittest.TestCase):
    """Which path a call takes, and what it answers."""

    def test_pool_size_is_the_plain_ceiling(self):
        self.assertEqual(_engine(ceiling=16, budget=42.0).batch_pool_size, 16)
        self.assertEqual(_engine(ceiling=1, budget=42.0).batch_pool_size, 1)

    def test_a_ceiling_of_one_still_renders_chunk_by_chunk(self):
        engine = _engine(ceiling=1, budget=42.0)
        seen = []

        def _convert(index, text):
            seen.append((index, text))
            return True

        engine.convert = _convert
        answers = engine.convert_batch([(3, 'first'), (4, 'second')])
        self.assertEqual(answers, [True, True])
        self.assertEqual(seen, [(3, 'first'), (4, 'second')])

    def test_an_empty_call_is_an_empty_answer(self):
        engine = _engine(ceiling=64, budget=42.0)
        engine.convert = lambda index, text: True
        self.assertEqual(engine.convert_batch([]), [])

    def test_a_failed_slice_RAISES_naming_the_width_and_the_rows(self):
        # No per-item retry. It was here until Owen struck it on 2026-09-05: a
        # slice that failed because the WIDTH is wrong re-renders row by row,
        # succeeds, and the run reports success while the fact worth learning is
        # gone. `convert` raises when one row fails; a slice does the same.
        engine = _engine(ceiling=4, budget=42.0)
        engine._budget = mock.Mock(cap_frames=lambda text: 300)
        engine._seed_for = lambda index: 0
        engine._mlx_prompts_for = lambda texts: [(None, 700) for _t in texts]
        engine._generate_delayed_rows_batch = mock.Mock(
            side_effect=RuntimeError('Metal out of memory'))
        engine.convert = mock.Mock(
            side_effect=AssertionError('the slice was retried per item'))

        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as caught:
                engine.convert_batch([(5, 'first'), (6, 'second')])

        message = str(caught.exception)
        self.assertIn('Metal out of memory', message)
        self.assertIn('2 rows', message)
        self.assertIn('[5, 6]', message)
        engine.convert.assert_not_called()

    def test_convert_writes_through_the_one_shared_writer(self):
        # The batched path and the single-row path MUST land byte-identically,
        # which is only guaranteed while there is one `sf.write` call site.
        engine = _engine(ceiling=1, budget=42.0)
        written = []
        calls = []
        # An IN-BAND take (7 chars at ~17 chars/s), which the length ladder
        # passes through untouched and never renders a second time; the stub
        # takes the ladder's `seed`. Five seconds for seven characters would be
        # a run-on and the ladder would - correctly - re-roll it.
        import numpy as np
        clip = np.zeros(int(len('a chunk') / 17.0 * 24000), dtype=np.float32)
        engine.render_audio = (lambda text, seed=None, index=0, sampling=None, take=0:
                               calls.append((text, seed, index)) or clip)
        engine._write_sentence = lambda number, audio: written.append((number, audio)) or True
        self.assertTrue(engine.convert(7, 'a chunk'))
        self.assertEqual(calls, [('a chunk', None, 7)])
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0][0], 7)
        self.assertIs(written[0][1], clip)


# ---------------------------------------------------------------------------
# The LISTEN ladder: generate_batch_stream
# ---------------------------------------------------------------------------


class _Pcm(str):
    """A stand-in waveform: compares like the string it prints as, and answers
    `.copy()` the way the streamed rung's `on_chunk(row, 0, audio.copy())`
    requires. A real waveform is a numpy array; nothing here is about its
    samples."""

    def copy(self):
        return self


def _stream_engine(*, ceiling: int, budget: float = 42.0) -> HiggsV3MlxEngine:
    """`_engine` plus the four collaborators `generate_batch_stream` reaches for.

    Everything that would touch MLX is replaced: prompts are `(None, positions)`
    pairs, the batcher answers with the row texts it was given, and the codec
    "decodes" a row's codes into a string. What is under test is the ROW
    BOOKKEEPING - which rows go solo, which go batched, and who is told what.
    """
    engine = _engine(ceiling=ceiling, budget=budget)
    engine.voice = 'testvoice'
    engine._budget = mock.Mock(cap_frames=lambda text: 300)
    engine._seed_for = lambda index: 1000 + index
    engine.codec = lambda: mock.Mock(decode=lambda codes: _Pcm(f'decoded:{codes}'))
    engine._mlx_prompts_for = lambda texts: [(None, 700) for _t in texts]
    return engine


class StreamLadderTest(unittest.TestCase):
    """Streamed rows solo and first; the read-ahead batched behind them."""

    @staticmethod
    def _recorder():
        """(on_chunk, on_row, log) where `log` is one ordered event list."""
        events = []
        return (lambda row, seq, pcm: events.append(('chunk', row, seq, pcm)),
                lambda row, pcm: events.append(('row', row, pcm)),
                events)

    def test_stream_rows_render_solo_first_in_ascending_order(self):
        engine = _stream_engine(ceiling=8)
        solo = []
        engine.render_audio = (lambda text, index=0, should_stop=None, sampling=None, take=0:
                               solo.append(index) or _Pcm(f'solo:{index}'))
        batched = []

        def _batch(texts, caps, seed, should_stop=None, prompts=None,
                   group_no=1, group_count=1, on_retire=None, sampling=None):
            batched.append(list(texts))
            for position, text in enumerate(texts):
                on_retire(position, f'codes:{text}')
            return [f'codes:{text}' for text in texts]

        engine._generate_delayed_rows_batch = _batch
        on_chunk, on_row, events = self._recorder()
        engine.generate_batch_stream(
            ['t0', 't1', 't2', 't3'], None, {2, 1}, on_chunk, on_row)

        # SOLO FIRST, ascending - not batch order, not the order stream_rows
        # happened to iterate in.
        self.assertEqual(solo, [1, 2])
        self.assertEqual(batched, [['t0', 't3']])
        self.assertEqual(events, [
            ('chunk', 1, 0, 'solo:1'),
            ('row', 1, 'solo:1'),
            ('chunk', 2, 0, 'solo:2'),
            ('row', 2, 'solo:2'),
            ('row', 0, 'decoded:codes:t0'),
            ('row', 3, 'decoded:codes:t3'),
        ])

    def test_on_chunk_fires_only_for_stream_rows_and_on_row_exactly_once(self):
        engine = _stream_engine(ceiling=8)
        engine.render_audio = (lambda text, index=0, should_stop=None, sampling=None, take=0:
                               _Pcm(f'solo:{index}'))
        engine._generate_delayed_rows_batch = (
            lambda texts, caps, seed, should_stop=None, prompts=None,
            group_no=1, group_count=1, on_retire=None, sampling=None, take=0:
            [on_retire(p, p) for p in range(len(texts))] and None
            or [p for p in range(len(texts))])
        on_chunk, on_row, events = self._recorder()
        engine.generate_batch_stream(
            ['a', 'b', 'c', 'd', 'e'], None, {3}, on_chunk, on_row)

        chunked = [e[1] for e in events if e[0] == 'chunk']
        rowed = [e[1] for e in events if e[0] == 'row']
        self.assertEqual(chunked, [3], 'on_chunk fired for a row nobody asked to stream')
        self.assertEqual(sorted(rowed), [0, 1, 2, 3, 4])
        self.assertEqual(len(rowed), len(set(rowed)), 'a row was answered twice')

    def test_a_ceiling_of_one_renders_the_read_ahead_serially_in_order(self):
        engine = _stream_engine(ceiling=1)
        seen = []
        engine.render_audio = (lambda text, index=0, should_stop=None, sampling=None, take=0:
                               seen.append(index) or _Pcm(f'solo:{index}'))
        engine._generate_delayed_rows_batch = mock.Mock(
            side_effect=AssertionError('an unconfigured process batched'))
        on_chunk, on_row, events = self._recorder()
        engine.generate_batch_stream(['a', 'b', 'c'], None, {1}, on_chunk, on_row)

        # The streamed row first, then the rest IN ORDER, all through render_audio.
        self.assertEqual(seen, [1, 0, 2])
        self.assertEqual(events, [
            ('chunk', 1, 0, 'solo:1'),
            ('row', 1, 'solo:1'),
            ('row', 0, 'solo:0'),
            ('row', 2, 'solo:2'),
        ])

    def test_a_failed_group_RAISES_naming_the_width_and_the_rows(self):
        # No fallback to serial: the caller (serve/worker.py) turns the raise into
        # a failed batch_item for every row it has not already answered, which is
        # exactly what a single-row failure does on the serial rung.
        engine = _stream_engine(ceiling=2)
        engine.render_audio = (lambda text, index=0, should_stop=None, sampling=None, take=0:
                               _Pcm(f'solo:{index}'))
        calls = []

        def _batch(texts, caps, seed, should_stop=None, prompts=None,
                   group_no=1, group_count=1, on_retire=None, sampling=None):
            calls.append(group_no)
            if group_no == 1:
                for position, text in enumerate(texts):
                    on_retire(position, text)
                return list(texts)
            raise RuntimeError('Metal out of memory')

        engine._generate_delayed_rows_batch = _batch
        on_chunk, on_row, events = self._recorder()
        with self.assertRaises(RuntimeError) as caught:
            engine.generate_batch_stream(
                ['a', 'b', 'c', 'd'], None, set(), on_chunk, on_row)

        message = str(caught.exception)
        self.assertIn('Metal out of memory', message)
        self.assertIn('2 rows', message)          # the group's WIDTH
        self.assertIn('[2, 3]', message)          # and its ROW INDICES
        self.assertEqual(calls, [1, 2], 'the failed group was retried')
        # Rows already handed over STAND; nothing re-renders them.
        self.assertEqual([e[1] for e in events if e[0] == 'row'], [0, 1])

    def test_a_stop_between_solo_rows_ends_the_call_with_no_further_answer(self):
        engine = _stream_engine(ceiling=8)
        engine.render_audio = (lambda text, index=0, should_stop=None, sampling=None, take=0:
                               _Pcm(f'solo:{index}'))
        engine._generate_delayed_rows_batch = mock.Mock(
            side_effect=AssertionError('the batch ran after a stop'))
        on_chunk, on_row, events = self._recorder()
        stops = iter([False, True])
        engine.generate_batch_stream(['a', 'b', 'c'], None, {0, 1},
                                     on_chunk, on_row,
                                     should_stop=lambda: next(stops))
        # Row 0 rendered; row 1's check stopped the call, so neither it nor the
        # read-ahead behind it was ever touched.
        self.assertEqual(events, [('chunk', 0, 0, 'solo:0'), ('row', 0, 'solo:0')])

    def test_a_stopped_batch_keeps_the_rows_that_already_retired(self):
        engine = _stream_engine(ceiling=4)
        engine.render_audio = (lambda text, index=0, should_stop=None, sampling=None, take=0:
                               _Pcm(f'solo:{index}'))

        def _batch(texts, caps, seed, should_stop=None, prompts=None,
                   group_no=1, group_count=1, on_retire=None, sampling=None):
            on_retire(0, 'first')
            return None            # should_stop went true mid-batch

        engine._generate_delayed_rows_batch = _batch
        on_chunk, on_row, events = self._recorder()
        engine.generate_batch_stream(['a', 'b', 'c'], None, set(),
                                     on_chunk, on_row,
                                     should_stop=lambda: False)
        self.assertEqual(events, [('row', 0, 'decoded:first')])

    def test_the_seed_is_the_first_row_of_each_group(self):
        engine = _stream_engine(ceiling=2)
        engine.render_audio = (lambda text, index=0, should_stop=None,
                               sampling=None, take=0: 'x')
        seeds = []

        def _batch(texts, caps, seed, should_stop=None, prompts=None,
                   group_no=1, group_count=1, on_retire=None, sampling=None):
            seeds.append(seed)
            return list(texts)

        engine._generate_delayed_rows_batch = _batch
        on_chunk, on_row, _events = self._recorder()
        engine.generate_batch_stream(['a', 'b', 'c', 'd'], None, set(),
                                     on_chunk, on_row)
        # `_seed_for` is 1000 + index; groups are [0, 1] and [2, 3].
        self.assertEqual(seeds, [1000, 1002])

    def test_a_mixed_voice_batch_is_still_refused(self):
        engine = _stream_engine(ceiling=8)
        engine.render_audio = (lambda text, index=0, should_stop=None,
                               sampling=None, take=0: 'x')
        with self.assertRaises(ValueError) as caught:
            engine.generate_batch_stream(['a', 'b'], ['testvoice', 'other'],
                                         set(), None, lambda r, p: None)
        self.assertIn('other', str(caught.exception))


class OnRetireTest(unittest.TestCase):
    """`_generate_delayed_rows_batch(on_retire=...)`, against a fake sampler.

    The real loop is mlx-audio's and needs a GPU. What is checkable here is the
    BOOKKEEPING: which row is handed over, when, with what shape, and that the
    returned list still holds every row exactly once.
    """

    def test_each_row_is_handed_over_at_ITS_OWN_retirement(self):
        from tests.higgs_worker.higgs_mlx_fake import fake_mlx_batch
        engine = _engine(ceiling=4, budget=42.0)
        # Row 0 finishes in 2 steps, row 1 in 5, row 2 in 3.
        retired = []
        with fake_mlx_batch(engine, done_at=[2, 5, 3]):
            out = engine._generate_delayed_rows_batch(
                ['a', 'b', 'c'], [10, 10, 10], None,
                on_retire=lambda row, rows: retired.append((row, rows.shape)))

        # RETIREMENT ORDER, not row order: that is the whole point.
        self.assertEqual([row for row, _shape in retired], [0, 2, 1])
        self.assertEqual([shape for _row, shape in retired], [(2, 8), (3, 8), (5, 8)])
        self.assertEqual([m.shape for m in out], [(2, 8), (5, 8), (3, 8)])
        for matrix in out:
            self.assertEqual(matrix.dtype, np.int64)

    def test_a_row_is_evaluated_once_and_the_full_list_still_comes_back(self):
        from tests.higgs_worker.higgs_mlx_fake import fake_mlx_batch
        engine = _engine(ceiling=4, budget=42.0)
        handed = {}
        with fake_mlx_batch(engine, done_at=[1, 2]):
            out = engine._generate_delayed_rows_batch(
                ['a', 'b'], [10, 10], None,
                on_retire=lambda row, rows: handed.__setitem__(row, rows))
        self.assertEqual(sorted(handed), [0, 1])
        # The SAME arrays: the tail must not re-stack a row it already evaluated.
        for row, matrix in handed.items():
            self.assertIs(out[row], matrix)

    def test_without_on_retire_the_answer_is_unchanged(self):
        from tests.higgs_worker.higgs_mlx_fake import fake_mlx_batch
        engine = _engine(ceiling=4, budget=42.0)
        with fake_mlx_batch(engine, done_at=[2, 3]):
            out = engine._generate_delayed_rows_batch(['a', 'b'], [10, 10], None)
        self.assertEqual([m.shape for m in out], [(2, 8), (3, 8)])

    def test_a_row_abandoned_by_should_stop_is_never_handed_over(self):
        from tests.higgs_worker.higgs_mlx_fake import fake_mlx_batch
        engine = _engine(ceiling=4, budget=42.0)
        retired = []
        steps = {'n': 0}

        def _stop():
            steps['n'] += 1
            return steps['n'] > 2

        with fake_mlx_batch(engine, done_at=[9, 9]):
            out = engine._generate_delayed_rows_batch(
                ['a', 'b'], [10, 10], None, should_stop=_stop,
                on_retire=lambda row, rows: retired.append(row))
        self.assertIsNone(out)
        self.assertEqual(retired, [])




class RetakeBatchTest(unittest.TestCase):
    """The length guard's retakes ride the batches (Owen, 2026-09-08: "batch
    them at the end. instead of serializing every single one").

    A batch of 32 rows costs about the wall time of ONE row here, so a solo
    re-roll costs a whole batch. What these prove is the SCHEDULING: one
    generation call for take 0, one more for every retake the guard wants -
    never one per off-length row.
    """

    RATE = 24000
    #: 300 characters, two sentences - over MIN_GUARD_CHARS (so both edges of
    #: the band apply) and splittable if it ever came to that.
    TEXTS = [(f'Chunk {tag}. ' + 'The night was long and the road was longer. '
              * 6).strip() for tag in 'ABCD']

    def _audio(self, chars, cps=17.0):
        return np.zeros(int(chars / cps * self.RATE), dtype=np.float32)

    def _engine_with_batches(self, bad_first_take):
        """An engine whose generation is a stub: the rows it 'renders' are the
        audio itself (`codec().decode` is identity), so the guard's arithmetic
        is the only thing under test."""
        engine = _engine(ceiling=8, budget=42.0)
        engine.config.seed = 1234
        engine._budget = mock.Mock(cap_frames=lambda text: 300)
        engine._seed_for = lambda index: 500 + index
        engine._mlx_prompts_for = lambda texts: [(None, 700) for _t in texts]
        engine.codec = lambda: mock.Mock(decode=lambda rows: rows)
        written = []
        engine._write_sentence = lambda number, audio: written.append((number, audio)) or True
        calls = []

        def _batch(texts, caps, seed, prompts=None, group_no=1, group_count=1,
                   should_stop=None, on_retire=None, sampling=None):
            calls.append({'texts': list(texts), 'seed': seed})
            first = len(calls) == 1
            return [self._audio(50) if (first and text in bad_first_take)
                    else self._audio(len(text)) for text in texts]

        engine._generate_delayed_rows_batch = _batch
        return engine, calls, written

    def test_two_off_length_rows_are_ONE_extra_generation_call_not_two(self):
        bad = {self.TEXTS[1], self.TEXTS[3]}
        engine, calls, written = self._engine_with_batches(bad)
        answers = engine.convert_batch(list(enumerate(self.TEXTS)))

        self.assertEqual(answers, [True, True, True, True])
        self.assertEqual(len(calls), 2,
                         'take 0 for the four rows, then ONE round for both retakes')
        self.assertEqual(calls[1]['texts'], [self.TEXTS[1], self.TEXTS[3]])
        self.assertEqual(calls[1]['seed'], truncation.reroll_seed(1234, 1, 1),
                         'the round takes its first request\'s seed')
        self.assertEqual(sorted(number for number, _audio in written), [0, 1, 2, 3])
        # The clean rows were written before the retake round ran.
        self.assertEqual([number for number, _audio in written][:2], [0, 2])

    def test_nothing_off_length_is_no_extra_call_at_all(self):
        engine, calls, written = self._engine_with_batches(set())
        self.assertEqual(engine.convert_batch(list(enumerate(self.TEXTS))),
                         [True, True, True, True])
        self.assertEqual(len(calls), 1)
        self.assertEqual([number for number, _audio in written], [0, 1, 2, 3])

    def test_the_retake_round_is_named_by_CHUNK_when_it_fails(self):
        # A failed slice raises naming its rows; on the retake round those rows
        # are ladder positions, and a reader needs the chunk numbers.
        # TWO bad rows: a round of one renders solo through `render_audio`,
        # which is the same call the serial ladder makes.
        engine, calls, _written = self._engine_with_batches({self.TEXTS[1], self.TEXTS[3]})
        real = engine._generate_delayed_rows_batch

        def _batch(texts, caps, seed, **kwargs):
            if len(calls) >= 1:
                raise RuntimeError('Metal out of memory')
            return real(texts, caps, seed, **kwargs)

        engine._generate_delayed_rows_batch = _batch
        with self.assertRaises(RuntimeError) as caught:
            engine.convert_batch(list(enumerate(self.TEXTS)))
        message = str(caught.exception)
        self.assertIn('Metal out of memory', message)
        self.assertIn('rows [1, 3]', message)


# ---------------------------------------------------------------------------
# THE SERIAL ARM OF THE GUARDED DRIVER: render_many at width 1
# ---------------------------------------------------------------------------
# `BATCH_SIZE` defaults to 1 on this backend (see the constant: "Unset means 1 -
# one row at a time, byte for byte the behaviour that shipped"), so
# `_render_many_serial` is the arm a Mac actually runs when crucible_higgs.serve - and
# therefore Crucible - asks it to render. Nothing reached it before 2026-09-13:
# `convert_batch` at a ceiling of 1 short-circuits to `convert` (which drives
# `truncation.render_guarded`, not this), and no other caller existed - so the
# driver Owen's ruling lifted above the file-writing layer shipped untested on
# its own default width.
#
# What these prove is the CONTRACT of crucible/docs/PHASE6-REMOTE-RENDER.md
# sections 2, 3 and 5: one requested index gives exactly one artifact whatever
# the ladder did to it, the verdict travels with the audio as DATA (a client on
# the far end of a socket has no shared stderr to scrape the
# `[HIGGS3][HIGGS_GUARD_EVENT]` lines off), and not one file is written.


class _SerialRenders:
    """`render_audio`, stubbed: one silent waveform per call at a chosen pace.

    THE TRAP THIS EXISTS TO AVOID is the one `serve/fake_engine.py` documents
    above `_rate_for`: a stub whose audio length is exactly proportional to its
    character count can NEVER fire the guard, because every take then sits at one
    identical chars/sec and the band is a ratio test. So the pace is a per-chunk
    TABLE - chars/sec for take 0, take 1, ... with the last entry repeating - and
    the guard fires, or does not, because the arithmetic says so.

    TWO RULES, both `_rate_for`'s, both load-bearing:
      * a SPLIT HALF IS NEVER BENT. Its text differs from the chunk's, so it
        renders at the clean pace and the split rung terminates - a table that
        bent the children too would drive every split to MAX_DEPTH and prove
        something other than what the test says it proves.
      * the attempt counter advances only on the chunk's OWN text, so a child
        render cannot eat the re-roll's entry.

    The waveform is zeros: `truncation.interior_hole_seconds` reports 0.0 for an
    all-silent take (`len(loud) < 2` - the silence touches both ends, so there is
    no INTERIOR run), which keeps the hole guard out of the way of the length
    arithmetic these tests are about.
    """

    RATE = 24000
    #: Dead centre of the band `_engine()` builds. That helper gives the config
    #: 20.0 / 14.5 and a voice with no pace fields, so `truncation.tracker_for`
    #: seeds the tracker at the geometric mean sqrt(20 x 14.5) = 17.03 and the
    #: band's edges sit at exactly 20.0 and 14.5 until PACE_WARMUP_CHUNKS (10)
    #: guarded takes have shipped - which no test here reaches, so the band does
    #: not move underneath any of them.
    CLEAN_CPS = 17.0
    #: Comfortably over the short edge: 40 characters per second of audio is a
    #: take that stopped less than half way through its text - the Fuhrer
    #: chunk-19 shape (1,127 characters in 3.0 s) the guard was written for.
    SHORT_CPS = 40.0

    def __init__(self, cps=None, watch=None):
        #: {chunk index: [chars/sec per take]}. Absent = clean at every rung.
        self.cps = {int(k): list(v) for k, v in (cps or {}).items()}
        #: (index, text, seed) per call, in call order.
        self.calls = []
        #: (index, text, in-flight snapshot) per call, when `watch` was given.
        self.in_flight_at = []
        self._watch = watch
        self._attempts = {}
        self._full_text = {}

    def __call__(self, text, seed=None, index=0, sampling=None, take=0):
        if self._watch is not None:
            self.in_flight_at.append((index, text, list(self._watch)))
        self.calls.append((index, text, seed))
        # Take 0 always comes first and always carries the whole chunk, so the
        # first text seen for an index IS that chunk's text; anything else is a
        # split half.
        first = self._full_text.setdefault(index, text)
        table = self.cps.get(index)
        if table is None or text != first:
            rate = self.CLEAN_CPS
        else:
            attempt = self._attempts.get(index, 0)
            self._attempts[index] = attempt + 1
            rate = float(table[min(attempt, len(table) - 1)])
        self.last_audio = np.zeros(int(len(text.strip()) / rate * self.RATE),
                                   dtype=np.float32)
        return self.last_audio


def _serial_engine(renders) -> HiggsV3MlxEngine:
    """`_engine(ceiling=1)` wired for `render_many`'s serial arm and nothing else.

    That arm touches exactly four collaborators - the marker strip, the
    control-token allowlist, the pace tracker and `render_audio` - so the two
    things it must NOT touch are booby-trapped rather than left to chance: a
    batch call is an AssertionError (nothing on this backend widens on its own;
    `_render_many_serial`'s docstring states the rule and `generate_batch_stream`
    rung 2 states it again), and the writer is a mock every test below asserts
    was never called.
    """
    engine = _engine(ceiling=1, budget=42.0)
    # A REAL base seed, so the re-roll rung asks for a seed no take 0 in this
    # book used and a test can name it. `_engine()` leaves it None, which is the
    # unseeded-engine case where `reroll_seed` stays None.
    engine.config.seed = 1234
    engine.render_audio = renders
    # THE DRIVER CALLS THE MEASURED DOOR (`render_audio_measured`), so that is
    # what a stub has to stand in for: `render_audio` is its one-value face and
    # stubbing only that would leave the real one rendering. The measure is
    # taken from the stub's own audio against the real cap, which is what the
    # served arm does for real (`FrameMeasure.from_audio`).
    engine.render_audio_measured = lambda text, seed=None, index=0, **kw: (
        renders(text, seed=seed, index=index, **kw),
        FrameMeasure.from_audio(renders.last_audio, v3_served.cap_frames(text),
                                engine.SAMPLE_RATE))
    engine._write_sentence = mock.Mock(
        side_effect=AssertionError('render_many wrote a file'))
    engine._generate_delayed_rows_batch = mock.Mock(
        side_effect=AssertionError('the serial arm built a batch'))
    return engine


def _band_tracker():
    """The band these tests judge against, stated by the CALLER.

    `truncation.tracker_for` takes the band explicitly since Owen's ruling of
    2026-09-19 - nothing reads it off the voice or off the engine any more - so
    a test that drives `render_many` has to say which band it means. This is
    the one `_engine()` used to produce: the config's 20.0 / 14.5 with its
    geometric centre, sqrt(20 x 14.5) = 17.03.
    """
    return truncation.tracker_for(truncation.engine_band(20.0, 14.5))


def _chunk_text(tag: str) -> str:
    """Seven sentences, 272 characters - over MIN_GUARD_CHARS (so BOTH edges of
    the band apply, not the short side alone) and splittable into 140 + 131,
    both over MIN_SPLIT_CHARS. Shaped like RetakeBatchTest.TEXTS, for the same
    reasons."""
    return ('Chunk ' + tag + '. '
            + 'The night was long and the road was longer. ' * 6).strip()


class RenderManySerialTest(unittest.TestCase):
    """`_render_many_serial`: the ladder at the shipped default width of 1."""

    def test_clean_chunks_yield_index_audio_and_a_clean_verdict(self):
        renders = _SerialRenders()
        engine = _serial_engine(renders)
        rows = [(0, _chunk_text('A')), (1, _chunk_text('B'))]
        out = list(engine.render_many(rows, tracker=_band_tracker()))

        self.assertEqual([index for index, _audio, _v, _m in out], [0, 1])
        for _index, audio, verdict, _m in out:
            self.assertEqual(verdict['verdict'], 'clean')
            self.assertIs(verdict['clean'], True)
            self.assertEqual(verdict['parts'], 1)
            # A clean take 0 emits NO guard event at all (`_LadderTask.offer`
            # finishes before it records one), so the common case costs one
            # empty list - `GuardPlan._build_verdict` says so, and this is the
            # assertion that holds it to it.
            self.assertEqual(verdict['takes'], [])
            self.assertEqual(audio.shape, (int(272 / 17.0 * 24000),))
            # The band is still centred on the RECORDED pace: two chunks is far
            # short of PACE_WARMUP_CHUNKS, so nothing here rides a moving band.
            self.assertEqual(verdict['band']['reference'], 17.03)
            self.assertIs(verdict['band']['warm'], False)
        # ONE render per chunk, at the engine's own seed rule (seed=None), and
        # the shipped audio is the very array the render returned.
        self.assertEqual([(i, s) for i, _t, s in renders.calls],
                         [(0, None), (1, None)])
        engine._write_sentence.assert_not_called()

    def test_an_off_length_take_zero_that_the_RE_ROLL_fixes(self):
        # Take 0 at 40 chars/s is a truncation - over the 20.0 short edge - and
        # the re-roll lands in band. Rung 1 of the ladder, the one Orpheus
        # measured as the backstop that actually works (memory:
        # orpheus-short-chunk-repeat).
        renders = _SerialRenders(cps={0: [_SerialRenders.SHORT_CPS,
                                          _SerialRenders.CLEAN_CPS]})
        engine = _serial_engine(renders)
        out = list(engine.render_many([(0, _chunk_text('A'))], tracker=_band_tracker()))

        self.assertEqual(len(out), 1)
        index, audio, verdict, _measure = out[0]
        self.assertEqual(index, 0)
        self.assertEqual(verdict['verdict'], 'rerolled')
        self.assertIs(verdict['clean'], True)
        self.assertEqual(verdict['parts'], 1, 'a re-roll is not a split')
        # THE EVIDENCE RIDES ALONG, verbatim: the take-0 verdict that fired
        # (chars, seconds, chars_per_second, both thresholds) and the re-roll
        # that closed it. That is the whole reason the driver carries a verdict
        # rather than only printing one.
        self.assertEqual([r['action'] for r in verdict['takes']],
                         ['short', 'rerolled'])
        self.assertEqual(verdict['takes'][0]['index'], 0)
        self.assertEqual(verdict['takes'][0]['max_chars_per_sec'], 20.0)
        self.assertAlmostEqual(verdict['takes'][0]['chars_per_second'], 40.0,
                               places=1)
        # The shipped audio is the RE-ROLL's, not take 0's.
        self.assertEqual(audio.shape, (int(272 / 17.0 * 24000),))
        # Take 0 asks for the engine's own seed rule; the re-roll asks for a
        # seed no take 0 in this book uses.
        self.assertEqual([s for _i, _t, s in renders.calls],
                         [None, truncation.reroll_seed(1234, 0, 1)])
        engine._write_sentence.assert_not_called()

    def test_a_SPLIT_still_yields_exactly_ONE_tuple_for_the_chunk(self):
        """THE CONTRACT, and the most important assertion in this file: one
        requested index, one artifact, always (PHASE6 section 5).

        Take 0 and the re-roll are both off-length, so rung 2 cuts the chunk at
        the sentence boundary nearest its middle and renders both halves. The
        halves are joined by `truncation.join_parts` INSIDE the ladder
        (`_LadderTask._finish` bubbles a completed child up to its parent, and
        only a ROOT ever reaches `finish_root`), so the driver never sees them: a
        caller that asked for chunk 0 gets chunk 0 - one tuple, one waveform -
        however many times the guard had to cut it up to make one.
        """
        renders = _SerialRenders(cps={0: [_SerialRenders.SHORT_CPS]})  # every take
        engine = _serial_engine(renders)
        out = list(engine.render_many([(0, _chunk_text('A'))], tracker=_band_tracker()))

        self.assertEqual(len(out), 1, 'a split chunk shipped as two artifacts')
        index, audio, verdict, _measure = out[0]
        self.assertEqual(index, 0)
        self.assertEqual(verdict['parts'], 2)
        self.assertEqual(verdict['verdict'], 'resplit')
        self.assertEqual([r['action'] for r in verdict['takes']],
                         ['short', 'resplit'])
        self.assertEqual(verdict['takes'][-1]['parts'], [140, 131])
        # Four renders: take 0, the re-roll, then each half's own take 0 - and
        # the halves are the real halves, not the whole chunk twice.
        self.assertEqual([t for _i, t, _s in renders.calls][2:],
                         truncation.split_halves(_chunk_text('A')))
        # 140 + 131 characters at the clean pace, plus RESPLIT_JOIN_SECONDS of
        # silence between them: the join is the ladder's, and it happened.
        expected = (int(140 / 17.0 * 24000) + int(131 / 17.0 * 24000)
                    + int(round(truncation.RESPLIT_JOIN_SECONDS * 24000)))
        self.assertEqual(audio.shape, (expected,))
        engine._write_sentence.assert_not_called()

    def test_not_one_file_is_written_however_hard_the_ladder_works(self):
        # The driver was lifted ABOVE the file-writing layer precisely so the
        # serve world - a pipe, and no `sentences_dir` - could reach the ladder at
        # all; `convert_batch` is now one of its two sinks and does the writing
        # itself. A write from in here would put the files back under the guard
        # and break the other sink.
        renders = _SerialRenders(cps={1: [_SerialRenders.SHORT_CPS],
                                      2: [_SerialRenders.SHORT_CPS,
                                          _SerialRenders.CLEAN_CPS]})
        engine = _serial_engine(renders)
        rows = [(0, _chunk_text('A')), (1, _chunk_text('B')), (2, _chunk_text('C'))]
        out = list(engine.render_many(rows, tracker=_band_tracker()))

        # One clean, one split, one re-rolled - every rung of the ladder walked.
        self.assertEqual([v['parts'] for _i, _a, v, _m in out], [1, 2, 1])
        self.assertEqual([v['verdict'] for _i, _a, v, _m in out],
                         ['clean', 'resplit', 'rerolled'])
        engine._write_sentence.assert_not_called()
        # ...and the fixture has no `sentences_dir` to write to, so a real
        # `_write_sentence` could not even name a path. If that ever changes,
        # this test goes blind and says so rather than passing quietly.
        self.assertFalse(hasattr(engine.config, 'sentences_dir'))

    def test_in_flight_names_a_chunk_from_its_first_render_until_its_verdict(self):
        """`in_flight` is the CALLER's list, kept exact so a cooperative stop
        knows precisely which rows were running - the same discipline
        `HiggsV3Engine.convert_many` keeps. A chunk part-way up the ladder
        (re-rolling, or waiting on its split halves) stays named."""
        held = []
        renders = _SerialRenders(cps={1: [_SerialRenders.SHORT_CPS]}, watch=held)
        engine = _serial_engine(renders)
        rows = [(0, _chunk_text('A')), (1, _chunk_text('B'))]
        out = list(engine.render_many(rows, in_flight=held, tracker=_band_tracker()))

        self.assertEqual(len(out), 2)
        # Chunk 0 alone while it renders; chunk 1 alone once 0 is decided - and
        # chunk 1 STILL NAMED on its re-roll and on both of its split halves,
        # which is the case a stop has to get right.
        self.assertEqual([snapshot for _i, _t, snapshot in renders.in_flight_at],
                         [[0], [1], [1], [1], [1]])
        self.assertEqual(held, [], 'a decided chunk was left in flight')

    def test_chunks_are_yielded_in_SUBMISSION_order_not_index_order(self):
        # The serial arm decides one chunk completely before it starts the next,
        # so the ladder's decision order IS submission order - even when the
        # first chunk takes four renders to decide and the ones behind it take
        # one each. Indices deliberately out of ascending order: the driver ships
        # what it was given, in the order it was given, and sorts nothing.
        renders = _SerialRenders(cps={7: [_SerialRenders.SHORT_CPS]})
        engine = _serial_engine(renders)
        rows = [(7, _chunk_text('A')), (3, _chunk_text('B')), (9, _chunk_text('C'))]
        out = list(engine.render_many(rows, tracker=_band_tracker()))

        self.assertEqual([index for index, _a, _v, _m in out], [7, 3, 9])
        self.assertEqual([v['parts'] for _i, _a, v, _m in out], [2, 1, 1])
        # ...and every render of chunk 7 happened before chunk 3's only one.
        self.assertEqual([i for i, _t, _s in renders.calls], [7, 7, 7, 7, 3, 9])

    def test_the_markers_are_stripped_AT_THIS_DOOR_before_the_guard_counts(self):
        """The strip is made in `render_many`, once per chunk, BEFORE the plan is
        built - not left to `render_audio`'s own strip at the model boundary.

        It matters because the GUARD COUNTS CHARACTERS: `[break]` is 7 characters
        the model is never given, so a chunk judged before the strip is judged -
        and split - on text that does not exist. (The strip at the model boundary
        is still there and still idempotent; this is about who COUNTS.)
        """
        renders = _SerialRenders()
        engine = _serial_engine(renders)
        marked = '[heading]' + _chunk_text('A') + ' [break]'
        out = list(engine.render_many([(4, marked)], tracker=_band_tracker()))

        clean = _chunk_text('A')
        self.assertEqual([t for _i, t, _s in renders.calls], [clean])
        self.assertEqual(out[0][2]['takes'], [])
        # The guard measured 272 characters, not the 289 the packer's markup
        # would have made it.
        self.assertEqual(len(marked), 289)
        self.assertEqual(out[0][1].shape, (int(len(clean) / 17.0 * 24000),))

    def test_a_chunk_that_is_ONLY_markers_is_refused_BY_NAME(self):
        # `render_many` is a generator, so the refusal arrives when the caller
        # pulls - and it arrives before ANY render, which is the point: a row with
        # nothing to say must not take a prefill (or, above width 1, a whole
        # batch) down with it.
        renders = _SerialRenders()
        engine = _serial_engine(renders)
        with self.assertRaises(ValueError) as caught:
            list(engine.render_many([(0, _chunk_text('A')), (5, '[break][heading]')],
                               tracker=_band_tracker()))
        message = str(caught.exception)
        self.assertIn('chunk 5', message)
        self.assertIn('[break][heading]', message)
        self.assertEqual(renders.calls, [],
                         'a render was issued before the whole call was validated')

    def test_no_rows_is_no_yields_and_no_renders(self):
        renders = _SerialRenders()
        engine = _serial_engine(renders)
        self.assertEqual(list(engine.render_many([], tracker=_band_tracker())), [])
        self.assertEqual(renders.calls, [])
        engine._write_sentence.assert_not_called()


class _FailingRenders(_SerialRenders):
    """`_SerialRenders` with a chosen take raised instead of returned.

    `fail_at` is `{chunk index: [take numbers that raise]}`, counted over that
    chunk's OWN text the way the pace table is - so "take 0 fails" and "the
    re-roll fails" are both expressible, and a split half is never bent.
    """

    class Boom(RuntimeError):
        pass

    def __init__(self, fail_at, cps=None, watch=None):
        super().__init__(cps=cps, watch=watch)
        self.fail_at = {int(k): set(v) for k, v in fail_at.items()}
        self._seen = {}

    def __call__(self, text, seed=None, index=0, sampling=None, take=0):
        first = self._full_text.get(index, text)
        if text == first:
            take = self._seen.get(index, 0)
            self._seen[index] = take + 1
            if take in self.fail_at.get(index, ()):
                self.calls.append((index, text, seed))
                raise self.Boom(f'decode came back misaligned for chunk {index}')
        return super().__call__(text, seed=seed, index=index)


class RenderManySerialFailureTest(unittest.TestCase):
    """ONE CHUNK'S FAILURE COSTS ONE CHUNK - the serial arm, 2026-09-13.

    THE DEFECT. `_render_many_serial` called `render_audio` with no `try`, and
    `render_many`'s docstring justified that with "THIS ARM NEVER YIELDS
    `audio is None`... a slice that fails raises". That rationale is about BATCH
    WIDTH - a batch that failed for a reason batching CAUSED is precisely the
    fact a per-item retry hides - and at the SHIPPED DEFAULT (`BATCH_ENV` unset
    -> `BATCH_SIZE = 1`) there is no slice for it to be about. So one chunk's
    decode failure, one `HiggsMlxStreamMisaligned`, took the whole call down and
    `serve/worker.py` reported every row not yet emitted as 'Batch generation
    failed': THE MAC FAILED A BOOK WHERE THE PC FAILED A SENTENCE.

    The served arm has always done it the other way - `HiggsV3Engine.
    convert_many`: "This chunk's failure, named, and the take goes on" - and one
    contract cannot have two answers. The width-failure raise stays where its
    reason is true: `_render_many_rounds` (asserted by `BatchFailureTest`).
    """

    def test_one_chunks_failure_is_one_None_and_the_rest_still_render(self):
        renders = _FailingRenders(fail_at={1: [0]})
        engine = _serial_engine(renders)
        rows = [(0, _chunk_text('A')), (1, _chunk_text('B')), (2, _chunk_text('C'))]
        out = list(engine.render_many(rows, tracker=_band_tracker()))

        self.assertEqual([index for index, _a, _v, _m in out], [0, 1, 2])
        # The failed chunk: audio None AND verdict None - the shared contract's
        # failure signal, which serve/worker.py turns into the ordinary
        # 'No audio generated' item rather than silence dressed as a success.
        self.assertEqual(out[1][1], None)
        self.assertEqual(out[1][2], None)
        # Its neighbours are untouched, and each rendered exactly once.
        for position in (0, 2):
            self.assertIsNotNone(out[position][1])
            self.assertEqual(out[position][2]['verdict'], 'clean')
        self.assertEqual([i for i, _t, _s in renders.calls], [0, 1, 2])
        engine._write_sentence.assert_not_called()

    def test_a_failure_PART_WAY_UP_THE_LADDER_abandons_the_whole_chunk(self):
        """Take 0 was off-length and the RE-ROLL raised. There is no audio to
        ship - the take that would have been accepted is the one that failed
        (`GuardPlan.abandon`) - so the chunk leaves the ladder, its split rung is
        never reached, and it never appears in `finished()`."""
        renders = _FailingRenders(fail_at={0: [1]},
                                  cps={0: [_SerialRenders.SHORT_CPS]})
        engine = _serial_engine(renders)
        out = list(engine.render_many([(0, _chunk_text('A')), (1, _chunk_text('B'))],
                                    tracker=_band_tracker()))

        self.assertEqual([(i, a is None) for i, a, _v, _m in out],
                         [(0, True), (1, False)])
        # Take 0, then the re-roll that raised - and NOT the two split halves the
        # ladder would have asked for next.
        self.assertEqual([i for i, _t, _s in renders.calls], [0, 0, 1])

    def test_the_failed_chunk_is_struck_off_in_flight(self):
        """`in_flight` is the caller's list and a cooperative stop deletes the
        half-written file of everything named in it. A chunk that will never be
        written again must not stay named."""
        held = []
        renders = _FailingRenders(fail_at={1: [0]}, watch=held)
        engine = _serial_engine(renders)
        rows = [(0, _chunk_text('A')), (1, _chunk_text('B'))]
        out = list(engine.render_many(rows, in_flight=held, tracker=_band_tracker()))

        self.assertEqual(len(out), 2)
        self.assertEqual(held, [], 'a failed chunk was left in flight')

    def test_the_ladder_is_left_EMPTY_by_a_failed_chunk(self):
        """`abandon` deletes the chunk's whole subtree and pops its records and
        its verdict, so a book's worth of failures cannot accumulate inside one
        long call - and nothing can later hand out a verdict for a chunk that
        shipped no audio. The plan is captured on the way past, because that is
        the only place it exists."""
        plans = []
        real_plan = truncation.GuardPlan

        def _capture(*args, **kwargs):
            plan = real_plan(*args, **kwargs)
            plans.append(plan)
            return plan

        renders = _FailingRenders(fail_at={0: [1], 2: [0]},
                                  cps={0: [_SerialRenders.SHORT_CPS]})
        engine = _serial_engine(renders)
        rows = [(0, _chunk_text('A')), (1, _chunk_text('B')), (2, _chunk_text('C'))]
        with mock.patch.object(truncation, 'GuardPlan', _capture):
            out = list(engine.render_many(rows, tracker=_band_tracker()))

        self.assertEqual([(i, a is None) for i, a, _v, _m in out],
                         [(0, True), (1, False), (2, True)])
        self.assertEqual(len(plans), 1, 'one plan for the whole call')
        plan = plans[0]
        self.assertEqual(plan.pending, 0, 'a failed chunk was left on the ladder')
        self.assertEqual(plan.finished(), [])
        for index in (0, 2):
            with self.subTest(chunk=index):
                self.assertIsNone(plan.verdict(index),
                                  'a chunk that shipped no audio still has a '
                                  'verdict on file')

    def test_a_BASE_exception_still_unwinds(self):
        """`Exception`, not `BaseException`. A `KeyboardInterrupt` (and the
        `SystemExit` a cooperative stop raises, and the `GeneratorExit` a
        consumer that closed this generator sends) is not this chunk's failure,
        and the rows behind it are not owed a render."""
        renders = _SerialRenders()

        def interrupted(text, seed=None, index=0, sampling=None, take=0):
            # `sampling=` IS LOAD-BEARING HERE, not tidying: without it the
            # call raises TypeError - an ordinary Exception - which
            # `_render_many_serial` catches per chunk, so the KeyboardInterrupt
            # below is never reached and this test passes for a reason that has
            # nothing to do with what it asserts.
            if index == 1:
                raise KeyboardInterrupt()
            return renders(text, seed=seed, index=index)

        engine = _serial_engine(interrupted)
        with self.assertRaises(KeyboardInterrupt):
            list(engine.render_many([(0, _chunk_text('A')), (1, _chunk_text('B'))],
                                    tracker=_band_tracker()))

    def test_ABOVE_width_one_a_failed_SLICE_still_RAISES(self):
        """The other half of the ruling, restated here so the two live side by
        side: the per-chunk path is the SERIAL arm's, and widening does not buy
        a per-item retry. `BatchFailureTest` asserts the message names the width
        and the rows."""
        engine = _engine(ceiling=4, budget=42.0)
        engine.config.seed = 1234
        engine._budget = mock.Mock(cap_frames=lambda text: 300)
        engine._seed_for = lambda index: 500 + index
        engine._mlx_prompts_for = lambda texts: [(None, 700) for _t in texts]
        engine._generate_delayed_rows_batch = mock.Mock(
            side_effect=RuntimeError('metal out of memory'))
        with self.assertRaises(RuntimeError) as caught:
            list(engine.render_many([(0, _chunk_text('A')), (1, _chunk_text('B'))],
                                    tracker=_band_tracker()))
        self.assertIn('rows [0, 1]', str(caught.exception))


# AT THE BOTTOM, WHERE IT HAS TO BE. This block sat at line 631 of a 1039-line
# file until 2026-09-13, so `python test_higgs_mlx_batch.py` ran everything above
# it and silently skipped the 408 lines below - RetakeBatchTest and
# RenderManySerialTest, both whole classes. pytest collects by inspection and
# never noticed, which is exactly why nobody did.
if __name__ == '__main__':
    unittest.main()
