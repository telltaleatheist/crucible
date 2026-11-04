"""An unjudged batch is still a BATCH — `retake: false` without a GPU.

MEASURED ON THE TRAINING PC, 2026-09-20, and this is the defect these tests
exist for. `serve/worker.py` picked the engine's batched driver `if rows and
retake:` and a sequential comprehension otherwise, so ONE flag decided two
unrelated things: whether the batch was judged, and whether it was batched at
all. A 128-chunk render at width 4 — `HIGGS_MAX_NUM_SEQS=4` exported, sglang
demonstrably willing (zero rejections across 1,024 renders at concurrency 4 on
the same checkpoint, direct) — sat at `#running-req: 1` on all 684 sampled
scheduler lines: 12.9 s/chunk against 4.16 s/render direct.

It fell on the caller who could least afford it. A screening render of a
fine-tuned checkpoint MUST send `retake: false` — it has no measured band, and
measuring one is what the render is for — so the one client rendering a thousand
chunks for throughput was the one client guaranteed to get them one at a time.

THE FIX IS A PLAN, NOT A POOL. `truncation.UnjudgedPlan` has `GuardPlan`'s exact
driver surface and judges nothing, so `render_many` stays the one driver and the
width is spent by the ENGINE — `v3_engine` pools threads, `mlx_backend` groups
into a BatchGenerator. A thread pool in the worker would have been right for the
served arm and wrong for MLX.

`GuardPlan(tracker=None)` is NOT this and the difference is the whole point: it
falls back to a fixed band and still judges, which for a screening checkpoint
means judging it against another model's numbers.
"""
import os
import sys
import threading
import time
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_ROOT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "crucible", "jobs", "tts", "higgs")   # holds crucible_higgs
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)

from crucible_higgs.engine.higgs import truncation  # noqa: E402


def _plan(**kwargs):
    return truncation.UnjudgedPlan(sample_rate=24000, base_seed=1234, **kwargs)


class UnjudgedPlanSurfaceTest(unittest.TestCase):
    """It must be drivable by exactly the loop that drives `GuardPlan`."""

    def test_it_offers_the_same_surface_the_driver_calls(self):
        """A missing method here is a driver that dies mid-batch on the arm
        nobody tests without a card."""
        plan = _plan()
        for name in ('add', 'next_request', 'offer', 'abandon', 'finished',
                     'verdict', 'measure'):
            self.assertTrue(callable(getattr(plan, name, None)),
                            f'UnjudgedPlan has no {name}()')
        self.assertIsInstance(plan.pending, int)

    def test_every_chunk_is_requested_once_and_ships_once(self):
        plan = _plan()
        for index, text in ((7, 'seven'), (8, 'eight'), (9, 'nine')):
            plan.add(index, text)
        self.assertEqual(plan.pending, 3)

        seen = []
        while True:
            request = plan.next_request()
            if request is None:
                break
            seen.append((request.index, request.text))
            plan.offer(request, np.zeros(10, dtype=np.float32))

        self.assertEqual(seen, [(7, 'seven'), (8, 'eight'), (9, 'nine')])
        self.assertEqual([index for index, _a, _c in plan.finished()], [7, 8, 9])
        self.assertEqual(plan.pending, 0)
        # Drained: a driver that ships as it goes must not ship twice.
        self.assertEqual(plan.finished(), [])

    def test_it_never_asks_for_a_second_take_of_anything(self):
        """THE DEFINITION OF UNJUDGED. A re-roll is a judgment, and this plan
        makes none — so however bad the audio is, nothing is asked for twice."""
        plan = _plan()
        plan.add(3, 'a chunk that came back as a single sample')
        request = plan.next_request()
        # Audio a guarded plan would certainly re-roll: one sample for 40 chars.
        plan.offer(request, np.zeros(1, dtype=np.float32))
        self.assertIsNone(plan.next_request())
        self.assertEqual(len(plan.finished()), 1)

    def test_the_seed_is_the_engines_own_rule_for_the_index(self):
        """`seed=None` means take 0 of that chunk in its own lane, which is what
        the sequential arm did and what Crucible's ladder depends on: the draw
        is a pure function of (index, take) and independent of the weights."""
        plan = _plan()
        plan.add(412, 'the four hundred and twelfth chunk')
        request = plan.next_request()
        self.assertIsNone(request.seed)
        self.assertEqual(request.index, 412)
        self.assertEqual(request.rung, 'take')
        self.assertEqual(request.depth, 0)

    def test_no_chunk_carries_a_verdict(self):
        """A `guard` key here would be narrator claiming to have judged a row it
        was told not to judge. `_emit_batch_item` turns None into a row with no
        `guard` key at all."""
        plan = _plan()
        plan.add(1, 'one')
        plan.offer(plan.next_request(), np.zeros(10, dtype=np.float32))
        self.assertIsNone(plan.verdict(1))

    def test_the_measurement_survives_even_though_the_judgment_does_not(self):
        """MEASURING IS THE WHOLE POINT of a screening render — it is what the
        ladder is being run to find out. Only the judging is switched off."""
        measure = truncation.FrameMeasure.counted_frames(1200, 7500)
        plan = _plan()
        plan.add(5, 'five')
        plan.offer(plan.next_request(), np.zeros(10, dtype=np.float32),
                   measure)
        self.assertIs(plan.measure(5), measure)

    def test_a_take_the_caller_already_has_is_not_rendered_again(self):
        """`first_take` is `GuardPlan`'s contract and the slab arms depend on
        it: a row decoded by the engine itself is handed over, not re-rendered."""
        audio = np.zeros(10, dtype=np.float32)
        plan = _plan()
        plan.add(2, 'two', first_take=audio)
        self.assertIsNone(plan.next_request())
        finished = plan.finished()
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0][0], 2)
        self.assertIs(finished[0][1], audio)

    def test_an_abandoned_chunk_returns_its_index_and_never_ships(self):
        """`GuardPlan`'s contract kept: the driver reports that row's failure,
        and a caller shipping `finished()` can never ship an abandoned chunk."""
        plan = _plan()
        plan.add(11, 'eleven')
        plan.add(12, 'twelve')
        first = plan.next_request()
        self.assertEqual(plan.abandon(first), 11)
        second = plan.next_request()
        plan.offer(second, np.zeros(10, dtype=np.float32))
        self.assertEqual([index for index, _a, _c in plan.finished()], [12])
        self.assertEqual(plan.pending, 0)


class UnjudgedIsNotAFixedBandTest(unittest.TestCase):
    """The distinction the whole fix rests on."""

    def test_a_guard_plan_with_no_tracker_still_judges(self):
        """`GuardPlan(tracker=None)` falls back to `self.fixed` — so it is NOT
        "do not judge", it is "judge against a constant". For a screening
        checkpoint that constant belongs to some other model, which is exactly
        what `retake: false` exists to refuse."""
        guarded = truncation.GuardPlan(
            sample_rate=24000, base_seed=1234, tracker=None,
            max_chars_per_sec=20.0, min_chars_per_sec=12.0)
        self.assertEqual(guarded.edges(), (20.0, 12.0))
        # And the unjudged plan has no band to report at all.
        self.assertFalse(hasattr(_plan(), 'edges'))


class UnjudgedBatchRunsConcurrentlyTest(unittest.TestCase):
    """THE MEASUREMENT, without a card.

    `v3_engine.render_many` keeps `width` requests in flight through a
    ThreadPoolExecutor. These drive that loop's shape with a fake render that
    records how many were ever in flight at once — which is the number the
    training PC read off sglang's scheduler as `#running-req`.
    """

    def _drive(self, plan, width, rows):
        """The pool half of `render_many`, reduced to what these tests are about:
        take a request whenever a slot is free, render, offer it back."""
        from concurrent.futures import ThreadPoolExecutor

        for index, text in rows:
            plan.add(index, text)

        live = 0
        peak = 0
        lock = threading.Lock()

        def render(_request):
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(0.02)
            with lock:
                live -= 1
            return np.zeros(10, dtype=np.float32)

        with ThreadPoolExecutor(max_workers=width) as pool:
            running = {}
            while True:
                while len(running) < width:
                    request = plan.next_request()
                    if request is None:
                        break
                    running[pool.submit(render, request)] = request
                if not running:
                    break
                done = [f for f in list(running) if f.done()]
                if not done:
                    time.sleep(0.002)
                    continue
                for future in done:
                    plan.offer(running.pop(future), future.result())
        return peak

    def test_an_unjudged_batch_keeps_width_rows_in_flight(self):
        """THE DEFECT, inverted. This was 1 — on all 684 sampled scheduler lines
        — because the batch never reached a driver that could run more."""
        peak = self._drive(_plan(), 4, [(i, f'chunk {i}') for i in range(16)])
        self.assertGreater(peak, 1, 'the unjudged batch rendered one at a time')
        self.assertLessEqual(peak, 4, 'it exceeded the width it was given')

    def test_width_one_is_honoured_and_is_not_an_accident(self):
        """A caller that asks for one row at a time gets one — the fix widens
        nothing that was not asked for."""
        peak = self._drive(_plan(), 1, [(i, f'chunk {i}') for i in range(6)])
        self.assertEqual(peak, 1)

    def test_every_row_still_ships_exactly_once_under_concurrency(self):
        """Concurrency must not cost the three guarantees the pool depends on:
        one item per requested index, exactly once."""
        plan = _plan()
        self._drive(plan, 4, [(i, f'chunk {i}') for i in range(16)])
        shipped = [index for index, _a, _c in plan.finished()]
        self.assertEqual(sorted(shipped), list(range(16)))
        self.assertEqual(len(shipped), len(set(shipped)))


if __name__ == '__main__':
    unittest.main()
