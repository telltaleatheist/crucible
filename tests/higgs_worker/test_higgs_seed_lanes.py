"""A chunk's seed is a function of (index, take) and nothing else.

THE LADDER'S WHOLE PREMISE (training PC, 2026-09-21, relayed from Owen): cell
`(prompt, take)` must be comparable across checkpoints, so four takes are four
matched lanes on every model. With `retake: false` the plan pins `attempt` to 0
and the seed is `seed + index` shifted into the take's lane (`in_take_lane`) —
and since 2026-09-20 an unjudged batch goes through the engine's batched driver
with the job's `width`. These tests pin the invariant that batching must not
bend: the seed a chunk renders at does not depend on batch position, batch
size, completion order, or the order rows were handed in.

Driven through the Higgs fake, which applies the real arms' rule word for word
(`FakeHiggsEngine._seed_for` / `_request_seed` mirror `HiggsV3Engine`'s and
`HiggsV3MlxEngine`'s) and records the seed each render ACTUALLY drew at. The
assertion is on what was applied, never on what was asked for.
"""
import os
import random
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_ROOT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "crucible", "jobs", "tts", "higgs")   # holds crucible_higgs
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)

from crucible_higgs.engine.higgs import truncation  # noqa: E402
from crucible_higgs.serve.fake_engine import (  # noqa: E402
    fake_engine_class, fake_engine_config)

BASE_SEED = 1234
TAKES = (0, 1, 2, 3)
ROWS = [(i, f'prompt number {i}, rendered for the ladder.') for i in range(12)]


def _engine(seed=BASE_SEED):
    config = fake_engine_config('higgs-v3', voice='deathstalker')
    config.seed = seed
    return fake_engine_class('higgs-v3')(config)


def _lanes(engine, rows, take, width):
    """Render `rows` unjudged at `take` and `width`; `{index: seed}` as drawn."""
    engine.renders_seen = []
    consumed = list(engine.render_many(
        rows, take_by_index={i: take for i, _t in rows}, tracker=None,
        width=width))
    shipped = sorted(i for i, _a, _v, _m in consumed)
    if shipped != sorted(i for i, _t in rows):
        raise AssertionError(f'rows shipped {shipped}, asked {sorted(rows)}')
    lanes = {}
    for row in engine.renders_seen:
        if row['index'] in lanes and lanes[row['index']] != row['seed']:
            raise AssertionError(
                f'chunk {row["index"]} was rendered at two seeds in one '
                f'unjudged pass: {lanes[row["index"]]} and {row["seed"]}')
        lanes[row['index']] = row['seed']
        if row['take'] != take:
            raise AssertionError(
                f'chunk {row["index"]} rendered at take {row["take"]}, asked {take}')
    return lanes


class SeedLanesSurviveBatchingTest(unittest.TestCase):

    def test_the_seed_is_the_rule_and_only_the_rule(self):
        """`in_take_lane(seed + index, take)`, for every cell, from the engine
        itself — so a lane can be recomputed by anyone holding the base seed."""
        engine = _engine()
        for take in TAKES:
            lanes = _lanes(engine, ROWS, take, width=4)
            for index, _text in ROWS:
                self.assertEqual(
                    lanes[index],
                    truncation.in_take_lane(BASE_SEED + index, take),
                    f'chunk {index} take {take} drew off the rule')

    def test_row_order_and_width_change_no_seed(self):
        """The same bank handed in shuffled, at widths 1 and 4 and 12, draws the
        same seed per cell every time. Batch position is not in the formula."""
        engine = _engine()
        for take in TAKES:
            reference = _lanes(engine, ROWS, take, width=1)
            for width in (4, 12):
                for trial in range(3):
                    shuffled = list(ROWS)
                    random.Random(trial * 7 + width).shuffle(shuffled)
                    self.assertEqual(
                        _lanes(engine, shuffled, take, width=width), reference,
                        f'take {take} width {width} trial {trial}: a seed moved '
                        'with the batch')

    def test_takes_are_disjoint_lanes_across_the_bank(self):
        """No seed is shared by two cells: not across chunks of one take, and
        not across takes of one chunk. That is what makes a cell comparable."""
        engine = _engine()
        seen = {}
        for take in TAKES:
            for index, seed in _lanes(engine, ROWS, take, width=4).items():
                self.assertNotIn(
                    seed, seen,
                    f'seed {seed} drawn by both {seen.get(seed)} and {(index, take)}')
                seen[seed] = (index, take)

    def test_a_subset_of_the_bank_draws_the_same_seeds_as_the_whole(self):
        """A resume renders the missing chunks alone; their seeds must be the
        ones the full run would have drawn, or a resumed cell is not that cell."""
        engine = _engine()
        whole = _lanes(engine, ROWS, 2, width=4)
        missing = [row for row in ROWS if row[0] in (3, 7, 11)]
        self.assertEqual(_lanes(engine, missing, 2, width=4),
                         {i: whole[i] for i in (3, 7, 11)})

    def test_an_unseeded_engine_stays_unseeded_on_every_lane(self):
        """`seed = None` means "do not seed at all", and batching must not
        invent one — an invented seed would be the batch's, not the chunk's."""
        engine = _engine(seed=None)
        for take in TAKES:
            for index, seed in _lanes(engine, ROWS, take, width=4).items():
                self.assertIsNone(seed, f'chunk {index} take {take} was seeded')


if __name__ == '__main__':
    unittest.main()
