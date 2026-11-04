"""`crucible_higgs.serve.worker`'s environment knobs: the refusals, and the OWNER.

TWO DEFECTS, ONE FILE, because they are two halves of the same rule
(crucible/docs/ARCHITECTURE.md R1: one fact, one owner, and where a copy must
exist it is CHECKED rather than authored twice).

**1. GARBAGE WAS COERCED TO A DEFAULT.** Four reads in `serve/worker.py` - the
stream gap (since deleted, see below), the warm-up's `n` and `ramp`, and the
grouping `cap` - sat behind
`except (TypeError, ValueError): <the default>` and `if x < 1: x = 1`.
`ORPHEUS_STREAM_BATCH='1 6'` (a stray space) therefore rendered every Listen
batch 16 wide on a box tuned for 4, and nothing said so. The same package
already had the opposite policy written down and enforced, in
`engine/higgs/mlx_backend.py`'s `_env_number`, whose docstring is the ruling:
"Garbage is never coerced and never defaulted past... 'it fell back to the
default' is exactly the sentence nobody can debug." One policy, two answers,
nothing comparing them. That function is now `narrator/env.py:env_number` and
both sides call it.

**2. TWO OF THE DEFAULTS ARE NOT NARRATOR'S TO CHOOSE.**
`electron/orpheus-worker-pool.ts` owns the streaming width and the ramp
(`STREAM_BATCH_CEILING_DEFAULT`, `STREAM_RAMP_WIDTH`) and sets both on every
serve spawn it makes, so narrator's numbers are a SECOND COPY - live only for a
worker started without that spawn (`python -m crucible_higgs.serve`, the CLI, these
tests), and free to drift silently otherwise. They are read out of the
TypeScript here and asserted equal, the way
`test_engine_protocol.test_the_engines_agree_with_the_assemblers_own_table`
holds the assembler's copy of `pads`/`edge_fade` to the engines'.

**THE THIRD KNOB IS GONE (2026-09-18).** `ORPHEUS_STREAM_GAP` used to be here as
the one default that WAS narrator's own - 0.3 s of silence appended to every
streamed row. It is deleted with the padding: the gap between two Listen rows is
`text/gaps.classify_gap`'s answer for the row, stated on the wire as `gapSec` and
inserted by the PLAYER, so the number has one owner and it is the same one the
book uses. What is pinned about it lives in
`test_engine_serve_protocol.py` (the field, per row) rather than here, because
there is no longer an environment variable to have an opinion about.
"""
import os
import re
import sys
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_ROOT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "crucible", "jobs", "tts", "higgs")   # holds crucible_higgs
_REPO_ROOT = os.path.dirname(_PYTHON_ROOT)                  # .../bookforge
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)

from crucible_higgs.env import env_number                                   # noqa: E402
from crucible_higgs.serve import worker as W                                # noqa: E402

#: electron/orpheus-worker-pool.ts - the OWNER of the width and the ramp.
POOL_TS = os.path.join(_REPO_ROOT, 'electron', 'orpheus-worker-pool.ts')


def _ts_const(name: str) -> int:
    """`const <name> = <int>;` out of the pool module, or an AssertionError.

    Read as TEXT rather than executed: this interpreter has no TypeScript, and
    the point is only to compare two literals. A rename on that side lands here
    as a named failure instead of a silent divergence.
    """
    if not os.path.isfile(POOL_TS):
        raise AssertionError(
            f'the owner of the streaming width is missing: {POOL_TS}. '
            'narrator keeps a copy of two of its numbers and this test is the '
            'only thing keeping them equal.')
    with open(POOL_TS, encoding='utf-8') as handle:
        source = handle.read()
    found = re.search(rf'^(?:export )?const {name}\s*=\s*(\d+);',
                      source, re.MULTILINE)
    if found is None:
        raise AssertionError(
            f'{name} is no longer a plain integer constant in '
            f'{os.path.basename(POOL_TS)}. It is the owner of a number narrator '
            'also holds; if it moved, narrator\'s copy has to move with it.')
    return int(found.group(1))


class TheOwnerOfEachDefaultTest(unittest.TestCase):

    def test_the_batch_width_default_is_the_pools(self):
        self.assertEqual(W.STREAM_BATCH_DEFAULT,
                         _ts_const('STREAM_BATCH_CEILING_DEFAULT'),
                         'electron/orpheus-worker-pool.ts owns the streaming '
                         'width; narrator holds a copy for spawns that are not '
                         'its, and the two just drifted')

    def test_the_ramp_default_is_the_pools(self):
        self.assertEqual(W.STREAM_RAMP_DEFAULT, _ts_const('STREAM_RAMP_WIDTH'))

    def test_the_stream_gap_knob_is_gone(self):
        """THE GAP IS NOT AN ENVIRONMENT ANY MORE, and a reappearing constant is
        a second owner of a number the book already has.

        It was 0.3 s appended to every streamed row here while `gaps.json` asked
        the assembler for 0.6 s - so Listen paced at half the book and ignored
        the voice's inject entirely. The rule moved to the one function that has
        always held it (`text/gaps.classify_gap`) and the number now travels as
        `gapSec` for the player to realize.
        """
        for gone in ('STREAM_GAP_SEC', 'STREAM_GAP_DEFAULT_SEC', 'STREAM_GAP_ENV'):
            self.assertFalse(hasattr(W, gone),
                             f'{gone} is back; the gap has one owner and it is '
                             'text/gaps.classify_gap')
        self.assertIn('gapSec', open(W.__file__, encoding='utf-8').read(),
                      'the worker no longer states the gap it classified')


class UnsetMeansTheDefaultTest(unittest.TestCase):

    def test_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(W.stream_batch_cap(), 16)
            self.assertEqual(W.stream_ramp_width(), 8)
            self.assertEqual(W.stream_warm_max(), 16)

    def test_exported_empty_means_unset(self):
        """What a shell hands over when a setting was cleared. It means "no
        answer", not "zero rows"."""
        with mock.patch.dict(os.environ, {W.STREAM_BATCH_ENV: '   ',
                                          W.STREAM_RAMP_ENV: ''}, clear=True):
            self.assertEqual(W.stream_batch_cap(), 16)
            self.assertEqual(W.stream_ramp_width(), 8)

    def test_a_value_is_read(self):
        with mock.patch.dict(os.environ, {W.STREAM_BATCH_ENV: '4',
                                          W.STREAM_RAMP_ENV: '2'}, clear=True):
            self.assertEqual(W.stream_batch_cap(), 4)
            self.assertEqual(W.stream_ramp_width(), 2)

    def test_the_warm_max_LAYERS_onto_the_grouping_cap(self):
        """Unset means "warm whatever the grouping cap is" - a layered default,
        not a fallback: the answer still comes from a variable somebody set."""
        with mock.patch.dict(os.environ, {W.STREAM_BATCH_ENV: '4'}, clear=True):
            self.assertEqual(W.stream_warm_max(), 4)
        with mock.patch.dict(os.environ, {W.STREAM_BATCH_ENV: '4',
                                          W.STREAM_WARM_MAX_ENV: '12'},
                             clear=True):
            self.assertEqual(W.stream_warm_max(), 12)


class GarbageIsRefusedByNameTest(unittest.TestCase):
    """THE DEFECT, in the form it shipped in."""

    def test_the_stray_space_that_started_this(self):
        # `ORPHEUS_STREAM_BATCH='1 6'`. It used to become 16 - the default - on a
        # box tuned for 4, with nothing in any log to trace it by.
        with mock.patch.dict(os.environ, {W.STREAM_BATCH_ENV: '1 6'}, clear=True):
            with self.assertRaises(ValueError) as caught:
                W.stream_batch_cap()
            message = str(caught.exception)
            self.assertIn(W.STREAM_BATCH_ENV, message)
            self.assertIn("'1 6'", message)

    def test_every_streaming_knob_refuses_garbage_by_name(self):
        for name, read in ((W.STREAM_BATCH_ENV, W.stream_batch_cap),
                           (W.STREAM_RAMP_ENV, W.stream_ramp_width),
                           (W.STREAM_WARM_MAX_ENV, W.stream_warm_max)):
            for value in ('banana', '8 rows', '1.5', '0', '-1'):
                with self.subTest(variable=name, value=value):
                    with mock.patch.dict(os.environ, {name: value}, clear=True):
                        with self.assertRaises(ValueError) as caught:
                            read()
                        self.assertIn(name, str(caught.exception))

    def test_a_garbage_grouping_cap_is_not_survivable_by_asking_for_the_warm_max(self):
        """The layered default must not become an escape hatch: if the variable
        it layers onto is garbage, the refusal is still the answer."""
        with mock.patch.dict(os.environ, {W.STREAM_BATCH_ENV: 'four'},
                             clear=True):
            with self.assertRaises(ValueError) as caught:
                W.stream_warm_max()
            self.assertIn(W.STREAM_BATCH_ENV, str(caught.exception))

    def test_zero_is_a_real_answer_for_a_gap_and_not_for_a_width(self):
        """Same reader, two minima, and that difference is the whole reason
        `minimum` is a parameter: a gap of 0 means "let the chunks butt together
        on the model's own pauses", while a width of 0 is not a narrower batch,
        it is a typo.

        The gap half is read through `text/gaps.classify_gap`'s
        `NARRATOR_SENTENCE_GAP` now - this worker has no gap variable of its own
        (see the module docstring) - so what is asserted here is the FLOOR
        behaviour it still has to keep."""
        from crucible_higgs.text.gaps import classify_gap_seconds
        with mock.patch.dict(os.environ, {'NARRATOR_SENTENCE_GAP': '0'},
                             clear=True):
            self.assertEqual(classify_gap_seconds('A sentence.'), (0.0, 0.0))
        with mock.patch.dict(os.environ, {W.STREAM_BATCH_ENV: '0'}, clear=True):
            with self.assertRaises(ValueError):
                W.stream_batch_cap()


class OnePolicyOneImplementationTest(unittest.TestCase):
    """R1, asserted as code rather than left as prose: the module that WROTE the
    rule and the module that was breaking it now call the same function."""

    def test_the_mlx_backend_reads_through_the_same_function(self):
        from crucible_higgs.engine.higgs import mlx_backend
        self.assertIs(mlx_backend.env_number, env_number)
        self.assertFalse(hasattr(mlx_backend, '_env_number'),
                         'the private copy is back; there is one reader')

    def test_no_streaming_knob_is_read_with_a_bare_os_environ_get(self):
        """AST over `serve/worker.py`: an `os.environ.get('ORPHEUS_STREAM_*')`
        anywhere in it is a read that has escaped the policy again.

        `ORPHEUS_SKIP_WARMUP` is deliberately NOT in scope - it is a flag
        compared to '1', not a number, and there is nothing for a numeric
        reader to say about it.
        """
        import ast
        source = open(W.__file__, encoding='utf-8').read()
        offenders = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == 'get'):
                continue
            value = func.value
            if not (isinstance(value, ast.Attribute) and value.attr == 'environ'):
                continue
            for arg in node.args:
                if (isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                        and arg.value.startswith('ORPHEUS_STREAM')):
                    offenders.append((node.lineno, arg.value))
        self.assertEqual(offenders, [])


if __name__ == '__main__':
    unittest.main()
