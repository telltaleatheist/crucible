"""`crucible_higgs.serve.worker`'s environment: the knobs that went, the reader that
stayed, and the warmup that replaced them.

Moved from narrator@dc05e7e7 `tests/test_serve_stream_env.py`. narrator's file was
about the ORPHEUS_STREAM_* knobs - the streaming width, the ramp and the warm max,
read through `env.env_number` and held equal to BookForge's Orpheus pool, which
owned them. Those knobs sized Orpheus's MLX grouping and its warmup, and they went
with Orpheus; their tests went with them. What stays:

  * the gap has no environment of its own on this worker - it is
    `text/gaps.classify_gap`'s answer, stated as `gapSec`;
  * `env_number` is the one numeric reader, and the MLX backend calls it;
  * no `ORPHEUS_*` variable is read by the worker at all;
  * THE WARMUP SHAPES. Every Crucible load is `warm: true`, and narrator warmed
    at the knobs' unset defaults - 1, ramp 8, full 16 - because Crucible never set
    them. The constants that replaced the knobs must keep exactly that warmup,
    and `WarmupShapesTest` pins it through a real worker and the fake's render log.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_ROOT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "crucible", "jobs", "tts", "higgs")   # holds crucible_higgs
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)

from crucible_higgs.env import env_number  # noqa: E402
from crucible_higgs.serve import worker as W  # noqa: E402


class TheKnobsAreGoneTest(unittest.TestCase):

    def test_the_orpheus_stream_knobs_are_gone(self):
        for gone in ('STREAM_BATCH_ENV', 'STREAM_RAMP_ENV', 'STREAM_WARM_MAX_ENV',
                     'STREAM_BATCH_DEFAULT', 'STREAM_RAMP_DEFAULT',
                     'stream_batch_cap', 'stream_ramp_width', 'stream_warm_max'):
            self.assertFalse(hasattr(W, gone), f'{gone} is back')

    def test_the_stream_gap_knob_is_gone(self):
        """THE GAP IS NOT AN ENVIRONMENT, and a reappearing constant is a second
        owner of a number the book already has (`text/gaps.classify_gap`)."""
        for gone in ('STREAM_GAP_SEC', 'STREAM_GAP_DEFAULT_SEC', 'STREAM_GAP_ENV'):
            self.assertFalse(hasattr(W, gone),
                             f'{gone} is back; the gap has one owner and it is '
                             'text/gaps.classify_gap')
        self.assertIn('gapSec', open(W.__file__, encoding='utf-8').read(),
                      'the worker no longer states the gap it classified')

    def test_the_worker_reads_no_orpheus_variable(self):
        """AST over `serve/worker.py`: no `os.environ.get('ORPHEUS_...')` and no
        `os.environ['ORPHEUS_...']` anywhere in it."""
        import ast
        source = open(W.__file__, encoding='utf-8').read()
        offenders = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and node.value.startswith('ORPHEUS_'):
                offenders.append((node.lineno, node.value))
        self.assertEqual(offenders, [])

    def test_zero_is_a_real_answer_for_a_gap(self):
        """A gap of 0 means "let the chunks butt together on the model's own
        pauses", and `classify_gap` keeps it - it is not a typo the way a width
        of 0 would be."""
        from crucible_higgs.text.gaps import classify_gap_seconds
        with mock.patch.dict(os.environ, {'NARRATOR_SENTENCE_GAP': '0'},
                             clear=True):
            self.assertEqual(classify_gap_seconds('A sentence.'), (0.0, 0.0))


class OnePolicyOneImplementationTest(unittest.TestCase):
    """R1, asserted as code: the one numeric env reader is `env.env_number`."""

    def test_the_mlx_backend_reads_through_the_same_function(self):
        from crucible_higgs.engine.higgs import mlx_backend
        self.assertIs(mlx_backend.env_number, env_number)
        self.assertFalse(hasattr(mlx_backend, '_env_number'),
                         'the private copy is back; there is one reader')

    def test_garbage_is_refused_by_name(self):
        with mock.patch.dict(os.environ, {'NARRATOR_TEST_WIDTH': '1 6'},
                             clear=True):
            with self.assertRaises(ValueError) as caught:
                env_number('NARRATOR_TEST_WIDTH', 16, int, 1, 'a width')
            self.assertIn('NARRATOR_TEST_WIDTH', str(caught.exception))
            self.assertIn("'1 6'", str(caught.exception))

    def test_unset_or_exported_empty_is_the_default(self):
        with mock.patch.dict(os.environ, {'NARRATOR_TEST_WIDTH': '  '},
                             clear=True):
            self.assertEqual(env_number('NARRATOR_TEST_WIDTH', 16, int, 1, 'w'), 16)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(env_number('NARRATOR_TEST_WIDTH', 16, int, 1, 'w'), 16)


class WarmupShapesTest(unittest.TestCase):
    """A `warm: true` load renders exactly what narrator rendered for every
    Crucible load: three single chunks, then a batch of 8 and a batch of 16 of
    the same sentence - 3 + 8 + 16 renders - and says so on a status line."""

    def test_a_warm_load_renders_widths_1_8_and_16(self):
        self.assertEqual((W.WARMUP_RAMP_WIDTH, W.WARMUP_FULL_WIDTH), (8, 16))
        root = tempfile.mkdtemp(prefix='narrator-warmup-')
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        log = os.path.join(root, 'renders.jsonl')
        env = dict(os.environ)
        env.update({'NARRATOR_ENGINE': 'higgs-v3', 'NARRATOR_FAKE_RENDER_LOG': log,
                    'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'})
        requests = (json.dumps({'action': 'load', 'voice': 'deathstalker',
                                'warm': True}) + '\n'
                    + json.dumps({'action': 'quit'}) + '\n')
        done = subprocess.run(
            [sys.executable, '-m', 'crucible_higgs.serve', '--fake-engine'],
            cwd=_PYTHON_ROOT, input=requests, capture_output=True, text=True,
            encoding='utf-8', env=env, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr)
        messages = [json.loads(line) for line in done.stdout.splitlines() if line]
        statuses = [m['message'] for m in messages if m['type'] == 'status']
        self.assertIn('Warming up voice (widths 1, 8, 16)...', statuses)
        self.assertIn('Warmup complete', statuses)
        self.assertEqual(messages[-1]['type'], 'loaded', messages)
        with open(log, encoding='utf-8') as handle:
            renders = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(len(renders), 3 + 8 + 16)
        self.assertEqual([r['text'] for r in renders[:3]],
                         ['Hello.', 'This is a brief warmup.',
                          'Here is a slightly longer warmup sentence to prepare '
                          'smooth playback.'])
        self.assertEqual({r['text'] for r in renders[3:]},
                         {'This is a brief warmup.'})
        # The batch warmups seed by POSITION (no caller index): 0..7, then 0..15.
        self.assertEqual([r['index'] for r in renders[3:]],
                         list(range(8)) + list(range(16)))

    def test_a_cold_load_renders_nothing(self):
        root = tempfile.mkdtemp(prefix='narrator-warmup-')
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        log = os.path.join(root, 'renders.jsonl')
        env = dict(os.environ)
        env.update({'NARRATOR_ENGINE': 'higgs-v3', 'NARRATOR_FAKE_RENDER_LOG': log})
        requests = (json.dumps({'action': 'load', 'voice': 'deathstalker',
                                'warm': False}) + '\n'
                    + json.dumps({'action': 'quit'}) + '\n')
        done = subprocess.run(
            [sys.executable, '-m', 'crucible_higgs.serve', '--fake-engine'],
            cwd=_PYTHON_ROOT, input=requests, capture_output=True, text=True,
            encoding='utf-8', env=env, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertFalse(os.path.exists(log))


if __name__ == '__main__':
    unittest.main()
