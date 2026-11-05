"""Importing the Higgs worker must not import torch, vLLM, mlx or transformers.

THIS IS A STRUCTURAL CONTRACT, not a nicety. Two things depend on it:

  * the test interpreter renders nothing and must still be able to exercise the
    protocol seam, the registry, the prompt/SML boundary, the truncation guard
    and every engine's arithmetic;
  * `crucible_higgs.serve`'s startup sends its 'ready' line BEFORE the heavy
    import, which is why Crucible gets an answer from a spawned worker in
    milliseconds instead of waiting out a model load.

Moved from narrator@dc05e7e7 `tests/test_engine_lazy_imports.py`. narrator's
version listed the Orpheus modules (caps, snac, vllm_backend, cuda_env ...);
those did not move, so the list here is the Higgs package and the serve worker.

numpy is deliberately NOT on the forbidden list: it is a hard dependency of the
frame arithmetic and of every waveform this package touches.
"""
import os
import subprocess
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_ROOT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "crucible", "jobs", "tts", "higgs")   # holds crucible_higgs

FORBIDDEN = ('torch', 'torchaudio', 'vllm', 'mlx', 'mlx_lm', 'mlx_audio',
             'transformers', 'snac', 'sglang', 'sglang_omni')

PROBE = """
import sys
import {module}
bad = sorted(m for m in {forbidden!r} if m in sys.modules)
print(','.join(bad))
"""


def _scrubbed_env():
    """The probe's environment, with every VLLM_*, ORPHEUS_* and HIGGS_* key
    removed, so a stray variable from a developer's shell cannot change what a
    probe imports and the probe describes the module rather than the shell."""
    return {k: v for k, v in os.environ.items()
            if not k.startswith(('VLLM_', 'ORPHEUS_', 'HIGGS_'))}


def _modules_pulled_in(module):
    """Import `module` in a FRESH interpreter and report which forbidden
    packages landed in sys.modules. A fresh process is the only honest test."""
    out = subprocess.run(
        [sys.executable, '-c', PROBE.format(module=module, forbidden=FORBIDDEN)],
        cwd=_PYTHON_ROOT, capture_output=True, text=True, timeout=180,
        env=_scrubbed_env())
    if out.returncode != 0:
        raise AssertionError(f'importing {module} failed:\n{out.stderr}')
    return [m for m in out.stdout.strip().split(',') if m]


class LazyImportTest(unittest.TestCase):

    def test_importing_the_engine_package_pulls_in_no_backend(self):
        self.assertEqual(_modules_pulled_in('crucible_higgs.engine'), [])

    def test_importing_each_seam_pulls_in_no_backend(self):
        for module in ('crucible_higgs', 'crucible_higgs.env',
                       'crucible_higgs.text.gaps', 'crucible_higgs.text.sml',
                       'crucible_higgs.text.sentences',
                       'crucible_higgs.engine.protocol',
                       'crucible_higgs.engine.registry',
                       'crucible_higgs.engine.item_sampling',
                       'crucible_higgs.engine.log'):
            with self.subTest(module=module):
                self.assertEqual(_modules_pulled_in(module), [])

    def test_importing_each_higgs_module_pulls_in_no_backend_either(self):
        """A backend MODULE is importable everywhere; only its FUNCTIONS need the
        library - the MLX arm included, on a machine with no MLX at all."""
        for module in ('crucible_higgs.engine.higgs',
                       'crucible_higgs.engine.higgs.codec',
                       'crucible_higgs.engine.higgs.config',
                       'crucible_higgs.engine.higgs.prompt',
                       'crucible_higgs.engine.higgs.truncation',
                       'crucible_higgs.engine.higgs.stall_guard',
                       'crucible_higgs.engine.higgs.served_common',
                       'crucible_higgs.engine.higgs.v3_served',
                       'crucible_higgs.engine.higgs.sgl_served',
                       'crucible_higgs.engine.higgs.v3_engine',
                       'crucible_higgs.engine.higgs.mlx_backend'):
            with self.subTest(module=module):
                self.assertEqual(_modules_pulled_in(module), [])

    def test_the_serve_worker_imports_without_a_backend(self):
        """`python -m crucible_higgs.serve` must reach its 'ready' line before any
        heavy import; the module itself therefore pulls in none of them."""
        self.assertEqual(_modules_pulled_in('crucible_higgs.serve'), [])
        self.assertEqual(_modules_pulled_in('crucible_higgs.serve.fake_engine'), [])

    def test_resolving_the_engine_class_pulls_in_no_backend(self):
        """The registry's factory imports the engine CLASS, which is still not
        the runtime: the served arm talks HTTP and the MLX arm imports mlx-audio
        inside `load_engine`."""
        self.assertEqual(
            _modules_pulled_in('crucible_higgs.engine.registry; '
                               "crucible_higgs.engine.registry.engine_class('higgs-v3')"),
            [])


if __name__ == '__main__':
    unittest.main()
