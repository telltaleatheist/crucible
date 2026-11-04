"""`import crucible_higgs.engine` must not import torch, vLLM or mlx.

THIS IS A STRUCTURAL CONTRACT, not a nicety. Three things depend on it:

  * the Windows test interpreter renders nothing and must still be able to
    exercise the caps registry, the prompt/SML boundary, the EOS floor and
    boost arithmetic, the SNAC window emitter and every guard verdict;
  * `crucible_higgs.serve`'s startup sends its 'ready' line BEFORE the heavy import,
    which is the whole reason the pool can spawn a worker and get an answer in
    milliseconds instead of waiting out a vLLM load;
  * the platform CUDA/env block (cuda_env) must be applied BEFORE torch is
    imported, and the only way to keep that ordering honest is for nothing at
    package-import time to import torch at all.

e2a's orpheus.py imported torch, torchaudio and numpy at module scope
(lines 13-15) and got the ordering right only because `lib.conf` happened to be
imported first through `common.headers`. narrator makes the property explicit
and testable instead.

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
             'transformers', 'snac')

PROBE = """
import sys
import {module}
bad = sorted(m for m in {forbidden!r} if m in sys.modules)
print(','.join(bad))
"""


def _scrubbed_env():
    """The probe's environment, with every VLLM_* and ORPHEUS_* key removed.

    THIS TEST WAS SELF-MASKING WITHOUT IT (found in adversarial review,
    2026-09-04). The probe inherits os.environ, and the Windows cudart lookup
    that used to run at import of vllm_backend SET VLLM_CUDART_SO_PATH in THIS
    process the first time any other test module imported crucible_higgs.engine. Every
    later probe then inherited that variable, took the "already set" branch, and
    skipped the torch import the test exists to catch - so a full `discover` run
    passed while running this module ALONE failed with 18 errors.

    A scrubbed environment makes the probe describe the module rather than the
    order the suite happened to run in. ORPHEUS_* goes too: those are the caps
    and switch channel, and a stray one from a developer's shell must not be
    able to change what a probe imports either.
    """
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('VLLM_') and not k.startswith('ORPHEUS_')}
    # Belt and braces for the specific variable that caused the masking, in case
    # the prefix rule is ever loosened.
    env.pop('VLLM_CUDART_SO_PATH', None)
    return env


def _modules_pulled_in(module):
    """Import `module` in a FRESH interpreter and report which forbidden
    packages landed in sys.modules. A fresh process is the only honest test:
    this one has already imported torch through the other test modules."""
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

    def test_importing_each_torch_free_seam_pulls_in_no_backend(self):
        for module in ('crucible_higgs.engine.caps', 'crucible_higgs.engine.prompt',
                       'crucible_higgs.engine.sampling', 'crucible_higgs.engine.snac',
                       'crucible_higgs.engine.guards', 'crucible_higgs.engine.text',
                       'crucible_higgs.engine.config', 'crucible_higgs.engine.errors',
                       'crucible_higgs.engine.cuda_env', 'crucible_higgs.engine.registry',
                       'crucible_higgs.engine.audio', 'crucible_higgs.engine.adapters',
                       'crucible_higgs.engine.asr_gate'):
            with self.subTest(module=module):
                self.assertEqual(_modules_pulled_in(module), [])

    def test_importing_each_backend_module_pulls_in_no_backend_either(self):
        """A backend MODULE is importable everywhere; only its FUNCTIONS need the
        library. That is what lets cuda_env.apply() run at import, before torch,
        on a machine that may never load a model at all."""
        for module in ('crucible_higgs.engine.vllm_backend',
                       'crucible_higgs.engine.transformers_backend',
                       'crucible_higgs.engine.mlx_backend'):
            with self.subTest(module=module):
                self.assertEqual(_modules_pulled_in(module), [])

    def test_the_serve_worker_imports_without_a_backend(self):
        """`python -m crucible_higgs.serve` must reach its 'ready' line before any
        heavy import; the module itself therefore pulls in none of them."""
        self.assertEqual(_modules_pulled_in('crucible_higgs.serve'), [])

    def test_importing_a_backend_does_not_touch_the_vllm_cudart_variable(self):
        """The regression guard for the masking bug this test had (see
        _scrubbed_env): resolving cudart imports torch, so it must NOT happen at
        module import. If it ever moves back, this fails on Windows in a fresh
        process regardless of what the suite ran before it."""
        probe = (
            'import os, sys\n'
            'before = os.environ.get("VLLM_CUDART_SO_PATH")\n'
            'import crucible_higgs.engine.vllm_backend\n'
            'print(repr(before), repr(os.environ.get("VLLM_CUDART_SO_PATH")), '
            '"torch" in sys.modules)\n'
        )
        out = subprocess.run([sys.executable, '-c', probe], cwd=_PYTHON_ROOT,
                             capture_output=True, text=True, timeout=180,
                             env=_scrubbed_env())
        self.assertEqual(out.returncode, 0, out.stderr)
        before, after, torch_loaded = out.stdout.split()
        self.assertEqual(before, 'None', 'the scrubbed env must not carry it in')
        self.assertEqual(after, 'None',
                         'importing vllm_backend must not resolve cudart (that '
                         'would import torch)')
        self.assertEqual(torch_loaded, 'False')

    def test_cuda_env_is_applied_at_backend_import(self):
        """The lib/conf.py block, at the moment e2a applied it: before torch."""
        probe = (
            'import os, sys\n'
            'assert "torch" not in sys.modules\n'
            'import crucible_higgs.engine.vllm_backend\n'
            'assert "torch" not in sys.modules, "vllm_backend imported torch at module scope"\n'
            'print(os.environ.get("CUDA_DEVICE_ORDER"), '
            'os.environ.get("TORCH_CUDA_ENABLE_CUDA_GRAPH"), '
            'os.environ.get("CUDA_CACHE_MAXSIZE"))\n'
        )
        out = subprocess.run([sys.executable, '-c', probe], cwd=_PYTHON_ROOT,
                             capture_output=True, text=True, timeout=180,
                             env=_scrubbed_env())
        self.assertEqual(out.returncode, 0, out.stderr)
        order, graphs, cache = out.stdout.split()
        self.assertEqual(order, 'PCI_BUS_ID')
        self.assertEqual(cache, '2147483648')
        # On win32 lib/conf.py sets 0 and orpheus.py's own block leaves it; on
        # every other platform BOTH set it to 1.
        self.assertEqual(graphs, '0' if sys.platform == 'win32' else '1')


if __name__ == '__main__':
    unittest.main()
