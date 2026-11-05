"""The engine's log lines must never land on the worker's JSON stdout.

THE BUG (measured 2026-09-05, Higgs v3 through narrator's serve worker on the
Mac):

    RuntimeError: non-JSON on stdout: '[HIGGS-MLX] loading Higgs v3 from ...'

`crucible_higgs.serve`'s stdout IS the JSON-lines protocol Crucible parses, so
every engine log line goes through `crucible_higgs.engine.log.log()`, whose stream
the HOST chooses - and the one host, the serve worker, chooses stderr.

Moved from narrator@dc05e7e7 `tests/test_engine_log_stream.py`. Dropped with the
code they guarded: the Orpheus log-call count, the five
`parallel-tts-bridge.ts` stdout patterns (all printed by `engine/orpheus/`), the
Orpheus serve run, and the `narrator.compat.worker` host-choice probe (compat
stayed in BookForge). Kept: the no-bare-print rule over `engine/`, the measured
log-call count against `engine/log.py`'s prose, the helper itself, and the real
worker subprocess keeping stdout JSON-only with the engine's line on stderr.
"""
import ast
import io
import json
import os
import subprocess
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_ROOT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "crucible", "jobs", "tts", "higgs")   # holds crucible_higgs
_ENGINE = os.path.join(_PYTHON_ROOT, 'crucible_higgs', 'engine')

#: `engine/log.py` holds the ONE legitimate `print` in the package - `log()`'s
#: own, which names its stream. Everything else in `engine/**` must go through
#: it, INCLUDING calls that name `file=sys.stderr` themselves: those bypass
#: `set_log_stream` so the host cannot route them.
_LOG_HELPER = os.path.join(_ENGINE, 'log.py')

#: MEASURED by the AST walk in `LogCallCountTest.test_the_count_is_what_the_prose_says`.
#: ONE constant, so the number cannot drift apart across docstrings.
#:
#:   engine/higgs/  33  mlx_backend 1 (_log), served_common 18, sgl_served 2,
#:                      truncation 3, v3_engine 4, v3_served 5
#:
#: narrator@dc05e7e7 counted 37 under higgs/; the four that did not move are the
#: v2 scaffold's `transformers_backend` (2) and BookForge's prep-budget helper in
#: `v3_engine` (2). The count moving when the engine gains a log line IS the test
#: working: update it here and in `engine/log.py`'s prose in the same commit.
LOG_CALLS_BY_PACKAGE = {'higgs': 33}
LOG_CALLS_TOTAL = sum(LOG_CALLS_BY_PACKAGE.values())          # 33


def _engine_modules():
    """Every .py under `engine/`, RECURSIVELY.

    The earlier version walked `engine/orpheus/` with a non-recursive `listdir`
    plus one named Higgs file, while its docstring claimed "the engine layer
    owns no bare print". A new bare `print` in `higgs/v3_served.py`,
    `higgs/engine.py`, `higgs/v3_engine.py` or `engine/protocol.py` would not
    have been caught - and that is the original bug.
    """
    for root, _dirs, files in os.walk(_ENGINE):
        if '__pycache__' in root:
            continue
        for name in sorted(files):
            if name.endswith('.py'):
                yield os.path.join(root, name)


def _print_calls(path):
    with io.open(path, encoding='utf-8') as handle:
        tree = ast.parse(handle.read())
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == 'print']


class NoBarePrintsTest(unittest.TestCase):
    """Structural: NOTHING under `engine/` calls `print`, at all.

    An AST walk, not a grep: a grep cannot tell a call from the word `print` in
    a docstring, and this package's docstrings talk about printing a great deal.

    AND `file=` DOES NOT EXCUSE IT. The earlier version accepted any
    `print(..., file=...)`, which is precisely the shape the 17 orpheus
    stream-diagnostic sites had before this work - correct-looking, and
    unroutable, because it bypasses `set_log_stream` and so pins a line to one
    stream whatever the host needs. `engine/log.py` is the single exemption.
    """

    #: THE SECOND EXEMPTION, and it is not an engine module at all:
    #: `higgs/launch/sgl_omni_entry.py` runs as a script by path in the SGLang
    #: server's process (`$HIGGS_SGL_ENV/bin/python`), where the worker package
    #: is nothing it can rely on, so it cannot import `engine.log`. It is packaged here because the shell
    #: launcher beside it execs it by path. Its one print is the line that says
    #: which context window the server is coming up with, on stderr, which is
    #: the server log narrator already reads - and there is no `set_log_stream`
    #: in that process to route it with.
    LAUNCH_ENTRY = os.path.join(_ENGINE, 'higgs', 'launch', 'sgl_omni_entry.py')

    def test_nothing_under_engine_calls_print(self):
        offenders = []
        exempt = {os.path.abspath(_LOG_HELPER), os.path.abspath(self.LAUNCH_ENTRY)}
        for path in _engine_modules():
            if os.path.abspath(path) in exempt:
                continue
            rel = os.path.relpath(path, _ENGINE).replace(os.sep, '/')
            offenders += [f'{rel}:{c.func.lineno}' for c in _print_calls(path)]
        self.assertEqual(
            offenders, [],
            'every one of these must go through crucible_higgs.engine.log.log(). A '
            'bare print goes to stdout, which is the JSON protocol of '
            'crucible_higgs.serve, and a print naming file= cannot be routed by '
            'the host at all.')

    def test_the_log_helper_is_the_one_exemption(self):
        """It must still hold exactly one print, or the rule above is guarding
        a helper that no longer writes anything."""
        self.assertEqual(len(_print_calls(_LOG_HELPER)), 1)

    def test_the_launch_entry_is_a_STANDALONE_script_and_stays_one(self):
        """Its exemption rests on it not being package code: it runs as a
        script by path. An import of the worker package (or of narrator) here
        would be a crash at launch, not a lint - so the rule is checked, not
        trusted."""
        with io.open(self.LAUNCH_ENTRY, encoding='utf-8') as handle:
            tree = ast.parse(handle.read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split('.')[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    self.fail('a relative import cannot work in the SGLang env')
                imported.add((node.module or '').split('.')[0])
        self.assertNotIn('narrator', imported)
        self.assertNotIn('crucible_higgs', imported)
        self.assertLessEqual(len(_print_calls(self.LAUNCH_ENTRY)), 1)


class LogCallCountTest(unittest.TestCase):
    """The sweep's size is MEASURED here, not asserted in prose four times.

    narrator's `engine/log.py` once said 116 while this file said 94, for a
    sweep that was 126. The number has one home and a test that counts it.
    """

    @staticmethod
    def _count(package):
        total = 0
        for path in _engine_modules():
            parts = os.path.relpath(path, _ENGINE).replace(os.sep, '/').split('/')
            if parts[0] != package:
                continue
            with io.open(path, encoding='utf-8') as handle:
                tree = ast.parse(handle.read())
            total += sum(1 for n in ast.walk(tree)
                         if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name) and n.func.id == 'log')
        return total

    def test_the_count_is_what_the_prose_says(self):
        measured = {p: self._count(p) for p in LOG_CALLS_BY_PACKAGE}
        self.assertEqual(measured, LOG_CALLS_BY_PACKAGE)
        self.assertEqual(sum(measured.values()), LOG_CALLS_TOTAL)

    def test_the_prose_agrees_with_the_count(self):
        """`engine/log.py`'s header states the total and the per-package split.
        It must state the ones this file measures.

        WHY THIS TEST EXISTS. The class above has kept the number honest since it
        was written - and the number STILL drifted, because `log.py`'s prose was
        never checked against it and drifted. Measuring one copy of a fact does not stop a second copy
        disagreeing; it only tells you which one to believe. So the second copy
        is checked here rather than trusted."""
        import crucible_higgs.engine.log as log_module
        prose = ' '.join((log_module.__doc__ or '').split())
        higgs = LOG_CALLS_BY_PACKAGE['higgs']
        self.assertIn(
            f'The engine has {LOG_CALLS_TOTAL} of them, all under `higgs/`', prose,
            f"engine/log.py's header must say the measured number: {higgs} "
            'under higgs/')


class LogStreamTest(unittest.TestCase):
    """The helper itself."""

    def setUp(self):
        from crucible_higgs.engine.log import current_log_stream, set_log_stream
        self.addCleanup(set_log_stream, current_log_stream())

    def test_the_default_is_stderr(self):
        """A host that forgets to choose gets the SAFE answer: logs on the log
        stream, never on a structured stdout it never declared."""
        from crucible_higgs.engine.log import log_stream, set_log_stream
        set_log_stream(None)
        self.assertIs(log_stream(), sys.stderr)

    def test_the_default_resolves_at_call_time(self):
        """Not a captured reference: a test (or a host) that swaps sys.stderr
        must be honoured, or every log line in a captured run vanishes."""
        from crucible_higgs.engine.log import log_stream, set_log_stream
        set_log_stream(None)
        original, sys.stderr = sys.stderr, io.StringIO()
        try:
            self.assertIs(log_stream(), sys.stderr)
        finally:
            sys.stderr = original

    def test_a_host_can_point_it_at_stdout(self):
        from crucible_higgs.engine.log import log, set_log_stream
        sink = io.StringIO()
        set_log_stream(sink)
        log('[HIGGS3] hello')
        self.assertEqual(sink.getvalue(), '[HIGGS3] hello\n')

    def test_it_forwards_print_kwargs(self):
        """Signature-compatible with the builtin, because the call sites were
        rewritten mechanically from `print` and some of them pass `end` / `sep`."""
        from crucible_higgs.engine.log import log, set_log_stream
        sink = io.StringIO()
        set_log_stream(sink)
        log('a', 'b', sep='-', end='')
        self.assertEqual(sink.getvalue(), 'a-b')

    def test_a_non_stream_is_refused_by_name(self):
        from crucible_higgs.engine.log import set_log_stream
        with self.assertRaises(ValueError):
            set_log_stream('stdout')


class ServeWorkerStdoutTest(unittest.TestCase):
    """END TO END: drive the REAL worker and assert stdout stays JSON-only.

    `--fake-engine` gives a worker with no model, so this runs anywhere - but it
    is the real `crucible_higgs.serve.main`, the real protocol loop and the real
    `set_log_stream` call. The fake engine logs through
    `crucible_higgs.engine.log.log()` at load exactly as a real one does, so the
    assertion is about routing.
    """

    def _run(self, engine_id, extra_env=None):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith('VLLM_')}
        env.update({'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8',
                    'NARRATOR_ENGINE': engine_id})
        env.update(extra_env or {})
        proc = subprocess.Popen(
            [sys.executable, '-m', 'crucible_higgs.serve', '--fake-engine'],
            cwd=_PYTHON_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=env)
        script = [{'action': 'load', 'voice': self.VOICE[engine_id],
                   'warm': False},
                  {'action': 'generate', 'text': 'One short line.'},
                  {'action': 'quit'}]
        out, err = proc.communicate(
            '\n'.join(json.dumps(m) for m in script) + '\n', timeout=180)
        return out, err

    VOICE = {'higgs-v3': 'deathstalker'}

    def _assert_json_only(self, out):
        lines = [l for l in out.splitlines() if l.strip()]
        self.assertTrue(lines, 'the worker produced no stdout at all')
        for line in lines:
            try:
                json.loads(line)
            except json.JSONDecodeError:
                self.fail(
                    'non-JSON line on the protocol stream: '
                    f'{line!r}\n\ncrucible_higgs.serve\'s stdout IS the JSON-lines '
                    'protocol. Engine logs belong on the stream '
                    'crucible_higgs.engine.log chooses.')
        return [json.loads(l) for l in lines]

    def test_the_higgs_worker_keeps_stdout_json_only(self):
        out, _err = self._run('higgs-v3')
        messages = self._assert_json_only(out)
        self.assertEqual(messages[0]['type'], 'ready')

    def test_an_engine_log_line_lands_on_stderr_and_never_on_stdout(self):
        """THE REGRESSION ITSELF, forced rather than hoped for.

        `FakeHiggsEngine.__init__` logs through `crucible_higgs.engine.log.log()` at load,
        exactly as a real engine does - so this drives the real worker, the real
        protocol loop and the real routing. If it ever goes back to stdout, this
        fails with the precise symptom the Mac hit.

        BOTH assertions matter. Without the stderr one, a fix that simply
        stopped logging would pass while proving nothing.
        """
        from crucible_higgs.serve.fake_engine import FAKE_LOAD_MARKER
        out, err = self._run('higgs-v3')
        self.assertNotIn(FAKE_LOAD_MARKER, out,
                         'an engine log line reached the JSON protocol stream')
        self.assertIn(FAKE_LOAD_MARKER, err,
                      'the engine never logged at all - this test proves '
                      'nothing unless the line was actually emitted somewhere')
        self._assert_json_only(out)


class HostChoiceTest(unittest.TestCase):
    """The serve worker CHOOSES stderr, explicitly, rather than relying on the
    default staying put."""

    def test_serve_worker_points_the_engine_log_at_stderr(self):
        probe = (
            'import sys\n'
            'import crucible_higgs.serve.worker as w\n'
            'from crucible_higgs.engine.log import log_stream\n'
            'try:\n'
            '    w.main(["--not-a-flag"])\n'   # exits 2 AFTER set_log_stream
            'except SystemExit:\n'
            '    pass\n'
            'sys.stdout.write("STDERR" if log_stream() is sys.stderr else "OTHER")\n')
        out = subprocess.run([sys.executable, '-c', probe], cwd=_PYTHON_ROOT,
                             capture_output=True, text=True)
        self.assertIn('STDERR', out.stdout,
                      f'serve.worker must keep engine logs OFF its JSON stdout.'
                      f'\nstderr:\n{out.stderr}')


if __name__ == '__main__':
    unittest.main()
