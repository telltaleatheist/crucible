"""The engine's log lines must never land on a host's STRUCTURED stdout.

THE BUG (measured 2026-09-05, Higgs v3 through `python -m crucible_higgs.serve` on
the Mac):

    RuntimeError: non-JSON on stdout: '[HIGGS-MLX] loading Higgs v3 from ...'

`crucible_higgs.serve`'s stdout IS the JSON-lines protocol. The engine layer had 94
bare `print` calls, every one of which could land between two protocol messages.

AND WHY "MOVE THEM ALL TO STDERR" IS THE WRONG FIX - the half this file exists
to stop anyone undoing. `electron/parallel-tts-bridge.ts` spawns
`narrator.compat.worker` and its worker **stdout** handler runs five parsers its
**stderr** handler does not:

    MODEL_LOAD_START_RE   /Loading Orpheus model with/    the load bar starting
    MODEL_LOAD_DONE_RE    /model loaded!/                 the load bar finishing
    REPAIR_START_RE       /hit the MLX audio-token cap/   the re-split ladder bar
    parseMlxHeartbeat()   "[ORPHEUS] MLX batch generating: ..."   the batch bar
    parseOrpheusGuardEvent()                              the guard-event index

Every one of those strings is printed by `engine/orpheus/`. So the stream is the
HOST's to choose, and the two hosts choose differently. Both directions are
pinned below, because each one silently breaks a different product surface.

NO STRING IS UNDER TEST HERE except as a destination: the rewrite that created
`log()` changed 94 call names and not one character of any message, and
`test_the_bridges_stdout_only_patterns_still_match` proves the exact strings the
bridge greps for are still emitted verbatim.
"""
import ast
import io
import json
import os
import re
import subprocess
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_ROOT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "crucible", "jobs", "tts", "higgs")   # holds crucible_higgs
_ENGINE = os.path.join(os.path.dirname(_HERE), 'engine')
_ORPHEUS = os.path.join(_ENGINE, 'orpheus')

#: `engine/log.py` holds the ONE legitimate `print` in the package - `log()`'s
#: own, which names its stream. Everything else in `engine/**` must go through
#: it, INCLUDING calls that name `file=sys.stderr` themselves: that is exactly
#: the shape the 17 orpheus stream-diagnostic sites had before this work, and it
#: bypasses `set_log_stream` so the host cannot route it.
_LOG_HELPER = os.path.join(_ENGINE, 'log.py')

#: MEASURED, 2026-09-05, by the AST walk in
#: `LogCallCountTest.test_the_count_is_what_the_prose_says`. ONE constant, so
#: the number cannot drift apart across four docstrings again (it did: 116 / 116
#: / 94 / 94 were all stated for one sweep, and 126 is the measurement).
#:
#:   engine/orpheus/  111  adapters 4, asr_gate 1, audio 2, engine 30, guards 8,
#:                         mlx_backend 34, sampling 1, snac 3,
#:                         transformers_backend 4, vllm_backend 24
#:   engine/higgs/      28  mlx_backend 1 (_log), served_common 14,
#:                         sgl_served 2, transformers_backend 2, v3_engine 5,
#:                         v3_served 4
#:
#: higgs/ went 15 -> 18 on 2026-09-05 when the v3 server stopped writing its
#: output to DEVNULL: v3_served gained the log-file line and the sentinel
#: proof's report, and v3_engine gained the "proof UNAVAILABLE" line for an
#: attached server whose operator named no log.
#:
#: 25 -> 28 on 2026-09-06, THE SECOND SERVING STACK. Most of the movement is a
#: MOVE and not growth: v3_served's lifecycle (the ownership scan, the watchdog,
#: the group signal, adoption, teardown) was extracted into `served_common.py`,
#: which both stacks now share, so 14 of its lines moved there and none of them
#: was written a second time. The three genuinely NEW lines each say what the
#: second stack does differently: sgl_served's two identity lines (which
#: checkpoint is up, read from the server's own environ because /v1/models
#: reports the served NAME as its root and can never say), and v3_engine's line
#: stating that the sentinel proof does not APPLY on that stack - so "not
#: applicable" and "not proved" never look the same in a run log.
#: MEASURED, 2026-09-13: higgs was declared 28 and was 32, and `engine/log.py`
#: separately said 25. THREE numbers for one fact - which is the exact failure
#: `LogCallCountTest`'s docstring says it exists to prevent, so the guard did not
#: fail: it went red, correctly, and was stepped over until now. Nothing in the
#: 2026-09-13 guard work added a log call; the drift is older than it and was
#: verified against HEAD before this line was changed.
#:
#: The count moving when the engine gains a log line IS the test working. Update
#: the number here in the same commit that adds the call, and the prose in
#: `engine/log.py` with it - `test_the_prose_agrees_with_the_count` now checks
#: that too, because a number this file keeps true and a docstring nobody checks
#: is how it drifted the first time.
#: 32 -> 33 on 2026-09-13, NARRATOR'S OWN PACKAGED LAUNCHER. `v3_served`'s
#: constructor now says WHICH OF THE THREE launchers a backend is on - attach,
#: the operator's override, or narrator's own shipped script - because the
#: `launching: <command>` line `start()` already writes cannot answer "whose
#: script is this": a path under site-packages and a path under a campaign
#: directory look alike in a log, and the difference is which flags a 19 GB
#: server came up with.
LOG_CALLS_BY_PACKAGE = {'orpheus': 111, 'higgs': 37}
LOG_CALLS_TOTAL = sum(LOG_CALLS_BY_PACKAGE.values())          # 148


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
    #: `higgs/launch/sgl_omni_entry.py` runs under the SGLANG ENV's interpreter
    #: (`$HIGGS_SGL_ENV/bin/python`), which has never heard of `narrator` and
    #: cannot import `engine.log`. It is packaged here because the shell
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
            'crucible_higgs.serve; a print naming file= cannot be routed by the host '
            'at all, and narrator.compat.worker needs these lines on STDOUT for '
            'parallel-tts-bridge.ts to parse.')

    def test_the_log_helper_is_the_one_exemption(self):
        """It must still hold exactly one print, or the rule above is guarding
        a helper that no longer writes anything."""
        self.assertEqual(len(_print_calls(_LOG_HELPER)), 1)

    def test_the_launch_entry_is_a_STANDALONE_script_and_stays_one(self):
        """Its exemption rests on it not being narrator code: it runs under
        another env's interpreter. An import of `narrator` here would be a
        crash at launch, not a lint - so the rule is checked, not trusted."""
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
        self.assertLessEqual(len(_print_calls(self.LAUNCH_ENTRY)), 1)


class LogCallCountTest(unittest.TestCase):
    """The sweep's size is MEASURED here, not asserted in prose four times.

    L2 in review: `engine/log.py` said 116 twice and this file said 94 twice,
    for a sweep that is 126. A commit whose whole argument is "measured, not
    assumed" cannot carry four disagreeing numbers, so the number now has one
    home and a test that counts it.
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
        never checked against it and said 25 while this file said 28 and the
        truth was 32. Measuring one copy of a fact does not stop a second copy
        disagreeing; it only tells you which one to believe. So the second copy
        is checked here rather than trusted."""
        import crucible_higgs.engine.log as log_module
        prose = log_module.__doc__ or ''
        total = LOG_CALLS_TOTAL
        higgs = LOG_CALLS_BY_PACKAGE['higgs']
        orpheus = LOG_CALLS_BY_PACKAGE['orpheus']
        self.assertIn(
            f'{total} of them ({orpheus} under `orpheus/`, {higgs} under',
            ' '.join(prose.split()),
            f"engine/log.py's header must say the measured numbers: {total} total, "
            f'{orpheus} orpheus, {higgs} higgs')


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
        log('[ORPHEUS] hello')
        self.assertEqual(sink.getvalue(), '[ORPHEUS] hello\n')

    def test_it_forwards_print_kwargs(self):
        """Signature-compatible with the builtin, because the 94 call sites were
        rewritten mechanically and some of them pass `end` / `sep`."""
        from crucible_higgs.engine.log import log, set_log_stream
        sink = io.StringIO()
        set_log_stream(sink)
        log('a', 'b', sep='-', end='')
        self.assertEqual(sink.getvalue(), 'a-b')

    def test_a_non_stream_is_refused_by_name(self):
        from crucible_higgs.engine.log import set_log_stream
        with self.assertRaises(ValueError):
            set_log_stream('stdout')


class BridgePatternsTest(unittest.TestCase):
    """The five stdout-only bridge patterns still have a real message to match.

    This is the guard on the OTHER half of the fix: those five parsers exist
    ONLY on `parallel-tts-bridge.ts`'s worker-stdout handler, so if someone
    later "tidies" one of these strings the audiobook progress bars stop moving
    and nothing fails.

    THE PATTERNS BELOW ARE THE BRIDGE'S OWN, COPIED VERBATIM, each with the .ts
    file and line it came from. The earlier version paraphrased two of them into
    weaker substrings and so pinned less than the bridge requires - measured in
    review: with the real `REPAIR_START_RE` and `HEARTBEAT_RE`, the reconstructed
    literals matched ZERO. Deleting the `sentence {i} ` prefix, or renaming the
    heartbeat's `rows`/`tokens` fields, passed the test and froze the bar. A
    paraphrase is not a pin; if the bridge's regex changes, this list must be
    re-copied from it.
    """

    #: (name, VERBATIM regex, where it lives in electron/, module that must
    #: still carry a matching message literal)
    PATTERNS = (
        ('MODEL_LOAD_START_RE',
         r'Loading .*TTS with voice|Loading Orpheus model with|Loading .* model',
         'parallel-tts-bridge.ts:2441', 'mlx_backend.py'),
        ('MODEL_LOAD_DONE_RE',
         r'TTS Loaded!|model loaded!',
         'parallel-tts-bridge.ts:2442', 'mlx_backend.py'),
        ('REPAIR_START_RE',
         r'sentence (\d+) (?:hit the MLX audio-token cap|produced no audio|'
         r'audio too short for text)',
         'parallel-tts-bridge.ts:2449', 'mlx_backend.py'),
        ('GENERATION_ACTIVITY_RE',
         r'audio-token cap|re-rendering split|Processed prompts|Adding requests|'
         r'MLX batch generating',
         'parallel-tts-bridge.ts:2428', 'mlx_backend.py'),
        ('HEARTBEAT_RE (parseMlxHeartbeat)',
         r'MLX batch generating:\s*(\d+) rows,\s*~(\d+) tokens'
         r'(?:\s*\(step (\d+)(?:\/(\d+))?\))?(?:,\s*(\d+)\/(\d+) rows done)?'
         r'(?:,\s*batch (\d+)\/(\d+))?',
         'mlx-batch-progress.ts:93', 'mlx_backend.py'),
    )

    @staticmethod
    def _docstring_nodes(tree):
        """The id() of every node that IS a docstring, so they can be excluded."""
        out = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    out.add(id(node.body[0].value))
        return out

    def _literals(self, filename):
        r"""Every MESSAGE literal in the module, f-strings reconstructed.

        TWO THINGS THE EARLIER VERSION GOT WRONG.

        1. It flattened an f-string by keeping only the `ast.Constant` parts and
           DROPPING every `{...}`, so `f'sentence {i} hit the ... cap'` became
           `'sentence  hit the ... cap'` - which the bridge's
           `sentence (\d+) ` can never match. Each interpolation is now
           replaced by `'0'`, which is what the field actually carries (an
           index, a count, a token total), so the reconstruction is something
           the bridge would really see.
        2. Its docstring claimed the AST "so a docstring's prose cannot satisfy
           a pattern that a real message must" - FALSE, `ast.walk` yields
           docstrings as `ast.Constant`. They are now genuinely excluded, so the
           claim and the code agree and prose in THIS package (which discusses
           these very strings at length) cannot satisfy a pin.
        """
        path = os.path.join(_ORPHEUS, filename)
        with io.open(path, encoding='utf-8') as handle:
            tree = ast.parse(handle.read())
        docstrings = self._docstring_nodes(tree)
        out = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) not in docstrings:
                    out.append(node.value)
            elif isinstance(node, ast.JoinedStr):
                out.append(''.join(
                    v.value if isinstance(v, ast.Constant)
                    and isinstance(v.value, str) else '0'
                    for v in node.values))
        return out

    def test_the_bridges_stdout_only_patterns_still_match(self):
        for name, pattern, source, filename in self.PATTERNS:
            with self.subTest(pattern=name):
                rx = re.compile(pattern, re.I)
                hits = [s for s in self._literals(filename) if rx.search(s)]
                self.assertTrue(
                    hits,
                    f'{name} ({source}) has nothing left to match in '
                    f'{filename}. The audiobook progress bar reads these off '
                    f"the compat worker's stdout; changing the wording "
                    f'silently stops it moving.')

    def test_a_docstring_cannot_satisfy_a_pin(self):
        """The correction to the claim above, asserted rather than stated.

        `engine/orpheus/mlx_backend.py`'s module docstring quotes the heartbeat
        line verbatim (it is a load-bearing string and the docstring says so),
        so if docstrings were included this file's most important pin would be
        satisfied by a COMMENT ABOUT the message rather than the message.
        """
        path = os.path.join(_ORPHEUS, 'mlx_backend.py')
        with io.open(path, encoding='utf-8') as handle:
            tree = ast.parse(handle.read())
        docstrings = self._docstring_nodes(tree)
        self.assertTrue(docstrings, 'the module has docstrings to exclude')
        quoted = [n.value for n in ast.walk(tree)
                  if isinstance(n, ast.Constant) and isinstance(n.value, str)
                  and id(n) in docstrings
                  and 'MLX batch generating' in n.value]
        self.assertTrue(
            quoted,
            'this test is only meaningful while a docstring quotes the '
            'heartbeat - it does today, deliberately')
        self.assertNotIn(quoted[0], self._literals('mlx_backend.py'),
                         'docstrings must be excluded from the literal set')


class ServeWorkerStdoutTest(unittest.TestCase):
    """END TO END: drive the REAL worker and assert stdout stays JSON-only.

    `--fake-engine` gives a worker with no model, so this runs anywhere - but it
    is the real `crucible_higgs.serve.main`, the real protocol loop and the real
    `set_log_stream` call. The engine log lines are FORCED by a sitecustomize-
    style stub (see `_probe`) that calls `crucible_higgs.engine.log.log()` at the
    moments an engine would, so the assertion is about routing rather than about
    whether this particular fake happens to be chatty.
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

    VOICE = {'orpheus': 'leah', 'higgs-v3': 'deathstalker'}

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

    def test_the_orpheus_worker_keeps_stdout_json_only(self):
        out, _err = self._run('orpheus')
        messages = self._assert_json_only(out)
        self.assertEqual(messages[0]['type'], 'ready')

    def test_the_higgs_worker_keeps_stdout_json_only(self):
        out, _err = self._run('higgs-v3')
        messages = self._assert_json_only(out)
        self.assertEqual(messages[0]['type'], 'ready')

    def test_an_engine_log_line_lands_on_stderr_and_never_on_stdout(self):
        """THE REGRESSION ITSELF, forced rather than hoped for.

        `FakeEngine.__init__` logs through `crucible_higgs.engine.log.log()` at load,
        exactly as a real engine does - so this drives the real worker, the real
        protocol loop and the real routing. If it ever goes back to stdout, this
        fails with the precise symptom the Mac hit.

        BOTH assertions matter. Without the stderr one, a fix that simply
        stopped logging would pass while proving nothing.
        """
        from crucible_higgs.serve.fake_engine import FAKE_LOAD_MARKER
        out, err = self._run('orpheus')
        self.assertNotIn(FAKE_LOAD_MARKER, out,
                         'an engine log line reached the JSON protocol stream')
        self.assertIn(FAKE_LOAD_MARKER, err,
                      'the engine never logged at all - this test proves '
                      'nothing unless the line was actually emitted somewhere')
        self._assert_json_only(out)


class HostChoiceTest(unittest.TestCase):
    """The OTHER direction: the compat worker really does put them on stdout.

    parallel-tts-bridge.ts parses that stdout for the model-load bar, the MLX
    batch bar, the repair bar and the guard-event index. A default that quietly
    became stderr would break all four with no error anywhere.
    """

    def test_compat_worker_points_the_engine_log_at_stdout(self):
        probe = (
            'import sys, io\n'
            'import narrator.compat.worker as w\n'
            'import narrator.compat.app as app\n'
            'from crucible_higgs.engine.log import log_stream\n'
            'seen = {}\n'
            'app.main = lambda argv, engine_factory=None: seen.setdefault(\n'
            '    "stream", log_stream()) and 0\n'
            'w.main(["--session", "x"])\n'
            'print("STDOUT" if seen["stream"] is sys.stdout else "OTHER")\n')
        out = subprocess.run([sys.executable, '-c', probe], cwd=_PYTHON_ROOT,
                             capture_output=True, text=True)
        self.assertIn('STDOUT', out.stdout,
                      f'compat.worker must keep engine logs on stdout for the '
                      f'bridge to parse.\nstderr:\n{out.stderr}')

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
