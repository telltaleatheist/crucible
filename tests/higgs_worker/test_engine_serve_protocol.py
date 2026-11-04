"""The serve worker's JSON-lines protocol is a CONTRACT with the Node pool.

electron/orpheus-worker-pool.ts (BookForge@3b4d0b17) resolves a whole batch by
reading `type`, `i`, `seq`, `format`, `data`, `duration`, `sampleRate`,
`streamed`, `chunks`, `message` and `batch_done` off these lines. A missing
`batch_done` hangs the batch until a 180 s timeout TAINTS the worker; a second
message for one `i` is dropped as stale, so a row answered twice loses audio; a
`batch_chunk` for a row the pool did not mark `stream:true` is logged and
thrown away.

None of that could be tested before without a 6 GB model on a GPU. This drives
narrator/serve/worker.py AS A SUBPROCESS with NARRATOR_FAKE_ENGINE=1, over real
stdin/stdout pipes, so the reader thread, the stdout lock, the per-item
bookkeeping and the one-answer-per-row guarantees are the real ones.

WHAT IS ASSERTED, against the pool's own reads:
  1. handshake: 'ready' first, with a device (and a backend when known);
  2. 'load' -> 'status'* then exactly one 'loaded' carrying voice + backend;
  3. 'generate' -> one 'audio' with format/data/duration/sampleRate;
  4. 'generate' with stream:true -> 'chunk'(seq 0) then 'done'(duration, chunks,
     cancelled);
  5. 'generate_batch' -> one 'batch_item' per `i`, EXACTLY once, then exactly one
     'batch_done' whose count is the item count, and it is LAST;
  6. an empty item is answered with audio (the "empty -> silence" contract), not
     a failure;
  7. an unloaded voice fails THAT ITEM and nothing else;
  8. fast start: a batch with one stream:true item emits that row's
     'batch_chunk's with seq 0..n-1 and no gaps, then ONE terminal
     'batch_item' carrying streamed:true + duration + chunks and NO data, while
     the non-streamed rows keep the ordinary data-carrying shape;
  9. a batch before any load answers every item with 'Model not loaded' and
     still emits 'batch_done';
 10. 'quit' ends the process.

Run: python -m unittest discover -s python/narrator/tests -t python -p "test_engine_*.py"
"""
import array
import base64
import json
import os
import subprocess
import sys
import time
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_ROOT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "crucible", "jobs", "tts", "higgs")   # holds crucible_higgs
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)


def fake_higgs_band() -> dict:
    """The band a batch must carry to be JUDGED by the fake Higgs engine.

    Since Owen's ruling of 2026-09-19 the band comes from the BATCH and from
    nowhere else - `retake: true` with no band is `retake_without_band` - so a
    test that wants the ladder has to state one. It is read off the fake's own
    constants rather than written out here: the fake renders
    HIGGS_FRAMES_PER_CHAR frames per character, its pace follows from that, and
    a hard-coded triple would drift the day the fake's pace changed and these
    tests would silently start judging every chunk a runaway.
    """
    from crucible_higgs.serve.fake_engine import FakeHiggsEngine as F
    return {'paceCharsPerSec': F._PACE,
            'maxCharsPerSec': F.MAX_CHARS_PER_SEC,
            'minCharsPerSec': F.MIN_CHARS_PER_SEC}


def pcm16(data: str) -> array.array:
    """The wire's `data` field as int16 samples.

    `array` rather than numpy on purpose: this file asserts what the NODE POOL
    reads, and the pool decodes base64 into an Int16Array with no numpy in
    sight. A gap is a run of exact zeros in this sequence and nothing else.
    """
    samples = array.array('h')
    samples.frombytes(base64.b64decode(data))
    return samples


def trailing_zeros(samples) -> int:
    """How many samples of exact silence a row ends in. This is the measurement
    the gap-owner keepers make: an appended gap is `int(rate * seconds)` zeros
    and a second one would double it."""
    n = 0
    for value in reversed(samples):
        if value != 0:
            break
        n += 1
    return n


class Worker:
    """One serve worker subprocess, driven line by line."""

    def __init__(self, extra_env=None):
        env = dict(os.environ)
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        # The warmup would render three sentences plus two batches through the
        # fake before 'loaded'. Cheap, but it makes the status stream noisy and
        # it is not what this test is about; the pool's own spawn sets this the
        # same way when it wants a cold ready.
        env['ORPHEUS_SKIP_WARMUP'] = '1'
        env.pop('NARRATOR_GOLDEN_LOCAL', None)
        if extra_env:
            env.update(extra_env)
        # --fake-engine is an ARGV flag, not an env var, precisely so a leaked
        # variable can never put the production entry point into sine-tone mode
        # (see serve/worker.py's note). The test types it, as the design intends.
        self.proc = subprocess.Popen(
            [sys.executable, '-m', 'crucible_higgs.serve', '--fake-engine'],
            cwd=_PYTHON_ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', env=env, bufsize=1,
        )

    def send(self, **request):
        self.proc.stdin.write(json.dumps(request) + '\n')
        self.proc.stdin.flush()

    def read(self):
        """One protocol message, or None at EOF. Every stdout line must be JSON:
        the pool parses them all, and a stray print would be a parse error there."""
        line = self.proc.stdout.readline()
        if not line:
            return None
        return json.loads(line)

    def died_because(self):
        """Why the worker produced nothing - its own stderr, and its exit code.

        `readline()` blocks, so "no output" is never a slow start: it means the
        process ENDED. Without this the failure reads "worker produced no output
        at all", which is indistinguishable between a crash, a refused engine
        and a missing dependency. The worker prints all three on stderr.
        """
        try:
            self.proc.wait(timeout=60)
        except Exception:
            pass
        try:
            err = self.proc.stderr.read() or ''
        except Exception:
            err = '(stderr unreadable)'
        return (f'worker exited {self.proc.returncode} before its handshake.\n'
                f'--- stderr ---\n{err.strip() or "(empty)"}')

    def read_until(self, *types, limit=400):
        """Every message up to and including the first of `types`."""
        out = []
        for _ in range(limit):
            msg = self.read()
            if msg is None:
                raise AssertionError(f'worker closed stdout before {types}: {out}')
            out.append(msg)
            if msg['type'] in types:
                return out
        raise AssertionError(f'no {types} in {limit} messages: {out}')

    def close(self):
        try:
            self.send(action='quit')
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)
        try:
            err = self.proc.stderr.read()
        except Exception:
            err = ''
        # Close both pipes explicitly: unittest keeps a reference to the test case
        # long enough for the GC to report them as unclosed files otherwise, and a
        # ResourceWarning in the output is noise a real failure has to compete with.
        for handle in (self.proc.stdout, self.proc.stderr, self.proc.stdin):
            try:
                handle.close()
            except Exception:
                pass
        return err


class _WorkerCase(unittest.TestCase):
    """One worker subprocess per test, plus the assertions every batch test here
    makes. Split out from ServeProtocolTest so the GUARDED batch tests below can
    drive the same real worker with a different NARRATOR_ENGINE - the invariants
    the pool depends on are the same ones whichever engine rendered the rows."""

    #: Extra environment for this class's worker. None = the Orpheus fake, which
    #: is what every test in ServeProtocolTest was written against.
    WORKER_ENV = None

    def _assert_batch_closed(self, msgs, expected_i):
        """THE THREE INVARIANTS THE POOL DEPENDS ON, asserted the same way for
        every batch test in this file.

        1. 'batch_done' is the LAST message of the batch. A batch that never
           emits it hangs every one of its sentences until the pool's 180 s
           timeout, which TAINTS the worker and blocks all queued work.
        2. EXACTLY ONE message per item index. The pool deletes a resolver when
           it fires, so a second message for one `i` is dropped as stale - a row
           answered twice is a row whose real audio may have been the one thrown
           away. A dict keyed by `i` silently hides that, so this counts.
        3. Every requested index is answered. An unanswered row hangs the same
           way a missing batch_done does.

        Returns the per-index message so a caller can go on to assert its shape.
        """
        self.assertEqual(msgs[-1]['type'], 'batch_done',
                         'batch_done must be the LAST message of a batch')
        self.assertEqual(msgs[-1]['count'], len(expected_i))
        items = [m for m in msgs[:-1] if m['type'] == 'batch_item']
        seen = [m['i'] for m in items]
        dupes = sorted({i for i in seen if seen.count(i) > 1})
        self.assertEqual(dupes, [], f'row(s) answered more than once: {dupes}')
        self.assertEqual(sorted(seen), sorted(expected_i),
                         'every requested row must be answered exactly once')
        self.assertEqual([m['type'] for m in msgs[:-1] if m['type'] not in
                          ('batch_item', 'batch_chunk', 'status')], [],
                         'a batch emits only batch_item / batch_chunk / status')
        return {m['i']: m for m in items}

    def setUp(self):
        self.w = Worker(extra_env=self.WORKER_ENV)
        self.addCleanup(self._shutdown)

    def _worker_with_env(self, extra):
        """Restart this test's worker with `extra` on top of WORKER_ENV.

        Several of the variables these tests steer with - the fake's rate table,
        `ORPHEUS_STREAM_GAP` - are read ONCE at import or per render inside the
        worker, so changing one means a new subprocess. One owner for that
        restart, because every caller must also keep WORKER_ENV. `None` is
        WORKER_ENV's declared value for "this class adds nothing", not a
        missing one, which is why it is read by name rather than defaulted.
        """
        self.w.close()
        env = {} if self.WORKER_ENV is None else dict(self.WORKER_ENV)
        env.update(extra)
        self.w = Worker(extra_env=env)

    def _shutdown(self):
        err = self.w.close()
        # Surfaced only on failure; a traceback on stderr is how a silently
        # swallowed worker bug would otherwise hide.
        if self.w.proc.returncode not in (0, None) and err:
            print(err, file=sys.stderr)

    def _ready(self):
        msg = self.w.read()
        if msg is None:
            self.fail(self.w.died_because())
        self.assertEqual(msg['type'], 'ready')
        return msg

    def _load(self, voice='leah', **kwargs):
        self.w.send(action='load', voice=voice, warm=False, **kwargs)
        msgs = self.w.read_until('loaded', 'error')
        return msgs


class ServeProtocolTest(_WorkerCase):

    # ---- 1, 2 --------------------------------------------------------------

    def test_handshake_and_load(self):
        ready = self._ready()
        self.assertIn('device', ready)
        # `backend` is optional by design: absent means "unknown", which the pool
        # reads as NOT per-request capable. The fake always knows it.
        self.assertEqual(ready.get('backend'), 'transformers')

        msgs = self._load('leah')
        self.assertEqual(msgs[-1]['type'], 'loaded', msgs)
        self.assertEqual(msgs[-1]['voice'], 'leah')
        self.assertEqual(msgs[-1]['backend'], 'transformers')
        self.assertTrue(all(m['type'] == 'status' for m in msgs[:-1]), msgs)

    def test_handshake_advertises_the_per_item_take_channel(self):
        """A client driving a take ladder can tell whether this narrator HAS one.

        Measured 2026-09-15: Crucible sent take 1's `{"temperature": 0.7}` on
        every `generate_batch` item to a narrator whose env pin predated
        `engine/item_sampling.py`. That narrator's `_resolve_row` read only
        `item['voice']`, so the rung was dropped in silence and take 1 came back
        BYTE-IDENTICAL to take 0 - reported as a successful take 1. Nothing
        compared the recipe's pin with Crucible's belief about it.

        The handshake is where that comparison becomes possible, so the fact
        rides on `ready`. It is a BUILD fact - `ready` is sent before any engine
        loads - and says only that an item's rung is parsed at all: BOTH halves,
        `sampling` and `take`, under ONE key, because a build has both or
        neither. Whether the loaded ENGINE has a given lever or a seed lane is
        `accept_item_sampling` / `accept_item_take`'s answer, per row, as
        `sampling_not_supported` / `take_not_supported`.
        """
        ready = self._ready()
        self.assertIs(ready.get('itemTake'), True, ready)
        # And the name it replaced is GONE rather than kept beside it: two keys
        # for one build fact would be the two-owners shape the rename removed.
        self.assertNotIn('itemSampling', ready)

    def test_unknown_stock_voice_is_refused_not_substituted(self):
        self._ready()
        self.w.send(action='load', voice='nosuchvoice', warm=False)
        msgs = self.w.read_until('loaded', 'error')
        self.assertEqual(msgs[-1]['type'], 'error', msgs)
        self.assertIn('nosuchvoice', msgs[-1]['message'])
        self.assertIn('Refusing to substitute', msgs[-1]['message'])

    def test_caps_cross_the_wire(self):
        """The catalog payload reaches the engine's registry verbatim, and a key
        that is neither a cap nor an explicitly-ignored one fails the LOAD."""
        self._ready()
        msgs = self._load('leah', caps={'maxCharsPerSec': 23.5, 'eosBoost': 8,
                                        'maxChars': 450, 'sentenceGap': 0.0})
        self.assertEqual(msgs[-1]['type'], 'loaded', msgs)

    # ---- 3, 4 --------------------------------------------------------------

    def test_generate_single(self):
        self._ready()
        self._load()
        self.w.send(action='generate', text='Hello there, listener.')
        msgs = self.w.read_until('audio', 'error')
        audio = msgs[-1]
        self.assertEqual(audio['type'], 'audio', msgs)
        self.assertEqual(audio['format'], 'pcm16')
        self.assertEqual(audio['sampleRate'], 24000)
        self.assertGreater(audio['duration'], 0.0)
        self.assertTrue(audio['data'])

    def test_generate_stream_mode(self):
        """`stream:true` on a SINGLE generate is the scheduler's
        streaming-first-sentence contract: one chunk with seq 0, then 'done'."""
        self._ready()
        self._load()
        self.w.send(action='generate', text='One streamed sentence.', stream=True)
        chunk = self.w.read_until('chunk', 'error')[-1]
        self.assertEqual(chunk['type'], 'chunk')
        self.assertEqual(chunk['seq'], 0)
        self.assertEqual(chunk['format'], 'pcm16')
        self.assertEqual(chunk['sampleRate'], 24000)
        done = self.w.read_until('done', 'error')[-1]
        self.assertEqual(done['type'], 'done')
        self.assertEqual(done['chunks'], 1)
        self.assertFalse(done['cancelled'])
        self.assertAlmostEqual(done['duration'], chunk['duration'], places=6)

    def test_generate_with_no_text_errors(self):
        self._ready()
        self._load()
        self.w.send(action='generate', text='')
        err = self.w.read_until('error')[-1]
        self.assertEqual(err['message'], 'No text provided')

    # ---- 5, 6, 7 -----------------------------------------------------------

    def test_generate_batch_one_item_each_then_batch_done(self):
        self._ready()
        self._load()
        items = [{'i': 10, 'text': 'First sentence of the block.'},
                 {'i': 11, 'text': 'Second sentence, a little longer than the first one.'},
                 {'i': 12, 'text': ''},
                 {'i': 13, 'text': 'Fourth.'}]
        self.w.send(action='generate_batch', items=items)
        msgs = self.w.read_until('batch_done')
        by_i = self._assert_batch_closed(msgs, [10, 11, 12, 13])

        for i in (10, 11, 13):
            item = by_i[i]
            self.assertEqual(item['format'], 'pcm16', item)
            self.assertEqual(item['sampleRate'], 24000)
            self.assertGreater(item['duration'], 0.0)
            self.assertTrue(item['data'])
            self.assertNotIn('message', item)
        # 11 is longer than 10, and the fake's duration is a function of length.
        self.assertGreater(by_i[11]['duration'], by_i[10]['duration'])
        # The EMPTY row is answered with a tiny silence, NOT with a failure:
        # "empty -> silence" is the worker's contract and the pool would mark a
        # message-carrying item as a failed sentence.
        self.assertIn('data', by_i[12])
        self.assertNotIn('message', by_i[12])
        self.assertLess(by_i[12]['duration'], 0.2)

    def test_an_unguarded_engine_sends_no_guard_key(self):
        """`guard` is ADDITIVE AND OPTIONAL (PHASE6-REMOTE-RENDER.md section 3).

        Orpheus offers no `render_many` - its guard is the older
        `_guard_truncation`, which reaches no verdict object - so its rows must
        carry exactly the field set orpheus-worker-pool.ts and the browser
        extension read today. A key appearing here would mean the new arm was
        selected by something other than the capability."""
        self._ready()
        self._load()
        self.w.send(action='generate_batch',
                    items=[{'i': 0, 'text': 'An ordinary Orpheus sentence.'}])
        by_i = self._assert_batch_closed(self.w.read_until('batch_done'), [0])
        self.assertIn('data', by_i[0])
        self.assertNotIn('guard', by_i[0])

    def test_unloaded_voice_fails_only_its_own_item(self):
        self._ready()
        self._load('leah')
        items = [{'i': 0, 'text': 'Ordinary sentence.'},
                 {'i': 1, 'text': 'Wrong voice.', 'voice': 'zoe'},
                 {'i': 2, 'text': 'Another ordinary one.'}]
        self.w.send(action='generate_batch', items=items)
        msgs = self.w.read_until('batch_done')
        by_i = self._assert_batch_closed(msgs, [0, 1, 2])
        self.assertIn('data', by_i[0])
        self.assertIn('data', by_i[2])
        self.assertNotIn('data', by_i[1])
        self.assertIn('zoe', by_i[1]['message'])

    def test_batch_before_load(self):
        self._ready()
        items = [{'i': 0, 'text': 'a'}, {'i': 1, 'text': 'b'}]
        self.w.send(action='generate_batch', items=items)
        msgs = self.w.read_until('batch_done')
        by_i = self._assert_batch_closed(msgs, [0, 1])
        for m in by_i.values():
            self.assertEqual(m['message'], 'Model not loaded')

    # ---- 8: fast start ------------------------------------------------------

    def test_fast_start_batch_chunks_then_streamed_item(self):
        self._ready()
        self._load()
        items = [
            {'i': 100, 'text': 'The row the listener is on, which streams.', 'stream': True},
            {'i': 101, 'text': 'Read-ahead behind it, which does not.'},
        ]
        self.w.send(action='generate_batch', items=items)
        msgs = self.w.read_until('batch_done')
        by_i = self._assert_batch_closed(msgs, [100, 101])

        chunks = [m for m in msgs if m['type'] == 'batch_chunk']
        item_msgs = [m for m in msgs if m['type'] == 'batch_item']
        self.assertEqual({m['i'] for m in chunks}, {100},
                         'only the row marked stream:true may emit batch_chunk')
        self.assertEqual([m['seq'] for m in chunks], list(range(len(chunks))),
                         'seq must run 0..n-1 with no gaps and no repeats')
        for c in chunks:
            self.assertEqual(c['format'], 'pcm16')
            self.assertEqual(c['sampleRate'], 24000)
            self.assertTrue(c['data'])
            self.assertGreater(c['duration'], 0.0)

        streamed = by_i[100]
        self.assertTrue(streamed['streamed'])
        self.assertNotIn('data', streamed,
                         'a streamed terminal carries totals only - its audio already left')
        self.assertEqual(streamed['chunks'], len(chunks))
        self.assertAlmostEqual(streamed['duration'],
                               sum(c['duration'] for c in chunks), places=5)
        # The non-streamed row in the SAME batch keeps the classic shape.
        self.assertIn('data', by_i[101])
        self.assertNotIn('streamed', by_i[101])

        # ORDER: every chunk of row 100 precedes its terminal batch_item.
        first_terminal = min(i for i, m in enumerate(msgs)
                             if m['type'] == 'batch_item' and m['i'] == 100)
        last_chunk = max(i for i, m in enumerate(msgs) if m['type'] == 'batch_chunk')
        self.assertLess(last_chunk, first_terminal)

    def test_both_listen_arms_state_the_gap_and_append_no_silence(self):
        """THE PLAYER REALIZES THE GAP, so narrator sends the NUMBER and bare
        speech (Owen, 2026-09-18: "yes, it paces like the book... maybe the
        browser extension should handle the gaps for itself"). On the stream
        there is no assembler but the player, and the RULE stays narrator's -
        `text/gaps.classify_gap`, the same call prep makes for the book.

        Both of fast start's arms are in this one batch and neither may pad:

          row 7   streams, so its audio left as raw chunks. There is no final
                  all-silent chunk any more - the gap is on its terminal record.
          row 8   does not stream, so it goes out whole through `finalize_audio`,
                  which appends nothing at either door now.

        `NARRATOR_SENTENCE_GAP` is the floor `classify_gap` reads - the voice's
        inject, set the same way the prep door sets it - so a value nothing else
        in this worker knows about proves the number came from that function and
        not from a constant.
        """
        gap_seconds = 0.42
        self._worker_with_env({'NARRATOR_SENTENCE_GAP': str(gap_seconds)})
        self._ready()
        self._load()
        items = [{'i': 7, 'text': 'The row the listener is on.', 'stream': True},
                 {'i': 8, 'text': 'Read-ahead behind it, buffered whole.'}]
        self.w.send(action='generate_batch', items=items)
        msgs = self.w.read_until('batch_done')
        by_i = self._assert_batch_closed(msgs, [7, 8])

        chunks = [m for m in msgs if m['type'] == 'batch_chunk']
        silent = [c for c in chunks
                  if trailing_zeros(pcm16(c['data'])) == len(pcm16(c['data']))]
        self.assertEqual(len(silent), 0,
                         'a streamed row still carries a silent gap chunk; the '
                         'player inserts the gap now')
        self.assertAlmostEqual(by_i[7]['gapSec'], gap_seconds, places=6)
        self.assertAlmostEqual(by_i[7]['duration'],
                               sum(c['duration'] for c in chunks), places=5)

        self.assertAlmostEqual(by_i[8]['gapSec'], gap_seconds, places=6)
        self.assertEqual(trailing_zeros(pcm16(by_i[8]['data'])), 0,
                         'the buffered Listen row still ends in appended silence')

    def test_the_gap_is_classify_gaps_own_answer_for_that_row(self):
        """EQUAL TO THE BOOK'S, row by row, because it is the same function.

        `[pause:2.5]` is the one thing that still moves `classify_gap`'s answer
        (the paragraph and section tiers were removed on 2026-07-17), so a batch
        of a plain sentence and a deliberate beat is the smallest pair that can
        tell "narrator classified this row" from "narrator sent its floor".
        Expected values are computed by CALLING `classify_gap_seconds` under the
        same env the worker runs with - a number typed in here would be a second
        copy of the rule.

        The classic arm on purpose: no item streams, so this is the batch path
        the extension's "buffer before playing" default takes.
        """
        from crucible_higgs.text.gaps import classify_gap_seconds
        gap_seconds = 0.42
        texts = {1: 'A bare sentence of prose.',
                 2: 'A deliberate beat follows this. [pause:2.5]'}
        with mock.patch.dict(os.environ,
                             {'NARRATOR_SENTENCE_GAP': str(gap_seconds)}):
            expected = {i: classify_gap_seconds(t)[1] for i, t in texts.items()}
        self.assertNotEqual(expected[1], expected[2], expected)

        self._worker_with_env({'NARRATOR_SENTENCE_GAP': str(gap_seconds)})
        self._ready()
        self._load()
        self.w.send(action='generate_batch',
                    items=[{'i': i, 'text': t} for i, t in sorted(texts.items())])
        by_i = self._assert_batch_closed(self.w.read_until('batch_done'), [1, 2])
        for i in texts:
            self.assertAlmostEqual(by_i[i]['gapSec'], expected[i], places=6,
                                   msg=f'row {i} did not get classify_gap\'s answer')

    def test_a_row_whose_gap_leads_is_refused_by_name(self):
        """THERE IS NO CHANNEL FOR A LEAD GAP ON THIS DOOR, and half an answer
        delivered silently is the shape this whole fix removes.

        `classify_gap` returns a non-zero LEAD only for a chunk that starts with
        an explicit `[pause:X]`; a heading returns `(0.0, floor)` like everything
        else. A lead has to be realized BEFORE the row's first sample, and on a
        fast-start row that sample has already been played by the time any
        terminal record arrives - so the row is failed BY NAME, on its own,
        rather than shipped with its opening beat dropped.
        """
        self._ready()
        self._load()
        items = [{'i': 0, 'text': '[pause:3.0] After a long beat.'},
                 {'i': 1, 'text': 'Its neighbour is untouched.'}]
        self.w.send(action='generate_batch', items=items)
        by_i = self._assert_batch_closed(self.w.read_until('batch_done'), [0, 1])
        self.assertTrue('data' not in by_i[0],
                        'the leading-pause row rendered instead of being refused')
        self.assertIn('lead', by_i[0].get('message', ''))
        self.assertIn('[pause:', by_i[0].get('message', ''))
        self.assertTrue('data' in by_i[1], 'the neighbour must render normally')

    # ---- 9, 10 --------------------------------------------------------------

    def test_cancel_is_acknowledged(self):
        self._ready()
        self._load()
        self.w.send(action='cancel')
        self.assertEqual(self.w.read_until('stopped')[-1]['type'], 'stopped')

    def test_cancel_DURING_a_batch_still_closes_it(self):
        """THE CASE THE WORKER'S DOCSTRING NAMES, and the one nothing tested.

        A 'cancel' arrives on stdin while the main thread is blocked inside the
        engine. The reader thread flips the flag; the engine abandons what it has
        not rendered and returns WITHOUT an on_row for those rows - so if the
        worker's `finally` sweep did not label them, they would never be answered
        and the pool would hang each one until its 180 s timeout, which taints
        the worker and blocks every queued batch behind it.

        What must hold, cancel or not: one message per row, exactly once, and
        'batch_done' last. The cancelled rows carry message 'cancelled'.
        """
        self._ready()
        self._load()
        n = 10
        items = [{'i': i, 'text': f'Sentence number {i} of a read-ahead block.',
                  'stream': True} for i in range(n)]
        self.w.send(action='generate_batch', items=items)
        # Written immediately behind the batch, so it lands while the fake is
        # still working through its rows (STREAM_ROW_SECONDS each).
        self.w.send(action='cancel')

        msgs = self.w.read_until('batch_done')
        by_i = self._assert_batch_closed(msgs, list(range(n)))

        cancelled = [i for i, m in by_i.items() if m.get('message') == 'cancelled']
        self.assertTrue(cancelled,
                        'the cancel must have abandoned at least one row; if this '
                        'is flaky the fake got faster than the pipe round-trip')
        # Whatever DID render still rendered properly - a cancel does not
        # retract audio that was already delivered.
        for i, m in by_i.items():
            if i in cancelled:
                self.assertNotIn('data', m)
                self.assertNotIn('streamed', m)
            else:
                self.assertTrue(m.get('streamed') or m.get('data'), m)
        # And the cancel is still acknowledged afterwards, in arrival order.
        self.assertEqual(self.w.read_until('stopped')[-1]['type'], 'stopped')

    def test_a_batch_after_a_cancel_is_not_suppressed(self):
        """The flag is cleared where the cancel is DEQUEUED, and only there - so
        a batch queued behind the acknowledgement renders normally. A flag that
        outlived its request would silently cancel the next block too."""
        self._ready()
        self._load()
        self.w.send(action='generate_batch',
                    items=[{'i': 0, 'text': 'One.', 'stream': True}])
        self.w.send(action='cancel')
        self._assert_batch_closed(self.w.read_until('batch_done'), [0])
        self.assertEqual(self.w.read_until('stopped')[-1]['type'], 'stopped')

        items = [{'i': 7, 'text': 'A fresh block after the cancel.'}]
        self.w.send(action='generate_batch', items=items)
        by_i = self._assert_batch_closed(self.w.read_until('batch_done'), [7])
        self.assertIn('data', by_i[7], 'the batch after a cancel must render')
        self.assertNotIn('message', by_i[7])

    def test_unknown_action(self):
        self._ready()
        self.w.send(action='fly')
        self.assertIn('Unknown action', self.w.read_until('error')[-1]['message'])

    def test_bad_json_is_reported_not_fatal(self):
        self._ready()
        self.w.proc.stdin.write('{not json\n')
        self.w.proc.stdin.flush()
        err = self.w.read_until('error')[-1]
        self.assertIn('Invalid JSON', err['message'])
        # Still alive afterwards.
        self._load()

    def test_quit_ends_the_process(self):
        self._ready()
        self.w.send(action='quit')
        self.w.proc.stdin.close()
        self.assertEqual(self.w.proc.wait(timeout=20), 0)

    def test_quit_STOPS_the_loaded_engine_before_the_process_ends(self):
        """MEASURED 2026-09-27 (Crucible 1.0.43, narrator 689446): 'quit' broke
        the loop and the process exited with no stop lines at all, so a served
        Higgs engine's launched server was left for the setsid watchdog to TERM
        later, while it still held its port. The engine's own cleanup must run,
        and be waited for, before the process ends."""
        self._ready()
        self._load()
        self.w.send(action='quit')
        self.w.proc.stdin.close()
        self.assertEqual(self.w.proc.wait(timeout=20), 0)
        err = self.w.proc.stderr.read()
        self.assertIn('exiting: stopping the loaded engine', err)
        self.assertIn('engine stopped', err)


class GuardedBatchTest(_WorkerCase):
    """THE SERVE WORLD'S BATCH NOW RUNS THE RETAKE LADDER, and says what it did.

    Owen ruled on 2026-09-13 that the model and its inference own the guard AND
    the retake decision (crucible/docs/PHASE6-REMOTE-RENDER.md). Before that,
    narrator had two rendering worlds: the audiobook one drove `truncation.
    GuardPlan` - PaceTracker, re-roll, split ladder - and the serve one, which is
    the door Crucible's render job drives, handed a Higgs engine one bare
    `render_audio()` per sentence and guarded nothing. `generate_batch` now routes
    an engine that offers `render_many` through that engine's own guarded driver
    and puts the verdict on `batch_item.guard`.

    Driven with NARRATOR_ENGINE=higgs-v3 + --fake-engine, so the WORKER is the
    real one - real reader thread, real stdout lock, real one-answer-per-row
    bookkeeping - and only the model is fake. FakeHiggsEngine.render_many builds a
    real `truncation.GuardPlan`, so the ladder under test is the shipping ladder;
    what the fake replaces is the audio, and NARRATOR_FAKE_HIGGS_RATE bends one
    chunk's duration so the guard fires deterministically instead of needing a
    card and a model that happens to misbehave.

    WHAT THESE TESTS CANNOT PROVE, and step 4 of PHASE6 section 7 is the gate for
    it: that the ladder makes the SAME decisions on a real card as it did before
    this wiring. A fake engine renders a sine wave against a band derived from its
    own pace; only a chapter rendered on a 3090 Ti and compared against the
    Mistborn pause map can say the policy did not move.
    """

    #: The fake's guard-steering knob is read per render, so the worker must be
    #: started with it already set - hence one subprocess per rate table.
    WORKER_ENV = {'NARRATOR_ENGINE': 'higgs-v3'}

    def _worker_with_rate(self, rate_json):
        """Restart this test's worker with a NARRATOR_FAKE_HIGGS_RATE table."""
        self._worker_with_env({'NARRATOR_FAKE_HIGGS_RATE': rate_json})

    def _higgs_batch(self, items, **extra):
        """A JUDGED batch: `retake: true` and the band it is judged against.

        Both are required together (`retake_without_band`), and this class is
        about the ladder, so every batch here asks for it. `extra` is for the
        tests that vary one key - a width, a malformed band, a batch that asks
        for no judging at all.
        """
        self._ready()
        self._load('deathstalker')
        request = {'retake': True, 'band': fake_higgs_band()}
        request.update(extra)
        self.w.send(action='generate_batch', items=items, **request)
        msgs = self.w.read_until('batch_done')
        return self._assert_batch_closed(msgs, [it['i'] for it in items])

    def test_the_worker_really_built_the_higgs_fake(self):
        """If this fails every other test in the class is testing Orpheus. `pads`
        is the tell: Higgs emits bare speech, Orpheus bakes its own silence in."""
        self._ready()
        loaded = self._load('deathstalker')[-1]
        self.assertEqual(loaded['type'], 'loaded', loaded)
        self.assertEqual(loaded['engine'], 'higgs-v3')
        self.assertFalse(loaded['pads'])

    def test_a_clean_batch_carries_a_clean_verdict(self):
        """The common case, and the one that costs nothing: a chunk the ladder
        never touched reports verdict 'clean', one part, and an EMPTY take list -
        `_LadderTask.offer` finishes before it records an event, so there is no
        evidence to carry because nothing happened."""
        items = [{'i': 0, 'text': 'The first chunk of the chapter, rendered clean.'},
                 {'i': 1, 'text': 'The second one, which is also perfectly ordinary.'},
                 {'i': 2, 'text': 'And a third.'}]
        by_i = self._higgs_batch(items)
        for i in (0, 1, 2):
            item = by_i[i]
            self.assertIn('data', item, item)
            self.assertNotIn('message', item)
            guard = item.get('guard')
            self.assertIsNotNone(guard, f'row {i} reached the wire with no verdict')
            self.assertEqual(guard['verdict'], 'clean', guard)
            self.assertTrue(guard['clean'])
            self.assertEqual(guard['parts'], 1)
            self.assertEqual(guard['takes'], [], 'a clean take 0 emits no event')
            # The band is the evidence behind the verdict, and it is the
            # TRACKER's - read fresh per take - not a constant.
            self.assertIn('max_chars_per_sec', guard['band'])
            self.assertIn('min_chars_per_sec', guard['band'])
            self.assertIn('observed', guard['band'])

    def test_the_render_door_emits_bare_speech(self):
        """THE GAP BELONGS TO WHOEVER ASSEMBLES (Owen, 2026-09-18): "whoever
        assembles them is who owns the gap. I think that's bookforge."

        This is the door Crucible's `tts` render job drives, and behind it
        BookForge's narrator assembler realizes `gaps.json` - 0.6 s, or the
        voice's inject - for a `pads=False` engine. `finalize_audio` used to
        append `STREAM_GAP_SEC` here as well, so every join of a remotely
        rendered book came out model tail + 0.30 baked into the chunk + 0.60
        from the assembler: the thirdreich "long on every join" defect, on
        every voice.

        MEASURED TWO WAYS, because one of them alone could pass by accident.
        A row rendered with a sentence gap configured must be BYTE-FOR-BYTE the
        row rendered with none - the gap reaches a rendered chunk through
        `gaps.json`, written at prep, and nothing at render time may anticipate
        it - and its audio must not end in silence, since the fake's tone runs to
        the last sample it decoded.

        AND IT CARRIES NO `gapSec`. That field is the LISTEN door's statement to
        a player that has no assembler behind it; on this door the assembler
        realizes `gaps.json`, and a second number on the wire would be a second
        owner of the same silence.
        """
        items = [{'i': 0, 'text': 'The chunk a remote render asks narrator for.'}]
        self._worker_with_env({'NARRATOR_SENTENCE_GAP': '0.5'})
        padded = self._higgs_batch(items)[0]
        self._worker_with_env({'NARRATOR_SENTENCE_GAP': '0'})
        bare = self._higgs_batch(items)[0]

        self.assertEqual(trailing_zeros(pcm16(padded['data'])), 0,
                         'the render door appended silence the assembler will '
                         'append again')
        self.assertEqual(padded['data'], bare['data'],
                         'NARRATOR_SENTENCE_GAP changed a RENDERED row; the gap '
                         'is realized by the assembler from gaps.json and this '
                         'door must not anticipate it')
        self.assertAlmostEqual(padded['duration'], bare['duration'], places=9)
        self.assertTrue('gapSec' not in padded,
                        'the render door stated a gap the assembler also owns')

    def test_a_bent_chunk_reports_its_reroll(self):
        """One bad take, then a good one: the ladder re-rolls and the verdict says
        so. `clean` stays True because the take that SHIPPED is inside the band -
        the verdict names what happened, not whether anything happened.

        The bent index is the CALLER'S `i`, not the position in the batch, which
        is also what this proves: the rate table is keyed by chunk index, and if
        the worker had passed the enumerate position the wrong row would bend."""
        self._worker_with_rate('{"41": 0.4}')
        items = [{'i': 40, 'text': 'The chunk before the bent one.'},
                 {'i': 41, 'text': 'The chunk whose first take comes back far too fast.'},
                 {'i': 42, 'text': 'The chunk after it.'}]
        by_i = self._higgs_batch(items)

        bent = by_i[41]
        self.assertIn('data', bent, bent)
        guard = bent['guard']
        self.assertEqual(guard['verdict'], 'rerolled', guard)
        self.assertTrue(guard['clean'], 'the re-roll landed inside the band')
        self.assertEqual(guard['parts'], 1, 'a re-roll is one chunk, not two')
        self.assertEqual(len(guard['takes']), 2,
                         'the bad take and the re-roll that replaced it')
        # The records are the LADDER'S OWN, forwarded verbatim - not a shape this
        # worker invents. `rung` names the step the ladder took after judging the
        # take, so take 0 is the one that sent it to the re-roll.
        self.assertEqual(guard['takes'][0]['rung'], 'reroll', guard['takes'][0])
        self.assertEqual(guard['takes'][0]['side'], 'short')
        for record in guard['takes']:
            for field in ('index', 'chars', 'seconds', 'chars_per_second',
                          'action', 'rung'):
                self.assertIn(field, record, record)
            self.assertEqual(record['index'], 41,
                             'the record is keyed by the CALLER\'s chunk index')
        # ONLY that row. Its neighbours are untouched - a guard that fired on the
        # batch instead of the chunk would show up here.
        for i in (40, 42):
            self.assertEqual(by_i[i]['guard']['verdict'], 'clean', by_i[i])

    def test_a_split_chunk_is_still_exactly_one_item(self):
        """THE HARD CASE (PHASE6 section 5). Take 0 and the re-roll are both bad,
        so the ladder splits the chunk and renders it as two halves - and joins
        them before it retires. One requested index, ONE artifact, always:
        returning two rows would push the ladder's private business into every
        consumer, break BookForge's `<i>.flac` resume scan and desynchronise the
        .sentences.vtt sidecar from the audio.

        `parts: 2` is how the split is reported, and it is the only sign of it.

        The text is two sentences of over `MIN_SPLIT_CHARS` (80) each on purpose:
        `split_halves` refuses a cut that would leave a half under that, and a
        chunk it cannot split is ACCEPTED off-length at the end of the ladder
        instead - a different and equally real verdict, but not this case."""
        self._worker_with_rate('{"41": [0.4, 0.4]}')
        items = [{'i': 40, 'text': 'The chunk before the split one.'},
                 {'i': 41, 'text': 'The first half of a chunk that will not settle '
                                   'no matter how many times it is rendered. '
                                   'And the second half of that very same chunk, '
                                   'which is every bit as stubborn about it.'},
                 {'i': 42, 'text': 'The chunk after it.'}]
        by_i = self._higgs_batch(items)

        split = by_i[41]
        self.assertIn('data', split, split)
        guard = split['guard']
        self.assertEqual(guard['verdict'], 'resplit', guard)
        self.assertEqual(guard['parts'], 2, 'the ladder split it into two halves')
        self.assertTrue(guard['clean'], 'the two halves both landed inside the band')
        self.assertEqual(len(guard['takes']), 2,
                         'take 0 and the re-roll that also came back bent; the '
                         'halves rendered clean and record nothing')
        self.assertEqual([r['rung'] for r in guard['takes']], ['reroll', 'split'])
        for i in (40, 42):
            self.assertEqual(by_i[i]['guard']['verdict'], 'clean', by_i[i])
        # _assert_batch_closed already counted one message per index and put
        # batch_done last; say the split-specific half of that out loud, because
        # it is the invariant this case is most likely to break.
        self.assertEqual(sorted(by_i), [40, 41, 42])

    def test_a_guarded_batch_keeps_the_empty_row_contract(self):
        """An empty chunk never goes on the ladder - there is no take to judge,
        and a Higgs `render_audio` refuses a blank chunk by name - so it keeps the
        tiny-silence answer every other batch path here gives it, and carries no
        verdict because nothing decided anything about it."""
        items = [{'i': 0, 'text': 'A real sentence.'},
                 {'i': 1, 'text': ''},
                 {'i': 2, 'text': 'Another real one.'}]
        by_i = self._higgs_batch(items)
        self.assertIn('data', by_i[1])
        self.assertNotIn('message', by_i[1])
        self.assertNotIn('guard', by_i[1])
        self.assertLess(by_i[1]['duration'], 0.2)
        for i in (0, 2):
            self.assertEqual(by_i[i]['guard']['verdict'], 'clean')

    def test_an_unloaded_voice_still_fails_only_its_own_row(self):
        """The guarded arm keeps the per-row refusal the sequential one had:
        `render_many` takes no voice and renders in whatever is loaded, so a row
        naming another voice must be refused rather than narrated by the wrong
        one and reported as a success."""
        items = [{'i': 0, 'text': 'Ordinary sentence.'},
                 {'i': 1, 'text': 'Wrong voice.', 'voice': 'someone-else'},
                 {'i': 2, 'text': 'Another ordinary one.'}]
        by_i = self._higgs_batch(items)
        self.assertIn('data', by_i[0])
        self.assertIn('data', by_i[2])
        self.assertNotIn('data', by_i[1])
        self.assertNotIn('guard', by_i[1])
        self.assertIn('someone-else', by_i[1]['message'])

    def test_a_cancel_stops_the_guarded_ladder_between_chunks(self):
        """THE HOLE OWEN HIT, and the reason this test exists at all.

        Measured on the Mac Studio, 2026-09-15: a Crucible `tts` job rendering
        `thirdreich` was cancelled at chunk 38 of 89. Crucible's door recorded
        `cancelling` and sent `{"action": "cancel"}`; this worker's reader thread
        set the flag, as it always did - and `_emit_guarded_batch` never read it.
        The ladder ran on for eleven more minutes, holding narrator's wire, the
        job's claim on the card, and the voice on it. Every OTHER batch arm here
        has been cancellable since the reader thread landed
        (`test_cancel_DURING_a_batch_still_closes_it` covers the streamed one);
        the guarded arm is the only one Crucible's render door drives, and it was
        written on 2026-09-13 without a check.

        What must hold is what holds for every other arm: one message per row
        exactly once, 'batch_done' last, abandoned rows carrying 'cancelled', and
        the rows that DID render still carrying real audio - a cancel stops work,
        it does not retract what was already delivered.

        NARRATOR_FAKE_HIGGS_ROW_MS is what gives the cancel somewhere to land: the
        fake retires a batch of sine waves faster than the pipe round-trip, which
        is exactly why a guarded cancel test could not have failed before.
        """
        self.w.close()
        env = dict(self.WORKER_ENV)
        env['NARRATOR_FAKE_HIGGS_ROW_MS'] = '120'
        self.w = Worker(extra_env=env)

        self._ready()
        self._load('deathstalker')
        n = 12
        items = [{'i': i, 'text': f'Sentence number {i} of a chapter being rendered.'}
                 for i in range(n)]
        # JUDGED: the cancel this test is about is the GUARDED arm's, which a
        # batch only takes when it asks for it (Owen, 2026-09-19).
        self.w.send(action='generate_batch', items=items,
                    retake=True, band=fake_higgs_band())
        # LONG ENOUGH THAT THE LADDER IS ALREADY RUNNING, and that is the whole
        # point of the delay: a cancel written immediately behind the batch is
        # caught by the check BEFORE the driver is started, which is a different
        # (and much easier) path. Two or three rows in, with nine still to go, is
        # where the between-chunks check is the only thing that can stop it.
        time.sleep(0.3)
        self.w.send(action='cancel')

        by_i = self._assert_batch_closed(self.w.read_until('batch_done'),
                                         list(range(n)))
        cancelled = [i for i, m in by_i.items() if m.get('message') == 'cancelled']
        rendered = [i for i in by_i if i not in cancelled]
        self.assertTrue(cancelled,
                        'the cancel must have abandoned at least one row; before '
                        'the fix every row rendered and this list was empty')
        self.assertTrue(rendered,
                        'the ladder must have been RUNNING when the cancel landed - '
                        'if nothing rendered, the pre-driver check stopped it and '
                        'the between-chunks one is untested')
        for i, m in by_i.items():
            if i in cancelled:
                self.assertNotIn('data', m, m)
                self.assertNotIn('guard', m, m)
            else:
                self.assertIn('data', m, m)
                self.assertIsNotNone(m.get('guard'), m)
        # The cancel is still acknowledged, in arrival order, after the batch it
        # aborted - which is what lets the NEXT batch render (the flag is cleared
        # only where it is dequeued).
        self.assertEqual(self.w.read_until('stopped')[-1]['type'], 'stopped')


class RetakeFlagTest(_WorkerCase):
    """WHO DECIDES WHETHER NARRATOR JUDGES (Owen's ruling, 2026-09-19).

    Until this landed, an engine that COULD guard a batch always did - the arm
    was picked by a capability probe on `render_many` and the caller had no
    say. The fine-tuning ladder is why that had to change: a screening render
    of a checkpoint has no measured pace and no measured cap, because MEASURING
    THEM IS WHAT THE RENDER IS FOR, so judging it means judging one model
    against another model's band. BookForge wants the opposite - the guarded
    render, with the band it read off the voice row - and both are one flag.

    Driven with NARRATOR_ENGINE=higgs-v3 + --fake-engine, so the WORKER and the
    ladder are the real ones and only the audio is fake.
    """

    WORKER_ENV = {'NARRATOR_ENGINE': 'higgs-v3'}

    def _send(self, items, **extra):
        # ONE handshake per worker, however many batches a test sends: `ready`
        # arrives once, so a second `_ready()` would wait for a line that is
        # never coming.
        if not getattr(self, '_prepared', False):
            self._ready()
            self._load('deathstalker')
            self._prepared = True
        self.w.send(action='generate_batch', items=items, **extra)
        return self._assert_batch_closed(self.w.read_until('batch_done'),
                                         [it['i'] for it in items])

    #: A chunk whose take 0 comes back FAR too short for its text - the shape
    #: the guard exists for (Fuhrer chunk 19: 1,127 characters in 3.0 s).
    BENT = json.dumps({'1': 0.35})

    ITEMS = [{'i': 0, 'text': 'An ordinary opening chunk of the chapter here.'},
             {'i': 1, 'text': 'The chunk whose first take comes back far too '
                              'short for the text it was given, which is what '
                              'the length guard exists to catch and re-roll.'}]

    def _with_rate(self, rate_json):
        """Restart on a rate table. A new subprocess is a new handshake."""
        self._worker_with_env({'NARRATOR_FAKE_HIGGS_RATE': rate_json})
        self._prepared = False

    def test_retake_true_judges_and_says_so(self):
        self._with_rate(self.BENT)
        by_i = self._send(self.ITEMS, retake=True, band=fake_higgs_band())
        self.assertIn('data', by_i[1], by_i[1])
        self.assertIsNotNone(by_i[1].get('guard'), by_i[1])
        self.assertEqual(by_i[1]['guard']['verdict'], 'rerolled', by_i[1])

    def test_retake_absent_renders_the_row_once_and_judges_NOTHING(self):
        """The bare arm: the same bent chunk ships as it came back, with no
        verdict at all. `guard` ABSENT - not 'clean' - because nobody judged
        it, and a clean verdict on an unjudged row would be narrator claiming
        to have looked."""
        self._with_rate(self.BENT)
        by_i = self._send(self.ITEMS)
        for i in (0, 1):
            self.assertIn('data', by_i[i], by_i[i])
            self.assertNotIn('guard', by_i[i], by_i[i])

    def test_retake_false_is_the_same_as_absent(self):
        by_i = self._send(self.ITEMS, retake=False)
        for i in (0, 1):
            self.assertIn('data', by_i[i], by_i[i])
            self.assertNotIn('guard', by_i[i], by_i[i])

    def test_retake_true_with_no_band_is_refused_BY_NAME(self):
        """THE DEFECT THIS REFUSAL EXISTS FOR. Before the ruling the tracker
        fell back to the ENGINE's default band when the voice carried none: a
        band centred at 15.0 chars/s against a book actually running near 17.2
        called healthy chunks run-ons and re-rolled them to MAX_DEPTH. So the
        batch that asks to be judged carries the band it is judged against, or
        it is refused whole."""
        by_i = self._send(self.ITEMS, retake=True)
        for i in (0, 1):
            self.assertNotIn('data', by_i[i], by_i[i])
            self.assertTrue(by_i[i]['message'].startswith('retake_without_band'),
                            by_i[i])

    def test_a_MALFORMED_band_is_refused_BY_NAME(self):
        good = fake_higgs_band()
        for band in ({k: v for k, v in good.items() if k != 'minCharsPerSec'},
                     {**good, 'minCharsPerSec': 0},
                     {**good, 'paceCharsPerSec': good['maxCharsPerSec'] * 2},
                     'a band'):
            with self.subTest(band=band):
                by_i = self._send(self.ITEMS, retake=True, band=band)
                for i in (0, 1):
                    self.assertTrue(
                        by_i[i]['message'].startswith('band_malformed'), by_i[i])

    def test_a_band_sent_WITHOUT_retake_is_accepted_and_not_used(self):
        """Owen: "it won't do anything with the number because it wasn't asked
        to." So a band is not even parsed on a batch that asks for no judging -
        a malformed one riding along is not a refusal, because nothing was
        going to read it."""
        by_i = self._send(self.ITEMS, band={'paceCharsPerSec': 'nonsense'})
        for i in (0, 1):
            self.assertIn('data', by_i[i], by_i[i])
            self.assertNotIn('guard', by_i[i], by_i[i])

    def test_a_retake_that_is_not_a_BOOLEAN_is_refused_BY_NAME(self):
        """Never coerced: `bool("false")` is True, so a client that spelled the
        flag wrong would get the exact opposite of what it asked for on every
        row of a book, silently, because both answers are ordinary audio."""
        for value in ('true', 'false', 1, 0):
            with self.subTest(retake=value):
                by_i = self._send(self.ITEMS, retake=value)
                self.assertTrue(
                    by_i[0]['message'].startswith('retake_malformed'), by_i[0])

    def test_every_row_carries_capped_ON_BOTH_ARMS(self):
        """THE MEASUREMENT IS UNCONDITIONAL. Whether a generation ran out of
        frame budget is a fact about the render, not a verdict on it, and the
        bare arm is exactly the one whose numbers nobody else has. Crucible
        reads a missing `capped` as "narrator did not say" and never as
        `false`, so it must be there whenever narrator does know."""
        for extra in ({}, {'retake': True, 'band': fake_higgs_band()}):
            with self.subTest(extra=extra):
                by_i = self._send(self.ITEMS, **extra)
                for i in (0, 1):
                    self.assertIn('capped', by_i[i], by_i[i])
                    self.assertIs(by_i[i]['capped'], False, by_i[i])

    def test_a_run_to_the_CEILING_is_reported_as_capped(self):
        """`cap_frames` is 2x the expected duration plus 150 frames of slack,
        so a take that reaches it did not stop because the model was finished.
        The fake's cap is the REAL formula over the real text
        (`FakeHiggsEngine.cap_frames`); only the audio is bent."""
        self._with_rate(json.dumps({'1': 40.0}))
        by_i = self._send(self.ITEMS)
        self.assertIs(by_i[0]['capped'], False, by_i[0])
        self.assertIs(by_i[1]['capped'], True, by_i[1])

    def test_an_engine_with_no_ladder_refuses_a_retake_BY_NAME(self):
        """Orpheus has no `render_many`. Rendering the batch unguarded would be
        a render that silently skipped the retake it was asked for and reported
        success."""
        self.w.close()
        self.w = Worker()                     # the Orpheus fake, the default
        self._prepared = True                 # this test does its own handshake
        self._ready()
        self._load('leah')
        self.w.send(action='generate_batch',
                    items=[{'i': 0, 'text': 'An ordinary sentence.'}],
                    retake=True, band=fake_higgs_band())
        by_i = self._assert_batch_closed(self.w.read_until('batch_done'), [0])
        self.assertTrue(by_i[0]['message'].startswith('retake_unsupported'),
                        by_i[0])

    def test_a_STREAMING_batch_cannot_ask_to_be_judged(self):
        """A streamed row leaves as it generates, so there is nothing left to
        re-roll by the time the ladder could decide. Refused rather than
        quietly rendered unjudged."""
        by_i = self._send([{'i': 0, 'text': 'A streamed sentence.', 'stream': True}],
                          retake=True, band=fake_higgs_band())
        self.assertTrue(by_i[0]['message'].startswith('retake_with_stream'),
                        by_i[0])


class BatchWidthTest(_WorkerCase):
    """`width`: how many rows of THIS batch narrator may keep in flight.

    MEASURED, 2026-09-19: SGLang started 16 wide at `mem_fraction_static` 0.60
    summed to 24.2 GB on a 24 GB card and WDDM paged the excess to host RAM -
    4 to 10 times slower, with no error and no log line. The fine-tuning ladder
    renders 4 wide on voices whose manifest says 16, and restarting a server to
    say so is not something a job can do. So a batch may narrow itself, and
    narrator does not reconfigure anything to honour it.
    """

    WORKER_ENV = {'NARRATOR_ENGINE': 'higgs-v3'}

    ITEMS = [{'i': i, 'text': f'Sentence number {i} of a chapter.'}
             for i in range(3)]

    def _send(self, **extra):
        if not getattr(self, '_prepared', False):
            self._ready()
            self._load('deathstalker')
            self._prepared = True
        self.w.send(action='generate_batch', items=self.ITEMS, **extra)
        return self._assert_batch_closed(self.w.read_until('batch_done'),
                                         [it['i'] for it in self.ITEMS])

    def test_a_width_inside_the_serving_width_reaches_the_driver(self):
        by_i = self._send(retake=True, band=fake_higgs_band(), width=1)
        for it in self.ITEMS:
            self.assertIn('data', by_i[it['i']], by_i[it['i']])

    def test_a_width_ABOVE_the_serving_width_is_refused_BY_NAME(self):
        """Refused, never clamped: a caller that believes it is running 16 wide
        while narrator quietly runs 4 reports a number nobody rendered at."""
        by_i = self._send(retake=True, band=fake_higgs_band(), width=999)
        for it in self.ITEMS:
            message = by_i[it['i']]['message']
            self.assertTrue(message.startswith('width_over_serving'), message)

    def test_a_malformed_width_is_refused_BY_NAME(self):
        for value in (0, -1, 2.0, True, '2'):
            with self.subTest(width=value):
                by_i = self._send(retake=True, band=fake_higgs_band(), width=value)
                self.assertTrue(
                    by_i[0]['message'].startswith('width_malformed'), by_i[0])

class VoiceCapsResetTest(unittest.TestCase):
    """A reload with an EMPTY caps payload must CLEAR the previous one.

    Driven IN-PROCESS rather than over the pipe, because what is being asserted
    is the state of a CLASS-LEVEL registry that outlives the engine - something
    the protocol deliberately never exposes. The worker object, its load path and
    its teardown are the real ones; only the engine is the fake.

    THE BUG THIS GUARDS (adversarial review, 2026-09-04). e2a's worker called
    `register_voice_caps(v, caps or {})` unconditionally; the first port
    registered only `if config.caps`. `_voice_caps` is class-level and survives
    `_teardown_engine`, and orpheus-worker-pool.ts sends `caps: {}` whenever a
    voice resolves to no catalog model - so: load a voice WITH catalog tuning,
    change modelDir (teardown + rebuild), reload the SAME voice with {}, and the
    first payload's eosFloor/eosBoost would still be attached to it. A whole
    session in the wrong tuning, reported as success.
    """

    def setUp(self):
        from crucible_higgs.serve import worker as W
        from crucible_higgs.serve.fake_engine import FakeEngine
        self.W, self.FakeEngine = W, FakeEngine
        self._saved_flag = W._FAKE_ENGINE
        W._FAKE_ENGINE = True          # the module global main() writes
        FakeEngine._voice_caps = {}
        self.addCleanup(self._restore)
        self.server = W.OrpheusStreamServer()

    def _restore(self):
        self.W._FAKE_ENGINE = self._saved_flag
        self.FakeEngine._voice_caps = {}

    def test_reload_with_empty_caps_clears_the_previous_payload(self):
        tuned = {'maxCharsPerSec': 22.6, 'eosBoost': 8, 'eosBoostStart': 2}
        self.assertTrue(self.server.load_voice('leah', caps=tuned, warm=False))
        self.assertEqual(self.FakeEngine._voice_caps['leah'],
                         {'maxCharsPerSec': 22.6, 'eosBoost': 8.0, 'eosBoostStart': 2.0})

        # A modelDir change is what forces the teardown; the pool then re-loads
        # the same token with whatever the catalog now says, which for an
        # unmodelled voice is {}.
        self.server._teardown_engine()
        self.assertEqual(self.FakeEngine._voice_caps['leah'],
                         {'maxCharsPerSec': 22.6, 'eosBoost': 8.0, 'eosBoostStart': 2.0},
                         'teardown does not (and must not) touch the class registry - '
                         'which is exactly why the reload has to clear it')

        self.assertTrue(self.server.load_voice('leah', caps={}, warm=False))
        self.assertEqual(self.FakeEngine._voice_caps['leah'], {},
                         'a reload with no catalog tuning must reset the voice to '
                         'env/class defaults, not inherit the previous payload')

    def test_a_warm_switch_back_also_clears(self):
        """The other branch: same weights, so no teardown - `set_voice` plus
        `_apply_voice_caps`. e2a always passed `caps or {}` there too."""
        self.assertTrue(self.server.load_voice('leah', caps={'eosBoost': 8}, warm=False))
        self.assertEqual(self.FakeEngine._voice_caps['leah'], {'eosBoost': 8.0})
        self.assertTrue(self.server.load_voice('leah', caps=None, warm=False))
        self.assertEqual(self.FakeEngine._voice_caps['leah'], {})

    def test_an_unknown_cap_key_fails_the_load(self):
        """Silently dropping it would mean a mis-tuned voice renders a whole book
        with no sign anything was wrong."""
        self.assertFalse(self.server.load_voice('leah', caps={'nonsense': 1},
                                                warm=False))


if __name__ == '__main__':
    unittest.main()
