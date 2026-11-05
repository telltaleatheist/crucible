"""The engine seam is a CONTRACT, asserted at runtime and not in a docstring.

`crucible_higgs.engine.protocol` is what the serve worker is written against. That
is only true if conformance is checked - a Protocol nobody isinstance()s is a
comment. Every type here is `runtime_checkable`, so this module asks the question
directly of both Higgs v3 arms.

What is asserted:
  1. HiggsV3Engine (served) and HiggsV3MlxEngine (in-process) satisfy `Engine`;
     their codecs satisfy `Codec` and the served budget satisfies `Budget`.
  2. The registry knows ONE id - higgs-v3 - and REFUSES any other by name,
     narrator's retired `orpheus` and `higgs-v2-scaffold` included.
  3. The facts the assembler and the worker read off an engine: `pads`,
     `edge_fade`, the codec's frame geometry, EOS reliability and the shape of
     `max_total_tokens`.
  4. No sampling lever of a retired engine is named in the protocol.
  5. Every engine writes an empty sentence's silence, at one length, through
     its own chunk writer; and the worker refuses an engine with no edge fade.

Moved from narrator@dc05e7e7 `tests/test_engine_protocol.py`. Dropped with the
engines they tested: every Orpheus and Higgs-v2-scaffold assertion, Orpheus's
chunk writer and budget, and the agreement test against
`narrator/assemble/engine_profiles.py` (assembly stays in BookForge; nothing here
holds a copy of that table).

An engine INSTANCE normally loads a model, which this interpreter cannot do. The
protocol checks therefore use an uninitialised instance (`object.__new__(cls)`)
with the two attributes `__init__` sets - `backend` and `voice` - assigned
directly. That is honest for what is being tested: isinstance against a
runtime_checkable Protocol is a hasattr sweep, so the value under test is the
CLASS's member set, and the two instance attributes are pinned separately against
`__init__`'s own source.
"""
import inspect
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_ROOT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "crucible", "jobs", "tts", "higgs")   # holds crucible_higgs
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)

from crucible_higgs.engine import registry                                  # noqa: E402
from crucible_higgs.engine.higgs import (HiggsV3Budget, HiggsV3Codec,       # noqa: E402
                                         HiggsV3Config, HiggsV3Engine,
                                         HiggsV3MlxCodec, HiggsV3MlxEngine,
                                         higgs_v3_stop_policy)
from crucible_higgs.engine.protocol import (BackendSpec, Budget, ClipsVoice,  # noqa: E402
                                            Codec, Engine, ReferenceClip,
                                            ServedBackend, SpeechRequest,
                                            TokenVoice)

# The members `Engine` names. Kept as a literal so a member SILENTLY dropped
# from the protocol is a failure here rather than an unnoticed relaxation.
#
# `_write_silence` IS ON THE LIST, underscore and all: it is what writes an empty
# sentence's file, and while the protocol declared it nowhere only Orpheus had
# one - an empty sentence reached a Higgs engine as an AttributeError. The
# leading underscore is history, not privacy (2026-09-13).
ENGINE_MEMBERS = ('ENGINE_ID', 'SAMPLE_RATE', 'pads', 'edge_fade', 'backend',
                  'voice', 'backend_spec', 'codec', 'budget', 'stop_policy',
                  'resolve_load_voice', 'convert', 'convert_batch',
                  '_write_silence', 'accept_item_sampling', 'accept_item_take',
                  'generate_batch_stream', 'cleanup')

ENGINES = (HiggsV3Engine, HiggsV3MlxEngine)


def _bare(cls, voice='deathstalker', backend='sglang-omni'):
    """An uninitialised engine carrying the two attributes __init__ sets."""
    engine = object.__new__(cls)
    engine.backend = backend
    engine.voice = voice
    return engine


def _v3_config(**kwargs):
    voice = ClipsVoice(
        clips=(ReferenceClip(path=__file__, transcript='A book-exact line.',
                             seconds=14.0),),
        name='deathstalker')
    return HiggsV3Config(voice=voice, stack='sglang-omni',
                         base_url='http://127.0.0.1:9', **kwargs)


class ProtocolConformanceTest(unittest.TestCase):

    def test_the_protocol_names_exactly_these_members(self):
        """Pins the surface itself: a member added to Engine without a decision
        shows up here, and one removed does too.

        DUNDERS are what is filtered out, not every underscore: `_write_silence`
        is a declared member (see ENGINE_MEMBERS).
        """
        declared = set(Engine.__annotations__)
        declared |= {name for name, value in vars(Engine).items()
                     if not name.startswith('__') and callable(value)}
        self.assertEqual(sorted(declared), sorted(ENGINE_MEMBERS))

    def test_both_v3_arms_satisfy_the_engine_protocol(self):
        self.assertIsInstance(_bare(HiggsV3Engine, backend='sglang-omni'), Engine)
        self.assertIsInstance(_bare(HiggsV3MlxEngine, backend='mlx'), Engine)

    def test_init_is_what_sets_backend_and_voice(self):
        """The two members the class body does not carry. Read off __init__'s
        source rather than assumed, so a refactor that stopped setting one would
        fail here instead of at the first isinstance in production."""
        for cls in ENGINES:
            source = inspect.getsource(cls.__init__)
            with self.subTest(engine=cls.__name__):
                self.assertIn('self.backend', source)
                self.assertIn('self.voice', source)

    def test_the_codecs_satisfy_the_codec_protocol(self):
        self.assertIsInstance(HiggsV3Codec(), Codec)
        self.assertIsInstance(HiggsV3MlxCodec(lambda codes: codes), Codec)

    def test_the_budget_satisfies_the_budget_protocol(self):
        self.assertIsInstance(HiggsV3Budget(_v3_config()), Budget)

    def test_the_v3_backend_satisfies_the_served_protocol(self):
        from crucible_higgs.engine.higgs.v3_served import HiggsV3ServedBackend
        backend = HiggsV3ServedBackend(base_url='http://127.0.0.1:8095')
        self.assertIsInstance(backend, ServedBackend)
        self.assertEqual(backend.spec.kind, 'served')

    def test_a_served_backend_with_neither_variable_runs_the_packaged_launcher(self):
        """Neither given is NOT a refusal: the worker runs ITS OWN packaged
        launcher, out of `crucible_higgs/engine/higgs/launch/`.

        That script is the definition of how a Higgs v3 server starts - the
        CUDA_HOME and FlashInfer workarounds, the per-stage memory fractions,
        the certified frames-7500 deploy profile - and it ships as package data
        of the `crucible` wheel. This is the proof it resolves to a real file,
        with its profile beside it.
        """
        from crucible_higgs.engine.higgs import v3_served
        from crucible_higgs.engine.higgs.v3_served import (BASE_URL_ENV,
                                                           LAUNCHER_PACKAGED,
                                                           SERVE_SCRIPT_ENV,
                                                           HiggsV3ServedBackend)
        for name in (BASE_URL_ENV, SERVE_SCRIPT_ENV):
            previous = os.environ.pop(name, None)
            if previous is not None:
                self.addCleanup(os.environ.__setitem__, name, previous)
        # A LAUNCHING backend states the server's admission width, and
        # `serve_concurrency()` refuses by name when it is unset - which is a
        # separate contract from the launcher and is asserted elsewhere.
        self._state('HIGGS_MAX_NUM_SEQS', '16')
        backend = HiggsV3ServedBackend()
        self.addCleanup(backend._close_log)
        self.assertEqual(backend.launcher_source, LAUNCHER_PACKAGED)
        self.assertEqual(backend.serve_script,
                         v3_served.packaged_serve_script())
        self.assertTrue(os.path.isfile(backend.serve_script),
                        backend.serve_script)
        self.assertEqual(v3_served.LAUNCH_PACKAGE, 'crucible_higgs.engine.higgs')
        # THE PROFILE TRAVELS WITH IT: the script reads it as its own sibling.
        self.assertTrue(
            os.path.isfile(os.path.join(
                os.path.dirname(backend.serve_script),
                v3_served.PACKAGED_DEPLOY_CONFIG)))

    def test_the_sglang_launcher_is_packaged_too(self):
        from crucible_higgs.engine.higgs import sgl_served
        script = sgl_served.packaged_serve_script()
        self.assertTrue(os.path.isfile(script), script)
        self.assertEqual(os.path.basename(script), sgl_served.PACKAGED_SERVE_SCRIPT)
        # The entry module the script execs, found at `$(dirname "$0")`.
        self.assertTrue(os.path.isfile(
            os.path.join(os.path.dirname(script), 'sgl_omni_entry.py')))

    def test_an_override_launcher_that_is_not_there_is_refused_by_name(self):
        """The override is not a suggestion. Substituting the packaged script
        for a path an operator named would start a server with different flags
        and report success."""
        from crucible_higgs.engine.higgs.v3_served import (BASE_URL_ENV,
                                                           SERVE_SCRIPT_ENV,
                                                           HiggsV3ServedBackend)
        previous = os.environ.pop(BASE_URL_ENV, None)
        if previous is not None:
            self.addCleanup(os.environ.__setitem__, BASE_URL_ENV, previous)
        missing = os.path.join(self.tempdir(), 'no-such-serve.sh')
        with self.assertRaises(ValueError) as caught:
            HiggsV3ServedBackend(serve_script=missing)
        self.assertIn(SERVE_SCRIPT_ENV, str(caught.exception))
        # The message carries the path as `{path!r}`.
        self.assertIn(repr(missing), str(caught.exception))

    def _state(self, name: str, value: str) -> None:
        previous = os.environ.get(name)
        os.environ[name] = value
        if previous is None:
            self.addCleanup(os.environ.pop, name, None)
        else:
            self.addCleanup(os.environ.__setitem__, name, previous)

    def tempdir(self) -> str:
        import shutil as _shutil
        import tempfile
        path = tempfile.mkdtemp(prefix='crucible-higgs-missing-')
        self.addCleanup(_shutil.rmtree, path, True)
        return path


class RegistryTest(unittest.TestCase):

    def test_the_known_ids(self):
        self.assertEqual(registry.ids(), ['higgs-v3'])

    def test_every_entry_is_a_pair_of_factories(self):
        """The registry must be READABLE - "which engines exist?" - on an
        interpreter with no backend installed, which is why its values are
        zero-import callables rather than classes (that importing them stays
        lazy is asserted in a fresh subprocess by test_engine_lazy_imports.py)."""
        for engine_id, entry in registry.ENGINES.items():
            with self.subTest(engine=engine_id):
                self.assertEqual(len(entry), 2)
                self.assertTrue(all(callable(f) for f in entry))

    def test_an_unknown_id_is_refused_by_name(self):
        for call in (lambda: registry.engine_class('llasa'),
                     lambda: registry.engine_config('llasa', voice='x')):
            with self.assertRaises(ValueError) as caught:
                call()
            message = str(caught.exception)
            self.assertIn('llasa', message)
            self.assertIn('higgs-v3', message)
            self.assertIn('Refusing to substitute', message)

    def test_the_retired_engines_are_unknown_ids_now(self):
        """narrator's `orpheus` (retired) and `higgs-v2-scaffold` (never
        shipped) did not move: asking for either is the same refusal as any
        other unknown id, never a substitution."""
        for engine_id in ('orpheus', 'higgs-v2-scaffold'):
            with self.subTest(engine=engine_id):
                with self.assertRaises(ValueError) as caught:
                    registry.engine_class(engine_id)
                self.assertIn(engine_id, str(caught.exception))

    def test_the_registry_resolves_the_arm_for_the_platform(self):
        """`higgs-v3` is ONE id with TWO backends, chosen by PLATFORM: on darwin
        the in-process MLX engine, everywhere else the served one."""
        expected = (HiggsV3MlxEngine if sys.platform.startswith('darwin')
                    else HiggsV3Engine)
        self.assertIs(registry.engine_class('higgs-v3'), expected)
        self.assertEqual(registry.higgs_v3_backend_for_platform('darwin'),
                         'inprocess-mlx')
        self.assertEqual(registry.higgs_v3_backend_for_platform('linux'), 'served')
        with self.assertRaises(ValueError):
            registry.higgs_v3_backend_for_platform('')

    def test_every_engine_reports_the_id_that_selected_it(self):
        """An engine whose ENGINE_ID disagrees with its registry key makes a
        log line name the wrong engine - on both arms."""
        for engine_id in registry.ids():
            with self.subTest(engine=engine_id):
                self.assertEqual(registry.engine_class(engine_id).ENGINE_ID,
                                 engine_id)
        for cls in ENGINES:
            self.assertEqual(cls.ENGINE_ID, 'higgs-v3')


class EngineFactsTest(unittest.TestCase):
    """The facts the assembler, the packer and the worker read off an engine
    instead of assuming."""

    def test_pads_and_edge_fade(self):
        for cls in ENGINES:
            with self.subTest(engine=cls.__name__):
                self.assertFalse(cls.pads,
                                 'Higgs emits bare speech; the assembler owns the gaps')
                self.assertEqual(cls.edge_fade.as_manifest(),
                                 {'in': 10.0, 'out': 25.0})

    def test_the_fade_is_asymmetric_and_that_is_the_point(self):
        """One float cannot say this. A chunk begins on an attack the ear
        expects and ends on a decay it does not, so the tail needs more than
        twice the head's window."""
        fade = HiggsV3Engine.edge_fade
        self.assertGreater(fade.out_ms, 2 * fade.in_ms)
        self.assertFalse(fade.is_none)

    def test_codec_geometry(self):
        for codec in (HiggsV3Codec(), HiggsV3MlxCodec(lambda codes: codes)):
            with self.subTest(codec=type(codec).__name__):
                self.assertEqual((codec.tokens_per_frame, codec.samples_per_frame,
                                  codec.trim_frames), (8, 960, 7))
                self.assertEqual(codec.sample_rate, 24000)
                self.assertEqual(codec.frames_per_second, 25.0)

    def test_the_served_codec_has_no_windowed_decoder_and_decodes_nothing(self):
        codec = HiggsV3Codec()
        self.assertIsNone(codec.streaming_decoder(lambda a, b: None),
                          'a delay-pattern codec has no sound windowed decode; '
                          'returning None is the statement that says so')
        with self.assertRaises(NotImplementedError):
            codec.decode([])

    def test_max_total_tokens_is_a_ceiling_the_prompt_eats_into(self):
        budget = HiggsV3Budget(_v3_config())
        window = _v3_config().context_tokens
        self.assertEqual(budget.max_total_tokens(0), window)
        self.assertEqual(budget.max_total_tokens(1632), window)
        with self.assertRaises(ValueError) as caught:
            budget.max_total_tokens(window)
        self.assertIn('nothing to generate', str(caught.exception))

    def test_the_stop_policy_trusts_eos_and_checks_coverage(self):
        policy = higgs_v3_stop_policy(_v3_config())
        self.assertTrue(policy.eos_reliable)
        self.assertFalse(policy.resplit_on_cap)
        self.assertEqual(policy.coverage_check, 'asr',
                         'duration is not a coverage proxy on this family')


class RetiredLeversStayOutOfTheProtocolTest(unittest.TestCase):

    def test_no_orpheus_lever_is_named_in_the_protocol(self):
        """StopPolicy exists so a scheduler never has to know what an eosFloor
        is; the retired engine's levers must not have leaked into the seam."""
        import crucible_higgs.engine.protocol as protocol
        source = inspect.getsource(protocol)
        for lever in ('eosBoost', 'eosFloor', 'eosBoostStart', 'eosFloorRate',
                      'repPenalty', 'minP'):
            with self.subTest(lever=lever):
                self.assertNotIn(f"'{lever}'", source)
                self.assertNotIn(f'"{lever}"', source)


class StdoutIsTheProtocolTest(unittest.TestCase):
    """No engine may print to STDOUT.

    `crucible_higgs.serve` writes one JSON object per line to stdout and Crucible
    parses every one of them; a bare `[HIGGS3] launching: ...` line there is a
    protocol error that kills the session. It is an easy mistake - a
    `print(..., flush=True)` looks like logging.
    """

    def _stdout_prints(self, path):
        import ast
        tree = ast.parse(open(path, encoding='utf-8').read())
        return [node.lineno for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and getattr(node.func, 'id', None) == 'print'
                and not any(kw.arg == 'file' for kw in node.keywords)]

    def test_the_higgs_engines_log_to_stderr(self):
        import crucible_higgs.engine.higgs as higgs
        directory = os.path.dirname(higgs.__file__)
        for name in sorted(os.listdir(directory)):
            if not name.endswith('.py'):
                continue
            path = os.path.join(directory, name)
            with self.subTest(module=name):
                self.assertEqual(
                    self._stdout_prints(path), [],
                    f'{name} prints to stdout, which is the JSON protocol')

    def test_the_fake_engine_too(self):
        from crucible_higgs.serve import fake_engine
        self.assertEqual(self._stdout_prints(fake_engine.__file__), [])

    def test_nothing_in_the_engine_calls_torchaudio_save(self):
        """torchaudio is not on the write path: every chunk goes through
        soundfile at PCM_16. Asserted over the AST."""
        import ast
        import crucible_higgs.engine.higgs as higgs
        directory = os.path.dirname(higgs.__file__)
        for name in sorted(os.listdir(directory)):
            if not name.endswith('.py'):
                continue
            tree = ast.parse(open(os.path.join(directory, name),
                                  encoding='utf-8').read())
            calls = [node.lineno for node in ast.walk(tree)
                     if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute)
                     and node.func.attr == 'save'
                     and getattr(node.func.value, 'id', None) == 'torchaudio']
            with self.subTest(module=name):
                self.assertEqual(calls, [], f'{name} writes with torchaudio')


class EmptySentenceSilenceTest(unittest.TestCase):
    """`_write_silence`, on EVERY engine.

    THE DEFECT THIS WOULD HAVE CAUGHT (found 2026-09-13): the protocol declared
    the member nowhere, so only Orpheus had one, and an empty sentence reached a
    Higgs engine as an AttributeError that aborted the take at that chunk. Two
    things are asserted, because either alone would have let it through: the
    protocol DECLARES the member (`ENGINE_MEMBERS`, and the isinstance checks
    above), and every engine's implementation actually writes the file.

    narrator drove this through `render/worker.py`'s
    `_write_empty_sentence_silence`, which stayed in BookForge; it is a one-line
    call of the member, so the member is called here directly.
    """

    #: narrator's `render/worker.py` RESUME_MIN_BYTES: a file at or under it is
    #: re-listed by every resume scan, so a blank sentence never looks rendered.
    RESUME_MIN_BYTES = 1024

    ENGINES = ('higgs-v3-served', 'higgs-v3-mlx')

    def _engine(self, which, directory):
        """A bare engine with only what the silence writer reads."""
        from types import SimpleNamespace
        config = SimpleNamespace(sentences_dir=directory, audio_format='flac')
        if which == 'higgs-v3-served':
            engine = _bare(HiggsV3Engine, backend='sglang-omni')
        elif which == 'higgs-v3-mlx':
            engine = _bare(HiggsV3MlxEngine, backend='mlx')
        else:
            raise AssertionError(f'no fixture for engine {which!r}')
        engine.config = config
        return engine

    def test_every_engine_writes_the_empty_sentence(self):
        import shutil
        import tempfile

        import soundfile as sf
        for which in self.ENGINES:
            directory = tempfile.mkdtemp(prefix='crucible-higgs-silence-')
            self.addCleanup(shutil.rmtree, directory, True)
            with self.subTest(engine=which):
                self._engine(which, directory)._write_silence(7)
                path = os.path.join(directory, '7.flac')
                self.assertTrue(os.path.isfile(path),
                                'the index has no file, so assembly has a hole')
                info = sf.info(path)
                # THE SAME WRITER CONTRACT as a rendered chunk beside it: mixed
                # bit depths in one session are what ffmpeg's concat demuxer
                # drops frames on, silently.
                self.assertEqual(info.subtype, 'PCM_16')
                self.assertEqual(info.format, 'FLAC')
                self.assertEqual(info.channels, 1)
                self.assertEqual(info.samplerate, 24000)

    def test_the_length_is_ONE_number_for_every_engine(self):
        import shutil
        import tempfile

        import soundfile as sf

        from crucible_higgs.engine.protocol import EMPTY_SENTENCE_SILENCE_SEC
        self.assertEqual(EMPTY_SENTENCE_SILENCE_SEC, 0.1)
        frames = set()
        for which in self.ENGINES:
            directory = tempfile.mkdtemp(prefix='crucible-higgs-silence-')
            self.addCleanup(shutil.rmtree, directory, True)
            self._engine(which, directory)._write_silence(0)
            frames.add(sf.info(os.path.join(directory, '0.flac')).frames)
        self.assertEqual(frames, {int(24000 * EMPTY_SENTENCE_SILENCE_SEC)})

    def test_it_is_silence_and_it_stays_under_the_resume_floor(self):
        import shutil
        import tempfile

        import numpy as np
        import soundfile as sf
        for which in self.ENGINES:
            directory = tempfile.mkdtemp(prefix='crucible-higgs-silence-')
            self.addCleanup(shutil.rmtree, directory, True)
            with self.subTest(engine=which):
                self._engine(which, directory)._write_silence(3)
                path = os.path.join(directory, '3.flac')
                audio, _rate = sf.read(path, dtype='float32')
                self.assertEqual(np.max(np.abs(audio)), 0.0)
                self.assertLessEqual(os.path.getsize(path), self.RESUME_MIN_BYTES)


class EdgeFadeIsRequiredTest(unittest.TestCase):
    """`_edge_fade_of` must RAISE, not assume EdgeFade(0, 0).

    EdgeFade(0, 0) is "this engine needs no fade", which is false of every codec
    that emits bare edges. A `pads = False` engine that forgot to declare one
    would have reported it and the assembler would have joined its chunks bare -
    a book that clicks at every join, shipped as a success.
    """

    def test_an_engine_without_edge_fade_is_refused_by_name(self):
        from crucible_higgs.serve import worker as W

        class Nameless:
            ENGINE_ID = 'forgot-to-declare'
            pads = False

        with self.assertRaises(RuntimeError) as caught:
            W._edge_fade_of(Nameless())
        message = str(caught.exception)
        self.assertIn('forgot-to-declare', message)
        self.assertIn('edge_fade', message)
        self.assertIn('click', message)

    def test_both_arms_and_the_fake_declare_one(self):
        from crucible_higgs.serve import worker as W
        from crucible_higgs.serve.fake_engine import FakeHiggsEngine
        for engine in (*ENGINES, FakeHiggsEngine):
            with self.subTest(engine=engine.__name__):
                self.assertEqual(W._edge_fade_of(engine), engine.edge_fade)


class BackendSpecTest(unittest.TestCase):

    def test_a_served_spec_must_carry_a_url(self):
        with self.assertRaises(ValueError) as caught:
            BackendSpec(kind='served', name='vllm-omni')
        self.assertIn('base_url', str(caught.exception))

    def test_an_inprocess_spec_must_not(self):
        with self.assertRaises(ValueError):
            BackendSpec(kind='inprocess', name='mlx', base_url='http://x')

    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            BackendSpec(kind='remote', name='x')

    def test_speech_request_defaults_to_the_servers_own_sampling(self):
        """An EMPTY sampling map means "server defaults" - which is what the
        delivered v3 render used, and what a client must not quietly replace."""
        request = SpeechRequest(text='hi', voice=TokenVoice('deathstalker'),
                                max_new_tokens=100)
        self.assertEqual(dict(request.sampling), {})


if __name__ == '__main__':
    unittest.main()
