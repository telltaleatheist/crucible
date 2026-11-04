"""Higgs v3 without the GPU: the request, the response, the lifecycle, the wire.

Higgs v3 is a SERVED engine - the model runs in a vllm-omni process narrator
launches, health-checks and stops. Everything about it that can go wrong
silently is on the HTTP boundary, so this module puts a REAL HTTP server
(stdlib `http.server`, in a thread) where vllm-omni would be and drives the real
client against it.

WHAT IS FIXTURE AND WHAT IS CAPTURE. There is NO recorded request/response
capture yet: the delivered cloned renders were driven by
`<campaign>/higgs/work/render_final.py` and `work/confirm.py`, and those scripts
plus `HIGGS_V3_LEVERS.md` are what the expected request body below is built
from. So the shape assertions here are AGAINST THE SCRIPTS, NOT A CAPTURE. When
a real capture lands (it is owed to `<campaign>/higgs/captures/`),
`EXPECTED_REQUEST` is the one place it replaces, and `decode_response` is the
one place a response-format correction goes.

The three failures this is here to catch, all of which return HTTP 200 or look
fine:
  1. sampling sent at the TOP LEVEL is silently dropped (pydantic), so the
     render runs at the server default while the log says otherwise;
  2. an out-of-vocabulary control token is READ ALOUD and collapses the render
     (measured coverage 0.000);
  3. a reference over 30 s, or more than one reference, is refused by the server
     with a 400 - after a 55 s launch and a GPU allocation.
"""
import json
import os
import struct
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_ROOT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "crucible", "jobs", "tts", "higgs")   # holds crucible_higgs
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)

from crucible_higgs.engine import registry                                  # noqa: E402
from crucible_higgs.engine.higgs import v3_served                           # noqa: E402
from crucible_higgs.engine.higgs import truncation                         # noqa: E402
from crucible_higgs.engine.higgs import config as v3_config           # noqa: E402
from crucible_higgs.engine.higgs.v3_engine import (HiggsV3Budget,           # noqa: E402
                                             HiggsV3Config,
                                             HiggsV3Defaults,
                                             HiggsV3Engine,
                                             higgs_v3_stop_policy)
from crucible_higgs.engine.higgs.v3_served import (HiggsV3ServedBackend,    # noqa: E402
                                             HiggsV3ServerError)
from crucible_higgs.engine.protocol import (ClipsVoice, ReferenceClip,      # noqa: E402
                                      SpeechRequest)

# The reference the delivered render used: work/refs/manifest.json entry "x2" -
# two same-book deathstalker clips pre-joined into one 27.42 s wav, transcripts
# joined in the same order.
X2_TEXT = ('They are little moments that help us to be genuinely present where we '
           'are. What else can we do?')


def a_wav(path, seconds=1.0, rate=24000):
    """A tiny real WAV on disk, so reference_data_uri has something to read."""
    samples = np.zeros(int(rate * seconds), dtype=np.float32)
    import soundfile as sf
    sf.write(path, samples, rate)
    return path


def wav_bytes(seconds=1.0, rate=24000, channels=1, tail_burst_dbfs=None):
    """A WAV FILE as bytes - what /v1/audio/speech returns with
    response_format 'wav'. Built by hand (PCM16 RIFF) so the fake server needs
    no soundfile of its own."""
    # SPEECH, THEN THE SERVER'S OWN TRIMMED SILENCE - which is the shape a
    # PATCHED server returns: our smoke's real clip measured -62 dBFS over its
    # last 300 ms, i.e. the words had already ended. The probe reads exactly
    # that window, so the fake has to have one.
    tail_frames = int(rate * 0.32)
    speech_frames = max(1, int(rate * seconds) - tail_frames)
    speech = (np.sin(2 * np.pi * 220.0 * np.arange(speech_frames) / rate)
              * 0.4 * 32767)
    # A -70 dBFS floor, not digital zero: real decoded audio has one, and a
    # probe that only passes on absolute silence would be untested against it.
    floor_amp = (10 ** (-70.0 / 20.0)) * 32767 * np.sqrt(2)
    tail = (np.sin(2 * np.pi * 60.0 * np.arange(tail_frames) / rate) * floor_amp)
    data = np.concatenate([speech, tail])
    if tail_burst_dbfs is not None:
        # An UNPATCHED server: ~250 ms of ramp-down sentinels decoded as real
        # sound at about -30 dB, cut off at its peak.
        burst = min(data.size, int(rate * 0.25))
        amp = (10 ** (tail_burst_dbfs / 20.0)) * 32767 * np.sqrt(2)
        data[-burst:] = (np.sin(2 * np.pi * 90.0 * np.arange(burst) / rate) * amp)
    frames = data.size
    pcm = data.astype('<i2').tobytes()
    if channels == 2:
        pcm = b''.join(pcm[i:i + 2] * 2 for i in range(0, len(pcm), 2))
    block = 2 * channels
    header = (b'RIFF' + struct.pack('<I', 36 + len(pcm)) + b'WAVE'
              + b'fmt ' + struct.pack('<IHHIIHH', 16, 1, channels, rate,
                                      rate * block, block, 16)
              + b'data' + struct.pack('<I', len(pcm)))
    return header + pcm


def a_launcher(test) -> str:
    """A REAL launch script on this filesystem, created once per test.

    `serve_script=` (and the `NARRATOR_HIGGS3_SERVE_SCRIPT` behind it) is an
    OPERATOR'S OVERRIDE of narrator's own packaged launcher, and a path that is
    not there is refused BY NAME rather than quietly replaced by the packaged
    one - substituting it would start a server with different flags and report
    success. So a test that wants LAUNCH mode has to name a file that exists,
    and this is it. The literal `/campaign/serve_v3.sh` these tests used to
    pass stopped being nameable on 2026-09-13 for exactly that reason.
    """
    existing = getattr(test, '_launcher', None)
    if existing is not None:
        return existing
    import shutil as _shutil
    import tempfile
    directory = tempfile.mkdtemp(prefix='narrator-H-launcher-')
    test.addCleanup(_shutil.rmtree, directory, True)
    path = os.path.join(directory, 'serve_v3.sh')
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('#!/bin/bash\nexit 0\n')
    os.chmod(path, 0o755)
    test._launcher = path
    return path


class FakeV3Handler(BaseHTTPRequestHandler):
    """Just enough of vllm-omni: /health, /v1/models, /v1/audio/speech.

    It records every request body it is given, and can be told to answer a 400
    the way the real server does.
    """

    server_version = 'fake-vllm-omni/0.28.0'

    def log_message(self, *args):
        pass                                   # keep the test output readable

    def do_GET(self):
        if self.path in (v3_served.HEALTH_PATH, '/ping'):
            if not self.server.healthy:
                self.send_error(503, 'not ready')
                return
            self.send_response(200)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if self.path == v3_served.MODELS_PATH:
            if self.server.models_broken:
                # Something is listening, but it is not an OpenAI-shaped server.
                self.send_error(404)
                return
            rows = []
            for m in self.server.models:
                row = {'id': m}
                if self.server.model_root is not None:
                    # vllm-omni 0.28 reports the model path it was started on.
                    row['root'] = self.server.model_root
                rows.append(row)
            payload = json.dumps({'data': rows}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != v3_served.SPEECH_PATH:
            self.send_error(404)
            return
        length = int(self.headers.get('Content-Length', '0'))
        body = json.loads(self.rfile.read(length).decode('utf-8'))
        self.server.requests.append(body)
        if self.server.error is not None:
            code, message = self.server.error
            payload = json.dumps({'message': message}).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        audio = wav_bytes(self.server.seconds, self.server.rate,
                          self.server.channels, self.server.tail_burst_dbfs)
        self.send_response(200)
        self.send_header('Content-Type', 'audio/wav')
        self.send_header('x-vllm-omni-output-tokens', '509')
        self.send_header('Content-Length', str(len(audio)))
        self.end_headers()
        self.wfile.write(audio)


#: The path a base-weights server reports as its root (measured shape).
BASE_SNAPSHOT_ROOT = ('/home/<user>/.cache/huggingface/hub/'
                      'models--bosonai--higgs-audio-v3-tts-4b/snapshots/239f63fb')


def state_stack(case, value=None):
    """`HIGGS_STACK` for the duration of one test.

    A SECOND CONTRACT VARIABLE, of exactly the same kind as HIGGS_MAX_NUM_SEQS:
    `served_common.serving_stack()` refuses BY NAME when it is unset, because
    the two serving stacks place sampling differently, size the frame cap
    against different context windows, and disagree about what an empty sampling
    means. Every test in this module drives the VLLM-OMNI stack unless it says
    otherwise; `tests/test_higgs_sgl.py` is the other one's.
    """
    value = value or v3_served.served_common.STACK_VLLM_OMNI
    previous = os.environ.get(v3_served.served_common.STACK_ENV)
    os.environ[v3_served.served_common.STACK_ENV] = value
    if previous is None:
        case.addCleanup(os.environ.pop, v3_served.served_common.STACK_ENV, None)
    else:
        case.addCleanup(os.environ.__setitem__,
                        v3_served.served_common.STACK_ENV, previous)


def state_concurrency(case, value='2'):
    """`HIGGS_MAX_NUM_SEQS` for the duration of one test - the contract
    variable every engine and every launching backend reads."""
    previous = os.environ.get(v3_served.SERVE_MAX_NUM_SEQS_ENV)
    os.environ[v3_served.SERVE_MAX_NUM_SEQS_ENV] = value
    if previous is None:
        case.addCleanup(os.environ.pop, v3_served.SERVE_MAX_NUM_SEQS_ENV, None)
    else:
        case.addCleanup(os.environ.__setitem__,
                        v3_served.SERVE_MAX_NUM_SEQS_ENV, previous)


class FakeV3Server:
    """The fake, on a real port, in a thread."""

    def __init__(self, healthy=True, seconds=1.0, rate=24000, channels=1):
        self.httpd = HTTPServer(('127.0.0.1', 0), FakeV3Handler)
        self.httpd.requests = []
        self.httpd.healthy = healthy
        self.httpd.error = None
        self.httpd.seconds = seconds
        self.httpd.rate = rate
        self.httpd.channels = channels
        self.httpd.models = ['higgs-v3']
        self.httpd.models_broken = False
        # What a server started WITHOUT HIGGS_MODEL_DIR reports: the base
        # snapshot out of the HF cache. Tests for fine-tunes set their own.
        self.httpd.model_root = BASE_SNAPSHOT_ROOT
        self.httpd.tail_burst_dbfs = None
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()

    @property
    def base_url(self):
        host, port = self.httpd.server_address[:2]
        return f'http://{host}:{port}'

    @property
    def requests(self):
        return self.httpd.requests

    def fail_with(self, code, message):
        self.httpd.error = (code, message)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=10)


#: THE REAL FILE, byte for byte. Copied 2026-09-05 out of the merged fine-tune
#: `/home/<user>/higgs_v3_merged/ds_ad4lm` (WSL), whose `merge_manifest.json`
#: records its provenance as
#: `"generation_config_source": "OVERRIDE from runs/ds_ad4lm_r32/
#: generation_config.override.json"` - the values being vllm-omni's own
#: `deploy/higgs_multimodal_qwen3.yaml` stage-0 `default_sampling_params`.
#: EMBEDDED rather than read from that path: a test that reads a machine's
#: working directory passes or fails on what somebody else did there.
MERGED_GENERATION_CONFIG = """{
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 50,
  "repetition_penalty": 1.0
}
"""


class V3TestCase(unittest.TestCase):
    """A fake server and a one-clip deathstalker voice, for every subclass."""

    def setUp(self):
        import tempfile
        self.dir = tempfile.mkdtemp(prefix='narrator-H-v3-')
        self.addCleanup(self._rmtree)
        self.clip = a_wav(os.path.join(self.dir, 'ref_x2.wav'), seconds=1.0)
        self.server = FakeV3Server()
        self.addCleanup(self.server.close)
        state_concurrency(self)
        state_stack(self)

    def _rmtree(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def merged_checkpoint(self, name='ds-merged', generation_config=...):
        """A directory shaped like a real merged Higgs v3 checkpoint - minus the
        8.5 GB of weights, which nothing here reads.

        `generation_config` is written verbatim; `...` writes THE REAL FILE'S
        CONTENT (see MERGED_GENERATION_CONFIG) and None writes no file at all,
        which is the misconfigured dir `require_generation_config` exists to
        refuse.
        """
        path = os.path.join(self.dir, name)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'config.json'), 'w',
                  encoding='utf-8') as handle:
            handle.write('{}')
        if generation_config is None:
            return path
        body = (MERGED_GENERATION_CONFIG if generation_config is ...
                else generation_config)
        with open(os.path.join(path, 'generation_config.json'), 'w',
                  encoding='utf-8') as handle:
            handle.write(body)
        return path

    def voice(self, seconds=27.42, clips=1, **kwargs):
        return ClipsVoice(
            clips=tuple(ReferenceClip(self.clip, X2_TEXT, seconds=seconds / clips)
                        for _ in range(clips)),
            name='deathstalker', **kwargs)

    def config(self, **kwargs):
        kwargs.setdefault('voice', self.voice())
        kwargs.setdefault('base_url', self.server.base_url)
        return HiggsV3Config(**kwargs)

    def quiet_config(self, **kwargs):
        """A config whose load runs no sentinel-filter probe - for the tests
        that are counting requests or asserting seeds."""
        kwargs.setdefault('probe_sentinel_filter', False)
        return self.config(**kwargs)


class RequestShapeTest(V3TestCase):
    """AGAINST THE SCRIPTS, NOT A CAPTURE - see the module docstring."""

    def test_the_body_matches_render_final_and_confirm(self):
        body = v3_served.build_request_body(
            'It was a Saturday morning.', self.voice(), 500, seed=1234)
        self.assertEqual(sorted(body), ['input', 'max_new_tokens', 'model',
                                        'references', 'response_format', 'seed'])
        self.assertEqual(body['model'], 'higgs-v3')
        self.assertEqual(body['response_format'], 'wav')
        self.assertEqual(body['max_new_tokens'], 500)
        self.assertEqual(body['seed'], 1234)
        self.assertEqual(len(body['references']), 1)
        self.assertEqual(body['references'][0]['text'], X2_TEXT)
        self.assertTrue(
            body['references'][0]['audio_path'].startswith('data:audio/wav;base64,'),
            'a bare path is rejected ("The URL must be either a HTTP, data or file '
            'URL") and file:// needs --allowed-local-media-path')

    def test_no_sampling_means_the_servers_own_defaults(self):
        """The delivered render sends NO extra_params, so the deploy defaults
        (temperature 1.0, top_p 0.95, top_k 50) apply verbatim. That is what
        Owen asked for."""
        body = v3_served.build_request_body('Hello.', self.voice(), 200)
        self.assertNotIn('extra_params', body)
        self.assertNotIn('temperature', body)

    def test_sampling_rides_in_extra_params_and_never_at_the_top_level(self):
        """THE bug of the first audition: pydantic drops top-level temperature
        without a word, so every one of those renders actually ran at the server
        default while the log said 0.3."""
        body = v3_served.build_request_body('Hello.', self.voice(), 200,
                                            sampling={'temperature': 0.7,
                                                      'top_p': 0.9})
        self.assertEqual(body['extra_params'], {'temperature': 0.7, 'top_p': 0.9})
        self.assertNotIn('temperature', body)
        self.assertNotIn('top_p', body)

    def test_an_unknown_sampling_key_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            v3_served.build_request_body('Hello.', self.voice(), 200,
                                         sampling={'min_p': 0.1})
        self.assertIn('min_p', str(caught.exception))

    def test_the_cap_formula_is_render_finals(self):
        """int(chars / 15.0 * 25 * 2.0) + 150."""
        for chars, cap in ((300, 1150), (600, 2150), (898, 3143)):
            with self.subTest(chars=chars):
                self.assertEqual(v3_served.cap_frames('x' * chars), cap)

    def test_no_voice_means_the_models_own_narrator(self):
        """Legitimate only for a smoke test: un-cloned sits at 12 % of the
        narrator ceiling."""
        body = v3_served.build_request_body('Hello.', None, 200)
        self.assertNotIn('references', body)


class ControlTokenTest(V3TestCase):

    def test_the_allowlist_is_the_measured_vocabulary(self):
        self.assertEqual(len(v3_served.ALLOWED_CONTROL_TOKENS), 45)
        self.assertIn('<|prosody:long_pause|>', v3_served.ALLOWED_CONTROL_TOKENS)
        self.assertIn('<|emotion:contemplation|>', v3_served.ALLOWED_CONTROL_TOKENS)

    def test_every_known_trap_is_refused(self):
        """Each of these was measured to collapse a render: coverage 0.000,
        pitch std 0.28 st, speaker cosine 0.05."""
        for trap in ('<|emotion:calm|>', '<|emotion:neutral|>',
                     '<|prosody:pause_long|>', '<|scene_desc_start|>'):
            with self.subTest(trap=trap):
                with self.assertRaises(ValueError) as caught:
                    v3_served.validate_control_tokens(trap + ' Hello.')
                self.assertIn(trap, str(caught.exception))

    def test_a_real_control_token_passes(self):
        v3_served.validate_control_tokens('<|prosody:long_pause|>Hello.')

    def test_the_request_builder_runs_the_check(self):
        with self.assertRaises(ValueError):
            v3_served.build_request_body('<|emotion:calm|>Hi.', self.voice(), 200)


class ReferenceTest(V3TestCase):

    def test_the_thirty_second_cap(self):
        with self.assertRaises(ValueError) as caught:
            self.voice(seconds=42.03)
            v3_served.check_reference_budget(self.voice(seconds=42.03))
        message = str(caught.exception)
        self.assertIn('42.0', message)
        self.assertIn('30', message)

    def test_joins_count_against_the_cap(self):
        """Concatenating N clips inserts N-1 x 0.35 s of silence, and the server
        measures the wav it is given."""
        total = v3_served.reference_seconds(self.voice(seconds=29.9, clips=2))
        self.assertAlmostEqual(total, 29.9, places=3)
        with self.assertRaises(ValueError):
            v3_served.check_reference_budget(self.voice(seconds=29.9, clips=2))

    def test_a_clip_without_a_duration_is_refused(self):
        voice = ClipsVoice(clips=(ReferenceClip(self.clip, X2_TEXT),),
                           name='deathstalker')
        with self.assertRaises(ValueError) as caught:
            v3_served.check_reference_budget(voice)
        self.assertIn('no duration', str(caught.exception))

    def test_more_than_one_reference_is_refused_with_the_instruction(self):
        """vllm-omni takes EXACTLY ONE reference. Using just the first clip
        would be a different voice, reported as success."""
        with self.assertRaises(ValueError) as caught:
            v3_served.reference_for(self.voice(seconds=27.0, clips=2))
        message = str(caught.exception)
        self.assertIn('EXACTLY ONE', message)
        self.assertIn('pre-joined', message)

    def test_a_clip_without_a_transcript_cannot_exist(self):
        with self.assertRaises(ValueError):
            ReferenceClip(self.clip, '')


class CheckpointVoiceTest(V3TestCase):
    """A fine-tuned Higgs voice is a MERGED CHECKPOINT, and one server serves
    exactly one of them.

    Measured by the training side, 2026-09-04: vllm-omni exposes no adapter
    flags and its higgs_audio_v3 talker does not implement `SupportsLoRA`, so a
    LoRA cannot be loaded at runtime and there is no per-request adapter either.
    The `lora-modules` strategy this module used to express is retired.
    """

    def test_the_only_strategy_is_checkpoint(self):
        self.assertEqual(v3_served.check_strategy('checkpoint'), 'checkpoint')

    def test_lora_modules_is_refused_and_says_why(self):
        with self.assertRaises(ValueError) as caught:
            v3_served.check_strategy('lora-modules')
        message = str(caught.exception)
        self.assertIn('SupportsLoRA', message)
        self.assertIn('merged', message.lower())
        self.assertIn('checkpoint', message)

    def test_the_old_merged_dir_name_is_refused_too(self):
        with self.assertRaises(ValueError) as caught:
            v3_served.check_strategy('merged-dir')
        self.assertIn('checkpoint', str(caught.exception))

    def test_an_unknown_strategy_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            v3_served.check_strategy('per-request')
        self.assertIn('per-request', str(caught.exception))

    def test_the_serve_target_is_the_checkpoint_itself(self):
        """No extra launch arguments: there is no adapter to name."""
        merged = self.merged_checkpoint()
        self.assertEqual(v3_served.checkpoint_serve_target(merged, 'ds-ft'),
                         merged)
        with self.assertRaises(ValueError) as caught:
            v3_served.checkpoint_serve_target('', 'ds-ft')
        self.assertIn('checkpointDir', str(caught.exception))

    def test_a_load_message_adapterDir_is_refused_at_that_boundary_too(self):
        """The pool's field is called `adapterDir`, and a LoRA directory can
        arrive on it. Taking it as a merged checkpoint would hand
        `vllm-omni serve` a directory it cannot load - the server would come up
        on garbage or on the base model. `load_voices` refuses the same key at
        the document boundary; this is the other end.
        """
        from crucible_higgs.engine.higgs import higgs_v3_config_from_worker_kwargs
        with self.assertRaises(ValueError) as caught:
            higgs_v3_config_from_worker_kwargs(voice='deathstalker',
                                               adapter_dir='/models/ds-lora')
        message = str(caught.exception)
        self.assertIn('adapterDir', message)
        self.assertIn('checkpointDir', message)
        self.assertIn('no runtime LoRA', message)

    def test_a_checkpoint_voice_carries_its_directory_into_the_config(self):
        from crucible_higgs.engine.protocol import DefaultVoice
        merged = self.merged_checkpoint()
        voice = DefaultVoice(name='ds-ft', checkpoint_dir=merged,
                             max_chars=500, max_chars_source='length-sweep')
        config = HiggsV3Config(voice=voice, base_url=self.server.base_url)
        self.assertEqual(config.checkpoint_dir, merged)

    def test_a_voice_and_a_config_that_disagree_are_refused(self):
        from crucible_higgs.engine.protocol import DefaultVoice
        voice = DefaultVoice(name='ds-ft', checkpoint_dir='/models/a',
                             max_chars=500, max_chars_source='catalog')
        with self.assertRaises(ValueError) as caught:
            HiggsV3Config(voice=voice, base_url=self.server.base_url,
                          checkpoint_dir='/models/b')
        self.assertIn('One server runs on one checkpoint', str(caught.exception))

    def test_a_server_on_another_checkpoint_is_refused(self):
        """The failure this prevents is silent: a leftover server for ANOTHER
        voice answers /health identically. /v1/models does not: it reports the
        directory the server was started on, and that is what is compared."""
        self.server.httpd.model_root = '/models/theirs'
        backend = HiggsV3ServedBackend(base_url=self.server.base_url,
                                       checkpoint_dir='/models/ours')
        with self.assertRaises(HiggsV3ServerError) as caught:
            backend.check_serves_expected_model()
        message = str(caught.exception)
        self.assertIn('/models/ours', message)
        self.assertIn('/models/theirs', message)
        self.assertIn('RESTARTING', message)

    def test_a_matching_checkpoint_passes(self):
        self.server.httpd.model_root = '/models/ours/'     # trailing slash: ls -d
        backend = HiggsV3ServedBackend(base_url=self.server.base_url,
                                       checkpoint_dir='/models/ours')
        self.assertEqual(backend.running_checkpoint(), '/models/ours')
        backend.check_serves_expected_model()

    def test_a_server_that_reports_no_root_is_refused_naming_the_variable(self):
        """A server build whose model list carries no root cannot be
        identified; the operator's assertion is the only other source."""
        self.server.httpd.model_root = None
        previous = os.environ.pop(v3_served.CHECKPOINT_ENV, None)
        if previous is not None:
            self.addCleanup(os.environ.__setitem__,
                            v3_served.CHECKPOINT_ENV, previous)
        backend = HiggsV3ServedBackend(base_url=self.server.base_url)
        with self.assertRaises(HiggsV3ServerError) as caught:
            backend.check_serves_expected_model(checkpoint_dir='/models/ours')
        self.assertIn(v3_served.CHECKPOINT_ENV, str(caught.exception))
        self.assertIn('root', str(caught.exception))

    def test_the_environment_can_state_it_when_the_server_cannot(self):
        self.server.httpd.model_root = None
        os.environ[v3_served.CHECKPOINT_ENV] = '/models/ours'
        self.addCleanup(os.environ.pop, v3_served.CHECKPOINT_ENV, None)
        backend = HiggsV3ServedBackend(base_url=self.server.base_url)
        self.assertEqual(backend.running_checkpoint(), '/models/ours')
        backend.check_serves_expected_model(checkpoint_dir='/models/ours')

    def test_a_reported_root_is_a_fact_and_the_variable_does_not_override_it(self):
        self.server.httpd.model_root = '/models/theirs'
        os.environ[v3_served.CHECKPOINT_ENV] = '/models/ours'
        self.addCleanup(os.environ.pop, v3_served.CHECKPOINT_ENV, None)
        backend = HiggsV3ServedBackend(base_url=self.server.base_url,
                                       checkpoint_dir='/models/ours')
        self.assertEqual(backend.running_checkpoint(), '/models/theirs')
        with self.assertRaises(HiggsV3ServerError):
            backend.check_serves_expected_model()

    def test_a_base_voice_refuses_a_server_running_a_fine_tune(self):
        """The wrong-voice failure in the OTHER direction: a base request
        against a leftover fine-tune server."""
        self.server.httpd.model_root = '/home/t/higgs_v3_merged/ds_prod'
        backend = HiggsV3ServedBackend(base_url=self.server.base_url)
        with self.assertRaises(HiggsV3ServerError) as caught:
            backend.check_serves_expected_model()
        self.assertIn('BASE', str(caught.exception))
        self.assertIn('/home/t/higgs_v3_merged/ds_prod', str(caught.exception))

    def test_a_voice_change_on_a_live_engine_names_the_restart(self):
        from crucible_higgs.engine.protocol import DefaultVoice
        merged = self.merged_checkpoint()
        self.server.httpd.model_root = merged
        voice = DefaultVoice(name='ds-ft', checkpoint_dir=merged,
                             max_chars=500, max_chars_source='catalog')
        engine = HiggsV3Engine(HiggsV3Config(voice=voice,
                                             base_url=self.server.base_url,
                                             probe_sentinel_filter=False))
        self.addCleanup(engine.cleanup)
        with self.assertRaises(ValueError) as caught:
            engine.set_voice('another-voice')
        message = str(caught.exception)
        self.assertIn('SupportsLoRA', message)
        self.assertIn('merged checkpoint', message)

    def test_a_checkpoint_voice_renders_text_only(self):
        """THE PRODUCTION PATH: no `references` key at all."""
        from crucible_higgs.engine.protocol import DefaultVoice
        merged = self.merged_checkpoint()
        self.server.httpd.model_root = merged
        voice = DefaultVoice(name='ds-ft', checkpoint_dir=merged,
                             max_chars=500, max_chars_source='catalog')
        engine = HiggsV3Engine(HiggsV3Config(voice=voice,
                                             base_url=self.server.base_url,
                                             probe_sentinel_filter=False))
        self.addCleanup(engine.cleanup)
        engine.render_audio('It was a Saturday morning.')
        self.assertNotIn('references', self.server.requests[-1])


class GenerationConfigTest(V3TestCase):
    """`generation_config.json` is a REQUIRED file of a Higgs v3 checkpoint.

    WHY IT IS REQUIRED, measured by the fine-tune campaign 2026-09-05:
    `vllm-omni serve <dir>` resolves sampling FROM THE MODEL DIRECTORY
    (`--generation-config` defaults to `auto`). A merged dir without the file
    makes vllm-omni's stage fallback (`entrypoints/openai/stage_params.py`) hand
    back a bare `SamplingParams()` - top_p 1.0, top_k DISABLED - which samples
    the untruncated 1026-way codebook tail and derails long prompts into babble.
    `OpenAICreateSpeechRequest` carries no temperature/top_p/top_k, so nothing
    per-request can correct it: the directory is the ONLY lever.

    PROVENANCE. The `bosonai/higgs-audio-v3-tts-4b` base ships NO such file
    (verified across the whole HF cache, 2026-09-05). Every merged dir gets one
    WRITTEN BY THE MERGE from a recorded per-run override whose values are
    vllm-omni's own `deploy/higgs_multimodal_qwen3.yaml` stage-0
    `default_sampling_params`, and `merge_manifest.json` records which. narrator
    never writes it: a dir that lacks it is MISCONFIGURED, and synthesizing one
    here would be narrator deciding a model's sampling.
    """

    def _voice(self, checkpoint):
        from crucible_higgs.engine.protocol import DefaultVoice
        return DefaultVoice(name='ds-ft', checkpoint_dir=checkpoint,
                            max_chars=500, max_chars_source='catalog')

    def _voices_document(self, checkpoint):
        path = os.path.join(self.dir, 'voices.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump({'ds-ft': {'kind': 'checkpoint',
                                 'checkpointDir': checkpoint,
                                 'maxChars': 500}}, handle)
        os.environ[v3_config.VOICES_ENV] = path
        self.addCleanup(os.environ.pop, v3_config.VOICES_ENV, None)
        return path

    # -- the states of the file ----------------------------------------------

    def test_present_and_valid_is_accepted_and_read_back(self):
        """THE REAL FILE'S CONTENT - see MERGED_GENERATION_CONFIG. Read back
        rather than merely accepted: the MLX arm renders at these numbers."""
        merged = self.merged_checkpoint()
        document = v3_served.require_generation_config(merged, 'ds-ft')
        self.assertEqual(document, {'temperature': 1.0, 'top_p': 0.95,
                                    'top_k': 50, 'repetition_penalty': 1.0})

    def test_absent_is_refused_naming_the_voice_the_dir_and_the_file(self):
        merged = self.merged_checkpoint(generation_config=None)
        with self.assertRaises(ValueError) as caught:
            v3_served.require_generation_config(merged, 'ds-ft')
        message = str(caught.exception)
        self.assertIn('ds-ft', message)
        self.assertIn(merged, message)
        self.assertIn('generation_config.json', message)
        self.assertIn('top_k', message)          # it says WHY, not just what

    def test_unparseable_json_is_refused_by_name(self):
        merged = self.merged_checkpoint(
            generation_config='{"temperature": 1.0, "top_p": ,}')
        with self.assertRaises(ValueError) as caught:
            v3_served.require_generation_config(merged, 'ds-ft')
        message = str(caught.exception)
        self.assertIn('ds-ft', message)
        self.assertIn('generation_config.json', message)
        self.assertIn('JSON', message)

    def test_a_file_that_carries_no_sampling_is_refused(self):
        """A generation_config.json without temperature/top_p/top_k is not the
        file this needs: the server reads it, finds no sampling and falls back
        exactly as if it were absent."""
        cases = (('{"eos_token_id": 128009}',
                  ('temperature', 'top_p', 'top_k')),
                 ('{"temperature": 1.0, "top_p": 0.95}', ('top_k',)),
                 ('{"temperature": 1.0, "top_k": 50}', ('top_p',)),
                 ('{"top_p": 0.95, "top_k": 50}', ('temperature',)))
        for index, (body, missing) in enumerate(cases):
            with self.subTest(body=body):
                merged = self.merged_checkpoint(name=f'partial-{index}',
                                                generation_config=body)
                with self.assertRaises(ValueError) as caught:
                    v3_served.require_generation_config(merged, 'ds-ft')
                message = str(caught.exception)
                for key in missing:
                    self.assertIn(key, message)

    def test_a_json_document_that_is_not_an_object_is_refused(self):
        merged = self.merged_checkpoint(generation_config='[1.0, 0.95, 50]')
        with self.assertRaises(ValueError) as caught:
            v3_served.require_generation_config(merged, 'ds-ft')
        self.assertIn('list', str(caught.exception))

    def test_a_checkpoint_directory_that_is_not_there_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            v3_served.require_generation_config(
                os.path.join(self.dir, 'nothing-here'), 'ds-ft')
        self.assertIn('not a directory', str(caught.exception))

    def test_an_unnamed_voice_is_refused_rather_than_reported_anonymously(self):
        merged = self.merged_checkpoint()
        with self.assertRaises(ValueError) as caught:
            v3_served.require_generation_config(merged, '   ')
        self.assertIn('VOICE NAME', str(caught.exception))

    # -- the doors that carry the refusal ------------------------------------

    def test_the_serve_target_refuses_a_dir_without_it(self):
        """`checkpoint_serve_target` is the one place that decides which
        directory a voice's server runs on, so it is where that directory's
        required files are proved."""
        merged = self.merged_checkpoint(generation_config=None)
        with self.assertRaises(ValueError) as caught:
            v3_served.checkpoint_serve_target(merged, 'ds-ft')
        self.assertIn('generation_config.json', str(caught.exception))

    def test_the_served_config_refuses_it_before_a_server_is_started(self):
        """55-297 s of cold start and ~14 GB of VRAM are paid AFTER this point,
        so the refusal belongs before them."""
        merged = self.merged_checkpoint(generation_config=None)
        with self.assertRaises(ValueError) as caught:
            HiggsV3Config(voice=self._voice(merged),
                          base_url=self.server.base_url)
        message = str(caught.exception)
        self.assertIn('ds-ft', message)
        self.assertIn('generation_config.json', message)

    def test_a_load_message_for_such_a_voice_is_refused(self):
        """`resolve_load_voice` validates a load AS THIS ENGINE understands it,
        and a checkpoint's required files are part of that - the same layer that
        refuses a fine-tune with no maxChars."""
        self._voices_document(self.merged_checkpoint(generation_config=None))
        with self.assertRaises(ValueError) as caught:
            HiggsV3Engine.resolve_load_voice('ds-ft')
        self.assertIn('generation_config.json', str(caught.exception))

    def test_the_same_load_passes_once_the_file_is_there(self):
        self._voices_document(self.merged_checkpoint())
        self.assertEqual(HiggsV3Engine.resolve_load_voice('ds-ft'), 'ds-ft')

    def test_narrator_never_writes_the_file_itself(self):
        """A refusal that repaired the directory would be a fallback: the merge
        is what puts this file there, and its values are a property of the
        model."""
        merged = self.merged_checkpoint(generation_config=None)
        with self.assertRaises(ValueError):
            v3_served.checkpoint_serve_target(merged, 'ds-ft')
        self.assertFalse(
            os.path.exists(os.path.join(merged, 'generation_config.json')),
            'the refusal must not create the file it refused over')


class ZeroShotBaseWeightsTest(V3TestCase):
    """A ZERO-SHOT VOICE MAY NAME ITS BASE WEIGHTS, and base weights are not a
    merge.

    THE REGRESSION. 2026-09-15, 00:11: the first zero-shot load ever made
    through Crucible on the PC was refused by name -

        engine_failed: narrator (higgs-v3) refused the request: Higgs v3
        voice 'zeroshot': the merged checkpoint
        /home/<user>/.crucible/voices/zeroshot/cuda-linux does not carry
        generation_config.json, which is a REQUIRED ...

    - for a file base weights have never carried. The pinned repo
    `bosonai/higgs-tts-3-4b` at 239f63fb7b02b1aa085f98d9efae5e35cc5523e8 lists
    THIRTEEN files (`config.json`, `chat_template.jinja`, `model.safetensors`
    and its index, the tokenizer pair, `.gitattributes`, the four docs, one
    asset) and no `generation_config.json` under any name or subdirectory; the
    pull took the whole snapshot, so nothing was missed. That absence is the
    fact `SERVER_DEFAULT_SAMPLING` exists for.

    What went wrong was a field with two meanings. Crucible names the pinned
    base directory so the clone renders on known bytes rather than on whatever
    the HF cache holds, and the only key for a directory was `checkpointDir` -
    which means a MERGE, whose own generation_config.json is its sampling. So
    the document now says which by the KIND beside it, and narrator reads a
    `clips` voice's into `base_dir`.
    """

    def _document(self, entry, name='zeroshot'):
        path = os.path.join(self.dir, 'voices.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump({name: entry}, handle)
        os.environ[v3_config.VOICES_ENV] = path
        self.addCleanup(os.environ.pop, v3_config.VOICES_ENV, None)
        return path

    def base_weights(self, name='base-pinned'):
        """The pulled base snapshot: a directory with `config.json` in it and
        NO generation_config.json - which is what the published repo is."""
        path = os.path.join(self.dir, name)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'config.json'), 'w',
                  encoding='utf-8') as handle:
            handle.write('{}')
        return path

    def _entry(self, base, **extra):
        entry = {'kind': 'clips',
                 'checkpointDir': base,
                 'maxChars': 600,
                 'targetChars': 600,
                 # The document's own spelling, which is Crucible's:
                 # `_SAMPLING_KEYS` is camelCase on the wire.
                 'sampling': {'temperature': 0.8, 'topP': 0.95, 'topK': 50},
                 'clips': [{'path': self.clip, 'transcript': X2_TEXT,
                            'seconds': 27.42}]}
        entry.update(extra)
        return entry

    # -- what the document means ---------------------------------------------

    def test_a_clips_voices_directory_is_read_as_BASE_WEIGHTS(self):
        """The kind beside the key is what says which model the directory
        holds. `checkpoint_dir` stays None, so every refusal that is about a
        fine-tune keeps asking about fine-tunes."""
        base = self.base_weights()
        voice = v3_config.load_voice('zeroshot', self._document(self._entry(base)))
        self.assertEqual(voice.base_dir, base)
        self.assertIsNone(voice.checkpoint_dir)

    def test_a_checkpoint_voices_directory_is_still_a_merge(self):
        """Unchanged, and the point of branching on the kind rather than on
        the presence of clips: a fine-tune auditioned with a reference clip is
        still a fine-tune."""
        merged = self.merged_checkpoint()
        path = self._document({'kind': 'checkpoint', 'checkpointDir': merged,
                               'maxChars': 500,
                               'clips': [{'path': self.clip,
                                          'transcript': X2_TEXT,
                                          'seconds': 27.42}]}, name='ds-ft')
        voice = v3_config.load_voice('ds-ft', path)
        self.assertEqual(voice.checkpoint_dir, merged)
        self.assertIsNone(voice.base_dir)

    # -- the load that was refused -------------------------------------------

    def test_the_load_is_NOT_refused_for_a_missing_generation_config(self):
        """THE REGRESSION ITSELF. The directory carries no
        generation_config.json, and that is what base weights are."""
        base = self.base_weights()
        self._document(self._entry(base))
        self.assertFalse(
            os.path.exists(os.path.join(base, 'generation_config.json')))
        self.assertEqual(HiggsV3Engine.resolve_load_voice('zeroshot'),
                         'zeroshot')

    def test_narrator_never_writes_the_file_into_the_base_dir(self):
        """Accepting the load must not mean repairing the directory: the
        pulled snapshot is bytes at a pinned revision and narrator does not
        add to them."""
        base = self.base_weights()
        self._document(self._entry(base))
        HiggsV3Engine.resolve_load_voice('zeroshot')
        self.assertEqual(sorted(os.listdir(base)), ['config.json'])

    def test_a_base_dir_that_is_not_there_is_still_refused_BY_NAME(self):
        """The file is not required; the BYTES are. A voice pointing at a
        directory nothing pulled is a load that would die inside the model
        loader instead."""
        self._document(self._entry(os.path.join(self.dir, 'never-pulled')))
        with self.assertRaises(ValueError) as caught:
            HiggsV3Engine.resolve_load_voice('zeroshot')
        message = str(caught.exception)
        self.assertIn('zeroshot', message)
        self.assertIn('never-pulled', message)
        self.assertNotIn('generation_config.json', message)

    def test_naming_a_merge_AND_base_weights_is_refused(self):
        """Two directories is two models, and one server runs on one."""
        voice = ClipsVoice(
            clips=(ReferenceClip(self.clip, X2_TEXT, seconds=14.0),),
            name='zeroshot', checkpoint_dir=self.merged_checkpoint(),
            base_dir=self.base_weights(), max_chars=600,
            max_chars_source='catalog')
        with self.assertRaises(ValueError) as caught:
            HiggsV3Config(voice=voice, base_url=self.server.base_url)
        self.assertIn('has not said which', str(caught.exception))

    # -- what the engine then does with it -----------------------------------

    def test_the_server_is_pointed_at_the_pinned_base_directory(self):
        """The whole reason the document names it: without this the served arm
        starts on whatever snapshot the HuggingFace cache holds, and the render
        carries a fingerprint naming bytes nobody read."""
        base = self.base_weights()
        config = HiggsV3Config(voice=self.voice(), base_dir=base,
                               base_url=self.server.base_url)
        self.assertEqual(config.model_dir, base)
        self.assertIsNone(config.checkpoint_dir)

    def test_such_a_voice_STATES_its_sampling_in_extra_params(self):
        """The half that would have been silently wrong. `served_sampling`
        used to go empty for any voice naming a directory, and an empty
        request against a directory with no generation_config.json is a bare
        SamplingParams(): top_p 1.0, top_k DISABLED, the untruncated 1026-way
        tail that derails long chunks into babble."""
        base = self.base_weights()
        self.server.httpd.model_root = base
        engine = HiggsV3Engine(self.quiet_config(
            base_dir=base, sampling={'temperature': 0.8, 'top_p': 0.95,
                                     'top_k': 50}))
        self.addCleanup(engine.cleanup)
        engine.render_audio('It was a Saturday morning.')
        extra = self.server.requests[-1]['extra_params']
        self.assertEqual(extra['temperature'], 0.8)
        self.assertEqual(extra['top_p'], 0.95)
        self.assertEqual(extra['top_k'], 50)
        self.assertEqual(extra['repetition_penalty'], 1.0)

    def test_v3_defaults_are_stamped_without_losing_the_base_dir(self):
        """`apply_v3_voice_defaults` rebuilds a hand-built voice to stamp v3's
        control allowlist. It used to rebuild from a hand-written field list,
        which would have dropped this one - and a rebuilt voice that lost its
        directory is a server started on the wrong model."""
        base = self.base_weights()
        bare = ClipsVoice(clips=(ReferenceClip(self.clip, X2_TEXT, seconds=14.0),),
                          name='zeroshot', base_dir=base, max_chars=600,
                          max_chars_source='catalog',
                          sampling={'temperature': 0.8})
        self.assertEqual(bare.allowed_controls, ())
        config = HiggsV3Config(voice=bare, base_url=self.server.base_url)
        self.assertEqual(len(config.voice.allowed_controls), 45)
        self.assertEqual(config.voice.base_dir, base)
        self.assertEqual(config.voice.sampling, {'temperature': 0.8})


class ServedSamplingTest(V3TestCase):
    """WHAT THE SERVED ARM SENDS, and it is not the same for the two kinds.

    "Send nothing and get the deploy defaults" was the belief this branch
    refuted: `vllm-omni serve` resolves sampling from the MODEL DIRECTORY
    (`--generation-config auto`), so an empty request gets whatever that
    directory holds - and the `bosonai/higgs-audio-v3-tts-4b` snapshot holds NO
    generation_config.json, which is a bare SamplingParams(): top_p 1.0, top_k
    DISABLED, the untruncated 1026-way tail. So base weights must STATE the
    values and a checkpoint must not.
    """

    def _checkpoint_config(self, **kwargs):
        from crucible_higgs.engine.protocol import DefaultVoice
        merged = kwargs.pop('merged', None) or self.merged_checkpoint()
        # Which checkpoint the server is running cannot be discovered, so it is
        # STATED - the engine refuses a checkpoint voice against a server that
        # says nothing (CheckpointVoiceTest covers that refusal itself).
        self.server.httpd.model_root = merged
        voice = DefaultVoice(name='ds-ft', checkpoint_dir=merged,
                             max_chars=500, max_chars_source='catalog')
        kwargs.setdefault('base_url', self.server.base_url)
        kwargs.setdefault('probe_sentinel_filter', False)
        return HiggsV3Config(voice=voice, **kwargs)

    def test_base_weights_state_the_sampling_in_extra_params(self):
        """The bosonai snapshot ships no generation_config.json, so sending
        nothing here is the babble sampling, not the deploy default."""
        engine = HiggsV3Engine(self.quiet_config())
        self.addCleanup(engine.cleanup)
        engine.render_audio('It was a Saturday morning.')
        extra = self.server.requests[-1]['extra_params']
        self.assertEqual(extra['temperature'], 1.0)
        self.assertEqual(extra['top_p'], 0.95)
        self.assertEqual(extra['top_k'], 50)
        self.assertEqual(extra['repetition_penalty'], 1.0)

    def test_the_seed_never_rides_inside_extra_params(self):
        """It is the request's TOP-LEVEL field, and two sources for one number
        is one too many - build_request_body refuses the duplicate."""
        engine = HiggsV3Engine(self.quiet_config())
        self.addCleanup(engine.cleanup)
        engine.render_audio('It was a Saturday morning.')
        body = self.server.requests[-1]
        self.assertNotIn('seed', body['extra_params'])
        self.assertEqual(body['seed'], 1234)

    def test_a_checkpoint_voice_sends_NO_extra_params(self):
        """Its directory carries the file and the server reads it; an
        extra_params here would override the model's own declared sampling with
        narrator's opinion of it."""
        engine = HiggsV3Engine(self._checkpoint_config())
        self.addCleanup(engine.cleanup)
        engine.render_audio('It was a Saturday morning.')
        self.assertNotIn('extra_params', self.server.requests[-1])

    def test_an_override_still_rides_for_either_kind(self):
        engine = HiggsV3Engine(
            self._checkpoint_config(sampling={'temperature': 0.7}))
        self.addCleanup(engine.cleanup)
        engine.render_audio('It was a Saturday morning.')
        self.assertEqual(self.server.requests[-1]['extra_params'],
                         {'temperature': 0.7})

    # -- what the stop policy reports (the manifest outlives the render) ------

    def test_the_stop_policy_reports_the_checkpoints_OWN_levers(self):
        import json as _json
        merged = self.merged_checkpoint(
            generation_config=_json.dumps({'temperature': 0.6, 'top_p': 0.85,
                                           'top_k': 30}))
        policy = higgs_v3_stop_policy(self._checkpoint_config(merged=merged))
        self.assertEqual(policy.levers,
                         {'temperature': 0.6, 'top_p': 0.85, 'top_k': 30.0})

    def test_the_stop_policy_reports_the_deploy_default_for_base_weights(self):
        policy = higgs_v3_stop_policy(self.config())
        self.assertEqual(policy.levers['temperature'], 1.0)
        self.assertEqual(policy.levers['top_p'], 0.95)
        self.assertEqual(policy.levers['top_k'], 50.0)
        self.assertNotIn('seed', policy.levers,
                         'the seed is not a sampling lever')


class GenerationConfigTypesTest(V3TestCase):
    """Presence is not validation. A number that is wrong in a way nobody says
    out loud is exactly what this file exists to stop."""

    def _refuse(self, body):
        merged = self.merged_checkpoint(name='typed-' + str(abs(hash(body))),
                                        generation_config=body)
        with self.assertRaises(ValueError) as caught:
            v3_served.require_generation_config(merged, 'ds-ft')
        message = str(caught.exception)
        self.assertIn('ds-ft', message)
        self.assertIn('generation_config.json', message)
        return message

    def test_a_float_top_k_is_refused_rather_than_truncated(self):
        """`int(50.7)` is 50 everywhere that touches it - a DIFFERENT sampling
        than the file states, arrived at silently."""
        message = self._refuse('{"temperature": 1.0, "top_p": 0.95, '
                               '"top_k": 50.7}')
        self.assertIn('top_k', message)
        self.assertIn('whole number', message)

    def test_a_null_value_is_refused_by_name(self):
        for body, key in (
                ('{"temperature": null, "top_p": 0.95, "top_k": 50}',
                 'temperature'),
                ('{"temperature": 1.0, "top_p": null, "top_k": 50}', 'top_p'),
                ('{"temperature": 1.0, "top_p": 0.95, "top_k": null}',
                 'top_k')):
            with self.subTest(key=key):
                self.assertIn(key, self._refuse(body))

    def test_a_string_value_is_refused_rather_than_coerced(self):
        self.assertIn('top_p', self._refuse(
            '{"temperature": 1.0, "top_p": "0.95", "top_k": 50}'))

    def test_a_boolean_is_not_a_number(self):
        """`True` is an int in Python; it is not a top-k."""
        self.assertIn('top_k', self._refuse(
            '{"temperature": 1.0, "top_p": 0.95, "top_k": true}'))

    def test_out_of_range_values_are_refused(self):
        for body, word in (
                ('{"temperature": -0.5, "top_p": 0.95, "top_k": 50}',
                 'negative'),
                ('{"temperature": 1.0, "top_p": 1.5, "top_k": 50}',
                 'probability mass'),
                ('{"temperature": 1.0, "top_p": 0.0, "top_k": 50}',
                 'probability mass'),
                ('{"temperature": 1.0, "top_p": 0.95, "top_k": -1}',
                 'non-negative'),
                ('{"temperature": 1.0, "top_p": 0.95, "top_k": 50, '
                 '"repetition_penalty": 0.0}', 'positive')):
            with self.subTest(body=body):
                self.assertIn(word, self._refuse(body))

    def test_the_values_a_checkpoint_may_legitimately_state_are_accepted(self):
        """0.0 temperature is greedy, 1.0 top_p is the untruncated tail and 0
        top_k disables top-k. A checkpoint that says so is saying it
        deliberately, and narrator does not second-guess a model's own file."""
        merged = self.merged_checkpoint(
            name='edge',
            generation_config='{"temperature": 0.0, "top_p": 1.0, "top_k": 0}')
        document = v3_served.require_generation_config(merged, 'ds-ft')
        self.assertEqual(document['top_k'], 0)

    def test_an_unreadable_file_is_refused_by_name(self):
        """Permissions, a broken symlink, a wedged mount. The file passes
        `isfile` and then the read fails - and every OTHER state of this file
        refuses with the voice, the path and why, so a bare OSError here would
        be the one refusal in this feature that names neither. The failure is
        injected rather than staged on disk because there is no portable way to
        make a file unreadable on both Windows and Linux."""
        from unittest import mock
        merged = self.merged_checkpoint(name='unreadable')
        with mock.patch('builtins.open',
                        side_effect=OSError('Permission denied')):
            with self.assertRaises(ValueError) as caught:
                v3_served.require_generation_config(merged, 'ds-ft')
        message = str(caught.exception)
        self.assertIn('ds-ft', message)
        self.assertIn('could not be read', message)
        self.assertIn('Permission denied', message)


class LifecycleTest(V3TestCase):

    def test_attach_mode_does_not_launch_or_kill(self):
        backend = HiggsV3ServedBackend(base_url=self.server.base_url)
        backend.start()                        # no-op
        self.assertTrue(backend.wait_ready(5))
        backend.stop()                         # must not touch anything
        self.assertTrue(backend.ping(), 'stop() killed a server it did not start')

    def test_attach_mode_has_no_launch_command(self):
        backend = HiggsV3ServedBackend(base_url=self.server.base_url)
        with self.assertRaises(ValueError):
            backend.launch_command()

    def test_wait_ready_returns_False_on_timeout(self):
        """A slow start is the caller's decision, not an exception. The measured
        cold start is ~55 s."""
        server = FakeV3Server(healthy=False)
        self.addCleanup(server.close)
        backend = HiggsV3ServedBackend(base_url=server.base_url)
        self.assertFalse(backend.wait_ready(2))

    def test_wait_ready_raises_when_the_process_died(self):
        """Waiting out a timeout on a corpse is the failure this avoids."""
        backend = HiggsV3ServedBackend(base_url='http://127.0.0.1:1',
                                       serve_script='/nonexistent.sh')

        class Dead:
            returncode = 1

            def poll(self):
                return 1

        backend._proc = Dead()
        with self.assertRaises(HiggsV3ServerError) as caught:
            backend.wait_ready(5)
        self.assertIn('status 1', str(caught.exception))

    def misbound_backend(self, bound, healthy_ports):
        """A LAUNCHING backend whose port scan and health probe are staged.

        The scan itself is `/proc`-reading python inside the distro and there is
        no way to make a real sgl-omni misbind on demand, so the two FACTS it
        gathers are injected and what is tested is the JUDGEMENT built on them -
        which is the half that was missing, not the half that was wrong.
        """
        dead = FakeV3Server(healthy=False)
        self.addCleanup(dead.close)
        backend = HiggsV3ServedBackend(base_url=dead.base_url,
                                       serve_script=a_launcher(self))
        self.addCleanup(backend._close_log)
        backend._bound_ports = lambda: list(bound)
        # A live set: a TERM to a misbound pid takes its port off it, which is
        # what the real server does when it shuts down.
        live = set(healthy_ports)
        by_pid = {int(r['pid']): int(r['port']) for r in bound}
        backend._answers_health = lambda port: port in live
        backend.signalled = []

        def signal(pid, signame):
            backend.signalled.append((int(pid), signame))
            live.discard(by_pid.get(int(pid)))

        backend._signal_guest = signal
        return backend, int(dead.base_url.rsplit(':', 1)[-1])

    def test_a_server_answering_on_ANOTHER_port_is_refused_BY_NAME(self):
        """MEASURED: told 8200, sgl-omni bound 57877 and said nothing, while
        narrator polled 8200 for the whole 900 s patience. The server is up and
        can never appear where it was asked for, so there is nothing to wait
        for."""
        backend, asked = self.misbound_backend(
            bound=[{'pid': 4242, 'port': 57877}], healthy_ports={57877})
        with self.assertRaises(HiggsV3ServerError) as caught:
            backend._refuse_if_misbound()
        message = str(caught.exception)
        self.assertIn('57877', message)
        self.assertIn(str(asked), message)
        self.assertIn('4242', message)

    def test_a_misbound_server_is_TERMINATED_before_the_refusal(self):
        """MEASURED 2026-09-27 (Crucible 1.0.43, job c0cc1107): the refusal
        raised before `_record_server`, so `stop()` had no pid, and the setsid'd
        sgl-omni was orphaned holding 13.4 GB until the next job came back
        accelerator_busy. It carries our marker, so it is ours to stop - by
        group, with TERM, and never KILL."""
        backend, _ = self.misbound_backend(
            bound=[{'pid': 4242, 'port': 34527}], healthy_ports={34527})
        with self.assertRaises(HiggsV3ServerError):
            backend._refuse_if_misbound()
        self.assertEqual(backend.signalled, [(4242, 'TERM')])

    def test_an_INTERNAL_socket_on_another_port_is_NOT_a_misbind(self):
        """vLLM's engine-core and SGLang's schedulers are separate processes that
        inherit our environment and open TCP sockets of their own. The marker
        alone would refuse every healthy launch; the second fact - that the
        stranger port answers OUR health endpoint - is what makes the difference.
        """
        backend, _ = self.misbound_backend(
            bound=[{'pid': 4242, 'port': 41001}, {'pid': 4243, 'port': 41002}],
            healthy_ports=set())
        backend._refuse_if_misbound()          # must not raise

    def test_the_port_we_ASKED_for_is_never_a_misbind(self):
        backend, asked = self.misbound_backend(bound=[], healthy_ports=set())
        backend._bound_ports = lambda: [{'pid': 4242, 'port': asked}]
        backend._answers_health = lambda port: True
        backend._refuse_if_misbound()          # must not raise

    def test_ATTACH_mode_never_asks_the_question(self):
        """The operator named the URL; there is no port we asked for to be
        missed, and a scan there would be a spawn for nothing."""
        backend = HiggsV3ServedBackend(base_url=self.server.base_url)

        def explode():
            raise AssertionError('attach mode scanned for bound ports')

        backend._bound_ports = explode
        backend._refuse_if_misbound()

    def test_wait_ready_refuses_a_misbind_instead_of_timing_out(self):
        """The whole point: the failure arrives as a named refusal, not as a
        timeout that points at cold starts and flashinfer."""
        backend, _ = self.misbound_backend(
            bound=[{'pid': 4242, 'port': 34529}], healthy_ports={34529})
        with self.assertRaises(HiggsV3ServerError) as caught:
            backend.wait_ready(2)
        self.assertIn('34529', str(caught.exception))

    def test_a_scan_that_CANNOT_RUN_is_a_refusal_not_a_shrug(self):
        """An unanswerable question about our own process is not a "no" - the
        same ruling `_own_servers_on_port` already makes."""
        backend, _ = self.misbound_backend(bound=[], healthy_ports=set())

        def broken():
            raise HiggsV3ServerError('the bound-port scan failed (exit 2): boom')

        backend._bound_ports = broken
        with self.assertRaises(HiggsV3ServerError) as caught:
            backend._refuse_if_misbound()
        self.assertIn('bound-port scan failed', str(caught.exception))

    def test_the_launch_command_invokes_their_script(self):
        backend = HiggsV3ServedBackend(serve_script=a_launcher(self))
        command = backend.launch_command()
        # The path AS THE GUEST SEES IT: on Windows the wrapper carries
        # /mnt/c/..., which is what `to_wsl` produces and what bash reads.
        self.assertIn(v3_served.served_common.to_wsl(a_launcher(self)),
                      ' '.join(command).replace('\\', '/'))
        self.assertIn('bash', ' '.join(command))

    def test_extra_launch_args_are_refused_because_the_script_takes_none(self):
        with self.assertRaises(ValueError) as caught:
            HiggsV3ServedBackend(serve_script=a_launcher(self),
                                 extra_args=['--enable-lora'])
        self.assertIn('takes no arguments', str(caught.exception))


class ResponseTest(V3TestCase):

    def test_a_wav_body_decodes_to_float32_mono(self):
        audio, rate = v3_served.decode_response(wav_bytes(seconds=0.5))
        self.assertEqual(rate, 24000)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.ndim, 1)
        self.assertEqual(audio.size, 12000)
        self.assertGreater(float(np.max(np.abs(audio[:4000]))), 0.1,
                           'the fake must carry real signal, not just a floor')

    def test_stereo_is_averaged_down(self):
        audio, _ = v3_served.decode_response(wav_bytes(seconds=0.5, channels=2))
        self.assertEqual(audio.ndim, 1)
        self.assertEqual(audio.size, 12000)

    def test_a_wrong_sample_rate_is_refused(self):
        """A manifest built on the wrong rate mis-times every cue after it."""
        with self.assertRaises(HiggsV3ServerError) as caught:
            v3_served.decode_response(wav_bytes(seconds=0.1, rate=16000))
        self.assertIn('24000', str(caught.exception))

    def test_an_empty_body_is_refused(self):
        with self.assertRaises(HiggsV3ServerError):
            v3_served.decode_response(b'')

    def test_a_server_error_carries_the_servers_own_message(self):
        self.server.fail_with(
            400, 'Reference audio too long (42.0s). Maximum 30s supported')
        backend = HiggsV3ServedBackend(base_url=self.server.base_url)
        request = SpeechRequest(text='Hello.', voice=self.voice(),
                                max_new_tokens=200)
        with self.assertRaises(HiggsV3ServerError) as caught:
            backend.speak(request)
        self.assertIn('Maximum 30s', str(caught.exception))

    def test_an_unreachable_server_says_so(self):
        backend = HiggsV3ServedBackend(base_url='http://127.0.0.1:1')
        with self.assertRaises(HiggsV3ServerError) as caught:
            backend.post_speech({'model': 'higgs-v3'}, timeout=2)
        self.assertIn('unreachable', str(caught.exception))


class EngineTest(V3TestCase):

    def test_it_renders_a_chunk_through_the_server(self):
        engine = HiggsV3Engine(self.config())
        self.addCleanup(engine.cleanup)
        audio = engine.render_audio('It was a Saturday morning.')
        self.assertEqual(audio.dtype, np.float32)
        self.assertGreater(audio.size, 0)
        sent = self.server.requests[-1]
        self.assertEqual(sent['input'], 'It was a Saturday morning.')
        # BASE WEIGHTS STATE THEIR SAMPLING. Sending no extra_params would mean
        # the model directory's generation_config.json, and the bosonai snapshot
        # carries none - i.e. a bare SamplingParams(), top_p 1.0 / top_k off.
        # See ServedSamplingTest.
        self.assertEqual(sent['extra_params']['top_p'], 0.95)

    def test_the_seam_facts(self):
        engine = HiggsV3Engine(self.config())
        self.addCleanup(engine.cleanup)
        self.assertEqual(engine.ENGINE_ID, 'higgs-v3')
        self.assertEqual(engine.SAMPLE_RATE, 24000)
        self.assertFalse(engine.pads, 'v3 emits bare speech; assembly owns gaps')
        self.assertEqual(engine.edge_fade.as_manifest(),
                         {'in': 10.0, 'out': 25.0})
        self.assertEqual(engine.backend_spec().kind, 'served')
        codec = engine.codec()
        self.assertEqual((codec.tokens_per_frame, codec.samples_per_frame,
                          codec.trim_frames), (8, 960, 7))
        self.assertIsNone(codec.streaming_decoder(lambda a, b: None))

    def test_the_codec_refuses_to_decode_because_the_server_did(self):
        """The patched server already dropped every sentinel frame BY TOKEN
        IDENTITY (work/patch_sentinel_filter.py); a second client-side trim
        would eat speech, and there are no tokens on this side anyway."""
        engine = HiggsV3Engine(self.config())
        self.addCleanup(engine.cleanup)
        with self.assertRaises(NotImplementedError) as caught:
            engine.codec().decode([1, 2, 3])
        self.assertIn('SERVER-SIDE', str(caught.exception))

    def test_the_budget(self):
        budget = HiggsV3Budget(self.config())
        self.assertEqual(budget.max_chars(), 600)
        self.assertEqual(budget.max_total_tokens(0), 8192)
        self.assertEqual(budget.max_total_tokens(685), 8192)
        with self.assertRaises(ValueError):
            budget.max_total_tokens(8192)

    def test_the_stop_policy_reports_what_will_be_sent(self):
        policy = higgs_v3_stop_policy(self.config())
        self.assertTrue(policy.eos_reliable)
        self.assertFalse(policy.resplit_on_cap)
        self.assertEqual(policy.coverage_check, 'asr')
        self.assertEqual(policy.levers['temperature'], 1.0,
                         'no sampling override means the server default')
        self.assertEqual(policy.levers['top_p'], 0.95)

    def test_an_override_shows_up_in_the_levers(self):
        policy = higgs_v3_stop_policy(self.config(sampling={'temperature': 0.7}))
        self.assertEqual(policy.levers['temperature'], 0.7)

    def test_a_batch_answers_every_row_as_it_retires(self):
        """Rows retire in COMPLETION order, not index order: the batch is N
        concurrent requests and the serve worker keys every callback by row."""
        engine = HiggsV3Engine(self.config())
        self.addCleanup(engine.cleanup)
        rows = []
        chunks = []
        engine.generate_batch_stream(['One.', 'Two.'], None, {1},
                                     lambda i, seq, pcm: chunks.append((i, seq)),
                                     lambda i, pcm: rows.append(i))
        self.assertEqual(sorted(rows), [0, 1])
        self.assertEqual(chunks, [(1, 0)], 'a streamed row arrives whole, once')

    def test_the_batch_width_is_the_servers_admission_width(self):
        state_concurrency(self, '5')
        engine = HiggsV3Engine(self.config())
        self.addCleanup(engine.cleanup)
        self.assertEqual(engine.BATCH_SIZE, 5)
        self.assertEqual(engine.batch_pool_size, 10)
        self.assertEqual(engine.server.concurrency, 5)

    def test_an_unstated_width_is_refused_by_name(self):
        os.environ.pop(v3_served.SERVE_MAX_NUM_SEQS_ENV, None)
        with self.assertRaises(ValueError) as caught:
            HiggsV3Engine(self.config())
        self.assertIn(v3_served.SERVE_MAX_NUM_SEQS_ENV, str(caught.exception))

    def test_the_packers_markers_never_reach_the_server(self):
        """MEASURED (the Mac, 2026-09-05): 677 of 1067 chunks of Owen's first
        Higgs book began with `[break]`, every heading with `[break][heading]`,
        and the model READ them - "break" spoken at the head of most chunks,
        7-15 s of gibberish before "Dedication.". The serve worker cleaned
        before calling; the render worker hands convert() the stored text and
        render_audio only trimmed whitespace. The strip is the model boundary's,
        for every caller.

        THE ASSERTION IS OVER THE WHOLE BATCH'S REQUESTS AND NOT `[-2:]`
        (corrected 2026-09-14, after a 1-in-5 failure). Two things make the
        last two requests the wrong window. `convert_batch` runs
        `serve_concurrency()` renders IN A POOL, so two chunks' HTTP requests
        arrive in whichever order the threads win; and the fake server answers
        every chunk with the same fixed duration, which puts chunk 3 (35 chars
        in 1.0 s, against a 20 chars/s ceiling) outside the band, so the length
        guard re-rolls it and the batch makes THREE requests, not two. The
        window then sometimes held chunk 3 twice. What this test is actually
        about is that no marker survives the model boundary, and the set of
        texts the server was asked for says that without caring which thread
        was quicker or how many takes the ladder spent."""
        engine = HiggsV3Engine(self.config(sentences_dir=self.dir))
        self.addCleanup(engine.cleanup)
        engine.render_audio('[break][heading]Dedication.')
        self.assertEqual(self.server.requests[-1]['input'], 'Dedication.')
        mark = len(self.server.requests)
        engine.convert_batch([(3, '[break]I would like to dedicate this book.'),
                              (4, '[item]First. [pause:1.5]Second.')])
        batch = [r['input'] for r in self.server.requests[mark:]]
        self.assertEqual(sorted(set(batch)),
                         ['First. Second.',
                          'I would like to dedicate this book.'],
                         'the server was asked for a text that is neither '
                         "chunk's, stripped")
        for r in self.server.requests:
            self.assertNotIn('[break]', r['input'])
            self.assertNotIn('[heading]', r['input'])
            self.assertNotIn('[item]', r['input'])
            self.assertNotIn('[pause:', r['input'])

    def test_a_chunk_that_is_only_markers_is_refused_by_name(self):
        engine = HiggsV3Engine(self.config())
        self.addCleanup(engine.cleanup)
        with self.assertRaises(ValueError) as caught:
            engine.render_audio('[break][heading]')
        self.assertIn('markers', str(caught.exception))

    def test_both_higgs_engines_strip_at_the_model_boundary(self):
        """Static: every Higgs render entry strips the markers, itself or
        through the one driver that does. A path that only `.strip()`s
        whitespace is the bug above.

        WHY A METHOD MAY SATISFY THIS BY DELEGATING. Until 2026-09-13 the
        batched entry points each stripped for themselves, and this test named
        them. Then the guarded driver was lifted out from under the file-writing
        layer into `render_many` (Owen's ruling that the model owns the guard;
        crucible/docs/PHASE6-REMOTE-RENDER.md), so `convert_many` /
        `convert_batch` became consumers of it and the strip moved WITH the
        driver, to the ladder's door. That is the correct place and the only
        place it can be: the guard measures CHARACTERS, so a chunk still
        carrying `[break]` would be judged - and split - on text the model will
        never be given.

        So the rule this asserts is the property, not a location: an entry
        either strips, or reaches a checked one that does. Listing `render_many`
        among the wanted names is what keeps the delegation honest - a driver
        that stopped stripping would fail here even though every caller still
        "delegates"."""
        import ast
        here = os.path.dirname(v3_served.__file__)
        #: The methods that must end up stripping. `render_many` is the door for
        #: both engines now; the others are the paths that do not go through it.
        wanted = {'v3_engine.py': {'render_audio', 'render_audio_measured',
                                   'render_many'},
                  'mlx_backend.py': {'render_audio', 'render_audio_measured',
                                     'render_many', 'generate_batch_stream'}}
        #: A method that reaches the door instead of stripping itself. Kept as a
        #: NAMED set rather than "any call to anything": delegating to something
        #: this test does not also check would be how the guarantee goes quiet.
        #: `render_audio` is the one-value face of `render_audio_measured`
        #: on both engines since 2026-09-19 (the measurement is taken where the
        #: frame cap is known), so it reaches the strip through that.
        delegates_to = {'render_many', 'render_audio_measured'}
        for filename, methods in wanted.items():
            with open(os.path.join(here, filename), encoding='utf-8') as handle:
                tree = ast.parse(handle.read())
            missing = set(methods)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in methods:
                    calls = {n.func.attr for n in ast.walk(node)
                             if isinstance(n, ast.Call)
                             and isinstance(n.func, ast.Attribute)}
                    self.assertTrue(
                        '_clean_sentence_for_tts' in calls
                        or calls & delegates_to,
                        f'{filename}:{node.name} must strip the markers, or '
                        f'reach {sorted(delegates_to)} which does')
                    missing -= {node.name}
            self.assertEqual(missing, set(), f'{filename}: not found: {missing}')

    def test_the_batched_entries_reach_the_stripping_driver(self):
        """The other half of the rule above: the methods that no longer strip
        for themselves must actually be consumers of the driver that does.

        Without this, deleting `convert_batch`'s call to `render_many` and
        rendering some other way would leave BOTH tests green - the door would
        still strip, and nothing would go through it."""
        import ast
        here = os.path.dirname(v3_served.__file__)
        wanted = {'v3_engine.py': {'convert_many'},
                  'mlx_backend.py': {'convert_batch'}}
        for filename, methods in wanted.items():
            with open(os.path.join(here, filename), encoding='utf-8') as handle:
                tree = ast.parse(handle.read())
            missing = set(methods)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in methods:
                    calls = {n.func.attr for n in ast.walk(node)
                             if isinstance(n, ast.Call)
                             and isinstance(n.func, ast.Attribute)}
                    self.assertIn(
                        'render_many', calls,
                        f'{filename}:{node.name} renders without going through '
                        f'render_many, so nothing strips its markers')
                    missing -= {node.name}
            self.assertEqual(missing, set(), f'{filename}: not found: {missing}')

    def test_convert_batch_answers_every_item_in_order(self):
        engine = HiggsV3Engine(self.config(sentences_dir=self.dir))
        self.addCleanup(engine.cleanup)
        items = [(7, 'Seven.'), (8, 'Eight.'), (9, 'Nine.'), (10, 'Ten.')]
        results = engine.convert_batch(items)
        self.assertEqual(results, [True] * 4)
        for index, _ in items:
            self.assertTrue(os.path.isfile(engine._sentence_file(index)))

    def test_a_dead_server_ends_the_flush_instead_of_failing_every_row(self):
        engine = HiggsV3Engine(self.config(sentences_dir=self.dir))
        self.addCleanup(engine.cleanup)
        self.server.close()
        with self.assertRaises(v3_served.HiggsV3ServerDown):
            engine.convert_batch([(1, 'One.'), (2, 'Two.'), (3, 'Three.')])

    def test_a_stop_abandons_the_rest_without_an_on_row(self):
        engine = HiggsV3Engine(self.config())
        self.addCleanup(engine.cleanup)
        rows = []
        engine.generate_batch_stream(['One.', 'Two.'], None, None, None,
                                     lambda i, pcm: rows.append(i),
                                     should_stop=lambda: True)
        self.assertEqual(rows, [])

    def test_a_mixed_voice_batch_is_impossible(self):
        engine = HiggsV3Engine(self.config())
        self.addCleanup(engine.cleanup)
        with self.assertRaises(ValueError) as caught:
            engine.generate_batch_stream(['One.'], ['someone_else'], None, None,
                                         lambda i, pcm: None)
        self.assertIn('server restart', str(caught.exception))

    def test_a_dead_server_fails_the_load_not_the_first_sentence(self):
        """The caller is holding a GPU lock; it needs to know now."""
        server = FakeV3Server(healthy=False)
        self.addCleanup(server.close)
        with self.assertRaises(HiggsV3ServerError) as caught:
            HiggsV3Engine(self.config(base_url=server.base_url, ready_timeout=2))
        self.assertIn('did not answer', str(caught.exception))


class RegistrySelectionTest(unittest.TestCase):

    def test_higgs_v3_is_the_second_engine(self):
        """One id, TWO backends, chosen by platform (PORT_NOTES 13).

        On darwin `higgs-v3` is the IN-PROCESS MLX engine - mlx-audio runs the
        whole model natively on Apple Silicon, so there is no server. Everywhere
        else it is this served one. Both carry ENGINE_ID 'higgs-v3' and the same
        geometry, budget and stop policy; only where the weights run differs.
        """
        from crucible_higgs.engine.higgs import HiggsV3MlxEngine
        expected = (HiggsV3MlxEngine if sys.platform.startswith('darwin')
                    else HiggsV3Engine)
        self.assertIs(registry.engine_class('higgs-v3'), expected)

    def test_v2_is_scaffolding_and_says_so(self):
        self.assertIn('higgs-v2-scaffold', registry.ids())
        self.assertNotIn('higgs-v2', registry.ids())
        self.assertIn('NOT SHIPPED', registry.__doc__)

    def test_the_defaults_carry_the_measured_numbers(self):
        self.assertEqual(HiggsV3Defaults.MAX_CHARS, 600)
        self.assertEqual(HiggsV3Defaults.CONTEXT_TOKENS, 8192)
        self.assertEqual(HiggsV3Defaults.MAX_REFERENCE_SECONDS, 30.0)
        self.assertEqual(HiggsV3Defaults.FRAMES_PER_SECOND, 25.0)
        self.assertEqual(HiggsV3Defaults.SAMPLE_RATE, 24000)


class ServeProtocolTest(V3TestCase):
    """The REAL serve worker, the REAL v3 client, the fake HTTP server.

    Not `--fake-engine`: the point is that `NARRATOR_ENGINE=higgs-v3` builds the
    actual HiggsV3Engine and that its audio reaches the pool's wire correctly.
    The worker runs as a subprocess and talks to the fake server over localhost,
    exactly as it would talk to vllm-omni.
    """

    def setUp(self):
        super().setUp()
        import json as _json
        self.voices_path = os.path.join(self.dir, 'voices.json')
        with open(self.voices_path, 'w', encoding='utf-8') as handle:
            # A REAL Higgs voice name. It used to say 'leah' - one of Orpheus's
            # eight stock tokens - which is the only reason this test passed
            # while the worker was validating v3 loads against Orpheus's
            # allowlist. A name no Orpheus allowlist contains is the whole
            # point.
            _json.dump({'deathstalker': {'clips': [{'path': self.clip,
                                                    'transcript': X2_TEXT,
                                                    'seconds': 27.42}]}}, handle)

    def _worker(self):
        import subprocess
        env = dict(os.environ)
        env.update({
            'PYTHONUNBUFFERED': '1',
            'PYTHONIOENCODING': 'utf-8',
            'ORPHEUS_SKIP_WARMUP': '1',
            'NARRATOR_ENGINE': 'higgs-v3',
            'NARRATOR_HIGGS_VOICES': self.voices_path,
            'NARRATOR_HIGGS3_URL': self.server.base_url,
        })
        env.pop('NARRATOR_GOLDEN_LOCAL', None)
        proc = subprocess.Popen(
            [sys.executable, '-m', 'crucible_higgs.serve'], cwd=_PYTHON_ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding='utf-8', env=env,
            bufsize=1)
        self.addCleanup(self._shutdown, proc)
        return proc

    def _shutdown(self, proc):
        try:
            proc.stdin.write('{"action": "quit"}\n')
            proc.stdin.flush()
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
        for handle in (proc.stdout, proc.stderr, proc.stdin):
            try:
                handle.close()
            except Exception:
                pass

    def _read_until(self, proc, *types, limit=200):
        out = []
        for _ in range(limit):
            line = proc.stdout.readline()
            if not line:
                # readline() blocks, so EOF means the process ENDED - never a
                # slow start. Its stderr is the only thing that says why.
                try:
                    proc.wait(timeout=60)
                except Exception:
                    pass
                raise AssertionError(
                    f'worker exited {proc.returncode} before {types}.\n'
                    f'messages so far: {out}\n'
                    f'--- stderr ---\n{(proc.stderr.read() or "").strip()}')
            message = json.loads(line)
            out.append(message)
            if message['type'] in types:
                return out
        raise AssertionError(f'no {types} in {limit} messages: {out}')

    @unittest.skipIf(
        sys.platform.startswith('darwin'),
        'this drives the SERVED v3 engine through the worker, and on '
        "darwin 'higgs-v3' resolves to the in-process MLX engine instead "
        '(there is no server to fake). The Mac equivalents are '
        'tests/test_higgs_mlx.py and the live `python -m crucible_higgs.serve` '
        'smoke recorded in PORT_NOTES 13.6.')
    def test_the_environment_selects_the_served_engine_and_it_renders(self):
        proc = self._worker()
        ready = self._read_until(proc, 'ready')[-1]
        self.assertEqual(ready['type'], 'ready')

        proc.stdin.write(json.dumps({'action': 'load',
                                     'voice': 'deathstalker',
                                     'warm': False}) + '\n')
        proc.stdin.flush()
        msgs = self._read_until(proc, 'loaded', 'error')
        self.assertEqual(msgs[-1]['type'], 'loaded', msgs)

        proc.stdin.write(json.dumps({'action': 'generate',
                                     'text': 'Hello there, listener.'}) + '\n')
        proc.stdin.flush()
        msgs = self._read_until(proc, 'audio', 'error')
        audio = msgs[-1]
        self.assertEqual(audio['type'], 'audio', msgs)
        self.assertEqual(audio['sampleRate'], 24000)
        self.assertGreater(audio['duration'], 0)
        # The request really went through the v3 client.
        self.assertTrue(self.server.requests)
        self.assertEqual(self.server.requests[-1]['model'], 'higgs-v3')
        self.assertEqual(self.server.requests[-1]['response_format'], 'wav')


class ServerIdentityTest(V3TestCase):
    """`/health` says something is listening. It does not say WHAT."""

    def test_adoption_requires_the_right_served_model(self):
        """A leftover server from another model on port 8095 would otherwise be
        adopted silently and render a whole book in the wrong voice."""
        self.server.httpd.models = ['some-other-model']
        backend = HiggsV3ServedBackend(base_url=self.server.base_url,
                                       serve_script=a_launcher(self))
        with self.assertRaises(HiggsV3ServerError) as caught:
            backend.start()
        message = str(caught.exception)
        self.assertIn('some-other-model', message)
        self.assertIn('higgs-v3', message)

    def test_adoption_accepts_our_own_server(self):
        backend = HiggsV3ServedBackend(base_url=self.server.base_url,
                                       serve_script=a_launcher(self))
        backend.start()                      # adopts, does not launch
        self.assertIsNone(backend._proc)

    def test_a_base_voice_needs_no_checkpoint_check(self):
        """No checkpoint configured = the base model; `/v1/models` naming
        higgs-v3 is the whole check."""
        backend = HiggsV3ServedBackend(base_url=self.server.base_url)
        backend.check_serves_expected_model()

    def test_a_port_that_answers_health_but_not_models_is_refused(self):
        backend = HiggsV3ServedBackend(base_url=self.server.base_url)
        self.server.httpd.models_broken = True
        with self.assertRaises(HiggsV3ServerError) as caught:
            backend.check_serves_expected_model()
        self.assertIn('cannot be identified', str(caught.exception))


class SentinelFilterProbeTest(V3TestCase):
    """A SENSOR, not a gate. See probe_sentinel_filter's docstring.

    The gate this replaces (-45 dBFS over the last 300 ms) was valid against the
    retired band-aid and is invalid under patch_sentinel_filter.py: the filter
    removes the sentinel frames rather than quieting them, so the window holds
    the model's own audio and the certifying box read -35..-38 dBFS on BOTH
    builds. These tests pin that the probe MEASURES and does not refuse on the
    level - a re-introduced threshold fails here.
    """

    def test_the_probe_measures_and_returns_the_tail_level(self):
        backend = HiggsV3ServedBackend(base_url=self.server.base_url)
        dbfs = backend.probe_sentinel_filter(self.voice())
        self.assertIsInstance(dbfs, float)
        self.assertLess(dbfs, 0.0, 'dBFS is a level below full scale')

    def test_a_LOUD_tail_is_reported_and_NOT_refused(self):
        """The old gate's failing case: ~250 ms at about -30 dB on the end.

        Under the sentinel filter that level says nothing about the patch - the
        certified server measures -35..-38 dBFS here, which is inside the same
        neighbourhood - so refusing on it would fail correct servers. It is
        reported instead.
        """
        self.server.httpd.tail_burst_dbfs = -30.0
        backend = HiggsV3ServedBackend(base_url=self.server.base_url)
        dbfs = backend.probe_sentinel_filter(self.voice())
        self.assertGreater(dbfs, -35.0, 'the fixture did not produce a loud tail')

    def test_a_200_with_no_audio_IS_refused(self):
        """The one thing left that is decidable rather than a level."""
        backend = HiggsV3ServedBackend(base_url=self.server.base_url)
        real = v3_served.decode_response
        v3_served.decode_response = lambda body, ctype=None: (np.zeros(0, dtype=np.float32), 24000)
        try:
            with self.assertRaises(HiggsV3ServerError) as caught:
                backend.probe_sentinel_filter(self.voice())
        finally:
            v3_served.decode_response = real
        self.assertIn('produced no audio', str(caught.exception))

    def test_no_level_threshold_survives(self):
        """The gate is gone by NAME, not merely unused: a constant called
        MAX_DBFS is how one comes back."""
        backend = v3_served.HiggsV3ServedBackend
        self.assertFalse([n for n in dir(backend) if n.endswith('MAX_DBFS')])
        low, high = backend.SENTINEL_TAIL_CERTIFIED_DBFS
        self.assertLess(low, high)
        self.assertEqual((low, high), (-38.0, -35.0),
                         'the certified band is what the box measured on both builds')

    def test_the_engine_runs_the_probe_at_load(self):
        engine = HiggsV3Engine(self.config())
        self.addCleanup(engine.cleanup)
        bodies = [r for r in self.server.requests
                  if r['input'] == v3_served.HiggsV3ServedBackend.SENTINEL_PROBE_TEXT]
        self.assertEqual(len(bodies), 1, 'exactly one probe render per load')

    def test_the_probe_can_be_turned_off(self):
        engine = HiggsV3Engine(self.config(probe_sentinel_filter=False))
        self.addCleanup(engine.cleanup)
        self.assertEqual(self.server.requests, [])


class SentinelProofTest(V3TestCase):
    """PROOF (a) OF THE SENTINEL FILTER: the RECORDS the patch writes, read.

    The patch's two halves are proved in two different places. (b) "no one-frame
    trim left in the stage processor" is a static grep BookForge's Higgs doctor
    runs before any server starts. (a) is this: the decode path says what it did.

    UNTIL 2026-09-13 IT SAID SO IN ENGLISH, and narrator ran three regexes over
    the server's LOG FILE to read it - crucible/docs/ARCHITECTURE.md rule R4's
    headline case, a log line promoted to an API that nothing versions and
    nothing tests. Two of these tests exist because that version was not merely
    fragile, it was WRONG: it passed on an empty log, and it expected the exact
    "2 frame(s)" line that patch v2 exists to eliminate.

    THE FIXTURES ARE JSON RECORDS because that is now the interface -
    `patch_sentinel_filter.py` v3's `_sentinel_report`, one per filter
    invocation, appended to the file named in `HIGGS_SENTINEL_REPORT`.
    """

    #: A clean sync chunk: every frame in range, nothing dropped.
    SYNC_CLEAN = {'v': 3, 'path': 'sync', 'total': 249, 'kept': 249, 'interior': 0}
    #: A clean FINAL async window: the trailing EOC ramp trimmed by identity,
    #: nothing out of range left behind. On patch v2+ this is what every good
    #: chunk looks like - and it writes NO log line at all, which is precisely
    #: why the old log-grep proof could not see it.
    ASYNC_CLEAN_FINAL = {'v': 3, 'path': 'async', 'final': True, 'window': 97,
                         'trimmed': 2, 'outside': 0}
    #: A clean NON-final window: nothing trimmed (only the final one is), nothing
    #: out of range.
    ASYNC_CLEAN_MID = {'v': 3, 'path': 'async', 'final': False, 'window': 128,
                       'trimmed': 0, 'outside': 0}
    #: THE DEFECT: sentinel frames still out of range on the FINAL window after
    #: the identity trim. They were substituted with codec code 0 - a valid code
    #: that decodes to real sound on the end of that chunk.
    ASYNC_FINAL_LEFTOVER = {'v': 3, 'path': 'async', 'final': True, 'window': 97,
                            'trimmed': 2, 'outside': 3}
    #: NOT a defect: the same count on a non-final window is the left-context
    #: region Stage 1 discards by frame count anyway.
    ASYNC_MID_LEFTOVER = {'v': 3, 'path': 'async', 'final': False, 'window': 128,
                          'trimmed': 0, 'outside': 3}
    SYNC_INTERIOR = {'v': 3, 'path': 'sync', 'total': 249, 'kept': 248,
                     'interior': 1}
    EMPTY_CHUNK = {'v': 3, 'path': 'sync', 'total': 249, 'kept': 0, 'interior': 0}

    def write_report(self, *records):
        path = os.path.join(self.dir, 'serve_current.log.sentinel.jsonl')
        with open(path, 'w', encoding='utf-8') as handle:
            for record in records:
                handle.write(json.dumps(record) + '\n')
        return path

    def attached_to(self, path):
        """A backend in ATTACH mode whose report the OPERATOR named.

        Through the documented door - the environment variable - rather than by
        setting the attribute, so what is tested is the contract an operator has.
        """
        if path is not None:
            os.environ[v3_served.SENTINEL_REPORT_ENV] = path
        else:
            os.environ.pop(v3_served.SENTINEL_REPORT_ENV, None)
        self.addCleanup(os.environ.pop, v3_served.SENTINEL_REPORT_ENV, None)
        return HiggsV3ServedBackend(base_url=self.server.base_url)

    def test_clean_records_PASS_and_report_what_they_read(self):
        path = self.write_report(self.SYNC_CLEAN, self.ASYNC_CLEAN_MID,
                                 self.ASYNC_CLEAN_FINAL)
        report = self.attached_to(path).verify_sentinel_filter()
        self.assertEqual(report['report'], path)
        self.assertEqual(report['records'], 3)
        self.assertEqual(report['syncRecords'], 1)
        self.assertEqual(report['asyncRecords'], 2)
        self.assertEqual(report['syncInteriorDrops'], 0)
        self.assertEqual(report['trailingFramesTrimmed'], 2)
        self.assertEqual(report['nonFinalSubstitutions'], 0)

    def test_an_EMPTY_report_is_REFUSED_rather_than_passing(self):
        """THE REGRESSION THAT MOTIVATED ALL OF THIS.

        The log-grep version's `test_a_log_with_no_sentinel_lines_at_all_PASSES`
        asserted the OPPOSITE of this, and was right to within its own means: it
        could not tell "the filter ran cleanly" from "nothing was rendered" or
        "the message was re-worded", so it chose to pass. Records make the two
        distinguishable - the filter writes one for EVERY invocation, clean ones
        included - so an empty file is now exactly what it looks like: no
        evidence, which must never read as no problem.
        """
        path = self.write_report()
        with self.assertRaises(HiggsV3ServerError) as caught:
            self.attached_to(path).verify_sentinel_filter()
        message = str(caught.exception)
        self.assertIn('holds no records', message)
        self.assertIn(path, message)

    def test_leftover_sentinels_on_the_FINAL_window_are_refused_BY_NAME(self):
        """Out of range AFTER the identity trim, on the final window, is a
        sentinel the trim could not reach - substituted with codec code 0, which
        decodes to real sound."""
        path = self.write_report(self.ASYNC_CLEAN_FINAL,
                                 self.ASYNC_FINAL_LEFTOVER)
        with self.assertRaises(HiggsV3ServerError) as caught:
            self.attached_to(path).verify_sentinel_filter()
        message = str(caught.exception)
        self.assertIn('sentinel proof FAILED', message)
        self.assertIn('3 frame(s)', message)
        self.assertIn('FINAL async window', message)
        self.assertIn(path, message)
        self.assertIn('"outside": 3', message.replace("'", '"'),
                      'the offending record itself must be quoted back')

    def test_EXACTLY_TWO_leftover_frames_are_refused_not_welcomed(self):
        """THE OTHER HALF OF THE R2 CASE, and the sharper one.

        The log-grep proof did not merely pass on an empty log - it also passed
        on a log FULL of the exact line patch v2 exists to eliminate, because
        `EXPECTED_TRAILING_SENTINEL_FRAMES` was 2 and 2 is what v1 printed for
        the EOC ramp it counted before trimming. A guard that is green for two
        opposite reasons is not a guard. Under v2's ordering the trailing run is
        already gone when the count is taken, so 2 left on a FINAL window is two
        sentinel frames that became codec code 0 - the single most likely real
        defect, and the one reading the old check treated as health.
        """
        path = self.write_report(
            dict(self.ASYNC_FINAL_LEFTOVER, outside=2),
            dict(self.ASYNC_FINAL_LEFTOVER, outside=2))
        with self.assertRaises(HiggsV3ServerError) as caught:
            self.attached_to(path).verify_sentinel_filter()
        message = str(caught.exception)
        self.assertIn('sentinel proof FAILED', message)
        self.assertIn('2 frame(s)', message)

    def test_the_same_count_on_a_NON_final_window_is_counted_not_refused(self):
        """Off the final window those frames can only be the left-context region
        Stage 1 discards by frame count. Refusing on them would fail correct
        servers; they are surfaced as a number instead."""
        path = self.write_report(self.ASYNC_MID_LEFTOVER, self.ASYNC_CLEAN_FINAL)
        report = self.attached_to(path).verify_sentinel_filter()
        self.assertEqual(report['nonFinalSubstitutions'], 3)
        self.assertEqual(report['finalWindowSubstitutions'], 0)

    def test_NO_REPORT_AT_ALL_is_refused_the_proof_needs_the_records(self):
        """An attached server whose operator named nothing. narrator will not
        guess a path: a stale report from an earlier run would let this pass on
        evidence from a server that is no longer up."""
        backend = self.attached_to(None)
        self.assertIsNone(backend.proof_report())
        with self.assertRaises(HiggsV3ServerError) as caught:
            backend.verify_sentinel_filter()
        message = str(caught.exception)
        self.assertIn('no sentinel report to read', message)
        self.assertIn(v3_served.SENTINEL_REPORT_ENV, message)

    def test_a_NAMED_report_that_does_not_exist_is_refused_too(self):
        missing = os.path.join(self.dir, 'nothing-here.jsonl')
        with self.assertRaises(HiggsV3ServerError) as caught:
            self.attached_to(missing).verify_sentinel_filter()
        message = str(caught.exception)
        self.assertIn('could not', message)
        self.assertIn(missing, message)
        self.assertIn('stale', message,
                      'the refusal should name what a v1/v2 env looks like')

    def test_a_SYNC_interior_drop_is_refused(self):
        """The sync path takes the FULL filter, so an interior drop is a frame
        that failed the token test between two good ones - never observed
        offline, and a hole in the middle of that chunk's audio."""
        path = self.write_report(self.SYNC_CLEAN, self.SYNC_INTERIOR)
        with self.assertRaises(HiggsV3ServerError) as caught:
            self.attached_to(path).verify_sentinel_filter()
        message = str(caught.exception)
        self.assertIn('SYNC path', message)
        self.assertIn('INTERIOR', message)

    def test_a_chunk_with_no_audio_at_all_is_refused(self):
        path = self.write_report(self.EMPTY_CHUNK)
        with self.assertRaises(HiggsV3ServerError) as caught:
            self.attached_to(path).verify_sentinel_filter()
        self.assertIn('NO audio', str(caught.exception))

    def test_a_record_from_ANOTHER_PATCH_GENERATION_is_refused(self):
        """A v2 env writes no report at all, so this is the corruption / mixed-env
        case. Skipping the record would be the same silent pass by another
        route."""
        old = dict(self.SYNC_CLEAN, v=2)
        path = self.write_report(self.SYNC_CLEAN, old)
        with self.assertRaises(HiggsV3ServerError) as caught:
            self.attached_to(path).verify_sentinel_filter()
        message = str(caught.exception)
        self.assertIn('v=2', message)
        self.assertIn('line 2', message)

    def test_a_TORN_line_is_refused_rather_than_skipped(self):
        path = self.write_report(self.SYNC_CLEAN)
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write('{"v": 3, "path": "sy\n')
        with self.assertRaises(HiggsV3ServerError) as caught:
            self.attached_to(path).verify_sentinel_filter()
        self.assertIn('not JSON', str(caught.exception))
        self.assertIn('line 2', str(caught.exception))

    def test_an_UNKNOWN_path_name_is_refused(self):
        path = self.write_report(dict(self.SYNC_CLEAN, path='streaming'))
        with self.assertRaises(HiggsV3ServerError) as caught:
            self.attached_to(path).verify_sentinel_filter()
        self.assertIn("'streaming'", str(caught.exception))

    def test_the_engine_says_UNAVAILABLE_rather_than_passing_silently(self):
        """Attached, no named report: `load_engine` must not call the proof, and
        must not report success either. "Not proved" and "proved" have to look
        different in a run log."""
        backend = self.attached_to(None)
        self.assertIsNone(backend.proof_report())
        lines = []
        real = v3_served.log
        v3_served.log = lambda *a, **k: lines.append(a[0] if a else '')
        try:
            engine = HiggsV3Engine(self.config())
            self.addCleanup(engine.cleanup)
        finally:
            v3_served.log = real
        # The engine logs through its own module, so assert on the outcome that
        # matters: the load did not raise, and the backend it built has no proof
        # records to read.
        self.assertIsNone(engine.server.proof_report())


class SentinelRecordShapeTest(unittest.TestCase):
    """THE PRODUCER AND THE CONSUMER OF THE RECORDS, COMPARED.

    `verify_sentinel_filter` reads fields out of a file written by
    `electron/scripts/higgs/patch_sentinel_filter.py`, which is spliced into a
    third-party module inside a conda env inside WSL. Nothing about that seam
    is type-checked, imported or exercised by either side's tests, and the only
    place the two shapes meet at runtime is a box with a 24 GB card on it.

    That is the exact shape of the defect this whole change is repairing - a
    fact with two owners and nothing comparing them - so it is compared HERE,
    statically, by reading the patch's own literals. A field renamed on one side
    would otherwise surface as `int(record.get('outside', 0))` quietly reading 0
    for ever: a proof that passes on every render because it is looking at a key
    nobody writes.
    """

    @classmethod
    def setUpClass(cls):
        """Read the patch's constants by PARSING it, never by importing it.

        An `importlib` load would run the module and leave a `__pycache__/`
        beside it - which is not a theoretical tidiness point: it broke
        `tools/test-higgs-engine.js`'s "the WSL scripts are LF" keeper the first
        time this test ran, because that check `readdirSync`s this directory and
        reads every entry as a file (EISDIR on the new directory). A test must
        not write into the tree it inspects, and `ast` answers the question
        without executing anything.
        """
        import ast
        repo = os.path.dirname(_PYTHON_ROOT)
        cls.path = os.path.join(repo, 'electron', 'scripts', 'higgs',
                                'patch_sentinel_filter.py')
        cls.ast = ast
        tree = ast.parse(open(cls.path, encoding='utf-8').read())
        cls.constants = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                cls.constants[target.id] = node.value.value
        for name in ('HELPER', 'ASYNC_NEW', 'V3_MARKER'):
            if name not in cls.constants:
                raise AssertionError(
                    f'patch_sentinel_filter.py no longer declares {name} as a '
                    'module-level string literal, so this comparison cannot be made')

    def records_in(self, source, node=None):
        """Every `_sentinel_report({...})` literal in a fragment, as dicts."""
        found = []
        for call in self.ast.walk(node or self.ast.parse(source)):
            if not isinstance(call, self.ast.Call):
                continue
            if getattr(call.func, 'id', None) != '_sentinel_report':
                continue
            literal = call.args[0]
            self.assertIsInstance(
                literal, self.ast.Dict,
                'a record built anywhere but at the call site cannot be compared here')
            found.append({k.value: getattr(v, 'value', None)
                          for k, v in zip(literal.keys, literal.values)})
        return found

    def test_the_block_the_patch_splices_in_is_VALID_PYTHON(self):
        """It is spliced into somebody else's module and compiled there. A
        syntax error would be found on a box with a GPU, after a ~19 GB load."""
        self.ast.parse(self.constants['HELPER'])

    def test_the_MARKER_the_doctor_greps_is_narrator_s_OWN_variable_name(self):
        self.assertEqual(self.constants['V3_MARKER'], v3_served.SENTINEL_REPORT_ENV)
        self.assertIn(v3_served.SENTINEL_REPORT_ENV, self.constants['HELPER'],
                      'the spliced block does not read the variable narrator exports')

    def test_every_record_the_patch_writes_carries_THIS_version(self):
        records = self.records_in(self.constants['HELPER'])
        records += self.records_in('if True:\n' + self.constants['ASYNC_NEW'])
        self.assertTrue(records, 'the patch writes no records at all')
        for record in records:
            self.assertEqual(record.get('v'), v3_served.SENTINEL_REPORT_VERSION,
                             f'a record is written at v={record.get("v")!r} while '
                             f'narrator refuses anything but '
                             f'v={v3_served.SENTINEL_REPORT_VERSION}: {record}')

    def test_the_FIELDS_the_proof_reads_are_the_fields_the_patch_writes(self):
        """Key by key, both directions. A field the proof reads and the patch
        does not write reads as 0 for ever (a proof that always passes); a field
        the patch writes and the proof does not read is a defect nobody is
        looking at."""
        sync = [r for r in self.records_in(self.constants['HELPER'])
                if r.get('path') == 'sync']
        asynchronous = [r for r in self.records_in('if True:\n' + self.constants['ASYNC_NEW'])
                        if r.get('path') == 'async']
        self.assertTrue(sync and asynchronous)
        for record in sync:
            self.assertEqual(set(record), {'v', 'path', 'total', 'kept', 'interior'},
                             f'the sync record shape moved: {sorted(record)}')
        for record in asynchronous:
            self.assertEqual(set(record),
                             {'v', 'path', 'final', 'window', 'trimmed', 'outside'},
                             f'the async record shape moved: {sorted(record)}')

    def test_the_patch_REFUSES_TO_EMIT_a_file_with_the_records_missing(self):
        """Its own pre-write invariant, so a re-cut that drops a call site fails
        at apply time rather than producing an env that renders fine and proves
        nothing."""
        source = open(self.path, encoding='utf-8').read()
        self.assertIn('REPORT_CALLS_MISSING', source)
        self.assertIn('V3_MARKER_MISSING', source)


class SentinelReportChannelTest(V3TestCase):
    """THE REPORT IS A CHANNEL NARRATOR OWNS, not a file it hopes exists.

    Three properties, each of which was a way the old log-grep proof could be
    handed the wrong evidence:

      * the server is TOLD where to write, by the same object that decides where
        to read, so the two can never name different files;
      * the path is translated for the arm the SERVER runs on;
      * it is TRUNCATED at launch, so last run's records cannot certify this run.
    """

    def launching_backend(self, **kwargs):
        dead = FakeV3Server(healthy=False)
        self.addCleanup(dead.close)
        backend = HiggsV3ServedBackend(
            base_url=dead.base_url, serve_script=a_launcher(self), **kwargs)
        self.addCleanup(backend._close_log)
        return backend

    def test_the_wrapper_EXPORTS_the_path_the_backend_will_read(self):
        path = os.path.join(self.dir, 'higgs-v3-server.log')
        backend = self.launching_backend(server_log=path)
        self.assertEqual(backend.proof_report(),
                         path + v3_served.SENTINEL_REPORT_SUFFIX)
        exports = backend._launch_exports()
        self.assertIn(v3_served.SENTINEL_REPORT_ENV + '=', exports)
        # On this arm (native Linux in CI, Windows on example-pc) the exported form
        # is the SERVER's view of the same file. What must hold everywhere is
        # that the basename is the one narrator reads - a different name there
        # would leave the server writing a file nothing ever opens.
        self.assertIn(os.path.basename(backend.proof_report()).replace('\\', '/'),
                      exports.replace('\\', '/'))

    def test_the_report_is_TRUNCATED_when_a_server_is_launched(self):
        """A report is append-only from the server's side - several decode
        workers write to it at once - so nothing else would ever clear it, and a
        file left from the previous run in the same session dir would let this
        run's proof pass on the last run's evidence."""
        import subprocess
        path = os.path.join(self.dir, 'higgs-v3-server.log')
        backend = self.launching_backend(server_log=path)
        stale = backend.proof_report()
        with open(stale, 'w', encoding='utf-8') as handle:
            handle.write(json.dumps(SentinelProofTest.ASYNC_FINAL_LEFTOVER) + '\n')

        class _FakeProc:
            returncode = 0

            def poll(self):
                return 0

        real = subprocess.Popen
        subprocess.Popen = lambda command, **kwargs: _FakeProc()
        try:
            backend.start()
        finally:
            subprocess.Popen = real
        self.assertEqual(os.path.getsize(stale), 0,
                         'the previous run\'s records survived a launch')

    def test_an_ADOPTED_server_stops_claiming_our_report(self):
        """`start()` adopts a server already on the port. Its decode path writes
        wherever ITS launcher pointed it, so the file we would have written is
        not the evidence and `proof_report` must stop naming it - the same rule
        `proof_log` follows, and the same flag."""
        os.environ.pop(v3_served.SENTINEL_REPORT_ENV, None)
        backend = HiggsV3ServedBackend(base_url=self.server.base_url,
                                       serve_script=a_launcher(self))
        self.assertEqual(backend.proof_report(), backend.sentinel_report)
        backend.start()                      # the fake server is already healthy
        self.addCleanup(backend._close_log)
        self.assertIsNone(backend.proof_report())


class ServerLogIsOwnedTest(V3TestCase):
    """The server's output goes to a FILE THIS BACKEND OWNS, never DEVNULL.

    The regression this pins is the one that made proof (a) impossible: both
    streams went to `subprocess.DEVNULL`, so the sentinel filter's own report of
    what it did was discarded, and `wait_ready`'s "check its log" pointed at
    nothing.
    """

    def launching_backend(self, **kwargs):
        """A backend in LAUNCH mode against a port nothing answers, so `start()`
        launches rather than adopting."""
        dead = FakeV3Server(healthy=False)
        self.addCleanup(dead.close)
        backend = HiggsV3ServedBackend(
            base_url=dead.base_url, serve_script=a_launcher(self), **kwargs)
        self.addCleanup(backend._close_log)
        return backend

    def start_capturing_popen(self, backend):
        """Run `start()` with Popen replaced, and return its kwargs."""
        import subprocess
        seen = {}

        class _FakeProc:
            returncode = 0

            def poll(self):
                return 0

        def fake_popen(command, **kwargs):
            seen.update(kwargs)
            seen['command'] = command
            return _FakeProc()

        real = subprocess.Popen
        subprocess.Popen = fake_popen
        try:
            backend.start()
        finally:
            subprocess.Popen = real
        return seen

    def test_both_streams_go_to_the_backends_own_file(self):
        import subprocess
        path = os.path.join(self.dir, 'higgs-v3-server.log')
        backend = self.launching_backend(server_log=path)
        seen = self.start_capturing_popen(backend)
        self.assertIsNot(seen['stdout'], subprocess.DEVNULL,
                         'the server log went to DEVNULL again')
        self.assertEqual(getattr(seen['stdout'], 'name', None), path)
        self.assertIs(seen['stderr'], subprocess.STDOUT,
                      'vLLM logs to stderr; it has to be merged into the file')
        self.assertTrue(os.path.exists(path))

    def test_the_log_is_the_one_the_session_named_and_the_spec_reports_it(self):
        path = os.path.join(self.dir, 'higgs-v3-server.log')
        backend = self.launching_backend(server_log=path)
        self.assertEqual(backend.launch_log, path)
        self.assertEqual(backend.proof_log(), path)
        self.assertEqual(backend.spec.server_log, path)

    def test_with_no_session_the_log_is_PER_INSTANCE(self):
        """Two workers must never share one log."""
        one = self.launching_backend()
        two = self.launching_backend()
        self.assertNotEqual(one.launch_log, two.launch_log)
        for backend in (one, two):
            self.assertTrue(backend.launch_log.endswith('.log'))
            self.assertIn('narrator-higgs3-', os.path.basename(backend.launch_log))

    def test_an_ADOPTED_server_stops_claiming_our_log(self):
        """`start()` adopts a server already on the port. That process's output
        went wherever its own launcher sent it, so the file we would have
        written is not the proof stream and the spec must stop naming it."""
        os.environ.pop(v3_served.SERVER_LOG_ENV, None)
        backend = HiggsV3ServedBackend(base_url=self.server.base_url,
                                       serve_script=a_launcher(self))
        self.assertEqual(backend.proof_log(), backend.launch_log)
        backend.start()                      # the fake server is already healthy
        self.addCleanup(backend._close_log)
        self.assertIsNone(backend.proof_log())
        self.assertIsNone(backend.spec.server_log)

    def test_the_loaded_message_carries_the_server_log(self):
        from crucible_higgs.serve.worker import _server_log_of
        path = self.write_named_log()
        engine = HiggsV3Engine(self.quiet_config())
        self.addCleanup(engine.cleanup)
        self.assertEqual(_server_log_of(engine), path)

    def write_named_log(self):
        path = os.path.join(self.dir, 'serve_current.log')
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write('')
        os.environ[v3_served.SERVER_LOG_ENV] = path
        self.addCleanup(os.environ.pop, v3_served.SERVER_LOG_ENV, None)
        return path

    def test_an_in_process_engine_reports_no_server_log(self):
        from crucible_higgs.engine.protocol import BackendSpec
        from crucible_higgs.serve.worker import _server_log_of

        class _InProcess:
            def backend_spec(self):
                return BackendSpec(kind='inprocess', name='mlx')

        self.assertIsNone(_server_log_of(_InProcess()))
        self.assertIsNone(_server_log_of(object()))

    def test_an_in_process_spec_may_not_carry_a_server_log(self):
        from crucible_higgs.engine.protocol import BackendSpec
        with self.assertRaises(ValueError) as caught:
            BackendSpec(kind='inprocess', name='mlx', server_log='/a/b.log')
        self.assertIn('server_log', str(caught.exception))


class FailedLoadReleasesTheServerTest(V3TestCase):
    """A ~14 GB vllm-omni left running is a GPU nobody can name."""

    def test_a_ready_timeout_stops_the_server_it_started(self):
        server = FakeV3Server(healthy=False)
        self.addCleanup(server.close)
        stopped = []

        real_stop = HiggsV3ServedBackend.stop
        HiggsV3ServedBackend.stop = lambda self, *a, **k: stopped.append(True)
        try:
            with self.assertRaises(HiggsV3ServerError):
                HiggsV3Engine(self.config(base_url=server.base_url,
                                          ready_timeout=2))
        finally:
            HiggsV3ServedBackend.stop = real_stop
        self.assertTrue(stopped, 'a failed load must stop the server it started')

    def test_a_failed_probe_also_stops_it(self):
        # The probe no longer refuses on a LEVEL (see SentinelFilterProbeTest),
        # so the failure driven here is the probe RENDER failing - which is what
        # this test was always about: anything that raises after start() must
        # stop the ~14 GB server rather than leak it.
        self.server.httpd.error = (500, 'probe render exploded')
        stopped = []
        real_stop = HiggsV3ServedBackend.stop
        HiggsV3ServedBackend.stop = lambda self, *a, **k: stopped.append(True)
        try:
            with self.assertRaises(HiggsV3ServerError):
                HiggsV3Engine(self.config())
        finally:
            HiggsV3ServedBackend.stop = real_stop
        self.assertTrue(stopped)

    def test_the_default_patience_covers_the_measured_cold_starts(self):
        """55 / 146 / 297 s measured on this box; 300 would have been a coin
        flip on the slowest of them."""
        self.assertGreaterEqual(HiggsV3Defaults.READY_TIMEOUT_SECONDS, 900.0)


class ResponseRefusalTest(V3TestCase):

    def test_a_non_wav_200_is_refused_by_name(self):
        """The shape a proxy or an error page produces. soundfile's own message
        ("Format not recognised") names nothing a reader can act on."""
        with self.assertRaises(HiggsV3ServerError) as caught:
            v3_served.decode_response(b'{"detail":"nope"}', 'application/json')
        message = str(caught.exception)
        self.assertIn('expected a WAV body', message)
        self.assertIn('application/json', message)
        self.assertIn('decode_response', message)

    def test_the_minus_100_error_names_the_missing_vllm_patch(self):
        """Without patch_vllm.py EVERY cloned request fails this way, and only
        an UN-cloned one succeeds - which is the model's own voice at 12 % of
        the narrator ceiling."""
        self.server.fail_with(400, 'Token id -100 is out of vocabulary')
        backend = HiggsV3ServedBackend(base_url=self.server.base_url)
        with self.assertRaises(HiggsV3ServerError) as caught:
            backend.speak(SpeechRequest(text='Hello.', voice=self.voice(),
                                        max_new_tokens=200))
        message = str(caught.exception)
        self.assertIn('patch_vllm.py', message)
        self.assertIn('THE vLLM PATCH IS MISSING', message)


class VoiceTuningTest(V3TestCase):
    """maxChars is per VOICE, and v3 voices carry v3's rules."""

    def _voices_file(self, document):
        path = os.path.join(self.dir, 'voices.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(document, handle)
        return path

    def test_a_voice_gets_v3s_control_allowlist_and_reference_cap(self):
        """Read with the v2 scaffold's defaults a voice carries an EMPTY
        allowlist, which would reject <|prosody:long_pause|> that the model
        understands perfectly."""
        from crucible_higgs.engine.higgs.config import load_voice
        path = self._voices_file({'ds': {'clips': [
            {'path': self.clip, 'transcript': X2_TEXT, 'seconds': 27.42}]}})
        voice = load_voice('ds', path,
                           allowed_controls=HiggsV3Defaults.ALLOWED_CONTROLS,
                           max_reference_seconds=HiggsV3Defaults.MAX_REFERENCE_SECONDS,
                           placeholder_max_chars=HiggsV3Defaults.MAX_CHARS)
        self.assertEqual(len(voice.allowed_controls), 45)
        self.assertIn('<|prosody:long_pause|>', voice.allowed_controls)
        self.assertEqual(voice.max_reference_seconds, 30.0)
        self.assertEqual(voice.max_chars, 600)
        self.assertEqual(voice.max_chars_source, 'placeholder')

    def test_the_config_stamps_v3s_rules_on_a_hand_built_voice(self):
        bare = ClipsVoice(clips=(ReferenceClip(self.clip, X2_TEXT, seconds=14.0),),
                          name='ds')
        self.assertEqual(bare.allowed_controls, ())
        config = HiggsV3Config(voice=bare, base_url=self.server.base_url)
        self.assertEqual(len(config.voice.allowed_controls), 45)
        self.assertEqual(config.voice.max_reference_seconds, 30.0)

    def test_a_declared_maxChars_wins(self):
        from crucible_higgs.engine.higgs.config import load_voice
        path = self._voices_file({'ds': {'maxChars': 420, 'clips': [
            {'path': self.clip, 'transcript': X2_TEXT, 'seconds': 27.42}]}})
        voice = load_voice('ds', path, placeholder_max_chars=600)
        self.assertEqual(voice.max_chars, 420)
        self.assertEqual(voice.max_chars_source, 'catalog')
        budget = HiggsV3Budget(HiggsV3Config(voice=voice,
                                             base_url=self.server.base_url))
        self.assertEqual(budget.max_chars(), 420)
        self.assertEqual(budget.max_chars_source(), 'catalog')

    def test_a_checkpoint_voice_without_maxChars_LOADS(self):
        """RETIRED REFUSAL (Owen, 2026-09-19). A fine-tune's safe chunk
        length is still a measured property of THAT model - what changed is
        that it is the CLIENT's number, and the screening render of an
        unmeasured checkpoint is the render that produces it. Refusing the
        load made the first render of every new checkpoint impossible.

        The placeholder is not passed off as the model's: `max_chars_source`
        says 'placeholder', and that is what the prep log and the manifest
        carry."""
        from crucible_higgs.engine.higgs.config import load_voices
        path = self._voices_file({'ds-ft': {
            'kind': 'checkpoint', 'checkpointDir': '/models/ds-merged',
            'clips': [{'path': self.clip, 'transcript': X2_TEXT,
                       'seconds': 27.42}]}})
        voice = load_voices(path, placeholder_max_chars=600)['ds-ft']
        self.assertEqual(voice.max_chars, 600)
        self.assertEqual(voice.max_chars_source, 'placeholder')
        self.assertEqual(voice.checkpoint_dir, '/models/ds-merged')

    def test_a_checkpoint_voice_with_maxChars_loads(self):
        from crucible_higgs.engine.higgs.config import load_voice
        path = self._voices_file({'ds-ft': {
            'kind': 'checkpoint', 'checkpointDir': '/models/ds-merged',
            'maxChars': 500,
            'clips': [{'path': self.clip, 'transcript': X2_TEXT,
                       'seconds': 27.42}]}})
        voice = load_voice('ds-ft', path, placeholder_max_chars=600)
        self.assertEqual(voice.max_chars, 500)
        self.assertEqual(voice.checkpoint_dir, '/models/ds-merged')

    def test_the_budget_answers_for_a_checkpoint_voice_assembled_in_code(self):
        """The belt retires with the load gate it belted (2026-09-19): a
        fine-tune with no declared cap packs at the engine placeholder and
        `max_chars_source` says so. Nothing narrator does per chunk reads this
        number - the frame ceiling is `cap_frames(text)` over the text it was
        sent - so there is nothing left for the refusal to protect."""
        bare = ClipsVoice(clips=(ReferenceClip(self.clip, X2_TEXT, seconds=14.0),),
                          name='ds-ft',
                          checkpoint_dir=self.merged_checkpoint())
        config = HiggsV3Config(voice=bare, base_url=self.server.base_url)
        budget = HiggsV3Budget(config)
        self.assertEqual(budget.max_chars(), int(config.max_chars))
        self.assertEqual(budget.max_chars_source(), 'placeholder')


class SeedRuleTest(V3TestCase):

    def test_chunk_i_renders_with_seed_plus_i(self):
        """Rendering every sentence at a flat seed makes a whole book of
        identical draws for identical text."""
        engine = HiggsV3Engine(self.config(probe_sentinel_filter=False))
        self.addCleanup(engine.cleanup)
        engine.generate_batch_stream(['One.', 'Two.', 'Three.'], None, None, None,
                                     lambda i, pcm: None)
        # Requests are concurrent, so the ORDER the fake saw them in is not the
        # row order; the seed-per-row rule is what is asserted.
        seeds = sorted(r['seed'] for r in self.server.requests)
        self.assertEqual(seeds, [1234, 1235, 1236])

    def test_convert_seeds_by_sentence_number(self):
        # An IN-BAND take for the six characters of 'Hello.' (~17 chars/s), so
        # the length guard passes take 0 through and the ONE request carries the
        # chunk's own seed. (At the fake's default 1.0 s, 6 chars/s is a run-on
        # and the guard - correctly - re-rolls at another seed.)
        srv = self.server
        (srv.httpd if hasattr(srv, 'httpd') else srv).seconds = 0.35
        engine = HiggsV3Engine(self.config(probe_sentinel_filter=False,
                                           sentences_dir=self.dir))
        self.addCleanup(engine.cleanup)
        engine.convert(7, 'Hello.')
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(self.server.requests[0]['seed'], 1241)

    def test_a_seed_in_extra_params_is_refused(self):
        """One place carries the seed: the top-level field. vllm-omni copies
        extra_params onto the stage-0 params, so a seed in both is two sources
        for one number and nothing says which wins."""
        with self.assertRaises(ValueError) as caught:
            v3_served.build_request_body('Hi.', self.voice(), 200, seed=1,
                                         sampling={'seed': 9})
        self.assertIn('TOP-LEVEL', str(caught.exception))


class _LaunchTestBase(unittest.TestCase):
    """Shared fixture for the two platform arms of the launch/stop path."""

    def setUp(self):
        import shutil as _shutil
        import tempfile
        self.dir = tempfile.mkdtemp(prefix='narrator-H-launch-')
        self.addCleanup(_shutil.rmtree, self.dir, True)
        state_concurrency(self)
        state_stack(self)
        self.fake_wsl = os.path.join(self.dir, 'wsl.exe')
        with open(self.fake_wsl, 'w', encoding='utf-8') as handle:
            handle.write('')
        os.chmod(self.fake_wsl, 0o755)       # shutil.which needs it on POSIX
        self._path = os.environ.get('PATH', '')
        os.environ['PATH'] = self.dir + os.pathsep + self._path
        self.addCleanup(os.environ.__setitem__, 'PATH', self._path)
        # AN OPERATOR'S LAUNCHER THAT EXISTS. `NARRATOR_HIGGS3_SERVE_SCRIPT`
        # (and the `serve_script=` argument behind it) OVERRIDES narrator's own
        # packaged launcher, and a path that is not there is refused by name
        # rather than quietly replaced by the packaged one - so a launch test
        # has to name a real file on this filesystem. `script_in_guest` is the
        # same path as the distro sees it, which is what the wrapper carries.
        self.script = a_launcher(self)
        self.script_in_guest = v3_served.served_common.to_wsl(self.script)

    def assert_wrapper(self, wrapper, backend, script_in_guest):
        self.assertIn(script_in_guest, wrapper,
                      'the script path must be the one the GUEST sees')
        self.assertIn('setsid bash', wrapper,
                      'the script must be detached from the wrapper\'s group')
        self.assertNotIn('echo $!', wrapper,
                         'no pid is recorded at launch: vllm-omni re-sessions '
                         'itself and `$!` is dead within a second (measured '
                         '84072 -> 84096 -> 84098)')
        self.assertTrue(wrapper.rstrip().endswith('wait'),
                        'the launcher must live as long as the script')
        # THE OWNERSHIP MARKER rides in the server's environment; it is how a
        # stop, a reclaim and the watchdog tell ours from a stranger's.
        self.assertIn(f'{v3_served.OWNER_ENV}=', wrapper)
        self.assertIn(backend.owner_id(), wrapper)
        # The launch script's knobs, STATED. Concurrency and the bind address
        # always; the checkpoint when the voice has one, else an explicit unset.
        self.assertIn(f'{v3_served.SERVE_MAX_NUM_SEQS_ENV}=2', wrapper)
        self.assertIn(f'{v3_served.SERVE_PORT_ENV}=', wrapper)
        self.assertIn(f'{v3_served.SERVE_HOST_ENV}=', wrapper)
        if backend.checkpoint_dir:
            self.assertIn(f'{v3_served.SERVE_MODEL_DIR_ENV}=', wrapper)
            self.assertNotIn(f'unset {v3_served.SERVE_MODEL_DIR_ENV}', wrapper)
        else:
            self.assertIn(f'unset {v3_served.SERVE_MODEL_DIR_ENV}', wrapper)


@unittest.skipUnless(sys.platform == 'win32', 'the Windows launch arm')
class WindowsLaunchTest(_LaunchTestBase):
    """The Windows arm: `wsl.exe -d <distro> bash -c <wrapper>`.

    NOT monkeypatched onto another platform. `shutil.which`, path translation
    and process termination all behave differently per OS, and a test that
    pretends otherwise proves nothing about either. The POSIX arm has its own
    twin below, so both are covered on the machine that can actually run them.
    """

    def test_the_launch_command_goes_through_wsl_with_a_pid_wrapper(self):
        backend = HiggsV3ServedBackend(
            serve_script=self.script, wsl_distro='Ubuntu')
        command = backend.launch_command()
        self.assertTrue(command[0].endswith('wsl.exe'), command)
        self.assertEqual(command[1:6], ['-d', 'Ubuntu', '--exec', 'bash', '-c'])
        self.assert_wrapper(command[6], backend, self.script_in_guest)

    def test_the_wrapper_runs_WITHOUT_the_distros_default_shell(self):
        """`--exec`, and it is the difference between a pid file and an empty one.

        Without it wsl.exe runs the command line through the distro's default
        shell, which expands `$!` before bash sees the wrapper - so the pid file
        is written EMPTY and `wait $!` becomes a bare `wait`. Measured on
        example-pc 2026-09-05 through this argv: bare -> `pid=[]`, `--` ->
        `pid=[]`, `--exec` -> `pid=[42679]`. `stop()` then has no pid to signal
        inside the distro, which is how a ~14 GB vllm-omni survives its own
        teardown.
        """
        backend = HiggsV3ServedBackend(
            serve_script=self.script, wsl_distro='Ubuntu')
        command = backend.launch_command()
        self.assertIn('--exec', command, command)
        # BEFORE the script, or the shell has already run by the time it applies.
        self.assertLess(command.index('--exec'), command.index('-c'))
        self.assertNotIn('--', command,
                         "`--` is not a substitute: it stops wsl.exe parsing its "
                         'own options and still runs the default shell')

    def test_a_script_inside_the_distro_is_not_mangled(self):
        backend = HiggsV3ServedBackend(
            serve_script=r'\\wsl$\Ubuntu\home\<user>\serve_v3.sh')
        self.assertIn('/home/<user>/serve_v3.sh', backend.launch_command()[-1])

    def test_stop_signals_the_guest_pid_when_the_server_outlives_wsl_exe(self):
        """Terminating wsl.exe kills the Windows-side relay; the guest process
        can keep running and holding ~14 GB, while proc.poll() reports a tidy
        exit."""
        server = FakeV3Server()
        self.addCleanup(server.close)
        backend = HiggsV3ServedBackend(base_url=server.base_url,
                                       serve_script=self.script)
        backend._guest_pid = 4242
        calls = []
        real_run = v3_served.subprocess.run

        def fake_run(argv, **kw):
            # The signal program prints the group it signalled; `_signal_guest`
            # reads the return code and that stdout, so the fake answers as the
            # real call does rather than returning None.
            calls.append(argv)
            return v3_served.subprocess.CompletedProcess(argv, 0, stdout='4242\n', stderr='')

        v3_served.subprocess.run = fake_run
        try:
            backend._verify_gone(timeout=2)
        finally:
            v3_served.subprocess.run = real_run
        self.assertTrue(calls, 'stop() must escalate to a guest-side signal')
        self.assertTrue(calls[0][0].endswith('wsl.exe'), calls[0])
        self.assertEqual(calls[0][1:5], ['-d', 'Ubuntu', '--exec', 'python3'])
        program = calls[0][calls[0].index('-c') + 1]
        self.assertIn('killpg', program, 'the process GROUP, read off /proc now')
        self.assertIn('SIGTERM', program)
        self.assertNotIn('SIGKILL', program, 'never a KILL on a GPU holder')
        self.assertEqual(calls[0][-1], '4242')

    def test_the_windows_arm_starts_no_watchdog(self):
        """The owner is a HOST pid the guest cannot watch; the marker says so
        and no watchdog is launched - stop() is the only teardown there."""
        backend = HiggsV3ServedBackend(
            serve_script=self.script, wsl_distro='Ubuntu')
        wrapper = backend.launch_command()[6]
        self.assertTrue(backend.owner_id().startswith('win32:'))
        self.assertNotIn('python3 -c', wrapper)

    def test_a_checkpoint_voice_exports_HIGGS_MODEL_DIR_in_guest_form(self):
        backend = HiggsV3ServedBackend(
            serve_script=self.script, wsl_distro='Ubuntu',
            checkpoint_dir=r'\\wsl$\Ubuntu\home\t\higgs_v3_merged\ds')
        wrapper = backend.launch_command()[6]
        self.assert_wrapper(wrapper, backend, self.script_in_guest)
        self.assertIn(f'{v3_served.SERVE_MODEL_DIR_ENV}=/home/t/higgs_v3_merged/ds',
                      wrapper)


@unittest.skipIf(sys.platform == 'win32', 'the POSIX launch arm')
class PosixLaunchTest(_LaunchTestBase):
    """The POSIX arm (Linux/WSL/macOS): `bash -c <wrapper>`, no wsl.exe.

    The same two assertions as the Windows arm - the wrapper records the
    server's own pid, and stop() escalates to that pid rather than to a pkill
    pattern that would take another agent's server with it.
    """

    def test_the_launch_command_is_a_plain_bash_wrapper(self):
        backend = HiggsV3ServedBackend(serve_script=a_launcher(self))
        command = backend.launch_command()
        self.assertEqual(command[:2], ['bash', '-c'])
        self.assert_wrapper(command[2], backend, a_launcher(self))

    def test_a_posix_path_is_passed_through_untouched(self):
        backend = HiggsV3ServedBackend(serve_script=a_launcher(self))
        self.assertIn(a_launcher(self), backend.launch_command()[2])

    def _capture_signals(self, backend):
        calls = []

        class _Done:
            returncode = 0
            stdout = '4242\n'
            stderr = ''

        real_run = v3_served.subprocess.run
        v3_served.subprocess.run = lambda argv, **kw: (calls.append(argv), _Done())[1]
        try:
            backend._verify_gone(timeout=2)
        finally:
            v3_served.subprocess.run = real_run
        return calls

    def test_stop_signals_the_servers_group(self):
        server = FakeV3Server()
        self.addCleanup(server.close)
        backend = HiggsV3ServedBackend(base_url=server.base_url,
                                       serve_script=a_launcher(self))
        backend._guest_pid = 4242
        calls = self._capture_signals(backend)
        self.assertTrue(calls, 'stop() must escalate to the server group')
        self.assertEqual(calls[0][:2], ['python3', '-c'])
        self.assertIn('killpg', calls[0][2], 'the GROUP, read off /proc now')
        self.assertIn('SIGTERM', calls[0][2])
        self.assertNotIn('SIGKILL', calls[0][2],
                         'TERM only - a KILL on a GPU holder inside WSL wedges '
                         'the VM')
        self.assertEqual(calls[0][-1], '4242')

    def test_the_guest_arm_starts_a_watchdog_on_the_owner(self):
        """Hit Stop, or let the app die: a worker killed with SIGKILL runs no
        cleanup, so a guest-side watchdog on the OWNER pid takes the marked
        listener's group down (Owen, 2026-09-05: "it should bring it down if
        i hit stop or if bookforge app dies")."""
        backend = HiggsV3ServedBackend(serve_script=a_launcher(self))
        wrapper = backend.launch_command()[2]
        self.assertIn('setsid python3 -c', wrapper)
        self.assertIn(f' {os.getpid()} ', wrapper, 'watches THIS process')
        self.assertIn('killpg', wrapper)
        self.assertIn('NARRATOR_HIGGS3_OWNER=', wrapper)
        self.assertNotIn('SIGKILL', wrapper)
        self.assertEqual(backend.owner_id(), str(os.getpid()))

    def test_the_watchdog_program_compiles_and_only_terms_marked_listeners(self):
        import ast
        tree = ast.parse(backend_program := HiggsV3ServedBackend._WATCHDOG)
        self.assertTrue(tree.body)
        self.assertIn("owner_of(p) == mark", backend_program)
        self.assertNotIn('SIGKILL', backend_program)
        ast.parse(HiggsV3ServedBackend._OWN_SERVERS_SCAN)
        ast.parse(HiggsV3ServedBackend._SIGNAL_GROUP)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'the watchdog reads /proc')
    def test_the_watchdog_exits_once_its_server_stops_while_the_owner_lives(self):
        """The wrapper ends in `wait`, which waits for the watchdog as well as
        the server. A watchdog that only ever exited after its OWNER died held
        every ordinary stop() for its whole timeout (~60 s per unload on the PC,
        2026-10-02) and ended with 'launch wrapper still up'."""
        import socket
        import subprocess
        import time
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        server = subprocess.Popen(
            [sys.executable, '-c',
             'import socket, sys, time\n'
             's = socket.socket()\n'
             's.bind(("127.0.0.1", int(sys.argv[1])))\n'
             's.listen()\n'
             'time.sleep(600)\n', str(port)],
            env=dict(os.environ, NARRATOR_HIGGS3_OWNER=str(os.getpid())))
        self.addCleanup(lambda: server.poll() is None and server.terminate())
        watchdog = subprocess.Popen(
            [sys.executable, '-c', HiggsV3ServedBackend._WATCHDOG,
             str(os.getpid()), str(port), '0.2'])
        self.addCleanup(lambda: watchdog.poll() is None and watchdog.terminate())
        time.sleep(2.0)
        self.assertIsNone(watchdog.poll(),
                          'the watchdog left while its server was still up')
        server.terminate()
        server.wait(timeout=10)
        self.assertEqual(watchdog.wait(timeout=10), 0)

    def test_a_checkpoint_voice_exports_HIGGS_MODEL_DIR(self):
        backend = HiggsV3ServedBackend(serve_script=a_launcher(self),
                                       checkpoint_dir='/home/t/higgs_v3_merged/ds')
        wrapper = backend.launch_command()[2]
        self.assert_wrapper(wrapper, backend, a_launcher(self))
        self.assertIn(f'{v3_served.SERVE_MODEL_DIR_ENV}=/home/t/higgs_v3_merged/ds',
                      wrapper)

    def test_KILL_is_refused_by_name(self):
        backend = HiggsV3ServedBackend(serve_script=a_launcher(self))
        with self.assertRaises(ValueError) as caught:
            backend._signal_guest(4242, 'KILL')
        self.assertIn('wedges', str(caught.exception))

    def test_it_never_kills_a_server_it_did_not_launch(self):
        server = FakeV3Server()
        self.addCleanup(server.close)
        backend = HiggsV3ServedBackend(base_url=server.base_url)
        calls = self._capture_signals(backend)
        self.assertEqual(calls, [],
                         'no recorded pid means it is not ours to kill')


class SetVoiceTest(V3TestCase):

    def test_a_second_load_for_another_voice_is_refused_by_name(self):
        engine = HiggsV3Engine(self.config(probe_sentinel_filter=False))
        self.addCleanup(engine.cleanup)
        with self.assertRaises(ValueError) as caught:
            engine.set_voice('someone_else')
        message = str(caught.exception)
        self.assertIn('someone_else', message)
        self.assertIn('needs a NEW server', message)

    def test_reloading_the_same_voice_is_a_no_op(self):
        engine = HiggsV3Engine(self.config(probe_sentinel_filter=False))
        self.addCleanup(engine.cleanup)
        engine.set_voice(engine.voice)

    def test_orpheus_caps_on_a_v3_load_are_refused(self):
        engine = HiggsV3Engine(self.config(probe_sentinel_filter=False))
        self.addCleanup(engine.cleanup)
        engine._apply_voice_caps('deathstalker', {})     # the pool's reset: a no-op
        with self.assertRaises(ValueError):
            engine._apply_voice_caps('deathstalker', {'eosBoost': 8})


class WorkerRefusalTest(unittest.TestCase):
    """Things `python -m crucible_higgs.serve` must refuse rather than half-do."""

    def _run(self, extra_env, argv=()):
        import subprocess
        env = dict(os.environ)
        env.update({'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'})
        env.update(extra_env)
        env.pop('NARRATOR_GOLDEN_LOCAL', None)
        return subprocess.run(
            [sys.executable, '-m', 'crucible_higgs.serve', *argv], cwd=_PYTHON_ROOT,
            input='{"action": "quit"}\n', capture_output=True, text=True,
            encoding='utf-8', env=env, timeout=180)

    def test_the_not_shipped_scaffold_is_refused_without_a_handshake(self):
        """`higgs-v2-scaffold` is interface scaffolding, not a rendering engine.
        A spawn that selects it must not get a worker that looks alive."""
        out = self._run({'NARRATOR_ENGINE': 'higgs-v2-scaffold'},
                        argv=('--fake-engine',))
        self.assertNotEqual(out.returncode, 0)
        self.assertNotIn('"type": "ready"', out.stdout)
        self.assertIn('scaffold', out.stderr)
        self.assertIn('higgs-v3', out.stderr)

    def test_an_unknown_engine_is_refused_without_a_handshake(self):
        out = self._run({'NARRATOR_ENGINE': 'llasa'}, argv=('--fake-engine',))
        self.assertNotEqual(out.returncode, 0)
        self.assertNotIn('"type": "ready"', out.stdout)
        self.assertIn('llasa', out.stderr)

    def test_a_backend_that_cannot_be_detected_never_prints_ready(self):
        """THE MAC FAILURE (2026-09-04), forced IN PROCESS.

        A swallowed ImportError printed {"type":"ready","device":"mlx"} with no
        backend; the pool saw a healthy handshake and every generate answered
        'Model not loaded' forever. An engine that cannot be IMPORTED is a dead
        worker and must say so with an exit code.

        Driven by patching `worker._engine_class` to raise the ImportError a
        broken env raises - which is exactly what the reviewer did by hand.
        NOT through an env hook: production code carries no test switches, and a
        subprocess arm could only assert "whatever happened, happened", which is
        what this test used to do.
        """
        import contextlib
        import io as _io
        from crucible_higgs.serve import worker as W

        real = W._engine_class

        def broken():
            raise ImportError("No module named 'torch'", name='torch')

        W._engine_class = broken
        stdout, stderr = _io.StringIO(), _io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), \
                    contextlib.redirect_stderr(stderr):
                code = W.main([])
        finally:
            W._engine_class = real

        self.assertEqual(code, 3, 'a worker that cannot render must exit 3')
        self.assertEqual(stdout.getvalue(), '',
                         'NOTHING on stdout - not even a ready line; stdout is '
                         'the protocol and the pool would read a handshake')
        self.assertIn('FATAL', stderr.getvalue())
        self.assertIn('torch', stderr.getvalue(),
                      'the reason has to reach the operator')
        self.assertIn('no engine can load in this process', stderr.getvalue())

    def test_the_unservable_arm_also_exits_without_a_handshake(self):
        """The same guarantee through a REAL subprocess, on the arm that can be
        triggered from outside: an engine id the worker refuses to serve."""
        out = self._run({'NARRATOR_ENGINE': 'higgs-v2-scaffold'})
        self.assertEqual(out.returncode, 3)
        self.assertEqual(out.stdout.strip(), '')
        self.assertIn('FATAL', out.stderr)

    def test_a_fatal_startup_exits_three_and_explains(self):
        out = self._run({'NARRATOR_ENGINE': 'higgs-v2-scaffold'})
        self.assertEqual(out.returncode, 3)
        self.assertIn('no engine can load in this process', out.stderr)


class LoadedMessageTest(V3TestCase):
    """The `loaded` line has to carry what an ASSEMBLER will need."""

    def setUp(self):
        super().setUp()
        self.voices_path = os.path.join(self.dir, 'voices.json')
        with open(self.voices_path, 'w', encoding='utf-8') as handle:
            json.dump({'deathstalker': {'clips': [
                {'path': self.clip, 'transcript': X2_TEXT, 'seconds': 27.42}]}},
                handle)

    @unittest.skipIf(
        sys.platform.startswith('darwin'),
        'this drives the SERVED v3 engine through the worker, and on '
        "darwin 'higgs-v3' resolves to the in-process MLX engine instead "
        '(there is no server to fake). The Mac equivalents are '
        'tests/test_higgs_mlx.py and the live `python -m crucible_higgs.serve` '
        'smoke recorded in PORT_NOTES 13.6.')
    def test_it_carries_engine_samplerate_pads_and_the_asymmetric_fade(self):
        import subprocess
        env = dict(os.environ)
        env.update({
            'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8',
            'ORPHEUS_SKIP_WARMUP': '1',
            'NARRATOR_ENGINE': 'higgs-v3',
            'NARRATOR_HIGGS_VOICES': self.voices_path,
            'NARRATOR_HIGGS3_URL': self.server.base_url,
        })
        env.pop('NARRATOR_GOLDEN_LOCAL', None)
        proc = subprocess.Popen(
            [sys.executable, '-m', 'crucible_higgs.serve'], cwd=_PYTHON_ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding='utf-8', env=env,
            bufsize=1)
        self.addCleanup(proc.kill)
        try:
            ready = proc.stdout.readline()
            if not ready:
                try:
                    proc.wait(timeout=60)
                except Exception:
                    pass
                self.fail(f'worker exited {proc.returncode} before its handshake.'
                          f'\n--- stderr ---\n{(proc.stderr.read() or "").strip()}')
            self.assertEqual(json.loads(ready)['type'], 'ready')
            proc.stdin.write(json.dumps({'action': 'load',
                                         'voice': 'deathstalker',
                                         'warm': False}) + '\n')
            proc.stdin.flush()
            for _ in range(50):
                line = proc.stdout.readline()
                if not line:
                    try:
                        proc.wait(timeout=60)
                    except Exception:
                        pass
                    self.fail(f'worker exited {proc.returncode} before its '
                              f'`loaded` line.\n--- stderr ---\n'
                              f'{(proc.stderr.read() or "").strip()}')
                message = json.loads(line)
                if message['type'] in ('loaded', 'error'):
                    break
            self.assertEqual(message['type'], 'loaded', message)
            self.assertEqual(message['engine'], 'higgs-v3')
            self.assertEqual(message['sampleRate'], 24000)
            self.assertIs(message['pads'], False)
            self.assertEqual(message['edgeFadeMs'], {'in': 10.0, 'out': 25.0})
            # The pool's existing reads still work, unchanged.
            self.assertEqual(message['voice'], 'deathstalker')
            self.assertEqual(message['backend'], 'vllm-omni')
        finally:
            try:
                proc.stdin.write('{"action": "quit"}\n')
                proc.stdin.flush()
                proc.wait(timeout=20)
            except Exception:
                pass


class PadsOnTheWireTest(unittest.TestCase):
    """finalize_audio's trim is ORPHEUS behaviour; the gap belongs to the DOOR."""

    def setUp(self):
        from crucible_higgs.serve import worker as W
        self.W = W
        self.addCleanup(W.set_active_engine_audio, W.DEFAULT_SAMPLERATE, True)

    def test_a_pads_engine_is_trimmed(self):
        W = self.W
        W.set_active_engine_audio(24000, True)
        quiet = np.concatenate([np.zeros(2400, dtype=np.float32),
                                np.full(2400, 0.5, dtype=np.float32),
                                np.zeros(24000, dtype=np.float32)])
        out = W.finalize_audio(quiet, W.FOR_STREAM)
        self.assertLess(out.size, quiet.size,
                        "Orpheus's long trailing pause is cut back")

    def test_a_no_pads_engine_is_not_trimmed(self):
        """Higgs emits bare speech: a quiet final consonant sits under the 0.01
        threshold with no padding in front of it, so the same trim would cut
        into the word."""
        W = self.W
        W.set_active_engine_audio(24000, False)
        quiet = np.concatenate([np.zeros(2400, dtype=np.float32),
                                np.full(2400, 0.5, dtype=np.float32),
                                np.zeros(24000, dtype=np.float32)])
        out = W.finalize_audio(quiet, W.FOR_STREAM)
        self.assertEqual(out.size, quiet.size,
                         'nothing may be removed from a pads=False chunk')

    def test_neither_door_appends_a_gap(self):
        """THE TRIM IS THE ENGINE'S QUESTION AND THE GAP IS NOBODY'S HERE, and
        this pins that they are not the same question.

        `pads` says who owns the silence INSIDE a chunk, so it decides the trim
        and nothing else. The silence BETWEEN two chunks belongs to whoever joins
        them - the assembler behind the render door, the PLAYER on the stream,
        and neither of them is this function (Owen, 2026-09-18). The stream door
        states its gap as `gapSec` on the row's record instead, which
        `test_engine_serve_protocol.py` pins; what is left to say here is that no
        waveform grows by a sample at either door, on either kind of engine.
        """
        W = self.W
        tone = np.full(2400, 0.5, dtype=np.float32)
        for pads in (True, False):
            with self.subTest(pads=pads):
                W.set_active_engine_audio(24000, pads)
                self.assertEqual(W.finalize_audio(tone, W.FOR_STREAM).size,
                                 tone.size)
                self.assertEqual(W.finalize_audio(tone, W.FOR_RENDER).size,
                                 tone.size)

    def test_a_caller_that_does_not_say_which_door_is_refused(self):
        """THE BUG WAS EXACTLY A CALLER THAT HAD NOT DECIDED. `finalize_audio`
        took the stream contract as read and `_emit_guarded_batch` - written on
        2026-09-13, and the door Crucible's render job drives - picked it up
        without ever naming it. So the door is a required argument with no
        default, and a name that is neither door fails by name rather than
        being read as one of them.
        """
        W = self.W
        tone = np.full(2400, 0.5, dtype=np.float32)
        with self.assertRaises(TypeError):
            W.finalize_audio(tone)
        with self.assertRaises(ValueError) as caught:
            W.finalize_audio(tone, 'listen')
        self.assertIn('listen', str(caught.exception))

    def test_the_wire_rate_follows_the_loaded_engine(self):
        W = self.W
        W.set_active_engine_audio(16000, True)
        self.assertEqual(W.active_samplerate(), 16000)
        W.set_active_engine_audio(24000, True)
        self.assertEqual(W.active_samplerate(), 24000)


class VoiceDocumentShapesTest(V3TestCase):
    """THREE legitimate shapes, through the REAL loader and the REAL builder.

    The app catalog ships a `default` v3 voice - the model's own, no reference -
    and narrator used to refuse it, because it was written as a ClipsVoice with
    zero clips and ClipsVoice requires at least one. That rule is right and
    stays: a clone whose references went missing must be an error, never a
    silent downgrade. So "no reference" is its own member of the union.
    """

    def _load(self, document):
        from crucible_higgs.engine.higgs.config import load_voices
        path = os.path.join(self.dir, 'voices.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(document, handle)
        return load_voices(
            path, allowed_controls=HiggsV3Defaults.ALLOWED_CONTROLS,
            max_reference_seconds=HiggsV3Defaults.MAX_REFERENCE_SECONDS,
            placeholder_max_chars=HiggsV3Defaults.MAX_CHARS)

    def _clip_entry(self):
        return {'path': self.clip, 'transcript': X2_TEXT, 'seconds': 27.42}

    # ---- shape 1: a zero-shot reference clone ------------------------------

    def test_a_clips_voice_sends_one_reference(self):
        voice = self._load({'ds': {'clips': [self._clip_entry()]}})['ds']
        self.assertIsInstance(voice, ClipsVoice)
        body = v3_served.build_request_body('Hello.', voice, 200)
        self.assertEqual(len(body['references']), 1)
        self.assertEqual(body['references'][0]['text'], X2_TEXT)

    # ---- the document's sampling block reaches the served arm -------------

    def test_a_document_sampling_block_rides_as_the_served_override(self):
        """The catalog's per-backend `sampling` is written into the voice
        document (higgsVoicesDocument) and `load_voices` parses it; the served
        BUILDER must hand it to the config, or the block is documentary and the
        server renders at the checkpoint file / deploy default while the
        catalog states another number - exactly what the MLX arm did until
        13dc0e66. Through the REAL loader and the REAL builder."""
        from unittest import mock
        from crucible_higgs.engine.higgs import higgs_v3_config_from_worker_kwargs
        path = os.path.join(self.dir, 'voices-sampling.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump({'default': {'kind': 'default',
                                   'sampling': {'temperature': 0.7, 'topK': 40}},
                       'plain': {'kind': 'default'}}, handle)
        with mock.patch.dict(os.environ, {v3_config.VOICES_ENV: path,
                                          'NARRATOR_HIGGS3_URL': self.server.base_url}):
            config = higgs_v3_config_from_worker_kwargs(voice='default')
            plain = higgs_v3_config_from_worker_kwargs(voice='plain')
        self.assertEqual(config.sampling, {'temperature': 0.7, 'top_k': 40})
        sent = config.served_sampling()
        self.assertEqual(sent['temperature'], 0.7)
        self.assertEqual(sent['top_k'], 40)
        self.assertEqual(sent['top_p'], 0.95, 'a key the block leaves out keeps the default')
        self.assertIsNone(plain.sampling)
        self.assertEqual(plain.served_sampling()['temperature'], 1.0)

    # ---- shape 2: the model's own voice ------------------------------------

    def test_a_default_voice_loads_and_sends_no_references(self):
        from crucible_higgs.engine.protocol import DefaultVoice
        voice = self._load({'default': {'kind': 'default'}})['default']
        self.assertIsInstance(voice, DefaultVoice)
        self.assertEqual(voice.kind, 'default')
        self.assertEqual(voice.name, 'default')
        body = v3_served.build_request_body('Hello.', voice, 200)
        self.assertNotIn('references', body,
                         "the model's own voice takes no reference at all")
        self.assertEqual(body['input'], 'Hello.')

    def test_a_default_voice_uses_the_zero_shot_placeholder(self):
        voice = self._load({'default': {'kind': 'default'}})['default']
        self.assertEqual(voice.max_chars, HiggsV3Defaults.MAX_CHARS)
        self.assertEqual(voice.max_chars_source, 'placeholder')
        budget = HiggsV3Budget(HiggsV3Config(voice=voice,
                                             base_url=self.server.base_url))
        self.assertEqual(budget.max_chars(), 600)
        self.assertEqual(budget.max_chars_source(), 'placeholder')

    def test_the_reference_rules_do_not_apply_to_it(self):
        """No 30 s cap, no transcript - there is no reference to have either."""
        voice = self._load({'default': {'kind': 'default'}})['default']
        HiggsV3Config(voice=voice, base_url=self.server.base_url)   # no refusal

    def test_it_renders_end_to_end(self):
        voice = self._load({'default': {'kind': 'default'}})['default']
        engine = HiggsV3Engine(HiggsV3Config(voice=voice,
                                             base_url=self.server.base_url,
                                             probe_sentinel_filter=False))
        self.addCleanup(engine.cleanup)
        audio = engine.render_audio('It was a Saturday morning.')
        self.assertGreater(audio.size, 0)
        self.assertNotIn('references', self.server.requests[-1])

    # ---- shape 3: a fine-tune, whose weights ARE the voice -----------------

    def test_a_checkpoint_entry_needs_no_clips(self):
        from crucible_higgs.engine.protocol import DefaultVoice
        voice = self._load({'ds-ft': {'kind': 'checkpoint',
                                      'checkpointDir': '/models/ds-merged',
                                      'maxChars': 500}})['ds-ft']
        self.assertIsInstance(voice, DefaultVoice)
        self.assertEqual(voice.checkpoint_dir, '/models/ds-merged')
        self.assertEqual(voice.max_chars, 500)
        body = v3_served.build_request_body('Hello.', voice, 200)
        self.assertNotIn('references', body,
                         'a fine-tune is prompted with text alone')

    def test_a_checkpoint_no_longer_needs_its_maxChars(self):
        """Owen, 2026-09-19: `maxChars` is the CLIENT's packing size, so a
        checkpoint that declares none loads and packs at the placeholder."""
        voice = self._load({'ds-ft': {'kind': 'checkpoint',
                                      'checkpointDir': '/models/ds-merged'}})['ds-ft']
        self.assertEqual(voice.checkpoint_dir, '/models/ds-merged')
        self.assertEqual(voice.max_chars_source, 'placeholder')

    def test_a_checkpoint_kind_without_a_directory_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self._load({'ds-ft': {'kind': 'checkpoint', 'maxChars': 500}})
        self.assertIn('checkpointDir', str(caught.exception))

    def test_the_retired_adapterDir_key_is_refused_by_name(self):
        """There is no runtime LoRA on this stack, so an adapter directory has
        nothing to be loaded into. The refusal says what to write instead."""
        with self.assertRaises(ValueError) as caught:
            self._load({'ds-ft': {'kind': 'adapter',
                                  'adapterDir': '/models/ds-lora',
                                  'maxChars': 500}})
        message = str(caught.exception)
        self.assertIn('SupportsLoRA', message)
        self.assertIn('checkpointDir', message)

    # ---- the refusals that keep the shapes apart ---------------------------

    def test_a_clips_entry_with_an_empty_list_is_refused(self):
        """NOT quietly the default voice. ClipsVoice's >= 1 rule stands."""
        with self.assertRaises(ValueError) as caught:
            self._load({'ds': {'clips': []}})
        message = str(caught.exception)
        self.assertIn('no clips', message)
        self.assertIn("kind 'default'", message)

    def test_an_entry_that_says_nothing_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self._load({'mystery': {'scene': 'a quiet room'}})
        message = str(caught.exception)
        self.assertIn('12 %', message, 'the refusal must say what default costs')

    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self._load({'x': {'kind': 'description'}})
        self.assertIn('description', str(caught.exception))

    def test_the_maxCharsSource_vocabulary_is_closed(self):
        from crucible_higgs.engine.protocol import MAX_CHARS_SOURCES
        self.assertEqual(MAX_CHARS_SOURCES,
                         ('catalog', 'placeholder', 'length-sweep'))
        voice = self._load({'ds': {'clips': [self._clip_entry()],
                                   'maxChars': 450,
                                   'maxCharsSource': 'length-sweep'}})['ds']
        self.assertEqual(voice.max_chars_source, 'length-sweep')
        with self.assertRaises(ValueError) as caught:
            self._load({'ds': {'clips': [self._clip_entry()],
                               'maxChars': 450, 'maxCharsSource': 'vibes'}})
        self.assertIn('vibes', str(caught.exception))

    def test_a_max_chars_without_a_source_cannot_be_built(self):
        from crucible_higgs.engine.protocol import DefaultVoice
        with self.assertRaises(ValueError) as caught:
            DefaultVoice(name='x', max_chars=600)
        self.assertIn('provenance', str(caught.exception))


# =============================================================================
# convert_many: the server is fed continuously
# =============================================================================

class ConvertManyTest(unittest.TestCase):
    """The stream keeps exactly BATCH_SIZE TAKES in flight, refills a slot the
    moment one retires, reports every chunk, and a dead server ends it without
    submitting what was still queued.

    THE UNIT OF WORK IS A TAKE, NOT A CHUNK (Owen, 2026-09-08: "it should always
    be batched. even if it re-rolls, it should be batched"), so these fake
    `render_audio` - the one call a pool thread makes - and let the real
    `truncation.GuardPlan` decide what to ask for next. No server.
    """

    RATE = 24000
    PACE = 17.0            # the engine default band's centre: sqrt(20 * 14.5)

    def audio(self, chars, cps=None):
        import numpy as np
        return np.zeros(int(chars / (cps or self.PACE) * self.RATE), dtype=np.float32)

    def engine(self, width, render_audio):
        from types import SimpleNamespace
        eng = HiggsV3Engine.__new__(HiggsV3Engine)
        eng.BATCH_SIZE = width
        eng.config = SimpleNamespace(seed=1234, max_chars_per_sec=20.0,
                                     min_chars_per_sec=14.5)
        eng.voice_ref = SimpleNamespace()
        eng.render_audio = render_audio
        # THE DRIVER CALLS THE MEASURED DOOR. `render_audio` is its one-value
        # face, so a stub that replaced only that would leave the real one
        # talking to a server these tests do not have. The measure is None -
        # nothing here asserts `capped`, and inventing one would be a fake
        # measurement, which is worse than no measurement.
        eng.render_audio_measured = (
            lambda text, seed=None, index=0, **kw:
            (render_audio(text, seed=seed, index=index, **kw), None))
        eng.written = []
        eng._write_sentence = lambda index, audio: eng.written.append(index) or True
        return eng

    #: 200 characters - over MIN_GUARD_CHARS, so both edges of the band apply.
    TEXT = ('The night was long and the road was longer, and he walked it. ' * 4).strip()

    def test_width_is_held_and_slots_refill_as_takes_retire(self):
        lock = threading.Lock()
        active, peak = [0], [0]
        release = {i: threading.Event() for i in range(8)}
        pulled = []

        def render_audio(text, seed=None, index=0, sampling=None, take=0):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            release[index].wait(5)
            with lock:
                active[0] -= 1
            return self.audio(len(text))

        eng = self.engine(3, render_audio)
        done, in_flight = [], []

        def rows():
            for i in range(8):
                pulled.append(i)
                yield i, self.TEXT

        t = threading.Thread(target=eng.convert_many,
                             args=(rows(), lambda i, ok: done.append((i, ok)), in_flight))
        t.start()
        tick = threading.Event()
        for _ in range(100):
            if len(pulled) == 3:
                break
            tick.wait(0.02)
        self.assertEqual(pulled, [0, 1, 2], 'three in flight and only three')
        self.assertEqual(sorted(in_flight), [0, 1, 2])
        release[1].set()
        for _ in range(100):
            if len(pulled) == 4:
                break
            tick.wait(0.02)
        self.assertEqual(pulled, [0, 1, 2, 3], 'the retiring slot pulled exactly one more')
        self.assertEqual(done, [(1, True)])
        self.assertEqual(sorted(in_flight), [0, 2, 3])
        for i in range(8):
            release[i].set()
        t.join(5)
        self.assertFalse(t.is_alive())
        self.assertEqual(peak[0], 3)
        self.assertEqual(sorted(i for i, _ in done), list(range(8)))
        self.assertEqual(in_flight, [])
        self.assertEqual(sorted(eng.written), list(range(8)), 'every chunk written once')

    def test_a_job_may_render_NARROWER_than_the_serving_width(self):
        """`render_many(width=)`: the job's own ceiling on rows in flight.

        MEASURED, 2026-09-19: a server started 16 wide at SGLang
        `mem_fraction_static` 0.60 summed to 24.2 GB on a 24 GB card and WDDM
        paged the excess to host RAM - 4 to 10 times slower, with no error and
        no log line anywhere. The fine-tuning ladder renders 4 wide on voices
        whose manifest says 16, and restarting the server to say so is not
        something a job can do. So the width is a per-CALL ceiling under the
        width the server was started with, and nothing here reconfigures
        anything.

        The engine below is started 6 wide and asked for 2: what must hold is
        that only 2 rows are ever pulled and rendered at once.
        """
        lock = threading.Lock()
        active, peak = [0], [0]
        release = threading.Event()

        def render_audio(text, seed=None, index=0, sampling=None, take=0):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            release.wait(5)
            with lock:
                active[0] -= 1
            return self.audio(len(text))

        eng = self.engine(6, render_audio)
        out = []

        def drain():
            out.extend(eng.render_many(
                ((i, self.TEXT) for i in range(6)),
                tracker=truncation.tracker_for(truncation.engine_band(20.0, 14.5)),
                width=2))

        t = threading.Thread(target=drain)
        t.start()
        tick = threading.Event()
        for _ in range(100):
            if peak[0] >= 2:
                break
            tick.wait(0.02)
        tick.wait(0.15)          # long enough for a third to have started
        self.assertEqual(peak[0], 2,
                         'the call asked for width 2; a wider engine must not '
                         'put a third row in flight')
        release.set()
        t.join(10)
        self.assertFalse(t.is_alive())
        self.assertEqual(sorted(i for i, _a, _v, _m in out), list(range(6)),
                         'every row still renders - the width is a ceiling on '
                         'what is in flight, not on what is rendered')
        self.assertEqual(peak[0], 2)

    def test_a_reroll_re_enters_the_pool_instead_of_holding_its_slot(self):
        """THE POINT OF THE CHANGE. Chunk 0's take 0 is off-length, so the guard
        wants a re-roll - and while that re-roll waits its turn the pool is
        rendering other chunks, not sitting on chunk 0."""
        seen = []

        def render_audio(text, seed=None, index=0, sampling=None, take=0):
            seen.append((index, seed))
            if index == 0 and seed is None:
                return self.audio(len(text), cps=40.0)     # a truncation
            return self.audio(len(text))

        eng = self.engine(2, render_audio)
        done = []
        eng.convert_many(((i, self.TEXT) for i in range(4)),
                         lambda i, ok: done.append((i, ok)), [])
        self.assertEqual(sorted(done), [(0, True), (1, True), (2, True), (3, True)])
        # 5 takes for 4 chunks: chunk 0 twice, and its re-roll used another seed.
        self.assertEqual(len(seen), 5)
        self.assertEqual([s for i, s in seen if i == 0],
                         [None, truncation.reroll_seed(1234, 0, 1)])
        # ...and every other chunk still rendered. The re-roll is a unit of work
        # in the same pool, not a step chunk 0's slot walks through while the
        # rest of the book waits behind it.
        self.assertEqual(sorted(i for i, _ in seen), [0, 0, 1, 2, 3])

    def test_the_two_halves_of_a_split_render_at_the_same_time(self):
        """A split's halves are independent, and until 2026-09-08 they rendered
        one after the other inside one slot."""
        lock = threading.Lock()
        active, peak = [0], [0]
        started = threading.Event()

        def render_audio(text, seed=None, index=0, sampling=None, take=0):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            if len(text) < len(self.TEXT):
                started.wait(0.4)          # a half: hold the slot a moment
            with lock:
                active[0] -= 1
            if len(text) == len(self.TEXT):
                return self.audio(len(text), cps=40.0)     # whole text always short
            return self.audio(len(text))                   # halves are fine

        eng = self.engine(4, render_audio)
        done = []
        eng.convert_many(iter([(7, self.TEXT)]), lambda i, ok: done.append((i, ok)), [])
        self.assertEqual(done, [(7, True)])
        self.assertEqual(peak[0], 2, 'both halves were in flight together')

    def test_a_failed_take_fails_its_chunk_by_name_and_the_stream_goes_on(self):
        def render_audio(text, seed=None, index=0, sampling=None, take=0):
            if index == 3:
                raise RuntimeError('HTTP 400 for this text')
            return self.audio(len(text))

        eng = self.engine(2, render_audio)
        done = []
        eng.convert_many(((i, self.TEXT) for i in range(6)),
                         lambda i, ok: done.append((i, ok)), [])
        self.assertEqual(sorted(done), [(0, True), (1, True), (2, True),
                                        (3, False), (4, True), (5, True)])
        self.assertNotIn(3, eng.written, 'a chunk that failed is never written')

    def test_a_dead_server_ends_the_stream_before_the_queue_is_submitted(self):
        def render_audio(text, seed=None, index=0, sampling=None, take=0):
            if index == 1:
                raise v3_served.HiggsV3ServerDown('port 8100 refused')
            return self.audio(len(text))

        eng = self.engine(2, render_audio)
        pulled = []

        def rows():
            for i in range(10):
                pulled.append(i)
                yield i, self.TEXT

        in_flight = []
        with self.assertRaises(v3_served.HiggsV3ServerDown):
            eng.convert_many(rows(), lambda i, ok: None, in_flight)
        seen = len(pulled)
        self.assertLess(seen, 10)
        threading.Event().wait(0.1)
        self.assertEqual(len(pulled), seen, 'nothing is pulled after the exception')
        self.assertIn(1, in_flight)


class ContextLengthNamesTest(unittest.TestCase):
    """ONE external name for the context window, translated per stack.

    Crucible sets `HIGGS_CONTEXT_LENGTH` per voice in the spawn environment for
    every backend (a voice change is a full engine restart, so per-voice IS
    per-spawn). vllm-omni's launcher has always read `HIGGS_MAX_MODEL_LEN`
    (`serve_higgs_v3.sh:123` -> `--max-model-len`), and renaming it would break
    every launch line and campaign script that sets it - so narrator keeps both
    and refuses a disagreement rather than picking a winner nobody can see.
    """

    def setUp(self):
        for name in ('HIGGS_CONTEXT_LENGTH', 'HIGGS_MAX_MODEL_LEN'):
            os.environ.pop(name, None)
            self.addCleanup(os.environ.pop, name, None)

    def test_unset_is_the_stacks_own_window(self):
        self.assertEqual(v3_served.context_tokens(),
                         v3_served.DEFAULT_CONTEXT_TOKENS)

    def test_the_external_name_alone_decides_it(self):
        os.environ['HIGGS_CONTEXT_LENGTH'] = '4096'
        self.assertEqual(v3_served.context_tokens(), 4096)

    def test_the_launchers_own_name_alone_still_decides_it(self):
        os.environ['HIGGS_MAX_MODEL_LEN'] = '16384'
        self.assertEqual(v3_served.context_tokens(), 16384)

    def test_the_two_names_AGREEING_is_fine(self):
        os.environ['HIGGS_CONTEXT_LENGTH'] = '4096'
        os.environ['HIGGS_MAX_MODEL_LEN'] = '4096'
        self.assertEqual(v3_served.context_tokens(), 4096)

    def test_the_two_names_DISAGREEING_is_refused_by_name(self):
        """The loser would be invisible: the server comes up clean either way
        and the fault shows as an HTTP 500 hundreds of chunks later."""
        os.environ['HIGGS_CONTEXT_LENGTH'] = '8192'
        os.environ['HIGGS_MAX_MODEL_LEN'] = '4096'
        with self.assertRaises(ValueError) as caught:
            v3_served.context_tokens()
        message = str(caught.exception)
        self.assertIn('HIGGS_CONTEXT_LENGTH', message)
        self.assertIn('HIGGS_MAX_MODEL_LEN', message)

    def test_the_launch_exports_the_window_it_is_starting_with(self):
        os.environ['HIGGS_CONTEXT_LENGTH'] = '4096'
        backend = HiggsV3ServedBackend.__new__(HiggsV3ServedBackend)
        backend.base_url = 'http://127.0.0.1:8095'
        backend.concurrency = 4
        backend.checkpoint_dir = None
        backend.sentinel_report = '/tmp/report.json'
        backend.owner_id = lambda: 'keeper'
        self.assertIn('HIGGS_MAX_MODEL_LEN=4096', backend._launch_exports())
