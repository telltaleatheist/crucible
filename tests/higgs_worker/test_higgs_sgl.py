"""Higgs v3 on SGLang-Omni without the GPU: the request, the context, the wire.

The twin of `test_higgs_v3.py`, against the OTHER serving stack, and it exists
because the two are wrong in different ways. On vllm-omni the failure to catch is
sampling sent at the TOP LEVEL, where pydantic drops it in silence. Here it is
the exact opposite: `CreateSpeechRequest` HAS top-level `temperature`/`top_p`/
`top_k` fields, and `build_sglang_higgs_request` applies them ONLY when the
request carried them - so a request that sends NOTHING renders at top_k disabled
and the untruncated 1026-way codebook tail. Measured 2026-09-05: without top_k
one chunk ran to the cap with 80 s of silence.

THE FOUR FAILURES THIS MODULE IS HERE TO CATCH, none of which is visible in a
render's status:

  1. sampling not sent at all, or sent where the other stack puts it
     (`extra_params`), which this stack has no field for;
  2. the frame cap sent as `max_tokens` rather than `max_new_tokens`, which is
     not a field of CreateSpeechRequest and is therefore dropped;
  3. prompt tokens + `max_new_tokens` over 4,095, which is an HTTP 500 from
     inside the scheduler rather than a shorter render - and which
     `v3_served.cap_frames`' 2.0x ceiling reaches on its own at ~1,150
     characters, so it is not a corner case;
  4. a reference-clone voice rendered without its reference, which is the
     model's own speaker at 12 % of a fine-tune's ECAPA ceiling - or with its
     reference unaccounted for in the 4,096 positions, which is failure 3 again
     wearing a clip.

WHAT IS FIXTURE AND WHAT IS CAPTURE. The expected request body is built from the
training side's `higgs_ladder.py render` (which drove every night-3 measurement)
and from sglang-omni 0.1.4's own `serve/protocol.py` and
`models/higgs_tts/request_builders.py`, read on example-pc 2026-09-06. There is no
recorded capture; when one lands, `EXPECTED_REQUEST` is the one place it
replaces.
"""
import json
import io
import os
import pathlib
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_ROOT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "crucible", "jobs", "tts", "higgs")   # holds crucible_higgs
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)

from crucible_higgs.engine.higgs import served_common, sgl_served, v3_served   # noqa: E402
from crucible_higgs.engine.higgs.sgl_served import HiggsSglServedBackend       # noqa: E402
from crucible_higgs.engine.higgs.v3_engine import (HiggsV3Config,              # noqa: E402
                                             HiggsV3Engine,
                                             higgs_v3_stop_policy)
from crucible_higgs.engine.protocol import (ClipsVoice, DefaultVoice,          # noqa: E402
                                      ReferenceClip, SpeechRequest)
from tests.higgs_worker.test_higgs_v3 import (MERGED_GENERATION_CONFIG,      # noqa: E402
                                          X2_TEXT, a_wav, wav_bytes)


class FakeSglHandler(BaseHTTPRequestHandler):
    """Just enough of sgl-omni: /health, /v1/models, /v1/audio/speech.

    `/v1/models` answers the way the real one does -
    `ModelCard(id=model_name, root=model_name)`, the SERVED NAME in BOTH fields
    (serve/openai_api.py `_register_models`) - which is exactly why narrator
    cannot read the checkpoint out of it and reads the listener's environ
    instead. Getting this fixture wrong in the other direction (a `root` that is
    a path) would let a test pass on behaviour the real server does not have.
    """

    server_version = 'fake-sgl-omni/0.1.4'

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == sgl_served.HEALTH_PATH:
            if not self.server.healthy:
                self.send_error(503, 'not ready')
                return
            self.send_response(200)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if self.path == sgl_served.MODELS_PATH:
            if self.server.models_broken:
                self.send_error(404)
                return
            rows = [{'id': m, 'object': 'model', 'root': m}
                    for m in self.server.models]
            payload = json.dumps({'object': 'list', 'data': rows}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != sgl_served.SPEECH_PATH:
            self.send_error(404)
            return
        length = int(self.headers.get('Content-Length', '0'))
        body = json.loads(self.rfile.read(length).decode('utf-8'))
        self.server.requests.append(body)
        if self.server.error is not None:
            code, message = self.server.error
            payload = json.dumps({'detail': message}).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        audio = wav_bytes(self.server.seconds, self.server.rate, 1, None)
        self.send_response(200)
        self.send_header('Content-Type', 'audio/wav')
        self.send_header('Content-Length', str(len(audio)))
        self.end_headers()
        self.wfile.write(audio)


class FakeSglServer:
    """The fake, on a real port, in a thread."""

    def __init__(self, healthy=True, seconds=1.0, rate=24000):
        self.httpd = HTTPServer(('127.0.0.1', 0), FakeSglHandler)
        self.httpd.requests = []
        self.httpd.healthy = healthy
        self.httpd.error = None
        self.httpd.seconds = seconds
        self.httpd.rate = rate
        self.httpd.models = [sgl_served.SERVED_MODEL_NAME]
        self.httpd.models_broken = False
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
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


def state_env(case, name, value):
    """One environment variable for the duration of one test, restored after."""
    previous = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    if previous is None:
        case.addCleanup(os.environ.pop, name, None)
    else:
        case.addCleanup(os.environ.__setitem__, name, previous)


#: The sampling a validated merged checkpoint declares, which is what narrator
#: reads out of its `generation_config.json` and sends explicitly on this stack.
CHECKPOINT_SAMPLING = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 50,
                       'repetition_penalty': 1.0}


class SglTestCase(unittest.TestCase):
    """A fake server, a merged checkpoint, and the two contract variables."""

    def setUp(self):
        import tempfile
        self.dir = tempfile.mkdtemp(prefix='narrator-H-sgl-')
        self.addCleanup(self._rmtree)
        self.server = FakeSglServer()
        self.addCleanup(self.server.close)
        state_env(self, served_common.STACK_ENV, served_common.STACK_SGLANG_OMNI)
        state_env(self, v3_served.SERVE_MAX_NUM_SEQS_ENV, '2')

    def _rmtree(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def merged_checkpoint(self, name='ds-merged'):
        path = os.path.join(self.dir, name)
        os.makedirs(path, exist_ok=True)
        for filename, body in (('config.json', '{}'),
                               ('generation_config.json', MERGED_GENERATION_CONFIG)):
            with open(os.path.join(path, filename), 'w', encoding='utf-8') as handle:
                handle.write(body)
        return path

    def checkpoint_voice(self, name='deathstalker'):
        """The production shape on this stack: a fine-tune, prompted TEXT-ONLY."""
        return DefaultVoice(name=name, max_chars=600,
                            max_chars_source='length-sweep')

    def clips_voice(self, name='clone'):
        clip = a_wav(os.path.join(self.dir, 'ref.wav'), seconds=1.0)
        return ClipsVoice(clips=(ReferenceClip(clip, X2_TEXT, seconds=14.0),),
                          name=name)


# ---------------------------------------------------------------------------
# The request body
# ---------------------------------------------------------------------------

#: WHAT ONE CHUNK LOOKS LIKE ON THE WIRE. Every key is asserted, and so is the
#: ABSENCE of `extra_params` and of `max_tokens` - the two shapes that belong to
#: the other stack and are dropped in silence here.
EXPECTED_REQUEST = {
    'model': 'higgs-v3-ds',
    'input': 'It was a Saturday morning.',
    'response_format': 'wav',
    'max_new_tokens': 300,
    'seed': 1234,
    'temperature': 1.0,
    'top_p': 0.95,
    'top_k': 50,
    'repetition_penalty': 1.0,
}


class RequestShapeTest(SglTestCase):

    def test_the_body_carries_sampling_at_the_TOP_LEVEL(self):
        body = sgl_served.build_request_body(
            EXPECTED_REQUEST['input'], None, 300, seed=1234,
            sampling=CHECKPOINT_SAMPLING)
        self.assertEqual(body, EXPECTED_REQUEST)
        self.assertNotIn('extra_params', body,
                         'extra_params is vllm-omni\'s channel; CreateSpeechRequest '
                         'has no such field and pydantic drops it in silence')
        self.assertNotIn('max_tokens', body,
                         'the frame cap field is max_new_tokens; max_tokens is not '
                         'a field of CreateSpeechRequest at all')

    def test_a_default_voice_sends_no_references(self):
        body = sgl_served.build_request_body(
            'x' * 40, DefaultVoice(name='deathstalker'), 300,
            sampling=CHECKPOINT_SAMPLING)
        self.assertNotIn('references', body)

    def test_empty_sampling_is_REFUSED_by_name(self):
        """The whole reason this module exists. An empty mapping is correct on
        vllm-omni (the server reads the checkpoint's generation_config.json) and
        is the untruncated codebook tail here."""
        for empty in (None, {}):
            with self.assertRaises(ValueError) as caught:
                sgl_served.build_request_body('hello there', None, 300,
                                              sampling=empty)
            message = str(caught.exception)
            self.assertIn('no sampling', message)
            self.assertIn('top_k', message)
            self.assertIn('80 s of silence', message,
                          'the refusal must carry the measurement')

    def test_sampling_missing_a_lever_is_REFUSED_by_name(self):
        for missing in ('temperature', 'top_p', 'top_k'):
            partial = {k: v for k, v in CHECKPOINT_SAMPLING.items() if k != missing}
            with self.assertRaises(ValueError) as caught:
                sgl_served.build_request_body('hello there', None, 300,
                                              sampling=partial)
            self.assertIn(missing, str(caught.exception))

    def test_a_seed_inside_sampling_is_REFUSED(self):
        with self.assertRaises(ValueError) as caught:
            sgl_served.build_request_body(
                'hello there', None, 300,
                sampling={**CHECKPOINT_SAMPLING, 'seed': 7})
        self.assertIn('seed', str(caught.exception))

    def test_a_key_the_request_model_does_not_have_is_REFUSED(self):
        with self.assertRaises(ValueError) as caught:
            sgl_served.build_request_body(
                'hello there', None, 300,
                sampling={**CHECKPOINT_SAMPLING, 'min_p': 0.1})
        self.assertIn('min_p', str(caught.exception))

    def test_an_unknown_control_token_is_REFUSED(self):
        """Shared with the other stack: an out-of-vocabulary control token is
        read ALOUD as words and collapses the render (coverage 0.000)."""
        with self.assertRaises(ValueError) as caught:
            sgl_served.build_request_body(
                'Hello <|emotion:calm|> there', None, 300,
                sampling=CHECKPOINT_SAMPLING)
        self.assertIn('<|emotion:calm|>', str(caught.exception))

    def test_empty_text_is_REFUSED(self):
        with self.assertRaises(ValueError):
            sgl_served.build_request_body('   ', None, 300,
                                          sampling=CHECKPOINT_SAMPLING)


class ClipsVoiceReferenceTest(SglTestCase):
    """A reference-clone (zero-shot) voice renders on this stack: the clip rides
    in the body as base64 `data` - the `SpeechReference` branch that never
    touches the server's filesystem - and it is charged to the context."""

    def _b64(self, path):
        import base64
        with open(path, 'rb') as handle:
            return base64.b64encode(handle.read()).decode('ascii')

    def test_a_clips_voice_sends_ONE_reference_as_base64_data(self):
        voice = self.clips_voice()
        body = sgl_served.build_request_body('hello there', voice, 300,
                                             sampling=CHECKPOINT_SAMPLING)
        self.assertEqual(body['references'], [{
            'data': self._b64(voice.clips[0].path),
            'media_type': 'audio/wav',
            'text': X2_TEXT,
        }])
        # NOT the path route: `audio_path` is read by the server from its own
        # disk and needs --allowed-local-media-path, which the launcher does not
        # pass. And no `data:` prefix - `_normalize_reference` hands `data`
        # straight to base64 validation.
        self.assertNotIn('audio_path', body['references'][0])
        self.assertFalse(body['references'][0]['data'].startswith('data:'))

    def test_the_config_accepts_a_clips_voice_on_this_stack(self):
        config = HiggsV3Config(voice=self.clips_voice(), base_url=self.server.base_url)
        self.assertEqual(config.stack, served_common.STACK_SGLANG_OMNI)
        self.assertIsNone(config.checkpoint_dir, 'a zero-shot voice is the BASE weights')

    def test_the_reference_is_charged_to_the_prompt_bound(self):
        """`build_prompt`'s reference branch: ref_text_id + transcript +
        ref_audio_id + one placeholder per DELAYED row (T + 7 at 8 codebooks),
        T being the DECLARED seconds at 25 fps. 14 s -> 350 + 7 rows."""
        voice = self.clips_voice()
        without = sgl_served.prompt_token_bound('hello there')
        with_ref = sgl_served.prompt_token_bound('hello there', voice)
        expected = (sgl_served.REFERENCE_SCAFFOLD_TOKENS
                    + -(-len(X2_TEXT) // 3)
                    + 350 + sgl_served.REFERENCE_DELAY_ROWS)
        self.assertEqual(with_ref - without, expected)
        self.assertEqual(sgl_served.reference_token_bound(voice), expected)
        self.assertEqual(sgl_served.reference_token_bound(None), 0)
        self.assertEqual(sgl_served.reference_token_bound(self.checkpoint_voice()), 0)

    def test_the_frame_cap_leaves_room_for_the_reference(self):
        """At the zero-shot chunk length (600) a 14 s reference still leaves
        cap_frames its full 2.0x ceiling; at the fine-tunes' 1200 the context
        is the ceiling and the reference comes out of it."""
        voice = self.clips_voice()
        text = 'x' * 600
        self.assertEqual(sgl_served.frame_cap(text, voice), v3_served.cap_frames(text))
        long = 'x' * 1200
        cap = sgl_served.frame_cap(long, voice)
        self.assertLess(cap, sgl_served.frame_cap(long))
        self.assertLessEqual(cap + sgl_served.prompt_token_bound(long, voice),
                             sgl_served.max_context_positions())
        config = HiggsV3Config(voice=voice, base_url=self.server.base_url)
        self.assertEqual(config.cap_frames(long), cap,
                         "the config's cap charges the config's own voice")

    def test_a_chunk_the_reference_pushes_out_is_refused_naming_the_clip(self):
        voice = self.clips_voice()
        with self.assertRaises(ValueError) as caught:
            sgl_served.frame_cap('x' * 2200, voice)
        self.assertIn('reference clip', str(caught.exception))

    def test_the_backend_checks_the_context_WITH_the_reference(self):
        """`speak` re-checks prompt + cap at the wire; a cap sized without the
        clip is refused there rather than becoming an HTTP 500."""
        voice = self.clips_voice()
        backend = HiggsSglServedBackend(base_url=self.server.base_url)
        text = 'x' * 1200
        with self.assertRaises(ValueError):
            backend.speak(SpeechRequest(
                text=text, voice=voice, max_new_tokens=sgl_served.frame_cap(text),
                sampling=CHECKPOINT_SAMPLING))
        self.assertEqual(self.server.requests, [], 'nothing reached the wire')

    def test_a_non_wav_clip_is_refused_rather_than_mislabelled(self):
        flac = os.path.join(self.dir, 'ref.flac')
        with open(flac, 'wb') as handle:
            handle.write(b'fLaC')
        voice = ClipsVoice(clips=(ReferenceClip(flac, X2_TEXT, seconds=14.0),),
                           name='clone')
        with self.assertRaises(ValueError) as caught:
            sgl_served.reference_for(voice)
        self.assertIn('.wav', str(caught.exception))

    def test_two_clips_and_an_over_budget_clip_are_refused(self):
        clip = a_wav(os.path.join(self.dir, 'ref2.wav'), seconds=1.0)
        two = ClipsVoice(clips=(ReferenceClip(clip, X2_TEXT, seconds=10.0),
                                ReferenceClip(clip, X2_TEXT, seconds=10.0)),
                         name='two')
        with self.assertRaises(ValueError):
            sgl_served.reference_for(two)
        over = ClipsVoice(clips=(ReferenceClip(clip, X2_TEXT, seconds=42.0),),
                          name='over')
        with self.assertRaises(ValueError):
            sgl_served.reference_for(over)


# ---------------------------------------------------------------------------
# The 4096-token context
# ---------------------------------------------------------------------------


class ContextGuardTest(SglTestCase):

    def test_the_prompt_scaffold_is_exactly_three_tokens(self):
        """`[tts_id] + encode(text) + [text_id] + [audio_id]`, read off
        sglang_omni's HiggsTokenizerAdapter.build_prompt. Not an estimate."""
        self.assertEqual(sgl_served.PROMPT_SCAFFOLD_TOKENS, 3)
        self.assertEqual(sgl_served.prompt_token_bound(''), 3)

    def test_the_bound_is_above_the_measured_tokenizer_rate(self):
        """3.0 chars/token is a FLOOR: measured 3.25 at worst over the 50 packed
        Fuhrer chunks (26-1,190 chars) with ckpt-1080's tokenizer, and 4.56 on
        the longest. The bound must therefore sit ABOVE the real count for a
        chunk of that shape."""
        self.assertLessEqual(sgl_served.CHARS_PER_TOKEN_FLOOR, 3.25)
        # The longest measured chunk: 1,190 characters tokenized to 261 tokens.
        self.assertGreater(sgl_served.prompt_token_bound('x' * 1190), 261 + 3)

    def test_a_cap_that_fits_is_the_smaller_of_the_two_ceilings(self):
        text = 'x' * 600
        cap = sgl_served.frame_cap(text)
        self.assertLessEqual(cap, v3_served.cap_frames(text))
        self.assertLessEqual(
            cap + sgl_served.prompt_token_bound(text),
            sgl_served.max_context_positions())

    def test_a_long_chunk_is_capped_by_the_CONTEXT_not_by_cap_frames(self):
        """At ~1,150 characters `cap_frames`' 2.0x ceiling passes 4,096 on its
        own, so this is the ordinary case rather than a corner one."""
        text = 'x' * 1190
        self.assertGreater(v3_served.cap_frames(text),
                           sgl_served.max_context_positions())
        cap = sgl_served.frame_cap(text)
        self.assertEqual(
            cap,
            sgl_served.max_context_positions() - sgl_served.prompt_token_bound(text))

    def test_a_chunk_that_cannot_fit_is_REFUSED_by_name(self):
        with self.assertRaises(ValueError) as caught:
            sgl_served.frame_cap('x' * 2600)
        message = str(caught.exception)
        self.assertIn('2600-character', message)
        self.assertIn('4096', message)
        self.assertIn('targetChars', message, 'the refusal must name the lever')

    def test_the_backend_refuses_an_oversized_cap_before_the_wire(self):
        backend = HiggsSglServedBackend(base_url=self.server.base_url)
        with self.assertRaises(ValueError) as caught:
            backend.speak(SpeechRequest(
                text='hello there', voice=None, max_new_tokens=4090,
                sampling=CHECKPOINT_SAMPLING))
        self.assertIn('4095', str(caught.exception))
        self.assertEqual(self.server.requests, [],
                         'nothing may reach the server: it answers HTTP 500, not '
                         'a shorter render')


# ---------------------------------------------------------------------------
# The backend: identity, the wire, teardown
# ---------------------------------------------------------------------------


class ServerIdentityTest(SglTestCase):

    def _backend(self, **kwargs):
        return HiggsSglServedBackend(base_url=self.server.base_url, **kwargs)

    def test_a_server_serving_another_model_name_is_REFUSED(self):
        self.server.httpd.models = ['higgs-v3']      # the vllm-omni stack's name
        backend = self._backend()
        with self.assertRaises(served_common.HiggsServerError) as caught:
            backend.check_serves_expected_model()
        self.assertIn('higgs-v3-ds', str(caught.exception))

    def test_the_checkpoint_comes_from_the_LISTENERS_ENVIRON_not_v1_models(self):
        """`/v1/models` reports `root == model_name`, so it can never say which
        checkpoint is up. The scan reads HIGGS_MODEL_DIR out of the process."""
        checkpoint = self.merged_checkpoint()
        backend = self._backend(checkpoint_dir=checkpoint)
        backend._own_servers_on_port = lambda: [
            {'pid': 42, 'pgid': 42, 'owner': '1', 'modelDir': checkpoint}]
        backend.check_serves_expected_model()          # does not raise
        self.assertEqual(backend.running_checkpoint(), ('environ', checkpoint))

    def test_a_server_on_ANOTHER_checkpoint_is_REFUSED(self):
        backend = self._backend(checkpoint_dir=self.merged_checkpoint('wanted'))
        backend._own_servers_on_port = lambda: [
            {'pid': 42, 'pgid': 42, 'owner': '1',
             'modelDir': self.merged_checkpoint('running')}]
        with self.assertRaises(served_common.HiggsServerError) as caught:
            backend.check_serves_expected_model()
        self.assertIn('RESTARTING', str(caught.exception))

    def test_a_server_with_NO_model_dir_is_the_BASE_and_refuses_a_fine_tune(self):
        backend = self._backend(checkpoint_dir=self.merged_checkpoint())
        backend._own_servers_on_port = lambda: [
            {'pid': 42, 'pgid': 42, 'owner': '1', 'modelDir': None}]
        with self.assertRaises(served_common.HiggsServerError) as caught:
            backend.check_serves_expected_model()
        self.assertIn('BASE weights', str(caught.exception))

    def test_a_fine_tune_server_refuses_a_request_for_the_BASE(self):
        backend = self._backend()                      # no checkpoint = base
        backend._own_servers_on_port = lambda: [
            {'pid': 42, 'pgid': 42, 'owner': '1', 'modelDir': '/home/t/ds'}]
        with self.assertRaises(served_common.HiggsServerError) as caught:
            backend.check_serves_expected_model()
        self.assertIn('merged fine-tune', str(caught.exception))

    def test_an_unidentifiable_server_is_REFUSED_never_assumed_fine(self):
        backend = self._backend(checkpoint_dir=self.merged_checkpoint())
        backend._own_servers_on_port = lambda: []
        state_env(self, served_common.CHECKPOINT_ENV, None)
        with self.assertRaises(served_common.HiggsServerError) as caught:
            backend.check_serves_expected_model()
        self.assertIn(served_common.CHECKPOINT_ENV, str(caught.exception))

    def test_an_operator_assertion_is_used_only_when_there_is_no_fact(self):
        checkpoint = self.merged_checkpoint()
        backend = self._backend(checkpoint_dir=checkpoint)
        backend._own_servers_on_port = lambda: []
        state_env(self, served_common.CHECKPOINT_ENV, checkpoint)
        self.assertEqual(backend.running_checkpoint(), ('asserted', checkpoint))
        backend.check_serves_expected_model()          # does not raise


class WireTest(SglTestCase):

    def test_a_render_returns_24_kHz_mono(self):
        backend = HiggsSglServedBackend(base_url=self.server.base_url)
        audio, rate = backend.speak(SpeechRequest(
            text='It was a Saturday morning.', voice=None,
            max_new_tokens=sgl_served.frame_cap('It was a Saturday morning.'),
            seed=1234, sampling=CHECKPOINT_SAMPLING))
        self.assertEqual(rate, 24000)
        self.assertGreater(audio.size, 0)
        self.assertEqual(len(self.server.requests), 1)
        sent = self.server.requests[0]
        self.assertEqual(sent['model'], 'higgs-v3-ds')
        self.assertEqual(sent['top_k'], 50)
        self.assertIn('max_new_tokens', sent)

    def test_an_HTTP_500_names_the_context_as_the_usual_cause(self):
        self.server.fail_with(500, 'Internal Server Error')
        backend = HiggsSglServedBackend(base_url=self.server.base_url)
        with self.assertRaises(served_common.HiggsServerError) as caught:
            backend.speak(SpeechRequest(
                text='hello there', voice=None, max_new_tokens=300,
                sampling=CHECKPOINT_SAMPLING))
        self.assertIn('4095', str(caught.exception))

    def test_a_dead_server_raises_ServerDOWN_not_a_refusal(self):
        """A batch has to tell 'this chunk was refused' from 'the server is
        gone', or a dead port marks a whole book failed one sentence at a
        time."""
        dead = FakeSglServer(healthy=False)
        url = dead.base_url
        dead.close()
        backend = HiggsSglServedBackend(base_url=url)
        with self.assertRaises(served_common.HiggsServerDown):
            backend.speak(SpeechRequest(
                text='hello there', voice=None, max_new_tokens=300,
                sampling=CHECKPOINT_SAMPLING))

    def test_ServerDown_is_the_SAME_class_the_engine_catches(self):
        """`HiggsV3Engine.convert_many` ends a take on
        `v3_served.HiggsV3ServerDown`. A separate SGLang class would slip past
        that clause."""
        self.assertIs(served_common.HiggsServerDown, v3_served.HiggsV3ServerDown)
        self.assertIs(served_common.HiggsServerError, v3_served.HiggsV3ServerError)


class LaunchTest(SglTestCase):
    """The launch wrapper, which is the same shape as the vllm-omni arm's and
    carries this stack's own knobs."""

    def _launcher(self) -> str:
        """A REAL launch script on this filesystem, created once per test.

        `serve_script=` (and `NARRATOR_HIGGS_SGL_SERVE_SCRIPT` behind it) is an
        OPERATOR'S OVERRIDE of narrator's own packaged launcher as of
        2026-09-15, and a path that is not there is refused BY NAME rather than
        quietly replaced by the packaged one - substituting it would start a
        server with different flags and report success. So a test that wants
        LAUNCH mode has to name a file that exists. The literal
        `/campaign/serve_higgs_sgl.sh` these tests used to pass stopped being
        nameable for exactly the reason `test_higgs_v3.a_launcher` records about
        its own, one stack over and two days earlier.
        """
        existing = getattr(self, '_launcher_path', None)
        if existing is not None:
            return existing
        import shutil as _shutil
        import tempfile
        directory = tempfile.mkdtemp(prefix='narrator-SGL-launcher-')
        self.addCleanup(_shutil.rmtree, directory, True)
        path = os.path.join(directory, 'serve_higgs_sgl.sh')
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write('#!/bin/bash\nexit 0\n')
        os.chmod(path, 0o755)
        self._launcher_path = path
        return path

    def _backend(self):
        return HiggsSglServedBackend(serve_script=self._launcher(),
                                     wsl_distro='Ubuntu')

    def test_the_wrapper_exports_this_stacks_knobs_and_the_owner_marker(self):
        backend = self._backend()
        wrapper = backend.launch_command()[-1]
        self.assertIn(f'{sgl_served.SERVE_HOST_ENV}=', wrapper)
        self.assertIn(f'{sgl_served.SERVE_PORT_ENV}=', wrapper)
        self.assertIn(f'{sgl_served.SERVE_MAX_NUM_SEQS_ENV}=2', wrapper)
        self.assertIn(f'{served_common.OWNER_ENV}=', wrapper)
        self.assertIn(backend.owner_id(), wrapper)
        self.assertIn('setsid bash', wrapper)
        self.assertTrue(wrapper.rstrip().endswith('wait'))
        self.assertNotIn('SIGKILL', wrapper)

    def test_the_base_weights_UNSET_the_model_dir_rather_than_inherit_one(self):
        wrapper = self._backend().launch_command()[-1]
        self.assertIn(f'unset {sgl_served.SERVE_MODEL_DIR_ENV}', wrapper)

    def test_a_checkpoint_voice_exports_the_model_dir(self):
        backend = HiggsSglServedBackend(serve_script=self._launcher(),
                                        checkpoint_dir='/home/t/higgs_v3_merged/ds')
        wrapper = backend.launch_command()[-1]
        self.assertIn(f'{sgl_served.SERVE_MODEL_DIR_ENV}=/home/t/higgs_v3_merged/ds',
                      wrapper)
        self.assertNotIn(f'unset {sgl_served.SERVE_MODEL_DIR_ENV}', wrapper)

    @unittest.skipUnless(sys.platform == 'win32', 'the Windows launch arm')
    def test_the_windows_arm_uses_exec(self):
        command = self._backend().launch_command()
        self.assertTrue(command[0].endswith('wsl.exe'))
        self.assertEqual(command[1:6], ['-d', 'Ubuntu', '--exec', 'bash', '-c'])
        self.assertNotIn('--', command)

    def test_attach_mode_has_no_launch_command(self):
        backend = HiggsSglServedBackend(base_url=self.server.base_url)
        with self.assertRaises(ValueError):
            backend.launch_command()

    def test_neither_url_nor_script_RUNS_NARRATORS_OWN_LAUNCHER(self):
        """THE THIRD MODE, new on 2026-09-15. Until then this was a REFUSAL
        naming both variables, which meant the only way to start the stack Owen
        actually renders on was to hand narrator a path into a BookForge
        checkout — the exact lock-out 0eeb0267 removed for vllm-omni, still in
        place for the stack that works.

        The script it names is narrator's OWN, byte-identical to BookForge's
        (`tools/test-higgs-engine.js` pins the pair), so a client with no
        BookForge on disk launches the measured configuration."""
        state_env(self, sgl_served.BASE_URL_ENV, None)
        state_env(self, sgl_served.SERVE_SCRIPT_ENV, None)
        backend = HiggsSglServedBackend()
        self.assertEqual(backend.launcher_source, sgl_served.LAUNCHER_PACKAGED)
        self.assertEqual(os.path.basename(backend.serve_script),
                         sgl_served.PACKAGED_SERVE_SCRIPT)
        self.assertTrue(os.path.isfile(backend.serve_script))

    def test_the_packaged_launcher_is_the_one_that_ships(self):
        """It is package data, resolved through importlib.resources, and it is
        THIS stack's script rather than the vllm-omni one beside it. A launcher
        mix-up is not a crash: each script refuses the other's HIGGS_STACK, but
        only after a launch."""
        script = sgl_served.packaged_serve_script()
        self.assertTrue(os.path.isfile(script))
        body = pathlib.Path(script).read_text(encoding='utf-8')
        self.assertIn('sgl-omni', body)
        self.assertIn('--tts_engine.factory.max_running_requests', body)
        # ALONE, no sibling: SGLang-Omni has no deploy profile, so nothing here
        # reads `$(dirname "$0")` and none is shipped to be found.
        self.assertNotIn('HIGGS_DEPLOY_CONFIG', body)

    def test_an_override_that_is_not_there_is_refused_by_THIS_stacks_name(self):
        """The override is refused rather than silently replaced by the
        packaged script, and the refusal names the variable the operator
        actually set — not the vllm-omni arm's, whose check this shares."""
        state_env(self, sgl_served.BASE_URL_ENV, None)
        missing = os.path.join(self._launcher(), 'no', 'such.sh')
        with self.assertRaises(ValueError) as caught:
            HiggsSglServedBackend(serve_script=missing)
        self.assertIn(sgl_served.SERVE_SCRIPT_ENV, str(caught.exception))
        self.assertNotIn(v3_served.SERVE_SCRIPT_ENV, str(caught.exception))

    def test_the_attach_variable_is_this_stacks_OWN(self):
        """A stale NARRATOR_HIGGS3_URL must not point an SGLang engine at a
        vllm-omni server, which would answer /health and /v1/models in the right
        shapes and then drop half the request body.

        It no longer RAISES — with neither of this stack's variables set,
        narrator launches its own server — so what is asserted is that the
        other stack's url was not adopted as this one's."""
        state_env(self, sgl_served.BASE_URL_ENV, None)
        state_env(self, sgl_served.SERVE_SCRIPT_ENV, None)
        state_env(self, v3_served.BASE_URL_ENV, self.server.base_url)
        backend = HiggsSglServedBackend()
        self.assertEqual(backend.launcher_source, sgl_served.LAUNCHER_PACKAGED)
        self.assertNotEqual(backend.base_url, self.server.base_url)


# ---------------------------------------------------------------------------
# The engine, which is what actually picks the stack
# ---------------------------------------------------------------------------


class StackSelectionTest(SglTestCase):

    def test_HIGGS_STACK_unset_is_REFUSED_by_name(self):
        state_env(self, served_common.STACK_ENV, None)
        with self.assertRaises(ValueError) as caught:
            served_common.serving_stack()
        message = str(caught.exception)
        self.assertIn(served_common.STACK_ENV, message)
        self.assertIn('no default', message)

    def test_an_unknown_stack_is_REFUSED_by_name(self):
        state_env(self, served_common.STACK_ENV, 'tensorrt')
        with self.assertRaises(ValueError) as caught:
            served_common.serving_stack()
        self.assertIn('tensorrt', str(caught.exception))

    def test_the_engine_builds_the_SGLang_backend_and_reports_it(self):
        # ATTACH MODE: nothing here launched the fake server, so it carries no
        # ownership marker and its environ is not ours to read. The operator's
        # assertion is the only identity source left, and stating it is what an
        # operator attaching to their own server does.
        checkpoint = self.merged_checkpoint()
        state_env(self, served_common.CHECKPOINT_ENV, checkpoint)
        engine = HiggsV3Engine(HiggsV3Config(
            voice=self.checkpoint_voice(), base_url=self.server.base_url,
            checkpoint_dir=checkpoint))
        self.addCleanup(engine.cleanup)
        self.assertIsInstance(engine.server, HiggsSglServedBackend)
        self.assertEqual(engine.backend, served_common.STACK_SGLANG_OMNI)
        self.assertEqual(engine.backend_spec().name, 'sglang-omni')

    def test_a_checkpoint_voice_SENDS_its_sampling_on_this_stack(self):
        """The inverse of the vllm-omni rule, and the reason the stack has to be
        known: there, a checkpoint voice sends NOTHING because the server reads
        the directory itself."""
        checkpoint = self.merged_checkpoint()
        sgl = HiggsV3Config(voice=self.checkpoint_voice(),
                            base_url=self.server.base_url,
                            checkpoint_dir=checkpoint)
        self.assertEqual(sgl.served_sampling(), CHECKPOINT_SAMPLING)
        vllm = HiggsV3Config(voice=self.checkpoint_voice(),
                             stack=served_common.STACK_VLLM_OMNI,
                             base_url=self.server.base_url,
                             checkpoint_dir=checkpoint)
        self.assertEqual(vllm.served_sampling(), {})

    def test_the_frame_cap_is_the_stacks_own(self):
        checkpoint = self.merged_checkpoint()
        text = 'x' * 1190
        sgl = HiggsV3Config(voice=self.checkpoint_voice(),
                            base_url=self.server.base_url,
                            checkpoint_dir=checkpoint)
        vllm = HiggsV3Config(voice=self.checkpoint_voice(),
                             stack=served_common.STACK_VLLM_OMNI,
                             base_url=self.server.base_url,
                             checkpoint_dir=checkpoint)
        self.assertEqual(vllm.cap_frames(text), v3_served.cap_frames(text))
        self.assertEqual(sgl.cap_frames(text), sgl_served.frame_cap(text))
        self.assertLess(sgl.cap_frames(text), vllm.cap_frames(text))

    def test_the_stop_policy_records_the_stacks_cap(self):
        config = HiggsV3Config(voice=self.checkpoint_voice(),
                               base_url=self.server.base_url,
                               checkpoint_dir=self.merged_checkpoint())
        policy = higgs_v3_stop_policy(config)
        self.assertLessEqual(policy.max_new_tokens,
                             sgl_served.max_context_positions())

    def test_a_render_through_the_engine_sends_sampling_and_max_new_tokens(self):
        checkpoint = self.merged_checkpoint()
        state_env(self, served_common.CHECKPOINT_ENV, checkpoint)
        engine = HiggsV3Engine(HiggsV3Config(
            voice=self.checkpoint_voice(), base_url=self.server.base_url,
            checkpoint_dir=checkpoint))
        self.addCleanup(engine.cleanup)
        engine.render_audio('It was a Saturday morning.', index=3)
        sent = self.server.requests[-1]
        self.assertEqual(sent['top_k'], 50)
        self.assertEqual(sent['top_p'], 0.95)
        self.assertEqual(sent['temperature'], 1.0)
        self.assertEqual(sent['seed'], 1234 + 3, 'chunk i renders at seed + i')
        self.assertIn('max_new_tokens', sent)
        self.assertNotIn('extra_params', sent)


class MemFractionExportTest(unittest.TestCase):
    """`mem_fraction_static` is STATED at launch, not inherited in silence.

    The launch script has always read `HIGGS_SGL_MEM_FRACTION` from whatever
    environment it happened to be exec'd with (`serve_higgs_sgl.sh`:
    `--mem-fraction-static`, default 0.60), so the value reached the server but
    nothing narrator wrote said what it was. It is now one of the knobs
    narrator exports into the wrapper, and a nonsense one is refused HERE -
    before the launch is reported as started - rather than aborting inside a
    script narrator has already walked away from.
    """

    def _backend(self):
        return HiggsSglServedBackend.__new__(HiggsSglServedBackend)

    def _launchable(self):
        """Just enough backend to build the wrapper's export prefix."""
        backend = self._backend()
        backend.base_url = 'http://127.0.0.1:8200'
        backend.concurrency = 4
        backend.checkpoint_dir = None
        backend.owner_id = lambda: 'keeper'
        return backend

    def test_a_stated_fraction_is_exported(self):
        os.environ['HIGGS_SGL_MEM_FRACTION'] = '0.42'
        self.addCleanup(os.environ.pop, 'HIGGS_SGL_MEM_FRACTION', None)
        self.assertEqual(self._backend()._mem_fraction(), 0.42)
        self.assertIn('HIGGS_SGL_MEM_FRACTION=0.42',
                      self._launchable()._launch_exports(),
                      'the launch must STATE the fraction it started the '
                      'server at, not hope the script inherits one')

    def test_an_absent_fraction_exports_nothing(self):
        os.environ.pop('HIGGS_SGL_MEM_FRACTION', None)
        self.assertNotIn('HIGGS_SGL_MEM_FRACTION',
                         self._launchable()._launch_exports())

    def test_an_absent_fraction_states_nothing(self):
        os.environ.pop('HIGGS_SGL_MEM_FRACTION', None)
        self.assertIsNone(self._backend()._mem_fraction(),
                          "absent means the script's own default, which is "
                          'the script to own - not a second copy here')

    def test_a_fraction_outside_0_to_1_is_refused_by_name(self):
        for raw in ('0', '1', '1.5', '-0.1', 'sixty percent'):
            with self.subTest(raw=raw):
                os.environ['HIGGS_SGL_MEM_FRACTION'] = raw
                self.addCleanup(os.environ.pop, 'HIGGS_SGL_MEM_FRACTION', None)
                with self.assertRaises(ValueError) as caught:
                    self._backend()._mem_fraction()
                self.assertIn('HIGGS_SGL_MEM_FRACTION', str(caught.exception))


class ContextLengthEnvTest(unittest.TestCase):
    """`HIGGS_CONTEXT_LENGTH` is real on this stack, and narrator owns it.

    SGLang-Omni's Higgs builder hard-codes the window as a CLASS ATTRIBUTE
    (`sglang_omni/models/higgs_tts/engine_builder.py:29`,
    `context_length = 4096`) with no flag - three spellings were tried and
    recorded as failures on 2026-09-09, and the ladder's author had been
    rewriting that line in site-packages with a sed script, which the
    2026-09-15 env rebuild wiped. narrator instead ASSIGNS the attribute in the
    server process, from this variable, before the engine is built
    (`launch/sgl_omni_entry.py`), and reads the same variable here so the
    client sizes every request against the window the server is really serving.
    """

    def setUp(self):
        self.addCleanup(os.environ.pop, 'HIGGS_CONTEXT_LENGTH', None)
        os.environ.pop('HIGGS_CONTEXT_LENGTH', None)

    def test_unset_is_the_builders_own_4096(self):
        self.assertEqual(sgl_served.context_tokens(), 4096)
        self.assertEqual(sgl_served.max_context_positions(), 4095)

    def test_the_launched_value_is_what_every_request_is_sized_against(self):
        os.environ['HIGGS_CONTEXT_LENGTH'] = '8192'
        self.assertEqual(sgl_served.context_tokens(), 8192)
        self.assertEqual(sgl_served.max_context_positions(), 8191)
        # 1,759 characters: inside an 8,192-token window and over a 4,096-token
        # one. The window is what decides whether this chunk can be rendered at
        # all, which is the strongest thing it could decide.
        text = ('The night was long and the road was longer. ' * 40).strip()
        cap = sgl_served.frame_cap(text, None)
        self.assertLessEqual(sgl_served.prompt_token_bound(text) + cap,
                             sgl_served.max_context_positions())
        os.environ['HIGGS_CONTEXT_LENGTH'] = '4096'
        with self.assertRaises(ValueError) as caught:
            sgl_served.frame_cap(text, None)
        self.assertIn('4096', str(caught.exception),
                      'the refusal must name the window that refused it')

    def test_a_context_that_is_not_a_positive_int_is_refused_by_name(self):
        for raw in ('0', '-1', '8192.0', 'eight thousand'):
            with self.subTest(raw=raw):
                os.environ['HIGGS_CONTEXT_LENGTH'] = raw
                with self.assertRaises(ValueError) as caught:
                    sgl_served.context_tokens()
                self.assertIn('HIGGS_CONTEXT_LENGTH', str(caught.exception))

    def test_the_launch_STATES_the_window_it_asked_for(self):
        os.environ['HIGGS_CONTEXT_LENGTH'] = '8192'
        backend = HiggsSglServedBackend.__new__(HiggsSglServedBackend)
        backend.base_url = 'http://127.0.0.1:8200'
        backend.concurrency = 4
        backend.checkpoint_dir = None
        backend.owner_id = lambda: 'keeper'
        self.assertIn('HIGGS_CONTEXT_LENGTH=8192', backend._launch_exports())


class SglEntryModuleTest(unittest.TestCase):
    """narrator's own entry point for the SGLang server.

    It must be importable WITHOUT sglang_omni (it is packaged with narrator and
    read by this test on any platform), it must take the value from the
    environment, and it must assign the builder's class attribute - which is the
    only thing that has ever moved that number.
    """

    def _module(self):
        import importlib.util
        path = os.path.join(os.path.dirname(sgl_served.__file__), 'launch',
                            'sgl_omni_entry.py')
        self.assertTrue(os.path.isfile(path), path)
        spec = importlib.util.spec_from_file_location('sgl_omni_entry', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def setUp(self):
        self.addCleanup(os.environ.pop, 'HIGGS_CONTEXT_LENGTH', None)
        os.environ.pop('HIGGS_CONTEXT_LENGTH', None)

    def test_it_reads_the_same_variable_narrator_exports(self):
        module = self._module()
        self.assertEqual(module.CONTEXT_LENGTH_ENV,
                         served_common.CONTEXT_LENGTH_ENV)
        self.assertEqual(module.wanted_context_length(), 4096)
        os.environ['HIGGS_CONTEXT_LENGTH'] = '8192'
        self.assertEqual(module.wanted_context_length(), 8192)

    def test_it_ASSIGNS_the_builders_class_attribute(self):
        """The whole point. The attribute is evaluated at module import
        upstream, so nothing but an assignment after that import can move it -
        and a patched site-packages file is what the env rebuild wiped."""
        import sys as _sys
        import types
        module = self._module()
        builder = type('HiggsTtsEngineBuilder', (), {'context_length': 4096})
        pkg = types.ModuleType('sglang_omni')
        models = types.ModuleType('sglang_omni.models')
        higgs = types.ModuleType('sglang_omni.models.higgs_tts')
        eb = types.ModuleType('sglang_omni.models.higgs_tts.engine_builder')
        eb.HiggsTtsEngineBuilder = builder
        for name, mod in (('sglang_omni', pkg),
                          ('sglang_omni.models', models),
                          ('sglang_omni.models.higgs_tts', higgs),
                          ('sglang_omni.models.higgs_tts.engine_builder', eb)):
            _sys.modules[name] = mod
            self.addCleanup(_sys.modules.pop, name, None)
        before = module.apply_context_length(8192)
        self.assertEqual(before, 4096)
        self.assertEqual(builder.context_length, 8192,
                         'the builder must be serving the launched window')

    def test_a_missing_builder_is_a_hard_error(self):
        """Starting the server anyway would serve 4096 under a request for
        something else - the exact silence this entry exists to end."""
        module = self._module()
        with self.assertRaises(SystemExit) as caught:
            module.apply_context_length(8192)
        self.assertIn('engine_builder', str(caught.exception))

    def test_the_launcher_execs_the_entry_and_never_the_console_script(self):
        script = io.open(sgl_served.packaged_serve_script(),
                         encoding='utf-8').read()
        self.assertIn('sgl_omni_entry.py', script)
        self.assertNotIn('exec "$HIGGS_SGL_ENV/bin/sgl-omni" serve', script,
                         'the console script would build the engine with the '
                         'builder\'s own 4096 and report a clean start')


if __name__ == '__main__':
    unittest.main()
