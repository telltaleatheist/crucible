"""HiggsV3Engine - Higgs TTS 3 over a served backend, on EITHER serving stack.

Registry id `higgs-v3`. THE second engine (Owen, 2026-09-04 evening); Higgs v2
stayed only as interface scaffolding (`higgs-v2-scaffold`) because "it's
basically just Orpheus and we know Orpheus better".

TWO STACKS SERVE THE SAME MODEL, and `HIGGS_STACK` says which (BookForge sets it
from the catalog's `serving.stack`; `served_common.serving_stack()` refuses by
name when it is unset):

  `vllm-omni`    v3_served.py  - env higgs3, port 8095, 8192-token window,
                 sampling in `extra_params`, two required site-packages patches.
  `sglang-omni`  sgl_served.py - env sglomni, port 8200, a HARD-CODED 4096-token
                 window, sampling at the request TOP LEVEL and MANDATORY, no
                 patches, no reference clips. Measured 2026-09-05 on the same 50
                 chunks and the same checkpoint: 0 early stops and 0 sustained
                 voice switches at 16 in flight against vllm-omni's 4 and 6, at
                 2.5x the throughput. See that module's docstring.

The HTTP facts, the launch, the patches, the control-token allowlist, the 30 s
reference cap and the request/response shapes live in those two modules; what is
shared between them (ownership, the /proc listener scan, the watchdog, TERM-only
teardown) is `served_common.py`. This module is the `Engine` surface over all
three: codec geometry, budget, stop policy, and the render calls.

THE THREE PLACES THE STACK CHANGES WHAT THIS MODULE DOES, and nowhere else:
`HiggsV3Config.served_sampling` (empty vs the checkpoint's own numbers),
`HiggsV3Config.cap_frames` (which context window bounds the frame cap), and
which backend class `__init__` builds.

WHAT IS DIFFERENT FROM ORPHEUS

  backend       SERVED, not in-process (`BackendSpec.kind == 'served'`). The
                engine owns a server process: it starts it, waits out a
                55-297 s cold start (the measured range on this box) for health,
                and stops it. While up, the server preallocates to its
                gpu-memory-utilization target and OWNS the GPU.
  pads          False. Higgs emits bare speech; the manifest's
                gapBefore/gapAfter are live and the assembler realizes them.
  edge_fade     EdgeFade(10, 25) - ASYMMETRIC. Even with the server's
                content trim applied, a decoded chunk ends on a hard sample
                boundary, which is a click; a chunk also begins on one. The tail
                needs more than twice the head's window. Agrees with
                assemble.engine_profiles.PROFILES['higgs-v3'].
  no EOS levers there is no boost, no floor, no ratchet and no resplit ladder.
                v3 stops on its own; what it does instead is DROP THE TAIL of a
                long chunk, which no lever fixed. Hence max_chars 600 and
                `coverage_check = 'asr'`: a duration ratio measured 0.99 on a
                chunk that delivered 0.778 coverage with a 26 % insert rate.
  voice         a MERGED CHECKPOINT the server runs on (production: Owen,
                2026-09-04 - "fine-tuned voices only"), or one reference clip
                with its book-exact transcript (DIAGNOSTIC: how a voice is
                auditioned before anyone trains it). There is no runtime LoRA -
                vllm-omni has no adapter flags and its talker does not implement
                SupportsLoRA - so a voice change is a server restart.
  no streaming  the endpoint used is the buffered POST /v1/audio/speech, which
                returns a finished wav. vllm-omni also exposes a WebSocket
                `/v1/audio/speech/stream`, which narrator does NOT use: nothing
                here has been measured against it. `generate_batch_stream`
                therefore emits per row at retirement, and says so.
  batching      CONCURRENT REQUESTS, `BATCH_SIZE` of them at once. The server
                is a continuous batcher: stage 0 admits up to `max_num_seqs`
                sequences and schedules them together, and vllm-omni's own
                `/v1/audio/speech/batch` is an `asyncio.gather` over the items
                and nothing more (measured 2026-09-05). So the batch IS N
                in-flight POSTs, N = `HIGGS_MAX_NUM_SEQS` (`serve_concurrency`),
                the same number the launch script passes the server. The
                render worker's flush hands `convert_batch` a pool twice that
                wide so a retired slot is refilled while the slowest row is
                still decoding; `generate_batch_stream` retires rows AS THEY
                COMPLETE, not in index order - the serve worker keys every
                callback by row and the batch-stream contract permits it.
"""
import dataclasses
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..log import log
from ..protocol import (
    EMPTY_SENTENCE_SILENCE_SEC,
    BackendSpec,
    ClipsVoice,
    DefaultVoice,
    EdgeFade,
    SpeechRequest,
    StopPolicy,
)
from . import served_common, sgl_served, truncation, v3_served
from .codec import FrameMeasure
from .prompt import clean_text
from .served_common import STACK_SGLANG_OMNI
from .sgl_served import HiggsSglServedBackend
from .v3_served import HiggsV3ServedBackend


class HiggsV3Defaults:
    """Every v3 number, with the measurement behind it."""

    ENGINE_ID = 'higgs-v3'
    MODEL_ID = v3_served.MODEL_ID
    SAMPLE_RATE = v3_served.SAMPLE_RATE
    FRAMES_PER_SECOND = v3_served.FRAMES_PER_SECOND
    #: The window a config gets when nobody states one. The LAUNCHED value
    #: is `v3_served.context_tokens()`, read per load by
    #: `higgs_v3_config_from_worker_kwargs` - this is only the default that
    #: applies when a config is built in code with no environment at all.
    CONTEXT_TOKENS = v3_served.DEFAULT_CONTEXT_TOKENS
    # <= 600 chars is the measured safe zone; the delivered render used 300.
    MAX_CHARS = v3_served.MAX_CHARS
    # THE LENGTH BAND, ENFORCED since 2026-09-06 by `truncation.render_guarded`
    # on every chunk of both arms (it was advisory before that, and Fuhrer
    # chunk 19 shipped 3 s of a 1,127-character paragraph). Measured book pace
    # on deathstalker: 17.2 chars/s (5.83 s / 100 chars), every clean chunk
    # 0.94x-1.10x of it. Above MAX the take is too SHORT (an early stop:
    # 20.0 = ~0.86x); below MIN it is too LONG (a run-on tail: 14.5 = ~1.19x,
    # inside three of the four measured run-ons at 1.24x-1.54x, outside the
    # 1.11x one, which is the coverage audit's). See truncation.py.
    MAX_CHARS_PER_SEC = 20.0
    MIN_CHARS_PER_SEC = 14.5
    EDGE_FADE = EdgeFade(v3_served.EDGE_FADE_IN_MS,
                        v3_served.EDGE_FADE_OUT_MS)
    MAX_REFERENCE_SECONDS = v3_served.MAX_REFERENCE_SECONDS
    # v3's OWN control vocabulary (45 tokens). v2's default is empty, and a
    # voice loaded with v2's defaults would silently accept nothing - or, worse,
    # a v3 voice carrying v2's empty tuple would reject `<|prosody:long_pause|>`
    # while the model understands it perfectly.
    ALLOWED_CONTROLS = tuple(sorted(v3_served.ALLOWED_CONTROL_TOKENS))
    # COLD START IS NOT ONE NUMBER. Measured on this box, same script, same
    # env: 55 s (warm page cache), 146 s, and 297 s (cold weights, idle box,
    # narrator's own smoke). The spread is disk and first-run compilation, not
    # load. 900 s is therefore the default patience - long enough that a slow
    # start is never mistaken for a dead server, while `wait_ready` raises
    # IMMEDIATELY if the process actually died, so the wait is only ever paid
    # when something is genuinely still coming up.
    READY_TIMEOUT_SECONDS = 900.0


def apply_v3_voice_defaults(voice: ClipsVoice) -> ClipsVoice:
    """Stamp v3's own rules onto a voice: its 45-token control allowlist and its
    30 s reference cap.

    A voice read with the v2 scaffold's defaults carries an EMPTY allowlist and
    no cap. Empty is not "anything goes" - it is "this engine takes no control
    tokens", which is true of v2 and false of v3, and it would reject
    `<|prosody:long_pause|>` while the model understands it. A voice that
    already declared its own (from the document) is left alone.
    """
    if not isinstance(voice, ClipsVoice):
        # A DefaultVoice has no reference and no per-voice control allowlist -
        # there is nothing on it for v3's rules to stamp.
        return voice
    controls = (voice.allowed_controls
                or tuple(sorted(v3_served.ALLOWED_CONTROL_TOKENS)))
    cap = (voice.max_reference_seconds
           if voice.max_reference_seconds is not None
           else v3_served.MAX_REFERENCE_SECONDS)
    if (controls == voice.allowed_controls
            and cap == voice.max_reference_seconds):
        return voice
    # `replace` and NOT a fresh ClipsVoice(...) with a hand-written field
    # list: this stamps TWO fields and must carry every other one through
    # untouched. The hand-written list silently dropped `sampling`, the safe
    # band and the pace triple, and on 2026-09-15 it would have dropped
    # `base_dir` too - which is the directory the server runs on, so a
    # zero-shot voice whose controls needed stamping would have come up on
    # the wrong model.
    return dataclasses.replace(voice, allowed_controls=controls,
                               max_reference_seconds=cap)


@dataclass
class HiggsV3Config:
    """What one HiggsV3Engine is built from.

    `sampling` EMPTY means THE MODEL DIRECTORY'S OWN SAMPLING - see
    `served_sampling`. For a checkpoint voice that is the directory's
    `generation_config.json`, which the server reads for itself and which
    narrator has proved is there; for base weights there IS no such file in the
    bosonai snapshot, so v3's deploy default is sent EXPLICITLY rather than
    left to a bare `SamplingParams()` (top_p 1.0, top_k disabled - the babble
    case, PORT_NOTES 12.8d). Anything set here rides in `extra_params` on top.
    """
    voice: ClipsVoice
    #: WHICH SERVING STACK. `None` means "read `HIGGS_STACK`", which
    #: `__post_init__` does once and stores here, so everything downstream reads
    #: ONE resolved answer rather than the environment at four different moments.
    #: Refused by name when the variable is unset - see
    #: `served_common.serving_stack`.
    stack: Optional[str] = None
    base_url: Optional[str] = None
    serve_script: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    #: THE PINNED BASE WEIGHTS a zero-shot clone is conditioned on - the other
    #: directory a server can run on, and NEVER a merge (see
    #: `protocol.ClipsVoice.base_dir`). Kept apart from `checkpoint_dir`
    #: because only a merge carries the `generation_config.json` the sampling
    #: branches below are about; `model_dir` is what points the server.
    base_dir: Optional[str] = None
    sampling: Optional[dict] = None
    seed: Optional[int] = 1234
    sentences_dir: Optional[str] = None
    process_dir: Optional[str] = None
    audio_format: str = 'flac'
    max_chars: int = HiggsV3Defaults.MAX_CHARS
    max_chars_per_sec: float = HiggsV3Defaults.MAX_CHARS_PER_SEC
    min_chars_per_sec: float = HiggsV3Defaults.MIN_CHARS_PER_SEC
    context_tokens: int = HiggsV3Defaults.CONTEXT_TOKENS
    ready_timeout: float = HiggsV3Defaults.READY_TIMEOUT_SECONDS
    #: Run the sentinel-filter tail MEASUREMENT after /health (see
    #: HiggsV3ServedBackend.probe_sentinel_filter). ON by default, and cheap -
    #: one ~1 s render per server start. It reports the tail level against the
    #: certified band and refuses only a 200 that carries no audio; it is NOT
    #: the proof that the patch is applied, which is the doctor's static
    #: marker/absent-marker grep of the stage processor. See that method for
    #: what a real proof would be and the TODO that names it.
    probe_sentinel_filter: bool = True

    def __post_init__(self):
        # THE STACK, RESOLVED ONCE. Everything that differs between the two
        # serving stacks - where sampling rides, whether an empty sampling is
        # correct, how a frame cap is sized, which backend class is built - reads
        # this field and never the environment again.
        if self.stack is None:
            self.stack = served_common.serving_stack()
        elif self.stack not in served_common.STACKS:
            raise ValueError(
                f'HiggsV3Config(stack={self.stack!r}): the serving stacks are '
                f"{', '.join(served_common.STACKS)}.")
        if not isinstance(self.voice, (ClipsVoice, DefaultVoice)):
            raise ValueError(
                'HiggsV3Config(voice=...) takes a ClipsVoice (a reference clip with '
                "its book-exact transcript) or a DefaultVoice (the model's own "
                f'voice). Got {type(self.voice).__name__}.')
        if not (self.voice.name or '').strip():
            raise ValueError('HiggsV3Config needs a NAMED voice.')
        # A voice built directly (or loaded with another engine's defaults) may
        # carry v2's empty control allowlist and no reference cap. Stamp v3's on
        # it here so there is exactly ONE place a v3 voice acquires v3's rules.
        self.voice = apply_v3_voice_defaults(self.voice)
        # The 30 s cap and the one-reference rule, checked before a server is
        # ever started rather than after a 55 s launch and an HTTP 400.
        if isinstance(self.voice, ClipsVoice):
            v3_served.check_reference_budget(self.voice)
        # The voice may carry its own checkpoint; the config may name one
        # explicitly. They must not disagree.
        voice_checkpoint = getattr(self.voice, 'checkpoint_dir', None)
        if voice_checkpoint and self.checkpoint_dir and (
                voice_checkpoint != self.checkpoint_dir):
            raise ValueError(
                f"Higgs v3 voice '{self.voice.name}' names checkpoint "
                f'{voice_checkpoint} but the config names {self.checkpoint_dir}. One '
                'server runs on one checkpoint - refusing to pick.')
        if voice_checkpoint and not self.checkpoint_dir:
            self.checkpoint_dir = voice_checkpoint
        # THE SAME PAIR OF RULES FOR THE BASE WEIGHTS, and they are a separate
        # pair rather than the same one because the two directories are
        # different models: a merge IS the voice, base weights are what a
        # reference conditions.
        voice_base = getattr(self.voice, 'base_dir', None)
        if voice_base and self.base_dir and voice_base != self.base_dir:
            raise ValueError(
                f"Higgs v3 voice '{self.voice.name}' names base weights "
                f'{voice_base} but the config names {self.base_dir}. One '
                'server runs on one model - refusing to pick.')
        if voice_base and not self.base_dir:
            self.base_dir = voice_base
        if self.checkpoint_dir and self.base_dir:
            raise ValueError(
                f"Higgs v3 voice '{self.voice.name}' names a merged checkpoint "
                f'({self.checkpoint_dir}) AND base weights ({self.base_dir}). '
                "A fine-tune's voice is in its weights and a clone's is in its "
                'reference; a config that claims both has not said which model '
                'to serve.')
        if self.checkpoint_dir:
            # Also proves the directory carries the generation_config.json the
            # server will read its sampling out of - see
            # v3_served.require_generation_config.
            v3_served.checkpoint_serve_target(self.checkpoint_dir,
                                              self.voice.name)
        elif self.base_dir:
            # The bytes, and nothing more: base weights have never carried a
            # generation_config.json, which is exactly why `served_sampling`
            # STATES their sampling instead of letting the server resolve it.
            v3_served.require_base_weights_dir(self.base_dir, self.voice.name)

    #: The keys of `SERVER_DEFAULT_SAMPLING` that may ride in `extra_params`.
    #: `seed` is excluded: it is the request's TOP-LEVEL field and
    #: `build_request_body` refuses the duplicate.
    EXTRA_PARAM_KEYS = tuple(k for k in v3_served.SERVER_DEFAULT_SAMPLING
                             if k != 'seed')

    @property
    def model_dir(self):
        """WHICH DIRECTORY THE SERVER RUNS ON - the merge, or the pinned base
        weights, or `None` when the voice names neither and the launch
        script's own default decides.

        Distinct from `checkpoint_dir` on purpose. "Which model is served?"
        and "is that model a fine-tune whose own generation_config.json
        decides its sampling?" are two questions, and they stopped having one
        answer the moment a zero-shot voice could name its base weights
        (2026-09-15). Everything that POINTS the server reads this;
        everything that asks about SAMPLING reads `checkpoint_dir`.
        """
        return self.checkpoint_dir or self.base_dir

    def served_sampling(self) -> dict:
        """What rides in the request's `extra_params` - decided by voice KIND.

        THE SYMMETRY WITH THE MLX ARM (`HiggsV3MlxConfig.mlx_sampling`). Both
        answer the same question - "what sampling does this voice render at?" -
        and both branch on whether the voice IS a merged checkpoint, never on
        whether some file happens to exist.

          checkpoint voice   SEND NOTHING. The server reads
                             `<checkpointDir>/generation_config.json` for
                             itself, and `require_generation_config` has already
                             proved it is there and carries usable sampling. An
                             `extra_params` here would OVERRIDE the model's own
                             declared sampling with narrator's opinion of it.
          base weights       SEND v3's deploy default EXPLICITLY. The
                             `bosonai/higgs-audio-v3-tts-4b` snapshot ships no
                             `generation_config.json` at all (verified on the
                             WSL HF cache and the Mac's base copy), so sending
                             nothing here is NOT "the deploy defaults" - it is
                             vllm-omni's bare `SamplingParams()`: top_p 1.0,
                             top_k DISABLED, the untruncated 1026-way codebook
                             tail that derails long chunks into babble. That is
                             the exact failure this branch exists to prevent,
                             and until 2026-09-05 it was live for every
                             non-checkpoint voice on the served arm. SINCE
                             2026-09-15 "base weights" includes a ZERO-SHOT
                             voice that NAMES its base directory (`base_dir`,
                             pinned by whoever wrote the document so the clone
                             renders on known bytes). Naming the directory
                             does not put a generation_config.json in it, so
                             this branch reads `checkpoint_dir` - the merge -
                             and never `model_dir`.

        `sampling` is a named per-config override and is merged on top of
        either.

        ── AND ON SGLang-Omni IT IS NEVER EMPTY ────────────────────────────────

        The branch above is vllm-omni's, and it is right THERE and wrong on the
        other stack. SGLang-Omni reads no `generation_config.json` at all
        (`--generation-config` is a vLLM flag; this stack has no deploy profile
        either) and `build_sglang_higgs_request` sets top_p/top_k on the
        SamplingParams only when the REQUEST carried them - so "send nothing"
        means top_k disabled and the untruncated 1026-way codebook tail.
        MEASURED 2026-09-05: without top_k one chunk ran to the cap with 80 s of
        silence.

        So on that stack this returns `applied_sampling()` - what the model will
        actually sample at, read out of the checkpoint's own file by narrator
        instead of by the server. Same numbers, carried a different way, which is
        exactly the difference between the two stacks.
        """
        if self.stack == STACK_SGLANG_OMNI:
            return self.applied_sampling()
        # `self.checkpoint_dir` and not `self.model_dir`: a zero-shot voice
        # names a directory too - the BASE weights, pinned - and sending
        # nothing for those is the bare `SamplingParams()` this branch exists
        # to prevent, because base weights carry no generation_config.json for
        # the server to read. Only a MERGE is silent here.
        if self.checkpoint_dir:
            resolved = {}
        else:
            resolved = {k: v3_served.SERVER_DEFAULT_SAMPLING[k]
                        for k in self.EXTRA_PARAM_KEYS}
        resolved.update(self.sampling or {})
        return resolved

    def applied_sampling(self) -> dict:
        """What the model will actually SAMPLE at - which is not always what is
        sent (`served_sampling`), because a checkpoint's numbers live in its own
        directory and reach the server without passing through narrator.

        This is what `stop_policy` reports, and the manifest is the artifact that
        outlives the render: reporting the base default for a fine-tune would
        name sampling nobody used.

        A key the checkpoint's file omits is vLLM's own default; narrator has not
        measured those and does not state them, so only what the file says is
        reported.
        """
        if self.checkpoint_dir:
            document = v3_served.require_generation_config(
                self.checkpoint_dir, self.voice.name)
            resolved = {k: document[k] for k in self.EXTRA_PARAM_KEYS
                        if k in document}
        else:
            resolved = {k: v3_served.SERVER_DEFAULT_SAMPLING[k]
                        for k in self.EXTRA_PARAM_KEYS}
        resolved.update(self.sampling or {})
        return resolved

    def cap_frames(self, text: str) -> int:
        """`max_new_tokens` for one chunk, SIZED FOR THIS STACK.

        THE ONE SEAM where the two windows differ. vllm-omni serves an
        8,192-position window and `v3_served.cap_frames`' generous ceiling
        (2.0x the expected duration + 150 frames) costs nothing inside it.
        SGLang-Omni's Higgs builder hard-codes `context_length = 4096` with no
        flag, and prompt tokens + `max_new_tokens` over 4,095 is an HTTP 500 from
        inside the scheduler - not a shorter render - so `sgl_served.frame_cap`
        sends the smaller of the two ceilings and REFUSES BY NAME when what the
        context leaves would fall inside real speech.

        Every caller that sizes a request goes through here: `HiggsV3Budget
        .cap_frames` (which is what `render_audio` uses) and
        `higgs_v3_stop_policy` (which is what the manifest records).
        """
        if self.stack == STACK_SGLANG_OMNI:
            # The voice is charged for its reference clip, which rides in the
            # same 4,096 positions the cap has to fit under.
            return sgl_served.frame_cap(text, self.voice)
        return v3_served.cap_frames(text)


class HiggsV3Codec:
    """`protocol.Codec` for v3 - geometry only.

    THE SERVER DECODES. v3's tokens never reach this process: the endpoint
    returns a finished wav, and the patched server has already dropped every
    sentinel frame by token identity (work/patch_sentinel_filter.py, read to
    confirm it removes both the sentinel->0 substitution and the one-frame
    trim). So `decode()` refuses: a client-side trim on top would eat real
    speech, and there is nothing here to decode.

    The geometry is still reported, because the packer and the manifest need it:
    identical to v2 - 8 codebooks at 25 fps, 24 kHz, 960 samples per frame, a
    7-frame delay diagonal.
    """

    sample_rate = v3_served.SAMPLE_RATE
    frames_per_second = v3_served.FRAMES_PER_SECOND
    tokens_per_frame = 8
    samples_per_frame = int(v3_served.SAMPLE_RATE / v3_served.FRAMES_PER_SECOND)
    trim_frames = 7

    def frames_for_tokens(self, n_tokens: int) -> int:
        return int(n_tokens) // self.tokens_per_frame

    def audio_frames(self, generated_frames: int) -> int:
        return max(0, int(generated_frames) - self.trim_frames)

    def samples_for_frames(self, n_frames: int) -> int:
        return int(n_frames) * self.samples_per_frame

    def decode(self, tokens):
        raise NotImplementedError(
            'Higgs v3 decodes SERVER-SIDE: /v1/audio/speech returns a finished wav, '
            'and the patched server has already trimmed the sentinel tail by '
            'content. There are no audio tokens on this side to decode, and a '
            'second trim would eat speech. Use HiggsV3Engine.render_audio().')

    def streaming_decoder(self, decode_frames, label: str = ''):
        """None: the decode is behind the HTTP boundary. vllm-omni does expose a
        WebSocket `/v1/audio/speech/stream`, which nothing here has measured."""
        return None


class HiggsV3Budget:
    """`protocol.Budget` for v3."""

    def __init__(self, config: HiggsV3Config):
        self._config = config

    def _voice(self, voice):
        if voice is not None and voice != self._config.voice.name:
            raise ValueError(
                f"HiggsV3Budget: this engine serves '{self._config.voice.name}', not "
                f"'{voice}'. A v3 voice change is a server restart, so one engine "
                'answers for one voice.')
        return self._config.voice

    def max_chars(self, voice=None) -> int:
        """THE VOICE'S chunk size when the document declared one, else the
        engine's own placeholder - 600, the measured safe zone for the BASE
        model (900 chars drops the tail reproducibly, coverage 0.78-0.86, and a
        reference clip does not fix it; the delivered render used 300).

        `max_chars_source` says which of the two answered, and narrator's prep
        log prints it, so a book packed at the placeholder says so.

        NARRATOR NO LONGER REFUSES A FINE-TUNE THAT DECLARES NO `maxChars`
        (Owen, 2026-09-19). Until then a checkpoint voice with none raised here
        and at load: a fine-tune's safe chunk length is a measured property of
        that model, so narrator would not guess it. But `maxChars` IS THE
        CLIENT'S PACKING SIZE - the client packs the book and sends the chunks,
        and narrator renders what it is sent - and a screening render of a
        checkpoint nobody has measured yet is exactly the render that produces
        the measurement. Refusing it made the first render of every new
        checkpoint impossible. Nothing narrator does per chunk reads this
        number: the frame ceiling is `cap_frames(text)` over the text actually
        sent, and `higgs_v3_stop_policy`'s `max_new_tokens` is the ENGINE's own
        `config.max_chars`, never the voice's.
        """
        ref = self._voice(voice)
        if ref.max_chars is None:
            return int(self._config.max_chars)
        return int(ref.max_chars)

    def max_chars_source(self, voice=None) -> str:
        """'catalog' or 'placeholder' - so a log line can say which."""
        ref = self._voice(voice)
        return ref.max_chars_source or 'placeholder'

    def max_chars_per_sec(self, voice=None) -> float:
        return float(self._config.max_chars_per_sec)

    def max_total_tokens(self, prompt_tokens: int, voice=None) -> int:
        """The 8,192-position window (`--max-model-len 8192`) - a ceiling the
        prompt eats into. 27 s of reference is ~685 placeholder rows of it."""
        prompt_tokens = int(prompt_tokens)
        if prompt_tokens < 0:
            raise ValueError(f'prompt_tokens must be >= 0, got {prompt_tokens}')
        window = int(self._config.context_tokens)
        if prompt_tokens >= window:
            raise ValueError(
                f'Higgs v3 prompt is {prompt_tokens} positions against a '
                f'{window}-position window, leaving nothing to generate. Shorten the '
                'chunk or the reference - 27 s of reference audio is about 685 of '
                'those positions.')
        return window

    def cap_frames(self, text: str) -> int:
        """The config's, so the frame cap is sized for the stack that will serve
        it - see `HiggsV3Config.cap_frames`."""
        return self._config.cap_frames(text)


def higgs_v3_stop_policy(config: HiggsV3Config) -> StopPolicy:
    """v3 stops on its own and never hit the cap - but it DROPS TEXT, so
    `coverage_check` is 'asr' and it is not optional politeness.

    `levers` reports what the model will actually SAMPLE AT, which is not the
    same as what is sent: a checkpoint voice sends no `extra_params` because its
    own directory carries `generation_config.json`, and THOSE are the numbers
    the render used. Reporting the base default for a fine-tune would put
    sampling nobody used into the manifest - the artifact that outlives the
    render. See `HiggsV3Config.applied_sampling`.
    """
    sampling = config.applied_sampling()
    return StopPolicy(
        # The STACK'S cap for a full-length chunk - 4096 is a much lower ceiling
        # than 8192 and the manifest must record the one that applied.
        max_new_tokens=config.cap_frames('x' * int(config.max_chars)),
        eos_reliable=True,
        resplit_on_cap=False,
        max_chars_per_sec=float(config.max_chars_per_sec),
        min_chars_per_sec=float(config.min_chars_per_sec),
        levers={k: float(v) for k, v in sampling.items()},
        coverage_check='asr',
    )


class HiggsV3Engine:
    """One v3 server, owned."""

    ENGINE_ID = HiggsV3Defaults.ENGINE_ID
    SAMPLE_RATE = HiggsV3Defaults.SAMPLE_RATE
    pads = False
    edge_fade = HiggsV3Defaults.EDGE_FADE
    SUPPORTS_BATCH = True
    #: THE SERVER IS FED CONTINUOUSLY, not in pools. A batch on this backend is
    #: N concurrent requests, and the server itself batches continuously
    #: (vLLM's scheduler admits a new sequence the step after one retires), so
    #: the only way to leave it idle is to stop sending. `convert_batch` did
    #: exactly that once per pool: 32 rows through 16 threads, then a drain
    #: while the worker waited for the slowest row before handing over the next
    #: pool. MEASURED on Owen's witches render (2026-09-05, 259 requests): the
    #: server averaged 10.9 of 16 in flight and sat at 16 for 409 of 993
    #: seconds; GPU utilization swung 20-74%. `convert_many` keeps BATCH_SIZE in
    #: flight for the whole take and refills a slot the moment it retires, which
    #: is what the render worker uses when this attribute is True.
    CONTINUOUS_BATCH = True
    #: Class-level floor; the INSTANCE sets its own from `serve_concurrency()`
    #: (the render worker reads the attribute off the instance).
    BATCH_SIZE = 1

    def __init__(self, config: HiggsV3Config):
        if not isinstance(config, HiggsV3Config):
            raise ValueError(
                'HiggsV3Engine(config) takes a narrator.engine.higgs.HiggsV3Config; '
                f'got {type(config).__name__}.')
        self.config = config
        # THE STACK IS THE BACKEND'S NAME, and the config resolved it once (from
        # HIGGS_STACK, refused by name when unset). Reporting it here is what
        # puts the right one in the manifest and in every log line.
        self.backend = config.stack
        self.voice_ref = config.voice
        self.voice = config.voice.name
        self._codec = HiggsV3Codec()
        self._budget = HiggsV3Budget(config)
        self.params = {'samplerate': self.SAMPLE_RATE}
        # THE WIDTH OF THE BATCH = the server's admission width, one variable.
        # Resolved before the backend exists because a launch exports it.
        self.BATCH_SIZE = v3_served.serve_concurrency()
        # Twice the width per flush: the executor keeps BATCH_SIZE requests in
        # flight and refills a retired slot from the rest of the pool, so the
        # server is not left idling on the slowest row of every flush. A
        # cooperative stop discards at most one pool of finished work.
        self.batch_pool_size = 2 * self.BATCH_SIZE
        # THE CHECKPOINT GOES TO THE BACKEND: in launch mode it is exported as
        # HIGGS_MODEL_DIR so the server comes up ON this voice (it used not to
        # be, and the launcher served the base snapshot - Owen's first in-app
        # render, 2026-09-05); in both modes it is what the RUNNING server, as
        # reported by /v1/models, is held to in check_serves_expected_model.
        # THE SERVER'S LOG LIVES WITH THE RUN. `process_dir` is the session's
        # own directory - it already holds `session-state.json` and the Orpheus
        # guards' rejects - so a server this engine launches writes beside them
        # and the ledger can name one path for the whole run. With no session
        # (an audition, a test) the backend falls back to a per-instance file in
        # the temp dir and SAYS SO in its log line; what it never does is
        # discard the stream, which is what DEVNULL used to do and what left the
        # sentinel filter unprovable.
        #
        # WHICH BACKEND CLASS, decided by the stack and nothing else. The two
        # take the same constructor keywords on purpose - `base_url`,
        # `serve_script`, `checkpoint_dir`, `concurrency`, `server_log` - because
        # everything that differs between them is BELOW this line, inside the
        # class, and a caller that had to branch on the stack to build one would
        # be a second place the choice is made.
        #
        # `base_url` / `serve_script` fall through to the stack's OWN attach and
        # launch-script variables when the config states neither (each backend's
        # `__init__` reads them and refuses when both are absent), so a
        # NARRATOR_HIGGS3_URL cannot silently attach an SGLang engine to a
        # vllm-omni server: the names differ per stack.
        if config.stack == STACK_SGLANG_OMNI:
            self.server = HiggsSglServedBackend(
                base_url=config.base_url, serve_script=config.serve_script,
                checkpoint_dir=config.model_dir,
                concurrency=self.BATCH_SIZE,
                server_log=(os.path.join(config.process_dir,
                                         sgl_served.SERVER_LOG_NAME)
                            if config.process_dir else None))
        else:
            self.server = HiggsV3ServedBackend(
                base_url=config.base_url, serve_script=config.serve_script,
                checkpoint_dir=config.model_dir,
                concurrency=self.BATCH_SIZE,
                server_log=(os.path.join(config.process_dir,
                                         v3_served.SERVER_LOG_NAME)
                            if config.process_dir else None))
        self.load_engine()

    # ---- lifecycle ----------------------------------------------------------

    def load_engine(self):
        """Start the server (or adopt one), wait for health, take the
        sentinel-filter tail measurement and READ THE SERVER'S LOG for the
        token-level sentinel proof.

        A server that does not come up is a hard failure here rather than at the
        first sentence: the caller is holding a GPU lock and needs to know now.

        EVERY FAILURE AFTER `start()` STOPS THE SERVER. Without that, a timeout
        leaves a ~14 GB vllm-omni running with nothing holding a reference to
        it - the GPU stays occupied, the lock gets released by the caller's
        `finally`, and the next job OOMs against a process nobody can name.
        `stop()` is a no-op in attach mode, so this never kills somebody else's
        server.
        """
        self.server.start()
        try:
            if not self.server.wait_ready(self.config.ready_timeout):
                raise v3_served.HiggsV3ServerError(
                    f'Higgs v3 server at {self.server.base_url} did not answer '
                    f'{v3_served.HEALTH_PATH} within {self.config.ready_timeout:.0f}s. '
                    'Measured cold starts on this box range 55-297 s (page cache and '
                    'first-run compilation), so a longer wait is normal; this is not. '
                    'Check its log, and that both site-packages patches are applied '
                    'in the higgs3 env.')
            # Now that it answers, find the LISTENER (the pid the wrapper's
            # `$!` could never be - vllm-omni re-sessions itself) so stop() has
            # something of ours to signal. Adoption already knows the server.
            if self.server.serve_script and self.server._proc is not None:
                self.server._record_server()
            self.server.check_serves_expected_model(
                checkpoint_dir=self.config.model_dir)
            # THE SENTINEL PROOF IS vllm-omni'S, BOTH HALVES. The patch is a
            # site-packages fix to `vllm_omni/model_executor/
            # stage_input_processors/higgs_audio_v3.py`; SGLang-Omni has its own
            # stage processor, is not patched, and has no such line to read. So
            # the probe and the log proof are skipped on that stack - and said
            # out loud, because "not applicable" and "not proved" must not look
            # the same in a run log.
            if self.config.stack == STACK_SGLANG_OMNI:
                log('[HIGGSSGL] no sentinel-filter proof on this stack: the '
                    'patch is a vllm-omni site-packages fix and SGLang-Omni has '
                    'its own stage processor. Nothing here is unproved - there '
                    'is nothing of that kind to prove.', flush=True)
            elif self.config.probe_sentinel_filter:
                self.server.probe_sentinel_filter()
                # PROOF (a), on the RECORDS the probe render just wrote. It used
                # to be `proof_log()` and a grep of the server's log file; since
                # 2026-09-13 the patch writes structured records and this reads
                # those (crucible/docs/ARCHITECTURE.md R4). Skipped only when
                # there is no report to read at all - an attached server whose
                # operator named none - and that is said out loud rather than
                # passed over, because "not proved" and "proved" must never look
                # the same in a run log.
                if self.server.proof_report():
                    self.server.verify_sentinel_filter()
                else:
                    log('[HIGGS3] token-level sentinel proof UNAVAILABLE: attached '
                        'to a server this process did not start and no '
                        f'{v3_served.SENTINEL_REPORT_ENV} was named, so there is no '
                        'sentinel report to read. The static half of the proof (no '
                        'one-frame trim left in the stage processor) is unaffected '
                        '- it is a grep the Higgs doctor runs before any server '
                        'starts.', flush=True)
        except BaseException:
            self.server.stop()
            raise
        return self.server

    def cleanup(self):
        """Stop the server and release the GPU. Idempotent."""
        self.server.stop()

    @classmethod
    def detect_backend(cls) -> str:
        """THE SERVING STACK, from `HIGGS_STACK` - refused by name when unset.

        A CLASS METHOD ASKED BEFORE ANY ENGINE EXISTS (`serve/worker.py` reports
        it in its `ready` message), which is why it reads the environment rather
        than a config. It answers the same question `HiggsV3Config.stack` does
        and reads the same variable, so the two cannot disagree.
        """
        return served_common.serving_stack()

    @classmethod
    def resolve_load_voice(cls, voice, model_dir=None, adapter_dir=None,
                           base_dir=None) -> str:
        """Validate a `load` message's voice AS THIS ENGINE understands it.

        Orpheus's allowlist is eight stock TOKENS; v3's voices are entries in the
        NARRATOR_HIGGS_VOICES document, and a name like 'deathstalker' is neither
        stock-Orpheus nor a modelDir/adapterDir load. Checking a v3 load against
        Orpheus's list rejected every real v3 voice ("Unknown Orpheus voice
        'deathstalker'"), which is why voice validation belongs to the engine.

        Returns the voice name as the document spells it - NOT lower-cased:
        Orpheus tokens are case-folded because the prompt token is, and a v3
        voice name is a document key, where folding could miss an entry.
        """
        if model_dir:
            raise ValueError(
                f'Higgs v3 load carried modelDir={model_dir!r}. The served model is '
                "the launch script's argument, not a per-load field.")
        if base_dir:
            raise ValueError(
                f'Higgs v3 load carried baseDir={base_dir!r}. v3 has no shared-base + '
                'per-voice-adapter split.')
        name = (voice or '').strip()
        if not name:
            raise ValueError(
                'Higgs v3 load has no voice name. The name selects an entry in the '
                "NARRATOR_HIGGS_VOICES document; there is no default, and the model's "
                'own voice sits at 12 % of the narrator ceiling.')
        from .config import load_voice
        resolved = load_voice(
            name, allowed_controls=HiggsV3Defaults.ALLOWED_CONTROLS,
            max_reference_seconds=HiggsV3Defaults.MAX_REFERENCE_SECONDS,
            placeholder_max_chars=HiggsV3Defaults.MAX_CHARS)
        # The served directory's REQUIRED FILES are checked here, at the load
        # message, for the same reason `maxChars` is: the refusal belongs
        # before a 55-297 s server start and before anything holds a GPU.
        # WHICH files are required depends on which directory the voice names
        # - a merge's generation_config.json, base weights' bytes and nothing
        # else - and `voice_serve_target` is the one place that knows.
        v3_served.voice_serve_target(resolved)
        return name

    def backend_spec(self) -> BackendSpec:
        return self.server.spec

    # ---- the seam -----------------------------------------------------------

    def codec(self) -> HiggsV3Codec:
        return self._codec

    def budget(self) -> HiggsV3Budget:
        return self._budget

    def stop_policy(self, voice=None) -> StopPolicy:
        if voice is not None and voice != self.voice:
            raise ValueError(
                f"HiggsV3Engine.stop_policy({voice!r}): this engine serves "
                f"'{self.voice}'. A v3 voice is its reference (or its adapter, which "
                'is a server restart) - a second voice is a second engine.')
        return higgs_v3_stop_policy(self.config)

    def set_voice(self, voice: str, adapter_dir: str = None) -> None:
        """A SECOND load on a live v3 worker: refused by name.

        The serve worker's warm path calls this when a load arrives on an engine
        that is already up. For Orpheus it is a free prompt-prefix switch (or a
        LoRA re-point). For v3 it cannot be: a fine-tuned voice IS a merged
        checkpoint the server was started ON, and vllm-omni cannot load a voice
        into a running one - no adapter flags, and its talker does not implement
        `SupportsLoRA`. (A diagnostic clips voice is no better: its reference
        rides in every request.) Silently accepting would leave the worker
        reporting voice B while the server renders voice A for a whole book.

        Switching the SAME voice back to itself is a no-op, which is what a pool
        re-issuing an identical load actually wants.
        """
        want = (voice or '').strip()
        if want == self.voice and (adapter_dir or None) == (
                self.config.checkpoint_dir or None):
            return
        raise ValueError(
            f'HiggsV3Engine cannot switch voice in place: this server was started '
            f"for '{self.voice}'"
            + (f' on model {self.config.model_dir}'
               if self.config.model_dir else '')
            + f", and a load for '{want}'"
            + (f' (checkpoint {adapter_dir})' if adapter_dir else '')
            + f' needs a NEW server. Neither stack can load a voice into a running '
              f'one ({self.config.stack} here): vllm-omni has no adapter flags and '
              'its talker does not implement SupportsLoRA, and SGLang-Omni serves '
              'the one --model-path it was started on. Every fine-tuned voice is a '
              'whole merged checkpoint the server runs ON. Quit this worker and '
              'start one for that voice (~55 s warm, up to ~300 s cold on '
              'vllm-omni; ~110 s on SGLang-Omni).')

    def _apply_voice_caps(self, voice: str, caps: dict) -> None:
        """Orpheus's per-voice tuning registry has no v3 counterpart - v3's
        sampling comes from the MODEL DIRECTORY's `generation_config.json`
        (12.8d), with `extra_params` the only channel that can override it. An
        EMPTY payload is the pool's "no catalog tuning" signal and is accepted
        as the no-op it is; anything else is refused by name rather than looking
        applied."""
        if caps:
            raise ValueError(
                f'Higgs v3 load for {voice!r} carried caps={sorted(caps)}. Those are '
                'ORPHEUS tuning caps and v3 implements none of them.')

    # ---- rendering ----------------------------------------------------------

    def _clean_sentence_for_tts(self, sentence: str) -> str:
        """The SML strip, shared by every engine. The serve worker calls this on
        the engine before it renders, so it is part of the surface."""
        return clean_text(sentence)

    #: The levers a per-item rung may name on the SERVED arm. All four, because
    #: both serving stacks carry all four per request: vllm-omni's
    #: `extra_params` is validated against `SERVER_DEFAULT_SAMPLING` and
    #: SGLang-Omni's `CreateSpeechRequest` has them at the top level
    #: (`sgl_served.SAMPLING_KEYS`). `seed` is NOT a sampling lever here - it is
    #: the request's own top-level field and both builders refuse the duplicate.
    ITEM_SAMPLING_LEVERS = ('temperature', 'topP', 'topK', 'repetitionPenalty')

    def accept_item_sampling(self, raw, where: str = None):
        """One item's `sampling` off the wire -> this engine's dict, or None.

        THIS IS THE CHANNEL CRUCIBLE'S TAKE LADDER NEEDED. A rung is per
        RENDER; the voices document is per LOAD; until this existed Crucible
        refused every take above 0 by name (`sampling_not_wired`,
        crucible/docs/PHASE3-TTS.md section 4) because a reload per take is not
        a ladder. Both refusals - `sampling_malformed` and
        `sampling_not_supported` - are in `engine/item_sampling.py`.
        """
        from ..item_sampling import parse_item_sampling
        return parse_item_sampling(raw, where or f'HiggsV3Engine({self.voice!r})',
                                   levers=self.ITEM_SAMPLING_LEVERS)

    def accept_item_take(self, raw, where: str = None) -> int:
        """One item's `take` off the wire -> a whole number >= 0.

        THE OTHER HALF OF A RUNG. A rung is (sampling deltas, seed offset), and
        until 2026-09-15 this engine honoured only the first: every take of
        chunk i drew at `config.seed + i`, so take 0 and take N were
        byte-identical whenever their numbers matched. `_request_seed` is where
        the offset lands; this is where the wire's number is checked.
        `take_malformed` is the one refusal - see `engine/item_sampling.py`.
        """
        from ..item_sampling import parse_item_take
        return parse_item_take(raw, where or f'HiggsV3Engine({self.voice!r})')

    def _sampling_for(self, item_sampling) -> dict:
        """The sampling ONE request carries: `served_sampling()` with the
        item's rung laid over it, key by key.

        AN OVERLAY, NOT A REPLACEMENT, and on this arm that is the difference
        between a rung and a babble. `served_sampling()` on SGLang-Omni is the
        full resolved triple and the request is refused without it; a rung of
        `{temperature: 0.7}` replacing it wholesale would send no top_k, which
        is the untruncated 1026-way codebook tail - measured 2026-09-05 as one
        chunk running to the cap with 80 s of silence. On vllm-omni with a
        checkpoint voice `served_sampling()` is deliberately EMPTY (the server
        reads the checkpoint's own generation_config.json), so a rung there
        rides alone in `extra_params` and vllm-omni merges it over the file -
        which is exactly what "take 0, but cooler" means.
        """
        from ..item_sampling import apply_over
        return apply_over(self.config.served_sampling(), item_sampling)

    def _render_guarded(self, index: int, text: str, first_take=None):
        """`render_audio` under the truncation ladder - see `truncation.py`."""
        clean = self._clean_sentence_for_tts(text)
        return truncation.render_guarded(
            lambda part, seed: self.render_audio(part, seed=seed, index=index),
            clean, index, sample_rate=self.SAMPLE_RATE,
            # THE BAND FOLLOWS THE BOOK (Owen, 2026-09-08): one tracker per
            # engine, seeded from the ENGINE's own measured band (Fuhrer,
            # deathstalker) and re-centred on the shipped takes' own median.
            # This is narrator's own audiobook path; the serve door is given
            # its band by the batch instead (`_pace_tracker`).
            tracker=self._pace_tracker(),
            base_seed=self.config.seed, first_take=first_take)

    def _pace_tracker(self):
        """The engine's ONE `PaceTracker` for ITS OWN audiobook path, made on
        first use so the running pace spans the whole book and a test that
        builds the engine without `__init__` still gets one.

        ON THE ENGINE'S OWN BAND, stated at the call site (`engine_band`), and
        no longer on the voice entry's: the band has ONE owner and on the serve
        door that owner is the batch (Owen, 2026-09-19 - see
        `truncation.tracker_for`). `convert` / `convert_many` are narrator
        rendering a book itself, with no caller to be given a band by, so the
        engine's measured pair is the honest source there and this is the only
        place it is read.
        """
        tracker = getattr(self, '_pace', None)
        if tracker is None:
            tracker = truncation.tracker_for(
                truncation.engine_band(float(self.config.max_chars_per_sec),
                                       float(self.config.min_chars_per_sec)),
                'HiggsV3Engine._pace_tracker')
            self._pace = tracker
        return tracker

    def _seed_for(self, index: int):
        """THE seed rule, one place: chunk i renders with `seed + i`.

        `index` is the CHUNK's index - the sentence number for `convert`, the row
        for `generate_batch_stream`, and 0 only for a bare `render_audio` with no
        index of its own. render_final.py seeded chunk i with `1234 + i` for the
        same reason: a re-render of one chunk reproduces it, and two chunks of a
        batch never share a draw. Rendering every sentence at `seed` flat - which
        the served worker branch used to do - makes a whole book of identical
        draws for identical text.

        The seed travels as the request's TOP-LEVEL `seed` field and NEVER also
        inside `extra_params`; `build_request_body` refuses the duplicate.
        `HiggsV3Config.seed = None` means "do not seed at all".
        """
        if self.config.seed is None:
            return None
        return int(self.config.seed) + int(index)

    def _request_seed(self, seed, index: int, take: int):
        """THE SEED ONE REQUEST ACTUALLY CARRIES: the ladder's own seed when it
        chose one, this chunk's `seed + index` when it did not, shifted into
        take `take`'s lane either way.

        ONE DOOR, so the lane cannot be applied twice or missed once. Take 0
        returns exactly what `_seed_for` / the ladder returned, which is why
        every pre-ladder caller is untouched; a take above 0 moves the WHOLE
        chunk - its take 0, its re-roll and both halves of a split - into a
        lane no other take uses (`truncation.TAKE_SEED_STRIDE` carries the
        disjointness argument).
        """
        base = self._seed_for(index) if seed is None else seed
        return truncation.in_take_lane(base, take)

    def render_audio(self, text: str, seed=None, index: int = 0,
                     sampling=None, take: int = 0) -> np.ndarray:
        """One chunk of text -> a float32 mono waveform at 24 kHz.

        `index` is the chunk's own index and is what seeds it (see `_seed_for`);
        an explicit `seed` overrides. Callers that HAVE an index must pass it -
        `convert` and `generate_batch_stream` do - so a book is not rendered at
        one seed from end to end.

        `sampling` is THIS CHUNK's take-ladder rung, already parsed by
        `accept_item_sampling`. None renders at the voice's loaded sampling,
        which is take 0 - the existing behaviour, and the documented meaning of
        "no rung", not a substituted default.

        `take` is WHICH RUNG, and it moves the seed rather than the numbers
        (`_request_seed`). It is independent of `sampling` on purpose: a rung
        that declares no sampling override is still a different draw, because
        the lane moved. 0 is take 0 and changes nothing.
        """
        # THE MODEL BOUNDARY STRIPS THE MARKUP - here, once, for every caller.
        # `[break]` / `[heading]` / `[item]` / `[pause:X]` are narrator's own
        # markers: the packer writes them into the chunk text (677 of 1067
        # chunks of Owen's first Higgs book began with `[break]`; every heading
        # with `[break][heading]`), the assembler realizes them as gaps, and
        # Orpheus's convert() strips them before its prompt. This path only
        # trimmed whitespace, so Higgs READ THEM: "break" spoken at the head of
        # most chunks, "[heading]" as 7-15 s of gibberish before "Dedication."
        # (measured on the Mac, 2026-09-05; identical on the served arm). The
        # serve worker cleans before it calls, which is why Listen never showed
        # it; the render worker hands convert() the stored text, which is why
        # every book did.
        return self.render_audio_measured(text, seed=seed, index=index,
                                          sampling=sampling, take=take)[0]

    def render_audio_measured(self, text: str, seed=None, index: int = 0,
                              sampling=None, take: int = 0):
        """`render_audio`, with the `FrameMeasure` it spent - `(audio, measure)`.

        THE MEASUREMENT IS UNCONDITIONAL (Owen, 2026-09-19), so it is taken
        here, where the cap this render was sized at is still in hand, rather
        than reconstructed later by a caller doing the arithmetic again.

        INFERRED, NOT COUNTED, on this arm, and `FrameMeasure.counted` says so.
        The server returns a WAV and nothing else: no token count and no stop
        reason (`v3_served.decode_response`). The cap-hit is still decidable
        from the frames that came back - see `FrameMeasure.from_audio` - but
        the frame COUNT is a lower bound, so it is not reported as `tokens`.
        Closing that would mean a metadata channel on /v1/audio/speech, which
        is a server change and not narrator's to make.
        """
        clean = self._clean_sentence_for_tts(text)
        if not clean:
            raise ValueError(
                'HiggsV3Engine.render_audio(): the chunk has no text once its '
                f'markers are stripped ({(text or "").strip()!r}).')
        cap = self._budget.cap_frames(clean)
        request = SpeechRequest(
            text=clean, voice=self.voice_ref,
            max_new_tokens=cap,
            seed=self._request_seed(seed, index, take),
            sampling=self._sampling_for(sampling))
        audio, _rate = self.server.speak(request)
        return audio, FrameMeasure.from_audio(audio, cap, self.SAMPLE_RATE)

    def _sentence_file(self, sentence_number: int) -> str:
        if not self.config.sentences_dir:
            raise ValueError(
                'HiggsV3Engine.convert() needs a sentences_dir: this engine was built '
                'for in-memory generation, so there is nowhere to write chunk '
                f'{sentence_number}. Use generate_batch_stream / render_audio.')
        return os.path.join(self.config.sentences_dir,
                            f'{sentence_number}.{self.config.audio_format}')

    def convert(self, sentence_number: int, sentence: str) -> bool:
        """Render one chunk to `<sentences_dir>/<n>.<audio_format>`, EXACTLY AS
        DECODED - no trim, no fade, no pad. The server already trimmed the
        sentinel tail; the fades are the assembler's (`edge_fade`), and the
        gaps are the manifest's.

        THROUGH THE TRUNCATION LADDER (`truncation.render_guarded`): a take
        whose audio is too short for its text is re-rolled at another seed,
        then split at a sentence boundary and re-rendered - the Orpheus
        practice, asked for by Owen on 2026-09-06 after Fuhrer chunk 19
        shipped 3 s of a 1,127-character paragraph."""
        return self._write_sentence(sentence_number,
                                    self._render_guarded(sentence_number, sentence))

    def _write_sentence(self, sentence_number: int, audio) -> bool:
        """The chunk file, EXACTLY AS DECODED - no trim, no fade, no pad.

        ONE writer for every path on this arm - `convert` and the pool driver in
        `convert_many` - for the reason the MLX backend states at its own
        `_write_sentence`: a batched render must land byte-identically to a
        single-row one, and two copies of `sf.write` is how a subtype or a
        container drifts between them. PCM_16 is stated, never left to a library
        default (engine/orpheus/audio.py:write_chunk_file).
        """
        import soundfile as sf
        path = self._sentence_file(sentence_number)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, audio, self.SAMPLE_RATE, subtype='PCM_16',
                 format=self.config.audio_format.upper())
        return True

    def _write_silence(self, sentence_number: int) -> bool:
        """An empty sentence's placeholder clip - see `Engine._write_silence`.

        THROUGH `_write_sentence`, which is the whole point of there being one
        writer on this arm: the silence lands with the same subtype and the same
        container as every rendered chunk beside it. Nothing is generated - an
        empty chunk has nothing to say and `render_audio` refuses it by name.
        """
        return self._write_sentence(
            sentence_number,
            np.zeros(int(self.SAMPLE_RATE * EMPTY_SENTENCE_SILENCE_SEC),
                     dtype=np.float32))

    def convert_batch(self, items) -> list:
        """`items` through the SAME pool as a continuous take, and one bool per
        item in that order.

        A thin wrapper on `convert_many` since 2026-09-08, and that is the point:
        there is one driver on this arm, so a book rendered through the worker's
        continuous path and a list rendered through this one take the same route
        - `BATCH_SIZE` renders in flight, the guard's retakes among them (Owen:
        "it should always be batched. even if it re-rolls, it should be
        batched"). The two copies that stood here rendered whole ladders inside
        pool threads, which is exactly the serialisation that went.

        A dead server propagates, as it always has; a row that fails for its own
        reason is that row's False.
        """
        items = list(items)
        if not items:
            return []
        results = {}
        self.convert_many(iter(items), lambda index, ok: results.__setitem__(index, ok), [])
        return [bool(results.get(index, False)) for index, _text in items]

    def convert_many(self, rows, on_done, in_flight) -> None:
        """`BATCH_SIZE` RENDERS in flight for as long as there is work.

        `rows` is an ITERATOR of `(index, text)` and is pulled lazily: a slot is
        refilled the moment one retires, so the server never sees the drain a
        pool boundary made. `on_done(index, ok)` is called from THIS thread at
        each chunk's completion, so the worker's counters and progress lines stay
        single-threaded. `in_flight` is the worker's list of indices whose files
        may be half-written right now; it is kept exact here so a cooperative
        stop deletes precisely the rows that were running.

        ── EVERY TAKE IS A UNIT OF WORK, INCLUDING THE GUARD'S ─────────────────

        Owen, 2026-09-08: *"it should always be batched. even if it re-rolls, it
        should be batched. there are no situations in which serializing is
        superior."*

        Until then a slot held one chunk's WHOLE LADDER: take 0, then its re-roll,
        then - if it split - the left half rendered to completion before the right
        half was begun. The re-roll has to follow its own take 0 and nothing can
        change that, but the two halves of a split are independent and were
        rendering one after the other inside a single slot while other slots sat
        on unrelated chunks. So the unit of work here is now a `RenderRequest`
        from a `truncation.GuardPlan`, not a chunk: a slot is FREED the moment a
        take lands, and the retake it provoked re-enters the same pool as one more
        request. The ladder's policy is untouched (`truncation.py`: one policy,
        three drivers now - serial for a single chunk, rounds for the MLX slab,
        this pool for the server).

        THE PLAN IS THIS THREAD'S. `GuardPlan` is not thread-safe and does not
        need to be: the pool threads only render, and every `add` / `next_request`
        / `offer` happens here, at a retirement, exactly where `on_done` already
        happened. The files are written here too, for the same reason.

        A dead server (`HiggsV3ServerDown`, raised by any request) ends the take:
        nothing further is submitted, the requests already running are left to
        fail fast against the dead port, and the exception propagates. That is
        `convert_batch`'s policy, kept - every remaining row would fail the same
        way, and marking the book failed one sentence at a time is the thing this
        backend refuses to do. A request that fails for its OWN reason (an HTTP
        400 for its text, a decode failure) fails its chunk by name and the take
        goes on, which is the per-row policy this arm has always had.

        ── THE DRIVER MOVED OUT; THIS IS ITS FILE SINK ─────────────────────────

        Since 2026-09-13 every word above still describes what happens, but it
        happens in `render_many` - this method is the loop that writes what the
        driver yields. Owen ruled that day that the model and its inference own
        the guard AND the retake decision, and that a render must be able to
        happen on another computer; a guarded driver that writes files is
        unreachable to a caller that has no directory, which is why
        `narrator/serve/worker.py` - the path Crucible drives - went round the
        guard into a bare `render_audio()` and rendered unguarded
        (crucible/docs/PHASE6-REMOTE-RENDER.md section 0). Lifting the driver
        ABOVE the file layer is the whole fix: one plan, one ladder, two sinks.
        """
        for index, audio, _verdict, _measure in self.render_many(
                rows, in_flight, tracker=self._pace_tracker()):
            # The verdict is dropped HERE and nowhere else: this arm's analytics
            # have always come off the `[HIGGS3][HIGGS_GUARD_EVENT]` lines the
            # plan still prints, and `GuardPlan.verdict()` POPS - reading it and
            # discarding it is also what stops a book's records accumulating.
            if audio is None:
                on_done(index, False)
            else:
                self._write_sentence(index, audio)
                on_done(index, True)

    def render_many(self, rows, in_flight=None, sampling_by_index=None,
                    take_by_index=None, *, tracker, width=None):
        """THE GUARDED DRIVER: yields `(index, audio, verdict, measure)` as the
        ladder decides each chunk, and writes NOTHING.

        `tracker` is the `PaceTracker` this call judges against, and it is
        REQUIRED: the band has one owner per call and no default (Owen,
        2026-09-19). The serve door builds it from the band on the batch
        (`generate_batch`'s `band`); `convert_many` passes the engine's own
        (`_pace_tracker`). There is no arm that picks one for the caller - an
        inherited band was the defect, a band centred at 15.0 against a book
        running at 17.2 chars/s re-rolling healthy chunks to MAX_DEPTH.

        `measure` is the shipped take's `FrameMeasure` (`GuardPlan.measure`):
        what the render SPENT against its frame ceiling, joined across a
        chunk's parts when the ladder split it. A measurement, never a verdict
        - it is taken whether or not anything judged the chunk.

        `width` is how many takes this CALL may keep in flight, at most
        `BATCH_SIZE` - the width the server was started with
        (`v3_served.serve_concurrency`, `HIGGS_MAX_NUM_SEQS`). Absent means
        that serving width, which is the engine's own configuration and not a
        default substituted for a missing value. MEASURED, 2026-09-19: 16 wide
        at SGLang `mem_fraction_static` 0.60 summed to 24.2 GB on a 24 GB card
        and WDDM paged the excess to host RAM - 4-10x slower, with no error
        anywhere - so a job that knows it wants 4 has to be able to say 4
        without restarting the server. Nothing here reconfigures the server;
        this only limits what narrator asks of it at once.

        `BATCH_SIZE` renders in flight for as long as there is work, exactly as
        `convert_many` describes - that docstring is the rationale for every
        decision made below, and this is the same code it has described since
        2026-09-08, moved out from under `_write_sentence` so that a caller with
        no `sentences_dir` can have the guard too (docs/history/PHASE6-REMOTE-RENDER.md
        section 2). The audiobook world consumes it into files; the serve world
        consumes it into `batch_item` frames with `guard` attached.

        `verdict` is `GuardPlan.verdict(index)` - the conclusion the ladder
        reached, with the take records verbatim behind it. A chunk that FAILED
        for its own reason yields `(index, None, None)`: `audio is None` is the
        failure signal, and there is no verdict because `plan.abandon()` has
        already dropped the records of a chunk that never reached a decision.

        `in_flight` stays the CALLER'S list when it has one - the render
        worker's cleanup deletes exactly the rows it names - and is a throwaway
        when it does not, so the bookkeeping below has one shape rather than a
        `None` test in the hot loop.

        A generator, so the plan and the pool only exist while someone is
        iterating; a consumer that walks away closes it, `GeneratorExit` lands
        at a `yield` inside the `try`, and the pool is torn down by the same
        `BaseException` arm a cooperative stop uses.

        `sampling_by_index`, when given, maps EVERY chunk index in `rows` to
        its take-ladder rung (None for a chunk at take 0). It is a mapping and
        not a third element of each row because the ladder is keyed by chunk
        index and so is this: a split half and a re-roll carry their parent's
        index and must render at their parent's numbers, which is the whole
        point of a retake at a different temperature.

        THE MAP MUST COVER EVERY ROW, and a row it misses is refused by name
        rather than rendered at take 0. `rows` is pulled LAZILY here, so the
        check happens as each row is taken; an absent key would otherwise be a
        `.get()` that silently renders the ladder's rung 1 at rung 0's numbers,
        which is the one failure this whole channel exists to prevent.

        `take_by_index` is the OTHER half of the same rung and keeps the same
        rules: keyed by chunk index because a re-roll and both halves of a
        split must stay in their parent's seed lane, covering every row or
        refused by name. It moves the SEED, not the numbers, so a chunk at take
        3 with no sampling override still renders a draw take 0 never made.
        """
        rows = iter(rows)
        if in_flight is None:
            in_flight = []
        rungs = None if sampling_by_index is None else dict(sampling_by_index)
        takes = None if take_by_index is None else dict(take_by_index)

        def rung_for(index):
            if rungs is None:
                return None
            if index not in rungs:
                raise KeyError(
                    f'HiggsV3Engine.render_many: chunk {index} has no entry in '
                    'sampling_by_index. The map is given per CALL and must name '
                    'every row (None for a chunk at take 0); a missing key would '
                    "render a ladder's rung at take 0's numbers.")
            return rungs[index]

        def take_for(index):
            if takes is None:
                return 0
            if index not in takes:
                raise KeyError(
                    f'HiggsV3Engine.render_many: chunk {index} has no entry in '
                    'take_by_index. The map is given per CALL and must name every '
                    'row (0 for a chunk at take 0); a missing key would render a '
                    "retake in take 0's own seed lane.")
            return takes[index]
        width = int(self.BATCH_SIZE) if width is None else int(width)
        if width < 1:
            raise ValueError(f'render_many needs a width >= 1; got {width}.')
        # THE PLAN IS WHAT VARIES; THE DRIVER AND THE WIDTH DO NOT.
        # `tracker is None` means the batch said `retake: false` — it has no
        # band, so nothing is judged, nothing re-rolled and nothing split. It
        # does NOT mean `GuardPlan(tracker=None)`, which falls back to a fixed
        # band and would judge a screening checkpoint against numbers belonging
        # to some other model.
        #
        # Before 2026-09-20 an unjudged batch never reached here at all: the
        # serve worker ran a sequential comprehension instead, so `retake:
        # false` silently cost the caller its batching as well as its ladder
        # (measured: `#running-req` pinned at 1 across 684 scheduler lines at
        # width 4). One driver, one place that spends the width.
        plan = (truncation.UnjudgedPlan(
                    sample_rate=self.SAMPLE_RATE, base_seed=self.config.seed)
                if tracker is None else
                truncation.GuardPlan(
                    sample_rate=self.SAMPLE_RATE, base_seed=self.config.seed,
                    tracker=tracker))
        pool = ThreadPoolExecutor(max_workers=width, thread_name_prefix='higgs3-render')
        running = {}      # future -> RenderRequest
        outstanding = {}  # chunk index -> how many of its requests are in flight

        def render(request):
            """The pool thread's whole job: one take. No plan, no file, no lock.

            The rung is read by CHUNK INDEX, so every take of a chunk - take 0,
            its re-roll and both halves of a split - renders at the same
            numbers the caller asked that chunk for, and in that chunk's own
            take lane. `request.seed` is the ladder's (None for take 0 of the
            chunk, `reroll_seed` for a re-roll); `_request_seed` shifts
            whichever it is.

            MEASURED, not merely rendered: the pair goes to `plan.offer`,
            which carries the measure to whichever take ships."""
            return self.render_audio_measured(request.text, seed=request.seed,
                                              index=request.index,
                                              sampling=rung_for(request.index),
                                              take=take_for(request.index))

        def start(request) -> None:
            outstanding[request.index] = outstanding.get(request.index, 0) + 1
            if request.index not in in_flight:
                in_flight.append(request.index)
            running[pool.submit(render, request)] = request

        def fill() -> None:
            """EVERY free slot, every time. One submit per retirement was not
            enough the moment a take could produce TWO renders: a split's halves
            are both waiting the instant the re-roll fails, and filling one slot
            left the other half rendering after it - the serialisation this
            driver exists to remove."""
            while len(running) < width and submit_next():
                pass

        def submit_next() -> bool:
            """One more take in flight: a retake the guard is waiting on first,
            else the next chunk's take 0. RETAKES FIRST so a chunk that is part
            way through the ladder finishes rather than accumulating behind a
            book's worth of fresh work."""
            request = plan.next_request()
            if request is None:
                try:
                    index, text = next(rows)
                except StopIteration:
                    return False
                # The rung is resolved HERE, at the door, and not in the pool
                # thread: a row the caller's map does not name is a caller
                # defect, and it must fail the CALL rather than arrive as one
                # chunk's render failure.
                rung_for(index)
                # THE MARKER STRIP, AT THE LADDER'S DOOR, exactly as
                # `_render_guarded` does it for the single-chunk path.
                # `render_audio` strips again at the model boundary and is
                # idempotent about it, but the GUARD measures characters: a
                # chunk carrying `[break]` would be judged, and split, on text
                # that is not what the model will be given.
                plan.add(index, self._clean_sentence_for_tts(text))
                request = plan.next_request()
                if request is None:
                    # `add` without a first take always leaves a take-0 request.
                    raise RuntimeError(
                        f'Higgs v3: chunk {index} entered the ladder with nothing to render.')
            start(request)
            return True

        def retire(request, index_failed=None):
            """Bookkeeping for one finished request, and the chunks it completed.

            A GENERATOR - the caller `yield from`s it - so a decided chunk
            leaves on THIS thread at exactly the point the file used to be
            written, before the freed slots are refilled. Nothing about the
            ordering changes; only what happens to the audio does.
            """
            outstanding[request.index] -= 1
            if outstanding[request.index] <= 0:
                outstanding.pop(request.index, None)
                if request.index in in_flight:
                    in_flight.remove(request.index)
            for index, audio, _clean in plan.finished():
                # `finish_root` files the verdict BEFORE it appends to
                # `finished()`, so it is there; `verdict()` pops, so it is there
                # exactly once and only for a chunk the ladder really shipped.
                # The measure is filed and popped the same way.
                yield index, audio, plan.verdict(index), plan.measure(index)
            if index_failed is not None:
                # No audio, no verdict and nothing measured: a chunk that never
                # reached a decision spent takes nobody shipped.
                yield index_failed, None, None, None

        try:
            fill()
            while running:
                done, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for future in done:
                    request = running.pop(future)
                    try:
                        audio, measure = future.result()  # re-raises HiggsV3ServerDown
                    except v3_served.HiggsV3ServerDown:
                        raise
                    except Exception as exc:
                        # This chunk's failure, named, and the take goes on.
                        log(f'[HIGGS3] sentence {request.index} failed: {exc}', flush=True)
                        failed = plan.abandon(request)
                        yield from retire(request, index_failed=failed)
                        fill()
                        continue
                    plan.offer(request, audio, measure)
                    yield from retire(request)
                    fill()
        except BaseException:
            # A stop, a dead server, or a consumer that closed this generator
            # (`GeneratorExit` lands at the `yield from` above and is a
            # BaseException, which is why this arm catches that and not
            # Exception): nothing queued is submitted, and the requests already
            # running are not waited for - they fail fast against a killed port,
            # and `in_flight` still names their chunks for the worker's cleanup.
            # `shutdown(wait=True)` here would hold a cooperative stop open for
            # as long as the slowest HTTP timeout.
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)

    def generate_batch_stream(self, texts, voices, stream_rows, on_chunk, on_row,
                              should_stop=None, samplings=None, takes=None) -> None:
        """Whole rows, at retirement.

        The buffered endpoint returns a finished wav, so there is nothing to
        emit until the row is done. A row asked to stream gets its audio as a
        single `on_chunk(row, 0, audio)` before its `on_row` - the streaming
        channel, honestly filled, arriving when it actually arrived.

        `samplings`, when given, is aligned to `texts` and carries each row's
        take-ladder rung. Sampling is PER REQUEST on both serving stacks, so a
        batch here mixes rungs freely: the pool submits one HTTP request per
        row and each carries its own numbers.

        `takes` is aligned the same way and carries each row's take, which
        moves that row's SEED into its own lane. Mixing is free here for the
        same reason: the seed is a per-request field, one HTTP call per row.
        """
        from ..item_sampling import aligned, takes_aligned
        if not texts:
            return
        if voices is not None and len(voices) != len(texts):
            raise ValueError(
                f'HiggsV3Engine.generate_batch_stream: {len(voices)} voices for '
                f'{len(texts)} texts; voices must be aligned to texts or None')
        if voices is not None:
            wrong = sorted({v for v in voices if v and v != self.voice})
            if wrong:
                raise ValueError(
                    f'HiggsV3Engine.generate_batch_stream: rows ask for voice(s) '
                    f"{wrong}; this engine serves '{self.voice}'. A v3 voice change is "
                    'a server restart, so a mixed-voice batch is impossible.')
        stream_rows = set() if stream_rows is None else set(stream_rows)
        stray = [i for i in stream_rows if not (0 <= i < len(texts))]
        if stray:
            raise ValueError(
                f'HiggsV3Engine.generate_batch_stream: stream_rows names row(s) '
                f'{stray} outside the batch of {len(texts)}')
        if stream_rows and on_chunk is None:
            raise ValueError(
                'HiggsV3Engine.generate_batch_stream: stream_rows is non-empty but no '
                'on_chunk was given')
        blank = [i for i, t in enumerate(texts) if not (t or '').strip()]
        if blank:
            raise ValueError(
                f'HiggsV3Engine.generate_batch_stream: row(s) {blank} have no text '
                'after cleaning.')
        rungs = aligned(samplings, len(texts),
                        'HiggsV3Engine.generate_batch_stream')
        rows_takes = takes_aligned(takes, len(texts),
                                   'HiggsV3Engine.generate_batch_stream')

        def render(i: int, text: str):
            # Checked at the moment the row would be POSTed, not at submission:
            # rows queued behind the in-flight ones must not start once the
            # caller has asked for a stop. An abandoned row gets no on_row at
            # all, which is what the contract requires.
            if should_stop is not None and should_stop():
                return i, None, True
            return i, self.render_audio(text, index=i, sampling=rungs[i],
                                        take=rows_takes[i]), False

        width = min(self.BATCH_SIZE, len(texts))
        with ThreadPoolExecutor(max_workers=width,
                                thread_name_prefix='higgs3-stream') as pool:
            futures = [pool.submit(render, i, text)
                       for i, text in enumerate(texts)]
            # Callbacks run HERE, on the caller's thread, as rows retire - the
            # serve worker's callbacks take their own locks but were written
            # for one caller at a time, and this keeps that true.
            for future in as_completed(futures):
                try:
                    i, audio, abandoned = future.result()
                except v3_served.HiggsV3ServerDown:
                    raise
                except Exception as exc:
                    # Which row is not recoverable from the exception; find the
                    # future's index and report that row failed.
                    i = futures.index(future)
                    log(f'[HIGGS3] row {i} failed: {exc}', flush=True)
                    on_row(i, None)
                    continue
                if abandoned:
                    continue
                if i in stream_rows:
                    on_chunk(i, 0, audio.copy())
                on_row(i, audio)


def higgs_v3_config_from_worker_kwargs(voice=None, model_dir=None, base_dir=None,
                                       adapter_dir=None, caps=None, **extra):
    """Build a HiggsV3Config from the keywords `narrator.serve` hands an engine.

      voice       the voice NAME, looked up in the NARRATOR_HIGGS_VOICES document
      adapter_dir REFUSED - see below. A fine-tuned Higgs voice is a merged
                  CHECKPOINT and it comes from the voice document.
      model_dir   REFUSED - the model the server serves is the launch script's
                  argument, not a per-load field
      base_dir    REFUSED - v3 has no shared-base + adapter split of Orpheus's kind
      caps        REFUSED - the Orpheus cap names have no meaning here
    """
    from .config import load_voice
    if model_dir:
        raise ValueError(
            f'Higgs v3 load carried modelDir={model_dir!r}. The served model is the '
            "launch script's argument (serve_v3.sh), not a per-load field: changing "
            'it is a server restart, not a message.')
    if base_dir:
        raise ValueError(
            f'Higgs v3 load carried baseDir={base_dir!r}. v3 has no shared-base + '
            'per-voice-adapter split at all: a fine-tuned voice is a whole merged '
            'checkpoint the server runs ON (v3_served.CHECKPOINT_STRATEGY).')
    if caps:
        raise ValueError(
            f'Higgs v3 load carried caps={sorted(caps)}. Those are ORPHEUS tuning '
            'caps and v3 implements none of them - its sampling rides in '
            'extra_params and its default is the server\'s own. Refusing a payload '
            'that would look applied and do nothing.')
    if adapter_dir:
        # THE SAME REFUSAL `load_voices` MAKES, at the other boundary. The pool's
        # field is called `adapterDir`, and taking it as a merged checkpoint
        # would accept an actual LoRA directory here and hand it to
        # `vllm-omni serve` - which cannot load one (no adapter flags; the
        # talker does not implement SupportsLoRA) and would come up on garbage
        # or on the base model. The two are not interchangeable and the field
        # name cannot tell them apart, so this end refuses too.
        raise ValueError(
            f'Higgs v3 load carried adapterDir={adapter_dir!r}. There is no '
            'runtime LoRA on this stack, so an adapter directory has nothing to '
            'be loaded into - and it is NOT a merged checkpoint. A fine-tuned '
            "voice is declared in the NARRATOR_HIGGS_VOICES document as "
            '{"kind": "checkpoint", "checkpointDir": ...}, and the server is '
            'started on that directory.')
    if extra:
        raise ValueError(f'Higgs v3 load carried unknown keys: {sorted(extra)}.')
    if not (voice or '').strip():
        raise ValueError(
            'Higgs v3 load has no voice name. The name selects an entry in the '
            'NARRATOR_HIGGS_VOICES document; there is no default, and the model\'s '
            'own voice sits at 12 % of the narrator ceiling.')

    resolved = load_voice(
        voice.strip(),
        allowed_controls=HiggsV3Defaults.ALLOWED_CONTROLS,
        max_reference_seconds=HiggsV3Defaults.MAX_REFERENCE_SECONDS,
        placeholder_max_chars=HiggsV3Defaults.MAX_CHARS)
    strategy = (os.environ.get('NARRATOR_HIGGS3_ADAPTER_STRATEGY') or '').strip()
    if strategy:
        # Retained as a refusal, not a switch: the only strategy is 'checkpoint'
        # and the retired names say why in their own message.
        v3_served.check_strategy(strategy)
    # THE VOICE'S OWN SAMPLING, when the document states one, rides as the
    # config's override (`served_sampling` merges it over the checkpoint's file
    # or the deploy default; on SGLang `applied_sampling` does the same). The
    # MLX arm took this key on 2026-09-06 (13dc0e66); until this line the served
    # arm read the same document and dropped it, so the catalog's served block
    # was documentary here the way the MLX block had been on the Mac.
    return HiggsV3Config(
        voice=resolved,
        checkpoint_dir=getattr(resolved, 'checkpoint_dir', None),
        base_dir=getattr(resolved, 'base_dir', None),
        # THE WINDOW THE SERVER WAS LAUNCHED WITH, not the class default: this
        # bounds `max_total_tokens`, so a prompt is refused against the context
        # that actually exists rather than against 8192 forever.
        context_tokens=v3_served.context_tokens(),
        sampling=getattr(resolved, 'sampling', None))
