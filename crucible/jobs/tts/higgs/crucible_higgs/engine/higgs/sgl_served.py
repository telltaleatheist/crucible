"""Higgs TTS 3 served by **SGLang-Omni 0.1.4** - the second serving stack.

WHY THERE IS A SECOND STACK AT ALL. Measured by the training side on the night of
2026-09-05 (docs/history/HIGGS_FIELD_NOTES.md 4n, `night3/sgl/`): the same 50 packed chunks of
*Working Towards the Fuhrer*, the same merged checkpoint (ckpt-1080), the same
sampling, one seed, scored by `higgs_ladder.py score` (whisper coverage,
max_skip_words, insert_rate, early-stop) plus 3 s ECAPA windows for voice
switches:

    engine, in flight        early stops   damaged/50   sustained switches   chars/min
    vllm-omni 0.28.0, 1          0             5                0              1,064
    vllm-omni 0.28.0, 16         4            13                6             10,752
    SGLang-Omni 0.1.4, 16        0             5                0             26,666
    SGLang-Omni 0.1.4, 1         1             7                0              2,636

vllm-omni's damage at width is its BATCHED TALKER, not the model and not the
text: at 1 in flight the same build is clean, and the damage lands on the
NEWEST batch row - the request that prefills while others decode, which is the
row vLLM's condense moves when an earlier row retires (`pt_condense`: long B
damaged 3/6 while the short and long A were clean 12/12). Upstream knows half of
it (issue #6418, PR #6422's request-id keyed state pool, which IS in the
installed talker and does not fix it). The 6 sustained voice switches Owen hears
are the same defect: 12 of 50 chunks carry a 3 s ECAPA window under the 0.50 the
narrator never produces, against 0 at width 1 and 0 on this stack.

So this stack is 2.5x the throughput AND the clean one at 16 wide. What remains
damaged on every clean engine (~10 %) is text-domain - em-dashes missing from
the corpus, a footnote welded onto a chunk head, umlauts - and is addressed
elsewhere.

WHAT IS DIFFERENT FROM `v3_served.py`, and every one of these is a way to be
silently wrong:

  1. SAMPLING RIDES AT THE REQUEST'S **TOP LEVEL**, and it MUST BE SENT.
     `sglang_omni/serve/protocol.py:CreateSpeechRequest` has real
     `temperature` / `top_p` / `top_k` / `repetition_penalty` / `seed` fields -
     there is no `extra_params` - and
     `models/higgs_tts/request_builders.py:build_sglang_higgs_request` sets
     `top_p`/`top_k` on the SamplingParams ONLY when the request carried them.
     There is no deploy profile and no `generation_config.json` read: **a request
     that sends nothing samples the untruncated tail.** Measured 2026-09-05:
     without top_k one chunk ran to the cap with 80 s of silence. This is the
     exact opposite of vllm-omni, where a checkpoint voice sends NOTHING because
     the server reads the checkpoint's own `generation_config.json`
     (`--generation-config auto`). Hence `HiggsV3Config.served_sampling` branches
     on the stack, and `build_request_body` here REFUSES an empty mapping.
  2. THE FRAME CAP FIELD IS `max_new_tokens`, not `max_tokens`.
  3. THE CONTEXT IS A CLASS ATTRIBUTE, NOT A FLAG, and 4096 by default.
     `models/higgs_tts/engine_builder.py:29`:
     `class HiggsTtsEngineBuilder: context_length = 4096`. Three flag spellings
     were tried and recorded as failures (2026-09-09), so narrator launches the
     server through its own entry module, which ASSIGNS that attribute from
     `HIGGS_CONTEXT_LENGTH` in the server process before the engine is built
     (`launch/sgl_omni_entry.py`). `context_tokens()` here reads the same
     variable, so the client sizes requests against the window the server is
     actually serving. Prompt tokens + `max_new_tokens` must fit it or the
     request is an HTTP 500 - see `frame_cap`, which refuses BY NAME before
     anything is sent.

     SGLANG'S OTHER LENGTH REFUSAL IS NOT THIS ONE AND IS LEFT ALONE:
     `sglang_omni/serve/speech_service.py:58` caps the request TEXT at
     `MAX_SPEECH_INPUT_CHARS = 4096` and answers HTTP 400 before generation. It
     is a character count on the input, not a token window, and it is the
     server's to own.
  4. THE REFERENCE RIDES IN THE BODY, AS BASE64. `CreateSpeechRequest
     .references[]` is a `SpeechReference` with a `data` field (raw base64, no
     `data:` prefix) beside `media_type` and `text`; `speech_service.py
     ._normalize_reference` takes that branch FIRST and never touches the
     server's filesystem, so no `--allowed-local-media-path` is needed and
     `serve_higgs_sgl.sh` still hands the server no directory. (`audio_path`
     is the other branch - a server-local path or a `file://` URL - and this
     module never sends it.) The reference is charged to the context: one
     AUDIO_PLACEHOLDER per DELAYED reference row (`build_prompt(num_ref_tokens=
     delayed.shape[0])`, and `apply_delay_pattern` makes T frames T + 7 rows at
     8 codebooks) plus `ref_text_id`, the transcript's tokens and
     `ref_audio_id` - see `reference_token_bound`. A 30 s clip is ~760 of the
     4,096 positions, which at the zero-shot chunk length (600 chars, the
     measured wall - 900 drops the tail) leaves the frame cap its full 2.0x
     ceiling. Until 2026-09-06 a clips voice was refused on this stack outright,
     on the belief that `audio_path` was the only way in; the `data` branch was
     read off sglang-omni 0.1.4's own `serve/speech_service.py` and
     `models/higgs_tts/utils.py:load_audio_to_24k`.
  5. NO SENTINEL PATCH. That patch is a vllm-omni site-packages fix; SGLang-Omni
     has its own stage processor and `HIGGS_PATCHES` applies only to the other
     stack. There is therefore no `verify_sentinel_filter` here and BookForge's
     doctor does not report patch rows for this stack.
  6. IDENTITY IS NOT IN `/v1/models`. `sglang_omni/serve/openai_api.py`'s
     `_register_models` builds `ModelCard(id=model_name, root=model_name)` - the
     SERVED NAME, twice - so unlike vllm-omni 0.28 it never reports the model
     PATH. `running_checkpoint` therefore reads `HIGGS_MODEL_DIR` out of
     `/proc/<listener pid>/environ`, which is what our own launcher exported into
     it: a fact about the running process rather than a claim by the process
     asking.

THE LAUNCH (`engine/higgs/launch/serve_higgs_sgl.sh`, narrator's OWN copy since
2026-09-15 and shipped as package data; `electron/scripts/higgs/` holds the
byte-identical copy BookForge's installer deploys, kept in step by
`tools/test-higgs-engine.js` and dying with the legacy local render path. It is
transcribed from `night3/sgl/serve_sgl.sh`):

    sgl-omni serve --model-path <merged dir> --model-name higgs-v3-ds
      --host <host> --port <port> --mem-fraction-static 0.60
      --tts_engine.factory.max_running_requests <N>
      --tts_engine.factory.cuda_graph_max_bs <N>
      --tts_engine.factory.max_new_tokens 7500

in conda env `sglomni` (python 3.12, torch 2.13.0+cu130, sglang 0.5.18,
sglang-omni 0.1.4, flashinfer 0.6.17 + flashinfer-jit-cache cu130), with
`CUDA_HOME=<env>/lib/python3.12/site-packages/nvidia/cu13` and that directory
carrying `lib64 -> lib` and `libcudart.so -> libcudart.so.13`. Healthy in ~110 s,
~19 GB at 16 in flight, CUDA graphs captured on sm_86.

LICENCE is the model's, unchanged: Boson Higgs TTS 3 Research and
Non-Commercial.
"""
import contextlib
import importlib.resources
import json
import math
import os
import shlex
import shutil
import sys
import urllib.error
import urllib.request

from ..log import log
from ..protocol import BackendSpec, ClipsVoice, DefaultVoice
from . import served_common, v3_served
from .served_common import (
    CHECKPOINT_ENV,
    GuestOwnedServer,
    HiggsServerDown,
    HiggsServerError,
    to_wsl,
)

#: `--model-name` on the launch line, the `model` field of every request, and
#: the id `/v1/models` reports. DELIBERATELY NOT `higgs-v3`: a port is not proof
#: of identity, and the two stacks answer the same shaped endpoints, so a name
#: that differs is one more way a leftover server on the wrong port is caught.
#: What `CHECKPOINT_ENV` may say instead of a path, meaning "this server is on
#: the BASE weights". Lower-cased on read. A path is always absolute, so this
#: cannot collide with one.
BASE_ASSERTION = 'base'

SERVED_MODEL_NAME = 'higgs-v3-ds'

SPEECH_PATH = '/v1/audio/speech'
HEALTH_PATH = '/health'
MODELS_PATH = '/v1/models'

#: Attach to an ALREADY-RUNNING SGLang-Omni server instead of launching one.
BASE_URL_ENV = 'NARRATOR_HIGGS_SGL_URL'
#: Where the launch script lives, in the filesystem the spawn will read it from.
#: An OPERATOR'S OVERRIDE of narrator's own packaged launcher, exactly as
#: `v3_served.SERVE_SCRIPT_ENV` is - see `packaged_serve_script`.
SERVE_SCRIPT_ENV = 'NARRATOR_HIGGS_SGL_SERVE_SCRIPT'

#: NARRATOR'S OWN SGLang LAUNCHER, shipped as package data beside the vllm-omni
#: one (`launch/` beside this module, package data of `crucible` in Crucible's
#: pyproject.toml).
#:
#: WHY IT IS HERE AT ALL. Until 2026-09-15 this file had only the operator
#: override above, so the only copy of the script lived in BookForge's
#: `electron/scripts/higgs/` and any other client of narrator - Crucible first -
#: had to be handed a path into a BookForge checkout or this stack could not
#: start. That is precisely the dependency BookForge 0eeb0267 removed for
#: `serve_higgs_v3.sh`, and it was still in place for the stack Owen actually
#: renders on ("we dont use vllm-omni. we use sglang. vllm-omni doesnt work for
#: higgs", 2026-09-15).
#:
#: ONE FILE, NOT TWO, and that is a fact about the stack rather than an
#: omission. The vllm-omni launcher travels with `higgs_default_frames7500.yaml`
#: because stage 0's `max_tokens` is a HARD per-render ceiling that only a
#: deploy profile can raise. SGLang-Omni has no deploy profile at all - which is
#: also why sampling must ride on every request here - so there is no sibling
#: for `$(dirname "$0")` to find and none is invented.
PACKAGED_SERVE_SCRIPT = 'serve_higgs_sgl.sh'

#: WHERE THE LAUNCHER CAME FROM, reported on the ready path the way the
#: vllm-omni arm reports its own (`v3_served.LAUNCHER_*`). Three modes, decided
#: in the constructor and nowhere else.
LAUNCHER_ATTACH = v3_served.LAUNCHER_ATTACH
LAUNCHER_OPERATOR = v3_served.LAUNCHER_OPERATOR
LAUNCHER_PACKAGED = v3_served.LAUNCHER_PACKAGED
#: The WSL distro to run the launch script in, on Windows. Shared with the
#: vllm-omni arm: one machine, one guest.
WSL_DISTRO_ENV = v3_served.WSL_DISTRO_ENV
#: Where an ATTACHED server's log is, named by the operator. NO DEFAULT - see
#: `v3_served.SERVER_LOG_ENV` for why a guessed path is worse than none.
SERVER_LOG_ENV = 'NARRATOR_HIGGS_SGL_SERVER_LOG'

#: THE LAUNCH SCRIPT'S OWN KNOBS (`serve_higgs_sgl.sh`), exported into the
#: wrapper by narrator at launch exactly as the vllm-omni arm exports its set.
SERVE_HOST_ENV = 'HIGGS_SGL_HOST'
SERVE_PORT_ENV = 'HIGGS_SGL_PORT'
SERVE_MEM_FRACTION_ENV = 'HIGGS_SGL_MEM_FRACTION'
SERVE_MAX_NEW_TOKENS_ENV = 'HIGGS_SGL_MAX_NEW_TOKENS'
SERVE_CUDA_GRAPH_MAX_BS_ENV = 'HIGGS_SGL_CUDA_GRAPH_MAX_BS'
#: THE ONE CONCURRENCY NUMBER, shared with the vllm-omni arm on purpose: it is
#: the server's admission width AND the width of narrator's own batch, and one
#: number stated once is what stops the two disagreeing.
SERVE_MAX_NUM_SEQS_ENV = v3_served.SERVE_MAX_NUM_SEQS_ENV
SERVE_MODEL_DIR_ENV = served_common.SERVE_MODEL_DIR_ENV

#: The launch script's own bind defaults, mirrored so a launch with neither
#: variable set polls the port the script binds. Not a fallback that hides a
#: bug: both sides read the same two literals and narrator EXPORTS the pair it
#: chose into the wrapper. Port 8200 keeps this stack's server unmistakable from
#: vllm-omni's 8095 - a leftover on the wrong port is then a different port, not
#: a subtle identity question.
SERVE_DEFAULT_HOST = '127.0.0.1'
SERVE_DEFAULT_PORT = 8200

#: The log narrator writes a server it LAUNCHED to, inside the session's process
#: dir when there is one. One per SESSION, overwritten per start.
SERVER_LOG_NAME = 'higgs-sgl-server.log'

#: MEASURED COLD START: ~110 s to health on example-pc with warm weights
#: (`night3/logs/serve_events.log`, 2026-09-05). narrator waits far longer (see
#: `HiggsV3Defaults.READY_TIMEOUT_SECONDS`) because `wait_ready` raises
#: immediately if the process actually died, so patience costs nothing when the
#: server is genuinely coming up.
COLD_START_SECONDS = 110

# ---------------------------------------------------------------------------
# The 4096-token context, which is the whole of what makes this stack different
# to send to
# ---------------------------------------------------------------------------

#: THE BUILDER'S OWN CONTEXT. `sglang_omni/models/higgs_tts/engine_builder.py:29`:
#: `class HiggsTtsEngineBuilder: context_length = 4096`. A class attribute with
#: no CLI flag and no config path - three flag spellings were tried and recorded
#: as failures (2026-09-09), and the only thing that reaches it is an
#: ASSIGNMENT in the server process. narrator's launch entry does exactly that
#: (`launch/sgl_omni_entry.py`), which is why this is now a DEFAULT and not a
#: hard limit: unset `HIGGS_CONTEXT_LENGTH` serves this number, and a launch
#: that set one serves that.
BUILDER_DEFAULT_CONTEXT_TOKENS = 4096


def context_tokens() -> int:
    """THE CONTEXT THIS PROCESS'S SERVER WAS LAUNCHED WITH.

    A FUNCTION, not a constant, since 2026-09-19. It used to be `CONTEXT_TOKENS
    = 4096` and every request in the book was sized against that literal; when
    the entry module made the window settable, a constant would have gone on
    sizing chunks for a 4096-token server while an 8192-token one was running -
    the client under-filling every chunk and nobody able to see why. One
    variable (`HIGGS_CONTEXT_LENGTH`), one reader, both sides of the launch.
    """
    return served_common.context_length(BUILDER_DEFAULT_CONTEXT_TOKENS)


def max_context_positions() -> int:
    """Prompt tokens + `max_new_tokens` must be at most this, or the request is
    an HTTP 500 from inside the scheduler. One position is reserved for the
    position the last generated token occupies."""
    return context_tokens() - 1

#: THE PROMPT'S FIXED SCAFFOLD, EXACTLY THREE TOKENS, read off
#: `sglang_omni/models/higgs_tts/text_tokenizer.py:HiggsTokenizerAdapter
#: .build_prompt`: for a zero-shot request the ids are
#: `[tts_id] + encode(text) + [text_id] + [audio_id]`. Not an estimate - the
#: three are literal `ids.append(...)` calls with no branch a text-only request
#: can take. The reference branch is bounded separately - see
#: `reference_token_bound`.
PROMPT_SCAFFOLD_TOKENS = 3

#: THE REFERENCE BRANCH'S OWN SCAFFOLD, EXACTLY TWO TOKENS: `ref_text_id` before
#: the transcript and `ref_audio_id` before the placeholders, read off the same
#: `build_prompt` (`if reference_text and num_ref_tokens > 0: ids.append(
#: self.ref_text_id) ...; if num_ref_tokens > 0: ids.append(self.ref_audio_id)`).
REFERENCE_SCAFFOLD_TOKENS = 2

#: THE DELAY PATTERN'S EXTRA ROWS. `num_ref_tokens` is `delayed.shape[0]`
#: (`stages.py`, both the preprocessing and the GPU path), and
#: `utils.apply_delay_pattern` turns `[T, N]` raw codes into `[T + N - 1, N]` -
#: so a clip of T frames costs T + 7 placeholder positions at Higgs's 8
#: codebooks (`codec.py`'s `(batch, seq, 8)`).
REFERENCE_DELAY_ROWS = 7

#: CHARACTERS PER TOKEN, AS A **FLOOR**, so `prompt_token_bound` is an upper
#: bound on the prompt and never an average.
#:
#: MEASURED 2026-09-06 with `ds_ad4lm_prod_ckpt1080/tokenizer.json` (the Qwen3
#: text tokenizer this model family carries) over the 50 packed Fuhrer chunks the
#: whole night-3 comparison was run on, 26 to 1,190 characters:
#:
#:     min chars/token   3.25   (the 26-char chunk: 8 tokens)
#:     >= 250 chars      3.68 and up
#:     the longest       1,190 chars -> 261 tokens (4.56 chars/token)
#:
#: 3.0 sits under every one of those, so the bound is genuinely a ceiling on real
#: book prose. IT IS NOT A GUARANTEE for every possible string - a chunk of dense
#: CJK or of pure punctuation would tokenize below it - and the consequence of
#: being wrong is the HTTP 500 this guard exists to pre-empt, reported as that
#: chunk's failure. That is stated rather than papered over: the alternative,
#: loading the checkpoint's tokenizer here, would make every render depend on a
#: `tokenizers` import inside whichever env the worker happens to be in, and on
#: an attached server naming a checkpoint directory it may not have.
CHARS_PER_TOKEN_FLOOR = 3.0

#: THE MINIMUM SLACK a frame cap must leave over the chunk's EXPECTED length
#: before the cap is worth sending at all.
#:
#: `max_new_tokens` is a CEILING, not a target: `v3_served.cap_frames` asks for
#: 2.0x the expected duration plus 150 frames because a generous ceiling costs
#: nothing on a stack with an 8,192-position window. Here the context is a
#: SECOND, smaller ceiling, and the smaller of two ceilings is the one that
#: applies - so `frame_cap` sends `min(cap_frames, headroom)` rather than a
#: number the server would 500 on.
#:
#: What that `min` must NEVER become is a silent clamp that cuts speech, so the
#: point at which the remaining headroom stops being headroom is refused BY NAME:
#: below 1.2x the expected length the ceiling would start landing inside real
#: audio, and the honest answer is that this chunk does not fit this stack. At
#: the measured book pace (5.82 s per 100 characters, i.e. ~1,455 frames per
#: 1,000 characters) that boundary is around 1,900 characters - well above every
#: certified Higgs targetChars - so a refusal here means a packing decision to
#: revisit, not a routine event.
MIN_CAP_SLACK = 1.2


def reference_token_bound(voice) -> int:
    """An UPPER BOUND on the prompt positions `voice`'s reference costs; 0 for a
    voice with no reference (a DefaultVoice, or None).

    Per `build_prompt`'s reference branch: `REFERENCE_SCAFFOLD_TOKENS`, the
    transcript at `CHARS_PER_TOKEN_FLOOR`, and one placeholder per delayed
    reference row - the clip's DECLARED seconds at the codec's
    `v3_served.FRAMES_PER_SECOND` (25), rounded up, plus `REFERENCE_DELAY_ROWS`.
    The seconds come from the clip's declaration, never from opening the file,
    for the same reason `v3_served.reference_seconds` reads them that way; a
    clip that declares none is refused there BY NAME.
    """
    if not isinstance(voice, ClipsVoice):
        return 0
    seconds = v3_served.reference_seconds(voice)
    transcript_chars = sum(len(clip.transcript or '') for clip in voice.clips)
    rows = int(math.ceil(seconds * v3_served.FRAMES_PER_SECOND)) + REFERENCE_DELAY_ROWS
    return (REFERENCE_SCAFFOLD_TOKENS
            + int(math.ceil(transcript_chars / CHARS_PER_TOKEN_FLOOR))
            + rows)


def prompt_token_bound(text: str, voice=None) -> int:
    """An UPPER BOUND on the prompt tokens SGLang-Omni will build for `text` in
    `voice` (the reference, when the voice carries one - `reference_token_bound`).

    `PROMPT_SCAFFOLD_TOKENS` (exact, read off the builder) plus the text at
    `CHARS_PER_TOKEN_FLOOR` characters per token (a measured floor - see that
    constant). Deliberately one rule with no branch: a bound that is sometimes
    the tokenizer's real count and sometimes an estimate is a bound nobody can
    reason about, and the two would disagree exactly where it matters.
    """
    chars = len(text or '')
    return (PROMPT_SCAFFOLD_TOKENS
            + int(math.ceil(chars / CHARS_PER_TOKEN_FLOOR))
            + reference_token_bound(voice))


def expected_frames(text: str) -> int:
    """How many LM frames this chunk's audio is expected to take, with no slack:
    `v3_served.cap_frames`'s own model of the pace (15 chars/s at 25 fps) without
    its 2.0x ceiling factor and its 150-frame tail."""
    return int(len(text or '') / 15.0 * v3_served.FRAMES_PER_SECOND)


def frame_cap(text: str, voice=None) -> int:
    """`max_new_tokens` for one chunk on this stack, or a refusal BY NAME.
    `voice` is charged for its reference, when it carries one.

    THE TWO CEILINGS. `v3_served.cap_frames(text)` is narrator's own generous one
    (2.0x expected + 150). `max_context_positions() - prompt_token_bound(text)` is
    the stack's, and it is hard: prompt + `max_new_tokens` over 4,095 is an HTTP
    500 from inside the scheduler, not a shorter render. The smaller applies.

    THE REFUSAL. When what the context leaves is under `MIN_CAP_SLACK` x the
    chunk's expected length, the ceiling would fall inside real speech, and this
    raises instead - naming the chunk length, the numbers, and the lever
    (the voice's `targetChars`). Silently sending the smaller number there would
    be a cut chunk reported as a success, which is the failure mode this whole
    module exists downstream of.
    """
    if not (text or '').strip():
        raise ValueError('Higgs SGLang: no text to size a frame cap for.')
    bound = prompt_token_bound(text, voice)
    headroom = max_context_positions() - bound
    expected = expected_frames(text)
    if headroom < expected * MIN_CAP_SLACK:
        reference = reference_token_bound(voice)
        raise ValueError(
            f'Higgs SGLang-Omni: a {len(text)}-character chunk does not fit this '
            f"stack's context. SGLang-Omni's Higgs builder hard-codes "
            f'context_length {context_tokens()} (engine_builder.py, no flag), so '
            f'prompt + max_new_tokens must be at most {max_context_positions()}; '
            f'this prompt is at most {bound} tokens'
            + (f' ({reference} of them the reference clip and its transcript)'
               if reference else '')
            + f', leaving {headroom} frames '
            f'against an expected {expected} - under the {MIN_CAP_SLACK}x slack '
            'below which the cap would land inside real speech and cut the chunk '
            'while the request reported success. Lower this voice\'s targetChars '
            'in electron/data/higgs-models.json, or render it on the vllm-omni '
            'stack, whose window is 8192.')
    return min(v3_served.cap_frames(text), headroom)


# ---------------------------------------------------------------------------
# The reference
# ---------------------------------------------------------------------------

#: The one media type this module declares for a reference. The clip is read as
#: bytes and the server decodes them by THIS label (`load_audio_to_24k` ->
#: `io.load_base64(media_type, data)`), so a clip that is not a wav would be
#: decoded as one - refused by extension instead, see `reference_for`.
REFERENCE_MEDIA_TYPE = 'audio/wav'


def reference_for(voice: ClipsVoice) -> dict:
    """The single `references` entry for `voice`, in `SpeechReference`'s shape:
    `{data, media_type, text}` - raw base64 in `data`, NO `data:` prefix
    (`_normalize_reference` hands it straight to `load_base64`; a prefixed
    string fails its base64 validation).

    The one-clip rule and the 30 s budget are `v3_served.reference_for`'s, and
    they are the same rule here: sglang-omni's own reference ceiling is 100 s
    (`stages._MAX_REF_AUDIO_SEC`), but the measured zero-shot behaviour, the
    catalog's `referenceSecondsCap` and the MLX arm all speak the 30 s number,
    and a stack-specific budget is a voice that renders on one machine and is
    refused on the other.
    """
    if not isinstance(voice, ClipsVoice):
        raise ValueError(
            f'Higgs SGLang reference_for takes a ClipsVoice; got '
            f'{type(voice).__name__}.')
    v3_served.check_reference_budget(voice)
    if len(voice.clips) != 1:
        raise ValueError(
            f"Higgs SGLang voice '{voice.name}' has {len(voice.clips)} reference "
            'clips. One reference per request, the same rule as vllm-omni: '
            'several clips must be pre-joined into one wav (clips separated by '
            f'{v3_served.REFERENCE_JOIN_SECONDS} s of silence) with the '
            'transcripts joined in the same order.')
    clip = voice.clips[0]
    if not os.path.isfile(clip.path):
        raise ValueError(f'Higgs SGLang reference clip does not exist: {clip.path}')
    if os.path.splitext(clip.path)[1].lower() != '.wav':
        raise ValueError(
            f"Higgs SGLang reference clip {clip.path} is not a .wav. The request "
            f'declares the bytes as {REFERENCE_MEDIA_TYPE} and the server decodes '
            'them as that; convert the clip rather than mislabel it.')
    if not (clip.transcript or '').strip():
        raise ValueError(
            f"Higgs SGLang voice '{voice.name}': reference clip {clip.path} has no "
            'transcript. The server frames the clone prompt as <|ref_text|> '
            'transcript <|ref_audio|> clip, and a clip without its book-exact text '
            'conditions every chunk on words it never heard.')
    import base64
    with open(clip.path, 'rb') as handle:
        encoded = base64.b64encode(handle.read()).decode('ascii')
    return {'data': encoded, 'media_type': REFERENCE_MEDIA_TYPE,
            'text': clip.transcript}


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------

#: The sampling fields `CreateSpeechRequest` actually has, at the TOP LEVEL.
#: `seed` is excluded here and passed separately for the same reason it is on the
#: vllm-omni arm: it has exactly one home in the body and a duplicate is refused.
SAMPLING_KEYS = ('temperature', 'top_p', 'top_k', 'repetition_penalty')

_WHY_SAMPLING_IS_REQUIRED = (
    'SGLang-Omni applies NO top_k and NO top_p unless the request carries them: '
    'build_sglang_higgs_request sets them on the SamplingParams only when the '
    'request field is not None, and there is no deploy profile and no '
    'generation_config.json read on this stack. Measured 2026-09-05: without '
    'top_k one chunk ran to the cap with 80 s of silence. So an empty sampling '
    'here is not "the server\'s defaults" - it is the untruncated 1026-way '
    'codebook tail.')


def build_request_body(text: str, voice, max_new_tokens: int, seed=None,
                       sampling=None, model: str = SERVED_MODEL_NAME) -> dict:
    """The POST body for one chunk, as `higgs_ladder.py render` sends it.

    `voice` is a DefaultVoice (the model's own speaker, or a fine-tune whose
    weights ARE the voice - either way no `references` key), a ClipsVoice (one
    `references` entry, base64 in the body - `reference_for`), or None.

    `sampling` is REQUIRED and non-empty. That is the opposite of
    `v3_served.build_request_body`, where an empty mapping correctly means "the
    checkpoint's own generation_config.json decides" - here nothing reads that
    file and an empty mapping means the untruncated tail.
    """
    if not (text or '').strip():
        raise ValueError('Higgs SGLang request: no text')
    v3_served.validate_control_tokens(text)
    if voice is not None and not isinstance(voice, (DefaultVoice, ClipsVoice)):
        raise ValueError(
            f'Higgs SGLang-Omni got a {type(voice).__name__} voice. A voice here '
            'is a DefaultVoice (the merged checkpoint the server was started on, '
            "or the base model's own speaker) or a ClipsVoice (a reference clip "
            'in the request).')
    if not sampling:
        raise ValueError(
            'Higgs SGLang-Omni request carries no sampling. '
            + _WHY_SAMPLING_IS_REQUIRED
            + " HiggsV3Config.served_sampling() is what states it: from a "
            "MERGE's generation_config.json (which "
            'v3_served.require_generation_config has already proved is there and '
            'carries usable numbers), or from v3\'s deploy default for base '
            'weights - which is what a zero-shot voice renders at, whatever '
            'directory it names.')
    if 'seed' in sampling:
        raise ValueError(
            "Higgs SGLang sampling carries a seed. The seed is the request's own "
            '`seed` field and only that - pass it as build_request_body(..., '
            'seed=...). Two sources for one number is two renders nobody can '
            'reproduce.')
    stray = sorted(set(sampling) - set(SAMPLING_KEYS))
    if stray:
        raise ValueError(
            f'Higgs SGLang sampling carries key(s) CreateSpeechRequest does not '
            f"have: {', '.join(stray)}. Known: {', '.join(SAMPLING_KEYS)}. A "
            'field pydantic does not know is dropped without a word, and the '
            'render then runs at something nobody chose.')
    missing = [k for k in ('temperature', 'top_p', 'top_k')
               if sampling.get(k) is None]
    if missing:
        raise ValueError(
            f"Higgs SGLang sampling states no {', '.join(missing)}. "
            + _WHY_SAMPLING_IS_REQUIRED)
    body = {
        'model': model,
        'input': text,
        'response_format': 'wav',
        # THE FRAME CAP FIELD IS `max_new_tokens`. `max_tokens` is not a field of
        # CreateSpeechRequest at all, so sending it would be dropped in silence
        # and every chunk would run to the engine's own 7500-frame factory cap.
        'max_new_tokens': int(max_new_tokens),
    }
    if seed is not None:
        body['seed'] = int(seed)
    for key in SAMPLING_KEYS:
        if key in sampling:
            body[key] = sampling[key]
    if isinstance(voice, ClipsVoice):
        # The reference, base64 in the body. A DefaultVoice sends NO
        # `references` key at all - the absence is what "the model's own
        # speaker" / "the checkpoint's speaker" looks like on the wire.
        body['references'] = [reference_for(voice)]
    return body


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


def _guest_form(path: str) -> str:
    """A checkpoint path as the GUEST spells it, for comparing what the catalog
    says against what the running server's environ carries. On the Windows arm
    the catalog's `\\\\wsl$\\...` and `/home/...` forms both fold to the guest's
    own; elsewhere the path is already the guest's."""
    value = to_wsl(path or '') if sys.platform == 'win32' else (path or '')
    return value.rstrip('/') or value


#: Resolved once per process and KEPT, for `v3_served`'s reason stated one file
#: over: `importlib.resources.as_file` is a context manager because a package
#: may be a zip, in which case the path it yields is a temporary extraction that
#: disappears on exit - and this path is handed to `bash` minutes later, from
#: another method. So the ExitStack lives as long as the process does.
_packaged_launch_script = None
_packaged_launch_stack = None


def packaged_serve_script() -> str:
    """The absolute path of narrator's OWN `serve_higgs_sgl.sh`.

    THE SAME RESOLUTION AS `v3_served.packaged_serve_script`, on the DIRECTORY
    rather than on the file: this stack's script has no sibling to read, but the
    directory is what `importlib.resources` can make real in one step, and one
    mechanism for both launchers is one mechanism to get wrong.

    A package built without the package-data entry fails HERE, naming the entry,
    rather than two minutes into a launch as `bash: No such file`.
    """
    global _packaged_launch_script, _packaged_launch_stack
    if _packaged_launch_script is not None:
        return _packaged_launch_script
    resource = importlib.resources.files(
        v3_served.LAUNCH_PACKAGE).joinpath(v3_served.LAUNCH_DIR)
    stack = contextlib.ExitStack()
    try:
        directory = stack.enter_context(importlib.resources.as_file(resource))
    except Exception as exc:
        stack.close()
        raise ValueError(
            f'Higgs SGLang-Omni: narrator\'s packaged launcher '
            f'({v3_served.LAUNCH_PACKAGE}.{v3_served.LAUNCH_DIR}) could not be '
            f'made into a real directory on this filesystem '
            f'({type(exc).__name__}: {exc}). It is handed to bash, so it has to '
            'be a path. Install Crucible from a directory rather than a zip, or '
            f'name your own script in {SERVE_SCRIPT_ENV}.') from exc
    script = os.path.join(str(directory), PACKAGED_SERVE_SCRIPT)
    if not os.path.isfile(script):
        stack.close()
        raise ValueError(
            f'Higgs SGLang-Omni: narrator\'s packaged launcher directory '
            f'{directory} is missing {PACKAGED_SERVE_SCRIPT}. It ships as '
            'package data - [tool.setuptools.package-data] crucible = '
            '["jobs/tts/higgs/crucible_higgs/engine/higgs/launch/*"] in '
            "Crucible's pyproject.toml - and without it this stack cannot start "
            'at all.')
    _packaged_launch_stack = stack
    _packaged_launch_script = script
    return script


class HiggsSglServedBackend(GuestOwnedServer):
    """`narrator.engine.protocol.ServedBackend` for Higgs v3 under SGLang-Omni.

    The lifecycle - ownership marker, /proc listener scan, watchdog, adoption,
    TERM-only teardown - is `served_common.GuestOwnedServer`'s and is byte for
    byte the vllm-omni arm's. What is here is this stack's own: the launch line,
    the request, and an identity that has to be read out of the server's
    environment because `/v1/models` will not say it.
    """

    HEALTH_PATH = HEALTH_PATH
    LOG_TAG = '[HIGGSSGL]'
    SERVER_LOG_ENV = SERVER_LOG_ENV
    READY_FAILURE_HINT = (
        'The usual causes are flashinfer\'s JIT (the env needs CUDA_HOME pointed '
        'at nvidia/cu13 with lib64 -> lib and libcudart.so -> libcudart.so.13, '
        'plus the flashinfer-jit-cache cu130 wheel) and a mem-fraction-static '
        'the card cannot honour. Healthy takes ~110 s on example-pc.')

    def __init__(self, base_url: str = None, serve_script: str = None,
                 wsl_distro: str = None, checkpoint_dir: str = None,
                 server_log: str = None, concurrency: int = None):
        # THE LAUNCHER, IN THREE MODES, decided here and nowhere else - the
        # vllm-omni arm's comment, and now its behaviour:
        #
        #   attach    a base_url (argument or BASE_URL_ENV) names a server
        #             somebody else started. No launcher at all.
        #   operator  a serve_script (argument or SERVE_SCRIPT_ENV) overrides
        #             narrator's own. Refused when the path is not there.
        #   packaged  neither: narrator runs ITS OWN script, the one that ships
        #             in the package. See PACKAGED_SERVE_SCRIPT for why this is
        #             a definition and not a fallback.
        #
        # THE THIRD MODE IS NEW ON 2026-09-15 and it is what this whole file
        # lacked. Until then "neither is set" was a REFUSAL, so a client with no
        # BookForge checkout could not start the stack Owen actually renders on.
        base_url = (base_url or os.environ.get(BASE_URL_ENV) or '').strip()
        serve_script = (serve_script
                        or os.environ.get(SERVE_SCRIPT_ENV) or '').strip()
        if base_url:
            self.launcher_source = LAUNCHER_ATTACH
        elif serve_script:
            self.launcher_source = LAUNCHER_OPERATOR
            v3_served._check_override_script(serve_script, SERVE_SCRIPT_ENV)
        else:
            self.launcher_source = LAUNCHER_PACKAGED
            serve_script = packaged_serve_script()
        self.base_url = base_url or launch_base_url()
        self.serve_script = serve_script
        if concurrency is not None and int(concurrency) < 1:
            raise ValueError(
                f'Higgs SGLang: concurrency {concurrency} must be >= 1.')
        if concurrency is None and serve_script:
            concurrency = v3_served.serve_concurrency()
        self.concurrency = int(concurrency) if concurrency else None
        self.wsl_distro = (wsl_distro or os.environ.get(WSL_DISTRO_ENV)
                           or 'Ubuntu')
        self._named_log = (os.environ.get(SERVER_LOG_ENV) or '').strip() or None
        self.launch_log = ((server_log or '').strip()
                           or self.default_launch_log('narrator-higgs-sgl'))
        self._log_is_ours = bool(serve_script)
        self.server_log = self.launch_log if serve_script else self._named_log
        self._log_handle = None
        self.spec = BackendSpec(
            kind='served', name='sglang-omni', version='0.1.4',
            base_url=self.base_url,
            server_log=self.server_log,
            notes=('higgs-audio-v3-tts-4b; no site-packages patches (the '
                   'sentinel filter is vllm-omni\'s); 4096-token context'))
        self.checkpoint_dir = (checkpoint_dir or '').strip() or None
        self._proc = None
        self._guest_pid = None

    # -- launch --------------------------------------------------------------

    @staticmethod
    def _mem_fraction():
        """SGLang's `mem_fraction_static`, as THIS process was told it, or None.

        `HIGGS_SGL_MEM_FRACTION` is the launch script's own knob
        (`serve_higgs_sgl.sh`: `--mem-fraction-static`, default 0.60). It used
        to reach the script only by being inherited through the `bash` exec -
        which works, and which nothing states, so a launch could not say what
        fraction it started the server at. narrator now EXPORTS the value it
        was given, like every other knob it has an opinion about, and refuses a
        nonsense one here rather than letting the script discover it: the
        script's own check aborts a launch narrator has already reported as
        started.

        MEASURED, 2026-09-19: 16 concurrent requests at 0.60 summed to 24.2 GB
        on a 24 GB card and WDDM paged the excess to host RAM - 4 to 10 times
        slower, with no error anywhere. The fraction and the width are one
        budget, which is why both are stated at launch.
        """
        raw = (os.environ.get(SERVE_MEM_FRACTION_ENV) or '').strip()
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(
                f'{SERVE_MEM_FRACTION_ENV}={raw!r} is not a number. It is '
                "SGLang's mem_fraction_static and must be between 0 and 1 "
                '(exclusive).') from None
        if not (0.0 < value < 1.0):
            raise ValueError(
                f'{SERVE_MEM_FRACTION_ENV}={raw!r} is out of range: '
                'mem_fraction_static is a FRACTION of the card, strictly '
                'between 0 and 1.')
        return value

    def _launch_exports(self) -> str:
        """The `export ...` prefix of the wrapper: every launch-script knob
        narrator has an opinion about, stated explicitly.

        `HIGGS_MODEL_DIR` is exported for a checkpoint voice and explicitly
        UNSET for the base weights - unset, not left alone, because the worker's
        own environment may carry one from a caller and inheriting it would start
        a fine-tune under a request for the base speaker. It is also THE IDENTITY
        SOURCE on this stack (see `running_checkpoint`), so it is exported for
        that reason twice over.
        """
        host_port = self.base_url.split('://', 1)[-1].rstrip('/')
        host, _, port = host_port.rpartition(':')
        exports = [
            f'{SERVE_HOST_ENV}={shlex.quote(host)}',
            f'{SERVE_PORT_ENV}={shlex.quote(port)}',
            f'{SERVE_MAX_NUM_SEQS_ENV}={self.concurrency}',
            f'{served_common.OWNER_ENV}={shlex.quote(self.owner_id())}',
        ]
        # THE CONTEXT IS STATED TOO, and it is the one knob on this stack the
        # launcher cannot pass as a flag: `sgl_omni_entry.py` reads this
        # variable inside the server process and assigns the builder's class
        # attribute from it. Exported unconditionally - `context_tokens()` is
        # either the launch's own value or the builder's 4096, and writing the
        # number down is what makes the server log say which window is up.
        exports.append(f'{served_common.CONTEXT_LENGTH_ENV}={context_tokens()}')
        fraction = self._mem_fraction()
        if fraction is not None:
            # STATED, not inherited - see `_mem_fraction`. Absent means the
            # script's own documented default (0.60), which is the script's
            # number to own and not one narrator invents a second copy of.
            exports.append(f'{SERVE_MEM_FRACTION_ENV}={fraction}')
        if self.checkpoint_dir:
            exports.append(
                f'{SERVE_MODEL_DIR_ENV}={shlex.quote(_guest_form(self.checkpoint_dir))}')
            model = ''
        else:
            model = f'unset {SERVE_MODEL_DIR_ENV}; '
        return f'{model}export {" ".join(exports)}; '

    def _wrapper(self) -> str:
        """The shell the launcher actually runs: the exports, the script detached
        with `setsid` so a signal to the wrapper never reaches the server by
        accident, the guest-side watchdog (POSIX only - see `_watchdog_clause`),
        and `wait`. NO PID IS RECORDED: the listener on our port, found by its
        marker after health answers, is the only pid worth knowing."""
        script = (to_wsl(self.serve_script) if sys.platform == 'win32'
                  else self.serve_script)
        return ''.join([self._launch_exports(),
                        f'setsid bash {shlex.quote(script)} & ',
                        self._watchdog_clause(),
                        'wait'])

    def launch_command(self) -> list:
        """The command `start()` runs. Public so a test and a log line can see it
        without a GPU.

        `--exec` ON THE WINDOWS ARM IS LOAD-BEARING: without it `wsl.exe` hands
        the line to the distro's DEFAULT SHELL, which expands every `$` before
        `bash -c` sees the script. `--` is not a substitute (memory:
        wsl-exe-implicit-shell-trap).
        """
        if not self.serve_script:
            raise ValueError(
                f'This backend is in ATTACH mode ({BASE_URL_ENV}={self.base_url}); '
                'it has no launch command.')
        if sys.platform == 'win32':
            wsl = shutil.which('wsl.exe') or 'wsl.exe'
            return [wsl, '-d', self.wsl_distro, '--exec', 'bash', '-c',
                    self._wrapper()]
        return ['bash', '-c', self._wrapper()]

    # -- identity ------------------------------------------------------------

    def _models_payload(self) -> list:
        """The rows of `/v1/models`. Raises if it cannot be read."""
        try:
            with urllib.request.urlopen(self.base_url + MODELS_PATH,
                                        timeout=10) as response:
                payload = json.loads(response.read().decode('utf-8'))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise HiggsServerError(
                f'Higgs SGLang: {self.base_url}{MODELS_PATH} could not be read '
                f'({exc}). Something is listening on that port but it does not '
                'answer the OpenAI model list, so it cannot be identified - '
                'refusing to send a render to it.') from exc
        rows = payload.get('data') if isinstance(payload, dict) else None
        return [row for row in (rows or []) if isinstance(row, dict)]

    def served_models(self) -> list:
        """The model ids `/v1/models` reports."""
        return [row.get('id') for row in self._models_payload()]

    def running_checkpoint(self):
        """`(source, model_dir)` for the server on this port.

        NOT FROM `/v1/models`. `sglang_omni/serve/openai_api.py`'s
        `_register_models` answers `ModelCard(id=model_name, root=model_name)` -
        the SERVED NAME in both fields - so unlike vllm-omni 0.28 this stack
        never reports the model PATH there, and reading `root` would compare a
        checkpoint directory against the string "higgs-v3-ds" forever.

        TWO SOURCES, IN ORDER, AND THEY ARE DIFFERENT KINDS OF THING:

          'environ'   `HIGGS_MODEL_DIR` out of `/proc/<listener pid>/environ` -
                      the variable OUR launcher exported into that process. A
                      FACT about the running server, inherited by every fork and
                      rewritten by nothing. `None` beside this source means the
                      launcher exported none, which is the base weights and is an
                      answer rather than an absence.
          'asserted'  `NARRATOR_HIGGS3_CHECKPOINT`, for an ATTACHED server nobody
                      here launched, which therefore carries no marker to scan.
                      A CLAIM, and only consulted when there is no fact.

        `(None, None)` means neither - "unknown", which every caller treats as a
        refusal and never as "fine".
        """
        rows = self._own_servers_on_port()
        if rows:
            value = (rows[0].get('modelDir') or '').strip()
            return ('environ', value or None)
        asserted = (os.environ.get(CHECKPOINT_ENV) or '').strip()
        if asserted:
            # THE BASE SENTINEL. Until 2026-09-13 a checkpoint could only ever be
            # asserted as a PATH, so there was no way to say "this server runs the
            # base weights" — naming a path claimed a fine-tune and naming nothing
            # was refused outright. A server on the base was therefore
            # unattachable, which is not a corner: a ZERO-SHOT voice is base
            # weights conditioned by reference clips in the request, so the whole
            # zero-shot and streaming path runs against exactly such a server.
            #
            # `base` is deliberately a word a path cannot be — every real value
            # here is absolute — and it is still an ASSERTION an operator makes,
            # not a default. The refusal it lifts is "nothing can say"; the ones
            # that check what it says are untouched below.
            #
            # AND THIS IS THE LOCAL ANSWER ONLY. Owen ruled on 2026-09-13 that
            # the remote one is Crucible's: `GET /v1/activity` reports what is
            # resident, from `Residency`, which knows because it did the loading.
            #
            # The reason that ruling was needed is worth keeping next to this
            # code. A Higgs server CANNOT SAY WHAT WEIGHTS IT IS RUNNING —
            # sgl-omni's `/v1/models` answers the served NAME chosen at launch
            # (`higgs-v3-ds`), never the checkpoint path. So the only way to know
            # is `_own_servers_on_port()` reading `/proc/<pid>/environ`, which
            # needs the same machine AND narrator to have launched the process.
            # The moment the server is on the Mac or a droplet there is no
            # `/proc` to read and nothing to ask, so this whole identity check —
            # the one that stops a book rendering in the wrong narrator — cannot
            # work across a network at all.
            #
            # So do not grow this mechanism for the remote case. It is local by
            # construction; phases 6 and 7 move the question to the layer that
            # can answer it.
            if asserted.lower() == BASE_ASSERTION:
                return ('asserted-base', None)
            return ('asserted', asserted)
        return (None, None)

    def check_serves_expected_model(self, checkpoint_dir: str = None) -> None:
        """Prove the server on this port is OURS before anything is sent to it.

        Health answering 200 says only that SOMETHING is listening. This checks
        `/v1/models` carries `higgs-v3-ds`, and that the model the server was
        started ON is the one this voice needs. Every Higgs voice is a whole
        merged checkpoint and one server serves exactly one of them, so a
        leftover server from ANOTHER voice answers health and the model list
        identically and would render a whole book in the wrong narrator while
        every log line here named the right one - in either direction.
        """
        checkpoint_dir = checkpoint_dir or self.checkpoint_dir
        models = self.served_models()
        if SERVED_MODEL_NAME not in models:
            raise HiggsServerError(
                f'Higgs SGLang: the server at {self.base_url} serves '
                f"{models or '(none)'}, not '{SERVED_MODEL_NAME}'. That is "
                "somebody else's server, or the vllm-omni stack (which serves "
                "'higgs-v3'), on this port - refusing to render against it.")
        source, running = self.running_checkpoint()
        if source is None:
            raise HiggsServerError(
                f'Higgs SGLang: nothing can say which model the server at '
                f'{self.base_url} was started on. It carries no '
                f'{served_common.OWNER_ENV} (so narrator did not launch it and '
                f'its environ is not ours to read) and {CHECKPOINT_ENV} is not '
                'set. This voice is '
                + (f'the merged checkpoint {checkpoint_dir}' if checkpoint_dir
                   else 'the base weights')
                + ', and an unidentified server would render the whole book in '
                'whatever narrator it happens to hold. If this is your own '
                f'server, state its model directory in {CHECKPOINT_ENV} - or '
                f'{CHECKPOINT_ENV}={BASE_ASSERTION} if it is serving the base '
                'weights, which is what a zero-shot voice needs.')
        if checkpoint_dir:
            want = _guest_form(checkpoint_dir)
            if running is None:
                raise HiggsServerError(
                    f'Higgs SGLang: this voice is the merged checkpoint {want}, '
                    f'but the server at {self.base_url} was started with no '
                    f'{SERVE_MODEL_DIR_ENV} at all ({source}) - it is serving the '
                    'BASE weights, a different speaker at 12 % of the '
                    "fine-tune's ECAPA ceiling. Serving this voice means "
                    'RESTARTING on its checkpoint.')
            if _guest_form(running) != want:
                raise HiggsServerError(
                    f'Higgs SGLang: the server at {self.base_url} is running on '
                    f'{running} ({source}), but this voice is {want}. There is no '
                    'runtime LoRA on this stack either, so serving another voice '
                    'means RESTARTING on that checkpoint.')
            log(f'{self.LOG_TAG} serving checkpoint {want} (from the server\'s own '
                f'{source})', flush=True)
            return
        if running is not None:
            raise HiggsServerError(
                f'Higgs SGLang: this voice is the BASE weights, but the server at '
                f'{self.base_url} was started on {running} ({source}), which is a '
                'merged fine-tune. Serving the base means RESTARTING on the base '
                'snapshot.')
        log(f'{self.LOG_TAG} serving the base weights '
            + (f'({CHECKPOINT_ENV}={BASE_ASSERTION}, asserted by the operator)'
               if source == 'asserted-base'
               else f'(the server carries no {SERVE_MODEL_DIR_ENV})'), flush=True)

    # -- use -----------------------------------------------------------------

    def speak(self, request):
        """One `SpeechRequest` -> (float32 mono, sample rate).

        `request.max_new_tokens` is the caller's frame cap and is checked against
        the 4,096-token context HERE as well as where it was computed, because a
        cap that breaches it is an HTTP 500 and this is the last place before the
        wire.
        """
        self._check_context(request.text, request.max_new_tokens, request.voice)
        body = build_request_body(request.text, request.voice,
                                  request.max_new_tokens, seed=request.seed,
                                  sampling=request.sampling)
        payload, content_type = self.post_speech(body, with_content_type=True)
        return v3_served.decode_response(payload, content_type)

    def _check_context(self, text: str, max_new_tokens: int, voice=None) -> None:
        """prompt + cap <= 4,095, refused by name before the wire. The voice's
        reference, when it has one, is part of the prompt."""
        bound = prompt_token_bound(text, voice)
        total = bound + int(max_new_tokens)
        if total > max_context_positions():
            raise ValueError(
                f'Higgs SGLang-Omni: this request would need at most {bound} '
                f'prompt tokens plus a {int(max_new_tokens)}-frame cap = {total} '
                f'positions, over the {max_context_positions()} SGLang-Omni\'s '
                f'Higgs builder allows (context_length {context_tokens()}, '
                'hard-coded in engine_builder.py). The server answers HTTP 500 '
                'for this, not a shorter render. Size the cap with '
                'sgl_served.frame_cap(), which is what the engine uses.')

    def post_speech(self, body: dict, timeout: float = 1800,
                    with_content_type: bool = False):
        """POST `body` to /v1/audio/speech and return the raw response bytes (or
        `(bytes, content_type)` when asked)."""
        payload = json.dumps(body).encode('utf-8')
        req = urllib.request.Request(
            self.base_url + SPEECH_PATH, data=payload,
            headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = response.read()
                ctype = response.headers.get('Content-Type')
                return (data, ctype) if with_content_type else data
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode('utf-8', 'replace')[:800]
            if exc.code == 500:
                # The one 5xx whose text does not say what to do about it, and
                # the one this stack actually produces in normal use.
                raise HiggsServerError(
                    f'Higgs SGLang HTTP 500: {detail}\n'
                    'THE USUAL CAUSE IS THE CONTEXT: SGLang-Omni\'s Higgs builder '
                    f'hard-codes context_length {context_tokens()}, and prompt '
                    f'tokens + max_new_tokens over {max_context_positions()} fail '
                    'inside the scheduler rather than rendering short. '
                    'sgl_served.frame_cap() sizes a request to fit and refuses by '
                    'name when a chunk cannot; a 500 reaching here means the cap '
                    'came from somewhere else, or the prompt tokenized denser '
                    f'than the measured {CHARS_PER_TOKEN_FLOOR} characters per '
                    'token this bound assumes.') from exc
            raise HiggsServerError(
                f'Higgs SGLang HTTP {exc.code}: {detail}') from exc
        except urllib.error.URLError as exc:
            raise HiggsServerDown(
                f'Higgs SGLang-Omni server at {self.base_url} is unreachable: '
                f'{exc.reason}. Is it started (serve_higgs_sgl.sh; ~110 s to '
                'health) and is the sglomni env intact?') from exc


def launch_base_url() -> str:
    """Where a server narrator LAUNCHES will answer: the `HIGGS_SGL_HOST` /
    `HIGGS_SGL_PORT` pair the launch script binds, from the environment when
    BookForge stated them and the script's own literals otherwise. The pair is
    exported into the wrapper too, so the script binds exactly what this
    returns."""
    host = (os.environ.get(SERVE_HOST_ENV) or '').strip() or SERVE_DEFAULT_HOST
    raw_port = (os.environ.get(SERVE_PORT_ENV) or '').strip()
    if raw_port:
        try:
            port = int(raw_port)
        except ValueError:
            raise ValueError(
                f'Higgs SGLang: {SERVE_PORT_ENV}={raw_port!r} is not a port '
                'number.') from None
    else:
        port = SERVE_DEFAULT_PORT
    return f'http://{host}:{port}'
