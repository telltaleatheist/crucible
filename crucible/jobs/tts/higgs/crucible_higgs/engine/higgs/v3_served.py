"""Higgs TTS 3 (`higgs-audio-v3-tts-4b`) as a SERVED backend.

Higgs v3 is narrator's second engine (Owen, 2026-09-04 evening). It does NOT run
in this process: `model_type: higgs_multimodal_qwen3` has no HF modeling class -
it is implemented inside vllm-omni - and its dependency set (torch 2.13.0+cu130,
vllm 0.28.0) cannot share an environment with Orpheus's vLLM 0.7.3. So something
must LAUNCH a server, wait for its health endpoint, use it over HTTP, and kill
it. That lifecycle is `BackendSpec.kind == 'served'` and it is what this module
implements.

EVERY FACT BELOW IS MEASURED, from the campaign
`E:\\training\\_campaigns\\2026-09-01-cod-full-rebuild\\higgs\\`: `serve_v3.sh`
(the launch), `work/render_final.py` + `work/confirm.py` (the request),
`work/patch_vllm.py` + `work/patch_sentinel_filter.py` (the two required
patches),
`HIGGS_V3_LEVERS.md` (the lever sweep, the delivered render, the control-token
vocabulary), `work/refs/manifest.json` (the reference clips) and
`work/serve_v3c.log` (the route table).

THE ENVIRONMENT (example-pc, WSL Ubuntu, RTX 3090 Ti)

    env      /home/<user>/anaconda3/envs/higgs3 (python 3.11,
             torch 2.13.0+cu130, vllm 0.28.0, vllm-omni 0.28.0)
    launch   BookForge's serve_higgs_v3.sh (transcribed from the campaign's
             serve_v3.sh) - vllm-omni serve <model dir> --served-model-name
             higgs-v3 --host/--port --trust-remote-code --stage-overrides
             <per-stage memory / max_num_seqs / max_model_len>
             --attention-backend FLASH_ATTN --omni, with CUDA_HOME pointed at
             the pip CUDA 13 wheel, VLLM_USE_FLASHINFER_SAMPLER=0,
             VLLM_DISABLE_FLASHINFER_PREFILL=1, TORCH_CUDA_ARCH_LIST=8.6.
             The model dir, the bind address and the concurrency come from
             the HIGGS_* variables narrator exports (`_launch_exports`).
             PER-STAGE, not the global flags the campaign used: vllm-omni
             applies a global --gpu-memory-utilization to EVERY stage, and
             this server is two of them (talker + codec), so 0.60 reserved
             1.2 cards - measured 24.2 of 24.5 GB on 2026-09-05.
    cost     COLD START 55-297 s to health=200 (measured on this box, same
             script, same env: 55 s warm page cache, 146 s, and 297 s cold, in
             narrator's own smoke). The spread is disk and first-run
             compilation. The server preallocates to the utilization target
             (0.60 of 24 GB), so it OWNS the GPU while up.
    stop     SIGTERM the `vllm-omni serve` process (their work/release_gpu.sh
             does exactly this and then drops the shared lock)

narrator INVOKES THEIR SCRIPT. It does not write a launcher of its own: the
script is where the CUDA_HOME/flashinfer workarounds live, and a second copy of
them would drift.

TWO SITE-PACKAGES PATCHES ARE REQUIRED, and must be RE-APPLIED AFTER ANY PIP
UPGRADE in that env:

  work/patch_vllm.py       vLLM 0.28 rejects any prompt id < 0; vllm-omni's
                           clone path deliberately emits AUDIO_PLACEHOLDER_ID
                           == -100 for the talker to substitute reference
                           embeddings at prefill. Without the patch EVERY clone
                           request is HTTP 400 "Token id -100 is out of
                           vocabulary" - which is what made cloning look broken
                           in the first audition.
  work/patch_sentinel_filter.py
                           filters frames by TOKEN IDENTITY: a frame is kept iff
                           all 8 codebooks are in [0, 1023]. THIS IS
                           SERVER-SIDE. Upstream substituted every out-of-range
                           code with 0 - a VALID codec code that decodes to real
                           sound, so the substitution CONVERTED the ramp-down
                           BOC/EOC sentinels into audio - and then trimmed
                           exactly one of the seven frames they smear across,
                           leaving ~240 ms of garbage on every chunk. The patch
                           removes the substitution and the positional trim
                           entirely (`[:, :-1]` occurs twice in the pristine
                           file and zero times after it). The streaming path
                           gets the trailing run only, because Stage 1 trims
                           left_context/right_holdback BY FRAME COUNT and
                           dropping a leading or interior frame would desync
                           those trims and cut real speech. So a patched server
                           returns audio whose tail is already clean and THE
                           CLIENT MUST NOT TRIM AGAIN. What remains is a hard
                           sample boundary, which is a click: `edge_fade` -
                           EdgeFade(10 in, 25 out) - is the assembler's job.
                           SUPERSEDES work/patch_tail_trim.py (retired
                           2026-09-05), which reasoned about WHERE sentinels
                           usually sit and kept the 0-substitution for every one
                           outside the trailing run.

Both belong in a managed-env recipe at cut-over; see ../PORT_NOTES.md 12.7.

WHAT THE REQUEST LOOKS LIKE, and the two ways it goes silently wrong:

  * SAMPLING MUST RIDE IN `extra_params`. `temperature` / `top_p` / `top_k` are
    not fields of `OpenAICreateSpeechRequest`; pydantic drops them without a
    word, and the whole first audition therefore ran at the server default
    while reporting 0.3. AND "the server default" IS THE MODEL DIRECTORY:
    sending nothing means whatever `<model dir>/generation_config.json` holds,
    because `--generation-config` defaults to `auto` and `serve_v3.sh` passes no
    override. A validated merged checkpoint holds temperature 1.0, top_p 0.95,
    top_k 50, repetition_penalty 1.0; a directory WITHOUT the file - which is
    every unmerged `bosonai/higgs-audio-v3-tts-4b` snapshot - gets a bare
    `SamplingParams()` instead: top_p 1.0 and top_k DISABLED, the untruncated
    1026-way codebook tail, which derails long chunks into babble. See
    `require_generation_config` and ../PORT_NOTES.md 12.8d. So base weights are
    sent `SERVER_DEFAULT_SAMPLING` EXPLICITLY (HiggsV3Config.served_sampling)
    and only a checkpoint voice sends nothing.
  * A CONTROL TOKEN THAT IS NOT IN THE VOCABULARY IS READ ALOUD AS WORDS and
    derails generation into a degenerate loop - ASR coverage 0.000, pitch std
    0.28 st, speaker cosine 0.05. `<|emotion:calm|>`, `<|emotion:neutral|>`,
    `<|prosody:pause_long|>` and the v2-only `<|scene_desc_start|>` are all
    traps. Hence `validate_control_tokens`, run on every request.

EXACTLY ONE REFERENCE. vllm-omni refuses multi-shot cloning, so "two clips"
means ONE concatenated wav (clips joined by 0.35 s of silence) with the
transcripts joined in the same order - and the total is capped at 30 s (42 s is
HTTP 400 "Reference audio too long"). Measured: no reference 0.093 speaker
cosine, one clip 0.680, two clips 0.692, against a 0.766 narrator self-ceiling;
same-book vs cross-book is worth +0.076, the second clip only +0.012.

LICENCE: Boson Higgs TTS 3 Research and Non-Commercial. Fine for personal use
and, under the Creator Use Grant, for credited creator content; production
deployment or embedding in a product needs separate licensing.
"""
import contextlib
import importlib.resources
import io
import json
import os
import re
import shlex
import shutil
import subprocess  # noqa: F401  (tests patch v3_served.subprocess.run)
import sys
import urllib.error
import urllib.request

import numpy as np

from ..log import log
from ..protocol import BackendSpec, ClipsVoice, DefaultVoice
from . import served_common
from .served_common import GuestOwnedServer

MODEL_ID = 'bosonai/higgs-audio-v3-tts-4b'
SERVED_MODEL_NAME = 'higgs-v3'
DEFAULT_BASE_URL = 'http://127.0.0.1:8095'
SPEECH_PATH = '/v1/audio/speech'
HEALTH_PATH = '/health'
MODELS_PATH = '/v1/models'

#: AN OPERATOR'S OVERRIDE of the launcher below. Set it to run a script that is
#: not narrator's own - a campaign's `serve_v3.sh`, an env's installed copy, a
#: one-off with different flags. A path that does not exist is REFUSED by name;
#: it is never quietly replaced by the packaged one.
SERVE_SCRIPT_ENV = 'NARRATOR_HIGGS3_SERVE_SCRIPT'
#: NARRATOR'S OWN LAUNCHER, shipped inside the package as data
#: (`crucible_higgs/engine/higgs/launch/`, declared in Crucible's pyproject.toml's
#: [tool.setuptools.package-data]) and resolved through `importlib.resources`.
#:
#: THIS IS NOT A FALLBACK. How narrator launches its Higgs v3 server is
#: narrator's own definition - the CUDA_HOME and FlashInfer workarounds, the
#: per-stage memory fractions and the certified frames-7500 deploy profile are
#: all in that file, and a client that has never heard of BookForge cannot be
#: expected to supply them. Until 2026-09-13 the only copy lived in BookForge's
#: `electron/scripts/higgs/`, so narrator refused to start at all for any other
#: caller: Crucible's first real `tts` render died with
#: `NARRATOR_HIGGS3_SERVE_SCRIPT ... Neither is set`.
#: THIS PACKAGE, by its own name: the worker runs as `crucible_higgs` out of
#: Crucible's install (`crucible/jobs/tts/higgs/`), and the launch directory is
#: package data of `crucible` there (pyproject.toml [tool.setuptools.package-data]).
LAUNCH_PACKAGE = __package__
LAUNCH_DIR = 'launch'
PACKAGED_SERVE_SCRIPT = 'serve_higgs_v3.sh'
#: The certified deploy profile that ships BESIDE the script; the script reads
#: it as `$(dirname "$0")/<this>`, so the two must land in one directory. Named
#: here so the resolution below can prove it is there rather than letting
#: vllm-omni silently auto-discover its own 2048-frame profile.
PACKAGED_DEPLOY_CONFIG = 'higgs_default_frames7500.yaml'
#: Which of the three launcher modes a backend is in, reported on its own log
#: line at construction (see `HiggsV3ServedBackend.__init__`).
LAUNCHER_ATTACH = 'attach'
LAUNCHER_OPERATOR = 'operator'
LAUNCHER_PACKAGED = 'packaged'

#: Attach to an ALREADY-RUNNING server instead of launching one.
BASE_URL_ENV = 'NARRATOR_HIGGS3_URL'
#: The WSL distro to run the launch script in, on Windows.
WSL_DISTRO_ENV = 'NARRATOR_HIGGS3_WSL_DISTRO'
#: WHICH MERGED CHECKPOINT an ATTACHED server is running, when the server
#: cannot say so itself.
#:
#: Normally it CAN: vllm-omni 0.28's `/v1/models` carries `"root": <the model
#: path the server was started on>` beside the served name (measured on
#: example-pc 2026-09-05: root = the base snapshot path when the launcher was
#: started without HIGGS_MODEL_DIR), and `running_checkpoint()` reads it. This
#: variable is the operator's assertion for a server build whose model list
#: does NOT carry a root; it is consulted only then, and never overrides a root
#: the server reports - a reported path is a fact and an env var is a claim.
CHECKPOINT_ENV = served_common.CHECKPOINT_ENV
#: THE LAUNCH SCRIPT'S OWN KNOBS (`serve_higgs_v3.sh`), exported into the
#: wrapper's environment by narrator at launch. narrator states every one of
#: them rather than inheriting whatever the worker's environment happens to
#: hold, because each is a place the two sides can silently disagree:
#:   HIGGS_MODEL_DIR    the merged checkpoint the server comes up ON. Left
#:                      unset, the script serves the base snapshot - a
#:                      different speaker - and `check_serves_expected_model`
#:                      then (correctly) refuses after a 55-297 s launch.
#:                      That was Owen's first in-app Higgs render (2026-09-05).
#:   HIGGS_HOST/PORT    where the script binds, which must be where narrator
#:                      polls `/health` and posts renders.
#:   HIGGS_MAX_NUM_SEQS how many sequences stage 0 admits at once - and
#:                      therefore how wide narrator's own batch is
#:                      (`serve_concurrency`). One number, stated once.
SERVE_MODEL_DIR_ENV = served_common.SERVE_MODEL_DIR_ENV
SERVE_HOST_ENV = 'HIGGS_HOST'
SERVE_PORT_ENV = 'HIGGS_PORT'
SERVE_MAX_NUM_SEQS_ENV = 'HIGGS_MAX_NUM_SEQS'
#: The launch script's own bind defaults, mirrored so that a launch with
#: neither variable set polls the port the script binds. Not a fallback that
#: hides a bug: both sides read the same two literals, and narrator EXPORTS
#: the pair it chose into the wrapper, so they cannot drift apart.
SERVE_DEFAULT_HOST = '127.0.0.1'
SERVE_DEFAULT_PORT = 8095
#: How the base snapshot is recognised in a reported model root: the HF cache
#: directory name of `bosonai/higgs-audio-v3-tts-4b`. A server whose root
#: carries this is serving the BASE weights; any other root is a fine-tune.
BASE_SNAPSHOT_MARKER = 'models--bosonai--higgs-audio-v3-tts-4b'
#: THE OWNERSHIP MARKER: exported into the launch wrapper's environment, and
#: therefore into `/proc/<pid>/environ` of the server and every process it
#: forks. Its value is the pid of the narrator process that launched it.
#:
#: This replaced a pid FILE holding the wrapper's `$!`. MEASURED 2026-09-05 on
#: Owen's first Windows render: `setsid bash serve_higgs_v3.sh &` recorded
#: 84072, but vllm-omni puts itself into a session of its own (84096 -> 84098,
#: sid 84098), so the recorded pid was dead within a second, `kill -TERM
#: -- -84072` was "No such process", and the server outlived the job holding
#: 24 GB. A pid remembered at launch cannot follow a process that re-parents
#: itself; a marker in its environment can, because environ is inherited by
#: every fork and exec and nothing in vllm-omni rewrites it. So a server is
#: OURS iff the process listening on our port carries this variable, and its
#: process GROUP (read off /proc/<pid>/stat, not remembered) is what a stop
#: signals - measured: the two `VLLM::StageEngineCoreProc` children share the
#: listener's pgid.
#: How long one guest-side command (a /proc scan, a group signal) may take
#: before it is reported as unanswered. Owen, 2026-09-05: "timeouts are intended
#: to kill something if its waiting for an obscenely long time ... it should be
#: like 10 minutes." These are wedge detectors, not budgets: a WSL VM that takes
#: a minute to answer under load is slow, not gone, and a 30 s ceiling turned
#: slow into "could not scan" on a healthy machine.
GUEST_COMMAND_TIMEOUT_SECONDS = served_common.GUEST_COMMAND_TIMEOUT_SECONDS

OWNER_ENV = served_common.OWNER_ENV
#: How long the guest-side watchdog sleeps between looks at the owner.
WATCHDOG_INTERVAL_SECONDS = served_common.WATCHDOG_INTERVAL_SECONDS

#: WHERE AN ATTACHED SERVER'S LOG IS, named by the operator. NO DEFAULT.
#:
#: In LAUNCH mode narrator owns the server's output and writes it itself (see
#: `server_log`), so this is not needed. In ATTACH mode the server belongs to
#: somebody else and its output went wherever they sent it; narrator cannot
#: discover that and will not guess, because the wrong file is worse than none -
#: a stale log from an earlier run would let the sentinel proof pass on evidence
#: from a server that is no longer up.
#:
#: The training side's own launcher tees to
#: `E:\\training\\_campaigns\\2026-09-01-cod-full-rebuild\\higgs\\v3_ft\\logs\\serve_current.log`,
#: OVERWRITTEN PER START, and that is the path an operator points this at when
#: attaching to their server. Give it in the form the READING process sees: a
#: narrator worker runs inside WSL, so that file is `/mnt/e/training/...` there.
SERVER_LOG_ENV = 'NARRATOR_HIGGS3_SERVER_LOG'

#: `vllm_omni/deploy/higgs_multimodal_qwen3.yaml` stage 0 - THE DEPLOY DEFAULT
#: FOR THE BASE WEIGHTS, and nothing more than that.
#:
#: It is NOT what a request gets by sending no `extra_params`. `vllm-omni serve`
#: on the CLI never reads that YAML; it resolves sampling from the MODEL
#: DIRECTORY (`--generation-config` defaults to `auto`), so an empty request
#: gets `<model dir>/generation_config.json` - and, when the directory has no
#: such file, a bare `SamplingParams()`: top_p 1.0, top_k DISABLED. The
#: `bosonai/higgs-audio-v3-tts-4b` snapshot ships no such file, which is why a
#: merged checkpoint must carry one (`require_generation_config`) and why base
#: weights are sent these values EXPLICITLY rather than assumed
#: (`HiggsV3Config.served_sampling`). See ../PORT_NOTES.md 12.8d.
#:
#: `seed` rides at the request's TOP LEVEL and never inside `extra_params`
#: (`build_request_body` refuses the duplicate), so it is excluded wherever this
#: mapping is used as a sampling payload.
SERVER_DEFAULT_SAMPLING = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 50,
                           'repetition_penalty': 1.0, 'seed': 42}

#: The log file narrator writes a server it LAUNCHED to, inside the session's
#: process dir when there is one. narrator has no other log FILE anywhere - its
#: own engine lines go to a STREAM the host chooses (`engine/log.py`) - so this
#: is a new artifact, and it lives beside `session-state.json` because that is
#: the directory that already holds a run's own per-run files (the Orpheus
#: guards file their rejects there too). One per SESSION, overwritten per start,
#: exactly like the training side's `serve_current.log`.
SERVER_LOG_NAME = 'higgs-v3-server.log'

#: WHERE THE SENTINEL FILTER WRITES ITS RECORDS, and the one variable that makes
#: proof (a) a real channel instead of a log scrape.
#:
#: `electron/scripts/higgs/patch_sentinel_filter.py` v3 reads this out of its own
#: process environment at import and appends one JSON record per filter
#: invocation - clean ones included. narrator exports it into the launch wrapper
#: beside HIGGS_HOST/HIGGS_PORT/HIGGS_MODEL_DIR, so in LAUNCH mode the path the
#: server writes and the path narrator reads are decided in one place and cannot
#: name different files. In ATTACH mode narrator reads the variable out of its
#: OWN environment: an operator who started the server itself states the same
#: file here that they gave it, exactly as they state its log in SERVER_LOG_ENV.
#: There is NO DEFAULT, for the same reason: a stale report from an earlier run
#: would let this proof pass on evidence from a server that is no longer up.
SENTINEL_REPORT_ENV = 'HIGGS_SENTINEL_REPORT'

#: The suffix narrator appends to `launch_log` to name the report it owns. It
#: rides beside the log deliberately: one per SESSION when the engine gave us a
#: process dir, one per INSTANCE otherwise, truncated at every launch - the same
#: three properties the log has, for the same reason (two workers must never
#: share one, and last run's evidence must never certify this run).
SENTINEL_REPORT_SUFFIX = '.sentinel.jsonl'

#: The record `v` this narrator reads. A record carrying any other version is
#: REFUSED rather than skipped: it means the env holds a patch generation this
#: code was not written against, and quietly ignoring the records would be the
#: "no evidence reads as no problem" failure this whole proof exists to prevent.
SENTINEL_REPORT_VERSION = 3

# ── THREE REGEXES OVER THE SERVER'S LOG FILE LIVED HERE UNTIL 2026-09-13 ─────
#
# `_ASYNC_SENTINEL_RE`, `_SYNC_INTERIOR_RE` and `_EMPTY_CHUNK_RE` matched the
# three `logger.warning` messages the sentinel filter can write, and
# `verify_sentinel_filter` opened the server's LOG FILE and ran them over every
# line to decide whether a 19 GB model could render a book. That is
# crucible/docs/ARCHITECTURE.md rule R4's headline example - a log line that
# became an API nothing versions and nothing tests. vLLM re-words a warning, or
# the log formatter changes, or the file rotates, and the proof either refuses a
# healthy server or, far worse, passes on nothing at all.
#
# It also carried a belief that was already FALSE. The deleted constant
# `EXPECTED_TRAILING_SENTINEL_FRAMES = 2` documented v1 of the patch, which
# counted out-of-range frames BEFORE it trimmed the trailing run and so counted
# the model's normal 2-frame EOC ramp. **v2 (2026-09-05) reordered exactly
# that**: it trims by identity first, so on a correct build the async warning
# does not fire on a clean final window at all. Against a v2 env the old proof
# therefore passed two different ways it should not have - vacuously, on a log
# with no matching lines, and on a log full of the very "2 frame(s)" lines v2
# exists to eliminate, because 2 was what it expected.
#
# The fix is at the SOURCE: patch v3 writes one structured record per filter
# invocation to `$HIGGS_SENTINEL_REPORT` (SENTINEL_REPORT_ENV), and
# `verify_sentinel_filter` reads records. The warnings are untouched and still
# go to the log for a human - R4's instruction is "promote the fact to an event",
# never "stop logging".

#: Total reference audio the server accepts. 42 s returns HTTP 400.
MAX_REFERENCE_SECONDS = 30.0
#: Silence between concatenated reference clips, per HIGGS_V3_LEVERS.md 5.
REFERENCE_JOIN_SECONDS = 0.35

#: Chunk size. <= 600 chars is the measured safe zone; 900 drops the tail
#: reproducibly (coverage 0.78-0.86, tail coverage 0.00-0.31) and a reference
#: clip does NOT fix it. The delivered render used 300. 600 is a PLACEHOLDER
#: until the catalog carries it per (engine, voice).
MAX_CHARS = 600
DELIVERED_MAX_CHARS = 300

#: The fused context window the vllm-omni launcher serves when nobody says
#: otherwise (`--max-model-len`, from `HIGGS_MAX_MODEL_LEN`).
DEFAULT_CONTEXT_TOKENS = 8192

#: THE LAUNCHER'S OWN NAME for the same number. It predates
#: `HIGGS_CONTEXT_LENGTH` and is what `serve_higgs_v3.sh:123` reads, so it stays
#: - renaming it would break every launch line and every campaign script that
#: sets it. narrator TRANSLATES instead: one external name across the three
#: stacks, each stack's own spelling underneath.
SERVE_MAX_MODEL_LEN_ENV = 'HIGGS_MAX_MODEL_LEN'


def context_tokens() -> int:
    """THE CONTEXT THIS STACK IS LAUNCHED WITH, from either name, and a
    REFUSAL when both are set and disagree.

    TWO NAMES FOR ONE NUMBER IS THE FAILURE, NOT THE FEATURE. `HIGGS_MAX_MODEL_LEN`
    is the launcher's, `HIGGS_CONTEXT_LENGTH` is the one Crucible sets per voice
    for every stack. Picking a winner silently would mean a server started at
    one window and a client sizing chunks against the other - which is invisible
    until a chunk is an HTTP 500 or, worse, quietly short. So a disagreement is
    named and the launch stops.
    """
    external = served_common.context_length(DEFAULT_CONTEXT_TOKENS)
    raw = (os.environ.get(SERVE_MAX_MODEL_LEN_ENV) or '').strip()
    if not raw:
        return external
    try:
        launcher = int(raw)
    except ValueError:
        raise ValueError(
            f'{SERVE_MAX_MODEL_LEN_ENV}={raw!r} is not a whole number of '
            'tokens.') from None
    if launcher < 1:
        raise ValueError(
            f'{SERVE_MAX_MODEL_LEN_ENV}={raw!r} must be at least 1 token.')
    stated = (os.environ.get(served_common.CONTEXT_LENGTH_ENV) or '').strip()
    if stated and launcher != external:
        raise ValueError(
            f'{served_common.CONTEXT_LENGTH_ENV}={external} and '
            f'{SERVE_MAX_MODEL_LEN_ENV}={launcher} are two names for ONE number '
            'and they disagree. They are the model context the server is started '
            'with AND the window every request is sized against; narrator will '
            'not pick one, because the loser is invisible - the server comes up '
            'clean either way and the fault shows as an HTTP 500 hundreds of '
            'chunks later. Set one, or set both to the same value.')
    return launcher
#: LM frames per second of audio, and the sample rate. Identical to v2.
FRAMES_PER_SECOND = 25.0
SAMPLE_RATE = 24000

#: Assembly fades. 10 ms in / 25 ms out, measured to take a chunk edge from
#: -30 dB to -45..-48 dB against a -70 dB floor.
EDGE_FADE_IN_MS = 10.0
EDGE_FADE_OUT_MS = 25.0

#: Every inline control token v3 actually has, read off
#: `tokenizer.get_added_vocab()` (work/added_vocab.json, 84 added tokens).
#: THIS IS AN ALLOWLIST, NOT DOCUMENTATION - see the module docstring.
ALLOWED_CONTROL_TOKENS = frozenset((
    '<|emotion:affection|>', '<|emotion:amusement|>', '<|emotion:anger|>',
    '<|emotion:arousal|>', '<|emotion:awe|>', '<|emotion:bitterness|>',
    '<|emotion:confusion|>', '<|emotion:contemplation|>',
    '<|emotion:contentment|>', '<|emotion:determination|>',
    '<|emotion:disgust|>', '<|emotion:elation|>', '<|emotion:enthusiasm|>',
    '<|emotion:fear|>', '<|emotion:helplessness|>', '<|emotion:longing|>',
    '<|emotion:pride|>', '<|emotion:relief|>', '<|emotion:sadness|>',
    '<|emotion:shame|>', '<|emotion:surprise|>',
    '<|env:music|>', '<|env:noise|>',
    '<|prosody:expressive_high|>', '<|prosody:expressive_low|>',
    '<|prosody:long_pause|>', '<|prosody:pause|>', '<|prosody:pitch_high|>',
    '<|prosody:pitch_low|>', '<|prosody:speed_fast|>',
    '<|prosody:speed_slow|>', '<|prosody:speed_very_fast|>',
    '<|prosody:speed_very_slow|>',
    '<|sfx:burping|>', '<|sfx:cough|>', '<|sfx:crying|>', '<|sfx:humming|>',
    '<|sfx:laughter|>', '<|sfx:screaming|>', '<|sfx:sigh|>', '<|sfx:sneeze|>',
    '<|sfx:sniff|>',
    '<|style:shouting|>', '<|style:singing|>', '<|style:whispering|>',
))

# Anything shaped like a control token. Deliberately loose: the point is to
# catch `<|emotion:calm|>` and `<|scene_desc_start|>` BEFORE the model reads
# them out loud, so the pattern must match the traps too.
_CONTROL_TOKEN_RE = re.compile(r'<\|[^|>]{1,64}\|>')


#: THE SERVER ERRORS, AND THEY BELONG TO BOTH STACKS.
#:
#: These are ALIASES of `served_common.HiggsServerError` / `HiggsServerDown`
#: rather than v3-only subclasses, and that is load-bearing rather than tidy.
#: `HiggsV3Engine` runs against EITHER stack and its per-row failure policy is
#: written as `except v3_served.HiggsV3ServerDown: raise` - the one exception
#: that must end a whole take instead of failing one chunk. A separate SGLang
#: class would slip past that clause and every remaining row of a book would be
#: marked failed one at a time against a dead port, which is exactly the
#: behaviour `convert_many` exists to refuse.
HiggsV3ServerError = served_common.HiggsServerError
HiggsV3ServerDown = served_common.HiggsServerDown


#: Resolved once per process and KEPT. `importlib.resources.as_file` is a
#: context manager because a package may be a zip, in which case the path it
#: yields is a temporary extraction that disappears on exit - and this path is
#: handed to `bash` minutes later, from another method. So the ExitStack lives
#: as long as the process does, exactly as the resources documentation's
#: "keep the file around" example does.
_packaged_launch_dir = None
_packaged_launch_stack = None


def packaged_serve_script() -> str:
    """The absolute path of narrator's OWN `serve_higgs_v3.sh`.

    Resolved through `importlib.resources` on the DIRECTORY rather than on the
    script, because the script reads its deploy profile as a sibling: extracting
    one file alone would give bash a script whose `$(dirname "$0")` holds
    nothing. Both are asserted present, so a package built without the
    package-data entry fails here - naming the entry - rather than three minutes
    into a launch.
    """
    global _packaged_launch_dir, _packaged_launch_stack
    if _packaged_launch_dir is not None:
        return _packaged_launch_dir
    resource = importlib.resources.files(LAUNCH_PACKAGE).joinpath(LAUNCH_DIR)
    stack = contextlib.ExitStack()
    try:
        directory = stack.enter_context(importlib.resources.as_file(resource))
    except Exception as exc:
        stack.close()
        raise ValueError(
            f'Higgs v3: narrator\'s packaged launcher ({LAUNCH_PACKAGE}.'
            f'{LAUNCH_DIR}) could not be made into a real directory on this '
            f'filesystem ({type(exc).__name__}: {exc}). It is handed to bash, so '
            'it has to be a path. Install Crucible from a directory rather than '
            f'a zip, or name your own script in {SERVE_SCRIPT_ENV}.') from exc
    script = os.path.join(str(directory), PACKAGED_SERVE_SCRIPT)
    profile = os.path.join(str(directory), PACKAGED_DEPLOY_CONFIG)
    missing = [os.path.basename(p) for p in (script, profile)
               if not os.path.isfile(p)]
    if missing:
        stack.close()
        raise ValueError(
            f'Higgs v3: narrator\'s packaged launcher directory {directory} is '
            f'missing {missing}. Both files ship as package data - '
            '[tool.setuptools.package-data] crucible = '
            '["jobs/tts/higgs/crucible_higgs/engine/higgs/launch/*"] in '
            "Crucible's pyproject.toml - and the "
            'script reads the profile as its own sibling, so one without the '
            'other is a server that truncates every long chunk at 81.92 s.')
    _packaged_launch_stack = stack
    _packaged_launch_dir = script
    return script


def _check_override_script(path: str, variable: str) -> None:
    r"""An operator's launcher that is not there is a REFUSAL, not a reason to
    run narrator's own.

    `variable` NAMES THE OVERRIDE THAT WAS SET, and it is a required argument
    rather than this module's own constant because BOTH stacks use this check
    (`sgl_served` since 2026-09-15) and each has its own variable. A refusal
    about NARRATOR_HIGGS_SGL_SERVE_SCRIPT that printed NARRATOR_HIGGS3_
    SERVE_SCRIPT would send a reader to unset a variable they never set.

    Somebody who named a script meant that script; substituting the packaged one
    would start a server with different flags, a different env prefix and
    possibly a different deploy profile, and report success.

    WHAT CANNOT BE CHECKED is a GUEST path on the Windows arm - exactly the
    three forms `served_common.to_wsl` rewrites into the distro:

      /home/<user>/.../serve_higgs_v3.sh    BookForge's own
                                              (higgs-spawn.ts
                                              `serveScriptGuestPath`)
      \\wsl$\Ubuntu\...                       the same file named as a share
      \\wsl.localhost\Ubuntu\...              ditto, the newer spelling

    The first is not on the Windows filesystem at all, so `os.path.isfile`
    would answer False about a file that exists. The two UNC forms are only
    reachable while the distro is RUNNING, so the same answer would depend on
    whether WSL happened to be up. Saying nothing about a path this process
    cannot see is not a fallback - bash inside the guest reports it by name two
    seconds later, which is the same refusal one layer down.
    """
    if sys.platform == 'win32':
        guest = path.replace('\\', '/').lower()
        if guest.startswith(('/', '//wsl$/', '//wsl.localhost/')):
            return
    if not os.path.isfile(path):
        raise ValueError(
            f'Higgs v3: {variable}={path!r} is not a file. That variable '
            "OVERRIDES narrator's own packaged launcher, so an unreadable path "
            'is refused rather than silently replaced by it - a server started '
            'from the wrong script is a render nobody can account for. Unset the '
            "variable to use narrator's own launcher.")


def serve_concurrency() -> int:
    """How many requests the server admits at once - `HIGGS_MAX_NUM_SEQS`.

    THE ONE NUMBER BEHIND BATCHING on the served arm. vllm-omni's continuous
    batcher schedules up to stage 0's `max_num_seqs` sequences together, and its
    own `/v1/audio/speech/batch` endpoint is nothing more than an
    `asyncio.gather` over the items (serving_speech.py:create_speech_batch,
    measured 2026-09-05) - so N concurrent POSTs to the plain endpoint ARE the
    batch, and there is nothing to be gained from the batch endpoint except a
    response that arrives when the slowest item does.

    Read from the environment because the same variable is what the launch
    script passes as `max_num_seqs`: BookForge sets it from the catalog's
    serving block, narrator exports it into the wrapper at launch and sizes
    `BATCH_SIZE` from it. Refused by name when absent - a guessed width is
    either a server idling at 1 or a queue the render never asked for.
    """
    raw = (os.environ.get(SERVE_MAX_NUM_SEQS_ENV) or '').strip()
    if not raw:
        raise ValueError(
            f'Higgs v3: {SERVE_MAX_NUM_SEQS_ENV} is not set. It is the number of '
            'sequences the server admits at once (the launch script passes it as '
            'stage 0 max_num_seqs) and the width of narrator\'s batch; BookForge '
            "sets it from the catalog's serving block. There is no default.")
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f'Higgs v3: {SERVE_MAX_NUM_SEQS_ENV}={raw!r} is not an integer.') from None
    if value < 1:
        raise ValueError(
            f'Higgs v3: {SERVE_MAX_NUM_SEQS_ENV}={value} must be at least 1.')
    return value


def launch_base_url() -> str:
    """Where a server narrator LAUNCHES will answer: the `HIGGS_HOST` /
    `HIGGS_PORT` pair the launch script binds, from the environment when
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
                f'Higgs v3: {SERVE_PORT_ENV}={raw_port!r} is not a port number.') from None
    else:
        port = SERVE_DEFAULT_PORT
    return f'http://{host}:{port}'


# ---------------------------------------------------------------------------
# The text: control tokens
# ---------------------------------------------------------------------------


def validate_control_tokens(text: str) -> None:
    """Refuse any `<|...|>` sequence v3 does not have in its vocabulary.

    Not a nicety. An out-of-vocabulary token is split into 7-10 ordinary text
    pieces and READ ALOUD, and the render collapses: measured ASR coverage
    0.000, pitch std 0.28 st (a near-constant tone), speaker cosine 0.05. The
    request still returns HTTP 200, so nothing downstream would notice.
    """
    unknown = sorted({t for t in _CONTROL_TOKEN_RE.findall(text or '')
                      if t not in ALLOWED_CONTROL_TOKENS})
    if unknown:
        raise ValueError(
            f'Higgs v3 text carries control token(s) it does not have: '
            f"{', '.join(unknown)}. An unknown token is NOT ignored - it is read "
            'aloud as words and collapses the render (coverage 0.000). Note there '
            'is no <|emotion:neutral|>, no <|emotion:calm|>, the pause token is '
            '<|prosody:long_pause|> (not pause_long), and <|scene_desc_start|> is '
            'a v2 token that does not exist in v3.')


# ---------------------------------------------------------------------------
# The voice: one reference, 30 seconds, a data URI
# ---------------------------------------------------------------------------


def reference_seconds(voice: ClipsVoice) -> float:
    """Total reference audio, from the clips' declared durations. Raises if any
    clip has none - the 30 s cap cannot be enforced on a guess, and the server
    enforces it with an HTTP 400 either way."""
    missing = [c.path for c in voice.clips if c.seconds is None]
    if missing:
        raise ValueError(
            f"Higgs v3 voice '{voice.name}' has clip(s) with no duration: "
            f"{', '.join(missing)}. The server caps total reference audio at "
            f'{MAX_REFERENCE_SECONDS:.0f} s and rejects an over-long reference with '
            'HTTP 400, so the duration has to be known before the request is built.')
    return float(sum(c.seconds for c in voice.clips))


def check_reference_budget(voice: ClipsVoice) -> float:
    """The 30 s cap, checked client-side so the refusal names the clips."""
    total = reference_seconds(voice)
    # Concatenating N clips inserts N-1 joins of silence, which count.
    total += REFERENCE_JOIN_SECONDS * max(0, len(voice.clips) - 1)
    if total > MAX_REFERENCE_SECONDS:
        raise ValueError(
            f"Higgs v3 voice '{voice.name}': {total:.1f} s of reference audio across "
            f'{len(voice.clips)} clip(s) (joins included) exceeds the server\'s '
            f'{MAX_REFERENCE_SECONDS:.0f} s cap - it answers HTTP 400 "Reference audio '
            'too long". Two of our ~14 s clips is the practical maximum, and the '
            'second clip is worth only +0.012 speaker cosine anyway; same-BOOK clips '
            'are worth +0.076.')
    return total


def reference_data_uri(path: str) -> str:
    """A `data:audio/wav;base64,...` URI for one wav.

    A bare filesystem path is rejected ("The URL must be either a HTTP, data or
    file URL"), and a `file://` URL needs the server launched with
    `--allowed-local-media-path`. The data URI needs neither, which is why every
    render script uses one.
    """
    import base64
    if not os.path.isfile(path):
        raise ValueError(f'Higgs v3 reference clip does not exist: {path}')
    with open(path, 'rb') as handle:
        return 'data:audio/wav;base64,' + base64.b64encode(handle.read()).decode('ascii')


def reference_for(voice: ClipsVoice) -> dict:
    """The single `references` entry for `voice`.

    vllm-omni accepts EXACTLY ONE reference ("'references' only supports a
    single reference; multi-shot voice clone is not supported"), so a
    multi-clip voice must arrive as one CONCATENATED wav whose transcript is
    the clips' transcripts joined in the same order. narrator does not
    concatenate here: the joined wav is a corpus artifact (see the campaign's
    `work/refs/manifest.json`, where `x2` is exactly that), so a multi-clip
    ClipsVoice is refused with the instruction rather than silently using its
    first clip - which would be a different voice at 89 % of ceiling instead of
    90 %, reported as success.
    """
    check_reference_budget(voice)
    if len(voice.clips) != 1:
        raise ValueError(
            f"Higgs v3 voice '{voice.name}' has {len(voice.clips)} clips. vllm-omni "
            'takes EXACTLY ONE reference; several clips must be pre-joined into one '
            f'wav (clips separated by {REFERENCE_JOIN_SECONDS} s of silence) with the '
            'transcripts joined in the same order, and that joined wav given as a '
            'single clip. Refusing to use just the first one.')
    clip = voice.clips[0]
    return {'audio_path': reference_data_uri(clip.path), 'text': clip.transcript}


# ---------------------------------------------------------------------------
# Fine-tuned voices: MERGED CHECKPOINTS, one server each
# ---------------------------------------------------------------------------

#: The only way vllm-omni serves a fine-tuned Higgs voice.
#:
#: **THERE IS NO RUNTIME LoRA.** Measured by the training side, 2026-09-04
#: (HIGGS_FIELD_NOTES.md): vllm-omni exposes no adapter flags at all, and its
#: higgs_audio_v3 talker class does not implement `SupportsLoRA`. So a voice
#: cannot be attached to a running server, and there is no per-request adapter
#: either. Every fine-tuned voice ships as a MERGED CHECKPOINT DIRECTORY
#: (~8.5 GB, Boson's own layout) and the server is started ON that directory.
#: The LoRA is the archival artifact; merging is a CPU step that happens outside
#: narrator.
#:
#: The consequence for this module: the running server is KEYED ON ITS
#: CHECKPOINT DIR, and a request for a different voice is a server RESTART
#: (~55 s warm, up to ~300 s cold).
CHECKPOINT_STRATEGY = 'checkpoint'

#: Refused by name. It was one of two candidate strategies while vllm-omni's
#: LoRA support was unknown; it is now known not to exist.
RETIRED_STRATEGIES = {
    'lora-modules': (
        'vllm-omni cannot load a LoRA at runtime: it exposes no adapter flags '
        'and its higgs_audio_v3 talker does not implement SupportsLoRA '
        '(measured 2026-09-04). A fine-tuned Higgs voice is a MERGED CHECKPOINT '
        "directory served on its own - use kind 'checkpoint' with a "
        'checkpointDir.'),
    'merged-dir': (
        "renamed to 'checkpoint' - the voice IS the checkpoint, and calling it a "
        'merged ADAPTER dir kept implying a base model it sits on top of.'),
}


def check_strategy(strategy: str) -> str:
    """The only strategy is `checkpoint`. Anything else is refused BY NAME.

    A wrong strategy is a server that comes up serving the BASE voice and
    renders a whole book in it, so this never guesses.
    """
    if strategy == CHECKPOINT_STRATEGY:
        return strategy
    if strategy in RETIRED_STRATEGIES:
        raise ValueError(
            f"Higgs v3 strategy '{strategy}' is retired: "
            f'{RETIRED_STRATEGIES[strategy]}')
    raise ValueError(
        f"Unknown Higgs v3 voice strategy '{strategy}'. The only one is "
        f"'{CHECKPOINT_STRATEGY}': a merged checkpoint directory, served on its "
        'own.')


#: THE SAMPLING AUTHORITY OF A MERGED CHECKPOINT, and a REQUIRED file of one.
#:
#: `vllm-omni serve <dir>` resolves sampling from the MODEL DIRECTORY:
#: `--generation-config` defaults to `auto`, so this file - and only this file -
#: sets temperature / top_p / top_k for every request the server answers.
#: `OpenAICreateSpeechRequest` has no temperature / top_p / top_k fields (see the
#: module docstring: pydantic drops them silently), so there is no per-request
#: lever that can correct it.
#:
#: MEASURED, 2026-09-05 (the fine-tune campaign): a merged dir WITHOUT this file
#: makes vllm-omni's stage fallback (`entrypoints/openai/stage_params.py`) hand
#: back a bare `SamplingParams()` - temperature 1.0, **top_p 1.0, top_k
#: DISABLED** - which samples the untruncated 1026-way codebook tail and derails
#: long prompts into babble (seed-dependent collapse to 3-10 s of audio at >= 600
#: chars). With the file present the same server renders the same prompts
#: correctly.
GENERATION_CONFIG_FILE = 'generation_config.json'

#: The keys that make that file the SAMPLING file. A `generation_config.json`
#: carrying none of them (an eos_token_id stub, say) is not the file this needs -
#: the server would read it, find no sampling, and fall back exactly as if it
#: were absent. So its presence is checked by CONTENT, not by name.
GENERATION_CONFIG_SAMPLING_KEYS = ('temperature', 'top_p', 'top_k')

#: What a correct one holds for this model family, recorded so a reader can see
#: what "present and valid" looks like. NOT a default and NEVER written by
#: narrator: the FILE is the authority and the merge is what puts it there.
#: These are `vllm_omni/deploy/higgs_multimodal_qwen3.yaml` stage 0's
#: `default_sampling_params` - which `vllm-omni serve` on the CLI does not read,
#: which is why the values have to be materialised into the model directory.
GENERATION_CONFIG_EXPECTED = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 50,
                              'repetition_penalty': 1.0}

_WHY_GENERATION_CONFIG = (
    'vllm-omni serves a checkpoint DIRECTORY and resolves sampling from it '
    '(--generation-config defaults to "auto"), so that file is the temperature / '
    'top_p / top_k the server actually uses; without it vLLM falls back to a bare '
    'SamplingParams (top_p 1.0, top_k DISABLED), which samples the untruncated '
    '1026-way codebook tail and derails long chunks into babble - and '
    'OpenAICreateSpeechRequest has no sampling fields, so no request can correct '
    'it.')


def require_base_weights_dir(base_dir: str, voice_name: str) -> str:
    """A zero-shot voice's BASE weights directory must BE a directory - and
    that is the whole of what is asked of it.

    NO `generation_config.json`, and that is the point of having this beside
    `require_generation_config` rather than reusing it (2026-09-15). The
    published base has never carried one: `bosonai/higgs-tts-3-4b` at
    239f63fb7b02 lists thirteen files - `config.json`, `chat_template.jinja`,
    the weights, the tokenizer pair, the index, the docs - and no
    `generation_config.json` under any name or subdirectory. That absence is
    not a defect to route around; it is the fact `SERVER_DEFAULT_SAMPLING`
    exists for, and narrator states base sampling EXPLICITLY on both arms
    because of it (`HiggsV3Config.served_sampling`,
    `HiggsV3MlxConfig.mlx_sampling`).

    WHY A VOICE NAMES BASE WEIGHTS AT ALL. Crucible pulls the base at the
    manifest's pinned revision and names that directory in the voices document
    so a clone renders on the bytes the pin names, instead of on whatever
    snapshot the HuggingFace cache happens to hold
    (`crucible/narratorvoices.py`, PHASE3-TTS.md section 4b). Before
    2026-09-15 the only field for it was `checkpointDir`, which means a MERGE,
    and the first zero-shot load Crucible ever made was refused by name for a
    file base weights never had.
    """
    name = _require_voice_name(voice_name, 'require_base_weights_dir')
    if not (base_dir or '').strip():
        raise ValueError(
            f"Higgs v3 voice '{name}' names an empty base weights directory. "
            'A zero-shot clone is the BASE model conditioned on a reference; '
            'with no directory there is nothing to condition.')
    if not os.path.isdir(base_dir):
        raise ValueError(
            f"Higgs v3 voice '{name}' names the base weights directory "
            f'{base_dir}, which is not a directory. That directory IS what '
            'the model is loaded from - there is nothing to serve.')
    return base_dir


def _require_voice_name(voice_name: str, caller: str) -> str:
    name = (voice_name or '').strip()
    if not name:
        raise ValueError(
            f'{caller}() needs the VOICE NAME: every refusal it makes has to '
            'say which voice is misconfigured, and an unnamed one is a refusal '
            'nobody can act on.')
    return name


def require_generation_config(checkpoint_dir: str, voice_name: str) -> dict:
    """Read a merged checkpoint's `generation_config.json`, or refuse BY NAME.

    THE ONE PLACE the file is validated, for both arms (see
    `checkpoint_serve_target`). Returns the parsed document so a caller that
    needs the values - the MLX backend, which has no server to read them for it -
    takes them from the FILE and never from a constant here.

    Nothing is copied, synthesized or defaulted. A merged dir that does not
    carry this file is MISCONFIGURED, not under-specified: the merge that built
    it is what puts the file there (it asserts byte-equality with the base, or
    copies a recorded per-run override), and narrator writing one would be
    narrator deciding a model's sampling.

    ASKED OF A MERGE ONLY. Base weights carry no such file and never did, so a
    voice that names ITS BASE WEIGHTS reaches `require_base_weights_dir`
    instead; `voice_serve_target` is the one place that chooses between them.
    """
    name = _require_voice_name(voice_name, 'require_generation_config')
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(
            f"Higgs v3 voice '{name}' names the merged checkpoint directory "
            f'{checkpoint_dir}, which is not a directory. The checkpoint IS the '
            'voice - there is nothing to serve, and nothing to read its sampling '
            'from.')
    path = os.path.join(checkpoint_dir, GENERATION_CONFIG_FILE)
    if not os.path.isfile(path):
        raise ValueError(
            f"Higgs v3 voice '{name}': the merged checkpoint {checkpoint_dir} does "
            f'not carry {GENERATION_CONFIG_FILE}, which is a REQUIRED file of a '
            f'Higgs v3 checkpoint - {_WHY_GENERATION_CONFIG} Re-merge the '
            'checkpoint (the merge writes it) rather than dropping one in manually.')
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            text = handle.read()
    except OSError as exc:
        # Permissions, a DIRECTORY named generation_config.json, a broken
        # symlink, an unreadable mount. Every other state of this file is
        # refused with the voice, the path and why; a bare OSError from here
        # would be the one refusal in this feature that names neither.
        raise ValueError(
            f"Higgs v3 voice '{name}': {path} exists but could not be read "
            f'({exc}). {_WHY_GENERATION_CONFIG}') from exc
    try:
        document = json.loads(text)
    except ValueError as exc:
        raise ValueError(
            f"Higgs v3 voice '{name}': {path} is not parseable JSON ({exc}). "
            f'{_WHY_GENERATION_CONFIG} A file the server cannot parse is a file it '
            'does not apply.') from exc
    if not isinstance(document, dict):
        raise ValueError(
            f"Higgs v3 voice '{name}': {path} holds a "
            f'{type(document).__name__}, not a JSON object of sampling '
            f'parameters. {_WHY_GENERATION_CONFIG}')
    missing = [key for key in GENERATION_CONFIG_SAMPLING_KEYS
               if key not in document]
    if missing:
        raise ValueError(
            f"Higgs v3 voice '{name}': {path} carries no "
            f"{', '.join(missing)}. A generation_config.json that does not carry "
            'sampling is not the file this needs - the server reads it, finds no '
            f'sampling and falls back exactly as if it were absent. '
            f'{_WHY_GENERATION_CONFIG} It should hold '
            f'{GENERATION_CONFIG_EXPECTED} for this model family.')
    _check_sampling_types(document, path, name)
    return document


def _check_sampling_types(document: dict, path: str, name: str) -> None:
    """The three (four, with `repetition_penalty`) values must be NUMBERS OF THE
    RIGHT KIND, refused by name exactly as absence is.

    Presence without type is not validation. `"top_k": 50.7` truncates silently
    to 50 the moment anything calls `int()` on it; `"temperature": null` raises a
    bare TypeError from whichever caller touched it first, naming neither the
    voice nor the file; and on the served arm nothing reads the values at all, so
    a malformed one goes straight to vLLM. A number that is wrong in a way
    nobody says out loud is the failure this whole file exists to stop.

    `bool` is excluded deliberately: `True` is an `int` in Python and
    `"top_k": true` is not a top-k.
    """
    for key in ('temperature', 'top_p', 'repetition_penalty'):
        if key not in document:
            # Only `repetition_penalty` can be absent here - the other two are
            # required above - and an absent one is vLLM's own default, which
            # narrator neither states nor checks.
            continue
        value = document[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Higgs v3 voice '{name}': {path} gives {key} as "
                f'{value!r} ({type(value).__name__}), which is not a number. '
                f'{_WHY_GENERATION_CONFIG}')
        value = float(value)
        if value != value or value in (float('inf'), float('-inf')):
            raise ValueError(
                f"Higgs v3 voice '{name}': {path} gives {key} as {value!r}, "
                'which is not a finite number.')
        if key == 'temperature' and value < 0.0:
            raise ValueError(
                f"Higgs v3 voice '{name}': {path} gives temperature {value!r}. "
                'Temperature is a divisor of the logits and cannot be negative; '
                '0.0 is greedy decoding.')
        if key == 'top_p' and not 0.0 < value <= 1.0:
            raise ValueError(
                f"Higgs v3 voice '{name}': {path} gives top_p {value!r}. "
                'top_p is a probability mass and must be in (0.0, 1.0]. (1.0 is '
                'accepted and means the UNTRUNCATED tail - if a checkpoint '
                'really wants that, it says so here and narrator does not '
                'second-guess it.)')
        if key == 'repetition_penalty' and value <= 0.0:
            raise ValueError(
                f"Higgs v3 voice '{name}': {path} gives repetition_penalty "
                f'{value!r}, which must be positive (1.0 is no penalty).')
    top_k = document['top_k']
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError(
            f"Higgs v3 voice '{name}': {path} gives top_k as {top_k!r} "
            f'({type(top_k).__name__}). top_k is a COUNT of candidate tokens and '
            'must be a whole number - a float here is silently truncated by '
            'every consumer, which is a different sampling than the file states.')
    if top_k < 0:
        raise ValueError(
            f"Higgs v3 voice '{name}': {path} gives top_k {top_k!r}. It must be "
            'a non-negative count. (0 is accepted and DISABLES top-k - a '
            'checkpoint that states that is stating it deliberately.)')


def checkpoint_serve_target(checkpoint_dir: str, voice_name: str) -> str:
    """What `vllm-omni serve <...>` is pointed at for a MERGED FINE-TUNE.

    It IS the checkpoint dir - there are no extra launch arguments, because
    there is no adapter to name. Kept as a function so the one place that
    decides "which directory does this voice's server run on" has a name and a
    test - and, since 2026-09-05, so that the one place also PROVES the directory
    carries the sampling the server will read out of it
    (`require_generation_config`). Both v3 arms reach it through
    `voice_serve_target`.
    """
    if not (checkpoint_dir or '').strip():
        raise ValueError(
            'Higgs v3: a fine-tuned voice needs its merged checkpoint directory '
            '(checkpointDir). There is no adapter to load onto a base server.')
    require_generation_config(checkpoint_dir, voice_name)
    return checkpoint_dir


def voice_serve_target(voice):
    """THE DIRECTORY THIS VOICE'S MODEL IS LOADED FROM, validated for what it
    is - or `None` when the voice names none and the arm's own base variable
    decides.

    ONE function because there is one question, and two answers because a
    voice may name its weights in either of two fields, which mean different
    things (2026-09-15):

      `checkpoint_dir`   a MERGED FINE-TUNE. The weights ARE the voice, and
                         the directory must carry the `generation_config.json`
                         the server reads its sampling out of
                         (`checkpoint_serve_target`).
      `base_dir`         the BASE weights a zero-shot clone is conditioned on,
                         pinned by whoever wrote the voices document. No
                         `generation_config.json` is asked of it and none
                         exists (`require_base_weights_dir`); base sampling is
                         STATED by narrator instead.

    Both at once is refused: one server runs on one directory, and a voice
    that names two has not said which.
    """
    checkpoint = (getattr(voice, 'checkpoint_dir', None) or '').strip()
    base = (getattr(voice, 'base_dir', None) or '').strip()
    if checkpoint and base:
        raise ValueError(
            f"Higgs v3 voice '{voice.name}' names a merged checkpoint "
            f'({checkpoint}) AND base weights ({base}). Those are different '
            'models and one server runs on one directory: a fine-tune\'s '
            'voice is in its weights and a clone\'s is in its reference, so a '
            'voice that claims both has not said which model to load.')
    if checkpoint:
        return checkpoint_serve_target(checkpoint, voice.name)
    if base:
        return require_base_weights_dir(base, voice.name)
    return None


# ---------------------------------------------------------------------------
# The request and the response
# ---------------------------------------------------------------------------


def cap_frames(text: str, chars_per_sec: float = 15.0, slack: float = 2.0,
               slack_frames: int = 150) -> int:
    """`max_new_tokens`, in FRAMES, by work/render_final.py's formula:
    `int(len(text) / 15.0 * 25 * 2.0) + 150`.

    (work/confirm.py uses the same one. render_v3.py, the earlier script, used
    1.8x + 100 - the v2 formula; the delivered render used this.)
    """
    expected_seconds = len(text or '') / chars_per_sec
    return int(expected_seconds * FRAMES_PER_SECOND * slack) + slack_frames


def build_request_body(text: str, voice, max_new_tokens: int, seed=None,
                       sampling=None, model: str = SERVED_MODEL_NAME) -> dict:
    """The POST body for one chunk, as work/render_final.py and work/confirm.py
    send it.

    `voice` is a ClipsVoice (reference cloning), a DefaultVoice (the model's
    own voice, or a fine-tune whose weights are the voice - no `references` key
    at all), or None (the same, unnamed).

    `sampling` EMPTY OR NONE sends no `extra_params` at all, which means THE
    MODEL DIRECTORY'S `generation_config.json` decides - a validated merged
    checkpoint's four numbers, or, for a directory without the file, a bare
    `SamplingParams()` (top_p 1.0, top_k DISABLED: the babble case, 12.8d). So
    an empty `sampling` is only correct for a checkpoint voice; base weights
    must state the values, and `HiggsV3Config.served_sampling` is what decides
    which of the two this is. Anything given rides in `extra_params`, never at
    the top level - see the module docstring.
    """
    if not (text or '').strip():
        raise ValueError('Higgs v3 request: no text')
    validate_control_tokens(text)
    body = {
        'model': model,
        'input': text,
        'response_format': 'wav',
        'max_new_tokens': int(max_new_tokens),
    }
    if seed is not None:
        body['seed'] = int(seed)
    if voice is not None and not isinstance(voice, DefaultVoice):
        # A DefaultVoice sends NO `references` key: it IS the model's own voice,
        # which v3 serves text-only. The 30 s cap and the transcript rules have
        # nothing to apply to.
        body['references'] = [reference_for(voice)]
    if sampling:
        if 'seed' in sampling:
            # ONE place carries the seed: the top-level field. vllm-omni copies
            # extra_params onto the stage-0 sampling params, so a seed in both
            # is two sources for one number and nothing says which wins.
            raise ValueError(
                "Higgs v3 sampling carries a seed. The seed is the request's "
                'TOP-LEVEL `seed` field and only that - pass it as build_request_body'
                "(..., seed=...), never inside extra_params.")
        stray = sorted(set(sampling) - set(SERVER_DEFAULT_SAMPLING))
        if stray:
            raise ValueError(
                f'Higgs v3 sampling carries key(s) the stage-0 params do not have: '
                f"{', '.join(stray)}. Known: "
                f"{', '.join(sorted(SERVER_DEFAULT_SAMPLING))}.")
        body['extra_params'] = dict(sampling)
    return body


def decode_response(body: bytes, content_type: str = None):
    """The response body -> (float32 mono at 24 kHz, sample rate).

    FORMAT ASSUMPTION, STATED AS ONE: with `"response_format": "wav"` the
    endpoint returns a WAV FILE as the raw response body - that is what every
    render script does (`sf.read(io.BytesIO(r.content))`) and there is no
    recorded capture in the campaign to check byte for byte. If a capture ever
    lands in `<campaign>/higgs/captures/`, the fixture in
    tests/test_higgs_v3.py replaces the script-derived one and this function is
    where any correction goes - it is the single place the format is decided.

    A 200 whose body is not a WAV is REFUSED by name (see below).

    Multi-channel is averaged down, as the scripts do. NO TRIM AND NO FADE is
    applied: the patched server already drops every sentinel frame by token
    identity (work/patch_sentinel_filter.py), and the fades belong to assembly
    (`edge_fade`).
    """
    import soundfile as sf
    if not body:
        raise HiggsV3ServerError('Higgs v3 returned an empty body')
    # A 200 that is not a WAV is the shape a proxy, an error page or a changed
    # `response_format` produces, and soundfile's own message for it ("Error
    # opening <_io.BytesIO ...>: Format not recognised") names nothing a reader
    # can act on. Check the RIFF/WAVE magic and say what actually arrived.
    if not (body[:4] == b'RIFF' and body[8:12] == b'WAVE'):
        head = body[:16]
        raise HiggsV3ServerError(
            'Higgs v3: expected a WAV body from /v1/audio/speech; got content-type '
            f'{content_type!r}, first bytes {head!r}. The request asks for '
            '"response_format": "wav" and every render script reads the body as a '
            'WAV file - see decode_response. A JSON or HTML body here is usually a '
            'proxy or an error page reaching us with status 200.')
    audio, rate = sf.read(io.BytesIO(body), dtype='float32', always_2d=True)
    audio = audio.mean(axis=1)
    if int(rate) != SAMPLE_RATE:
        raise HiggsV3ServerError(
            f'Higgs v3 returned {rate} Hz audio; this codec is fixed at '
            f'{SAMPLE_RATE} Hz and a manifest built on the wrong rate mis-times '
            'every cue after it.')
    return np.asarray(audio, dtype=np.float32).reshape(-1), int(rate)


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


class HiggsV3ServedBackend(GuestOwnedServer):
    """`narrator.engine.protocol.ServedBackend` for Higgs v3 under **vllm-omni**.

    The SIBLING STACK is `sgl_served.HiggsSglServedBackend` (SGLang-Omni 0.1.4),
    and everything the two share - the ownership marker, the /proc listener scan,
    the watchdog, adoption, TERM-only teardown - is `served_common`. What is here
    is vllm-omni's: the launch line, `extra_params` sampling, the sentinel-filter
    proof, and `/v1/models`' `root` as the model identity.

    Two modes, chosen by what it is given:

      ATTACH   `base_url` names a server somebody else started (the operator,
               or a previous session). `start()` is a no-op, `stop()` refuses to
               kill a process it did not launch.
      LAUNCH   `serve_script` names the campaign's `serve_v3.sh`. `start()` runs
               it - through `wsl.exe -d <distro> bash <script>` on Windows,
               directly on Linux - and `stop()` terminates it.

    narrator never writes its own launch line: the script carries the CUDA_HOME
    and flashinfer workarounds without which the server does not start at all.
    """

    HEALTH_PATH = HEALTH_PATH
    LOG_TAG = '[HIGGS3]'
    SERVER_LOG_ENV = SERVER_LOG_ENV
    READY_FAILURE_HINT = (
        'The usual causes are the flashinfer JIT (needs CUDA_HOME + '
        'VLLM_USE_FLASHINFER_SAMPLER=0) and OOM at a gpu-memory-utilization the '
        'card cannot honour.')

    def __init__(self, base_url: str = None, serve_script: str = None,
                 wsl_distro: str = None, extra_args=None,
                 checkpoint_dir: str = None, server_log: str = None,
                 concurrency: int = None):
        # THE LAUNCHER, IN THREE MODES, decided here and nowhere else:
        #
        #   attach    a base_url (argument or BASE_URL_ENV) names a server
        #             somebody else started. No launcher at all.
        #   operator  a serve_script (argument or SERVE_SCRIPT_ENV) overrides
        #             narrator's own. Refused when the path is not there.
        #   packaged  neither: narrator runs ITS OWN script, the one that ships
        #             in the package. See PACKAGED_SERVE_SCRIPT for why this is
        #             a definition and not a fallback.
        base_url = (base_url or os.environ.get(BASE_URL_ENV) or '').strip()
        serve_script = (serve_script
                        or os.environ.get(SERVE_SCRIPT_ENV) or '').strip()
        if base_url:
            self.launcher_source = LAUNCHER_ATTACH
        elif serve_script:
            self.launcher_source = LAUNCHER_OPERATOR
            _check_override_script(serve_script, SERVE_SCRIPT_ENV)
        else:
            self.launcher_source = LAUNCHER_PACKAGED
            serve_script = packaged_serve_script()
        if extra_args:
            raise ValueError(
                'serve_v3.sh takes no arguments - it `exec`s a fixed vllm-omni '
                f'command line - so {sorted(extra_args)} cannot be passed through it. '
                'Either the script gains a "$@" passthrough (their file, not ours), '
                'or launch the server yourself with those arguments and point '
                f'{BASE_URL_ENV} at it.')
        # ATTACH: the URL names the server. LAUNCH: the server will bind
        # whatever HIGGS_HOST/HIGGS_PORT the wrapper exports, and that pair is
        # decided HERE so the poll and the bind cannot name different ports.
        self.base_url = base_url or launch_base_url()
        self.serve_script = serve_script
        # THE WIDTH OF THE BATCH, and stage 0's max_num_seqs. Required for a
        # launch (the wrapper exports it); optional for an attach, where the
        # engine states it from the same variable.
        if concurrency is not None and int(concurrency) < 1:
            raise ValueError(f'Higgs v3: concurrency {concurrency} must be >= 1.')
        if concurrency is None and serve_script:
            # A launching backend must state it (the wrapper exports it); the
            # one source is the contract variable, which refuses when unset.
            concurrency = serve_concurrency()
        self.concurrency = int(concurrency) if concurrency else None
        self.wsl_distro = (wsl_distro or os.environ.get(WSL_DISTRO_ENV)
                           or 'Ubuntu')
        # THE SERVER'S OUTPUT IS EVIDENCE, so it goes to a file this backend
        # owns rather than to DEVNULL.
        #
        # It used to be DEVNULL on both streams, which threw away the only
        # record of what the decode path did - and the sentinel filter's own
        # proof is written into exactly that stream (`verify_sentinel_filter`).
        # It also meant a server that died at startup left its reason nowhere,
        # so `wait_ready`'s "check its log" pointed at nothing.
        #
        # TWO PATHS, kept apart on purpose:
        #   launch_log  where WE write, when we start the server. The session's
        #               process dir when the engine gave us one, else a
        #               per-instance file beside the pid file (same naming, same
        #               reason: two workers must never share one).
        #   named_log   what an OPERATOR says an ATTACHED server writes to. No
        #               default - see SERVER_LOG_ENV.
        self._named_log = (os.environ.get(SERVER_LOG_ENV) or '').strip() or None
        self.launch_log = ((server_log or '').strip()
                           or self.default_launch_log('narrator-higgs3'))
        # THE SENTINEL REPORT, the machine-readable half of the same evidence.
        # Same two-path shape as the log and for the same reasons: ours when we
        # launch (named from `launch_log`, so it inherits its per-session or
        # per-instance uniqueness), the operator's when we attach. See
        # SENTINEL_REPORT_ENV for why there is no default on the attach side.
        self.sentinel_report = self.launch_log + SENTINEL_REPORT_SUFFIX
        self._named_report = (os.environ.get(SENTINEL_REPORT_ENV)
                              or '').strip() or None
        # Which of the two is the PROOF stream is decided by which mode this
        # backend is in, and `start()` corrects it if it adopts a server that
        # was already up (that server's output is not ours either).
        self._log_is_ours = bool(serve_script)
        self.server_log = self.launch_log if serve_script else self._named_log
        self._log_handle = None
        self.spec = BackendSpec(
            kind='served', name='vllm-omni', version='0.28.0',
            base_url=self.base_url,
            server_log=self.server_log,
            notes=('higgs-audio-v3-tts-4b; requires patch_vllm.py and '
                   'patch_sentinel_filter.py in the higgs3 env'))
        # THE SERVER IS KEYED ON THIS. A fine-tuned Higgs voice is a merged
        # checkpoint the server runs ON, so "which voice is up" and "which
        # directory is up" are the same question - and a request for another one
        # is a restart, not a message. In LAUNCH mode this is the directory the
        # wrapper exports as HIGGS_MODEL_DIR; in either mode it is what
        # `check_serves_expected_model` holds the RUNNING server to. None means
        # the base weights.
        self.checkpoint_dir = (checkpoint_dir or '').strip() or None
        self._proc = None
        self._guest_pid = None
        # WHICH OF THE THREE, said once, at construction. `start()` already logs
        # the command it runs, and that line cannot answer "whose script is
        # this" - a path under site-packages and a path under a campaign
        # directory look alike in a log, and the difference is which flags a
        # 19 GB server came up with.
        log(f'{self.LOG_TAG} launcher: ' + {
            LAUNCHER_ATTACH: f'none - attaching to {self.base_url} '
                             f'({BASE_URL_ENV})',
            LAUNCHER_OPERATOR: f"{self.serve_script} (the operator's "
                               f'{SERVE_SCRIPT_ENV})',
            LAUNCHER_PACKAGED: f"{self.serve_script} (narrator's own, shipped "
                               'in the package)',
        }[self.launcher_source], flush=True)

    # -- lifecycle -----------------------------------------------------------
    #
    # `owner_id`, `start`, `stop`, `ping`, `wait_ready`, the /proc listener scan,
    # the watchdog, the group signal and the log handling are `served_common
    # .GuestOwnedServer`'s - they are the same problem on both stacks and were
    # solved once. What is below is vllm-omni's own launch line.

    def proof_report(self):
        """The sentinel report a proof reads, or None.

        Exactly `proof_log`'s shape, one line below it: ours when we launched
        the server, the operator's named file when we attached to (or adopted)
        somebody else's, and None when neither. `_log_is_ours` is the single
        flag both follow, because the two files are one decision - the streams
        of a server we did not start are not ours to read either, and `start()`
        clears that flag when it adopts a server that was already up.
        """
        return self.sentinel_report if self._log_is_ours else self._named_report

    def _open_log(self) -> None:
        """Open the launch log AND truncate the sentinel report.

        THE REPORT IS TRUNCATED HERE, at the one moment narrator is about to
        launch a server it owns - after `start()` has ruled out attach mode, an
        already-running process of ours and an adoptable stranger, and
        immediately before the Popen. A report is APPEND-ONLY from the server's
        side (several decode workers write to it at once), so nothing else would
        ever clear it, and a file left over from the previous run in the same
        session dir would let this run's proof pass on the last run's evidence.
        That is the precise hazard SERVER_LOG_ENV's comment refuses to guess a
        path over, and it must not come back in through the report.

        Failure is LOUD, exactly as it is for the log: a server whose evidence
        goes nowhere renders ~19 GB worth of audiobook that nothing can certify.
        """
        super()._open_log()
        try:
            with open(self.sentinel_report, 'wb'):
                pass
        except OSError as exc:
            raise HiggsV3ServerError(
                f'Higgs v3: could not truncate the sentinel report '
                f'{self.sentinel_report} ({exc}). That file is where the '
                'patched decode path records what it did to every chunk, and '
                'a run that cannot clear it would be proved by the PREVIOUS '
                "run's records.") from exc

    def _launch_exports(self) -> str:
        """The `export ...` prefix of the wrapper: every launch-script knob
        narrator has an opinion about, stated explicitly.

        HIGGS_MODEL_DIR is exported for a checkpoint voice and UNSET for the
        base weights - unset, not left alone, because the worker's own
        environment may carry one from a caller and inheriting it would start a
        fine-tune under a request for the base speaker.
        """
        host_port = self.base_url.split('://', 1)[-1].rstrip('/')
        host, _, port = host_port.rpartition(':')
        # THE REPORT PATH IS EXPORTED IN THE FORM THE SERVER SEES. narrator may
        # be a Windows process launching the server inside the distro, in which
        # case the same file is `C:\...` here and `/mnt/c/...` there - the same
        # split `serve_script` and `checkpoint_dir` already have, and getting it
        # wrong would leave the server writing a path that does not exist while
        # narrator waited to read one that was never created.
        report = (_to_wsl(self.sentinel_report) if sys.platform == 'win32'
                  else self.sentinel_report)
        exports = [
            f'{SERVE_HOST_ENV}={shlex.quote(host)}',
            f'{SERVE_PORT_ENV}={shlex.quote(port)}',
            f'{SERVE_MAX_NUM_SEQS_ENV}={self.concurrency}',
            f'{OWNER_ENV}={shlex.quote(self.owner_id())}',
            f'{SENTINEL_REPORT_ENV}={shlex.quote(report)}',
            # THE CONTEXT, TRANSLATED. The launcher's name is
            # `HIGGS_MAX_MODEL_LEN`; the name Crucible sets per voice is
            # `HIGGS_CONTEXT_LENGTH`. `context_tokens()` reads both and refuses
            # a disagreement, so what is exported here is the one number this
            # server is starting with - stated, not inherited, like every other
            # knob narrator has an opinion about.
            f'{SERVE_MAX_MODEL_LEN_ENV}={context_tokens()}',
        ]
        if self.checkpoint_dir:
            target = (_to_wsl(self.checkpoint_dir) if sys.platform == 'win32'
                      else self.checkpoint_dir)
            exports.append(f'{SERVE_MODEL_DIR_ENV}={shlex.quote(target)}')
            model = ''
        else:
            model = f'unset {SERVE_MODEL_DIR_ENV}; '
        return f'{model}export {" ".join(exports)}; '

    def _wrapper(self) -> str:
        """The shell the launcher actually runs.

        Three things, in order:

        1. The exports: the launch script's knobs and the OWNER marker
           (`_launch_exports`). The marker rides into the server's environ and
           is how `_server_on_port` tells OURS from a stranger's.
        2. `setsid bash <script> &` - the script detached from this shell's
           group so a signal to the wrapper never reaches the server by
           accident, and `wait` so this shell lives as long as the script.
           NO PID IS RECORDED HERE: vllm-omni re-sessions itself after exec
           (measured 84072 -> 84096 -> 84098), so `$!` is dead within a second
           of the server coming up. The listener on our port, found by
           `/proc/net/tcp` after `/health` answers, is the only pid worth
           knowing, and its group is read at the moment of the signal.
        3. THE WATCHDOG, on the guest arm only: a stdlib python3 process,
           detached with setsid, that watches the OWNER pid (this narrator
           worker) and, when it is gone, SIGTERMs the group of whatever is
           listening on our port WITH OUR MARKER - then waits for the port to
           free and exits. That is what makes "hit Stop" and "the app died"
           bring the server down: a worker killed with SIGKILL never runs its
           own cleanup, and until this existed the server outlived every such
           job holding 24 GB (Owen, 2026-09-05: "it should bring it down if i
           hit stop or if bookforge app dies"). The watchdog also exits on its
           own once the marked listener it saw is gone while the owner lives
           (an ordinary stop), so it never lingers - and it must not: the
           wrapper's `wait` waits for it too, and a watchdog that outlived its
           server held every stop() for its full timeout. On the
           Windows arm the owner is a host pid the guest cannot see, so no
           watchdog is started and `stop()` is the only teardown - stated in
           OWNER_ENV's prefix, not guessed at.
        """
        script = (_to_wsl(self.serve_script) if sys.platform == 'win32'
                  else self.serve_script)
        return ''.join([self._launch_exports(),
                        f'setsid bash {shlex.quote(script)} & ',
                        self._watchdog_clause(),
                        'wait'])

    def launch_command(self) -> list:
        """The command `start()` runs. Public so a test and a log line can see
        it without a GPU.

        `--exec` ON THE WINDOWS ARM IS LOAD-BEARING, and its absence was a live
        bug. Without it `wsl.exe` hands the command line to the distro's DEFAULT
        SHELL, which expands every `$` before `bash -c` sees the script - so
        `_wrapper()`'s `echo $! > <pidfile>; wait $!` wrote an EMPTY pid file and
        degenerated into a bare `wait`. Measured on example-pc 2026-09-05 through
        this exact argv: bare and `--` both write `pid=[]`; `--exec` writes
        `pid=[42679]`. The consequence is not cosmetic - `stop()` could then
        never signal the server BY PID inside the distro, which is the one path
        that stops a ~14 GB vllm-omni after the Windows-side `wsl.exe` relay has
        been terminated out from under it. `--` is NOT a substitute: it stops
        wsl.exe parsing its own options and still runs the default shell.
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

    def verify_sentinel_filter(self) -> dict:
        """PROOF (a) OF THE SENTINEL FILTER: read the records the patch WRITES.

        The other half, (b) "no one-frame trim left in the stage processor", is
        a static grep BookForge's doctor runs before any server starts. This is
        the half that needs the running decode path to say what it did.

        IT USED TO GREP THE SERVER'S LOG FILE. Three regexes over vLLM's
        formatted warnings decided whether a 19 GB model could render a book -
        crucible/docs/ARCHITECTURE.md R4's headline example, and a proof that
        could pass on nothing at all the moment a message was re-worded. Patch
        v3 (2026-09-13) now appends one JSON record per filter invocation to the
        file named in SENTINEL_REPORT_ENV, and this reads those records. The
        warnings still go to the log, unchanged, for a human to read.

        WHAT IS ASSERTED, and why each one:

          * THE REPORT EXISTS AND IS NOT EMPTY. No report, an unreadable one, or
            one with zero records is a REFUSAL. The filter records EVERY
            invocation including clean ones, so an empty file means the decode
            path did not run, or ran in an env whose patch predates the report -
            and "no evidence" must never read as "no problem". THIS IS THE
            ASSERTION THE LOG-GREP VERSION COULD NOT MAKE: a log with no
            matching lines was indistinguishable from a clean render, and it
            passed.
          * EVERY RECORD IS THIS VERSION. A record carrying any `v` but
            SENTINEL_REPORT_VERSION is an env holding a patch generation this
            code was not written against; skipping it would be the same silent
            pass by another route.
          * ZERO SYNC-PATH INTERIOR DROPS. The sync path takes the FULL filter,
            so an interior drop there is a frame that failed the token test
            while sitting between two good ones - not an expected shape on any
            real generation, and never observed offline. One is a refusal.
          * ZERO EMPTY CHUNKS (a sync record whose `kept` is 0): every frame of
            that chunk carried a stream sentinel, so it produced no audio at all.
          * ZERO OUT-OF-RANGE FRAMES LEFT ON A FINAL ASYNC WINDOW. This is the
            assertion the old one was groping for and had backwards. Patch v2
            removes the trailing sentinel run BY IDENTITY before it counts, so
            on the FINAL window anything still out of range is a sentinel the
            trim could not reach - and it was then substituted with codec code
            0, a VALID code that decodes to real, audible sound on the end of
            that chunk. The old check expected exactly `2` there, which is what
            v1 printed and what v2 exists to eliminate.
          * A NON-FINAL window's leftovers are COUNTED, NOT REFUSED. There they
            can only be the left-context region Stage 1 discards by frame count
            anyway (patch_sentinel_filter.py, async step 2). The count is
            returned so a ledger can carry it; it is not evidence of damage, and
            refusing on it would fail correct servers.

        Returns a report - path, records read, per-path counts - so a caller can
        put it in a ledger.
        """
        path = self.proof_report()
        if not path:
            raise HiggsV3ServerError(
                'Higgs v3 sentinel proof: there is no sentinel report to read. '
                'This backend attached to a server it did not start, so the file '
                'its decode path writes is wherever that operator pointed it. '
                f'Name it in {SENTINEL_REPORT_ENV} - the SAME path that server '
                'carries in its own environment - and narrator will read it. '
                'narrator will not guess one, because a stale report from an '
                'earlier run would let this proof pass on evidence from a server '
                'that is no longer up.')
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                raw = handle.read().splitlines()
        except OSError as exc:
            raise HiggsV3ServerError(
                f'Higgs v3 sentinel proof: the sentinel report {path} could not '
                f'be read ({exc}). The proof IS the record - refusing to report a '
                'render as proved when nothing was read. An env still carrying '
                'patch v1 or v2 writes no report at all, and BookForge\'s Higgs '
                'doctor reports that env as patch:higgs-sentinel-filter=stale.'
            ) from exc

        counts = {'sync': 0, 'async': 0}
        non_final_substitutions = 0
        trailing_frames_trimmed = 0
        for number, text in enumerate(raw, start=1):
            text = text.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except ValueError as exc:
                raise HiggsV3ServerError(
                    f'Higgs v3 sentinel proof: {path} line {number} is not JSON '
                    f'({exc}): {text[:200]!r}. The filter emits one compact record '
                    'per line with a single write() to an O_APPEND fd; a torn line '
                    'means something other than the filter is writing to this '
                    'file.') from exc
            if not isinstance(record, dict):
                raise HiggsV3ServerError(
                    f'Higgs v3 sentinel proof: {path} line {number} is a '
                    f'{type(record).__name__}, not a record: {text[:200]!r}')
            if record.get('v') != SENTINEL_REPORT_VERSION:
                raise HiggsV3ServerError(
                    f'Higgs v3 sentinel proof: {path} line {number} carries '
                    f'v={record.get("v")!r}, not v={SENTINEL_REPORT_VERSION}. That '
                    'is a patch generation this narrator was not written against. '
                    'Re-run the installer so the env carries the matching '
                    'patch_sentinel_filter.py; a render made against the other one '
                    'is not read as proved by this one.')
            where = record.get('path')
            if where == 'sync':
                counts['sync'] += 1
                interior = int(record.get('interior', 0))
                if interior:
                    raise HiggsV3ServerError(
                        f'Higgs v3 sentinel proof FAILED in {path}: the SYNC path '
                        f'dropped {interior} INTERIOR sentinel frame(s) - a frame '
                        'that failed the token test while sitting between two good '
                        'ones. That is not an expected shape on any real generation '
                        'and has never been observed offline; the filter drops it '
                        'and records it rather than splicing silently. The audio '
                        'for that chunk has a frame missing from its middle. The '
                        f'record was: {text}')
                if int(record.get('total', 0)) and not int(record.get('kept', 0)):
                    raise HiggsV3ServerError(
                        f'Higgs v3 sentinel proof FAILED in {path}: a chunk was '
                        'emitted with NO audio - every one of its frames carried a '
                        f'stream sentinel. The record was: {text}')
            elif where == 'async':
                counts['async'] += 1
                trailing_frames_trimmed += int(record.get('trimmed', 0))
                outside = int(record.get('outside', 0))
                if outside and record.get('final') is True:
                    raise HiggsV3ServerError(
                        f'Higgs v3 sentinel proof FAILED in {path}: {outside} '
                        'frame(s) of the FINAL async window were still out of range '
                        'AFTER the trailing sentinel run had been removed by '
                        'identity. Those are sentinels the trim could not reach, '
                        'and they were then substituted with codec code 0 - a VALID '
                        'code that decodes to real sound on the end of that chunk. '
                        f'The record was: {text}')
                if not record.get('final'):
                    non_final_substitutions += outside
            else:
                raise HiggsV3ServerError(
                    f'Higgs v3 sentinel proof: {path} line {number} names path '
                    f'{where!r}, which is neither "sync" nor "async": {text[:200]!r}')

        total = counts['sync'] + counts['async']
        if not total:
            raise HiggsV3ServerError(
                f'Higgs v3 sentinel proof: {path} holds no records. The filter '
                'writes one for EVERY invocation, clean ones included, so an empty '
                'report means the decode path never ran under this narrator - '
                'refusing to report a render as proved on an empty file. This is '
                'exactly the case the old log-grep proof PASSED: a log with no '
                'matching lines looked the same as a clean render.')

        log(f'{self.LOG_TAG} sentinel proof OK: {total} record(s) from {path} '
            f'({counts["sync"]} sync, {counts["async"]} async); 0 sync interior '
            f'drops, 0 empty chunks, 0 out-of-range frames on a final window; '
            f'{trailing_frames_trimmed} trailing sentinel frame(s) trimmed, '
            f'{non_final_substitutions} non-final left-context substitution(s)',
            flush=True)
        return {
            'report': path,
            'records': total,
            'syncRecords': counts['sync'],
            'asyncRecords': counts['async'],
            'syncInteriorDrops': 0,
            'emptyChunks': 0,
            'finalWindowSubstitutions': 0,
            'trailingFramesTrimmed': trailing_frames_trimmed,
            'nonFinalSubstitutions': non_final_substitutions,
        }

    def _models_payload(self) -> list:
        """The rows of `/v1/models`. Raises if it cannot be read."""
        try:
            with urllib.request.urlopen(self.base_url + MODELS_PATH,
                                        timeout=10) as response:
                payload = json.loads(response.read().decode('utf-8'))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise HiggsV3ServerError(
                f'Higgs v3: {self.base_url}{MODELS_PATH} could not be read ({exc}). '
                'Something is listening on that port but it does not answer the '
                'OpenAI model list, so it cannot be identified - refusing to send a '
                'render to it.') from exc
        rows = payload.get('data') if isinstance(payload, dict) else None
        return [row for row in (rows or []) if isinstance(row, dict)]

    def served_models(self) -> list:
        """The model ids `/v1/models` reports. Raises if it cannot be read."""
        return [row.get('id') for row in self._models_payload()]

    def running_checkpoint(self):
        """The model path the server on this URL was started on, or None.

        DISCOVERED: vllm-omni's `/v1/models` row for the served name carries
        `"root": <model path>` - the directory `vllm-omni serve` was pointed at,
        which for a fine-tune is the merged checkpoint and for the base weights
        is the HF-cache snapshot (measured 2026-09-05; see CHECKPOINT_ENV).
        When the row has no root - another server build - the operator's
        `NARRATOR_HIGGS3_CHECKPOINT` is taken as their assertion, and with
        neither the answer is None, which every caller treats as "unknown",
        never as "fine".
        """
        for row in self._models_payload():
            if row.get('id') != SERVED_MODEL_NAME:
                continue
            root = row.get('root')
            if isinstance(root, str) and root.strip():
                return root.strip().rstrip('/') or root.strip()
            break
        asserted = (os.environ.get(CHECKPOINT_ENV) or '').strip()
        return asserted or None

    def check_serves_expected_model(self, checkpoint_dir: str = None) -> None:
        """Prove the server on this port is OURS before anything is sent to it.

        `/health` answering 200 says only that SOMETHING is listening. This
        checks `/v1/models` carries `higgs-v3`, and that the model the server
        was started on is the one this voice needs: the merged checkpoint for a
        fine-tune, the base snapshot for the base voice. Every Higgs voice is a
        whole merged checkpoint and one server serves exactly one of them, so a
        leftover server from ANOTHER voice answers `/health` and `/v1/models`
        identically and would render a whole book in the wrong narrator while
        every log line here named the right one - in either direction.
        """
        checkpoint_dir = checkpoint_dir or self.checkpoint_dir
        models = self.served_models()
        if SERVED_MODEL_NAME not in models:
            raise HiggsV3ServerError(
                f"Higgs v3: the server at {self.base_url} serves "
                f"{models or '(none)'}, not '{SERVED_MODEL_NAME}'. That is somebody "
                "else's server (or one left over from another model) on this port - "
                'refusing to render against it. Stop it, or point '
                'NARRATOR_HIGGS3_URL somewhere else.')
        running = self.running_checkpoint()
        if running is None:
            raise HiggsV3ServerError(
                f'Higgs v3: the server at {self.base_url} does not report which '
                f'model it was started on ({MODELS_PATH} carries no "root" for '
                f"'{SERVED_MODEL_NAME}'), and {CHECKPOINT_ENV} is not set. This "
                'voice is '
                + (f'the merged checkpoint {checkpoint_dir}' if checkpoint_dir
                   else 'the base weights')
                + ', and an unidentified server would render the whole book in '
                'whatever narrator it happens to hold. If this is your own server, '
                f'state its model directory in {CHECKPOINT_ENV}.')
        if checkpoint_dir:
            if os.path.normpath(running) != os.path.normpath(checkpoint_dir):
                # Both are known; they disagree, which is decidable and fatal.
                raise HiggsV3ServerError(
                    f'Higgs v3: the server at {self.base_url} is running on '
                    f'{running}, but this voice is {checkpoint_dir}. vllm-omni cannot '
                    'load a voice into a running server - it has no adapter flags and '
                    'its talker does not implement SupportsLoRA - so serving another '
                    'voice means RESTARTING on that checkpoint.')
            log(f'[HIGGS3] serving checkpoint {checkpoint_dir} (reported by the '
                'server)', flush=True)
        else:
            if BASE_SNAPSHOT_MARKER not in running:
                raise HiggsV3ServerError(
                    f'Higgs v3: this voice is the BASE weights, but the server at '
                    f'{self.base_url} is running on {running}, which is a merged '
                    f'fine-tune (no {BASE_SNAPSHOT_MARKER!r} in its path). Serving '
                    'the base means RESTARTING on the base snapshot.')
            log(f'[HIGGS3] serving the base snapshot {running} (reported by the '
                'server)', flush=True)

    #: The sentinel-filter tail measurement: RMS in dBFS over the last 300 ms
    #: of a one-word render.
    #:
    #: IT IS A SENSOR AND NOT A GATE, and that is a correction rather than a
    #: relaxation. Until 2026-09-05 this window was gated at -45 dBFS, derived
    #: from two points: our own smoke read -62.4 dBFS against a server carrying
    #: the retired `patch_tail_trim.py`, and the campaign's diagnosis put an
    #: UNPATCHED tail near -31 dBFS. `patch_sentinel_filter.py` invalidates the
    #: gate: it drops the sentinel frames entirely, so what the window now holds
    #: is THE MODEL'S OWN AUDIO, and the certifying box measured -35 to -38 dBFS
    #: here on BOTH builds. A -45 dB gate would fail a correct server, and no
    #: level distinguishes the two builds at all - the tail measurement cannot
    #: decide this question, whichever number is chosen.
    #:
    #: So the band below is RECORDED, LOGGED AND COMPARED, and nothing is
    #: refused on it. No threshold is invented to keep a gate alive.
    SENTINEL_TAIL_WINDOW_SECONDS = 0.3
    SENTINEL_TAIL_CERTIFIED_DBFS = (-38.0, -35.0)
    SENTINEL_PROBE_TEXT = 'Yes.'
    SENTINEL_PROBE_SEED = 4242

    def probe_sentinel_filter(self, voice=None) -> float:
        """Render one fixed-seed word and REPORT its tail level in dBFS.

        WHAT THIS NO LONGER DOES: prove `work/patch_sentinel_filter.py` is
        applied. It used to (as `probe_tail_trim`), because an unpatched server
        left ~240 ms of decoded sentinel garbage on the end of every chunk at
        about -30 dB and a quiet tail was therefore evidence. Under the sentinel
        filter the trailing frames are GONE rather than quiet, so the window
        holds ordinary speech decay: -35 to -38 dBFS on the certifying box, on
        the patched and the band-aided build alike (measured 2026-09-05). A
        level gate here would now fail correct servers and could never separate
        the two builds. Keeping it would have been a number defended for its own
        sake.

        WHAT WOULD PROVE THE PATCH, both halves measured by the fine-tuning
        session:

          (a) THE SERVER'S OWN LOG, READ. `verify_sentinel_filter` - DONE
              2026-09-05, and the reason `start()` no longer sends the
              launcher's streams to DEVNULL. It counts the trailing-ramp lines
              (every one must report exactly
              EXPECTED_TRAILING_SENTINEL_FRAMES = 2; any other count is a
              sentinel the trim did not reach) and refuses a single SYNC-path
              interior drop - a frame that failed the token test between two
              good ones, which offline classification of every saved talker
              matrix puts at zero on all real shapes and which the detector has
              never fired on. A missing or unreadable log is itself a refusal:
              the proof IS the stream.
          (b) NO TRIM CODE LEFT IN THE STAGE PROCESSOR. `[:, :-1]` occurs twice
              in the pristine `higgs_audio_v3.py` and zero times after the
              patch. This half IS enforced today, statically and before any
              server starts: BookForge's Higgs doctor greps the file in
              site-packages for `_filter_sentinel_frames` AND for the absence of
              `[:, :-1]`, and reports `trim-survived` for a half-applied or
              stacked file (electron/tool-paths.ts HIGGS_PATCHES,
              electron/scripts/higgs/install_higgs_env.sh).

        NOT A FAILURE: the async warning at higgs_audio_v3.py:403 ("frame(s)
        carry a stream sentinel outside the trailing run") is an INSTRUMENTATION
        BUG in the patch - the count is taken BEFORE the trailing-run trim, so it
        counts the normal 2-frame EOC ramp. Expect exactly "2 frame(s)" per
        chunk, in sequential renders as much as concurrent ones. COUNT them and
        report the count; do not read them as contamination.

        Cost: one ~1 s generation, once per server start. Returns the dBFS.
        """
        body = build_request_body(
            self.SENTINEL_PROBE_TEXT, voice,
            cap_frames(self.SENTINEL_PROBE_TEXT),
            seed=self.SENTINEL_PROBE_SEED)
        payload, ctype = self.post_speech(body, timeout=600,
                                          with_content_type=True)
        audio, rate = decode_response(payload, ctype)
        window = audio[-int(rate * self.SENTINEL_TAIL_WINDOW_SECONDS):]
        if window.size == 0:
            # The one thing this probe still REFUSES on, and it is decidable: a
            # server that answers 200 with no audio at all is not a level
            # question.
            raise HiggsV3ServerError(
                'Higgs v3 sentinel-filter probe: the probe render produced no audio.')
        rms = float(np.sqrt(np.mean(np.square(window.astype(np.float64)))))
        dbfs = 20.0 * float(np.log10(max(rms, 1e-12)))
        low, high = self.SENTINEL_TAIL_CERTIFIED_DBFS
        where = ('inside' if low <= dbfs <= high else 'OUTSIDE')
        log(f'[HIGGS3] sentinel-filter probe: {dbfs:.1f} dBFS over the last '
            f'{self.SENTINEL_TAIL_WINDOW_SECONDS * 1000:.0f} ms, {where} the '
            f'certified band {low:.0f}..{high:.0f} dBFS. REPORTED, NOT GATED - the '
            'tail level does not distinguish a filtered server from a band-aided '
            'one; the patch is proved by the doctor\'s marker/absent-marker grep '
            'of the stage processor.', flush=True)
        return dbfs

    # -- use -----------------------------------------------------------------

    def speak(self, request):
        """One `SpeechRequest` -> (float32 mono, sample rate).

        `request.voice` is a ClipsVoice; `request.sampling` empty means the
        server's own defaults. A non-200 raises with the server's own message.
        """
        body = build_request_body(request.text, request.voice,
                                  request.max_new_tokens, seed=request.seed,
                                  sampling=request.sampling)
        payload, content_type = self.post_speech(body, with_content_type=True)
        return decode_response(payload, content_type)

    def post_speech(self, body: dict, timeout: float = 1800,
                    with_content_type: bool = False):
        """POST `body` to /v1/audio/speech and return the raw response bytes
        (or `(bytes, content_type)` when asked)."""
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
            if 'Token id -100' in detail:
                # The one 400 whose text does not say what to do about it.
                raise HiggsV3ServerError(
                    f'Higgs v3 HTTP {exc.code}: {detail}\n'
                    'THE vLLM PATCH IS MISSING. vllm-omni builds every voice-clone '
                    'prompt with AUDIO_PLACEHOLDER_ID == -100, which the talker '
                    'substitutes with reference embeddings at prefill, but vLLM '
                    "0.28's blanket negative-id check rejects it first. Apply "
                    'work/patch_vllm.py in the higgs3 env and restart the server. '
                    'Without it EVERY cloned request fails this way and only an '
                    'un-cloned one succeeds - which renders in the model\'s own '
                    'voice, at 12 % of the narrator ceiling.') from exc
            raise HiggsV3ServerError(
                f'Higgs v3 HTTP {exc.code}: {detail}') from exc
        except urllib.error.URLError as exc:
            raise HiggsV3ServerDown(
                f'Higgs v3 server at {self.base_url} is unreachable: {exc.reason}. '
                'Is it started (serve_v3.sh; cold start measured 55-297 s) and are '
                'both site-packages patches applied?') from exc


#: A Windows path -> the path WSL sees. `served_common.to_wsl`, kept under this
#: name because both stacks' launch paths and every test that checks the UNC
#: forms reach for it here.
_to_wsl = served_common.to_wsl
