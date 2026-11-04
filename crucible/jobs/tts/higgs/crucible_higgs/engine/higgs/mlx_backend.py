"""Higgs TTS 3, IN-PROCESS on Apple Silicon, through mlx-audio.

Registry id `higgs-v3`, backend `mlx`, `BackendSpec.kind == 'inprocess'`. The
same engine as `v3_engine.HiggsV3Engine`, on the other side of a platform
split: on Windows/Linux v3 runs in a vllm-omni SERVER over HTTP (there is no HF
modeling class and its torch cannot share an env with Orpheus's vLLM); on the
Mac `mlx_audio.tts.models.higgs_audio_v3` implements the whole model natively,
so the weights load into THIS process and there is nothing to launch, health-
check or kill.

Owen, 2026-09-05: "make sure the Mac has Higgs built in for streaming the model
via the browser extension. I use that constantly on the Mac."

WHAT THIS MODULE OWNS AND WHAT IT BORROWS

  borrowed   the model object, its weights, the tokenizer, the prompt builder,
             the codec, and the per-step sampler
             (`higgs_audio_v3.generation.step`) - all of mlx-audio 0.4.8.
  owned      the generation LOOP, the decode, and every number narrator states.

The loop is narrator's rather than `Model.generate`'s for three measured
reasons, each of which is a contract narrator cannot express through that call:

  1. `Model.generate` applies `fade_in_ms=30, fade_out_ms=15` to the waveform
     BY DEFAULT. narrator's fade is the ASSEMBLER's (`edge_fade`,
     EdgeFade(10, 25)) and no engine may bake one in - a chunk file that is
     already faded is faded twice.
  2. `Model.generate` has no `should_stop`. It is a generator that yields ONCE,
     at the end, so a cancel could only land between rows - and one v3 chunk is
     up to 600 characters, tens of seconds of generation.
  3. `Model._decode_audio` hands `reverse_delay_pattern`'s output straight to
     the codec with no check that every code is a real code. See THE DECODE.

THE DECODE, AT THE TOKEN LEVEL
------------------------------
v3's audio vocabulary is 8 codebooks x 1024 real codes plus two STREAM
SENTINELS: 1024 (BOC, begin-of-codes) and 1025 (EOC, end-of-codes). The codec's
codebooks hold exactly 1024 entries (`VectorQuantizer(codebook_size=1024)`), so
1024 and 1025 are OUT OF RANGE for it. mlx's `nn.Embedding` does not bounds-
check a gather, so a sentinel reaching the codec is not an error - it is
whatever memory that index lands on, decoded as sound.

The delay pattern offsets codebook c by c rows: audio frame t of codebook c
sits at stream row t + c. Reverting takes the diagonal,
`raw[t, c] = delayed[t + c, c]`, and consumes Q - 1 = 7 rows.

WHAT THAT MEANS FOR THE THREE ENDINGS mlx-audio's sampler can produce:

  ramp-up      rows 0..6 carry FORCED BOC in every codebook above the diagonal
               (`generation.step`, the `delay_count < n` branch). Those sit
               strictly ABOVE the diagonal and the revert reads only ON and
               BELOW it, so no BOC can reach the codec from the head. Measured
               below, not assumed.
  clean end    codebook 0 emits EOC at row e; the sampler then runs n - 2 = 6
               more rows and stops, so L = e + 7 and T = L - 7 = e. The EOC
               diagonal sits at `delayed[e + c, c]`, which is raw frame t = e -
               one past the last frame the revert produces. **On a clean ending
               the revert is EXACT and nothing needs trimming.** This is why
               the "every chunk ends in garbage" story was wrong, and why a
               blind trailing trim is a band-aid: on the shape it was written
               for it removes real audio.
  ragged end   the ramp-down is SAMPLED, not forced (unlike the ramp-up).
               Nothing makes the model put EOC exactly on the diagonal, and
               nothing stops it emitting a sentinel mid-stream. Any such code
               lands INSIDE a raw frame the revert keeps, and goes to the codec.

So the fix is neither a trim nor a substitution. It is a filter by TOKEN
IDENTITY, applied after the revert: **a frame is kept iff all 8 of its
codebooks are in [0, 1023]**. Nothing out of range is handed to the codec at
all. This is the same root fix the training side landed for vllm-omni
(work/patch_sentinel_filter.py), whose upstream substituted 0 for every
sentinel - and 0 is a VALID code that decodes to real sound, so the
substitution was the defect, not the cure.

`real_code_frames` reports what it dropped. A leading or trailing drop is a
ragged ending; an INTERIOR drop is not an expected shape and is logged, because
a gate is a defect sensor, not a silent repair. No fade is added on top: a cut
in the code domain lands on a frame boundary the codec never rendered, so there
is nothing to click. The 10/25 ms `edge_fade` narrator declares is the
ASSEMBLER's join contract and is unrelated.

`revert_delay_pattern` and `real_code_frames` are PURE NUMPY and import no mlx,
so they are unit-testable on the Windows interpreter and are pinned against
saved token matrices from the vllm-omni investigation
(`tests/golden/higgs_sentinel/`).

STREAMING
---------
`generate_batch_stream` emits WHOLE ROWS at retirement, and says so. That is not
a shortcut: a delay-pattern codec's window is incomplete in its last 7 frames by
construction (codebook c of frame t has not been emitted until step t + c), and
the ragged-ending filter above can only run once generation has finished, so a
mid-row window cannot tell a ramp-down from speech. `codec().streaming_decoder()`
returns None for the same reason. `should_stop` is checked EVERY generation
step, so a cancel lands in milliseconds even though audio does not.

That is also why the Listen path's batch is a LADDER rather than one batch: the
row being listened to renders SOLO (its time-to-first-audio is its whole
generation, so width would be added straight onto the listener's wait), and only
the read-ahead behind it is batched - each of those rows decoded and delivered
the moment it retires (`_generate_delayed_rows_batch(on_retire=...)`) instead of
when its group finishes. See `generate_batch_stream`.

NO MLX AT IMPORT. Every mlx / mlx_lm / mlx_audio import is inside a function,
exactly as `engine/orpheus/mlx_backend.py` does it, so this module imports on a
machine with no MLX at all (`tests/test_engine_lazy_imports.py`).
"""
import os
import time as _time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...env import env_number
from ..log import log
from ..protocol import EMPTY_SENTENCE_SILENCE_SEC, BackendSpec, ClipsVoice, DefaultVoice, StopPolicy
from . import served_common, truncation, v3_served
from .codec import FrameMeasure
from .prompt import clean_text
from .stall_guard import RowStall, counted, stall_guard_from_env
from .v3_engine import HiggsV3Defaults, apply_v3_voice_defaults

#: Codebooks per frame, and the size of one codebook. The two stream sentinels
#: live immediately above the real codes.
NUM_CODEBOOKS = 8
NUM_REAL_CODES = 1024
AUDIO_BOC_ID = 1024      # begin-of-codes, forced by the sampler's ramp-up
AUDIO_EOC_ID = 1025      # end-of-codes, SAMPLED on the ramp-down

#: Where the base weights are. NO DEFAULT AND NO SEARCH: an unset variable is a
#: refusal naming it, because guessing a model directory is how a render ends up
#: in a different model's voice. The Mac's copy lives under
#: `~/Library/Application Support/BookForge/runtime/higgs-models/base`.
MODEL_ENV = 'NARRATOR_HIGGS3_MLX_MODEL'

#: The exact mlx-audio release this backend is written against. It is PINNED,
#: not floored: 0.5.1 cannot render Orpheus at all (pyproject `orpheus-mlx`),
#: and the private members used below are not a public API.
MLX_AUDIO_VERSION = '0.4.8'

#: The mlx-audio model package narrator loads, named EXPLICITLY rather than
#: sniffed out of `config.json` + the model path. See `load_engine`.
MLX_AUDIO_ARCH = 'higgs_audio_v3'

#: THE BATCH CEILING. Unset means 1 - one row at a time, byte for byte the
#: behaviour that shipped - so nothing widens because a machine happened to have
#: the memory. BookForge asks for a width explicitly (electron/higgs-spawn.ts,
#: the darwin WORKER arm only), taken from the same Orpheus memory tier that
#: sizes the Orpheus MLX batch.
BATCH_ENV = 'NARRATOR_HIGGS3_MLX_BATCH'

#: Total unified memory ONE batch may occupy, weights and pinned buffer cache
#: included. `_mlx_width_for_depth` narrows a deep batch to stay inside it. 42 is
#: what BookForge's `extreme` tier hands Orpheus today.
MEM_BUDGET_ENV = 'NARRATOR_HIGGS3_MLX_MEM_BUDGET_GB'

#: The buffer-cache cap this backend already sets at load (`load_engine`). It is
#: READ AGAIN by the headroom math because a pinned cache is a known constant
#: that KV cannot have - one variable, two readers, never a second name.
CACHE_LIMIT_ENV = 'HIGGS_MLX_CACHE_LIMIT_GB'
CACHE_LIMIT_DEFAULT_GB = 8.0

#: THE FRACTION NAME, shared with the CUDA stack on purpose - the same variable
#: Crucible sets per voice in the spawn environment, and the same one
#: `sgl_served` exports into its launcher. Here it means a fraction of the
#: machine's UNIFIED memory, and `mlx_mem_budget_gb` turns it into the absolute
#: GB this backend's headroom math has always worked in. The pinned buffer
#: cache (`CACHE_LIMIT_ENV`) does NOT follow it: it is a cap on a cache that
#: lives inside the budget, already subtracted by `_mlx_kv_headroom_gb`, and
#: scaling both would take it off twice.
SERVE_MEM_FRACTION_ENV = 'HIGGS_SGL_MEM_FRACTION'


# `_env_number` USED TO BE DEFINED HERE. It moved to `narrator/env.py` on
# 2026-09-13, unchanged, because it states a POLICY - garbage is refused by
# name, never coerced, never defaulted past - and `serve/worker.py` was breaking
# that policy in four places while this module enforced it in three. A rule with
# one implementation inside one backend is a rule the rest of the tree cannot
# obey (ARCHITECTURE.md R1). No alias is left behind: nothing outside this
# module ever imported the private name, and a shim kept "just in case" is a
# second name for the one thing this move existed to stop having two of.


def mlx_batch_ceiling() -> int:
    """`BATCH_SIZE`: the widest batch this process will ever generate at once."""
    return int(env_number(BATCH_ENV, 1, int, 1,
                          'the widest Higgs MLX batch, in rows'))


def mlx_device_memory_gb() -> float:
    """The machine's PHYSICAL unified memory, in GB, as Metal reports it.

    `mx.metal.device_info()['memory_size']` is bytes of device memory - on
    Apple silicon that is the unified pool the whole machine shares, which is
    what a FRACTION has to be a fraction of. Imported here rather than at module
    scope: this module is imported on Windows (where there is no mlx) by tests
    and by the registry, and only a caller that asked for a fraction needs an
    answer.

    NO GUESS. A machine whose Metal cannot say is a machine narrator cannot turn
    a fraction into gigabytes on, and it says so rather than assuming 64.
    """
    try:
        import mlx.core as mx
        info = mx.metal.device_info()
        size = info['memory_size']
    except Exception as exc:                        # noqa: BLE001 - reported whole
        raise ValueError(
            f'{SERVE_MEM_FRACTION_ENV} was set, but narrator cannot read this '
            f"device's memory size from Metal ({type(exc).__name__}: {exc}), so "
            'it has nothing to take a fraction OF. Set '
            f'{MEM_BUDGET_ENV} in gigabytes instead.') from exc
    return float(size) / 1e9


def mlx_mem_budget_gb() -> float:
    """`MLX_MEM_BUDGET_GB`: the whole batch's unified-memory budget.

    TWO WAYS TO SAY IT, AND THEY MAY NOT DISAGREE (Owen, 2026-09-19).
    `HIGGS_SGL_MEM_FRACTION` is the fraction-of-the-card name the CUDA stack
    already takes, and Crucible sets it per voice in the spawn environment for
    every backend; the Mac is 64 GB unified, so the same fraction is a real
    number here too - `fraction x mlx_device_memory_gb()`. `MEM_BUDGET_ENV` is
    the absolute-GB name this backend has always had.

    A value set BOTH ways that does not agree is refused by name rather than
    letting one silently win: the loser would be invisible - the batch simply
    runs narrower or wider than the operator believes, and the only symptom is
    a number in a log nobody compares.
    """
    absolute = (os.environ.get(MEM_BUDGET_ENV) or '').strip()
    fraction = (os.environ.get(SERVE_MEM_FRACTION_ENV) or '').strip()
    if not fraction:
        return float(env_number(MEM_BUDGET_ENV, 42.0, float, 1.0,
                                'the Higgs MLX batch memory budget, in GB'))
    share = float(env_number(SERVE_MEM_FRACTION_ENV, 0.0, float, 0.0,
                             "the Higgs MLX batch's share of unified memory"))
    if not (0.0 < share < 1.0):
        raise ValueError(
            f'{SERVE_MEM_FRACTION_ENV}={fraction!r} is out of range: it is a '
            'FRACTION of the device, strictly between 0 and 1.')
    derived = share * mlx_device_memory_gb()
    if absolute:
        stated = float(env_number(MEM_BUDGET_ENV, 42.0, float, 1.0,
                                  'the Higgs MLX batch memory budget, in GB'))
        # A tenth of a gigabyte: the two are the same statement rounded
        # differently, not two different budgets.
        if abs(stated - derived) > 0.1:
            raise ValueError(
                f'{SERVE_MEM_FRACTION_ENV}={share:g} of this device\'s '
                f'{mlx_device_memory_gb():.1f} GB is {derived:.1f} GB, but '
                f'{MEM_BUDGET_ENV}={stated:g} GB says otherwise. They are two '
                'ways of stating ONE budget and narrator will not pick between '
                'them - the loser is invisible, and the batch would run at a '
                'width nobody chose. Set one, or make them agree.')
    return derived


def mlx_cache_limit_gb() -> float:
    """The pinned MLX buffer cache, in GB. The SAME variable `load_engine` sets
    `mx.set_cache_limit` from - read here so the headroom math subtracts the
    memory that was actually pinned rather than a number of its own."""
    return float(env_number(CACHE_LIMIT_ENV, CACHE_LIMIT_DEFAULT_GB, float, 0.0,
                            'the pinned MLX buffer cache, in GB'))


def _lower_cb0(logits, position, codes: list, penalty: float):
    """Lower codebook 0's logit for each of `codes` by `penalty` (the stall
    guard's one write). `position` is the row's place in a batched (B, n, V)
    logits array, or None for one row's (n, V). A scatter into at most a
    window's worth of entries - never a full-codebook bias."""
    import mlx.core as mx
    idx = mx.array(codes, dtype=mx.int32)
    if position is None:
        return logits.at[0, idx].add(-float(penalty))
    return logits.at[int(position), 0, idx].add(-float(penalty))


def _observe_stall(stall: RowStall, codes, steady: bool, row) -> None:
    """Record one row's sampled cb0 code. The code is read only on a steady
    step - the sampler has already evaluated it there (`step` reads cb0 to watch
    for EOC) - and a guard that has just engaged says so once."""
    was = stall.penalty() > 0
    stall.observe(int(codes[0].item()) if steady else -1, steady)
    if not was and stall.penalty() > 0:
        _log(f'Higgs v3 MLX stall guard ENGAGED on row {row}: codebook 0 has repeated '
             f'{stall.run} frames (window of {len(stall.ring)} codes); lowering it.')


def _observe_stall_batch(stalls, active, sampled_rows, steady) -> None:
    """`_observe_stall` for a batch step: every steady row's cb0 is read in ONE
    evaluation, not one sync per row."""
    import mlx.core as mx
    positions = [p for p, s in enumerate(steady) if s]
    values = {}
    if positions:
        stacked = np.asarray(mx.stack([sampled_rows[p][0] for p in positions]))
        values = {p: int(v) for p, v in zip(positions, stacked.tolist())}
    for position, row in enumerate(active):
        was = stalls[row].penalty() > 0
        stalls[row].observe(values.get(position, -1), steady[position])
        if not was and stalls[row].penalty() > 0:
            _log(f'Higgs v3 MLX stall guard ENGAGED on row {row}: codebook 0 has repeated '
                 f'{stalls[row].run} frames (window of {len(stalls[row].ring)} codes); '
                 'lowering it.')


def _log(message: str) -> None:
    """One ASCII log line, to the HOST's log stream.

    NOT stdout-by-default: `narrator.serve`'s stdout IS the JSON-lines protocol,
    and a bare `print` from the engine layer lands between two protocol messages
    and breaks the client's parse. Found the first time this backend was driven
    through the real worker (2026-09-05): the load banner arrived where a
    `loaded` message was expected.

    Routed through `narrator.engine.log` so there is ONE mechanism for the whole
    engine layer rather than two. The default is stderr; the only host that
    redirects it to stdout is `narrator.compat.worker`, whose stdout
    parallel-tts-bridge.ts parses - and all but ONE of these strings are inert
    there (checked: "loading Higgs v3 from ..." has no "model" token for
    MODEL_LOAD_START_RE, and "model loaded in 2.8s" lacks the "!"
    MODEL_LOAD_DONE_RE requires).

    THE ONE EXCEPTION IS LOAD-BEARING ON PURPOSE: the batch heartbeat,
    "MLX batch generating: <N> rows, ~<T> tokens (step <S>/<D>), <R>/<N> rows
    done, batch <G>/<C>", is read by `parallel-tts-bridge.ts`'s
    GENERATION_ACTIVITY_RE (liveness - a batch is otherwise silent for minutes
    and the watchdog would kill the worker) and parsed field by field by
    `mlx-batch-progress.ts` (the within-batch progress bar). NEITHER anchors the
    engine prefix, so `[HIGGS-MLX]` is read exactly as `[ORPHEUS]` is; the
    wording and the field ORDER are the contract.
    """
    log(f'[HIGGS-MLX] {message}')


# ---------------------------------------------------------------------------
# The decode, as pure numpy (no mlx; unit-tested on Windows)
# ---------------------------------------------------------------------------


class HiggsMlxStreamMisaligned(ValueError):
    """The delayed-row matrix cannot be turned into audio frames.

    Named for `engine.orpheus.errors.TokenStreamMisaligned` and for the same
    reason: a stream that cannot be interpreted is audio nobody can trust, so
    the row fails loudly instead of decoding whatever happens to be there.
    """


def revert_delay_pattern(delayed_lq) -> np.ndarray:
    """(L, 8) delayed stream rows -> (L - 7, 8) audio frames.

    `raw[t, c] = delayed[t + c, c]`: codebook c is delayed by c rows, so
    reverting takes the diagonal and consumes Q - 1 = 7 rows. Identical in
    content to `mlx_audio...higgs_audio_v3.generation.reverse_delay_pattern`
    (which returns the same thing as an mx.array); narrator keeps its own so the
    arithmetic is testable without mlx and so the FILTER below can run on the
    result before anything reaches the codec.

    A generation shorter than the diagonal carries no audio frames at all and
    returns an empty (0, 8) - it is not an error, it is a row that stopped
    before it said anything.
    """
    matrix = np.asarray(delayed_lq)
    if matrix.ndim != 2 or matrix.shape[1] != NUM_CODEBOOKS:
        raise HiggsMlxStreamMisaligned(
            f'Higgs v3 delayed rows must be (steps, {NUM_CODEBOOKS}); got shape '
            f'{tuple(matrix.shape)}')
    matrix = matrix.astype(np.int64, copy=False)
    keep = int(matrix.shape[0]) - (NUM_CODEBOOKS - 1)
    if keep <= 0:
        return np.zeros((0, NUM_CODEBOOKS), dtype=np.int64)
    return np.stack([matrix[c:c + keep, c] for c in range(NUM_CODEBOOKS)],
                    axis=1)


@dataclass(frozen=True)
class FrameFilterReport:
    """What `real_code_frames` dropped, and from where.

    `interior` is the one that matters: leading and trailing drops are a ragged
    ending, which the sampler can legitimately produce, while an interior drop
    means a sentinel appeared in the middle of speech and something is wrong
    upstream of the decode.
    """
    total: int
    kept: int
    leading: int
    interior: int
    trailing: int

    @property
    def dropped(self) -> int:
        return self.total - self.kept

    def as_log(self) -> str:
        return (f'frames {self.kept}/{self.total} kept '
                f'(lead {self.leading}, interior {self.interior}, '
                f'tail {self.trailing})')


def real_code_frames(raw_tq):
    """Keep only frames whose 8 codebooks are ALL real codes. -> (frames, report)

    THE ROOT FIX, decided by TOKEN IDENTITY and never by position or by audio
    content. Out-of-range values are the stream sentinels BOC=1024 / EOC=1025
    (and any negative pad). They are NOT substituted with 0: 0 is a valid
    codebook entry that decodes to real sound, which is exactly how the
    equivalent upstream code turned a ramp-down into an audible burst.

    Returns the kept frames and a `FrameFilterReport`. Nothing is trimmed on a
    clean ending, because on a clean ending the delay-pattern revert already
    ends one frame before the EOC diagonal and every frame is real.
    """
    frames = np.asarray(raw_tq)
    if frames.ndim != 2 or frames.shape[1] != NUM_CODEBOOKS:
        raise HiggsMlxStreamMisaligned(
            f'Higgs v3 audio frames must be (frames, {NUM_CODEBOOKS}); got shape '
            f'{tuple(frames.shape)}')
    total = int(frames.shape[0])
    if total == 0:
        return frames, FrameFilterReport(0, 0, 0, 0, 0)
    valid = ((frames >= 0) & (frames < NUM_REAL_CODES)).all(axis=1)
    kept_idx = np.nonzero(valid)[0]
    kept = int(kept_idx.size)
    if kept == 0:
        return (frames[:0], FrameFilterReport(total, 0, total, 0, 0))
    lo, hi = int(kept_idx[0]), int(kept_idx[-1])
    leading = lo
    trailing = total - 1 - hi
    interior = int((~valid[lo:hi + 1]).sum())
    return frames[valid], FrameFilterReport(total, kept, leading, interior,
                                            trailing)


def _report_frames(report: FrameFilterReport, label: str) -> None:
    """ASCII log lines, and a LOUD one for the shape that should not happen."""
    if report.interior:
        _log(f'{label}: {report.interior} INTERIOR sentinel frame(s) dropped - '
             f'this is not an expected shape ({report.as_log()})')
    if report.kept == 0 and report.total:
        _log(f'{label}: every frame carried a stream sentinel; this row '
             'produced no audio')


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def model_dir_from_env() -> str:
    """The base weights directory, or a refusal naming the variable.

    NOT a search and NOT a default: an engine that guesses where its weights are
    is one that can render a whole book in the wrong model and report success.
    """
    path = (os.environ.get(MODEL_ENV) or '').strip()
    if not path:
        raise ValueError(
            f'{MODEL_ENV} is not set. The Higgs v3 MLX backend loads the merged '
            f'weights from a directory (or an HF repo id); there is no default and '
            f'no search. Point {MODEL_ENV} at the base checkpoint - on this Mac '
            f'that is "$HOME/Library/Application Support/BookForge/runtime/'
            f'higgs-models/base" ({v3_served.MODEL_ID}, ~8.5 GB).')
    return path


@dataclass
class HiggsV3MlxConfig:
    """What one HiggsV3MlxEngine is built from.

    `model_dir` is the weights the model object is loaded FROM. For a
    `checkpoint` voice that is the voice's own merged directory - its weights
    ARE the voice, exactly as on the served arm, and the only difference is that
    here loading it is a `load_model` call rather than a server restart. For a
    `clips` or `default` voice it is the base checkpoint - the directory the
    voice names in `base_dir` when it names one (whoever writes the voices
    document pins the base weights, so a clone renders on known bytes), else
    `MODEL_ENV`.

    `sampling` EMPTY means THE MODEL DIRECTORY'S OWN SAMPLING - see
    `mlx_sampling`, which reads it from the checkpoint's
    `generation_config.json` rather than from a constant here. The remaining
    keys of `v3_served.SERVER_DEFAULT_SAMPLING` are the SERVER's
    (`repetition_penalty`, `seed`) and have no counterpart in mlx-audio's
    sampler; passing one is refused by name rather than silently ignored.
    """
    voice: object
    model_dir: str
    sampling: Optional[dict] = None
    seed: Optional[int] = 1234
    sentences_dir: Optional[str] = None
    process_dir: Optional[str] = None
    audio_format: str = 'flac'
    max_chars: int = HiggsV3Defaults.MAX_CHARS
    max_chars_per_sec: float = HiggsV3Defaults.MAX_CHARS_PER_SEC
    min_chars_per_sec: float = HiggsV3Defaults.MIN_CHARS_PER_SEC
    context_tokens: int = HiggsV3Defaults.CONTEXT_TOKENS

    #: The sampling keys mlx-audio's `step()` actually takes.
    MLX_SAMPLING_KEYS = ('temperature', 'top_p', 'top_k')

    def __post_init__(self):
        if not isinstance(self.voice, (ClipsVoice, DefaultVoice)):
            raise ValueError(
                'HiggsV3MlxConfig(voice=...) takes a ClipsVoice (reference clips '
                "with their book-exact transcripts) or a DefaultVoice (a fine-tuned "
                'checkpoint, or the model\'s own voice). Got '
                f'{type(self.voice).__name__}.')
        if not (self.voice.name or '').strip():
            raise ValueError('HiggsV3MlxConfig needs a NAMED voice.')
        self.voice = apply_v3_voice_defaults(self.voice)
        if isinstance(self.voice, ClipsVoice):
            v3_served.check_reference_budget(self.voice)
        if not (self.model_dir or '').strip():
            raise ValueError(
                f"HiggsV3MlxConfig for voice '{self.voice.name}' has no model_dir. "
                'The MLX backend loads weights from a directory; see '
                f'{MODEL_ENV}.')
        unknown = sorted(set(self.sampling or {}) - set(self.MLX_SAMPLING_KEYS))
        if unknown:
            raise ValueError(
                f'Higgs v3 MLX sampling carried {unknown}. mlx-audio\'s sampler takes '
                f"only {list(self.MLX_SAMPLING_KEYS)}; the rest of the SERVED "
                "defaults ('repetition_penalty', 'seed') are vllm-omni's and have no "
                'counterpart here. Refusing a lever that would look applied and do '
                'nothing.')
        # VALIDATE NOW, at construction - the same moment `HiggsV3Config` does.
        # Both real doors already validate before building a config, so this
        # closes an asymmetry rather than a live hole: a caller that constructs
        # the dataclass directly should not get an object whose first refusal
        # arrives from inside the generation loop.
        self.mlx_sampling()

    def mlx_sampling(self) -> dict:
        """The three levers actually handed to `step()` - FROM THE MODEL DIR.

        **mlx-audio DOES NOT READ `generation_config.json`.** Measured against
        mlx-audio 0.4.8 on the Mac, 2026-09-05: `grep -rn generation_config` over
        `tts/models/higgs_audio_v3/` returns NOTHING, while three other model
        packages in the same release do read it (`moss_tts.py:349`,
        `qwen3_tts.py:2914`, `stt/models/whisper/whisper.py:717`), so its absence
        here is a gap and not a convention. `Model.generate`
        (`higgs_audio_v3/model.py:751-753`) defaults `temperature=1.0`,
        `top_p=None`, `top_k=None`, and `generation.step`
        (`generation.py:114-141`) takes all three as required keyword arguments -
        `_apply_top_k`/`_apply_top_p` (`generation.py:62-77`) are no-ops for
        None. So on this runtime the CALLER is the only thing that can apply the
        checkpoint's sampling, and passing nothing samples the untruncated
        codebook tail exactly as an unconfigured vllm-omni does.

        So narrator reads the file itself, and the FILE is the authority:

          checkpoint voice   `<checkpointDir>/generation_config.json`, which
                             `v3_served.require_generation_config` has already
                             proved is there and carries all three keys. That is
                             the same file the served arm's vllm-omni reads, so
                             both arms render one voice at one sampling.
          base weights       (a `default` voice, or a zero-shot clone -
                             including one that NAMES its base weights in
                             `base_dir`, since 2026-09-15) there is NO file to
                             read: the
                             `bosonai/higgs-audio-v3-tts-4b` snapshot ships none
                             (verified 2026-09-05 on the WSL HF cache AND on the
                             Mac's `runtime/higgs-models/base`), which is the
                             whole reason a MERGED dir has to carry one. The
                             authority for base weights is therefore v3's
                             documented deploy default,
                             `v3_served.SERVER_DEFAULT_SAMPLING` - stated here,
                             not guessed. The SERVED arm now states the same
                             values for the same weights, explicitly, in
                             `extra_params` (`HiggsV3Config.served_sampling`),
                             so base weights render alike on both arms.

        `sampling` is a named per-config override on top and stays what it was.

        A `repetition_penalty` OTHER THAN 1.0 in the file is REFUSED, not
        dropped. mlx-audio's `higgs_audio_v3` has no repetition penalty at all -
        `Model.generate` takes no such argument and the word appears nowhere in
        the package (PORT_NOTES 13.11) - so a checkpoint whose file asks for one
        would render on the Mac with no penalty while the SAME directory on the
        served arm applies it: one voice, two samplings, silently. 1.0 is a
        no-op and is accepted as the nothing it is. This is the same rule
        `__post_init__` applies to a user-supplied `repetition_penalty`; a lever
        the runtime cannot honour is a refusal at either door.
        """
        # `checkpoint_dir`, the MERGE - never `base_dir`. A zero-shot voice
        # may name its base weights (pinned, so the clone renders on known
        # bytes) and that directory has no generation_config.json in it; its
        # authority is SERVER_DEFAULT_SAMPLING below, exactly as for a voice
        # that names no directory at all (2026-09-15).
        if self.voice.checkpoint_dir:
            document = v3_served.require_generation_config(
                self.voice.checkpoint_dir, self.voice.name)
            # Absence is a real state - the key is optional in the file and an
            # absent penalty is no penalty - so it is tested for, not defaulted.
            penalty = document['repetition_penalty'] if (
                'repetition_penalty' in document) else 1.0
            if float(penalty) != 1.0:
                raise ValueError(
                    f"Higgs v3 voice '{self.voice.name}': its "
                    f'{v3_served.GENERATION_CONFIG_FILE} in '
                    f'{self.voice.checkpoint_dir} asks for repetition_penalty '
                    f'{penalty}, and mlx-audio has NO repetition penalty - '
                    'higgs_audio_v3.Model.generate takes no such argument and the '
                    'lever does not exist anywhere in the package. Rendering here '
                    'anyway would give this checkpoint one sampling on the Mac and '
                    'another on the vllm-omni server, from the same file. Render '
                    'this voice on the served arm, or re-merge it with a '
                    'repetition_penalty of 1.0.')
        else:
            document = v3_served.SERVER_DEFAULT_SAMPLING
        resolved = {k: document[k] for k in self.MLX_SAMPLING_KEYS}
        resolved.update(self.sampling or {})
        return {'temperature': float(resolved['temperature']),
                'top_p': float(resolved['top_p']),
                'top_k': int(resolved['top_k'])}


# ---------------------------------------------------------------------------
# Codec and budget
# ---------------------------------------------------------------------------


class HiggsV3MlxCodec:
    """`protocol.Codec` for v3 on MLX - and unlike the served arm's codec, this
    one really decodes: the tokens ARE in this process.

    Geometry is v3's, unchanged: 8 codebooks at 25 fps, 24 kHz, 960 samples per
    frame, a 7-frame delay diagonal.
    """

    sample_rate = v3_served.SAMPLE_RATE
    frames_per_second = v3_served.FRAMES_PER_SECOND
    tokens_per_frame = NUM_CODEBOOKS
    samples_per_frame = int(v3_served.SAMPLE_RATE / v3_served.FRAMES_PER_SECOND)
    trim_frames = NUM_CODEBOOKS - 1

    def __init__(self, audio_decoder, label: str = ''):
        if not callable(audio_decoder):
            raise ValueError(
                'HiggsV3MlxCodec(audio_decoder) needs a callable taking a '
                '(frames, 8) int matrix and returning a float32 waveform '
                f'(the mlx-audio HiggsAudioTokenizer.decode); got '
                f'{type(audio_decoder).__name__}.')
        self._audio_decoder = audio_decoder
        self._label = label

    def frames_for_tokens(self, n_tokens: int) -> int:
        return int(n_tokens) // self.tokens_per_frame

    def audio_frames(self, generated_frames: int) -> int:
        """`generated_frames - 7`: reverting the delay pattern costs the
        diagonal. The sentinel filter may take MORE than this on a ragged
        ending, so it is a ceiling, not a promise."""
        return max(0, int(generated_frames) - self.trim_frames)

    def samples_for_frames(self, n_frames: int) -> int:
        return int(n_frames) * self.samples_per_frame

    def seconds_for_frames(self, n_frames: float) -> float:
        return float(n_frames) / self.frames_per_second

    def decode(self, tokens):
        """The delayed rows one generation produced -> 1-D float32 at 24 kHz.

        `tokens` is the (steps, 8) matrix of what the sampler emitted, exactly
        as generated - references and ramp-up included. Revert, filter by token
        identity, decode. Nothing else: no trim, no substitution, no fade.
        """
        raw, report = real_code_frames(revert_delay_pattern(tokens))
        _report_frames(report, self._label or 'decode')
        if raw.shape[0] == 0:
            raise HiggsMlxStreamMisaligned(
                'Higgs v3 MLX decode: no real-code frames survived the delay-pattern '
                'revert and the sentinel filter, so the row produced no speech. That '
                'is a failed render, not a silent one.')
        audio = self._audio_decoder(raw)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        expected = self.samples_for_frames(int(raw.shape[0]))
        if audio.size != expected:
            raise HiggsMlxStreamMisaligned(
                f'Higgs v3 MLX decode: {raw.shape[0]} frames should decode to '
                f'{expected} samples at {self.sample_rate} Hz, got {audio.size}. The '
                'codec and this arithmetic disagree - refusing to ship audio of an '
                'unknown length.')
        return audio

    def streaming_decoder(self, decode_frames, label: str = ''):
        """None. A delay-pattern codec's window is incomplete in its last 7
        frames by construction, and the sentinel filter can only run once
        generation has finished, so a mid-row window cannot tell a ramp-down
        from speech. Rows are emitted at retirement instead; faking a stream out
        of independent decodes would ship audio the listener has already heard
        by the time anything could notice it was wrong."""
        return None


class HiggsV3MlxBudget:
    """`protocol.Budget` for v3 on MLX. Same numbers as the served arm - they
    are properties of the MODEL, not of the runtime."""

    def __init__(self, config: HiggsV3MlxConfig):
        self._config = config

    def _voice(self, voice):
        if voice is not None and voice != self._config.voice.name:
            raise ValueError(
                f"HiggsV3MlxBudget: this engine serves '{self._config.voice.name}', "
                f"not '{voice}'. One loaded model answers for one voice.")
        return self._config.voice

    def max_chars(self, voice=None) -> int:
        """THE VOICE's chunk size, or the engine placeholder when the document
        declared none; `max_chars_source` says which answered.

        A FINE-TUNE WITH NO `maxChars` IS NO LONGER REFUSED (Owen, 2026-09-19)
        - see `HiggsV3Budget.max_chars` for the whole reasoning. In short:
        `maxChars` is the CLIENT's packing size, the screening render of a new
        checkpoint is what MEASURES it, and nothing narrator does per chunk
        reads it (the frame ceiling is `cap_frames(text)` over the text it was
        actually sent).
        """
        ref = self._voice(voice)
        if ref.max_chars is None:
            return int(self._config.max_chars)
        return int(ref.max_chars)

    def max_chars_source(self, voice=None) -> str:
        return self._voice(voice).max_chars_source or 'placeholder'

    def max_chars_per_sec(self, voice=None) -> float:
        return float(self._config.max_chars_per_sec)

    def max_total_tokens(self, prompt_tokens: int, voice=None) -> int:
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
        return v3_served.cap_frames(text)


def higgs_v3_mlx_stop_policy(config: HiggsV3MlxConfig) -> StopPolicy:
    """v3 stops on its own; what it does instead is DROP TEXT, so
    `coverage_check` is 'asr'. Identical to the served arm's policy but for
    `levers`, which reports the three sampling values this runtime can actually
    apply rather than the server's five."""
    return StopPolicy(
        max_new_tokens=v3_served.cap_frames('x' * int(config.max_chars)),
        eos_reliable=True,
        resplit_on_cap=False,
        max_chars_per_sec=float(config.max_chars_per_sec),
        min_chars_per_sec=float(config.min_chars_per_sec),
        levers={k: float(v) for k, v in config.mlx_sampling().items()},
        coverage_check='asr',
    )


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


class HiggsV3MlxEngine:
    """One Higgs v3 model, loaded into this process by MLX."""

    ENGINE_ID = HiggsV3Defaults.ENGINE_ID          # 'higgs-v3'
    SAMPLE_RATE = HiggsV3Defaults.SAMPLE_RATE      # 24000
    pads = False
    edge_fade = HiggsV3Defaults.EDGE_FADE          # EdgeFade(10, 25)
    SUPPORTS_BATCH = True
    #: THE CEILING, not the width. 1 unless BookForge asks for more
    #: (`NARRATOR_HIGGS3_MLX_BATCH`), so an unconfigured process renders exactly
    #: as it did single-row; the instance overrides this in `__init__`.
    #:
    #: The batched loop is narrator's own mirror of mlx-audio's
    #: `Model.batch_generate` - left-padded prompt embeddings behind
    #: `BatchKVCache(left_padding)`, `_step_batch_sampler` for boc/eoc, and a
    #: `filter(keep)` over the cache as rows retire. narrator keeps its own copy
    #: for the three reasons the single-row loop exists (no baked-in fade, a
    #: `should_stop`, and rows returned for the sentinel filter) plus a fourth:
    #: `batch_generate` takes ONE `limit` for the whole batch, where narrator
    #: caps each row at its own `cap_frames`.
    #:
    #: The WIDTH actually generated is `_mlx_width_for_depth`, derived from the
    #: batch's own depth against `MLX_MEM_BUDGET_GB`; one seed is drawn per
    #: batch (`_seed_for` of its first index), so a batch is reproducible and
    #: two batches never share a draw.
    #:
    #: NOTHING HERE IS MEASURED. Owen measures throughput and audio identity on
    #: the Mac; this side only guarantees the arithmetic and the row bookkeeping.
    BATCH_SIZE = 1

    #: Resident bf16 weights, in GB. PROVENANCE: `model.safetensors` of
    #: ds_ad4lm_prod_ckpt1080 is 8,489,897,458 bytes = 8.49 GB, and the base
    #: checkpoint is the same architecture at the same precision.
    MLX_WEIGHTS_GB = 8.5

    #: ARITHMETIC KV bytes per generated position per row, in MB, for the v3
    #: backbone: 36 layers x (K+V) x 8 kv heads x 128 head_dim x 2 bytes =
    #: 147,456 B = 0.140625 MB. This is the DOCUMENTED DEFAULT; `load_engine`
    #: recomputes it from the checkpoint's own `text_config` so a differently
    #: shaped Higgs is budgeted as itself rather than as this one.
    MLX_KV_MB_PER_TOKEN_ROW = 0.140625

    def __init__(self, config: HiggsV3MlxConfig):
        if not isinstance(config, HiggsV3MlxConfig):
            raise ValueError(
                'HiggsV3MlxEngine(config) takes a '
                'narrator.engine.higgs.HiggsV3MlxConfig; got '
                f'{type(config).__name__}.')
        self.config = config
        self.backend = 'mlx'
        self.voice_ref = config.voice
        self.voice = config.voice.name
        self.params = {'samplerate': self.SAMPLE_RATE}
        self._model = None
        self._reference_codes = None
        self._reference_texts = None
        self._codec_obj = None
        self._budget = HiggsV3MlxBudget(config)
        # The batch ceiling and its budget are read ONCE, here, so a mid-book
        # environment change cannot re-shape a running render - and so a bad
        # value refuses at construction rather than at the first batch.
        self.BATCH_SIZE = mlx_batch_ceiling()
        self.MLX_MEM_BUDGET_GB = mlx_mem_budget_gb()
        # Resolved ONCE, from the model directory's generation_config.json (see
        # HiggsV3MlxConfig.mlx_sampling). Re-reading the file per chunk would ask
        # the same question thousands of times a book and would let the answer
        # change mid-render.
        self._sampling = config.mlx_sampling()
        # THE STALL GUARD (stall_guard.py), read ONCE for the same reason as the
        # batch ceiling: a malformed HIGGS_STALL_GUARD refuses here, at
        # construction, and a running render cannot change shape under it.
        self._stall_guard = stall_guard_from_env()
        _log('Higgs v3 MLX stall guard: '
             + ('off' if self._stall_guard is None else
                f'frames={self._stall_guard.frames} rate={self._stall_guard.rate:g} '
                f'max={self._stall_guard.max:g} window={self._stall_guard.window}'))
        self.load_engine()

    # ---- lifecycle ----------------------------------------------------------

    @classmethod
    def detect_backend(cls) -> str:
        """'mlx'. The import is the probe: a Mac without mlx-audio cannot serve
        this engine, and saying so here is what stops the worker handshaking as
        healthy and then failing every generate (PORT_NOTES 12.11)."""
        import mlx_audio.tts.utils  # noqa: F401
        return 'mlx'

    def load_engine(self):
        """Load the weights, the tokenizer and the codec, and encode the voice's
        reference clips ONCE.

        `mlx_audio.tts.utils.load_model` reads `config.json`, builds
        `higgs_audio_v3.Model`, applies its `sanitize()` weight-name map and
        then its `post_load_hook`, which builds the tokenizer from
        `tokenizer.json` and the codec from the SAME safetensors shards
        (`HiggsAudioTokenizer.from_higgs_tts_checkpoint`). So the OFFICIAL HF
        weights load directly - there is no MLX conversion step, no quantized
        mlx-community repo needed and no separate codec download.

        `model_type` IS PASSED EXPLICITLY, and that is not belt-and-braces.
        v3's `config.json` says `model_type: "higgs_multimodal_qwen3"`, and
        mlx-audio 0.4.8 does carry the remapping for it
        (`tts/utils.py:MODEL_REMAPPING`) - but `utils.get_model_class` can never
        reach it: the branch that applies a remapping is
        `elif model_type_mapped is not None`, guarded by
        `if model_name is not None and model_type_mapped != model_type`, and a
        real remapping ALWAYS differs from its key, so the first branch always
        wins and the alias is never consulted. That branch instead scans the
        model PATH's components for something that happens to be named like a
        model directory. Measured on this Mac, 2026-09-05:

            ValueError: Model type higgs_multimodal_qwen3 not supported for tts.

        Naming the weights directory `higgs_audio_v3` would satisfy that scan -
        and that is exactly the wrong fix: it makes a load depend on a directory
        name, and it leaves the same path scan free to pick a DIFFERENT
        architecture out of any other component of the path (`llama`, `spark`
        and `dense` are all real model directories). `model_type=` skips the
        sniffing entirely and states which architecture narrator wants. The
        assertion in `_require_mlx_audio_surface` proves it is what it got.
        """
        import time

        import mlx.core as mx
        from mlx_audio.tts.utils import load_model

        target = self.config.model_dir
        # Bound the MLX buffer cache, exactly as the Orpheus MLX backend does at
        # load. Without a limit the allocator keeps every freed buffer, and a
        # single long Higgs render (per-step generation buffers in many distinct
        # sizes, then the codec decode) grew one process to ~50 GB resident on
        # the M1 Ultra before `cleanup()`'s `mx.clear_cache()` ever ran (Owen,
        # 2026-09-05, watching the MLX cap sweep). A bounded cache keeps the
        # footprint flat for the whole run and still reuses buffers, which a
        # per-chunk flush does not. The knob is the same shape as Orpheus's:
        # HIGGS_MLX_CACHE_LIMIT_GB, a documented tunable with a measured default,
        # not a fallback for a value that should have been set.
        cache_gb = mlx_cache_limit_gb()
        mx.set_cache_limit(int(cache_gb * 1e9))
        _log(f'Higgs MLX buffer cache limited to {cache_gb:g} GB')
        _log(f'loading Higgs v3 from {target}')
        started = time.perf_counter()
        model = load_model(target, model_type=MLX_AUDIO_ARCH)
        self._require_mlx_audio_surface(model)
        mx.eval(model.parameters())
        elapsed = time.perf_counter() - started
        _log(f'model loaded in {elapsed:.1f}s '
             f'(peak {mx.get_peak_memory() / 1e9:.2f} GB)')
        self._model = model
        self._adopt_kv_coefficient(model)
        self._announce_batch_budget()
        self._codec_obj = HiggsV3MlxCodec(
            self._decode_frames, label=f'voice {self.voice}')
        self._encode_references()
        # WHERE THE NUMBERS CAME FROM, honestly: since 2026-09-06 the voice
        # document can carry `sampling` (the catalog's engine-level block, or a
        # per-voice one with a stated reason), and it overrides the file. A
        # line that named the file while the document decided would be the
        # exact inherited-invisibly trap the document exists to end.
        if self.config.sampling:
            source = ('the voice document (catalog sampling), over ' + (
                os.path.join(self.config.voice.checkpoint_dir,
                             v3_served.GENERATION_CONFIG_FILE)
                if self.config.voice.checkpoint_dir else "v3's deploy default"))
        elif self.config.voice.checkpoint_dir:
            source = os.path.join(self.config.voice.checkpoint_dir,
                                  v3_served.GENERATION_CONFIG_FILE)
        else:
            source = ("v3's deploy default (the base weights carry no "
                      'generation_config.json)')
        _log(f'sampling {self._sampling} from {source}')
        return model

    @staticmethod
    def _require_mlx_audio_surface(model) -> None:
        """The loop below drives mlx-audio's model through members that are not
        a public API. Say so, and REFUSE by name when one is missing, rather
        than discovering it half a book in.

        Pinned to mlx-audio MLX_AUDIO_VERSION; a newer release that renames one
        of these must be measured, not adapted to at runtime.
        """
        # WHICH ARCHITECTURE ACTUALLY LOADED. mlx-audio's type resolution scans
        # the model PATH for a component named like one of its model packages
        # (see load_engine), so a directory called `llama` or `dense` anywhere in
        # the path could still override the explicit `model_type`. Check the
        # class narrator got rather than the one it asked for.
        module = type(model).__module__
        if not module.endswith(f'.{MLX_AUDIO_ARCH}.model'):
            raise RuntimeError(
                f'mlx-audio loaded {module}.{type(model).__name__} for a Higgs v3 '
                f'checkpoint; narrator asked for {MLX_AUDIO_ARCH}. mlx-audio picks an '
                'architecture partly from the model PATH, so a path component named '
                'like another model package can hijack the load - rename the weights '
                'directory rather than render a book with the wrong model.')
        needed = ('_build_prompt_embeddings', '_audio_logits',
                  '_embed_audio_codes', 'backbone', 'config', 'codec')
        missing = [name for name in needed if not hasattr(model, name)]
        if missing:
            raise RuntimeError(
                f'mlx-audio\'s higgs_audio_v3 model is missing {missing}. narrator\'s '
                f'MLX backend drives it through those members and is written against '
                f'mlx-audio {MLX_AUDIO_VERSION}; a release that renamed them needs the '
                'loop re-measured, not a guess at the new names.')
        if model.codec is None:
            raise RuntimeError(
                'mlx-audio loaded the Higgs v3 model with NO codec '
                '(post_load_hook found neither an audio_tokenizer/ directory nor '
                'codec tensors in the safetensors shards). There is nothing to turn '
                'tokens into audio - refusing to report a loaded engine.')
        if int(model.config.audio_num_codebooks) != NUM_CODEBOOKS:
            raise RuntimeError(
                f'Higgs v3 checkpoint declares '
                f'{model.config.audio_num_codebooks} codebooks; this backend\'s '
                f'delay-pattern arithmetic is written for {NUM_CODEBOOKS}.')
        if int(model.config.sample_rate) != v3_served.SAMPLE_RATE:
            raise RuntimeError(
                f'Higgs v3 checkpoint declares sample rate '
                f'{model.config.sample_rate}; narrator states '
                f'{v3_served.SAMPLE_RATE} on the wire and in every manifest.')

    def _adopt_kv_coefficient(self, model) -> None:
        """Budget the KV cache from THIS checkpoint's shape, not from a constant.

        36 layers x (K + V) x 8 kv heads x 128 head_dim x 2 bytes is 0.140625 MB
        per position per row for the v3 backbone, and the class constant says so
        - but the number that matters is the one the loaded model actually has,
        and mlx-audio parses it into `config.text_config`. A checkpoint with a
        different depth is then budgeted as itself. Anything missing leaves the
        documented default in place and SAYS so; it is a budget input, not a
        correctness input.
        """
        text_config = getattr(getattr(model, 'config', None), 'text_config', None)
        try:
            layers = int(text_config.num_hidden_layers)
            kv_heads = int(text_config.num_key_value_heads)
            head_dim = int(text_config.head_dim)
        except (AttributeError, TypeError, ValueError):
            _log(f'checkpoint declares no layer/head geometry; budgeting KV at the '
                 f'documented {self.MLX_KV_MB_PER_TOKEN_ROW:g} MB per position per row')
            return
        # x2 for K and V, x2 bytes for bf16.
        self.MLX_KV_MB_PER_TOKEN_ROW = (
            layers * 2 * kv_heads * head_dim * 2) / (1024.0 * 1024.0)

    def _announce_batch_budget(self) -> None:
        """ONE line at load saying what a batch may cost, and ONE saying that no
        one has measured it yet. Silent when the ceiling is 1: an unwidened
        process has no batch to budget and says nothing new.

        A BUDGET THAT CANNOT HOLD THE WEIGHTS TURNS BATCHING OFF, LOUDLY, and
        does not fail the load. BookForge hands over the Orpheus memory tier's
        budget, and Orpheus's weights are 6.9 GB against Higgs's 8.5: the small
        tiers (13 GB at 'light') can hold this model but not a batch of it.
        Refusing the LOAD there would break single-row rendering, which works
        fine on that machine, over a feature it was never going to get. So the
        FEATURE is refused, by name, in the log, once - the engine then renders
        exactly as it did before this change.
        """
        width = int(self.BATCH_SIZE or 1)
        if width <= 1:
            return
        try:
            headroom = self._mlx_kv_headroom_gb()
        except ValueError as refusal:
            self.BATCH_SIZE = 1
            _log(f'batched Higgs MLX rendering is OFF: {refusal} Rendering one '
                 'row at a time, exactly as an unconfigured process does.')
            return
        depth = v3_served.cap_frames('x' * int(self.config.max_chars))
        _log(f'Higgs MLX batch budget {self.MLX_MEM_BUDGET_GB:g} GB '
             f'({headroom:.1f} GB for KV at {self.MLX_KV_MB_PER_TOKEN_ROW:.4f} MB '
             f'per position per row -> max {self._mlx_width_for_depth(depth)} rows '
             f'at a {depth}-frame chunk, ceiling {width})')
        # THE CERTIFICATE IS SINGLE-ROW. The Mac's 900-char maxChars was measured
        # one row at a time; nothing has measured what a left-padded batch does to
        # it, and the catalog is unchanged. Say it out loud rather than letting a
        # widened render inherit a certificate it was not given.
        _log('batched Higgs MLX rendering is UNCERTIFIED: the voice maxChars '
             'certificate was measured SINGLE-ROW. Owen measures the batched '
             f'path; {BATCH_ENV}=1 restores it.')

    def _encode_references(self) -> None:
        """A ClipsVoice's reference audio, encoded ONCE at load.

        The reference is re-used verbatim for every chunk, so encoding it per
        request would pay the codec's encoder (HuBERT + the RVQ) thousands of
        times for an identical answer. mlx-audio exposes exactly this:
        `encode_reference_audio` returns the DELAYED reference codes that
        `generate(ref_audio_codes=...)` takes.
        """
        voice = self.voice_ref
        if not isinstance(voice, ClipsVoice):
            self._reference_codes = None
            self._reference_texts = None
            return
        codes, texts = [], []
        for clip in voice.clips:
            codes.append(self._model.encode_reference_audio(clip.path))
            texts.append(clip.transcript)
        self._reference_codes = codes
        self._reference_texts = texts
        total = v3_served.reference_seconds(voice)
        _log(f'encoded {len(codes)} reference clip(s) ({total:.1f}s) '
             f'for voice {self.voice}')

    def cleanup(self) -> None:
        """Release the model and let MLX give the memory back. Idempotent."""
        self._model = None
        self._codec_obj = None
        self._reference_codes = None
        self._reference_texts = None
        try:
            import mlx.core as mx
        except ImportError:
            return
        mx.clear_cache()

    # ---- the seam -----------------------------------------------------------

    @classmethod
    def resolve_load_voice(cls, voice, model_dir=None, adapter_dir=None,
                           base_dir=None) -> str:
        """The SAME refusals as the served arm, for the same reasons.

        `modelDir` is refused even though this backend does load a directory:
        which weights it loads is decided by the VOICE (a `checkpoint` voice
        names its own merged directory) or by MODEL_ENV, never by a per-load
        field - a load that could re-point the weights would let one worker
        answer for two different models under one name.
        """
        if model_dir:
            raise ValueError(
                f'Higgs v3 load carried modelDir={model_dir!r}. Which weights the MLX '
                f'backend loads comes from the voice document (a "checkpoint" voice '
                f'names its own merged directory) or from {MODEL_ENV}, not from a '
                'per-load field.')
        if base_dir:
            raise ValueError(
                f'Higgs v3 load carried baseDir={base_dir!r}. v3 has no shared-base + '
                'per-voice-adapter split.')
        if adapter_dir:
            raise ValueError(
                f'Higgs v3 load carried adapterDir={adapter_dir!r}. A fine-tuned Higgs '
                'voice is a MERGED checkpoint declared in the NARRATOR_HIGGS_VOICES '
                'document ({"kind": "checkpoint", "checkpointDir": ...}), not an '
                'adapter directory - and mlx-audio loads whole weights, not adapters.')
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
        # The named directory's REQUIRED FILES, at the load message - and for
        # a MERGE the generation_config.json is not just the server's
        # sampling, it is THIS process's (see HiggsV3MlxConfig.mlx_sampling).
        # A zero-shot voice's `base_dir` is base weights, which carry no such
        # file and are asked for none.
        v3_served.voice_serve_target(resolved)
        return name

    def backend_spec(self) -> BackendSpec:
        """IN-PROCESS. This is the whole point of the Mac arm: the weights are
        in this interpreter, so there is no server to launch, health-check or
        kill, and no HTTP between a chunk and its audio."""
        return BackendSpec(
            kind='inprocess', name='mlx',
            notes=f'mlx-audio {MLX_AUDIO_VERSION} higgs_audio_v3, '
                  f'weights {self.config.model_dir}')

    def codec(self) -> HiggsV3MlxCodec:
        if self._codec_obj is None:
            raise RuntimeError(
                'HiggsV3MlxEngine.codec(): the engine has been cleaned up; its codec '
                'went with the model.')
        return self._codec_obj

    def budget(self) -> HiggsV3MlxBudget:
        return self._budget

    def stop_policy(self, voice=None) -> StopPolicy:
        if voice is not None and voice != self.voice:
            raise ValueError(
                f'HiggsV3MlxEngine.stop_policy({voice!r}): this engine holds '
                f"'{self.voice}'. A second voice is a second engine.")
        return higgs_v3_mlx_stop_policy(self.config)

    def set_voice(self, voice: str, adapter_dir: str = None) -> None:
        """A second load on a live worker.

        The SAME voice is a no-op, which is what a pool re-issuing an identical
        load wants. A DIFFERENT voice is refused by name: a v3 voice is either a
        merged checkpoint (different weights - this engine holds one model) or a
        reference set encoded against those weights, and swapping either in
        place would leave the worker reporting voice B while it renders voice A.
        Unlike the served arm the fix is cheap - quit and reload, no 55-300 s
        server start - but it is still a reload.
        """
        want = (voice or '').strip()
        if adapter_dir:
            raise ValueError(
                f'HiggsV3MlxEngine.set_voice({want!r}, adapterDir={adapter_dir!r}): '
                'there is no runtime adapter on this stack.')
        if want == self.voice:
            return
        raise ValueError(
            f'HiggsV3MlxEngine cannot switch voice in place: this process loaded '
            f"'{self.voice}' from {self.config.model_dir}, and a load for '{want}' "
            'needs the weights and/or the encoded reference of a different voice. '
            'Quit this worker and start one for that voice.')

    def _apply_voice_caps(self, voice: str, caps: dict) -> None:
        """The Orpheus tuning caps have no meaning here. An EMPTY payload is the
        pool's "no catalog tuning" signal and is the no-op it looks like;
        anything else is refused by name."""
        if caps:
            raise ValueError(
                f'Higgs v3 load for {voice!r} carried caps={sorted(caps)}. Those are '
                'ORPHEUS tuning caps and v3 implements none of them.')

    def _clean_sentence_for_tts(self, sentence: str) -> str:
        """The SML strip, shared by every engine."""
        return clean_text(sentence)

    # ---- generation ---------------------------------------------------------

    def _seed_for(self, index: int):
        """THE seed rule, one place: chunk i renders with `seed + i`.

        Same rule as the served arm. A re-render of one chunk reproduces it, and
        two chunks of a batch never share a draw - rendering every sentence at a
        flat seed makes a whole book of identical draws for identical text.
        """
        if self.config.seed is None:
            return None
        return int(self.config.seed) + int(index)

    def _request_seed(self, seed, index: int, take: int):
        """THE SEED ONE RENDER ACTUALLY DRAWS AT - the served arm's
        `HiggsV3Engine._request_seed`, word for word, because the two arms must
        put take N in the same lane or one machine's take 3 is another's take
        0: the ladder's seed when it chose one, `seed + index` when it did not,
        shifted into take `take`'s lane either way.

        ON THIS BACKEND ONE SEED SERVES A WHOLE SLAB (`mx.random.seed` is drawn
        per batch), so the lane is applied to the BUCKET's seed and
        `_mlx_batch_groups` breaks a group when the take changes - the same
        split it already makes when the sampling changes, and for the same
        reason: a slab cannot carry two draws.
        """
        base = self._seed_for(index) if seed is None else seed
        return truncation.in_take_lane(base, take)

    def _decode_frames(self, frames_tq):
        """(frames, 8) real codes -> a float32 waveform. The codec call, and
        nothing else - every filter has already run in HiggsV3MlxCodec.decode."""
        import mlx.core as mx
        codes = mx.array(np.ascontiguousarray(frames_tq.astype(np.int32)))
        audio = self._model.codec.decode(codes)
        mx.eval(audio)
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def _generate_delayed_rows(self, text: str, cap: int, seed,
                               should_stop=None, sampling=None):
        """Run the sampler and return the (steps, 8) delayed rows, or None if
        `should_stop` went true mid-row.

        `sampling` IS the numbers for this row - the caller has already laid
        any take-ladder rung over the engine's own (`_sampling_for`). None
        means the engine's, which is what every caller passed before the
        ladder existed.

        This is `Model.generate`'s loop with three differences and no others:
        `should_stop` is checked every step, no fade is applied (there is no
        waveform here at all - the caller decodes), and the rows are returned as
        emitted so the decode can filter them by token identity.
        """
        import mlx.core as mx
        from mlx_audio.tts.models.higgs_audio_v3.generation import HiggsSamplerState, step
        from mlx_lm.models.cache import make_prompt_cache

        model = self._model
        if model is None:
            raise RuntimeError(
                'HiggsV3MlxEngine: no model is loaded (cleanup() has run).')
        if seed is not None:
            mx.random.seed(int(seed))

        references = self._references_for()
        prompt_embeds, prompt_tokens = model._build_prompt_embeddings(
            text, references)
        mx.eval(prompt_embeds)
        # The context window is a CEILING the prompt eats into; refuse a prompt
        # that already fills it rather than generating into nothing.
        self._budget.max_total_tokens(int(prompt_embeds.shape[1]))

        cache = make_prompt_cache(model)
        hidden = model.backbone(
            mx.zeros((1, prompt_embeds.shape[1]), dtype=mx.int32),
            cache=cache, input_embeddings=prompt_embeds)
        last_hidden = hidden[:, -1, :]

        state = HiggsSamplerState(num_codebooks=NUM_CODEBOOKS)
        if sampling is None:
            sampling = self._sampling
        stall = None if self._stall_guard is None else RowStall(self._stall_guard)
        rows = []
        for _ in range(int(cap)):
            if should_stop is not None and should_stop():
                return None
            logits = model._audio_logits(last_hidden)[0]
            # THE STALL GUARD, before `step` samples (so before temperature /
            # top-k / top-p): decided from the frames already sampled.
            was_counted = stall is not None and counted(state, NUM_CODEBOOKS)
            if was_counted and stall.penalty() > 0:
                logits = _lower_cb0(logits, None, stall.codes(), stall.penalty())
            codes = step(logits, state,
                         temperature=sampling['temperature'],
                         top_p=sampling['top_p'], top_k=sampling['top_k'],
                         boc_id=int(model.config.audio_boc_token_id),
                         eoc_id=int(model.config.audio_eoc_token_id))
            if stall is not None:
                _observe_stall(stall, codes, was_counted and not state.generation_done, 0)
            rows.append(codes)
            if state.generation_done:
                break
            next_embed = model._embed_audio_codes(codes)[None]
            hidden = model.backbone(mx.zeros((1, 1), dtype=mx.int32),
                                    cache=cache, input_embeddings=next_embed)
            last_hidden = hidden[:, -1, :]
        if not rows:
            raise HiggsMlxStreamMisaligned(
                f'Higgs v3 MLX: the sampler emitted no rows at all for a '
                f'{len(text)}-char chunk (cap {cap} frames, prompt {prompt_tokens} '
                'tokens).')
        stacked = mx.stack(rows, axis=0)
        mx.eval(stacked)
        return np.asarray(stacked).astype(np.int64)

    # ---- batch scheduling ---------------------------------------------------

    @property
    def batch_pool_size(self) -> int:
        """How many chunks the worker accumulates before calling
        `convert_batch`. Plain `BATCH_SIZE`: there is no continuous generator on
        this backend, so a deeper pool would only delay the worker's per-chunk
        progress lines without widening anything."""
        return max(1, int(self.BATCH_SIZE or 1))

    def _mlx_kv_headroom_gb(self) -> float:
        """GB left for KV once the resident weights and the PINNED buffer cache
        are paid for.

        Refuses when the budget cannot even hold those two: there is no honest
        width to fall back to, and running at full width anyway would defeat the
        whole point of a budget. Called at load (`_announce_batch_budget`) so a
        bad budget fails before any audio is rendered rather than mid-book.
        """
        cache_gb = mlx_cache_limit_gb()
        headroom = self.MLX_MEM_BUDGET_GB - self.MLX_WEIGHTS_GB - cache_gb
        if headroom <= 0:
            raise ValueError(
                f'{MEM_BUDGET_ENV}={self.MLX_MEM_BUDGET_GB:g} leaves no room for the KV '
                f'cache: {self.MLX_WEIGHTS_GB:g} GB of Higgs v3 weights + {cache_gb:g} GB '
                f'of pinned buffer cache ({CACHE_LIMIT_ENV}) already meet or exceed it. '
                f'Raise the budget, lower the cache limit, or set {BATCH_ENV}=1 to '
                'render one row at a time.')
        return headroom

    def _mlx_width_for_depth(self, depth: int, ceiling=None) -> int:
        """Widest batch whose KV stays inside the budget when every row may run
        to `depth` positions (its prompt plus its own frame cap):

            width x depth x MLX_KV_MB_PER_TOKEN_ROW
                <= MLX_MEM_BUDGET_GB - MLX_WEIGHTS_GB - the pinned cache

        Never more than `BATCH_SIZE`, never less than 1 - one row is the same
        work the single-row path would do anyway, so it is always attemptable.

        Worked example at the shipped defaults (budget 42, weights 8.5, cache 8,
        depth 2000): headroom 25.5 GB; per-row KV = 2000 x 0.140625 / 1024 =
        0.2747 GB; width = min(ceiling, 92).
        """
        # `ceiling` is ONE CALL'S width when a job asked for a narrower one
        # (`render_many(width=)`), never wider: the serving ceiling is what the
        # process was started under and a request cannot raise it.
        width = max(1, int(self.BATCH_SIZE or 1))
        if ceiling is not None:
            width = max(1, min(width, int(ceiling)))
        if depth <= 0:
            return width
        headroom = self._mlx_kv_headroom_gb()
        kv_gb_per_row = depth * self.MLX_KV_MB_PER_TOKEN_ROW / 1024.0
        return max(1, min(width, int(headroom / kv_gb_per_row)))

    def _mlx_batch_groups(self, entries: list, ceiling=None) -> list:
        """entries: `(index, text, prompt_positions, cap_frames, sampling,
        take)` in BOOK ORDER. Returns `(bucket, depth)` pairs, each bucket a
        CONSECUTIVE slice whose rows all render at the SAME sampling AND the
        same take.

        A GROUP BREAKS ON A SAMPLING CHANGE, and that is not a nicety.
        `_generate_delayed_rows_batch` runs one `_step_batch_sampler` over
        every active row with ONE temperature / top_p / top_k, so a slab
        carrying two take-ladder rungs would render one of them at the other's
        numbers and report both as asked for. Splitting is the only honest
        answer: this backend cannot mix, so it batches by group instead
        (`engine/item_sampling.py:group_key`). `sampling` None is its own group -
        the rows at take 0.

        IT BREAKS ON A TAKE CHANGE FOR THE SAME REASON (2026-09-15). One
        `mx.random.seed` is drawn per SLAB, so every row in a bucket shares one
        RNG stream and the bucket's seed is one row's
        (`_request_seed(..., take)`). Two takes in one slab would therefore
        both draw out of whichever take's lane the first row named - which is
        precisely the "take N rendered as take 0" this channel exists to stop,
        and it would be invisible because the audio WOULD differ, just not in
        the way the take asked for. The rung's two halves are both in the
        group key.

        No sorting and no length bucketing: mlx-audio's batch prefill LEFT-pads
        every prompt to the longest in the batch and hands the padding to
        `BatchKVCache(left_padding)`, so a short row batched beside a long one is
        masked correctly rather than silently corrupted. Book order also keeps
        chunks finishing roughly in the order the progress bar expects.

        `depth` is the deepest a row in the slice can reach - its prompt
        positions plus its own `cap_frames`, because every generated frame is one
        more backbone position on top of the prompt. A slice of short chunks
        therefore gets a shallow depth and, through `_mlx_width_for_depth`, buys
        back width.

        A slice too deep for its own width SHRINKS to the width that fits and
        the next slice starts there - 64 rows capped at 50 run 50, then 50, not
        32+32. See the comment on the loop for why the even split it replaced
        (2026-09-09) was throwing away rows.
        """
        from ..item_sampling import group_key
        width = max(1, int(self.BATCH_SIZE or 1))

        def _depth(rows):
            return max(int(e[2]) + int(e[3]) for e in rows)

        def _rung_key(entry):
            """THE WHOLE RUNG, both halves: the sampling's order-independent
            key and the take. Two rows share a slab only when both match."""
            return group_key(entry[4]), int(entry[5])

        def _run_length(start, limit):
            """How many CONSECUTIVE entries from `start` share one rung,
            looked at NO FURTHER than `limit` rows - the window can never be
            wider than that anyway, and a book of 1400 chunks at one rung would
            otherwise re-walk its whole tail once per window.

            The slab cannot straddle two rungs, so this is a ceiling on the
            window and not a preference."""
            key = _rung_key(entries[start])
            end = start + 1
            while end < start + limit and _rung_key(entries[end]) == key:
                end += 1
            return end - start

        groups = []
        i, n = 0, len(entries)
        while i < n:
            asked = _run_length(i, min(width, n - i))
            take = asked
            # A WINDOW THAT DOES NOT FIT SHRINKS TO WHAT DOES; it is not split
            # into equal parts (2026-09-09). The even split read the ceiling as a
            # UNIT to be divided — 64 rows capped at 50 became 32+32 rather than
            # 50+14, to avoid a near-solo tail. But the ceiling is a MAXIMUM and
            # the book runs on, so there is no tail to avoid: a window of 50 just
            # means the next window starts at 50. Dividing instead threw away
            # every row between `allowed` and `take`, and it cost the most
            # exactly when it was closest to fitting.
            #
            # MEASURED on Owen's Streicher render (2026-09-09): "narrowed 64 rows
            # -> 2 x ~32 (depth 2989 positions, cap 62, budget 42 GB)". 62 rows
            # fitted; it asked for 64, missed by two, and ran 32. Over 774 chunks
            # that is 24 batches instead of 12, and a batch costs about the wall
            # time of its deepest row however many rows ride in it.
            #
            # Re-measured on each shrink because a smaller window can be
            # SHALLOWER — its deepest row may be one of the rows just dropped —
            # which buys width back. `take` only ever decreases, so this ends.
            while True:
                window = entries[i:i + take]
                depth = _depth(window)
                allowed = self._mlx_width_for_depth(depth, ceiling)
                if allowed >= take or take <= 1:
                    break
                take = allowed
            if take < asked:
                _log(f'MLX batch narrowed {asked} rows -> {take} '
                     f'(depth {depth} positions, cap {allowed}, budget '
                     f'{self.MLX_MEM_BUDGET_GB:g} GB)')
            groups.append((window, depth))
            i += take
        return groups

    # ---- batched generation -------------------------------------------------

    def _mlx_prompts_for(self, texts: list) -> list:
        """`(prompt_embeds, positions)` per text, built ONCE.

        Positions are `prompt_embeds.shape[1]`, NOT `len(prompt.token_ids)`: an
        audio reference expands into one backbone position per delayed code row,
        so the token count understates the KV a reference costs - and the KV is
        what the budget is about.
        """
        model = self._model
        if model is None:
            raise RuntimeError(
                'HiggsV3MlxEngine: no model is loaded (cleanup() has run).')
        import mlx.core as mx
        references = self._references_for()
        prompts = []
        for text in texts:
            prompt_embeds, _prompt_tokens = model._build_prompt_embeddings(
                text, references)
            mx.eval(prompt_embeds)
            positions = int(prompt_embeds.shape[1])
            # The SAME refusal the single-row path makes, per row, BEFORE any
            # prefill: a prompt that already fills the window has nothing to
            # generate into, and finding that out inside a 64-row batch would
            # take the other 63 rows down with it.
            self._budget.max_total_tokens(positions)
            prompts.append((prompt_embeds, positions))
        return prompts

    def _generate_delayed_rows_batch(self, texts: list, caps: list, seed,
                                     should_stop=None, prompts=None,
                                     group_no: int = 1, group_count: int = 1,
                                     on_retire=None, sampling=None):
        """Generate a whole batch and return one `(steps, 8)` int64 matrix per
        row, in row order - or None if `should_stop` went true mid-batch.

        This is mlx-audio's `Model.batch_generate` loop with narrator's four
        differences and no others: no fade (there is no waveform here - the
        caller decodes), a `should_stop` checked every step, the rows returned as
        emitted so the sentinel filter can run on them, and a PER-ROW cap instead
        of one `limit` for the batch.

        THE PADDING IS LEFT PADDING, and that is the whole correctness story.
        Every prompt is padded at its FRONT to the longest in the batch and the
        pad widths are handed to `BatchKVCache(left_padding)`, which is what
        masks them out of attention. Right-padding would put the pad between the
        prompt and the first generated frame, and every short row in the batch
        would generate from garbage - the mixed-length corruption this loop is
        written to avoid.

        A row retires when its sampler says `generation_done` OR when it has
        emitted its own `cap` rows - the same two endings the single-row loop
        has. Retirement filters the cache by POSITION IN THE CURRENT ACTIVE LIST
        (`keep`), never by original row index, because `filter` re-slices the
        live batch dimension.

        `on_retire(row, rows_np)` - OPTIONAL, and the whole reason the Listen
        path can batch its read-ahead. A batch is otherwise ATOMIC to its
        caller: nothing leaves until the deepest row in it has finished, which
        on a 64-row batch is minutes after the shallowest row was ready. Called
        the moment a row retires, with that row's own `(steps, 8)` matrix
        already stacked and EVALUATED, so the caller can decode and emit it
        while the remaining rows keep generating. The return value is unchanged
        either way: the full list in row order, or None if `should_stop` went
        true - and a row abandoned that way never retired, so it gets no
        `on_retire`.

        WHAT `on_retire` COSTS IS NOT MEASURED. It puts a codec decode on the
        same GPU the batch is still generating on, at a moment chosen by the
        batch rather than by the caller. Owen measures that on the Mac; this
        side guarantees only that each row is evaluated exactly once and handed
        over exactly once.

        ONE SAMPLING FOR THE WHOLE SLAB, and that is a fact of the loop, not a
        simplification: `_step_batch_sampler` takes ONE temperature / top_p /
        top_k for every active row at once. So this method cannot mix
        take-ladder rungs, and nothing above it may hand it rows that disagree
        - `_mlx_batch_groups` breaks a group on a sampling change so a slab is
        always homogeneous. `sampling` None means the engine's own.
        """
        import mlx.core as mx
        from mlx_audio.tts.models.higgs_audio_v3.generation import HiggsSamplerState
        from mlx_lm.models.cache import BatchKVCache

        model = self._model
        if model is None:
            raise RuntimeError(
                'HiggsV3MlxEngine: no model is loaded (cleanup() has run).')
        if len(caps) != len(texts):
            raise ValueError(
                f'HiggsV3MlxEngine._generate_delayed_rows_batch: {len(caps)} caps for '
                f'{len(texts)} texts')
        if not texts:
            return []
        # ONE SEED PER BATCH. Every row of a batch is drawn from one stream, so a
        # batch is reproducible as a batch; the caller passes `_seed_for` of the
        # batch's FIRST index so two batches never share a draw.
        if seed is not None:
            mx.random.seed(int(seed))

        if prompts is None:
            prompts = self._mlx_prompts_for(texts)
        lengths = [positions for _embeds, positions in prompts]
        max_prompt_length = max(lengths)
        left_padding = [max_prompt_length - length for length in lengths]
        padded = []
        for (prompt_embeds, _positions), pad in zip(prompts, left_padding):
            if pad:
                prompt_embeds = mx.pad(prompt_embeds, [(0, 0), (pad, 0), (0, 0)])
            padded.append(prompt_embeds)
        prompt_batch = mx.concatenate(padded, axis=0)

        batch_size = len(texts)
        batch_cache = [BatchKVCache(left_padding) for _ in model.layers]
        hidden = model.backbone(
            mx.zeros((batch_size, max_prompt_length), dtype=mx.int32),
            cache=batch_cache, input_embeddings=prompt_batch)
        last_hidden_batch = hidden[:, -1, :]
        mx.eval(last_hidden_batch)

        samplers = [HiggsSamplerState(num_codebooks=NUM_CODEBOOKS)
                    for _ in range(batch_size)]
        delayed_rows = [[] for _ in range(batch_size)]
        if sampling is None:
            sampling = self._sampling
        active = list(range(batch_size))
        limit = max(int(c) for c in caps)
        retired = 0
        heartbeat_seconds = 2.0
        last_beat = _time.time()
        # Rows stacked and evaluated AT RETIREMENT (when `on_retire` is
        # watching), remembered so the tail below never evaluates one twice.
        finished = {}

        def _finish(row):
            stacked = mx.stack(delayed_rows[row], axis=0)
            mx.eval(stacked)
            matrix = np.asarray(stacked).astype(np.int64)
            finished[row] = matrix
            return matrix

        stalls = (None if self._stall_guard is None
                  else [RowStall(self._stall_guard) for _ in range(batch_size)])
        for stepno in range(1, limit + 1):
            if not active:
                break
            if should_stop is not None and should_stop():
                return None
            logits_batch = model._audio_logits(last_hidden_batch)
            # THE STALL GUARD, per row, before the sampler - the same contract as
            # the single-row loop. Only rows past `frames` touch their logits.
            was_counted = ([] if stalls is None
                           else [counted(samplers[row], NUM_CODEBOOKS) for row in active])
            if stalls is not None:
                for position, row in enumerate(active):
                    if was_counted[position] and stalls[row].penalty() > 0:
                        logits_batch = _lower_cb0(logits_batch, position, stalls[row].codes(),
                                                  stalls[row].penalty())
            sampled_rows = model._step_batch_sampler(
                logits_batch,
                [samplers[i] for i in active],
                temperature=sampling['temperature'],
                top_p=sampling['top_p'], top_k=sampling['top_k'])
            if stalls is not None:
                _observe_stall_batch(stalls, active, sampled_rows, [
                    was_counted[position] and not samplers[row].generation_done
                    for position, row in enumerate(active)])

            next_active, next_codes, keep_positions = [], [], []
            for position, (row, codes) in enumerate(zip(active, sampled_rows)):
                delayed_rows[row].append(codes)
                # The two endings, in the single-row loop's order: the row is
                # kept only if the sampler has not finished AND it has not
                # reached its own frame cap.
                if samplers[row].generation_done or len(delayed_rows[row]) >= int(caps[row]):
                    retired += 1
                    # THE ROW IS FINISHED HERE, not when the batch is. Handing
                    # it over now is what lets the Listen path play a read-ahead
                    # sentence while its batch-mates are still generating.
                    if on_retire is not None:
                        on_retire(row, _finish(row))
                    continue
                keep_positions.append(position)
                next_active.append(row)
                next_codes.append(codes)

            now = _time.time()
            if now - last_beat >= heartbeat_seconds:
                last_beat = now
                # LOAD-BEARING LOG STRING - the ONLY signal inside a batch that
                # runs for minutes. parallel-tts-bridge.ts's
                # GENERATION_ACTIVITY_RE reads it as liveness and
                # mlx-batch-progress.ts parses its fields into the within-batch
                # bar; neither anchors the engine prefix, so the wording and
                # field ORDER after "MLX batch generating:" are what matter.
                _log(f'MLX batch generating: {batch_size} rows, ~{stepno} tokens '
                     f'(step {stepno}/{limit}), {retired}/{batch_size} rows done, '
                     f'batch {group_no}/{group_count}')

            if not next_active:
                break
            if len(next_active) != len(active):
                keep = mx.array(keep_positions, dtype=mx.int32)
                for layer_cache in batch_cache:
                    layer_cache.filter(keep)
            codes_batch = mx.stack(next_codes, axis=0)
            next_embed = model._embed_audio_codes(codes_batch)[:, None, :]
            hidden = model.backbone(
                mx.zeros((len(next_active), 1), dtype=mx.int32),
                cache=batch_cache, input_embeddings=next_embed)
            last_hidden_batch = hidden[:, -1, :]
            active = next_active

        out = []
        for row, rows in enumerate(delayed_rows):
            if not rows:
                raise HiggsMlxStreamMisaligned(
                    f'Higgs v3 MLX: the sampler emitted no rows at all for row {row} of '
                    f'a {batch_size}-row batch ({len(texts[row])} chars, cap '
                    f'{caps[row]} frames).')
            out.append(finished[row] if row in finished else _finish(row))
        return out

    def _references_for(self):
        """The encoded reference clips as mlx-audio's `ReferenceCodes`, or ()
        for a DefaultVoice (a fine-tuned checkpoint, or the model's own voice) -
        both of which are prompted with TEXT ALONE."""
        if self._reference_codes is None:
            return ()
        from mlx_audio.tts.models.higgs_audio_v3.prompt import ReferenceCodes
        return [ReferenceCodes(codes=codes, text=text)
                for codes, text in zip(self._reference_codes,
                                       self._reference_texts)]

    #: The levers a per-item rung may name on the MLX arm. NO
    #: `repetitionPenalty`: mlx-audio's `higgs_audio_v3` has no repetition
    #: penalty at all (PORT_NOTES 13.11) and `HiggsV3MlxConfig.__post_init__`
    #: already refuses it from the voices document, so a rung asking for it is
    #: `sampling_not_supported` here rather than a lever that looks applied and
    #: does nothing.
    ITEM_SAMPLING_LEVERS = ('temperature', 'topP', 'topK')

    def accept_item_sampling(self, raw, where: str = None):
        """One item's `sampling` off the wire -> this engine's dict, or None.
        Both refusals live in `engine/item_sampling.py`."""
        from ..item_sampling import parse_item_sampling
        return parse_item_sampling(raw, where or f'HiggsV3MlxEngine({self.voice!r})',
                                   levers=self.ITEM_SAMPLING_LEVERS)

    def accept_item_take(self, raw, where: str = None) -> int:
        """One item's `take` off the wire -> a whole number >= 0. The served
        arm's rule verbatim (`HiggsV3Engine.accept_item_take`): a rung is
        (sampling deltas, seed offset), and this is the offset's number.
        `take_malformed` is the refusal."""
        from ..item_sampling import parse_item_take
        return parse_item_take(raw, where or f'HiggsV3MlxEngine({self.voice!r})')

    def _sampling_for(self, item_sampling) -> dict:
        """The numbers ONE render runs at: `self._sampling` (resolved once from
        the checkpoint's generation_config.json) with the item's rung laid over
        it. An overlay, not a replacement - `engine/item_sampling.py` says why, and
        this arm needs all three keys present because `generation.step` takes
        them positionally."""
        from ..item_sampling import apply_over
        return apply_over(self._sampling, item_sampling)

    def render_audio(self, text: str, seed=None, index: int = 0,
                     should_stop=None, sampling=None, take: int = 0):
        """One chunk of text -> a float32 mono waveform at 24 kHz.

        `index` is the chunk's own index and is what seeds it (see `_seed_for`);
        an explicit `seed` overrides. Returns None only when `should_stop` went
        true mid-generation.

        `sampling` is this chunk's take-ladder rung (already parsed by
        `accept_item_sampling`); None renders at the voice's loaded sampling,
        which is take 0.

        `take` is WHICH RUNG, and it moves the seed rather than the numbers
        (`_request_seed`) - so a rung that declares no sampling override is
        still a different draw. 0 changes nothing.
        """
        return self.render_audio_measured(
            text, seed=seed, index=index, should_stop=should_stop,
            sampling=sampling, take=take)[0]

    def render_audio_measured(self, text: str, seed=None, index: int = 0,
                              should_stop=None, sampling=None, take: int = 0):
        """`render_audio`, with the `FrameMeasure` it spent - `(audio, measure)`.

        COUNTED, not inferred, on this arm: the loop runs in THIS process, so
        the frames the model generated are the rows it handed back and the cap
        is the number they were generated against. That is the engine's own
        stop reason, read from its own output, which is why this arm reports
        `tokens` and the served arm cannot (see
        `HiggsV3Engine.render_audio_measured`).

        A stop mid-generation returns `(None, None)`: nothing was decided and
        nothing is claimed about the cap.
        """
        # THE MODEL BOUNDARY STRIPS THE MARKUP - once, for every caller. See
        # HiggsV3Engine.render_audio: this path only trimmed whitespace, so the
        # packer's `[break]` / `[heading]` reached the prompt and were READ
        # ("break" at the head of 677 of 1067 chunks; 7-15 s of gibberish before
        # "Dedication."), measured on this arm 2026-09-05.
        clean = self._clean_sentence_for_tts(text)
        if not clean:
            raise ValueError(
                'HiggsV3MlxEngine.render_audio(): the chunk has no text once its '
                f'markers are stripped ({(text or "").strip()!r}).')
        # The 45-token allowlist. An UNKNOWN control token is not ignored: the
        # model reads it out loud as words and the chunk collapses.
        v3_served.validate_control_tokens(clean)
        cap = self._budget.cap_frames(clean)
        rows = self._generate_delayed_rows(
            clean, cap, self._request_seed(seed, index, take),
            should_stop=should_stop, sampling=self._sampling_for(sampling))
        if rows is None:
            return None, None
        return (self.codec().decode(rows),
                FrameMeasure.counted_frames(len(rows), cap))

    def _sentence_file(self, sentence_number: int) -> str:
        if not self.config.sentences_dir:
            raise ValueError(
                'HiggsV3MlxEngine.convert() needs a sentences_dir: this engine was '
                'built for in-memory generation, so there is nowhere to write chunk '
                f'{sentence_number}. Use generate_batch_stream / render_audio.')
        return os.path.join(self.config.sentences_dir,
                            f'{sentence_number}.{self.config.audio_format}')

    def _write_sentence(self, sentence_number: int, audio) -> bool:
        """The chunk file, EXACTLY AS DECODED - no trim, no fade, no pad. The
        fades are the assembler's (`edge_fade`) and the gaps are the manifest's.

        ONE writer for both paths: a batched render must land byte-identically
        to a single-row one, and two copies of `sf.write` is how a subtype or a
        container drifts between them."""
        import soundfile as sf
        path = self._sentence_file(sentence_number)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, audio, self.SAMPLE_RATE, subtype='PCM_16',
                 format=self.config.audio_format.upper())
        return True

    def _write_silence(self, sentence_number: int) -> bool:
        """An empty sentence's placeholder clip - see `Engine._write_silence`.

        THROUGH `_write_sentence`, for the reason that writer exists: the
        silence lands with the same subtype and container as every rendered
        chunk beside it. Nothing is generated - `render_audio` refuses an empty
        chunk by name, and the ladder's door refuses one before any prefill.
        """
        return self._write_sentence(
            sentence_number,
            np.zeros(int(self.SAMPLE_RATE * EMPTY_SENTENCE_SILENCE_SEC),
                     dtype=np.float32))

    def convert(self, sentence_number: int, sentence: str) -> bool:
        """Render one chunk to `<sentences_dir>/<n>.<audio_format>`, through
        the truncation ladder (`truncation.render_guarded`, shared with the
        served arm)."""
        return self._write_sentence(
            sentence_number, self._render_guarded(sentence_number, sentence))

    def _render_guarded(self, index: int, text: str, first_take=None):
        """`render_audio` under the truncation ladder - see `truncation.py`.
        `first_take` is the batch path's already-decoded row, so a row that
        passes the guard is never rendered twice."""
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

        ON THE ENGINE'S OWN BAND (`truncation.engine_band`), never on the voice
        entry's: the band has ONE owner, and on the serve door that owner is
        the batch (Owen, 2026-09-19 - `truncation.tracker_for`). `convert` /
        `convert_batch` are narrator rendering a book itself, with no caller to
        be given a band by, which is why the engine's measured pair is the
        honest source here and is read in this one place.
        """
        tracker = getattr(self, '_pace', None)
        if tracker is None:
            tracker = truncation.tracker_for(
                truncation.engine_band(float(self.config.max_chars_per_sec),
                                       float(self.config.min_chars_per_sec)),
                'HiggsV3MlxEngine._pace_tracker')
            self._pace = tracker
        return tracker

    def convert_batch(self, items) -> list:
        """Render `items` - `(index, text)` in BOOK ORDER - and answer one bool
        per item, in that order.

        SERIAL at the default ceiling of 1 (see `BATCH_SIZE`), which is what an
        unconfigured process does: chunk by chunk through `convert`, the one
        path that renders and writes a single row. Above it this method is
        nothing but a SINK on `render_many` - the driver runs the slices, the
        ladder and the retake rounds, and every chunk it decides is written to
        `<sentences_dir>/<i>.<audio_format>` through the one shared writer.

        THAT SPLIT IS NEW ON 2026-09-13 AND IS PLUMBING ONLY. Owen ruled that
        the model and its inference own the guard AND the retake decision, and
        that a rendered chunk has to be able to travel to another machine. The
        driver used to live HERE, below the file-writing layer, so the serve
        world - which has a pipe and no `sentences_dir` - could not reach the
        ladder at all and rendered through a bare `render_audio`: one take, no
        band, no re-roll, no split. Lifting the driver above the writer is what
        lets both worlds share it (crucible/docs/PHASE6-REMOTE-RENDER.md
        sections 0 and 2). Not one guard decision moved with it.

        A GENERATION FAILURE IS LOUD. It raises, naming the slice's width and
        its row indices on top of the original error, exactly as `convert`
        raises when one row fails - there is no per-item retry. A retry was
        here until Owen struck it on 2026-09-05: it is a silent fallback in the
        shape the repo forbids. A batch that failed for a reason a single row
        would also fail for (a poisoned Metal context, a prompt over the window)
        re-renders the whole slice one row at a time to arrive at the same
        error N times, and a batch that failed for a reason batching CAUSED is
        precisely the fact the retry hides - the run reports success and nobody
        learns the width is wrong.

        There is no `should_stop` here: the worker owns the stop for the
        audiobook path and drops a flush's in-flight indices itself.
        """
        items = [(int(index), text) for index, text in items]
        if int(self.BATCH_SIZE or 1) <= 1 or not items:
            return [self.convert(index, text) for index, text in items]
        results = {}
        # DRAINED, not sampled. `render_many` is a generator, so no row renders
        # until something pulls on it and the LAST slice's retake rounds only
        # run because this loop runs to exhaustion - the same "every row of
        # `items` is decided before this returns" the method has always kept.
        for index, audio, _verdict, _measure in self.render_many(
                items, tracker=self._pace_tracker()):
            # No `width`: `convert_batch` IS the engine's own book render, so
            # the engine's own ceiling is the right one and there is no job to
            # have asked for another.
            results[index] = self._write_sentence(index, audio)
        return [results.get(index, False) for index, _text in items]

    # -- the driver, above the files ----------------------------------------

    def render_many(self, rows, in_flight=None, sampling_by_index=None,
                    take_by_index=None, *, tracker, width=None):
        """THE GUARDED DRIVER, AND NOT ONE FILE WRITTEN: `(index, audio,
        verdict, measure)` for every chunk in `rows`, yielded the moment the
        ladder decides it.

        `tracker` is the `PaceTracker` this call judges against, and it is
        REQUIRED - the band has one owner per call and no default (Owen,
        2026-09-19). The serve door builds it from the band on the batch;
        `convert_batch` passes the engine's own (`_pace_tracker`). `measure` is
        the shipped take's `FrameMeasure`, joined across the parts when the
        ladder split the chunk: a measurement, taken whether or not anything
        judged the chunk.

        `width` is how many rows this CALL may keep in flight, at most this
        engine's `BATCH_SIZE` ceiling (`NARRATOR_HIGGS3_MLX_BATCH`). Absent
        means that ceiling - the engine's own configuration, not a default. It
        rides down to `_mlx_batch_groups`, so a narrower call really does build
        narrower slabs rather than merely being asked to: the KV of a slab is
        `width x depth`, and the measurement that produced this argument is a
        card that paged rather than failed when the sum went over.

        Owen's ruling of 2026-09-13 - the model and its inference own the guard
        and the retake decision, and the chunks have to be able to travel back
        from another computer. Until then this backend had two rendering worlds
        and only one of them was guarded: `convert_batch` ran `GuardPlan` and
        wrote the sentence files in the same method, while `narrator.serve` (and
        so Crucible, which drives it) came in through a bare `render_audio` and
        got one unjudged take. The guard was unreachable without also acquiring
        a `sentences_dir`. This generator IS that driver with the writer taken
        off it; `convert_batch` is now one of its two sinks and the serve world
        is the other. crucible/docs/PHASE6-REMOTE-RENDER.md sections 0 and 2.

        SAME `GuardPlan`, same `PaceTracker`, same bands, same `MIN_GUARD_CHARS`
        rule, same round ordering, same accept rule, same batch-seed rule for a
        re-roll. A changed decision here would have to be re-measured against the
        Mistborn pause map before anyone could trust it, and a plumbing change
        does not (PHASE6 section 8). `verdict` is `GuardPlan.verdict(index)` -
        the ladder's own conclusion plus its take records verbatim - because a
        driver shipping a chunk over a socket has no shared stderr for its
        caller to scrape the `[HIGGS3][HIGGS_GUARD_EVENT]` lines off. The lines
        are still printed; the records are now also readable as data.

        `rows` is `(index, text)` in BOOK ORDER and is MATERIALISED, unlike the
        served arm's lazily-pulled `convert_many`: this backend slices by prompt
        POSITIONS (`_mlx_batch_groups`), so it has never been able to choose a
        width without the whole list in hand.

        THE MARKER STRIP IS MADE HERE, AT THE LADDER'S DOOR, ONCE PER CHUNK and
        whatever the width - the position `convert_batch` used to make it from,
        and the position `HiggsV3Engine.convert_many` makes it from on the
        served arm. `render_audio` strips again at the model boundary and is
        idempotent about it, but the GUARD counts CHARACTERS: a chunk still
        carrying the packer's `[break]` / `[heading]` would be judged, and
        split, on text the model is never going to be given. The control-token
        allowlist is checked here for the same reason `convert_batch` checked it
        here - BEFORE any prefill, where a bad row does not take a batch down.

        `in_flight`, when given, is the caller's list of chunk indices with a
        render outstanding right now, kept exact the way
        `HiggsV3Engine.convert_many` keeps it so a cooperative stop knows
        precisely which rows were running. A chunk joins it when its first
        render is issued and leaves it when the ladder DECIDES it, so a chunk
        part-way up the ladder - re-rolling, or waiting on its split halves -
        stays named. It is the caller's list and is mutated in place.

        WHICH ARM RAISES, AND WHY ONLY THAT ONE (corrected 2026-09-13).

        `audio is None` is the shared contract's "this chunk failed", and the
        SERIAL arm (`_render_many_serial`, the shipped default width) yields it,
        exactly as the served arm does at `HiggsV3Engine.convert_many`: the
        render is wrapped, the chunk is struck off the ladder with
        `plan.abandon`, and the take goes on. One bad sentence costs one
        sentence.

        THE WIDTH FAILURE STILL RAISES, on `_render_many_rounds` only: a SLICE
        that fails takes the call down, naming its width and its rows
        (`_batch_failure`), because Owen struck the per-item retry on 2026-09-05
        and a batch that failed for a reason BATCHING CAUSED - a left-padded
        prefill, a shared cache, a width the memory could not hold - is
        precisely the fact a per-item retry hides. That rationale is about
        WIDTH, and at `BATCH_SIZE = 1` there is no slice for it to be about:
        letting it stand there meant one chunk's decode failure, one
        `HiggsMlxStreamMisaligned`, reported every row not yet emitted as
        'Batch generation failed' on a machine that had rendered nothing as a
        batch. The Mac failed a book where the PC failed a sentence.

        THE DOOR ABOVE IS STILL A WHOLE-CALL REFUSAL and is deliberately not
        part of that: a chunk that is only markers, or that carries a control
        token, is refused BEFORE any render, for both arms, because it is an
        INPUT defect rather than a render failure and the caller handed it in.

        NOT THREAD-SAFE, and deliberately not made so: `GuardPlan` is one
        batch's on one thread, and every `add` / `round` / `offer` / `finished`
        below happens on the thread that drives this generator. The only shared,
        locked thing is the engine's one `PaceTracker`.

        `sampling_by_index`, when given, maps EVERY chunk index in `rows` to
        its take-ladder rung (None = take 0). Keyed by chunk index and not
        carried on the row, because a re-roll and both halves of a split carry
        their parent's index and must render at their parent's numbers. A row
        the map does not name is refused by name - a `.get()` here would render
        rung 1 at rung 0's numbers, which is the whole failure the channel
        exists to prevent. On THIS backend a mixed-rung batch is also a SPLIT
        batch: the slab sampler takes one set of numbers, so `_mlx_batch_groups`
        breaks a group whenever the sampling changes.

        `take_by_index` is the rung's other half - which lane the chunk's seed
        draws from - keyed by chunk index for the same reason and covering
        every row the same way. It splits a slab too (`_mlx_batch_groups`):
        one `mx.random.seed` per batch means one take per batch.
        """
        held = [] if in_flight is None else in_flight
        rungs = None if sampling_by_index is None else dict(sampling_by_index)
        takes = None if take_by_index is None else dict(take_by_index)
        # ONE STRIP, ONE ALLOWLIST CHECK, BOTH WIDTHS - see the docstring. This
        # is `convert_batch`'s own loop, moved up with the driver and now
        # covering the serial arm too, which used to leave the refusal to
        # `render_audio` and so refused an empty chunk one rung later.
        cleaned = []
        for index, text in rows:
            clean = self._clean_sentence_for_tts(text)
            if not clean:
                raise ValueError(
                    f'HiggsV3MlxEngine.render_many(): chunk {index} has no text '
                    f'once its markers are stripped ({(text or "").strip()!r})')
            v3_served.validate_control_tokens(clean)
            if rungs is not None and int(index) not in rungs:
                raise KeyError(
                    f'HiggsV3MlxEngine.render_many: chunk {index} has no entry in '
                    'sampling_by_index. The map is given per CALL and must name '
                    'every row (None for a chunk at take 0); a missing key would '
                    "render a ladder's rung at take 0's numbers.")
            if takes is not None and int(index) not in takes:
                raise KeyError(
                    f'HiggsV3MlxEngine.render_many: chunk {index} has no entry in '
                    'take_by_index. The map is given per CALL and must name every '
                    'row (0 for a chunk at take 0); a missing key would render a '
                    "retake in take 0's own seed lane.")
            cleaned.append((int(index), clean))
        if not cleaned:
            return
        # The rung per chunk, in the SAME shape the two drivers below read: a
        # callable so the serial arm, the slab arm and the retake rounds all
        # resolve one chunk's numbers one way.
        def rung_for(index):
            return None if rungs is None else rungs[int(index)]

        def take_for(index):
            return 0 if takes is None else takes[int(index)]
        # ONE PLAN FOR THE WHOLE CALL, never one per chunk: the band re-centres
        # on the takes already shipped (Owen, 2026-09-08, "the calculated and
        # recorded characters per second"), and a plan per chunk would throw the
        # book's own pace away between every sentence.
        # AND THE PLAN IS WHAT VARIES, exactly as on the served arm. A batch
        # that said `retake: false` has no band, so it gets a plan that judges
        # nothing — not `GuardPlan(tracker=None)`, which falls back to a fixed
        # band and would judge a screening checkpoint against another model's
        # numbers. The grouping below is untouched either way: the width is this
        # engine's to spend, and an unjudged batch spends it the same as a
        # judged one. Before 2026-09-20 an unjudged batch never reached here and
        # was rendered one row at a time by the serve worker instead.
        plan = (truncation.UnjudgedPlan(sample_rate=self.SAMPLE_RATE,
                                        base_seed=self.config.seed)
                if tracker is None else
                truncation.GuardPlan(sample_rate=self.SAMPLE_RATE,
                                     base_seed=self.config.seed,
                                     tracker=tracker))
        ceiling = int(self.BATCH_SIZE or 1) if width is None else int(width)
        if ceiling < 1:
            raise ValueError(f'render_many needs a width >= 1; got {ceiling}.')
        if ceiling <= 1:
            yield from self._render_many_serial(plan, cleaned, held, rung_for,
                                                take_for)
            return
        yield from self._render_many_rounds(plan, cleaned, held, rung_for,
                                            take_for, ceiling)

    def _render_many_serial(self, plan, cleaned, held, rung_for, take_for):
        """`render_many` at the shipped default width of 1: one chunk at a
        time, the ladder walked depth-first by taking `next_request()` every
        time. `cleaned` is `(index, text)` already through the marker strip.

        `rung_for(index)` is the chunk's take-ladder sampling and
        `take_for(index)` its seed lane, both resolved by CHUNK index so every
        take of a chunk - take 0, the re-roll, both halves of a split - runs at
        the numbers and in the lane that chunk was asked for.

        This is the loop `truncation.render_guarded` runs for `convert`, written
        out rather than called for one reason: that function returns the audio
        alone and drops the verdict, which is the whole thing this door exists
        to carry. The renders are the same renders in the same order - a solo
        `render_audio` per take, never a one-row slab, because a batch of one is
        not byte for byte the single-row path and nothing on this backend widens
        on its own (`generate_batch_stream`, rung 2, states the same rule).

        ONE CHUNK'S FAILURE COSTS ONE CHUNK, which is the whole difference
        between this arm and `_render_many_rounds`, and it is the served arm's
        rule verbatim (`HiggsV3Engine.convert_many`: "This chunk's failure,
        named, and the take goes on"). A take that raises - a decode that came
        back misaligned (`HiggsMlxStreamMisaligned`), a codec that refused its
        frames, a row the sampler could not finish - abandons that chunk's whole
        tree (`GuardPlan.abandon`: a chunk part-way up the ladder has no audio
        to ship, because the take that would have been accepted is the one that
        failed), strikes it off `in_flight`, and yields `(index, None, None)`.
        The chunks behind it still render.

        `Exception`, NOT `BaseException`: a `GeneratorExit` from a consumer that
        closed this generator, a `KeyboardInterrupt`, and the `SystemExit` a
        cooperative stop raises must all still unwind - none of them is this
        chunk's failure, and the rows behind them are not owed a render.
        """
        for index, text in cleaned:
            plan.add(index, text)
            if index not in held:
                held.append(index)
            while True:
                request = plan.next_request()
                if request is None:
                    break
                try:
                    take, measure = self.render_audio_measured(
                        request.text, seed=request.seed,
                        index=request.index,
                        sampling=rung_for(request.index),
                        take=take_for(request.index))
                except Exception as exc:
                    # NAMED on the host's log stream first, the way the served
                    # arm names it: the tuple tells a caller THAT the chunk
                    # failed and the line tells them why.
                    _log(f'sentence {request.index} failed: '
                         f'{type(exc).__name__}: {exc}')
                    failed = plan.abandon(request)
                    if failed in held:
                        held.remove(failed)
                    yield failed, None, None, None
                    # The abandoned chunk's whole tree is gone from the plan, so
                    # the next `next_request()` is ASKED - and answers None -
                    # rather than assumed. Nothing else is in flight on this arm.
                    continue
                plan.offer(request, take, measure)
            yield from self._decided(plan, held)

    def _render_many_rounds(self, plan, cleaned, held, rung_for, take_for,
                            ceiling=None):
        """`render_many` above width 1: consecutive memory-budgeted slices
        (`_mlx_batch_groups`), each generated in one left-padded batch and
        decoded per row, then the guard's retakes a ROUND at a time. `cleaned`
        is `(index, text)` already through the marker strip and the allowlist.

        THE LENGTH GUARD'S RETAKES RIDE THE SAME BATCHES: a row whose take 0 is
        off-length is recorded rather than re-rendered on the spot, and after
        the last slice every recorded retake renders together
        (`_render_retake_rounds`). Every row is still decided by the time this
        generator is exhausted - the deferral is inside the call, exactly as
        Orpheus's `_render_deferred_resplits` defers inside its own batch.
        """
        # Built ONCE for the whole call, because the SLICING needs the prompt
        # POSITIONS: how deep a batch can get is its prompt plus its frame cap,
        # and a prompt carries one position per delayed reference code row.
        prompts = self._mlx_prompts_for([clean for _index, clean in cleaned])
        prompt_by_index = {index: prompt
                           for (index, _clean), prompt in zip(cleaned, prompts)}
        # THE RUNG RIDES ON THE ENTRY because the SLICER needs it - BOTH
        # halves of it. A slab samples every active row at one temperature AND
        # draws one seed for the whole batch, so `_mlx_batch_groups` breaks a
        # group when either the sampling or the take changes and a bucket is
        # homogeneous in both.
        entries = [(index, clean, positions, self._budget.cap_frames(clean),
                    rung_for(index), take_for(index))
                   for (index, clean), (_embeds, positions)
                   in zip(cleaned, prompts)]

        # THE GUARD'S RETAKES ARE BATCHED, NOT SERIAL (Owen, 2026-09-08: "take
        # note of which ones we need to re-render and add them to the task list,
        # and batch them at the end. instead of serializing every single one").
        # A batch of 32 rows costs about the wall time of ONE row here, so a
        # solo re-roll costs a whole batch: his Shift slice had 4 re-rolls in
        # ~28 chunks. The ladder is unchanged - `GuardPlan` is the same policy,
        # driven a round at a time (truncation.py, "ONE POLICY, THREE DRIVERS").
        groups = self._mlx_batch_groups(entries, ceiling)
        for group_no, (bucket, depth) in enumerate(groups, 1):
            texts = [e[1] for e in bucket]
            caps = [e[3] for e in bucket]
            for entry in bucket:
                if entry[0] not in held:
                    held.append(entry[0])
            try:
                rows_per_row = self._generate_delayed_rows_batch(
                    # The bucket's OWN take lane: every row in it shares one,
                    # which is what `_mlx_batch_groups` guarantees.
                    texts, caps,
                    self._request_seed(None, bucket[0][0], bucket[0][5]),
                    prompts=[prompt_by_index[e[0]] for e in bucket],
                    group_no=group_no, group_count=len(groups),
                    sampling=self._sampling_for(bucket[0][4]))
            except Exception as bucket_err:
                raise self._batch_failure(bucket, depth, bucket_err) from bucket_err
            for entry, codes in zip(bucket, rows_per_row):
                # The batched take is take 0 of the ladder: a row inside the
                # band is shipped as decoded, a row that stopped early or ran on
                # is RECORDED and rendered with the rest of this call's retakes.
                # MEASURED from the row's OWN frames against the row's OWN cap
                # (`entry[3]`) - a slab is generated to the deepest cap in its
                # bucket, so measuring against the bucket's depth would call
                # every short row in it capped.
                plan.add(entry[0], entry[1],
                         first_take=self.codec().decode(codes),
                         first_measure=FrameMeasure.counted_frames(len(codes),
                                                                   entry[3]))
            yield from self._decided(plan, held)
        yield from self._render_retake_rounds(plan, held, rung_for, take_for,
                                              ceiling)

    def _decided(self, plan, held):
        """Every chunk the ladder has decided since the last call, with the
        verdict it reached, and its index struck off `held`.

        `GuardPlan.verdict` POPS - a book is not a small number of chunks and
        the records are kept only until the driver has them - so this is the one
        place that reads it, immediately after the `finished()` that produced it.
        """
        for index, audio, _clean in plan.finished():
            if index in held:
                held.remove(index)
            yield index, audio, plan.verdict(index), plan.measure(index)

    #: A round advances every waiting chunk one rung, and the ladder is at most
    #: two rungs deep at each of `MAX_DEPTH` + 1 levels. A run past this is a
    #: ladder that does not terminate, which is a bug to see, not to survive.
    MAX_RETAKE_ROUNDS = 2 * (truncation.MAX_DEPTH + 1)

    def _render_retake_rounds(self, plan, held, rung_for, take_for,
                              ceiling=None):
        """The guard's retakes for this call, a ROUND at a time, each round in
        one batch; every chunk a round decides is yielded as
        `(index, audio, verdict)`.

        Round 1 is every off-length row's re-roll; round 2 is the split halves
        of the rows the re-roll did not fix, FLATTENED ACROSS CHUNKS so three
        chunks' six halves are one batch, not six solo renders; and so on to
        `MAX_DEPTH`. Rows land by index, so the ship order is unchanged.

        Nothing joins `held` here: a chunk on a retake round has been in flight
        since its take 0 and only leaves when `_decided` strikes it off.
        """
        for round_no in range(1, self.MAX_RETAKE_ROUNDS + 1):
            requests = plan.round()
            if not requests:
                return
            _log(f'length guard: retake round {round_no}, {len(requests)} take(s) '
                 f'for chunk(s) {sorted({request.index for request in requests})}')
            rendered = self._render_requests(requests, rung_for, take_for,
                                             ceiling)
            for request, (take, measure) in zip(requests, rendered):
                plan.offer(request, take, measure)
            yield from self._decided(plan, held)
        raise RuntimeError(
            f'Higgs v3 MLX: the length guard asked for a {self.MAX_RETAKE_ROUNDS + 1}th '
            f'retake round ({plan.pending} chunk(s) still on the ladder); the ladder '
            'is meant to terminate at MAX_DEPTH.')

    def _render_requests(self, requests: list, rung_for, take_for,
                         ceiling=None) -> list:
        """`truncation.RenderRequest`s rendered as batches; `(audio, measure)`
        per request, aligned to `requests`.

        ONE SEED PER BUCKET, as everywhere on this backend (`mx.random.seed` is
        drawn per batch): the bucket takes its first request's seed, which for a
        re-roll is `reroll_seed` - a seed no take 0 in this book used - and for
        a split half is that chunk's own, and either way it is shifted into
        that chunk's TAKE LANE (`_request_seed`), which is the lane every row
        in the bucket shares because `_mlx_batch_groups` breaks a group when
        the take changes. The text differs from take 0's either way, so the
        draw does too.
        """
        width = max(1, int(self.BATCH_SIZE or 1) if ceiling is None else int(ceiling))
        if width <= 1 or len(requests) == 1:
            return [self.render_audio_measured(
                        request.text, seed=request.seed,
                        index=request.index,
                        sampling=rung_for(request.index),
                        take=take_for(request.index))
                    for request in requests]
        # Keyed by POSITION, not by chunk index: the two halves of one chunk are
        # two requests carrying the same index, and they render side by side.
        prompts = self._mlx_prompts_for([request.text for request in requests])
        # THE RUNG IS THE CHUNK'S, read through `request.index`: two halves of
        # one chunk are two requests at one position each, and both must keep
        # their chunk's numbers. It rides on the entry so the slicer can break
        # a group on it - a retake round mixing two chunks' rungs would
        # otherwise land in one slab at one temperature.
        entries = [(position, request.text, positions,
                    self._budget.cap_frames(request.text),
                    rung_for(request.index), take_for(request.index))
                   for position, (request, (_embeds, positions))
                   in enumerate(zip(requests, prompts))]
        out = [None] * len(requests)
        groups = self._mlx_batch_groups(entries, ceiling)
        for group_no, (bucket, depth) in enumerate(groups, 1):
            first = requests[bucket[0][0]]
            seed = self._request_seed(first.seed, first.index, bucket[0][5])
            try:
                rows_per_row = self._generate_delayed_rows_batch(
                    [entry[1] for entry in bucket], [entry[3] for entry in bucket], seed,
                    prompts=[prompts[entry[0]] for entry in bucket],
                    group_no=group_no, group_count=len(groups),
                    sampling=self._sampling_for(bucket[0][4]))
            except Exception as bucket_err:
                # Named by CHUNK index, which is what a reader of the log has.
                named = [(requests[entry[0]].index,) + tuple(entry[1:]) for entry in bucket]
                raise self._batch_failure(named, depth, bucket_err) from bucket_err
            for entry, rows in zip(bucket, rows_per_row):
                # Against the ROW's own cap (`entry[3]`), not the bucket's
                # depth - see the same note on the take-0 slab above.
                out[entry[0]] = (self.codec().decode(rows),
                                 FrameMeasure.counted_frames(len(rows), entry[3]))
        return out

    @staticmethod
    def _batch_failure(bucket, depth, err) -> RuntimeError:
        """The ONE sentence a failed slice fails with, for both batch paths.

        It names the WIDTH and the ROW INDICES on top of the original error,
        because those are the two facts a single-row failure cannot carry and
        the two a reader needs to tell "this chunk is bad" from "this width is
        too wide". Raised, never swallowed - see `convert_batch`.
        """
        rows = [entry[0] for entry in bucket]
        return RuntimeError(
            f'Higgs v3 MLX: a batch of {len(bucket)} rows (depth {depth} positions, '
            f'rows {rows}) failed: {err}')

    def generate_batch_stream(self, texts, voices, stream_rows, on_chunk, on_row,
                              should_stop=None, samplings=None, takes=None) -> None:
        """Whole rows, at retirement - the honest cadence for this codec.

        `samplings`, when given, is aligned to `texts` and carries each row's
        take-ladder rung. Rung 1 (the streamed rows) renders solo, so it honours
        any rung; rung 2's batcher CANNOT mix - one slab, one sampler - so
        `_mlx_batch_groups` breaks a group when the rung changes and the
        read-ahead is split into homogeneous slabs rather than rendered at one
        row's numbers.

        `takes` is aligned the same way and is the rung's other half: which
        seed lane the row draws from. Rung 1 honours any take; rung 2's slab
        draws ONE seed, so a take change breaks a group exactly as a sampling
        change does.

        A row asked to stream gets its audio as a single `on_chunk(row, 0, pcm)`
        before its `on_row`: the streaming channel, filled with what there
        actually was, when there actually was it. See the module docstring for
        why a mid-row window is not sound here.

        A row abandoned because `should_stop` went true gets NO `on_row`, so a
        caller can never mistake an abandoned row for a finished one.

        THE LADDER: STREAM ROWS SOLO FIRST, THEN THE READ-AHEAD BATCHED
        --------------------------------------------------------------
        The Listen path sends ONE batch carrying the block being listened to
        (`stream_rows`) and the read-ahead behind it. Those two halves want
        opposite things, so they are rendered by different rungs:

          1. Every streamed row renders SOLO, in ascending row order, through
             `render_audio` - byte for byte the call the single-row path makes.
             This backend cannot token-stream (a delay-pattern codec has no
             windowed decode; see STREAMING in the module docstring), so a
             "streamed" row is a whole sentence delivered as one chunk, and its
             time-to-first-audio is its WHOLE generation. Batching it would add
             every other row's generation to the listener's wait for a word.
             Solo is the narrowest and fastest thing this backend has.
          2. Everything else - nobody is waiting on it yet - goes through the
             memory-budgeted batcher, and each row is decoded and handed over
             THE MOMENT IT RETIRES (`on_retire`), so a read-ahead sentence is
             not held hostage by the deepest row in its group.

        At the shipped default (`NARRATOR_HIGGS3_MLX_BATCH` = 1) rung 2 is a
        serial `render_audio` loop, which is byte for byte what this method did
        before batching existed. Nothing widens on its own.

        A non-streamed row gets `on_row` and NO `on_chunk`, on both rungs: the
        chunk channel is the streamed rows', and a chunk for a row the caller
        never asked to stream is an emission it has nowhere to put.

        A GROUP THAT RAISES IS NOT RETRIED. The exception propagates, carrying
        the group's width and row indices - which is exactly what a single-row
        failure does on the serial rung, and what the caller
        (`serve/worker.py::_generate_batch_streaming`) turns into a failed
        `batch_item` for every row it has not already answered. Rows already
        handed over stand. Narrowing and re-rendering here would be a silent
        fallback that hides the one thing worth learning: that this width does
        not work on this machine.
        """
        from ..item_sampling import aligned, takes_aligned
        if not texts:
            return
        if voices is not None and len(voices) != len(texts):
            raise ValueError(
                f'HiggsV3MlxEngine.generate_batch_stream: {len(voices)} voices for '
                f'{len(texts)} texts; voices must be aligned to texts or None')
        if voices is not None:
            wrong = sorted({v for v in voices if v and v != self.voice})
            if wrong:
                raise ValueError(
                    f'HiggsV3MlxEngine.generate_batch_stream: rows ask for voice(s) '
                    f"{wrong}; this engine holds '{self.voice}'. One loaded model "
                    'answers for one voice, so a mixed-voice batch is impossible.')
        stream_rows = set() if stream_rows is None else set(stream_rows)
        stray = [i for i in stream_rows if not (0 <= i < len(texts))]
        if stray:
            raise ValueError(
                f'HiggsV3MlxEngine.generate_batch_stream: stream_rows names row(s) '
                f'{stray} outside the batch of {len(texts)}')
        if stream_rows and on_chunk is None:
            raise ValueError(
                'HiggsV3MlxEngine.generate_batch_stream: stream_rows is non-empty but '
                'no on_chunk was given')
        blank = [i for i, t in enumerate(texts) if not (t or '').strip()]
        if blank:
            raise ValueError(
                f'HiggsV3MlxEngine.generate_batch_stream: row(s) {blank} have no text '
                'after cleaning.')
        rungs = aligned(samplings, len(texts),
                        'HiggsV3MlxEngine.generate_batch_stream')
        rows_takes = takes_aligned(takes, len(texts),
                                   'HiggsV3MlxEngine.generate_batch_stream')

        # ---- rung 1: the streamed rows, solo, in ascending order ----------
        for row in sorted(stream_rows):
            if should_stop is not None and should_stop():
                return
            audio = self.render_audio(texts[row], index=row,
                                      should_stop=should_stop,
                                      sampling=rungs[row], take=rows_takes[row])
            if audio is None:      # should_stop went true mid-row
                return
            on_chunk(row, 0, audio.copy())
            on_row(row, audio)

        read_ahead = [i for i in range(len(texts)) if i not in stream_rows]
        if not read_ahead:
            return
        if should_stop is not None and should_stop():
            # Before the prompts, not after: building them is a prefill's worth
            # of embedding work on rows nobody is waiting for any more.
            return

        # ---- rung 2: everything else ---------------------------------------
        width = max(1, int(self.BATCH_SIZE or 1))
        if width <= 1:
            # The unconfigured process, unchanged: one row at a time, in order.
            for row in read_ahead:
                if should_stop is not None and should_stop():
                    return
                audio = self.render_audio(texts[row], index=row,
                                          should_stop=should_stop,
                                          sampling=rungs[row], take=rows_takes[row])
                if audio is None:
                    return
                on_row(row, audio)
            return

        # The control-token allowlist, per row, BEFORE any prefill - the refusal
        # `render_audio` makes on the serial rung, made here where it does not
        # take a batch down mid-prefill.
        cleaned = {}
        for row in read_ahead:
            # THE SAME STRIP AS EVERY OTHER RENDER ENTRY - the packer's markers
            # ([break]/[heading]/[item]/[pause:X]) are stripped at the model
            # boundary here too; the solo and serial rungs get it through
            # render_audio, and a rung that built its prompt from the raw text
            # would read the markers aloud on the read-ahead alone.
            clean = self._clean_sentence_for_tts(texts[row])
            if not clean:
                raise ValueError(
                    f'HiggsV3MlxEngine.generate_batch_stream: row {row} has no '
                    f'spoken text after the marker strip: {texts[row]!r}')
            v3_served.validate_control_tokens(clean)
            cleaned[row] = clean

        # Built ONCE, because the SLICING needs the prompt POSITIONS: how deep a
        # batch can get is its prompt plus its frame cap.
        prompts = self._mlx_prompts_for([cleaned[row] for row in read_ahead])
        prompt_by_row = dict(zip(read_ahead, prompts))
        entries = [(row, cleaned[row], positions,
                    self._budget.cap_frames(cleaned[row]), rungs[row],
                    rows_takes[row])
                   for row, (_embeds, positions) in zip(read_ahead, prompts)]
        groups = self._mlx_batch_groups(entries)
        _log(f'stream batch: {len(stream_rows)} row(s) solo, {len(read_ahead)} '
             f'read-ahead in {len(groups)} group(s) of '
             f'{[len(bucket) for bucket, _d in groups]}')

        for group_no, (bucket, depth) in enumerate(groups, 1):
            rows = [entry[0] for entry in bucket]

            def _retire(position, codes, _rows=rows):
                """One read-ahead row, decoded and delivered AT RETIREMENT.
                `position` indexes this group; `_rows` maps it back to the
                caller's row index."""
                on_row(_rows[position], self.codec().decode(codes))

            try:
                out = self._generate_delayed_rows_batch(
                    [entry[1] for entry in bucket],
                    [entry[3] for entry in bucket],
                    self._request_seed(None, rows[0], bucket[0][5]),
                    should_stop=should_stop,
                    prompts=[prompt_by_row[row] for row in rows],
                    group_no=group_no, group_count=len(groups),
                    on_retire=_retire,
                    sampling=self._sampling_for(bucket[0][4]))
            except Exception as bucket_err:
                raise self._batch_failure(bucket, depth, bucket_err) from bucket_err
            if out is None:
                # should_stop went true mid-batch. Rows that had already retired
                # were handed over by `_retire`; the abandoned ones get nothing,
                # which is the contract this method has always kept.
                return


# ---------------------------------------------------------------------------
# The registry's factory
# ---------------------------------------------------------------------------


def higgs_v3_mlx_config_from_worker_kwargs(voice=None, model_dir=None,
                                           base_dir=None, adapter_dir=None,
                                           caps=None, **extra):
    """Build a HiggsV3MlxConfig from the keywords `narrator.serve` hands an
    engine. The refusals are `resolve_load_voice`'s, at the other boundary."""
    from .config import load_voice
    HiggsV3MlxEngine.resolve_load_voice(
        voice, model_dir=model_dir, adapter_dir=adapter_dir, base_dir=base_dir)
    if caps:
        raise ValueError(
            f'Higgs v3 load carried caps={sorted(caps)}. Those are ORPHEUS tuning '
            'caps and v3 implements none of them.')
    if extra:
        raise ValueError(f'Higgs v3 load carried unknown keys: {sorted(extra)}.')
    resolved = load_voice(
        voice.strip(),
        allowed_controls=HiggsV3Defaults.ALLOWED_CONTROLS,
        max_reference_seconds=HiggsV3Defaults.MAX_REFERENCE_SECONDS,
        placeholder_max_chars=HiggsV3Defaults.MAX_CHARS)
    # A fine-tuned voice's WEIGHTS are the voice, so the model loaded is the
    # voice's own checkpoint. A zero-shot voice may name a directory too - the
    # BASE weights, pinned by whoever wrote the document - and that is the
    # model this process loads; `model_dir_from_env()` is for a voice that
    # names none at all.
    named = v3_served.voice_serve_target(resolved)
    # THE VOICE'S OWN SAMPLING, when the document states one, over the
    # checkpoint file's - see `mlx_sampling`. None keeps the file's values.
    return HiggsV3MlxConfig(voice=resolved,
                            model_dir=named or model_dir_from_env(),
                            # THE WINDOW THIS LOAD ASKS FOR. One external name
                            # across all three backends (`HIGGS_CONTEXT_LENGTH`,
                            # set per voice in the spawn env); on this arm there
                            # is no server to launch, so it lands straight on
                            # the config the engine bounds prompts against.
                            # Unset = HiggsV3Defaults.CONTEXT_TOKENS, exactly
                            # what this factory produced before.
                            context_tokens=served_common.context_length(
                                HiggsV3Defaults.CONTEXT_TOKENS),
                            sampling=getattr(resolved, 'sampling', None))
