"""narrator — the managed subprocess that is to `tts` what vLLM is to `llm`.

PHASE3-TTS.md section 4. Crucible does not reimplement Higgs's frame budget,
its guard or its codec arithmetic; it runs the code that already has them.
`python/narrator` in the BookForge repo is an installable
package whose `serve` entry point loads a voice once and answers sentence
requests over stdin and stdout, and this file is the client for that wire.

Three things make it unlike every engine before it, and each one is a seam
somewhere else in this package rather than a special case here:

- **Readiness is a line, not a route.** narrator prints
  `{"type": "ready", "device", "backend"}` on stdout when the process is up.
  `SubprocessEngine.announced_ready()` is the seam; the default `/v1/models`
  poll is untouched and neither HTTP engine overrides it.
- **The wire is the pipes.** `SubprocessEngine.stdio()` is the seam: narrator
  keeps stdin and stdout as text-mode pipes and sends only stderr to
  `~/.crucible/logs/engine-<voice>.log`. Everything the HTTP engines do is
  unchanged.
- **Nothing else can reach it.** vLLM binds a loopback port and the proxy talks
  to it; narrator binds nothing, so this object *is* the channel. Hence
  `base_url` refuses rather than returning a port nothing is listening on.

The correlation problem, and why this file does not solve it
------------------------------------------------------------
`generate_batch` retires rows **out of order** — a short row finishes while a
long one is still generating, and `tests/fake_narrator.py` retires in reverse on
purpose so that a consumer relying on arrival order fails in the suite rather
than on a book. The row's identity is the `i` narrator echoes back, and it is the
*caller's* number: the render door sends its chunk indices as `i` and reads them
straight off each `batch_item`.

So `converse()` yields every line as it lands and **reorders nothing**. Buffering
into caller order here would defeat the two things the shape exists for: the
render door writes each FLAC as its row retires, overlapped with the next row's
generation, and the streaming door (section 7) needs `batch_chunk` lines out of
the same iterator *while* a row is still generating. One reader, one iterator,
and the consumer keys on `i`.

What the reader thread does and does not swallow
------------------------------------------------
fd 1 is the wire and nothing else — the same rule, learned from the same
incident, as `crucible/workers.py`: narrator's aligner once had a library log to
stdout on a 401-chunk book and corrupt the result stream. A line on stdout that
is not a JSON object carrying a `type` is therefore a **refusal naming the
line**, never something skipped. narrator's own diagnostics go to stderr and
into the log, which is where a reader should look for them.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

from ..errors import JobCancelled
from .base import LOG_TAIL_LINES, EngineError, SubprocessEngine

if TYPE_CHECKING:  # `crucible.narratorvoices` imports this package; no cycle at runtime
    from ..narratorvoices import VoicesDocument

#: How narrator is started. It takes no configuration on the command line — its
#: whole interface is the protocol on stdin and stdout plus the environment — so
#: the argv is this and nothing else. Which engine it serves is `NARRATOR_ENGINE`
#: below, which is the variable narrator's own `engine_id()` reads.
MODULE = "narrator.serve"

#: The environment variable that decides which of narrator's engines a process
#: serves. narrator refuses an unknown value by name at start-up rather than
#: defaulting, which is why Crucible may set it and then trust the `ready` line.
ENGINE_VARIABLE = "NARRATOR_ENGINE"

#: WHAT A `higgs-v3` WORKER NEEDS BESIDES ITS ENGINE ID, and every one of them
#: is refused BY NAME by narrator rather than defaulted. Crucible's first real
#: `tts` render (2026-09-13) died on the first of them — `narrator (higgs-v3)
#: exited 3` before `ready`, `HIGGS_STACK is not set` — because `environment()`
#: named only NARRATOR_ENGINE and PYTHONUNBUFFERED.
#:
#:   HIGGS_STACK         which serving stack this process renders on.
#:                       `served_common.serving_stack()` raises when it is
#:                       unset, and is called from `HiggsV3Engine
#:                       .detect_backend()` — a CLASSMETHOD `narrator.serve`
#:                       calls before any voice loads, which is why the worker
#:                       dies before `ready` rather than on the first chunk.
#:                       The two stacks place sampling differently and size the
#:                       frame cap against different context windows, so a
#:                       guessed one is a book rendered at sampling nobody
#:                       chose. STATED FROM THE ENV SPEC (`jobenv.tts_env`),
#:                       because the stack is a property of what the recipe
#:                       installed, not of the voice.
#:   HIGGS_ENV           the prefix the SERVER runs out of. narrator's packaged
#:                       `serve_higgs_v3.sh` builds CUDA_HOME, PATH,
#:                       LD_LIBRARY_PATH and `$HIGGS_ENV/bin/vllm-omni` from it
#:                       and refuses (exit 5) when it is unset. For Crucible it
#:                       is THE TTS ENV ITSELF — `higgs_env_prefix()` below,
#:                       which derives it from the interpreter Crucible was
#:                       handed and confirms it by reading what is on disk
#:                       there.
#:   HIGGS_MAX_NUM_SEQS  stage 0's admission width AND the width of narrator's
#:                       own batch (`v3_served.serve_concurrency()`, which
#:                       raises by name). STATED FROM THE VOICE MANIFEST's
#:                       `[voice.serving]`.
#:
#: NARRATOR_HIGGS3_MLX_BATCH is the same question on the OTHER arm and it is
#: below, at `MLX_TIERS`, not here: the three above are the served arm's
#: and this one is `mlx-darwin`'s, where narrator starts no server and batches
#: in process. It is not stated from the voice manifest — `max_num_seqs` is a
#: vLLM stage-0 admission width measured on a 24 GB card and means nothing to a
#: Metal backend — but from a table of widths measured on THIS arm.
#:
#: NARRATOR_HIGGS3_SERVE_SCRIPT is deliberately NOT here. narrator ships its own
#: launcher as package data as of BookForge 0eeb0267 and runs it when no
#: override is named; an operator's path into somebody's checkout is exactly
#: what that commit removed the need for.
#:
#: THE LAUNCHER'S OTHER SIX KNOBS ARE UNSET HERE, AND THAT IS INERT RATHER THAN
#: LOST — but it is inert only because two files agree, and an agreement nothing
#: compares is the shape `docs/ARCHITECTURE.md` was written about. So the
#: comparison is written down. BookForge states each of these from its catalog
#: (`electron/data/higgs-models.json`'s shared `serving` block, through
#: `higgsSpawnEnv`); Crucible states none of them and takes the default in
#: narrator's own `engine/higgs/launch/serve_higgs_v3.sh`. The two columns are
#: the same numbers, and they are the same numbers ON PURPOSE: that script's
#: defaults were set to the measured catalog values on 2026-09-13 precisely so
#: a caller who has only read the file is correct.
#:
#:     variable                  launcher default         catalog value
#:     HIGGS_HOST                127.0.0.1                127.0.0.1
#:     HIGGS_PORT                8095                     8095
#:     HIGGS_GPU_MEM_UTIL        0.35                     0.35
#:     HIGGS_CODEC_GPU_MEM_UTIL  0.10                     0.1
#:     HIGGS_MAX_MODEL_LEN       8192                     8192
#:     HIGGS_DEPLOY_CONFIG       the packaged             higgs_default_
#:                               higgs_default_           frames7500.yaml
#:                               frames7500.yaml          (the same bytes)
#:
#: THE LAST ROW IS THE ONE THAT WOULD HAVE HURT. Unset meant "vllm-omni's own
#: auto-discovered profile" until 2026-09-13, and that profile caps stage 0 at
#: 2048 frames = 81.92 s — every long chunk cut mid-sentence, with the request
#: reporting success. It is now `${VAR-<sibling>}` (the `-` form, not `:-`), so
#: silence means the certified frames-7500 profile and the auto one stays
#: reachable as the empty string. A reader who finds this table stale should
#: fix the table, not add six variables: the launcher is the owner, and a
#: second statement of a number is how the two come to disagree.
#:
#: NARRATOR_HIGGS_VOICES (and, on the MLX arm, NARRATOR_HIGGS3_MLX_MODEL) is
#: the fourth thing a `higgs-v3` worker needs, ON BOTH ARMS, and it is not a
#: constant here because it is not a value: it is the PATH of a document
#: Crucible writes at every load, `crucible/narratorvoices.py`, handed to this
#: engine at construction as `voices`. Crucible's first real render found it
#: on the served arm and the keeper found it again on the Mac (2026-09-14):
#: narrator resolves a Higgs v3 voice BY NAME in that document and refuses a
#: `modelDir` on the `load` message by name, so a worker with no document
#: cannot load any voice at all.
STACK_VARIABLE = "HIGGS_STACK"
MAX_NUM_SEQS_VARIABLE = "HIGGS_MAX_NUM_SEQS"

#: SGLANG'S `--mem-fraction-static`, from `[voice.serving].mem_fraction`.
#: narrator's `engine/higgs/launch/serve_higgs_sgl.sh:59` reads it and defaults
#: it to 0.60, so an unset variable is that script's stated number rather than
#: an absence. See `crucible/voices.py:_SERVING_OPTIONAL` for the measurement
#: that made it a field.
MEM_FRACTION_VARIABLE = "HIGGS_SGL_MEM_FRACTION"

#: THE ENGINE'S CONTEXT IN TOKENS, from `[voice.serving].context_length`.
#: **No narrator on any pin reads this yet** — see `environment()`, which says
#: so at the line that sets it — and the name is the one narrator is growing the
#: reader under. Stated here so the two sides cannot pick different spellings.
CONTEXT_LENGTH_VARIABLE = "HIGGS_CONTEXT_LENGTH"

#: THE ENV PREFIX VARIABLE IS THE STACK'S, NOT ONE NAME FOR BOTH — and each
#: launcher reads ONLY its own.
#:
#: `serve_higgs_v3.sh` builds CUDA_HOME, PATH, LD_LIBRARY_PATH and
#: `$HIGGS_ENV/bin/vllm-omni` out of `HIGGS_ENV` and exits 5 when it is unset.
#: `serve_higgs_sgl.sh` does the identical job out of `HIGGS_SGL_ENV` — and
#: DEFAULTS IT to `$HOME/anaconda3/envs/sglomni` rather than refusing, because
#: the script is also run by hand on the machine it was transcribed from.
#:
#: THAT DEFAULT IS WHY THIS IS A TABLE AND NOT A CONSTANT. Sending the SGLang
#: launcher `HIGGS_ENV` would set a variable it never reads, leave
#: `HIGGS_SGL_ENV` unset, and send it looking for `sgl-omni` inside a conda env
#: that does not exist on a Crucible host — `exit 5`, several minutes after a
#: load began, naming a directory nobody configured. BookForge's `higgsSpawnEnv`
#: branches on exactly this and emits `HIGGS_SGL_ENV` + `NARRATOR_HIGGS_SGL_
#: SERVE_SCRIPT` on one arm and `HIGGS_ENV` + `NARRATOR_HIGGS3_SERVE_SCRIPT` on
#: the other, with the comment that nothing from the vllm-omni half comes along.
#:
#: The BINARY is per stack for the same reason: it is the most direct evidence
#: on disk that a directory is the tree the stack was installed into, and the
#: two stacks install different ones.
STACK_ENV_PREFIX_VARIABLE: dict[str, str] = {
    "vllm-omni": "HIGGS_ENV",
    "sglang-omni": "HIGGS_SGL_ENV",
}
STACK_LAUNCH_BINARY: dict[str, str] = {
    "vllm-omni": "vllm-omni",
    "sglang-omni": "sgl-omni",
}

#: The vllm-omni arm's name, kept as a module constant because `crucible doctor`
#: and the tests refer to it and because it is what `higgs_env_prefix`'s refusal
#: says when no stack is in hand.
ENV_PREFIX_VARIABLE = STACK_ENV_PREFIX_VARIABLE["vllm-omni"]


def env_prefix_variable_for(serving_stack: str) -> str:
    """`HIGGS_ENV` or `HIGGS_SGL_ENV`, or a refusal naming the stack.

    No default. A stack this build does not know is a launcher whose variables
    nobody here has read, and guessing one of the two would configure the wrong
    server — or, on the SGLang arm, no server at all while its own hardcoded
    conda default takes over.
    """
    variable = STACK_ENV_PREFIX_VARIABLE.get(serving_stack)
    if variable is None:
        raise EngineError(
            f"no env-prefix variable for serving stack {serving_stack!r}; this "
            f"build knows {sorted(STACK_ENV_PREFIX_VARIABLE)}. Each launcher "
            "reads only its own name for the prefix it runs out of, and the "
            "SGLang one DEFAULTS to a conda env rather than refusing — so a "
            "guess here is a server started out of a directory nobody named"
        )
    return variable

#: The narrator engine those three belong to. Written as a constant so the
#: refusals below read as a rule rather than as a special case: they are the
#: shape a narrator engine that reads no `HIGGS_*` variable would arrive into,
#: and since Owen's ruling of 2026-09-14 (`voices.NARRATOR_ENGINE_SAMPLING`)
#: `higgs-v3` is the only engine Crucible names at all.
HIGGS_V3 = "higgs-v3"

#: THE IN-PROCESS ARM'S RENDER WIDTH — `mlx-darwin`'s counterpart of
#: `HIGGS_MAX_NUM_SEQS`, and the one variable this file used to leave unset on
#: that arm.
#:
#: WHAT IT COST TO LEAVE IT UNSET (measured on owens-mac-studio, 2026-09-15).
#: narrator's `HiggsV3MlxEngine.BATCH_SIZE` is `mlx_batch_ceiling()`, which is
#: **1 unless this variable asks for more** — "so an unconfigured process
#: renders exactly as it did single-row" (`engine/higgs/mlx_backend.py`). At 1,
#: `render_many` takes `_render_many_serial` and the Mac renders one chunk at a
#: time. BookForge's own darwin worker has always asked for a width
#: (`electron/higgs-spawn.ts:higgsMlxBatchEnv` -> the memory tier's 64) and
#: Crucible never learned to, so every render that moved from the app to this
#: server silently lost the batch. Owen's `thirdreich` book was running at
#: 2.9 chunks/min where the same machine did ~130 raw sent/min in September
#: (BookForge 626980a2, deathstalker at MLX batch 62).
#:
#: THE CURVE, one voice (`thirdreich`), one corpus (16 and 64 chunks of 521-572
#: chars, the voice's own 500-700 band), one resident load per run, this env's
#: own interpreter:
#:
#:     width  1   1,799 chars/min   1.99x realtime   (what this file shipped)
#:     width 16   6,749 chars/min   7.43x realtime   3.75x
#:     width 32   9,662 chars/min  10.78x realtime   5.37x
#:     width 64  12,579 chars/min  13.97x realtime   6.99x
#:
#: 64 IS THE NUMBER FOR THE SAME REASON 16 IS THE SERVED ARM'S: it is the width
#: the throughput was measured at, and it is the width BookForge's Mac has been
#: asking for since 2026-09-05. It is a CEILING and not an allocation —
#: narrator's `_mlx_width_for_depth` narrows it against
#: `NARRATOR_HIGGS3_MLX_MEM_BUDGET_GB` when a slice is deep (Owen's Streicher
#: render ran 62 of the 64 it asked for), so asking for 64 is asking the
#: budgeter to do its job rather than promising the memory.
#:
#: A TABLE KEYED BY ENGINE, like `ttsstream.STREAM_BATCH_WIDTH`, and for its
#: reason: the numbers must be MEASURED and there is deliberately no default. A
#: second narrator engine on this arm adds a row after somebody measures it;
#: until then `mlx_render_profile` refuses it by name, because a guessed width
#: is wrong in both directions — too low renders a book one chunk at a time
#: while every variable looks configured, too high asks for memory the machine
#: does not have.
#:
#: THE WIDTH IS NOT ONE NUMBER, AND 64 ALONE WAS HALF A DECISION. The first fix
#: for the 7x (2026-09-15, `MLX_RENDER_WIDTH = {"higgs-v3": 64}`) copied the
#: CONSTANT BookForge's 64 GB Mac happens to ask for and left the budget at
#: narrator's own 42 GB default — right on that machine and wrong on any other,
#: which is `docs/ARCHITECTURE.md`'s one-fact-two-owners exactly. BookForge
#: never hardcoded 64: `electron/higgs-spawn.ts:higgsMlxBatchEnv` reads
#: `orpheusMemoryProfile(resolveConcreteOrpheusTier(null, null))` and emits the
#: WIDTH and the MEMORY BUDGET out of the same tier row, in the same breath,
#: because narrator narrows a deep batch's width against that budget
#: (`_mlx_width_for_depth`). So the row is ported here whole, and the tier is
#: chosen the way BookForge's `orpheusAutoSuggestion` chooses it on darwin with
#: no VRAM to read: by BANDS of the machine's own total memory.
#:
#: THE BANDS AND THE ROWS ARE BookForge'S, TRANSCRIBED (`electron/
#: orpheus-memory.ts`, `MLX_TIERS` + the darwin branch of
#: `orpheusAutoSuggestion`). They are measured on an M1 Ultra against real book
#: sentences and nothing here re-derives them:
#:
#:     >= 60 GiB   extreme   width 64   budget 42 GB   cache 8 GB
#:     >= 44 GiB   fast      width 72   budget 34 GB   cache 8 GB
#:     >= 28 GiB   moderate  width 48   budget 22 GB   cache 6 GB
#:      < 28 GiB   light     width 24   budget 13 GB   cache 3 GB
#:
#: `fast` IS WIDER THAN `extreme` AND THAT IS NOT A TYPO. BookForge's extreme
#: row went 96 -> 64 on 2026-09-01 (Owen's call) because continuous batching
#: keeps the KV cache at full depth permanently, so the 55 GB "worst case"
#: became the steady state; the budget came down 55 -> 42 with it. `fast` was
#: never re-measured and keeps its 72. Transcribing the table means transcribing
#: that, not tidying it into a monotone.
#:
#: WHAT CRUCIBLE CANNOT LEARN, and does not pretend to. BookForge's tier is also
#: a SETTING (Settings -> memory tier) and carries an `autoCeiling` that
#: ratchets DOWN after an out-of-memory failure, persisted per machine in
#: `orpheus-memory.json`. A Crucible server has neither, so this is the band
#: alone — the same answer BookForge gives on a fresh install with the tier left
#: on `auto`. An operator who needs a different one sets
#: `NARRATOR_HIGGS3_MLX_BATCH` in the environment narrator inherits, which is
#: the override BookForge honours too.
#:
#: THIS IS THE RENDER DOOR'S WIDTH, AND THE STREAMING DOOR IS NOT AN EXCEPTION
#: TO IT — it just does not use it. BookForge runs TWO processes and gives them
#: two values: the audiobook worker gets the tier's width, and the Listen
#: server gets `orpheus-worker-pool.ts`'s `streamBatchCeiling()`, which for
#: Higgs is `HIGGS_STREAM_BATCH_WIDTH = 1` — measured 2026-09-11, where a 4-row
#: group was exactly as fast as four solo rows and cost the listener a hole
#: after the first sentence. Crucible runs ONE resident engine for both doors,
#: so it cannot hold two values, and it does not need to: the streaming door's
#: width is the number of ROWS it hands `generate_batch`
#: (`ttsstream.STREAM_BATCH_WIDTH`, which is that same measured 1), and
#: narrator's read-ahead batches only over the rows it was given. A ceiling of
#: 64 with one row in hand is one row. The two would part company only if that
#: table were ever raised, and raising it is a measurement, not an edit.
MLX_BATCH_VARIABLE = "NARRATOR_HIGGS3_MLX_BATCH"
MLX_MEM_BUDGET_VARIABLE = "NARRATOR_HIGGS3_MLX_MEM_BUDGET_GB"
MLX_CACHE_LIMIT_VARIABLE = "HIGGS_MLX_CACHE_LIMIT_GB"

#: The resident Higgs v3 weights on the MLX arm, as narrator's own backend
#: counts them (`HiggsV3MlxEngine.MLX_WEIGHTS_GB`). Read here for ONE purpose —
#: proving a row's budget can hold the weights and the pinned cache before a
#: process exists — and never to size anything: the sizing is narrator's.
HIGGS_V3_MLX_WEIGHTS_GB = 8.5


@dataclass(frozen=True)
class MlxTier:
    """One row of the transcribed table: a memory band and what it buys.

    THE THREE NUMBERS TRAVEL TOGETHER. narrator's headroom arithmetic is
    `budget - weights - cache` (`_mlx_kv_headroom_gb`), so a budget taken from
    one row while the cache keeps narrator's default is two rows' worth of
    memory in one sum — which is how a plausible pair produces a load that
    refuses. They are one row here for the same reason the width and the budget
    are one decision.
    """

    name: str
    #: The band's floor, in MiB of total memory, compared the way BookForge
    #: compares it (`os.totalmem()` rounded to MiB against 60_000 / 44_000 /
    #: 28_000). The last row's floor is 0: it is what is left.
    min_total_mib: int
    #: `NARRATOR_HIGGS3_MLX_BATCH` — a CEILING to ask for, never an allocation.
    #: narrator narrows it per slice (`_mlx_width_for_depth`); Owen's Streicher
    #: render ran 62 of the 64 it asked for.
    width: int
    #: `NARRATOR_HIGGS3_MLX_MEM_BUDGET_GB` — the whole batch's unified-memory
    #: budget, weights and pinned cache included.
    mem_budget_gb: float
    #: `HIGGS_MLX_CACHE_LIMIT_GB` — the pinned MLX buffer cache. BookForge emits
    #: only two of its row's three numbers for Higgs and leaves this at
    #: narrator's 8 GB default; that is fine on the 64 GB Mac (the row says 8)
    #: and is arithmetically impossible on the `light` row, where 13 - 8.5 - 8
    #: is negative and narrator refuses the load by name. All three come off the
    #: row here, which is the port rather than a departure from it.
    cache_limit_gb: float


MLX_TIERS: dict[str, tuple[MlxTier, ...]] = {
    HIGGS_V3: (
        MlxTier("extreme", 60_000, 64, 42.0, 8.0),
        MlxTier("fast", 44_000, 72, 34.0, 8.0),
        MlxTier("moderate", 28_000, 48, 22.0, 6.0),
        MlxTier("light", 0, 24, 13.0, 3.0),
    ),
}


def mlx_render_profile(narrator_engine: str, total_bytes: int) -> MlxTier:
    """The `MLX_TIERS` row this machine's memory selects, or a refusal.

    No default anywhere, for `ttsstream.batch_width_for`'s reason written one
    arm over: an engine nobody has measured a render width for is an engine
    whose batching is unknown, and 1 is not a safe answer — it is the answer
    that cost this server a 7x. Nor is the MACHINE guessed at: a Mac whose
    memory Crucible could not read is refused here rather than handed the 64 GB
    machine's row.
    """
    rows = MLX_TIERS.get(narrator_engine)
    if rows is None:
        raise EngineError(
            f"no measured MLX render tiers for narrator engine "
            f"{narrator_engine!r}; this build knows {sorted(MLX_TIERS)}. "
            f"{MLX_BATCH_VARIABLE} is the width narrator's in-process backend "
            "batches at and it defaults to 1 — one chunk at a time — so "
            "leaving it unset is a measured 7x, not a safe fallback"
        )
    if not isinstance(total_bytes, int) or total_bytes <= 0:
        raise EngineError(
            f"cannot choose an MLX render tier for {narrator_engine!r} from "
            f"total_bytes={total_bytes!r}: the tier is a BAND of this "
            f"machine's own memory ({MLX_BATCH_VARIABLE} and "
            f"{MLX_MEM_BUDGET_VARIABLE} come out of the same row), and "
            "`accelerator.probe_unified_memory` is what reads it. A machine "
            "whose memory could not be read gets no row — the 64 GB Mac's is "
            "not a safe stand-in for a 16 GB one"
        )
    total_mib = round(total_bytes / (1024 * 1024))
    for row in rows:
        if total_mib >= row.min_total_mib:
            break
    else:  # unreachable: the last row's floor is 0
        raise EngineError(
            f"the {narrator_engine!r} MLX tier table has no row for "
            f"{total_mib} MiB; its last row must have a floor of 0"
        )
    headroom = row.mem_budget_gb - HIGGS_V3_MLX_WEIGHTS_GB - row.cache_limit_gb
    if headroom <= 0:
        # THE TABLE ITSELF IS WRONG, and it is caught here rather than at a
        # load. narrator raises the same sum from inside the process
        # (`_mlx_kv_headroom_gb`) after it has read 8.5 GB of weights off disk;
        # a row that cannot hold its own weights and cache is a row nobody can
        # render on, and saying so before the process exists names the row.
        raise EngineError(
            f"the {narrator_engine!r} MLX tier {row.name!r} budgets "
            f"{row.mem_budget_gb:g} GB, which cannot hold "
            f"{HIGGS_V3_MLX_WEIGHTS_GB:g} GB of weights plus a "
            f"{row.cache_limit_gb:g} GB pinned buffer cache "
            f"({MLX_CACHE_LIMIT_VARIABLE}) — narrator refuses that sum at load "
            "and there is no KV left to batch with"
        )
    return row

#: How long `stop()` gives the `quit` action before falling back on SIGTERM.
#: narrator's teardown releases CUDA from inside the process, and on a loaded
#: SGLang-Omni that takes seconds rather than milliseconds.
QUIT_GRACE_SECONDS = 30.0

#: How often the conversation loop wakes to notice a cancel, a dead process or a
#: missed deadline. Not a timeout on anything: a row that takes ninety seconds is
#: a row that is being generated.
POLL_SECONDS = 0.5

#: How long a cooperative cancel may go UNANSWERED before the engine is declared
#: unstoppable. Measured on the Mac, 2026-09-15: a `tts` render of `thirdreich`
#: was cancelled at chunk 38 of 89, the door recorded `cancelling`, this file
#: sent `{"action": "cancel"}` — and narrator went on retiring rows for eleven
#: more minutes, because the arm Crucible's render door drives
#: (`serve/worker.py:_emit_guarded_batch`, the `render_many` ladder) never read
#: the flag its stdin reader had set. Every other arm did.
#:
#: NOT A SILENCE TIMEOUT, and it is the opposite case to
#: `RENDER_SILENCE_TIMEOUT_SECONDS`. A silence timeout asks "is this process
#: alive"; this asks "did it hear me". narrator was talking the whole time — a
#: `batch_item` every twenty seconds, each one resetting the silence clock — so
#: no liveness check could ever have ended it.
#:
#: THE CLOCK STARTS AT THE CANCEL AND IS NOT RESET BY A LINE, for the same
#: reason. It is generous enough that a narrator which honours the cancel
#: between chunks always beats it (one chunk's render is ~20 s on MLX, and a
#: chunk part-way up the retake ladder may be a second render behind that), and
#: short enough that the operator who pressed stop is not waiting on a book.
CANCEL_GRACE_SECONDS = 120.0

#: How long a `load` may go without narrator saying anything at all. It is a
#: SILENCE timeout and any line resets it, the same discipline `crucible/
#: workers.py` uses: narrator on `cuda-linux` starts SGLang-Omni underneath
#: itself, which is minutes of weight reading before the `loaded` line.
LOAD_SILENCE_TIMEOUT_SECONDS = 900.0

#: The binary `serve_higgs_v3.sh` execs, relative to `$HIGGS_ENV`. The launcher
#: also builds `$HIGGS_ENV/lib/python3.11/site-packages/nvidia/cu13` as
#: CUDA_HOME and puts `$HIGGS_ENV/bin` on PATH — so the prefix is the tree the
#: STACK is installed into, and this path is the most direct evidence on disk
#: that a directory is that tree.
#: The vllm-omni arm's, kept for callers with no stack in hand. The per-stack
#: answer is `STACK_LAUNCH_BINARY`, which is what `higgs_env_prefix` reads.
LAUNCH_BINARY = ("bin", STACK_LAUNCH_BINARY["vllm-omni"])


def higgs_env_prefix(python: Path, serving_stack: str) -> Path:
    """`$HIGGS_ENV` / `$HIGGS_SGL_ENV` for a tts env, READ off the env.

    THE DEFECT THIS EXISTS FOR (live WSL server, 2026-09-14): every `tts` job
    failed at engine start with `HIGGS_ENV is the prefix its server runs out
    of, and the tts env python .../envs/tts-higgs-v3/bin/python does not sit
    in one — /home/telltale/anaconda3/envs/crucible/pyvenv.cfg is not there`.
    Two mistakes, one line (`Path(python).resolve().parent.parent`, a7ab9af):

    1. **`.resolve()` walked out of the env.** `workerenv.install_worker_env`
       builds the env with `sys.executable -m venv`, and a venv's `bin/python`
       is a SYMLINK to the interpreter it was built from — here the conda env
       the server itself runs in. Resolving it therefore lands on the BASE
       interpreter's prefix, which is not the env Crucible installed vllm-omni
       into. The prefix is where the env IS, so the symlink is not followed.
    2. **`pyvenv.cfg` was made the definition of a prefix.** A conda env has
       none (it has `conda-meta/`), and a downloaded python-build-standalone
       tree (`crucible/interpreter.py`) has neither — so the check refused
       two of the three layouts Crucible can be pointed at, and the one it
       refused tonight was the base of the very venv it had just walked into.

    What is read, in the order the evidence answers the launcher's question:

    * `bin/<the stack's server>` — the file the script execs, `vllm-omni` or
      `sgl-omni`. Definitive on all three layouts, and the only one that says
      the STACK is here and not merely a python. It is the stack's OWN binary
      since 2026-09-15: looking for vllm-omni inside an SGLang env would find
      nothing and fall through to the weaker checks below, so the one piece of
      evidence that actually distinguishes a stack tree from a bare venv would
      never fire on the stack Owen renders on.
    * `pyvenv.cfg` — a venv. Its own prefix, never its parent's.
    * `conda-meta/` — a conda env, which is its own prefix.

    Anything else is refused BY NAME here rather than at the end of a launch,
    where it reads as `$HIGGS_ENV/bin/vllm-omni: No such file` — or, on the
    SGLang arm, as a launcher quietly taking its own hardcoded conda default.
    """
    variable = env_prefix_variable_for(serving_stack)
    binary = STACK_LAUNCH_BINARY[serving_stack]
    root = Path(python).parent.parent
    if (root / "bin" / binary).exists():
        return root
    if (root / "pyvenv.cfg").is_file():
        return root
    if (root / "conda-meta").is_dir():
        return root
    raise EngineError(
        f"{variable} is the prefix narrator's server runs out of, "
        f"and {root} — the prefix of the tts env python {python} — is not "
        f"one: it carries no bin/{binary}, no pyvenv.cfg (a venv) "
        "and no conda-meta/ (a conda env). narrator's launch script builds "
        f"CUDA_HOME, PATH, LD_LIBRARY_PATH and the {binary} binary from that "
        "prefix and refuses when it is unset"
    )


class EngineWouldNotStop(EngineError):
    """narrator was sent a cancel, kept working, and outlasted the grace.

    An `EngineError` on purpose, so every `except EngineError` already written
    against this wire keeps its meaning: the streaming door closes the session
    and names the engine (`ttsstream.py`), and the settlement then takes the
    card back through the one unload door. What the subclass adds is a caller
    that wants to say something sharper — the render door, which turns it into
    the cancel the operator asked for rather than into a failed render, and
    which takes the voice off the card BY NAME rather than leaving it resident
    behind a lease that would otherwise keep it there.

    IT IS THE ENGINE'S FAULT AND IT IS NAMED AS SUCH. A process that reads a
    cancel (its stdin reader does, and sets a flag) and then renders another
    fifty chunks is not a slow engine, it is one whose flag nothing reads. The
    only honest thing to do with it is to stop it.
    """


@dataclass(frozen=True)
class _Garbled:
    """A line on stdout that is not a message. Carried, not dropped."""

    line: str


class _Ended:
    """narrator's stdout reached end of file."""


_ENDED = _Ended()


class NarratorEngine(SubprocessEngine):
    """`python -m narrator.serve`, and the JSON-lines channel to it.

    One instance per resident voice, built by `build_voice_engine()` from the
    voice manifest's `narrator_engine`. The name carries the engine id, so a log
    line, a timeout and a refusal all say `narrator (higgs-v3)` rather than
    `narrator` — on `cuda-linux` there are two envs and two engines and the
    question a reader has is always which.
    """

    def __init__(
        self,
        narrator_engine: str,
        python: Path,
        log_path: Path,
        *,
        serving_stack: str | None,
        max_num_seqs: int | None,
        mem_fraction: float | None,
        context_length: int | None,
        voices: VoicesDocument | None,
        mlx_total_bytes: int | None,
    ) -> None:
        """`serving_stack` comes from the env spec, `max_num_seqs` from the
        voice manifest, `voices` from `narratorvoices.write_document`,
        `mlx_total_bytes` from the accelerator probe, and for `higgs-v3` ALL
        FOUR ARE REQUIRED HERE — the first two on the served arm, the memory on
        the in-process one, the document on both.

        NONE HAS A DEFAULT, keyword-only and mandatory. `None` is a real
        answer — "narrator starts no server out of this env", "this engine
        reads no document", "this is not the in-process arm" — and a default
        would make FORGETTING to pass one indistinguishable from saying it,
        which is precisely how a worker ends up spawned without HIGGS_STACK. A
        caller must state all four; `build_voice_engine` is where they come
        from.

        Refused at CONSTRUCTION and not at spawn, because the alternative is a
        worker that starts, reads 8.5 GB off disk and exits 3 before it says
        `ready` — which is exactly how this was found. A refusal that arrives
        before the process does names the missing thing instead of leaving a
        reader to find `HIGGS_STACK is not set` at the end of an engine log.

        `serving_stack` is None on `mlx-darwin`, where narrator starts no
        server and reads none of these, and for any engine with no row in
        `jobenv.CUDA_LINUX_SERVING_STACK`. `voices` is None for an engine
        outside `narratorvoices.DOCUMENT_READERS` — one whose weights ride the
        `load` message and which reads no `NARRATOR_HIGGS_*` variable.
        `mlx_total_bytes` is None on every arm that is NOT the in-process one,
        and refused there, because a memory figure handed to an engine that
        renders through a server configures nothing.
        """
        super().__init__(python=python, log_path=log_path)
        self._narrator_engine = narrator_engine
        self._serving_stack = serving_stack
        self._max_num_seqs = max_num_seqs
        # NOT VALIDATED AGAIN HERE. `voices.py:_check_serving` is the one owner
        # of what a fraction and a context length may be — (0, 1) and positive,
        # each owing its note — and a second copy of those rules in this file is
        # the two-owners shape `docs/ARCHITECTURE.md` catalogues. What this
        # class owns is whether the variable is EMITTED, which is below.
        self._mem_fraction = mem_fraction
        self._context_length = context_length
        if narrator_engine == HIGGS_V3:
            if voices is None:
                # THE DOCUMENT IS HOW A HIGGS VOICE IS NAMED, on both arms.
                # Without it there is no load this worker could accept, so the
                # refusal is here and not at the first `load`.
                raise EngineError(
                    f"cannot start {self.name} without a voices document: "
                    "narrator resolves a Higgs v3 voice by name in the "
                    "NARRATOR_HIGGS_VOICES document and refuses a modelDir on the "
                    "load message, on the served arm and the MLX arm alike. "
                    "crucible/narratorvoices.py writes it from the voice "
                    "manifest and the pulled weights at every load"
                )
        elif voices is not None:
            # A DOCUMENT FOR AN ENGINE THAT READS NONE. An engine outside
            # `narratorvoices.DOCUMENT_READERS` takes its weights on the `load`
            # message and never reads the variable; handing it one would leave
            # two statements of where the weights are, one of them read by
            # nothing.
            raise EngineError(
                f"{self.name} was given a voices document ({voices.path}), but "
                f"only {HIGGS_V3!r} resolves a voice by name in one; any other "
                "engine takes its weights on the load message"
            )
        self._voices = voices
        if narrator_engine == HIGGS_V3 and serving_stack is not None:
            # THE SERVED ARM. `serving_stack` set is what "narrator will start a
            # server out of this env" means, and it is the one condition under
            # which all three variables have a reader.
            if max_num_seqs is None:
                raise EngineError(
                    f"cannot start {self.name} on the {serving_stack} stack "
                    f"without {MAX_NUM_SEQS_VARIABLE}: it is stage 0's "
                    "admission width and the width of narrator's own batch, "
                    "and narrator refuses it by name "
                    "(v3_served.serve_concurrency). It comes from the voice "
                    "manifest's [voice.serving].max_num_seqs"
                )
            if max_num_seqs < 1:
                raise EngineError(
                    f"{MAX_NUM_SEQS_VARIABLE}={max_num_seqs} for {self.name} "
                    "must be at least 1"
                )
            # `<env>/bin/python` -> `<env>`, confirmed by what is on disk
            # there. `higgs_env_prefix` is where the two ways this was wrong
            # are written down; its refusal names this engine here.
            try:
                self._env_prefix: Path | None = higgs_env_prefix(
                    python, serving_stack)
            except EngineError as refusal:
                raise EngineError(
                    f"cannot start {self.name}: {refusal}"
                ) from refusal
        elif serving_stack is not None:
            # A STACK ON AN ENGINE THAT HAS NONE. `HIGGS_*` is Higgs v3's
            # vocabulary, and an engine that renders in process reads not one
            # of these names. Silently dropping the value would leave the env
            # recipe and this file disagreeing about what that env starts.
            raise EngineError(
                f"{self.name} was given serving_stack={serving_stack!r}, but "
                f"only {HIGGS_V3!r} starts a server underneath narrator and "
                "reads the HIGGS_* variables. Either the env recipe installed "
                "a stack this engine cannot use, or jobenv.tts_env named one "
                "it should not have"
            )
        else:
            self._env_prefix = None
        if narrator_engine == HIGGS_V3 and serving_stack is None:
            # THE IN-PROCESS ARM — `mlx-darwin`. It starts no server, so none of
            # the three above has a reader here; what it DOES have is a batch
            # width AND the memory budget that width is narrowed against, and
            # narrator's defaults for them are 1 and 42 GB. Resolved at
            # CONSTRUCTION for this file's stated reason: a refusal that arrives
            # before the process does names the missing thing, and the
            # alternative — a worker that starts and renders a whole book one
            # chunk at a time, or one that budgets 42 GB on a 16 GB machine —
            # announces nothing at all.
            if mlx_total_bytes is None:
                raise EngineError(
                    f"cannot start {self.name} on the in-process arm without "
                    "the machine's total memory: the batch width "
                    f"({MLX_BATCH_VARIABLE}) and the budget it is narrowed "
                    f"against ({MLX_MEM_BUDGET_VARIABLE}) come out of ONE "
                    "measured row chosen by that figure, and narrator's own "
                    "defaults for them are 1 — one chunk at a time, a measured "
                    "7x — and 42 GB, which is a 64 GB machine's number. "
                    "`accelerator.probe_unified_memory` reads it and "
                    "`residency.load_voice` passes it"
                )
            self._mlx_tier: MlxTier | None = mlx_render_profile(
                narrator_engine, mlx_total_bytes
            )
        else:
            # Every other arm reads no `NARRATOR_HIGGS3_MLX_*` variable: the
            # served arm renders through vllm-omni, and an engine that is not
            # `higgs-v3` owes its own set here (see `environment`) rather than
            # inheriting Higgs's vocabulary.
            if mlx_total_bytes is not None:
                # A MEMORY FIGURE FOR AN ARM THAT SIZES NOTHING FROM IT, refused
                # the way a stack on an in-process engine is: the served arm's
                # memory is the launch script's two GPU fractions, and a second
                # statement of "how much memory may this have" that nothing
                # reads is the lever-that-reports-success shape again.
                raise EngineError(
                    f"{self.name} was given mlx_total_bytes={mlx_total_bytes}, "
                    "but only the in-process arm (a higgs-v3 engine whose env "
                    "starts no serving stack) sizes a batch from this "
                    "machine's memory. A served narrator's memory is its "
                    "launcher's GPU fractions"
                )
            self._mlx_tier = None
        #: One writer at a time. narrator holds a lock over its own stdout for
        #: the mirror-image reason (two half-written lines are not two messages);
        #: the render door and a cancel arriving from the queue thread are two
        #: writers, and an interleaved write would be the same corruption in the
        #: other direction.
        self._writer = threading.Lock()
        self._inbox: queue.Queue[dict[str, Any] | _Garbled | _Ended] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._ready_message: dict[str, Any] | None = None

    # ------------------------------------------------------------- the spawn

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"narrator ({self._narrator_engine})"

    @property
    def narrator_engine(self) -> str:
        return self._narrator_engine

    @property
    def base_url(self) -> str:
        """There is none, and saying so is the point.

        Every other engine answers OpenAI routes on a loopback port and the proxy
        forwards to them. narrator answers no HTTP at all, so a `base_url` here
        would be a port nothing is listening on, published on a `/v1/health` row
        and eventually fetched by something. `ResidentVoice` deliberately carries
        no such field either.
        """
        raise EngineError(
            f"{self.name} has no base url: its wire is newline-delimited JSON "
            "over stdin and stdout, not HTTP. Talk to it through this engine "
            "object (PHASE3-TTS.md section 4)"
        )

    def command(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> list[str]:
        """`<tts env python> -m narrator.serve`, and nothing else.

        `model_dir`, `served_name` and `port` are not on it. The voice travels
        on the `load` message, because narrator is a resident server that
        switches voices without respawning; the weights travel in the
        NARRATOR_HIGGS_VOICES document for `higgs-v3` (see `load`); the port is
        not used at all, and `Residency` finds one anyway for the reason it
        says there.
        """
        return [str(self._python), "-m", MODULE]

    def environment(self) -> dict[str, str]:
        """Everything narrator refuses to start without, and nothing else.

        The three `HIGGS_*` variables are emitted ONLY on the arm that reads
        them — a `higgs-v3` env whose recipe installs a serving stack. On
        `mlx-darwin` narrator builds `HiggsV3MlxEngine` from
        `HiggsV3MlxConfig`, neither of which reads `HIGGS_STACK` or
        `HIGGS_MAX_NUM_SEQS` (the MLX `detect_backend()` returns 'mlx' off an
        import), and there is no launch script for `HIGGS_ENV` to mean anything
        to. Setting them there would be three levers read by nothing.

        THAT ARM HAS A WIDTH OF ITS OWN, and until 2026-09-15 this method left
        it unset. `NARRATOR_HIGGS3_MLX_BATCH` is what `HiggsV3MlxEngine` reads
        instead of `HIGGS_MAX_NUM_SEQS`, it defaults to **1**, and at 1
        `render_many` renders one chunk at a time. So the symmetry the paragraph
        above states was only half true: the served arm was told its width and
        the in-process arm was told nothing, which is not the same as "reads
        none of these" — it is a 7x, measured on owens-mac-studio and written
        down at `MLX_TIERS`. Emitted only here, because the served arm reads it
        no more than the MLX arm reads `HIGGS_STACK`.

        AND IT IS THREE VARIABLES, not one. The width is a ceiling narrator
        narrows against `NARRATOR_HIGGS3_MLX_MEM_BUDGET_GB`, whose own headroom
        is that budget less the weights less `HIGGS_MLX_CACHE_LIMIT_GB` — so
        they are one decision and come off one `MlxTier` row, chosen by this
        machine's own memory. Stating the width alone (which is what the first
        fix did) is asking a budgeter to do its job while telling it the
        budget of a machine somebody else owns.

        A SECOND ENGINE WILL OWE ITS OWN SET HERE, and finding it is that
        engine's first job rather than something guessed in advance: narrator's
        `serve/worker.py` reads each engine's configuration from the
        environment and from the `load` message, and which half Crucible owns
        is the same question `load()` defers on for `caps`.
        """
        environment = {
            ENGINE_VARIABLE: self._narrator_engine,
            # narrator writes its progress to stderr and its protocol to stdout.
            # Without this, a pipe makes CPython block-buffer both, and a `ready`
            # line can sit in a 4 KB buffer for the whole of a load — which reads
            # from here as a readiness timeout on an engine that was up.
            "PYTHONUNBUFFERED": "1",
        }
        if self._env_prefix is not None:
            # `__init__` refuses unless all three are answerable, so this block
            # is all-or-nothing by construction rather than by three checks.
            assert self._serving_stack is not None
            assert self._max_num_seqs is not None
            environment[STACK_VARIABLE] = self._serving_stack
            # THE STACK'S OWN NAME FOR ITS PREFIX. `HIGGS_ENV` on vllm-omni,
            # `HIGGS_SGL_ENV` on SGLang-Omni — see `STACK_ENV_PREFIX_VARIABLE`
            # for why sending the wrong one is worse than sending none.
            environment[env_prefix_variable_for(self._serving_stack)] = str(
                self._env_prefix
            )
            environment[MAX_NUM_SEQS_VARIABLE] = str(self._max_num_seqs)
        # THE TWO LEVERS THE MANIFEST MAY STATE, ON EITHER ARM (2026-09-19).
        # Outside the `_env_prefix` block above deliberately: `HIGGS_STACK`,
        # `HIGGS_ENV` and `HIGGS_MAX_NUM_SEQS` are the SERVED arm's vocabulary
        # and mean nothing in process, but Owen ruled the same day that darwin
        # is to be configured the same way — "context limits and such" — so
        # these two are stated wherever the manifest states them and narrator
        # answers for the arm it is on.
        #
        # ONLY WHEN THE MANIFEST STATED ONE. Absent means narrator's own
        # launcher default, which is a number in a file with an owner
        # (`serve_higgs_sgl.sh:59` writes 0.60), and writing it back here would
        # be Crucible restating a value it did not choose — the shape that put
        # narrator's `CHARS_PER_SEC` 15.0 into two voice manifests.
        if self._mem_fraction is not None:
            # `:g` for `MLX_MEM_BUDGET_VARIABLE`'s reason: narrator parses with
            # `float()` and the launcher's `case` test takes `0.48` either way,
            # and a `/proc/<pid>/environ` read from two machines compares
            # without a reader wondering what a `.0` means.
            environment[MEM_FRACTION_VARIABLE] = f"{self._mem_fraction:g}"
        if self._context_length is not None:
            # NARRATOR DOES NOT READ THIS YET, and that is said out loud rather
            # than discovered. At bookforge HEAD (2026-09-19)
            # `engine/higgs/sgl_served.py:217-221` states the 4096 as a class
            # attribute of SGLang-Omni's `HiggsTtsEngineBuilder` "with no CLI
            # flag and no config path - the value cannot be raised from here,
            # from the launcher, or from a request", and no `HIGGS_CONTEXT_
            # LENGTH` exists anywhere in that tree. The variable is the agreed
            # name for the channel narrator is growing on its own branch; until
            # the tts env's pin moves to it, this is set and read by nothing.
            environment[CONTEXT_LENGTH_VARIABLE] = str(self._context_length)
        if self._mlx_tier is not None:
            # ONE ROW, THREE VARIABLES, and that is the whole point of the row.
            # The width is a CEILING TO ASK FOR, never a promise to allocate:
            # narrator narrows it per slice against the budget below
            # (`_mlx_width_for_depth`), and the budget's own headroom is
            # `budget - weights - cache`, so stating two of the three and
            # letting narrator default the third is two rows' worth of memory in
            # one sum. `__init__` refuses an engine with no measured row and a
            # machine with no readable memory, so these are numbers rather than
            # guesses.
            environment[MLX_BATCH_VARIABLE] = str(self._mlx_tier.width)
            # `:g` rather than `str()`: narrator parses these with `float()`
            # either way, and "42" is what BookForge's own spawn writes, so a
            # log line or a `/proc/<pid>/environ` read from the two arms of one
            # fleet compares without a reader wondering what the `.0` means.
            environment[MLX_MEM_BUDGET_VARIABLE] = f"{self._mlx_tier.mem_budget_gb:g}"
            environment[MLX_CACHE_LIMIT_VARIABLE] = (
                f"{self._mlx_tier.cache_limit_gb:g}"
            )
        if self._voices is not None:
            # NARRATOR_HIGGS_VOICES on both Higgs arms, plus the MLX arm's base
            # weights when the document has a voice that loads them. The
            # document decides which; see `narratorvoices.VoicesDocument`.
            environment.update(self._voices.environment())
        return environment

    def stdio(self, log_handle: Any) -> dict[str, Any]:
        """stdin and stdout stay pipes; only stderr goes to the log.

        The seam PHASE3-TTS.md section 4 said `start()` would need. UTF-8 is
        stated rather than inherited: the text on these pipes is a book, and the
        locale of whatever shell started the server has no business deciding how
        an em-dash crosses it.
        """
        return {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": log_handle,
            "text": True,
            "encoding": "utf-8",
            "errors": "strict",
            "bufsize": 1,
        }

    # ------------------------------------------------------------ the reader

    def attach(self, process: subprocess.Popen[Any]) -> None:
        self._ready_message = None
        self._reader = threading.Thread(
            target=self._pump,
            args=(process.stdout,),
            name=f"narrator-stdout-{self._narrator_engine}",
            daemon=True,
        )
        self._reader.start()

    def _pump(self, stream: Any) -> None:
        """Every line of narrator's stdout, classified once, on one thread.

        `ready` is taken out here rather than queued, because it is a lifecycle
        announcement and not an answer to anything: `ready()` polls for it while
        `converse()` has not been called yet, and putting it in the same queue
        would make the two compete for it. A SECOND `ready` is a protocol error —
        narrator prints exactly one, and two would mean the process restarted
        underneath a conversation.
        """
        try:
            for line in stream:
                text = line.strip()
                if not text:
                    continue
                try:
                    message = json.loads(text)
                except json.JSONDecodeError:
                    self._inbox.put(_Garbled(text))
                    continue
                if not isinstance(message, dict) or not isinstance(
                    message.get("type"), str
                ):
                    self._inbox.put(_Garbled(text))
                    continue
                if message["type"] == "ready" and self._ready_message is None:
                    self._ready_message = message
                    continue
                self._inbox.put(message)
        except (ValueError, OSError):
            # The stream was closed under the reader — `detach()` does exactly
            # that when the engine is stopped. End of file is end of file.
            pass
        finally:
            self._inbox.put(_ENDED)

    def detach(self) -> None:
        """Close the pipes and let the reader go. Order matters here.

        stdin is closed first and unconditionally: Crucible owns the write end
        and nothing else touches it, and closing it is what turns a worker
        blocked on a read it will never satisfy into one that sees EOF.

        **stdout is closed only once the reader has finished**, and that is not
        tidiness. `BufferedReader.close()` takes the object's own lock, which the
        reader thread is holding for as long as it is blocked inside `read()` —
        so closing a pipe another thread is reading deadlocks, and it deadlocks
        exactly in the case this method exists for: a worker that ignored SIGTERM
        and is still alive with its stdout open (measured 2026-09-13, in the test
        that proves `stop()` reports a timeout rather than escalating). When the
        process really is gone the reader is at end of file and joins at once, so
        the ordinary path closes the pipe as it always did; when it is not, the
        fd is left to the daemon thread and the caller has just been told, by
        name, that a process is still running.
        """
        process = self._process
        if process is not None and process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        reader = self._reader
        if reader is not None:
            reader.join(timeout=POLL_SECONDS * 4)
        if (
            process is not None
            and process.stdout is not None
            and (reader is None or not reader.is_alive())
        ):
            try:
                process.stdout.close()
            except OSError:
                pass
        self._reader = None
        self._ready_message = None
        while True:
            try:
                self._inbox.get_nowait()
            except queue.Empty:
                break

    # ---------------------------------------------------------- readiness

    def announced_ready(self) -> str | None:
        message = self._ready_message
        if message is None:
            return None
        return (
            f"{self.name} is ready on {message.get('device')} "
            f"(backend {message.get('backend')})"
        )

    def announces_item_take(self) -> bool:
        """Did the narrator on the other end of this wire say it parses a rung?

        `itemTake: true` on the `ready` line means this narrator BUILD reads a
        per-item rung off a `generate_batch` item — BOTH halves, `sampling` and
        `take` — which is the channel a take ladder needs
        (`narrator/engine/item_sampling.py`). A narrator without the key has no
        such channel: its `_resolve_row` reads `item['voice']` and nothing else,
        so a rung is DROPPED IN SILENCE and take N renders at take 0.

        ONE KEY FOR THE TWO FACTS, and the key was `itemSampling` for exactly
        one day. A rung is (sampling deltas, SEED OFFSET): narrator seeded
        chunk i at `config.seed + i` whatever the take until 2026-09-15, so a
        rung that declared no sampling override — which `[[voice.takes]]`
        permits — rendered take 0 byte for byte, and two take-0 re-rolls always
        did. Both halves land in one narrator module, are refused under this one
        code, and a build has both or neither; two capability keys would be two
        owners of one answer (docs/ARCHITECTURE.md). Nothing had shipped under
        the old name — the tts recipes pin a narrator older than either half.

        Measured 2026-09-15, which is why this exists. Two `tts` render jobs on
        voice `owen` — one 150-char sentence, take 0 and take 1, whose rung is
        `temperature = 0.7` — produced byte-identical 264,174-byte artifacts,
        and the run's own narrator log said `Applied extra_params:
        {'temperature': 0.8, ...}`. Crucible had built the item correctly; the
        env's pinned narrator (`envs/tts/higgs-v3-cuda-linux.txt`, bookforge
        0eeb0267) predated the channel by a day. The recipe's pin and this
        file's belief about it were one fact with two owners and nothing
        comparing them (docs/ARCHITECTURE.md). This is the comparison.

        False when the process is not up yet, which is not a claim about the
        build: callers ask it AFTER `ready`, with the engine in hand.

        It answers only "is there a channel". Whether the LOADED ENGINE has a
        particular lever, or a seed lane at all, is narrator's own answer, per
        row, and already has two names — `sampling_not_supported` and
        `take_not_supported`.
        """
        message = self._ready_message
        if message is None:
            return False
        return message.get("itemTake") is True

    def readiness_description(self) -> str:
        return "print a ready line on stdout"

    # ------------------------------------------------------------- the wire

    def send(self, message: dict[str, Any]) -> None:
        """One JSON object, one line, flushed."""
        process = self._process
        if process is None or process.stdin is None:
            raise EngineError(
                f"{self.name} is not running, so there is nothing to send to it"
            )
        line = json.dumps(message, ensure_ascii=False) + "\n"
        with self._writer:
            try:
                process.stdin.write(line)
                process.stdin.flush()
            except (BrokenPipeError, ValueError, OSError) as exc:
                code = process.poll()
                if code is not None:
                    raise EngineError(
                        f"{self.name} exited {code} and could not be sent a "
                        f"{message.get('action')!r}. Last {LOG_TAIL_LINES} lines "
                        f"of {self.log_path}:\n" + self.log_tail()
                    ) from None
                raise EngineError(
                    f"could not write to {self.name}: {exc}. Last "
                    f"{LOG_TAIL_LINES} lines of {self.log_path}:\n" + self.log_tail()
                ) from None

    def converse(
        self,
        message: dict[str, Any],
        *,
        terminal: frozenset[str],
        silence_timeout: float,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Send one message and yield narrator's answer, line by line.

        The iterator ends when a message of one of the `terminal` types has been
        yielded. It is a **stream in arrival order and nothing is reordered** —
        see the module docstring on why.

        `cancelled`, when it goes true, sends narrator one `{"action": "cancel"}`
        and then keeps reading. That is narrator's own contract: a cancel aborts
        what is in flight, the rows that will not be rendered come back as
        ordinary per-row failures, and `batch_done` arrives as always. Hanging up
        instead would leave the engine generating into nothing, and killing it
        would take the voice off the card for the next job. `JobCancelled` is
        raised once the terminal message has been seen, so the caller learns the
        run was cancelled rather than that it finished short.

        **And the cooperation is now BOUNDED.** `CANCEL_GRACE_SECONDS` after the
        cancel goes out, a narrator that has not reached a terminal message is
        answered with `EngineWouldNotStop` instead of being waited on forever.
        Until 2026-09-15 this method trusted the contract completely, and the
        contract was not kept: see that constant for the measurement.
        """
        self.send(message)
        return self._until(terminal, silence_timeout, cancelled)

    def _until(
        self,
        terminal: frozenset[str],
        silence_timeout: float,
        cancelled: Callable[[], bool] | None,
    ) -> Iterator[dict[str, Any]]:
        process = self._process
        if process is None:
            raise EngineError(f"{self.name} is not running")
        deadline = time.monotonic() + silence_timeout
        cancel_sent = False
        #: Set when the cancel goes out and NEVER reset by a line — see
        #: `CANCEL_GRACE_SECONDS`. A narrator that ignores the cancel is a
        #: narrator that is still talking, so the silence clock above cannot see
        #: it; this is the only clock that can.
        cancel_deadline = 0.0
        while True:
            if cancelled is not None and cancelled() and not cancel_sent:
                self.send({"action": "cancel"})
                cancel_sent = True
                cancel_deadline = time.monotonic() + CANCEL_GRACE_SECONDS
            if cancel_sent and time.monotonic() >= cancel_deadline:
                raise EngineWouldNotStop(
                    f"{self.name} was sent a cancel {CANCEL_GRACE_SECONDS:.0f}s "
                    "ago and has not finished what it was doing. Its stdin "
                    "reader sets a flag the moment a cancel lands, so an engine "
                    "still working after this long is one whose rendering arm "
                    "does not read that flag — this is the wire's contract "
                    "being broken, not a slow render. Last "
                    f"{LOG_TAIL_LINES} lines of {self.log_path}:\n"
                    + self.log_tail()
                )
            try:
                item = self._inbox.get(timeout=POLL_SECONDS)
            except queue.Empty:
                code = process.poll()
                if code is not None:
                    raise EngineError(
                        f"{self.name} exited {code} in the middle of a request. "
                        f"Last {LOG_TAIL_LINES} lines of {self.log_path}:\n"
                        + self.log_tail()
                    )
                if time.monotonic() >= deadline:
                    raise EngineError(
                        f"{self.name} said nothing at all for "
                        f"{silence_timeout:.0f}s. Last {LOG_TAIL_LINES} lines of "
                        f"{self.log_path}:\n" + self.log_tail()
                    )
                continue

            # Any line is proof of life, so the silence clock starts again.
            deadline = time.monotonic() + silence_timeout

            if isinstance(item, _Ended):
                raise EngineError(
                    f"{self.name} closed its stdout in the middle of a request "
                    f"(exit {process.poll()}). Last {LOG_TAIL_LINES} lines of "
                    f"{self.log_path}:\n" + self.log_tail()
                )
            if isinstance(item, _Garbled):
                raise EngineError(
                    f"{self.name} wrote a line to stdout that is not a protocol "
                    f"message: {item.line[:300]!r}. stdout carries the wire and "
                    "nothing else; narrator's own diagnostics belong on stderr, "
                    f"which is {self.log_path}"
                )
            if item["type"] == "error":
                # narrator's whole-request refusal: an unknown action, a voice it
                # cannot serve, a load that failed. A per-ROW failure is not this
                # — it is a `batch_item` carrying a `message` — so this ends the
                # conversation rather than being reported alongside the rows.
                raise EngineError(
                    f"{self.name} refused the request: "
                    f"{item.get('message', '(no message)')}"
                )
            yield item
            if item["type"] in terminal:
                if cancel_sent:
                    raise JobCancelled(f"{self.name} was cancelled mid-request")
                return

    # ---------------------------------------------------------------- load

    def load(
        self,
        *,
        voice: str,
        weights_dir: Path,
        warm: bool,
        on_progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Put a voice on the card and return narrator's own `loaded` line.

        This is what makes `load-voice` mean the weights are resident rather than
        only that a process is up: `ready` says narrator is listening, `loaded`
        says the engine underneath it has a voice in memory.

        **Where the weights ride depends on the engine, and the message says
        only what that engine reads.** `higgs-v3` REFUSES `modelDir` by name on
        both arms
        (`resolve_load_voice`: "the served model is the launch script's
        argument, not a per-load field") and resolves `voice` in the
        NARRATOR_HIGGS_VOICES document this engine was constructed with — so
        the message carries the voice id and `warm`, nothing else, and the
        document's entry is where `weights_dir` already is, as `checkpointDir`.
        `weights_dir` is still taken here so the two can be COMPARED: the
        document and this call are two statements of one fact, and a load that
        checks them agree is the difference between one owner and two.

        **No `caps` are sent, and that is a decision rather than an omission.**
        narrator's caps channel is `register_voice_caps`, whose key vocabulary
        is its older engine's (`temperature`, `topP`, `minP`, `repPenalty`, the
        four `eos*` levers, `maxCharsPerSec`) and which **raises on a key it
        does not know**;
        `higgs_v3_config_from_worker_kwargs` refuses the whole payload by name.
        A Higgs voice's sampling reaches narrator through the DOCUMENT instead
        (`narratorvoices.voice_entry`, key `sampling`), which is the channel
        narrator's `load_voices` reads it from on both arms.
        """
        request: dict[str, Any] = {
            "action": "load",
            "voice": voice,
            # Explicit, though narrator's own default is true: a first load may
            # spend time on discarded warm-up renders, and a load-voice job is
            # an operator's explicit order that would rather pay it here than in
            # the first chunk of a book.
            "warm": warm,
        }
        if self._voices is None:
            request["modelDir"] = str(weights_dir)
        else:
            # Refused HERE, by name, for a voice the document does not carry or
            # a directory it does not agree with — narrator would refuse the
            # first the same way, after the process is up.
            named = self._voices.weights_for(voice)
            if named != weights_dir:
                raise EngineError(
                    f"{self.name} was asked to load {voice!r} from {weights_dir}, "
                    f"but {self._voices.path} names {named} for it. The document "
                    "is what narrator reads; two directories for one voice is a "
                    "load nobody can vouch for"
                )
        loaded: dict[str, Any] | None = None
        for message in self.converse(
            request,
            terminal=frozenset({"loaded"}),
            silence_timeout=LOAD_SILENCE_TIMEOUT_SECONDS,
        ):
            if message["type"] == "loaded":
                loaded = message
            elif on_progress is not None:
                on_progress(f"{self.name}: {message['type']} {message}")
        if loaded is None:  # unreachable: `loaded` is the terminal type
            raise EngineError(f"{self.name} ended its load without a loaded line")
        return loaded

    # ---------------------------------------------------------------- stop

    def stop(self) -> None:
        """`quit` on stdin first, then SIGTERM. Never SIGKILL.

        narrator's own module docstring calls the stdin `quit` action its primary
        teardown: it unwinds the stdin loop from inside the process, runs the
        atexit hooks and releases the GPU. SIGTERM reaches the same place through
        a handler that raises `SystemExit(143)`, and it is the backstop for a
        worker that has stopped reading its stdin. Both are here because the
        first is cleaner and the second always arrives. Neither is SIGKILL —
        force-killing a process stuck in a WSL dxg GPU wait wedges the whole WSL
        VM until Windows reboots, which is why `SubprocessEngine.stop()` reports
        a timeout instead of escalating.
        """
        process = self._process
        if process is not None and process.poll() is None:
            try:
                self.send({"action": "quit"})
            except EngineError:
                # The pipe is gone, so the worker is already on its way out and
                # SIGTERM below is the only thing left to send — which is what
                # would have been sent anyway. Not swallowed silently: whatever
                # killed it wrote to stderr, and that is this engine's log.
                pass
            else:
                try:
                    process.wait(timeout=QUIT_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
        super().stop()
