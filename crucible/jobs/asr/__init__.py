"""The `asr` job type: one audio file in, one transcript out.

PHASE4-AUDIO.md section 3. This is a real gap rather than an oversight in the
app: `generate-sentences` is a GPU queue step, it is the only ASR site in
BookForge that takes the arbiter lease, and the whole-m4b align door needs a
rough transcript before the aligner runs.

**There is no default model.** A job names one or it is refused, because an ASR
pass at the wrong size is a transcript that looks fine, is worse, and has nothing
in it to say so. `jobs.resolve_model` does that refusal for free, since this type
advertises several models and none of them is preferred.

What is the client's and what is the server's
---------------------------------------------
The client says *what to transcribe* and *how to read it*: the model, the
language, the two switches that change what whisper is asked for
(`vad_filter`, `word_timestamps`), and optionally the text to prime it with
(`initial_prompt`). Both switches are required — they default to
true in BookForge, and a default here would mean a transcript silently produced
under different rules than the caller assumed.

Everything about *how it is run* is the server's and appears nowhere on the wire:

- **`compute_type`.** `float16` on an accelerator, `int8` on CPU. Crucible has no
  CPU backend, so it is `float16`, always. BookForge also has a **one-shot CPU
  fallback** here (`transcribe-bridge.ts`: a CUDA load that fails is retried once
  on CPU at int8). That does not come across, and the reason is that it is not a
  fallback at all, it is a silent substitution: the run still produces a
  transcript, the transcript is a different transcript, and nothing in the output
  says which one you got. Crucible refuses instead.
- **Windowing.** 900-second windows, each extended 15 seconds past its own
  boundary so a sentence straddling the cut is spoken in full inside it. Those
  are BookForge's measured numbers: an 18-hour file handed to
  `model.transcribe()` in one piece frames the whole signal into about 19 GiB of
  float64 and OOMs, and a 900-second window keeps the peak independent of book
  length. A client sends one file and never learns any of this.
- **Decoding.** 16 kHz mono float32 through ffmpeg, once, in the worker.

The initial prompt and the windows
----------------------------------
`initial_prompt` (optional; see `AsrParams`) is handed to **every** window, not
only the first. The reason is in both libraries' source, not a preference:

- Each window is its own `transcribe()` call, and each call starts its token
  history empty — faster-whisper 1.2.1 `generate_segments` sets
  `all_tokens = []` and puts the prompt at its head; mlx-whisper 0.4.3
  `transcribe` does the same. `condition_on_previous_text` conditions each
  30-second segment on the segments before it *inside one call*; it carries
  nothing from one 900-second window to the next. A prompt given to window 0
  alone would reach the first fifteen minutes and none of the other seventeen
  hours.
- Inside a call the prompt is not permanent either: it is the head of a history
  that is cut to its last 223 tokens, so it scrolls out after a few segments of
  speech, and a temperature fallback above 0.5 resets the history outright. So
  "every window" means the start of every fifteen minutes is primed — which is
  what a caller sending a title and its proper nouns wants, and the most the
  engines can give it.

The overlap seconds a window shares with its neighbour are heard twice, once
primed at the head of the later window; the server's dedup keeps whichever
segment starts first, the same as without a prompt. `transcript.json` records
`initial_prompt` (null when none was sent).

What comes back
---------------
`transcript.json`, holding whisper's own segments in absolute book time with the
window overlaps removed, plus what produced them. Sentence-cue grouping and the
WebVTT stay in BookForge, for the reason section 2 gives about `align`: Crucible
returns what the model said and asserts nothing about the client's units.

The one deviation from the app worth knowing about is where the overlap
duplicates are dropped. BookForge groups words into sentence cues first and
dedupes the *cues*; here the grouping does not exist yet, so the same rule — sort
by start, drop anything that begins inside a kept span, with the same 0.1 s
tolerance — is applied to the *segments*. The boundary behaviour is therefore
close but not identical, and it is written down here rather than discovered later.

A failed window fails the job
-----------------------------
The worker keeps going after a window fails, so one run finds every bad stretch
instead of one per re-run. But the job then fails, naming the windows, and
publishes no artifact. A hole in the middle of a transcript is invisible in the
output — which is the same argument as the one against a default model, and it
gets the same answer.

Qwen3-ASR: the same job type, a different run (2026-09-24)
----------------------------------------------------------
`qwen3-asr-1.7b` is a model under this job type, not a new verb: one audio file
in, `transcript.json` out, the same `language` / `vad_filter` /
`word_timestamps` params. Its engines (`vllm` on cuda-linux, `mlx-audio` on
mlx-darwin) are run by `qwen.py` rather than by the whisper window loop,
because a word-timestamped transcript there is two models (the ASR model and
the Qwen3 aligner) and a loop guard between them (`loopguard.py`). What
differs on the wire, all of it refused by name rather than ignored:

- `context` (optional) is Qwen's system-turn context; `initial_prompt` is
  whisper's primed transcript. Each engine refuses the other's field — they
  are two mechanisms with two limits, not two spellings of one.
- `language` must be a language the aligner supports, and never `auto`.
- `vad_filter: true` is refused: neither Qwen engine has a VAD.

docs/PHASE25-QWEN-ASR.md is the contract.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictStr,
    ValidationError,
    field_validator,
)

from ... import accelerator, hosttools, jobenv, weights, workerenv, workers
from ...asrmodels import (
    QWEN_ASR_ENGINES,
    QWEN_CONTEXT_MAX_TOKENS,
    AsrManifest,
    AsrManifestError,
    RENAMED_ASR_IDS,
    load_all_asr_manifests,
    retired_asr_id_note,
)
from ...config import Config
from ...errors import ApiError, JobError
# The one place `<id>@<revision>` is spelled. Two spellings of a fingerprint
# is two names for one set of weights, which is the thing it exists to stop.
from ...manifests import fingerprint
from ..align import QWEN3_LANGUAGES
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor
from . import qwen

__all__ = ["AsrJobType", "AsrParams"]

JOB_TYPE = "asr"

#: The audio window, and how far each window reaches past its own end. Engine
#: knowledge, measured by BookForge against real books
#: (`electron/scripts/transcribe_audiobook.py`), not a wire parameter.
WINDOW_SECONDS = 900
OVERLAP_SECONDS = 15

#: How long the server waits on a worker that has said *nothing at all* before it
#: gives up on it. Not a run deadline: every message resets it, and the worker
#: reports decode progress from its first seconds, so this only fires on a worker
#: that is genuinely wedged. 900 s covers loading a 3 GB model from a cold disk.
READY_SILENCE_TIMEOUT_SECONDS = 900.0

#: Two segments whose spans overlap by more than this are the same speech heard
#: twice, once in each of two consecutive windows. BookForge's number.
OVERLAP_TOLERANCE_SECONDS = 0.1

#: `compute_type` and the device, per ENGINE — because this job type has two,
#: and they are two libraries rather than two builds of one.
#:
#: There is no CPU entry in either table and there is no default in either:
#: Crucible has no CPU backend, and a backend nobody has decided about must be
#: a refusal naming it rather than a `cuda` handed to a Mac.
#:
#: Both engines land on `float16` and that is a coincidence worth stating
#: rather than a shared constant: faster-whisper's is CTranslate2's
#: `compute_type`, and mlx-whisper's is the `fp16=True` its `transcribe()`
#: defaults to and turns into `mx.float16`. They mean the same precision by two
#: different routes, which is why the worker is told the value rather than
#: assuming it.
#:
#: The device names are each library's own. `cuda` is CTranslate2's;
#: `metal` is MLX's one and only device, and it is deliberately NOT `mps` —
#: `mps` is torch's name for the same silicon and the `align` job type uses it
#: because the aligner IS torch. Two libraries, two spellings, and neither
#: worker will accept the other's (`crucible/jobs/asr/mlx_worker.py` refuses by
#: name).
COMPUTE_TYPE_FOR_ENGINE: dict[str, str] = {
    "faster-whisper": "float16",
    "mlx-whisper": "float16",
}
DEVICE_FOR_ENGINE: dict[str, str] = {
    "faster-whisper": "cuda",
    "mlx-whisper": "metal",
}

#: Which script runs each engine. A SECOND WORKER and not a branch inside the
#: first: the two run in different envs on different machines and share no
#: import — one needs `faster_whisper` (CTranslate2), the other `mlx_whisper`
#: (MLX), and neither library exists in the other's env. The wire between
#: server and worker is identical, which is what keeps `transcript.json` one
#: document whichever machine produced it.
WORKER_SCRIPT_FOR_ENGINE: dict[str, Path] = {
    "faster-whisper": Path(__file__).resolve().parent / "worker.py",
    "mlx-whisper": Path(__file__).resolve().parent / "mlx_worker.py",
}

#: Engines with no voice-activity detector at all. faster-whisper ships Silero;
#: mlx-whisper ships nothing of the kind. A job that asks for `vad_filter` on
#: one of these is refused BY NAME rather than transcribed without it, for the
#: reason the module docstring gives about the CPU fallback: the run would
#: produce a transcript under different rules and nothing in the file would say
#: so.
#:
#: Both Qwen engines are in it too: Qwen3-ASR is handed a piece of audio and
#: transcribes it, and neither vLLM nor mlx-audio has a detector in front.
ENGINES_WITHOUT_VAD: frozenset[str] = frozenset({"mlx-whisper", *QWEN_ASR_ENGINES})

#: Which env's python runs each engine's worker. Whisper's two engines are the
#: `asr` env's (`envs/asr/*.txt`); the Qwen engines run in the LLM env, which
#: already pins vLLM 0.29.0 on cuda-linux and mlx-audio 0.5.5 on mlx-darwin, so
#: Qwen3-ASR costs no new env on either machine (docs/PHASE25 section 6). The
#: aligner a Qwen job also runs is the `align` env's, resolved in `qwen.py`.
ENV_FOR_ENGINE: dict[str, str] = {
    "faster-whisper": "asr",
    "mlx-whisper": "asr",
    "vllm": "llm",
    "mlx-audio": "llm",
}

#: THE LONGEST CONTEXT ANY QWEN JOB MAY SEND, IN CHARACTERS: a cheap refusal
#: before the job is queued. The real limit is `QWEN_CONTEXT_MAX_TOKENS` (1,024
#: of the model's own tokens), counted by the worker with the model's tokenizer
#: after it loads; English runs about four characters a token, so 8,192
#: characters is twice what could ever fit and only stops a client that sent
#: a document where a sentence belongs.
CONTEXT_MAX_CHARS = 8192

#: What a context may not contain: the chat template's own control tokens. The
#: context is placed verbatim inside the system turn, so `<|im_end|>` in it
#: would end that turn and start whatever followed. vLLM's own transcription
#: door STRIPS these silently (`_sanitize_transcription_user_text`); this
#: server refuses instead, because a context that was quietly edited is a
#: transcript made under a prompt nobody wrote.
_CHAT_CONTROL = re.compile(r"<\|[^|]*\|>|<asr_text>")


def _for_engine(table: dict[str, Any], engine: str, what: str) -> Any:
    """One engine's entry, or a refusal naming it. Never a default."""
    found = table.get(engine)
    if found is None:
        raise JobError(
            "engine_unsupported",
            f"there is no {what} for asr engine {engine!r}; this build runs "
            f"{sorted(table)}",
        )
    return found

#: The language codes faster-whisper accepts, read from
#: `faster_whisper/tokenizer.py`'s `_LANGUAGE_CODES` at master on 2026-09-13.
#: Checked here so a typo is a 400 naming the code rather than a job that dies
#: mid-stream inside the worker — whisper raises on an unknown code only when the
#: tokenizer is built, which is after the model is on the card.
WHISPER_LANGUAGES = frozenset(
    """af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo
    fr gl gu ha haw he hi hr ht hu hy id is it ja jw ka kk km kn ko la lb ln lo
    lt lv mg mi mk ml mn mr ms mt my ne nl nn no oc pa pl ps pt ro ru sa sd si
    sk sl sn so sq sr su sv sw ta te tg th tk tl tr tt uk ur uz vi yi yo zh
    yue""".split()
)

#: What a client sends instead of a code to ask whisper to detect the language.
#: It is a value, not an absence: "detect it" is a decision, and a job that did
#: not make it is a job that did not say what it wanted.
AUTO_LANGUAGE = "auto"

class AsrParams(BaseModel):
    """`params` for an asr job. Unknown keys are refused, not ignored.

    `initial_prompt` — the one param with a default, and why
    ---------------------------------------------------------
    Text whisper reads as if it were the transcript so far, before it hears the
    first second: the way to tell it how a title, a name or a coined word is
    spelled. Both engines take it in `transcribe()` under this name
    (faster-whisper 1.2.1 `WhisperModel.transcribe(initial_prompt=...)`,
    mlx-whisper 0.4.3 `transcribe(initial_prompt=...)`), and both do the same
    thing with it: encode `" " + prompt.strip()` and put it at the head of the
    token history the first 30-second segment is conditioned on.

    Every other param here is required because every one changes the
    transcript, and so does this one. It is still **optional on the wire, with
    `None` meaning no prompt**, because the asr client already in the fleet —
    BookForge's `electron/crucible/asr.ts`, through the `@crucible/client`
    1.0.23 it pins, whose `asr()` sends exactly the three keys above — would be
    refused by a fourth required key: every one of its transcripts a 400 on the
    day this server is deployed, for a feature it never asked for. The absent case is not a
    silent substitution: no prompt is precisely what those clients were getting
    and what they meant, and `transcript.json` records `initial_prompt: null`
    so the document still says which rule it was made under. A client that
    wants to be explicit sends `null`; the SDK sends the key whenever its
    caller states it.

    A non-string is refused (`StrictStr`: no `5` turned into `"5"`), and so is a
    blank one — `""` and `None` would be two spellings of "no prompt", and
    faster-whisper does not even treat them alike (it encodes `" "` for `""`).

    Its LENGTH is checked by the worker, not here. Both libraries keep only the
    last `max_length // 2 - 1` prompt tokens (223 on every whisper), so a longer
    prompt would lose its beginning with no error; the count needs the model's
    own tokenizer, which exists only in the worker's env, so each worker counts
    with it after loading and fails the run by name before the first window.


    `context` — Qwen3-ASR's, and optional for the same reason
    ---------------------------------------------------------
    The text Qwen3-ASR reads in its SYSTEM turn before it hears the audio: an
    instruction and a vocabulary, e.g. ContentStudio's "Verbatim transcript of
    a livestream. Transcribe every disfluency exactly as spoken, including
    filler sounds: um, uh, ah, er, hmm, and false starts and repeated words."
    (which took a 10-minute window from 9 fillers to 19). It is NOT
    `initial_prompt` under another name, and the two are not interchangeable:
    whisper reads its prompt as the transcript so far, keeps its last 223
    tokens and lets it scroll out after a few segments; Qwen reads the context
    as an instruction, whole, for every piece, up to 1,024 tokens here
    (`asrmodels.QWEN_CONTEXT_MAX_TOKENS`). One field for both would carry two
    limits and two meanings under one name, so each engine refuses the other's
    field by name (`_refuse_what_this_engine_has_not_got`).

    Optional with `None` meaning no context, for `initial_prompt`'s reason: the
    asr clients already in the fleet send three keys. Blank is refused, a
    non-string is refused, and so is a context carrying the chat template's
    own control tokens (`_CHAT_CONTROL`).
    """

    model_config = ConfigDict(extra="forbid")

    language: str
    vad_filter: bool
    word_timestamps: bool
    initial_prompt: StrictStr | None = None
    context: StrictStr | None = None

    @field_validator("initial_prompt")
    @classmethod
    def prompt_is_not_blank(cls, value: str | None) -> str | None:
        if value is not None and value.strip() == "":
            raise ValueError(
                "initial_prompt is blank; send null for no prompt, or the text "
                "whisper should be primed with (a title, the names in it)"
            )
        return value

    @field_validator("context")
    @classmethod
    def context_is_plain_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value.strip() == "":
            raise ValueError(
                "context is blank; send null for no context, or the instruction "
                "and vocabulary Qwen3-ASR should read before the audio"
            )
        if len(value) > CONTEXT_MAX_CHARS:
            raise ValueError(
                f"context is {len(value)} characters; Qwen3-ASR is given at most "
                f"{QWEN_CONTEXT_MAX_TOKENS} tokens of it, which is well under "
                f"{CONTEXT_MAX_CHARS} characters. Send the instruction and the "
                "names, not the document"
            )
        found = _CHAT_CONTROL.search(value)
        if found is not None:
            raise ValueError(
                f"context contains {found.group(0)!r}, one of the chat "
                "template's own control tokens; the context is placed verbatim "
                "inside the system turn, where that would end the turn"
            )
        return value

    @field_validator("language")
    @classmethod
    def known_language(cls, value: str) -> str:
        if value == AUTO_LANGUAGE or value in WHISPER_LANGUAGES:
            return value
        raise ValueError(
            f"{value!r} is not a language faster-whisper knows; send a code from "
            f"{sorted(WHISPER_LANGUAGES)} or {AUTO_LANGUAGE!r} to have it detected"
        )

    def whisper_language(self) -> str | None:
        """What the worker passes to `model.transcribe`. None means detect."""
        return None if self.language == AUTO_LANGUAGE else self.language


# ------------------------------------------------------------------ helpers


def adopt_renamed_asr_weights(config: Config) -> list[str]:
    """Move weights pulled under a RENAMED asr id into the new id's folder.

    Run once per server start, before anything is served (`crucible serve`).
    Owen's lineup ruling of 2026-09-24 renamed four ids (`RENAMED_ASR_IDS`) and
    the store is laid out by id, so without this a Mac that had pulled
    `mlx-whisper-large-v3-turbo` would read `whisper-large-v3-turbo` as not
    installed and download the same 1.6 GB again beside the old copy. The
    store moves a directory only when its stamp names the new block's exact
    pin (`weights.adopt_renamed`), so this never puts different bytes under an
    id. Idempotent: once moved there is nothing under the old id to find.

    A rename whose target this build does not ship is a defect in THIS build,
    not weather, and is refused by name.
    """
    manifests = load_all_asr_manifests()
    lines: list[str] = []
    for old_id, new_id in sorted(RENAMED_ASR_IDS.items()):
        manifest = manifests.get(new_id)
        if manifest is None:
            raise AsrManifestError(
                f"RENAMED_ASR_IDS maps {old_id!r} to {new_id!r}, and this build "
                f"ships no such asr manifest (it ships {sorted(manifests)})"
            )
        lines.extend(
            weights.adopt_renamed(config, old_id, manifest, manifest.backends)
        )
    return lines


def _manifests() -> dict[str, AsrManifest]:
    try:
        return load_all_asr_manifests()
    except AsrManifestError as exc:
        raise ApiError(
            500,
            "asr_manifests_unreadable",
            f"this server cannot read its ASR model manifests: {exc}",
        ) from None


def _known(model_id: str) -> AsrManifest:
    manifests = _manifests()
    manifest = manifests.get(model_id)
    if manifest is None:
        # A removed id is refused like any other unknown id — it is NOT an
        # alias (Owen, 2026-09-24) — and the sentence says what replaced it,
        # so a client reading the refusal can fix its call in one edit.
        note = retired_asr_id_note(model_id)
        raise ApiError(
            400,
            "unknown_model",
            f"no ASR manifest for model {model_id!r}; this build ships "
            f"{sorted(manifests)}" + ("" if note is None else f". {note}"),
            {"model": model_id, "offered": sorted(manifests)},
        )
    return manifest


def _params(params: dict[str, Any]) -> AsrParams:
    try:
        return AsrParams.model_validate(params)
    except ValidationError as exc:
        raise ApiError(
            400,
            "invalid_params",
            "asr params are not valid: "
            + "; ".join(
                f"{'.'.join(str(p) for p in problem['loc']) or '<root>'}: "
                f"{problem['msg']}"
                for problem in exc.errors()
            ),
        ) from None


def _most_a_job_needs(spec: Any, backend_kind: str) -> int:
    """What the card must hold for the largest job this model can run.

    A whisper model's own estimate. A Qwen model's estimate PLUS its aligner's,
    because a word-timestamped job holds both for its whole run
    (`qwen.need_bytes`); the figure a listing shows is the figure the guard
    will ask for.
    """
    if spec.engine in QWEN_ASR_ENGINES:
        return qwen.need_bytes(spec, backend_kind, with_aligner=True)
    return spec.memory_bytes_estimate


def _env_ready(config: Config, env: str, backend_kind: str) -> tuple[bool, str]:
    """Is one env installed here, and what does its status say."""
    try:
        if env == "llm":
            status = jobenv.env_status(
                config.home, jobenv.llm_env(backend_kind), backend_kind
            )
        else:
            status = workerenv.env_status(config.home, env, backend_kind)
    except (workerenv.WorkerEnvError, jobenv.EnvError) as exc:
        return False, str(exc)
    return status.installed, status.detail


def _python_for(config: Config, engine: str, backend_kind: str, model_id: str) -> Path:
    """The interpreter an engine's worker runs in, or `env_missing` by name."""
    env = _for_engine(ENV_FOR_ENGINE, engine, "env")
    try:
        if env == "llm":
            spec = jobenv.llm_env(backend_kind)
            return jobenv.require_env(config.home, spec, backend_kind)
        return workerenv.require_env(config.home, env, backend_kind)
    except (workerenv.WorkerEnvError, jobenv.EnvError) as exc:
        directory = (
            jobenv.env_dir(config.home, jobenv.llm_env(backend_kind))
            if env == "llm"
            else workerenv.worker_env_dir(config.home, env)
        )
        raise ApiError(
            409,
            "env_missing",
            f"cannot run {model_id!r}: its engine {engine!r} runs in the {env} "
            f"env, and {exc}",
            {"model": model_id, "env": str(directory)},
        ) from None


def ffmpeg_path() -> str | None:
    """Where ffmpeg is on this host, or None.

    A module-level probe, for the reason `crucible/accelerator.py` gives about
    its own: a test replaces it and asserts on the refusal, instead of asserting
    on whatever happens to be installed on the machine running the suite. The
    search itself goes through `crucible/hosttools.py`, which is the one owner of
    *which PATH was searched* — the fact every refusal below has to name.
    """
    return hosttools.which("ffmpeg")


def _require_ffmpeg() -> str:
    """ffmpeg's path, or a refusal by name before the job is queued.

    The worker decodes through ffmpeg rather than through faster-whisper's own
    PyAV decoder, which silently truncates some assembled m4b files (see
    `worker.py`). So ffmpeg is not optional, and its absence is a fact about the
    host that should be a 409 at submit time rather than a job that dies a minute
    in.
    """
    found = ffmpeg_path()
    if found is None:
        raise ApiError(
            409,
            "ffmpeg_missing",
            "there is no ffmpeg on this server's PATH, and asr decodes every input "
            "through it — faster-whisper's own PyAV decoder silently truncates some "
            "m4b files, which ends a transcript hours early with no error. "
            + hosttools.searched_note(),
            {"path": hosttools.search_path()},
        )
    return found


# ------------------------------------------------------------------ job type


class AsrJobType:
    """`POST /v1/jobs {"type": "asr", "model": "<id>", "inputs": {...}}`."""

    name = JOB_TYPE

    def __init__(
        self,
        config: Config,
        backend: Any,
        owned_pids: Callable[[], frozenset[int]],
    ) -> None:
        self._config = config
        self._backend = backend
        # The accelerator guard must not report Crucible's own resident engine as
        # somebody else's process holding the card, so this type is handed the
        # same owned-pid set the `llm` types use. It is a callable and not a set
        # because the answer changes every time a model is loaded or unloaded.
        self._owned_pids = owned_pids

    # ----------------------------------------------------------- describing

    def describe_models(self) -> list[ModelDescriptor]:
        backend_kind = self._config.backend_kind
        rows: list[ModelDescriptor] = []
        for manifest in _manifests().values():
            if manifest.supports(backend_kind):
                spec = manifest.spec(backend_kind)
                revision, source, estimate = (
                    spec.revision,
                    spec.hf_repo,
                    _most_a_job_needs(spec, backend_kind),
                )
                # The same predicate `check` and `_require_loadable` read: the
                # puller's stamp, at the revision this host's block pins.
                installed = (
                    weights.installed(self._config, manifest, spec) is not None
                )
            else:
                # A backend this manifest has no block for has nothing to install.
                revision, source, estimate, installed = "", "", 0, False
            rows.append(
                ModelDescriptor(
                    id=manifest.id,
                    revision=revision,
                    source=source,
                    installed=installed,
                    # Nothing is ever resident for `asr`: the worker loads the
                    # model, transcribes one file and exits. The aligner is the
                    # job type that stays resident across a book, and it is
                    # section 2's, not this one's.
                    resident=False,
                    vram_bytes=estimate,
                )
            )
        return rows

    def retired_model_note(self, model: str) -> str | None:
        """What replaced an asr id this build removed, for `unknown_model`.

        Read by `jobs.resolve_model` when a request names an id this type does
        not serve. Never resolves anything: the old ids are not aliases (Owen,
        2026-09-24), and a request naming one is refused either way.
        """
        return retired_asr_id_note(model)

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        """The `model` block of a transcript's provenance sidecar.

        A transcript is an artifact like any other and has to say which weights
        produced it: `whisper-tiny` and `whisper-large-v3-turbo` disagree about
        a hard passage, and a cue list that does not name its model is a cue
        list nobody can re-derive. The revision is this host's backend pin,
        which is a statement about bytes — `weights.require_installed` refuses
        weights pulled at any other one.

        `engine` AND `hf_repo` SINCE 2026-09-24. Until then an asr id named one
        engine's conversion and the id alone said which bytes; Owen's ruling of
        that day made `whisper-large-v3-turbo` and `whisper-tiny` one id across
        both backends, so the id is a CTranslate2 conversion on the PC and an
        MLX one on the Mac. The sidecar already names the backend beside this
        block; these two keys make the block say which conversion by itself,
        so a transcript read without its manifest still names its bytes.
        """
        if model is None:
            raise JobError("model_required", f"{self.name} needs a model")
        manifest = _known(model)
        spec = manifest.backends.get(self._config.backend_kind)
        if spec is None:
            # Unreachable through the API: `preflight` refuses
            # `backend_unsupported` before a job exists. A sidecar still has to
            # say something true if it is reached another way, and inventing a
            # revision is not it.
            return {
                "id": model,
                "revision": None,
                "fingerprint": None,
                "engine": None,
                "hf_repo": None,
            }
        return {
            "id": model,
            "revision": spec.revision,
            "fingerprint": fingerprint(model, spec.revision),
            "engine": spec.engine,
            "hf_repo": spec.hf_repo,
        }

    def vram_estimate(self, model: str | None) -> int:
        if model is None:
            raise JobError("model_required", f"{self.name} needs a model")
        manifest = _known(model)
        if not manifest.supports(self._config.backend_kind):
            return 0
        return _most_a_job_needs(
            manifest.spec(self._config.backend_kind), self._config.backend_kind
        )

    def check(self, backend: Any) -> JobTypeStatus:
        """Ready when some installed model has its engine's env installed.

        Per ENGINE since 2026-09-24: whisper runs in the `asr` env and Qwen3-ASR
        in the `llm` env (`ENV_FOR_ENGINE`), so "the asr env is missing" no
        longer means this type cannot run — a host with only the llm and align
        envs transcribes with `qwen3-asr-1.7b`. Each env's own detail is kept,
        so a reader learns which one is missing.
        """
        if ffmpeg_path() is None:
            return JobTypeStatus(
                ready=False,
                detail="there is no ffmpeg on PATH, and asr decodes every input "
                "through it",
            )
        try:
            manifests = _manifests()
        except ApiError as exc:
            return JobTypeStatus(ready=False, detail=exc.message)
        envs: dict[str, tuple[bool, str]] = {}
        for env in sorted(set(ENV_FOR_ENGINE.values())):
            envs[env] = _env_ready(self._config, env, backend.kind)
        runnable: list[str] = []
        pulled_without_env: list[str] = []
        for manifest in manifests.values():
            if not manifest.supports(backend.kind):
                continue
            spec = manifest.spec(backend.kind)
            if weights.installed(self._config, manifest, spec) is None:
                continue
            if envs[_for_engine(ENV_FOR_ENGINE, spec.engine, "env")][0]:
                runnable.append(manifest.id)
            else:
                pulled_without_env.append(manifest.id)
        detail = "; ".join(f"{env}: {text}" for env, (_, text) in envs.items())
        if not runnable:
            missing = (
                f"; installed but their env is not: {pulled_without_env}"
                if pulled_without_env
                else "; no ASR model is installed — `crucible models pull <id>`"
            )
            return JobTypeStatus(ready=False, detail=detail + missing)
        return JobTypeStatus(ready=True, detail=f"{detail}; runnable: {runnable}")

    # ------------------------------------------------------------ preflight

    def _spec_for(self, model_id: str) -> tuple[AsrManifest, Any]:
        """The manifest and this host's block, or `backend_unsupported` by name."""
        backend_kind = self._backend.kind
        manifest = _known(model_id)
        if not manifest.supports(backend_kind):
            raise ApiError(
                400,
                "backend_unsupported",
                f"ASR model {model_id!r} has no {backend_kind} block; "
                f"{manifest.path.name} declares {sorted(manifest.backends)}",
                {
                    "model": model_id,
                    "backend": backend_kind,
                    "declared": sorted(manifest.backends),
                },
            )
        return manifest, manifest.spec(backend_kind)

    def _require_runnable(
        self, model_id: str, params: AsrParams
    ) -> tuple[AsrManifest, Any, Path, Path, "qwen.AlignerPlan | None"]:
        """Manifest, spec, env python, weights dir and the aligner (Qwen with
        word timestamps only), or the named refusal.

        The order is `llm`'s and for `llm`'s reason: what no amount of installing
        can fix first, then what an install or a pull would fix, then the live
        accelerator. Nobody is told to download 3 GB of weights for a model that
        will never fit.
        """
        backend_kind = self._backend.kind
        manifest, spec = self._spec_for(model_id)
        accelerator.refuse_if_larger_than_host(
            model_id=model_id,
            need_bytes=self._need_bytes(spec, params),
            host_total_bytes=self._backend.gpu.vram_bytes,
            host_name=self._backend.gpu.name,
        )
        python = _python_for(self._config, spec.engine, backend_kind, model_id)
        try:
            installed = weights.require_installed(self._config, manifest, spec)
        except weights.WeightsError as exc:
            raise ApiError(
                409,
                "model_not_installed",
                str(exc),
                {"model": model_id, "hf_repo": spec.hf_repo, "revision": spec.revision},
            ) from None
        aligner = None
        if spec.engine in QWEN_ASR_ENGINES and params.word_timestamps:
            aligner = qwen.plan_aligner(self._config, spec, backend_kind)
        return manifest, spec, python, installed.path, aligner

    def _need_bytes(self, spec: Any, params: AsrParams) -> int:
        """What THIS job needs: the aligner is on the card only with timestamps."""
        if spec.engine in QWEN_ASR_ENGINES:
            return qwen.need_bytes(
                spec, self._backend.kind, with_aligner=params.word_timestamps
            )
        return spec.memory_bytes_estimate

    def _refuse_what_this_engine_has_not_got(
        self, model_id: str, params: AsrParams
    ) -> None:
        """A param this model's engine cannot honour is a 400, never a no-op.

        Asked BEFORE the env and weights checks: a caller who cannot have what
        they asked for should not first be told to install 2 GB. The engine is
        the MODEL's (its manifest block on this host) since 2026-09-24, because
        a host now has two — whisper and Qwen3-ASR — and they differ.

        `vad_filter: true` where there is no VAD. faster-whisper ships Silero;
        `mlx-whisper` has no voice-activity detector at all — its
        `no_speech_threshold` is the model's own per-segment judgement, which is
        a different mechanism on different evidence and not a substitute — and
        neither Qwen engine has one either. Running the job anyway would produce
        a transcript under rules the caller did not ask for, with nothing in
        `transcript.json` to say which rules those were: the same failure
        PHASE4-AUDIO.md section 3 refuses the CPU fallback for. `vad_filter:
        false` runs perfectly well, so the refusal names the value.

        The prompt fields cross neither way (`AsrParams`): `initial_prompt` on a
        Qwen engine and `context` on a whisper one.

        A Qwen engine is always TOLD the language, and it must be one the
        aligner places words in: ContentStudio's measurement is that
        auto-detection costs the 1.7B time, and the aligner has no detection
        at all — it takes a language name. Eleven languages, the aligner's own
        list (`crucible/jobs/align`); `auto` and the nineteen languages that
        Qwen3-ASR transcribes but the aligner cannot place are refused by name.
        """
        _, spec = self._spec_for(model_id)
        engine = spec.engine
        if engine in ENGINES_WITHOUT_VAD and params.vad_filter:
            raise ApiError(
                400,
                "vad_unsupported_by_engine",
                f"{model_id!r} transcribes with {engine!r}, and that engine has no "
                "voice-activity detector at all — faster-whisper's is Silero, "
                "and mlx-whisper and the Qwen3-ASR engines ship nothing of the "
                "kind. Send vad_filter: false and get a transcript this server "
                "can describe, rather than one produced under rules nothing in "
                "the file records",
                {"backend": self._backend.kind, "engine": engine, "vad_filter": True},
            )
        qwen_engine = engine in QWEN_ASR_ENGINES
        if qwen_engine and params.initial_prompt is not None:
            raise ApiError(
                400,
                "initial_prompt_unsupported_by_engine",
                f"{model_id!r} is Qwen3-ASR, which has no initial_prompt: that is "
                "whisper's primed transcript, 223 tokens that scroll out. Send "
                "`context`, the instruction and vocabulary Qwen reads in its "
                "system turn before every piece",
                {"engine": engine, "field": "initial_prompt"},
            )
        if not qwen_engine and params.context is not None:
            raise ApiError(
                400,
                "context_unsupported_by_engine",
                f"{model_id!r} is whisper ({engine!r}), which has no context: "
                "that is Qwen3-ASR's system-turn instruction. Send "
                "`initial_prompt`, the text whisper is primed with as if it "
                "were the transcript so far",
                {"engine": engine, "field": "context"},
            )
        if qwen_engine and params.language not in QWEN3_LANGUAGES:
            auto = (
                "; `auto` is refused because detection costs the 1.7B time and "
                "the aligner has none"
                if params.language == AUTO_LANGUAGE
                else ""
            )
            raise ApiError(
                400,
                "language_unsupported_by_engine",
                f"{model_id!r} is always told its language, and it must be one "
                f"the aligner places words in: {sorted(QWEN3_LANGUAGES)}. "
                f"{params.language!r} is not{auto}",
                {"engine": engine, "language": params.language},
            )

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        if model is None:  # unreachable: resolve_model requires one
            raise ApiError(400, "model_required", f"{self.name} needs a model")
        checked = _params(params)
        self._refuse_what_this_engine_has_not_got(model, checked)
        _require_ffmpeg()
        _, spec, _, _, _ = self._require_runnable(model, checked)
        accelerator.guard(
            self._config.backend_kind,
            model_id=model,
            need_bytes=self._need_bytes(spec, checked),
            owned_pids=self._owned_pids(),
            desktop_allowance_bytes=self._config.desktop_allowance_bytes,
            # Deliberately no `reclaimable_bytes`. An `llm` load may unload the
            # previous resident to make room for itself; an asr job never unloads
            # somebody's model to run a transcript, so the memory a resident
            # engine holds is not memory this job can have.
        )

    # ------------------------------------------------------------------ run

    def run(self, job: Job, ctx: JobContext) -> None:
        params = AsrParams.model_validate(job.params)
        model = job.model
        if model is None:  # unreachable: resolve_model requires one
            raise JobError("model_required", f"{self.name} needs a model")
        audio = self._one_input(ctx)

        try:
            self._refuse_what_this_engine_has_not_got(model, params)
            ffmpeg = _require_ffmpeg()
            _, spec, python, weights_dir, aligner = self._require_runnable(
                model, params
            )
            # The card can change between the queue and the lane, so the guard
            # runs again here against the same rules.
            state = accelerator.guard(
                self._config.backend_kind,
                model_id=model,
                need_bytes=self._need_bytes(spec, params),
                owned_pids=self._owned_pids(),
                desktop_allowance_bytes=self._config.desktop_allowance_bytes,
            )
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None
        ctx.warming(state.detail)

        # Zeros rather than absent fields: every `stage` line carries the same
        # three numbers, so a consumer reads one shape and never has to ask
        # whether this particular event happens to have them. A `total_s` of 0
        # is what "the container has not been probed yet" looks like, and it is
        # what BookForge's own decode line reports before ffprobe answers.
        ctx.progress(
            0.0,
            f"decoding {audio.name}",
            stage="decoding",
            processed_s=0.0,
            total_s=0.0,
            cues=0,
        )

        if spec.engine in QWEN_ASR_ENGINES:
            document = qwen.QwenAsrRun(
                config=self._config,
                backend=self._backend,
                ctx=ctx,
                job=job,
                model=model,
                spec=spec,
                python=python,
                weights_dir=weights_dir,
                aligner=aligner,
                ffmpeg=ffmpeg,
                audio=audio,
                language=params.language,
                context=params.context,
                word_timestamps=params.word_timestamps,
            ).run()
        else:
            document = self._whisper(
                ctx, job, model, spec, python, weights_dir, ffmpeg, audio, params
            )

        path = ctx.scratch / "transcript.json"
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        ctx.artifact("transcript.json", path)
        ctx.progress(
            1.0,
            f"{len(document['segments'])} segments over "
            f"{document['duration_s']:.0f}s of audio",
            stage="transcribing",
            processed_s=document["duration_s"],
            total_s=document["duration_s"],
            cues=len(document["segments"]),
        )

    def _whisper(
        self,
        ctx: JobContext,
        job: Job,
        model: str,
        spec: Any,
        python: Path,
        weights_dir: Path,
        ffmpeg: str,
        audio: Path,
        params: AsrParams,
    ) -> dict[str, Any]:
        """The whisper engines' run: one worker, 900-second windows, one exit."""
        outcome = self._transcribe(
            ctx, job, spec.engine, python, weights_dir, ffmpeg, audio, params
        )

        windows = outcome.ready["windows"]
        try:
            results = workers.require_positional_results(outcome, windows, "window")
        except workers.WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None

        failures = [
            f"window {index} ({index * WINDOW_SECONDS}s): {result['error']}"
            for index, result in enumerate(results)
            if "error" in result
        ]
        if failures:
            # No artifact. A transcript with a fifteen-minute hole in the middle
            # looks exactly like a transcript without one.
            raise JobError(
                "asr_window_failed",
                f"{len(failures)} of {windows} window(s) failed, so the transcript "
                "would have holes in it and nothing in the file would say where: "
                + "; ".join(failures),
            )

        return self._transcript(model, spec, params, outcome, results)

    @staticmethod
    def _one_input(ctx: JobContext) -> Path:
        """The single audio file, or a refusal naming what arrived instead."""
        inputs = ctx.inputs()
        if len(inputs) != 1:
            raise JobError(
                "invalid_inputs",
                f"an asr job takes exactly one audio file; this one has "
                f"{len(inputs)}: {sorted(inputs)}",
            )
        return next(iter(inputs.values()))

    def _transcribe(
        self,
        ctx: JobContext,
        job: Job,
        engine: str,
        python: Path,
        weights_dir: Path,
        ffmpeg: str,
        audio: Path,
        params: AsrParams,
    ) -> workers.WorkerOutcome:
        request = {
            "model_dir": str(weights_dir),
            "ffmpeg": ffmpeg,
            "audio": str(audio),
            "language": params.whisper_language(),
            "vad_filter": params.vad_filter,
            "word_timestamps": params.word_timestamps,
            # Always sent, null included: the worker's wire has no optional
            # keys. Applied to EVERY window — see "The initial prompt and the
            # windows" in this module's docstring.
            "initial_prompt": params.initial_prompt,
            "device": _for_engine(DEVICE_FOR_ENGINE, engine, "device"),
            "compute_type": _for_engine(
                COMPUTE_TYPE_FOR_ENGINE, engine, "compute_type"
            ),
            "window_s": WINDOW_SECONDS,
            "overlap_s": OVERLAP_SECONDS,
        }

        def on_ready(message: dict[str, Any]) -> None:
            ctx.warming(
                f"{message['duration_s']:.0f}s of audio decoded, "
                f"{message['windows']} window(s) of {WINDOW_SECONDS}s to transcribe "
                f"on {message['device']} at {message['compute_type']}"
            )

        def on_progress(message: dict[str, Any]) -> None:
            processed = float(message["processed_s"])
            total = float(message["total_s"])
            stage = message["stage"]
            # The decode phase drives no fraction. It is real work with a real
            # position, but none of the transcript exists yet, and a bar that
            # counts the decode as progress towards the transcript is a bar that
            # lies. `stage` plus the two second counts is exactly what
            # BookForge's own parser reads off its DECODE and PROGRESS lines.
            fraction = 0.0 if stage == "decoding" else (
                min(1.0, processed / total) if total > 0 else 0.0
            )
            ctx.progress(
                fraction,
                f"{stage} {processed:.0f}s of {total:.0f}s, "
                f"{message['cues']} segments",
                stage=stage,
                processed_s=processed,
                total_s=total,
                cues=message["cues"],
            )

        try:
            return workers.run_worker(
                python=python,
                script=_for_engine(
                    WORKER_SCRIPT_FOR_ENGINE, engine, "worker script"
                ),
                request=request,
                # The CUDA libraries pip put inside this env, on the loader
                # path. ctranslate2 resolves cuBLAS at the first matrix
                # multiply rather than at load, so without this the model
                # constructs fine and the first window fails with
                # "Library libcublas.so.12 is not found or cannot be loaded"
                # — measured on owens-pc against a doctor reporting ready.
                # See `workerenv.worker_environment`.
                environment=workerenv.worker_environment(
                    workerenv.worker_env_dir(self._config.home, JOB_TYPE)
                ),
                log_path=self._config.logs_dir / f"asr-{job.id}.log",
                ready_silence_timeout=READY_SILENCE_TIMEOUT_SECONDS,
                on_ready=on_ready,
                on_progress=on_progress,
                cancelled=lambda: ctx.cancelled,
            )
        except workers.WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None

    # ----------------------------------------------------------- transcript

    @staticmethod
    def _transcript(
        model: str,
        spec: Any,
        params: AsrParams,
        outcome: workers.WorkerOutcome,
        results: tuple[dict[str, Any], ...],
    ) -> dict[str, Any]:
        """Window-relative results into one absolute, deduplicated document.

        A result's window is its **position** in the stream and nothing else —
        the worker reports no index, because an index a worker reports is an
        index a worker can get wrong. Window `n` starts at `n * WINDOW_SECONDS`
        by construction, so the shift is arithmetic the server does.
        """
        segments: list[dict[str, Any]] = []
        for index, result in enumerate(results):
            offset = index * float(WINDOW_SECONDS)
            for segment in result["segments"]:
                shifted = dict(segment)
                shifted["start"] = segment["start"] + offset
                shifted["end"] = segment["end"] + offset
                if "words" in segment:
                    shifted["words"] = [
                        {
                            **word,
                            "start": word["start"] + offset,
                            "end": word["end"] + offset,
                        }
                        for word in segment["words"]
                    ]
                segments.append(shifted)

        # Each window reaches OVERLAP_SECONDS past its own boundary, so the last
        # seconds of every window are spoken again at the start of the next one.
        # Sort by start and drop anything that begins inside a span already kept.
        segments.sort(key=lambda row: row["start"])
        deduplicated: list[dict[str, Any]] = []
        for segment in segments:
            if (
                deduplicated
                and segment["start"]
                < deduplicated[-1]["end"] - OVERLAP_TOLERANCE_SECONDS
            ):
                continue
            deduplicated.append(segment)

        # Every window detects the language independently when none was given.
        # The first window's answer is the document's, because that is the one
        # BookForge shows and the one a re-run reproduces; the rest are the same
        # answer on any real book and the disagreement is not something Crucible
        # is in a position to adjudicate.
        first = results[0]
        return {
            "model": model,
            "revision": spec.revision,
            "hf_repo": spec.hf_repo,
            "language": first["language"],
            "language_probability": first["language_probability"],
            "language_requested": params.language,
            "vad_filter": params.vad_filter,
            "word_timestamps": params.word_timestamps,
            # null when none was sent, so the document names the rule it was
            # made under whichever way the client spelled "no prompt".
            "initial_prompt": params.initial_prompt,
            "duration_s": outcome.ready["duration_s"],
            "window_s": WINDOW_SECONDS,
            "overlap_s": OVERLAP_SECONDS,
            "windows": outcome.ready["windows"],
            "segments": deduplicated,
        }
