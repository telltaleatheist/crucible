"""The `align` job type: chunks of audio and their text in, timestamps out.

PHASE4-AUDIO.md section 2. The easiest type in phase 4 and the one that unblocks
a feature which **cannot run on Owen's PC at all today**: the library is on `Z:`
(`\\\\TITAN\\iO`), WSL cannot mount a network drive, and
`whisperx-align-bridge.ts:642` refuses whole-m4b alignment by name when the qwen
env is `viaWsl`. Bytes cross the wire here, so the filesystem stops being part of
the contract.

The one structurally new thing: the model stays resident
--------------------------------------------------------
`asr` loads a model, transcribes one file and exits. An aligner that did that
would read 1.7 GB of weights per chunk, and a book is hundreds of chunks. So this
is the first *worker* that outlives a job — `workers.WorkerSession`, held in
`Residency` beside a model and a voice as a third kind of resident thing
(`crucible/residency.py`). A card holds one thing; loading an aligner unloads a
voice and vice versa, which is not a courtesy between modules but the reason the
holder is generalised at all.

It is a third KIND and not a third engine. An LLM and a voice are servers with a
lifecycle of start/ready/stop; the aligner is the same JSON-lines worker every
phase 4 type talks to, kept open. It has no `base_url`, nothing to proxy to and
no readiness route, so it gets its own resident row rather than three null fields
in somebody else's.

What is the client's and what is the server's
---------------------------------------------
The client says *what to align*: the model, the language, and one `{index, text}`
per chunk with one audio input each. Everything about *how* is the server's and
appears nowhere on the wire — the 16 kHz mono float32 decode, `bfloat16`, the
device, and the 300-second ceiling.

Four hard facts, none of them wire parameters:

- **16 kHz mono float32, decoded with ffmpeg.** The sample rate is what the
  feature extractor was trained at; a client should not have to know it exists.
- **`bfloat16`**, from the manifest, because that is what the model card runs and
  what the bake-off measured.
- **`QWEN3_MAX_AUDIO_S = 300` is a REFUSAL, not a split.** The model card's own
  limit is timestamps "within up to 5 minutes". Chunking a longer clip here would
  silently change the alignment, so a chunk past it comes back as a failed chunk
  naming its duration.
- **Eleven languages, and an unknown one is refused before the job is queued.**
  The model does not fall back to English for a language it was not trained on;
  it places words badly, and a silently mis-aligned book is worse than a refused
  one.

**No retries and no second backend, ever.** A failed chunk is reported, the run
continues, and no other aligner is tried. `--backend qwen3` is hardcoded in the
app for exactly this reason; here it is the manifest's engine and there is
nothing to fall back to.

What stays in BookForge, and it is most of the value
----------------------------------------------------
The item-to-word mapping, the `_normalized` letter-sequence equality check that
refuses a model which rewrote the text (`aligner.py:681`), every derived score,
and the coverage report. Crucible returns one timestamped item per *its own*
tokenization — 665 items for a 668-word English window, measured in the
`qwen-align` env on 2026-09-08 — and asserts nothing about words. A server that
started asserting things about words would be a server that had opinions about
audiobooks.

A `cue` per chunk, so a killed run costs only what it had not reached
---------------------------------------------------------------------
Each chunk's items are emitted as a `cue` event the moment they land, as well as
being collected into `alignment.json` at the end. A 1,400-chunk book is hours;
the artifact is all-or-nothing and the cues are not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ... import accelerator, hosttools, weights, workerenv, workers
from ...alignmodels import (
    AlignBackendSpec,
    AlignManifest,
    AlignManifestError,
    load_all_align_manifests,
)
from ...backend import CUDA_LINUX, MLX_DARWIN
from ...config import Config
from ...errors import ApiError, JobCancelled, JobError
from ...manifests import fingerprint
from ...residency import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    KIND_ALIGN,
    Residency,
    describe_resident,
)
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor

__all__ = ["AlignJobType", "AlignParams", "UnloadAlignerJobType"]

JOB_TYPE = "align"

#: The model card's own limit: it "supports timestamp prediction ... within up to
#: 5 minutes". **A refusal, not a split.** Narrator chunks are around 90 s and the
#: corpus cutter windows to five minutes itself, so anything past this is a
#: caller's bug; chunking it here would move every timestamp after the cut and
#: nothing in the output would say so (`python/narrator/align/aligner.py:569`).
QWEN3_MAX_AUDIO_S = 300.0

#: THE WHOLE SUPPORTED LANGUAGE LIST: ISO code -> the English NAME the model's
#: `align(language=...)` takes. Checked before the job is queued rather than
#: inside the worker, because the alternative is a model load and a book's worth
#: of badly placed words (`aligner.py:575`). The model does not fall back to
#: English for a language it was not trained on — it just places words badly.
QWEN3_LANGUAGES: dict[str, str] = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "yue": "Cantonese",
}

#: What `device_map=` is given, per backend. The DTYPE comes off the manifest —
#: it is a property of what the bake-off measured — and this is the other half:
#: which piece of silicon torch is told to put the checkpoint on.
#:
#: There is no CPU entry because there is no CPU backend, and there is no
#: default: a backend nobody has decided a device for must be a refusal naming
#: it, not a silent `cuda` handed to a Mac. The names are torch's own, and
#: `mps` is the one narrator already uses on this machine
#: (`python/narrator/align/aligner.py` lists it in `GPU_DEVICES` and picks
#: float32 only on `cpu`, bfloat16 on `mps` exactly as on `cuda`).
DEVICE_FOR_BACKEND: dict[str, str] = {
    CUDA_LINUX: "cuda",
    MLX_DARWIN: "mps",
}


def device_for(backend_kind: str) -> str:
    """The torch device this backend aligns on. Refuses an unknown backend."""
    found = DEVICE_FOR_BACKEND.get(backend_kind)
    if found is None:
        raise JobError(
            "backend_unsupported",
            f"there is no align device for backend {backend_kind!r}; this build "
            f"aligns on {sorted(DEVICE_FOR_BACKEND)}",
        )
    return found


#: How long the server waits on a worker that has said *nothing at all* before it
#: gives up on it. Not a run deadline: every message resets it. 900 s covers
#: reading 1.7 GB of weights from a cold disk, which is the only part of an align
#: session that is ever quiet for long.
READY_SILENCE_TIMEOUT_SECONDS = 900.0

WORKER_SCRIPT = Path(__file__).resolve().parent / "worker.py"


class AlignChunk(BaseModel):
    """One chunk: the index its audio is named after, and its spoken text."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    text: str

    @field_validator("text")
    @classmethod
    def not_empty(cls, value: str) -> str:
        if value.strip() == "":
            raise ValueError(
                "a chunk's text is empty; a forced aligner places the text it is "
                "given and there is nothing here to place"
            )
        return value


class AlignParams(BaseModel):
    """`params` for an align job. Unknown keys are refused, not ignored."""

    model_config = ConfigDict(extra="forbid")

    language: str
    chunks: list[AlignChunk] = Field(min_length=1)

    @field_validator("language")
    @classmethod
    def known_language(cls, value: str) -> str:
        if value in QWEN3_LANGUAGES:
            return value
        raise ValueError(
            f"{value!r} is not a language Qwen3-ForcedAligner supports; it takes "
            f"one of {sorted(QWEN3_LANGUAGES)}. It does not fall back to English "
            "for a language it was not trained on — it places words badly, and a "
            "silently mis-aligned book is worse than a refused one"
        )

    @field_validator("chunks")
    @classmethod
    def unique_indexes(cls, value: list[AlignChunk]) -> list[AlignChunk]:
        seen = [chunk.index for chunk in value]
        duplicates = sorted({index for index in seen if seen.count(index) > 1})
        if duplicates:
            raise ValueError(
                f"chunk index {duplicates} appears more than once; an index names "
                "one chunk and one input file"
            )
        return value

    def model_language(self) -> str:
        """The English name `model.align` takes, from the ISO code the client sent."""
        return QWEN3_LANGUAGES[self.language]


# ------------------------------------------------------------------ helpers


def _manifests() -> dict[str, AlignManifest]:
    try:
        return load_all_align_manifests()
    except AlignManifestError as exc:
        raise ApiError(
            500,
            "align_manifests_unreadable",
            f"this server cannot read its align manifests: {exc}",
        ) from None


def _known(model_id: str) -> AlignManifest:
    manifests = _manifests()
    manifest = manifests.get(model_id)
    if manifest is None:
        raise ApiError(
            400,
            "unknown_model",
            f"no align manifest for {model_id!r}; this build ships {sorted(manifests)}",
        )
    return manifest


def _params(model: type[BaseModel], params: dict[str, Any], job_type: str) -> Any:
    try:
        return model.model_validate(params)
    except ValidationError as exc:
        raise ApiError(
            400,
            "invalid_params",
            f"{job_type} params are not valid: "
            + "; ".join(
                f"{'.'.join(str(p) for p in problem['loc']) or '<root>'}: "
                f"{problem['msg']}"
                for problem in exc.errors()
            ),
        ) from None


def ffmpeg_path() -> str | None:
    """Where ffmpeg is on this host, or None.

    A module-level probe, for the reason `crucible/accelerator.py` gives about
    its own: a test replaces it and asserts on the refusal, instead of asserting
    on whatever happens to be installed on the machine running the suite. The
    search goes through `crucible/hosttools.py`, the one owner of *which PATH
    was searched* — which is what the refusals below have to name.
    """
    return hosttools.which("ffmpeg")


def _require_ffmpeg() -> str:
    found = ffmpeg_path()
    if found is None:
        raise ApiError(
            409,
            "ffmpeg_missing",
            "there is no ffmpeg on this server's PATH, and align decodes every "
            "chunk through it to 16 kHz mono float32 — the rate the model's "
            "feature extractor was trained at, which is why it is not something a "
            "client is asked to do. " + hosttools.searched_note(),
            {"path": hosttools.search_path()},
        )
    return found


def _align_provenance(backend_kind: str, model: str | None) -> dict[str, Any] | None:
    """The `model` block of an alignment's provenance sidecar.

    An `alignment.json` that does not name the weights that produced it is a set
    of timestamps nobody can re-derive, and two revisions of one aligner are two
    sets of timestamps.
    """
    if model is None:
        return None
    manifest = _known(model)
    spec = manifest.backends.get(backend_kind)
    if spec is None:
        # Unreachable through the API: `preflight` refuses `backend_unsupported`
        # before a job exists. A sidecar still has to say something true if it is
        # reached another way, and inventing a revision is not it.
        return {"id": model, "revision": None, "fingerprint": None}
    return {
        "id": model,
        "revision": spec.revision,
        "fingerprint": fingerprint(model, spec.revision),
    }


def _descriptors(config: Config, residency: Residency) -> list[ModelDescriptor]:
    backend_kind = config.backend_kind
    rows: list[ModelDescriptor] = []
    for manifest in _manifests().values():
        if manifest.supports(backend_kind):
            spec = manifest.spec(backend_kind)
            revision, source, estimate = (
                spec.revision,
                spec.hf_repo,
                spec.memory_bytes_estimate,
            )
            # The same predicate `check` and `_require_loadable` read: the
            # puller's stamp, at the revision this host's block pins.
            installed = weights.installed(config, manifest, spec) is not None
        else:
            # A backend this manifest has no block for has nothing to install.
            revision, source, estimate, installed = "", "", 0, False
        rows.append(
            ModelDescriptor(
                id=manifest.id,
                revision=revision,
                source=source,
                installed=installed,
                # Unlike `asr`, this one can be true: the aligner stays on the
                # card between jobs, which is the whole point of section 2.
                resident=residency.is_resident(KIND_ALIGN, manifest.id),
                vram_bytes=estimate,
            )
        )
    return rows


# ------------------------------------------------------------------ job type


class AlignJobType:
    """`POST /v1/jobs {"type": "align", "model": "qwen3-aligner", "inputs": {...}}`."""

    name = JOB_TYPE

    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency

    @property
    def residency(self) -> Residency:
        return self._residency

    # ----------------------------------------------------------- describing

    def describe_models(self) -> list[ModelDescriptor]:
        return _descriptors(self._config, self._residency)

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        if model is None:
            raise JobError("model_required", f"{self.name} needs a model")
        return _align_provenance(self._config.backend_kind, model)

    def vram_estimate(self, model: str | None) -> int:
        if model is None:
            raise JobError("model_required", f"{self.name} needs a model")
        manifest = _known(model)
        if not manifest.supports(self._config.backend_kind):
            return 0
        return manifest.spec(self._config.backend_kind).memory_bytes_estimate

    def check(self, backend: Any) -> JobTypeStatus:
        try:
            env = workerenv.env_status(self._config.home, JOB_TYPE, backend.kind)
        except workerenv.WorkerEnvError as exc:
            return JobTypeStatus(ready=False, detail=str(exc))
        if not env.installed:
            return JobTypeStatus(ready=False, detail=env.detail)
        if ffmpeg_path() is None:
            return JobTypeStatus(
                ready=False,
                detail=f"{env.detail}; but there is no ffmpeg on PATH, and align "
                "decodes every chunk through it. " + hosttools.searched_note(),
            )
        try:
            manifests = _manifests()
        except ApiError as exc:
            return JobTypeStatus(ready=False, detail=exc.message)
        installed = [
            manifest.id
            for manifest in manifests.values()
            if manifest.supports(backend.kind)
            and weights.installed(self._config, manifest, manifest.spec(backend.kind))
            is not None
        ]
        if not installed:
            return JobTypeStatus(
                ready=False,
                detail=(
                    f"{env.detail}; no aligner is installed — "
                    "`crucible models pull qwen3-aligner`"
                ),
            )
        return JobTypeStatus(ready=True, detail=f"{env.detail}; installed: {installed}")

    # ------------------------------------------------------------ preflight

    def _require_runnable(
        self, model_id: str
    ) -> tuple[AlignManifest, AlignBackendSpec, Path, Path]:
        """Manifest, spec, env python and weights dir, or the named refusal.

        The order is `llm`'s and for `llm`'s reason: what no amount of installing
        can fix first, then what an install or a pull would fix, then the live
        accelerator.
        """
        backend_kind = self._backend.kind
        manifest = _known(model_id)
        if not manifest.supports(backend_kind):
            raise ApiError(
                400,
                "backend_unsupported",
                f"aligner {model_id!r} has no {backend_kind} block; "
                f"{manifest.path.name} declares {sorted(manifest.backends)}",
                {
                    "model": model_id,
                    "backend": backend_kind,
                    "declared": sorted(manifest.backends),
                },
            )
        spec = manifest.spec(backend_kind)
        accelerator.refuse_if_larger_than_host(
            model_id=model_id,
            need_bytes=spec.memory_bytes_estimate,
            host_total_bytes=self._backend.gpu.vram_bytes,
            host_name=self._backend.gpu.name,
        )
        try:
            python = workerenv.require_env(self._config.home, JOB_TYPE, backend_kind)
        except workerenv.WorkerEnvError as exc:
            raise ApiError(
                409,
                "env_missing",
                f"cannot run {model_id!r}: {exc}",
                {
                    "model": model_id,
                    "env": str(workerenv.worker_env_dir(self._config.home, JOB_TYPE)),
                },
            ) from None
        try:
            installed = weights.require_installed(self._config, manifest, spec)
        except weights.WeightsError as exc:
            raise ApiError(
                409,
                "model_not_installed",
                str(exc),
                {"model": model_id, "hf_repo": spec.hf_repo, "revision": spec.revision},
            ) from None
        return manifest, spec, python, installed.path

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        if model is None:  # unreachable: resolve_model requires one
            raise ApiError(400, "model_required", f"{self.name} needs a model")
        _params(AlignParams, params, self.name)
        _require_ffmpeg()
        _, spec, _, _ = self._require_runnable(model)
        # A streaming session holds the resident engine without occupying the
        # lane, so a free lane is not a free card (crucible/residency.py). An
        # align job that finds a voice resident LOADS OVER IT — `reclaimable_bytes`
        # below counts that voice as free memory, exactly as an llm load does —
        # and `Residency._refuse_mutation_if_claimed` is what stops it taking a
        # session's voice off the card mid-sentence.
        #
        # That backstop turns the job into a `failed` a minute later, which is the
        # shape this method exists to avoid: "refuse, by name, before the job is
        # queued" (`JobType.preflight`). `llm` and `tts` have asked this question
        # here since PHASE3-TTS.md section 7; `align` became a third mutator of
        # residency in phase 4 and did not inherit it. Admission is the server's
        # one scheduling answer (ARCHITECTURE.md section 3), so it is given here,
        # in full, in one place.
        self._residency.refuse_if_claimed(f"aligning with {model!r}")
        if self._residency.is_resident(KIND_ALIGN, model):
            # Already on the card and about to be reused. Running the guard would
            # refuse the job for memory the resident aligner is itself holding.
            return
        accelerator.guard(
            self._config.backend_kind,
            model_id=model,
            need_bytes=spec.memory_bytes_estimate,
            owned_pids=self._residency.owned_pids(),
            desktop_allowance_bytes=self._config.desktop_allowance_bytes,
            # An align load may unload the previous resident to make room for
            # itself, exactly as an llm load may — one card, one thing — so what
            # that resident holds counts as free.
            reclaimable_bytes=self._residency.reclaimable_bytes(excluding=model),
        )

    # ------------------------------------------------------------------ run

    def run(self, job: Job, ctx: JobContext) -> None:
        params = AlignParams.model_validate(job.params)
        model = job.model
        if model is None:  # unreachable: resolve_model requires one
            raise JobError("model_required", f"{self.name} needs a model")

        try:
            ffmpeg = _require_ffmpeg()
            manifest, spec, python, weights_dir = self._require_runnable(model)
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None

        audio = self._chunk_inputs(ctx, params)
        session = self._session(ctx, manifest, spec, weights_dir, python, model)

        request = {
            "op": "align",
            "language": params.model_language(),
            "max_audio_s": QWEN3_MAX_AUDIO_S,
            "ffmpeg": ffmpeg,
            # No index in a chunk and none in a result: position is the whole
            # identity, because an index a worker reports is an index a worker
            # can get wrong.
            "chunks": [
                {"audio": str(audio[chunk.index]), "text": chunk.text}
                for chunk in params.chunks
            ],
        }

        total = len(params.chunks)
        landed: list[dict[str, Any]] = []

        def on_progress(message: dict[str, Any]) -> None:
            processed = int(message["processed"])
            ctx.progress(
                min(1.0, processed / total),
                f"aligned {processed} of {total} chunk(s)",
                stage=message["stage"],
                processed=processed,
                total=total,
            )

        try:
            outcome = session.send(
                request,
                ready_silence_timeout=READY_SILENCE_TIMEOUT_SECONDS,
                on_progress=on_progress,
                cancelled=lambda: ctx.cancelled,
            )
        except workers.WorkerError as exc:
            # The session is dead or the worker broke the protocol. Take the
            # aligner off the card: `Residency` must not go on advertising a
            # resident thing whose process has gone.
            self._forget(ctx, model)
            raise JobError("worker_failed", str(exc)) from None
        except JobCancelled:
            # A cancel stops the worker mid-exchange, so the session is gone too
            # — `WorkerSession.send` discards it rather than hand the next job a
            # stream it can no longer parse. The resident row has to go with it,
            # or `/v1/health` advertises an aligner that is not there until some
            # later job notices and reloads.
            self._forget(ctx, model)
            raise

        try:
            results = workers.require_positional_results(outcome, total, "chunk")
        except workers.WorkerError as exc:
            self._forget(ctx, model)
            raise JobError("worker_failed", str(exc)) from None

        # A result is matched to its chunk by POSITION — the worker reported no
        # index at all — so the client's index comes back out of the params, in
        # the order the params listed them.
        for chunk, result in zip(params.chunks, results):
            row: dict[str, Any] = {"index": chunk.index}
            if "error" in result:
                row["error"] = result["error"]
            else:
                row["items"] = result["items"]
            landed.append(row)
            # The cue goes out as the chunk lands, so a run killed at chunk 900
            # of 1,400 has cost the client the 500 it had not reached and not the
            # 900 it had. A failed chunk gets a cue too, carrying `error` instead
            # of `items`, because a client watching this stream should learn
            # about the failure at the same moment as the successes around it.
            ctx.cue(row)

        document = {
            "model": model,
            "revision": spec.revision,
            "hf_repo": spec.hf_repo,
            "engine": spec.engine,
            "dtype": spec.dtype,
            "device": device_for(self._config.backend_kind),
            "language": params.language,
            "language_name": params.model_language(),
            "max_audio_s": QWEN3_MAX_AUDIO_S,
            "sample_rate": 16_000,
            # Said explicitly rather than left for a reader to infer from the
            # shape: these are the MODEL's tokens, not the caller's words. The
            # mapping onto words, the letter-sequence equality check and every
            # derived score stay in the client (PHASE4-AUDIO.md section 2).
            "items_are": "the model's own tokenization, not the caller's words",
            "chunks": landed,
        }
        path = ctx.scratch / "alignment.json"
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        ctx.artifact("alignment.json", path)

        failed = [row["index"] for row in landed if "error" in row]
        ctx.progress(
            1.0,
            f"{total - len(failed)} of {total} chunk(s) aligned"
            + (f", {len(failed)} failed: {failed}" if failed else ""),
            stage="aligning",
            processed=total,
            total=total,
        )
        ctx.done_extra(
            chunks=total, failed=failed, resident=self._residency.resident_id
        )

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _chunk_inputs(ctx: JobContext, params: AlignParams) -> dict[int, Path]:
        """`{index: path}`, or a refusal naming exactly what did not line up.

        One input per chunk, named `<index>.<ext>`. Both directions are checked:
        a chunk with no audio cannot be aligned, and an input with no chunk is a
        client that thinks it sent a chunk it did not. Neither is a thing to
        quietly drop — a book aligned with 1,399 of its 1,400 chunks reads as a
        complete answer.
        """
        inputs = ctx.inputs()
        by_index: dict[int, Path] = {}
        unnamed: list[str] = []
        for name, path in inputs.items():
            stem = Path(name).stem
            try:
                by_index[int(stem)] = path
            except ValueError:
                unnamed.append(name)
        if unnamed:
            raise JobError(
                "invalid_inputs",
                f"input(s) {sorted(unnamed)} are not named <index>.<ext>; an align "
                "input is matched to its text by the index in its filename",
            )
        wanted = {chunk.index for chunk in params.chunks}
        missing = sorted(wanted - set(by_index))
        extra = sorted(set(by_index) - wanted)
        if missing or extra:
            raise JobError(
                "invalid_inputs",
                "the chunks and the inputs do not line up: "
                + "; ".join(
                    part
                    for part in (
                        f"chunk(s) {missing} have no audio" if missing else "",
                        f"input(s) {extra} have no chunk" if extra else "",
                    )
                    if part
                ),
            )
        return by_index

    def _session(
        self,
        ctx: JobContext,
        manifest: AlignManifest,
        spec: AlignBackendSpec,
        weights_dir: Path,
        python: Path,
        model: str,
    ) -> workers.WorkerSession:
        """The resident aligner's worker, loading it first if it is not there.

        A load here rather than through a `load-aligner` job, and that is the one
        place this type departs from `llm` and `tts`. Those two are loaded by an
        explicit job because a client chooses *when* to spend 200 s of warm-up
        and against what else is queued. An aligner load is 20 s and is always
        immediately followed by the work it was loaded for, so making a client
        send two jobs to align one book would be ceremony. Taking it OFF the card
        is still an explicit door (`unload-aligner`), because that is a decision
        about somebody else's next job rather than about this one.
        """
        session = self._residency.aligner_session
        if session is not None and self._residency.is_resident(KIND_ALIGN, model):
            if session.alive:
                return session
            # The resident row outlived its process — the worker died between
            # jobs. Say so rather than sending a request into a closed pipe.
            ctx.warming(
                f"the resident {model} worker is gone (its log is "
                f"{session.log_path}); loading it again"
            )
            self._forget(ctx, model)

        try:
            state = accelerator.guard(
                self._config.backend_kind,
                model_id=model,
                need_bytes=spec.memory_bytes_estimate,
                owned_pids=self._residency.owned_pids(),
                desktop_allowance_bytes=self._config.desktop_allowance_bytes,
                reclaimable_bytes=self._residency.reclaimable_bytes(excluding=model),
            )
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None
        ctx.warming(state.detail)

        try:
            self._residency.load_aligner(
                manifest,
                spec,
                weights_dir,
                python,
                WORKER_SCRIPT,
                device=device_for(self._config.backend_kind),
                dtype=spec.dtype,
                max_audio_s=QWEN3_MAX_AUDIO_S,
                timeout=DEFAULT_READY_TIMEOUT_SECONDS,
                on_progress=ctx.warming,
            )
        except workers.WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None
        loaded = self._residency.aligner_session
        if loaded is None:  # pragma: no cover - load_aligner publishes or raises
            raise JobError(
                "worker_failed",
                f"{model} loaded but no session was published; this is a bug in "
                "crucible/residency.py",
            )
        return loaded

    def _forget(self, ctx: JobContext, model: str) -> None:
        """Take a dead aligner off the card without letting the tidy-up win.

        The failure being reported is the worker's, and a `stop()` that also
        fails must not replace it — the caller is about to raise the one error
        that explains what happened.

        NOT RAISING IS NOT THE SAME AS NOT SAYING, and until 2026-09-18 this
        did both. A `WorkerError` here is a worker that did not go on SIGTERM:
        `Residency.unload` unpublishes before it stops, so the resident row is
        gone and the process is not, and the card is held by something no row
        points at. That is exactly the fact a reader chasing a card that will
        not free needs, and it has nowhere else to appear. So it is said the
        way every other cleanup failure on this server is — a line in the log
        for whoever is watching the server, and a `note` on the stream of the
        job it happened to (`Settlement.settle_quietly`, `JobStore._settle`).
        """
        try:
            self._residency.unload(model)
        except (KeyError, workers.WorkerError) as exc:
            line = f"could not take {model} off the card: {type(exc).__name__}: {exc}"
            print(f"crucible: {line}", file=sys.stderr)
            ctx.note(line)


# --------------------------------------------------------------- unload job


class UnloadAlignerParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UnloadAlignerJobType:
    """`POST /v1/jobs {"type": "unload-aligner", "model": "<aligner id>"}`.

    The other half of section 2's residency, and the reason there is no
    `load-aligner` beside it is in `AlignJobType._session`. Without this door an
    aligner could only be taken off the card by loading something else, which
    would make "one card, one thing" a rule you can only obey by breaking it.
    """

    name = "unload-aligner"

    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency

    @property
    def residency(self) -> Residency:
        return self._residency

    def describe_models(self) -> list[ModelDescriptor]:
        return _descriptors(self._config, self._residency)

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        return _align_provenance(self._config.backend_kind, model)

    def vram_estimate(self, model: str | None) -> int:
        return 0

    def check(self, backend: Any) -> JobTypeStatus:
        aligner = self._residency.resident_aligner
        return JobTypeStatus(
            ready=True,
            detail=(
                f"resident: {aligner.aligner_id}"
                if aligner
                else "no aligner is resident"
            ),
        )

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        if model is None:  # unreachable: resolve_model requires one
            raise ApiError(400, "model_required", f"{self.name} needs an aligner")
        _params(UnloadAlignerParams, params, self.name)
        if self._residency.being_cleared(model):
            # And the same exception they make (T6, 2026-09-15): the settlement
            # clearing this very aligner is not a second holder, it is this
            # request already happening.
            return
        # The same refusal `unload-model` and `unload-voice` already make: taking
        # anything off the card while somebody holds it ends their conversation
        # mid-sentence, and `Residency.unload` refuses it anyway — from inside the
        # job, where it is a `failed` rather than an answer to the request.
        self._residency.refuse_if_claimed(f"unloading {model!r}")
        if not self._residency.is_resident(KIND_ALIGN, model):
            raise ApiError(
                409,
                "aligner_not_resident",
                f"{model!r} is not resident on this server; "
                + describe_resident(self._residency, KIND_ALIGN, "no aligner is"),
                {"requested": model, "resident": self._residency.resident_id},
            )

    def run(self, job: Job, ctx: JobContext) -> None:
        UnloadAlignerParams.model_validate(job.params)
        model = job.model
        if model is None:  # unreachable: resolve_model requires one
            raise JobError("model_required", f"{self.name} needs an aligner")
        if self._residency.await_clearance(model):
            # The settlement got there first, which is the card this job asked
            # for. Same terminal shape as an unload this job did itself.
            ctx.progress(0.0, f"unloading {model}")
            ctx.progress(1.0, f"{model} is unloaded — the card was cleared of it")
            ctx.done_extra(resident=self._residency.resident_id)
            return
        if not self._residency.is_resident(KIND_ALIGN, model):
            # Checked before `unload()` rather than caught from it: the holder
            # unloads by id alone, and a voice sharing an aligner's id would be
            # taken off the card by `unload-aligner`.
            raise JobError(
                "aligner_not_resident",
                f"{model!r} is not resident on this server; "
                + describe_resident(self._residency, KIND_ALIGN, "no aligner is"),
            )
        ctx.progress(0.0, f"unloading {model}")
        try:
            self._residency.unload(model)
        except workers.WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None
        ctx.progress(1.0, f"{model} is unloaded")
        ctx.done_extra(resident=self._residency.resident_id)
