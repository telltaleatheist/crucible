"""The `denoise` job type: one audio file in, its separated stems out.

PHASE4-AUDIO.md section 4.2. It is the pass BookForge runs over a session's
rendered sentences to strip the faint room hiss a fine-tuned voice reproduces —
those voices are trained on a deliberate ~-65 dBFS hiss bed, which is load-bearing
for reliable end-of-audio, so every raw render carries a hiss during speech that
cuts out at the digitally-silent assembly gaps.

It shares the `rvc` env and does not get one of its own
-------------------------------------------------------
audio-separator is torch, the rvc env already holds the exact torch it wants,
and BookForge runs the two out of one env for that reason today
(`electron/denoise-bridge.ts` reaches for the RVC env's python). So
`envs/rvc/<backend>.txt` pins `audio-separator` and there is no `envs/denoise/`;
`crucible install rvc` builds the env for both and decides both capability
flags. This is the first job type whose env belongs to another type, and the one
place that is written down as a fact rather than as a coincidence is
`workerenv.JOB_TYPES_SERVED_BY_ENV`.

What is the client's and what is the server's
---------------------------------------------
The client says **which model** and sends **one audio file at the model's native
rate**. That is the whole wire. Everything else is the server's: the separation
parameters (every one an audio-separator default, byte-identical to what
BookForge's own worker uses), `use_autocast` (CUDA-only, so it is read off the
backend), and the output format.

**Blocking is the client's**, and that is the same ruling that put chunking in
the client for `tts` (DESIGN.md section 3.1). BookForge concatenates a book's
sentences into ~22-minute blocks, denoises each, and slices the stems back at
recorded offsets; it keeps all of that. Crucible denoises one thing at a time.

The separator is RESIDENT, and one block per job is still the wire
------------------------------------------------------------------
Those two sentences are not in tension, and telling them apart is the whole of
Owen's ruling of 2026-09-15. The WIRE is one block per job, unchanged. What
changed is that the CHECKPOINT now stays on the card between jobs
(`residency.KIND_DENOISE`, the fourth resident kind, `_session` below).

This file used to argue the opposite — that "a separator loads once per job
either way", so nothing was lost. That is true of one job and false of a book:
the client sends **~44 jobs for a 15-hour book**, so the load was paid ~44 times.
BookForge had already measured it and already fixed it on its own side
(`electron/scripts/separator_worker.py`, bookforge `019afa52`): the per-block
spawn cost "10-25 s each for ~85 s of real work per block", which
`electron/denoise-bridge.ts:27-32` records as *"roughly a third of the pass. One
load now serves the whole book."* The warm figure from that commit is 7.4 s of an
11.0 s one-shot. Crucible could not see any of that, because a job type reasons
about one job and a pass is a property of the client.

`done.extra.load_seconds` is what makes it visible: the load time on the block
that loaded the separator, and `0.0` on every block after it. A pass whose blocks
all report a load is a pass that has lost the residency — which is exactly how
this hid the first time, with every job succeeding and every log clean.

Three invariants, each one BookForge's and each one measured
------------------------------------------------------------
- **The input must already be at the model's native rate.** 44.1 kHz for this
  model, whose librosa front-end crashes on anything else. Crucible does not
  resample: a stem returned at a rate the caller did not send is a stem whose
  sample offsets no longer mean anything, and the client is the one that knows
  what rate it wants back. The worker refuses by name before the model loads.
- **The primary stem comes back the same length, sample for sample.** That is
  what makes the client's offset slicing safe, and it is checked here rather
  than assumed. The other stems' figures are reported and not enforced: the
  length invariant is measured for the stem the app uses and nothing has
  measured it for the rest.
- **Exactly one output names the primary stem.** The app asserts this
  (`hits.length === 1` against `(dry)`) because zero means the model produced
  something else and two means the caller cannot tell which is the answer.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from ... import accelerator, workerenv, workers
from ...backend import CUDA_LINUX
from ...config import Config
from ...denoisemodels import (
    PULL_COMMAND,
    DenoiseBackendSpec,
    DenoiseManifest,
    DenoiseManifestError,
    denoise_models_root,
    load_all_denoise_manifests,
)
from ...denoisemodels import installed as model_installed
from ...denoisemodels import missing as missing_model_files
from ...errors import ApiError, JobError
from ...manifests import fingerprint
from ...residency import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    KIND_DENOISE,
    Residency,
    describe_resident,
)
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor

__all__ = [
    "DenoiseJobType",
    "DenoiseParams",
    "UnloadDenoiserJobType",
    "denoise_models_dir",
    "denoise_models_dir_for",
]

JOB_TYPE = "denoise"

#: The env this type runs in. Not its own: see the module docstring.
ENV_JOB_TYPE = "rvc"

#: What the separator writes. WAV because the client is slicing the result at
#: sample offsets and then resampling it back itself — a lossy intermediate
#: would put an encode and a decode in the middle of an operation whose whole
#: correctness argument is "the model returned exactly what it was given".
#: BookForge's worker passes the same thing.
OUTPUT_FORMAT = "WAV"

#: How long the server waits on a worker that has said *nothing at all*. Not a
#: run deadline: every message resets it. 900 s covers importing torch and
#: reading a 913 MB checkpoint off a cold disk.
READY_SILENCE_TIMEOUT_SECONDS = 900.0

#: What the engine's environment must say. Two of the three are the rvc env's
#: rather than audio-separator's, and they are here because the ENV is what
#: needs them: it bundles three OpenMP runtimes (torch, faiss-cpu and
#: scikit-learn each ship their own libomp), loading more than one aborts with
#: OMP Error #15, and with the duplicates co-loaded the cross-runtime barrier
#: SIGSEGVs the moment a thread pool spins up. See `jobs/rvc/__init__.py`, where
#: the same two were measured. The third keeps the worker's own lines flowing.
ENGINE_ENVIRONMENT: dict[str, str] = {
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}

WORKER_SCRIPT = Path(__file__).resolve().parent / "worker.py"


def denoise_models_dir_for(home: Path) -> Path:
    """audio-separator's `model_file_dir` under a Crucible home.

    **The layout is not decided here.** `crucible/denoisemodels.py` owns it,
    because `crucible denoise pull` has to place two files in exactly the tree
    this job reads them from, and two functions that agree today are two
    functions that can disagree tomorrow (ARCHITECTURE.md R1). This is the name
    the job side calls it by and nothing more.
    """
    return denoise_models_root(home)


def denoise_models_dir(config: Config) -> Path:
    """`denoise_models_dir_for(config.home)`. One definition, two callers."""
    return denoise_models_dir_for(config.home)


class DenoiseParams(BaseModel):
    """`params` for a denoise job: none, and that is the contract.

    Every knob audio-separator takes is an engine default BookForge measured and
    left alone, and `use_autocast` is decided by the backend rather than by the
    caller. A parameter here would be a number a client could set without any
    way to know what it does to a book — DESIGN.md section 3.1's rule, which
    says a knob crosses the seam one at a time, with a reason, and none has one
    yet.

    It is still a model with `extra="forbid"` rather than an ignored dict: a
    client that sends `{"model_filename": "..."}` is asking for something, and
    being told no by name beats being answered with something else.
    """

    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ helpers


def _manifests() -> dict[str, DenoiseManifest]:
    try:
        return load_all_denoise_manifests()
    except DenoiseManifestError as exc:
        raise ApiError(
            500,
            "denoise_manifests_unreadable",
            f"this server cannot read its denoise manifests: {exc}",
        ) from None


def _known(model_id: str) -> DenoiseManifest:
    manifests = _manifests()
    manifest = manifests.get(model_id)
    if manifest is None:
        raise ApiError(
            400,
            "unknown_model",
            f"no denoise manifest for {model_id!r}; this build ships "
            f"{sorted(manifests)}",
        )
    return manifest


def _params(params: dict[str, Any]) -> DenoiseParams:
    try:
        return DenoiseParams.model_validate(params)
    except ValidationError as exc:
        raise ApiError(
            400,
            "invalid_params",
            "denoise takes no params — every separation knob is an engine "
            "default this server does not put on the wire (PHASE4-AUDIO.md "
            "section 4.2): "
            + "; ".join(
                f"{'.'.join(str(p) for p in problem['loc']) or '<root>'}: "
                f"{problem['msg']}"
                for problem in exc.errors()
            ),
        ) from None


def _missing_files(config: Config, manifest: DenoiseManifest) -> list[str]:
    """Which of this model's two files are absent. One owner: the puller's."""
    return missing_model_files(config.home, manifest)


def _require_model_files(config: Config, manifest: DenoiseManifest) -> Path:
    """The separator's model directory, or `denoise_model_missing` by name.

    **Crucible fetches these two files itself and does not let audio-separator
    fetch them.** The library's own downloader pulls from a GitHub release,
    which DESIGN.md section 5 refuses as a source of weights, and a job that
    reached the network mid-run would be a job whose bytes nobody pinned. The
    manifest names a HuggingFace mirror at a pinned revision with a digest for
    each file, and `crucible denoise pull` is the one door that places them —
    so, as with `crucible/jobs/rvc/__init__.py`'s base assets, this refusal
    names a command that works rather than a directory to fill by hand.
    """
    root = denoise_models_dir(config)
    missing = _missing_files(config, manifest)
    if not missing:
        return root
    spec = manifest.backends.get(config.backend_kind)
    where = (
        f"{spec.hf_repo}@{spec.revision[:12]}" if spec is not None else "its upstream"
    )
    paths = (
        [spec.model_path, spec.config_path] if spec is not None else []
    )
    command = f"{PULL_COMMAND} {manifest.id}"
    raise ApiError(
        409,
        "denoise_model_missing",
        f"audio-separator needs {manifest.model_filename!r} and "
        f"{manifest.config_filename!r} in {root}, and {missing} are not there. "
        f"Crucible does not let the library fetch them: its own downloader "
        f"pulls from a GitHub release, which is not a source this server takes "
        f"weights from. The bytes are {where}, at {paths} — run "
        f"`{command}` to place them and try again",
        {
            "root": str(root),
            "missing": sorted(missing),
            "command": command,
            "hf_repo": None if spec is None else spec.hf_repo,
            "revision": None if spec is None else spec.revision,
            "paths": paths,
        },
    )


def _denoise_provenance(
    backend_kind: str, model: str | None
) -> dict[str, Any] | None:
    """The `model` block of a denoised stem's provenance sidecar.

    A denoised book that does not name the separator AND its revision is a book
    nobody can re-derive, and two checkpoints under one friendly name are two
    different results.
    """
    if model is None:
        return None
    manifest = _known(model)
    spec = manifest.backends.get(backend_kind)
    if spec is None:
        # Unreachable through the API: `preflight` refuses `backend_unsupported`
        # first. A sidecar still has to say something true if it is reached
        # another way, and inventing a revision is not it.
        return {"id": model, "revision": None, "fingerprint": None}
    return {
        "id": model,
        "revision": spec.revision,
        "fingerprint": fingerprint(model, spec.revision),
    }


# ------------------------------------------------------------------ job type


class DenoiseJobType:
    """`POST /v1/jobs {"type": "denoise", "model": "<id>", "inputs": {...}}`."""

    name = JOB_TYPE

    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        self._config = config
        self._backend = backend
        # THE HOLDER, not a bare `owned_pids` callable any more. This type now
        # loads and reuses a resident separator, so it needs the same object
        # `align` needs: the guard's owned pids come off it, and so does the
        # session it sends each block down.
        self._residency = residency
        #: What the load cost, by the job id that paid for it, so `done.extra`
        #: can report it once and then report `0.0` for every block that reused
        #: the session. Keyed by job rather than kept as one number because two
        #: jobs must never be able to claim one load.
        self._loaded_seconds: dict[str, float] = {}

    @property
    def residency(self) -> Residency:
        return self._residency

    def _owned_pids(self) -> frozenset[int]:
        """The pids Crucible's own resident thing holds.

        Kept as a method so every `accelerator.guard` call below reads the same
        way it did before the holder arrived — the guard must not report this
        server's own engine as somebody else's process on the card.
        """
        return self._residency.owned_pids()

    # ----------------------------------------------------------- describing

    def describe_models(self) -> list[ModelDescriptor]:
        backend_kind = self._config.backend_kind
        rows: list[ModelDescriptor] = []
        for manifest in _manifests().values():
            if manifest.supports(backend_kind):
                spec = manifest.spec(backend_kind)
                revision, source, estimate = (
                    spec.revision,
                    f"{spec.hf_repo}:{spec.model_path}",
                    spec.memory_bytes_estimate,
                )
                # The puller's own stamp at this pin, which is the fact a
                # puller reading this row wants: "do I need to pull". It is a
                # narrower fact than `check`'s `_missing_files`, which is
                # presence — a checkpoint somebody copied in by hand runs, and
                # `crucible denoise list` reports it as `present` and not
                # `installed` for exactly that reason. One word, one meaning.
                installed = (
                    model_installed(self._config.home, manifest, spec) is not None
                )
            else:
                # A backend this manifest has no block for has nothing to install.
                revision, source, estimate, installed = "", "", 0, False
            rows.append(
                ModelDescriptor(
                    id=manifest.id,
                    revision=revision,
                    # The repo AND the file: one repo holds every UVR model
                    # there is, so the repo alone would identify none of them.
                    source=source,
                    installed=installed,
                    # A REAL QUESTION since 2026-09-15. It used to read
                    # `resident=False` with a note saying holding the separator
                    # across jobs "would be a third kind of resident thing and a
                    # ruling nobody has made". Owen made it: a book is ~44 blocks
                    # and one load (see `_session`).
                    resident=self._residency.is_resident(KIND_DENOISE, manifest.id),
                    vram_bytes=estimate,
                )
            )
        return rows

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        if model is None:
            raise JobError("model_required", f"{self.name} needs a model")
        return _denoise_provenance(self._config.backend_kind, model)

    def vram_estimate(self, model: str | None) -> int:
        if model is None:
            raise JobError("model_required", f"{self.name} needs a model")
        manifest = _known(model)
        if not manifest.supports(self._config.backend_kind):
            return 0
        return manifest.spec(self._config.backend_kind).memory_bytes_estimate

    def check(self, backend: Any) -> JobTypeStatus:
        try:
            env = workerenv.env_status(self._config.home, ENV_JOB_TYPE, backend.kind)
        except workerenv.WorkerEnvError as exc:
            return JobTypeStatus(ready=False, detail=str(exc))
        if not env.installed:
            return JobTypeStatus(
                ready=False,
                detail=(
                    f"{env.detail} (denoise shares the rvc env — "
                    "`crucible install rvc`)"
                ),
            )
        try:
            manifests = _manifests()
        except ApiError as exc:
            return JobTypeStatus(ready=False, detail=exc.message)
        installed = [
            manifest.id
            for manifest in manifests.values()
            if manifest.supports(backend.kind) and not _missing_files(
                self._config, manifest
            )
        ]
        if not installed:
            root = denoise_models_dir(self._config)
            pullable = sorted(
                manifest.id
                for manifest in manifests.values()
                if manifest.supports(backend.kind)
            )
            return JobTypeStatus(
                ready=False,
                detail=(
                    f"{env.detail}; but no separator checkpoint is in {root} — "
                    f"`{PULL_COMMAND} <id>` fetches one, and this build ships "
                    f"{pullable}"
                ),
            )
        return JobTypeStatus(
            ready=True, detail=f"{env.detail}; installed: {installed}"
        )

    # ------------------------------------------------------------ preflight

    def _require_runnable(
        self, model_id: str
    ) -> tuple[DenoiseManifest, DenoiseBackendSpec, Path, Path]:
        """Manifest, spec, env python and model directory, or a named refusal.

        `rvc`'s order and `rvc`'s reason: what no amount of installing can fix
        first, then what an install or a fetch would fix, then the live card.
        """
        backend_kind = self._backend.kind
        manifest = _known(model_id)
        if not manifest.supports(backend_kind):
            raise ApiError(
                400,
                "backend_unsupported",
                f"denoise model {model_id!r} has no {backend_kind} block; "
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
            python = workerenv.require_env(
                self._config.home, ENV_JOB_TYPE, backend_kind
            )
        except workerenv.WorkerEnvError as exc:
            raise ApiError(
                409,
                "env_missing",
                f"cannot run {model_id!r}: {exc} (denoise shares the rvc env)",
                {
                    "model": model_id,
                    "env": str(
                        workerenv.worker_env_dir(self._config.home, ENV_JOB_TYPE)
                    ),
                },
            ) from None
        root = _require_model_files(self._config, manifest)
        return manifest, spec, python, root

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        if model is None:  # unreachable: resolve_model requires one
            raise ApiError(400, "model_required", f"{self.name} needs a model")
        _params(params)
        _manifest, spec, _python, _root = self._require_runnable(model)
        # A streaming session holds the resident engine without occupying the
        # lane, so a free lane is not a free card. `align` has asked this here
        # since it became a mutator of residency; `denoise` became the fourth on
        # 2026-09-15 and inherits the question with the kind. Admission is the
        # server's one scheduling answer (ARCHITECTURE.md section 3), so it is
        # given here rather than becoming a `failed` a minute later.
        self._residency.refuse_if_claimed(f"denoising with {model!r}")
        if self._residency.is_resident(KIND_DENOISE, model):
            # Already on the card and about to be reused — which is the normal
            # case for every block of a book after the first. Running the guard
            # would refuse the job for memory the resident separator is itself
            # holding.
            return
        accelerator.guard(
            self._config.backend_kind,
            model_id=model,
            need_bytes=spec.memory_bytes_estimate,
            owned_pids=self._owned_pids(),
            desktop_allowance_bytes=self._config.desktop_allowance_bytes,
            # A denoise load may now unload the previous resident to make room
            # for itself, exactly as an align load may — one card, one thing —
            # so what that resident holds counts as free. This USED to carry a
            # note saying the opposite ("a denoise never unloads somebody's
            # model"); that stopped being true when the separator became a
            # resident kind, and a guard that still believed it would refuse a
            # book for memory nothing is going to be using.
            reclaimable_bytes=self._residency.reclaimable_bytes(excluding=model),
        )

    # ------------------------------------------------------------------ run

    def run(self, job: Job, ctx: JobContext) -> None:
        DenoiseParams.model_validate(job.params)
        model = job.model
        if model is None:  # unreachable: resolve_model requires one
            raise JobError("model_required", f"{self.name} needs a model")

        try:
            manifest, spec, python, root = self._require_runnable(model)
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None

        source = self._input(ctx)
        output_dir = ctx.scratch / "stems"
        session = self._session(ctx, manifest, spec, root, python, model)

        request = {
            "op": "separate",
            "input": str(source),
            "output_dir": str(output_dir),
            "output_format": OUTPUT_FORMAT,
            "sample_rate": manifest.sample_rate,
        }

        def on_ready(message: dict[str, Any]) -> None:
            ctx.warming(
                f"{source.name}: {message['seconds']}s of "
                f"{message['channels']}-channel audio at {message['sample_rate']} Hz, "
                f"through {manifest.display}"
            )

        def on_progress(message: dict[str, Any]) -> None:
            ctx.progress(
                0.0,
                f"separating {source.name} through {manifest.model_filename}",
                stage=message["stage"],
            )

        try:
            outcome = session.send(
                request,
                ready_silence_timeout=READY_SILENCE_TIMEOUT_SECONDS,
                on_ready=on_ready,
                on_progress=on_progress,
                cancelled=lambda: ctx.cancelled,
            )
            # ONE result, because the unit of work is one input. The stems ride
            # inside it: a run that produced three stems has not done three
            # units of work, and results are matched to work by position.
            results = workers.require_positional_results(outcome, 1, "input")
        except workers.WorkerError as exc:
            # The session is dead or the worker broke the protocol. Take the
            # separator off the card: `Residency` must not go on advertising a
            # resident thing whose process has gone.
            self._forget(model)
            raise JobError("worker_failed", str(exc)) from None

        stems = results[0]["stems"]
        primary = self._check(manifest, outcome.ready, stems)
        # ONLY THE PRIMARY STEM IS PUBLISHED, and it is the only one anybody has
        # ever read. `denoise-bridge.ts` slices the `(dry)` stem and discards the
        # rest; publishing all of them meant the client downloaded a second
        # ~233 MB copy of each block's noise — about 10 GB across a 15-hour book
        # — to delete it. The others are still MEASURED and still reported below,
        # because "what the model produced" is a fact about the run; what changed
        # is that Crucible no longer ships bytes nothing asked for.
        ctx.artifact(primary["name"], output_dir / primary["name"])
        ctx.progress(
            1.0,
            f"{len(stems)} stem(s) from {source.name} through {manifest.display}",
            stage="separating",
        )
        ctx.done_extra(
            primary_stem=primary["name"],
            # Every stem the model wrote, named and measured. `artifacts` on the
            # done frame is what was PUBLISHED and this is what was PRODUCED;
            # they differ by design and a reader can see both.
            stems=[stem["name"] for stem in stems],
            sample_rate=primary["sample_rate"],
            frames=primary["frames"],
            separate_seconds=results[0]["separate_seconds"],
            # WHAT THE LOAD COST THIS JOB, and the number that makes the
            # residency visible: it is the load time on the block that loaded the
            # separator and `0.0` on every block after it. A pass whose blocks
            # all report a load is a pass that lost the residency, which is
            # exactly the regression this arrangement fixed and exactly the shape
            # that hid before — every job succeeding, every log clean.
            load_seconds=self._loaded_seconds.pop(job.id, 0.0),
            resident=self._residency.resident_id,
        )

    # -------------------------------------------------------------- helpers

    def _session(
        self,
        ctx: JobContext,
        manifest: DenoiseManifest,
        spec: DenoiseBackendSpec,
        model_file_dir: Path,
        python: Path,
        model: str,
    ) -> workers.WorkerSession:
        """The resident separator's worker, loading it first if it is not there.

        A load here rather than through a `load-denoiser` job, which is
        `AlignJobType._session`'s argument unchanged: `llm` and `tts` are loaded
        by an explicit job because a client chooses *when* to spend the warm-up
        and against what else is queued. A separator load is seconds and is
        always immediately followed by the block it was loaded for, so making a
        client send two jobs to denoise one block would be ceremony. Taking it
        OFF the card is still an explicit door (`unload-denoiser`), because that
        is a decision about somebody else's next job rather than about this one.

        THIS METHOD IS THE FIX. Every block of a book after the first takes the
        first branch and sends its request down a session that already has the
        checkpoint on the card.
        """
        session = self._residency.separator_session
        if session is not None and self._residency.is_resident(KIND_DENOISE, model):
            if session.alive:
                return session
            # The resident row outlived its process — the worker died between
            # blocks. Say so rather than sending a request into a closed pipe.
            ctx.warming(
                f"the resident {model} worker is gone (its log is "
                f"{session.log_path}); loading it again"
            )
            self._forget(model)

        try:
            state = accelerator.guard(
                self._config.backend_kind,
                model_id=model,
                need_bytes=spec.memory_bytes_estimate,
                owned_pids=self._owned_pids(),
                desktop_allowance_bytes=self._config.desktop_allowance_bytes,
                reclaimable_bytes=self._residency.reclaimable_bytes(excluding=model),
            )
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None
        ctx.warming(state.detail)

        began = time.perf_counter()
        try:
            self._residency.load_separator(
                manifest,
                spec,
                model_file_dir,
                python,
                WORKER_SCRIPT,
                # CUDA-only by audio-separator's own documentation, and a
                # property of the accelerator rather than of the model — so it is
                # decided here off the backend and never read from a manifest or
                # a request. It rides on the LOAD because it is what the model
                # was loaded with, and `ResidentSeparator` records it.
                use_autocast=self._backend.kind == CUDA_LINUX,
                environment=dict(ENGINE_ENVIRONMENT),
                timeout=DEFAULT_READY_TIMEOUT_SECONDS,
                on_progress=ctx.warming,
            )
        except workers.WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None
        self._loaded_seconds[ctx.job.id] = round(time.perf_counter() - began, 2)
        loaded = self._residency.separator_session
        if loaded is None:  # pragma: no cover - load_separator publishes or raises
            raise JobError(
                "worker_failed",
                f"{model} loaded but no session was published; this is a bug in "
                "crucible/residency.py",
            )
        return loaded

    def _forget(self, model: str) -> None:
        """Take a dead separator off the card without letting the tidy-up win.

        The failure being reported is the worker's, and a `stop()` that also
        fails must not replace it — the caller is about to raise the one error
        that explains what happened.
        """
        try:
            self._residency.unload(model)
        except (KeyError, workers.WorkerError):
            pass

    @staticmethod
    def _input(ctx: JobContext) -> Path:
        """The one audio file this job denoises, or a refusal.

        One, not many, and that did NOT change when the separator became
        resident — it is the half of the arrangement that was always right.
        Blocking stays in the client (PHASE4-AUDIO.md section 4.2): the app
        concatenates a session's sentences into ~22-minute blocks, sends each as
        its own job, and slices the stem back at recorded offsets. What was wrong
        was never the wire; it was that each of those ~44 jobs also paid a model
        load. `_session` is where that stopped.

        (`rvc` takes a directory for a different reason and still does: urvc's
        `convert-dir` is one process per 96 files by design, because the recycle
        needs a process to die.)
        """
        inputs = ctx.inputs()
        if not inputs:
            raise JobError(
                "invalid_inputs", "a denoise job with no audio to denoise is not a job"
            )
        if len(inputs) != 1:
            raise JobError(
                "invalid_inputs",
                f"this job carries {len(inputs)} inputs ({sorted(inputs)}); "
                "denoise takes exactly one audio file. The blocking is the "
                "client's, and a block is one file by the time it is sent",
            )
        return next(iter(inputs.values()))

    @staticmethod
    def _check(
        manifest: DenoiseManifest,
        ready: dict[str, Any],
        stems: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """The primary stem, having proved it is the only one and unchanged.

        Both rules are BookForge's, measured, and both are refusals rather than
        warnings — a stem that is subtly wrong assembles into a book that is
        subtly wrong and says nothing.
        """
        marker = f"({manifest.primary_stem})"
        hits = [stem for stem in stems if marker in stem["name"].lower()]
        if len(hits) != 1:
            raise JobError(
                "denoise_primary_stem_missing",
                f"{manifest.model_filename} produced {len(hits)} output(s) naming "
                f"{marker!r} and exactly one is the answer; it wrote "
                f"{[stem['name'] for stem in stems]}. Zero means the model "
                "produced something other than what this manifest says it "
                "produces; two means nothing here can say which one is the "
                "denoised audio",
            )
        primary = hits[0]
        if primary["sample_rate"] != manifest.sample_rate:
            raise JobError(
                "denoise_resampled",
                f"{primary['name']} came back at {primary['sample_rate']} Hz and "
                f"the input was {manifest.sample_rate} Hz — the model resampled "
                "it, which invalidates every sample offset the caller sliced by",
            )
        if primary["frames"] != ready["frames"]:
            # The invariant the client's offset slicing rests on: the roformer
            # returns exactly what it was given, sample for sample. Checked on
            # the primary stem only, because that is the one it was measured on
            # — the others' figures are reported and not enforced rather than
            # enforced on an assumption nobody has tested.
            raise JobError(
                "denoise_length_changed",
                f"{primary['name']} is {primary['frames']} frames and the input "
                f"was {ready['frames']} — the model changed the length. Slicing "
                "a stem back at the input's offsets is only safe because it does "
                "not",
            )
        return primary


# --------------------------------------------------------------- unload job


class UnloadDenoiserParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UnloadDenoiserJobType:
    """`POST /v1/jobs {"type": "unload-denoiser", "model": "<separator id>"}`.

    The other half of the residency ruling, and the reason there is no
    `load-denoiser` beside it is in `DenoiseJobType._session`. Without this door
    a separator could only be taken off the card by loading something else,
    which would make "one card, one thing" a rule you can only obey by breaking
    it. `unload-aligner`'s argument, unchanged, for the same shape of resident.
    """

    name = "unload-denoiser"

    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency

    @property
    def residency(self) -> Residency:
        return self._residency

    def describe_models(self) -> list[ModelDescriptor]:
        rows: list[ModelDescriptor] = []
        for manifest in _manifests().values():
            backend_kind = self._config.backend_kind
            supported = manifest.supports(backend_kind)
            spec = manifest.spec(backend_kind) if supported else None
            rows.append(
                ModelDescriptor(
                    id=manifest.id,
                    revision=spec.revision if spec else "",
                    source=f"{spec.hf_repo}:{spec.model_path}" if spec else "",
                    installed=(
                        model_installed(self._config.home, manifest, spec) is not None
                        if spec
                        else False
                    ),
                    resident=self._residency.is_resident(KIND_DENOISE, manifest.id),
                    vram_bytes=spec.memory_bytes_estimate if spec else 0,
                )
            )
        return rows

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        return _denoise_provenance(self._config.backend_kind, model)

    def vram_estimate(self, model: str | None) -> int:
        return 0

    def check(self, backend: Any) -> JobTypeStatus:
        separator = self._residency.resident_separator
        return JobTypeStatus(
            ready=True,
            detail=(
                f"resident: {separator.separator_id}"
                if separator
                else "no separator is resident"
            ),
        )

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        if model is None:  # unreachable: resolve_model requires one
            raise ApiError(400, "model_required", f"{self.name} needs a separator")
        try:
            UnloadDenoiserParams.model_validate(params)
        except ValidationError as exc:
            raise ApiError(
                400, "invalid_params", f"{self.name} takes no params: {exc}"
            ) from None
        if self._residency.being_cleared(model):
            # The settlement clearing this very separator is not a second
            # holder, it is this request already happening (`unload-aligner`'s
            # exception, T6 2026-09-15).
            return
        # Taking anything off the card while somebody holds it ends their
        # conversation mid-sentence, and `Residency.unload` refuses it anyway —
        # from inside the job, where it is a `failed` rather than an answer.
        self._residency.refuse_if_claimed(f"unloading {model!r}")
        if not self._residency.is_resident(KIND_DENOISE, model):
            raise ApiError(
                409,
                "separator_not_resident",
                f"{model!r} is not resident on this server; "
                + describe_resident(self._residency, KIND_DENOISE, "no separator is"),
                {"requested": model, "resident": self._residency.resident_id},
            )

    def run(self, job: Job, ctx: JobContext) -> None:
        UnloadDenoiserParams.model_validate(job.params)
        model = job.model
        if model is None:  # unreachable: resolve_model requires one
            raise JobError("model_required", f"{self.name} needs a separator")
        if self._residency.await_clearance(model):
            # The settlement got there first, which is the card this job asked
            # for. Same terminal shape as an unload this job did itself.
            ctx.progress(0.0, f"unloading {model}")
            ctx.progress(1.0, f"{model} is unloaded — the card was cleared of it")
            ctx.done_extra(resident=self._residency.resident_id)
            return
        if not self._residency.is_resident(KIND_DENOISE, model):
            # Checked before `unload()` rather than caught from it: the holder
            # unloads by id alone, and a voice sharing a separator's id would be
            # taken off the card by `unload-denoiser`.
            raise JobError(
                "separator_not_resident",
                f"{model!r} is not resident on this server; "
                + describe_resident(self._residency, KIND_DENOISE, "no separator is"),
            )
        ctx.progress(0.0, f"unloading {model}")
        try:
            self._residency.unload(model)
        except workers.WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None
        ctx.progress(1.0, f"{model} is unloaded")
        ctx.done_extra(resident=self._residency.resident_id)
