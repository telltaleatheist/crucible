"""The `rvc` job type: sentence audio in, the same sentences in another voice out.

PHASE4-AUDIO.md section 4. **Every input must produce an output** — a missing one
is a failed job, not a short answer — and that single sentence decides most of
what is below.

What is the client's and what is the server's
---------------------------------------------
The client says which model and the four numbers that change the sound:
`index_rate`, `protect_rate`, `n_semitones`, and optionally `f0_method` and
`hop_length`. Everything else is the server's: the batch size, the staging, the
environment hardening, and the fact that urvc is spawned rather than imported.

Three details that are not obvious, all measured, each enforced at one line
--------------------------------------------------------------------------
- **`protect_rate`'s scale is INVERTED.** `ultimate_rvc/rvc/infer/pipeline.py`
  gates the whole protection block on `if protect < 0.5`, so LOWER protects MORE
  and 0.5 disables protection outright — the opposite of urvc's own CLI help and
  of every RVC document on the internet. It is also a **no-op at index rate 0**,
  because the unprotected features it blends back in are cloned before feature
  retrieval and retrieval only runs above zero. The tuned deathstalker→Sigma
  recipe pairs protect 0.25 with index 0.5 for exactly that reason. The name is
  urvc's and stays; the note lives at the bound so that a reviewer does not
  "fix" it.
- **An absent `f0_method` or `hop_length` means the flag is omitted**, so urvc
  keeps its own tuned default. This is the one place in the whole server where
  "absent" is a meaningful wire value rather than a refusal, and the reason is
  that urvc's defaults are the measured ones: filling them in here would put
  Crucible's guess in a client's output with nothing to say it had happened.
- **Batching is a memory bound, not a throughput choice.** 96 files per recycled
  worker process, proven necessary on a 64 GB Mac (2026-07-17). It is engine
  knowledge: the server does it, the client never sees it, and the model reload
  it costs is the server's problem to reduce later. See `worker.py`.

The base models are the engine's, and Crucible pulls them now
-------------------------------------------------------------
urvc needs a contentvec embedder and an rmvpe/fcpe pitch predictor before it can
convert anything, and they are not per-model — they are the engine's. This file
used to say Crucible did not fetch them and could not, because the only source
written down anywhere was a 388 MB tarball on a **GitHub release** in BookForge,
which DESIGN.md section 5 refuses as a source of weights.

That was wrong about the world rather than about the rule. urvc's own first-run
downloader — the one `URVC_SKIP_INIT=1` turns off — fetches them from a
HuggingFace repo, which DESIGN.md allows, and `crucible/rvcbase.py` now pins
that repo at a revision with a digest per file. `crucible rvc pull-base` places
them under `~/.crucible/rvc-base/`, and a job without them is still refused by
name (`rvc_base_models_missing`) — but the refusal now names a command that
works. PLAN.md's owed ruling 3, discharged.

**Which files are needed is `rvcbase`'s to say, not this file's.** It used to be
a tuple here, and that tuple was already wrong: it checked for the embedder's
weights and not for the `config.json` beside them, without which transformers
will not load the directory at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ... import accelerator, rvcbase, weights, workerenv, workers
from ...config import Config
from ...errors import ApiError, JobError
from ...manifests import fingerprint
from ...rvcmodels import (
    RvcBackendSpec,
    RvcManifest,
    RvcManifestError,
    load_all_rvc_manifests,
)
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor

__all__ = ["RvcJobType", "RvcParams", "rvc_base_dir"]

JOB_TYPE = "rvc"

#: Files per recycled urvc process. BookForge's number, and a MEMORY bound: a
#: ten-minute chunk grew the converting process by about 1.5 GB and never
#: released it, so a 64 GB Mac hit swap on a book and RVC slowed about 5x
#: (2026-07-17). The env's per-file `torch.mps.empty_cache` patch helps and is
#: not sufficient; what is sufficient is the process exiting. Not a wire
#: parameter — a client that could set this could set it to a number that OOMs
#: the host.
BATCH_SIZE = 96

#: How long the server waits on a worker that has said *nothing at all*. Not a
#: run deadline: every message resets it, and urvc reports per file. 900 s covers
#: importing torch and loading the embedder on a cold disk.
READY_SILENCE_TIMEOUT_SECONDS = 900.0

#: What urvc's environment must say, and why each one earned its place
#: (`electron/rvc-bridge.ts:209`).
#:
#: - `URVC_SKIP_INIT` — skip the first-run model download and audio-separator
#:   init. Crucible pulls weights through one door and it is not an engine's
#:   start-up.
#: - `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` — a job must not reach the
#:   network. Every byte it uses was pulled at a pinned revision, and a silent
#:   fetch of "latest" would be a different conversion with nothing to say so.
#: - `KMP_DUPLICATE_LIB_OK` — the env bundles THREE OpenMP runtimes (torch,
#:   faiss-cpu and scikit-learn each ship their own libomp) and loading more than
#:   one aborts with OMP Error #15.
#: - `OMP_NUM_THREADS=1` — with the duplicates co-loaded, the cross-runtime
#:   barrier SIGSEGVs in `__kmp_suspend_initialize_thread` the moment conversion
#:   spins up its thread pool: exit 139 on the very first sentence, reproduced on
#:   BOTH mps and cpu, so the device was never the cause. One OpenMP thread
#:   removes the barrier. The heavy work is on the torch device, so serial OpenMP
#:   costs little.
#: - `PYTHONUNBUFFERED` — the worker parses urvc's progress line by line, and a
#:   block-buffered child would report a whole batch at once.
ENGINE_ENVIRONMENT: dict[str, str] = {
    "URVC_SKIP_INIT": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}

WORKER_SCRIPT = Path(__file__).resolve().parent / "worker.py"


def rvc_base_dir(config: Config) -> Path:
    """Where urvc's shared base assets live. See the module docstring."""
    return rvcbase.base_root(config)


def _base_assets() -> "rvcbase.RvcBaseAssets":
    """The declared set, or a 500 naming why this build cannot read its own.

    Read per call rather than at import: `crucible doctor` and a job both ask,
    the file is 2 KB, and a module-level load would make an unreadable
    declaration an ImportError at server start instead of a refusal with a
    reason in it.
    """
    try:
        return rvcbase.load_rvc_base()
    except rvcbase.RvcBaseError as exc:
        raise ApiError(
            500,
            "rvc_base_declaration_unreadable",
            f"this server cannot read its base-asset declaration: {exc}",
        ) from None


class RvcParams(BaseModel):
    """`params` for an rvc job. Unknown keys are refused, not ignored."""

    model_config = ConfigDict(extra="forbid")

    index_rate: float = Field(ge=0.0, le=1.0)

    #: THE SCALE IS INVERTED, AND THIS IS THE LINE A REVIEWER WILL TRY TO "FIX".
    #: `ultimate_rvc/rvc/infer/pipeline.py` gates the entire protection block on
    #: `if protect < 0.5`, so LOWER protects MORE and 0.5 turns protection OFF —
    #: backwards against urvc's own CLI help and against every RVC document
    #: online. The bound is therefore [0, 0.5] with 0.5 meaning "no protection",
    #: not "maximum protection", and a value above 0.5 is refused because it can
    #: only mean the caller believed the documented scale.
    #:
    #: It is also a NO-OP AT INDEX RATE 0: the unprotected features it blends
    #: back in are cloned before feature retrieval, and retrieval only runs above
    #: zero. The tuned deathstalker→Sigma recipe pairs protect 0.1 with index 0.3
    #: for that reason.
    protect_rate: float = Field(ge=0.0, le=0.5)

    #: Pitch shift. The app's recipes use -2; the range is urvc's.
    n_semitones: int = Field(ge=-24, le=24)

    #: **Absent means the flag is omitted**, so urvc keeps its own default. The
    #: one meaningful absence on the whole wire; see the module docstring.
    f0_method: str | None = None

    #: Crepe-family only, and absent for the same reason and with the same
    #: meaning as `f0_method`. Range is urvc's own 1-512.
    hop_length: int | None = Field(default=None, ge=1, le=512)


# ------------------------------------------------------------------ helpers


def _manifests() -> dict[str, RvcManifest]:
    try:
        return load_all_rvc_manifests()
    except RvcManifestError as exc:
        raise ApiError(
            500,
            "rvc_manifests_unreadable",
            f"this server cannot read its RVC manifests: {exc}",
        ) from None


def _known(model_id: str) -> RvcManifest:
    manifests = _manifests()
    manifest = manifests.get(model_id)
    if manifest is None:
        raise ApiError(
            400,
            "unknown_model",
            f"no RVC manifest for {model_id!r}; this build ships {sorted(manifests)}",
        )
    return manifest


def _params(params: dict[str, Any]) -> RvcParams:
    try:
        return RvcParams.model_validate(params)
    except ValidationError as exc:
        raise ApiError(
            400,
            "invalid_params",
            "rvc params are not valid: "
            + "; ".join(
                f"{'.'.join(str(p) for p in problem['loc']) or '<root>'}: "
                f"{problem['msg']}"
                for problem in exc.errors()
            ),
        ) from None


def _require_base_assets(config: Config) -> Path:
    """urvc's shared base assets, or `rvc_base_models_missing` by name."""
    assets = _base_assets()
    root = rvc_base_dir(config)
    absent = rvcbase.missing(config, assets)
    if absent:
        raise ApiError(
            409,
            "rvc_base_models_missing",
            f"ultimate-rvc needs its shared base assets and this host is missing "
            f"{[str(root / name) for name in absent]}. They are the engine's, not "
            f"any model's — the same files urvc's own first-run downloader would "
            f"have fetched, which URVC_SKIP_INIT turns off. Run "
            f"`{rvcbase.PULL_COMMAND}` to place them "
            f"({assets.total_bytes / 1e9:.2f} GB from "
            f"{assets.hf_repo}@{assets.revision[:12]}, verified file by file)",
            {
                "root": str(root),
                "missing": sorted(absent),
                "hf_repo": assets.hf_repo,
                "revision": assets.revision,
                "command": rvcbase.PULL_COMMAND,
            },
        )
    return root


def _rvc_provenance(backend_kind: str, model: str | None) -> dict[str, Any] | None:
    """The `model` block of a converted sentence's provenance sidecar.

    A converted audiobook that does not name the RVC model AND its revision is a
    book nobody can reproduce, and two trainings of one voice under one name are
    two voices — which is the whole reason `rvc/<id>.toml` exists instead of a
    folder name.
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


# ------------------------------------------------------------------ job type


class RvcJobType:
    """`POST /v1/jobs {"type": "rvc", "model": "<id>", "inputs": {...}}`."""

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
        # somebody else's process holding the card. A callable and not a set,
        # because the answer changes every time something is loaded or unloaded.
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
                    f"{spec.hf_repo}:{spec.archive}",
                    spec.memory_bytes_estimate,
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
                    # The repo AND the file. Seven of these share one repo, so a
                    # `source` of just the repo id would say the same thing for
                    # every row and identify none of them.
                    source=source,
                    installed=installed,
                    # Nothing is ever resident for `rvc`: the whole design is a
                    # process that exits every 96 files so the OS reclaims what
                    # it leaked.
                    resident=False,
                    vram_bytes=estimate,
                )
            )
        return rows

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        if model is None:
            raise JobError("model_required", f"{self.name} needs a model")
        return _rvc_provenance(self._config.backend_kind, model)

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
        root = rvc_base_dir(self._config)
        try:
            absent = rvcbase.missing(self._config, _base_assets())
        except ApiError as exc:
            return JobTypeStatus(ready=False, detail=exc.message)
        if absent:
            return JobTypeStatus(
                ready=False,
                detail=(
                    f"{env.detail}; but urvc's base assets are not at {root}: "
                    f"{sorted(absent)} missing — `{rvcbase.PULL_COMMAND}`"
                ),
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
                    f"{env.detail}; no RVC model is installed — "
                    "`crucible rvc pull <id>`"
                ),
            )
        return JobTypeStatus(ready=True, detail=f"{env.detail}; installed: {installed}")

    # ------------------------------------------------------------ preflight

    def _require_runnable(
        self, model_id: str
    ) -> tuple[RvcManifest, RvcBackendSpec, Path, Path]:
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
                f"RVC model {model_id!r} has no {backend_kind} block; "
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
        validated = _params(params)
        manifest, spec, _, _ = self._require_runnable(model)
        self._require_index(manifest, validated)
        _require_base_assets(self._config)
        accelerator.guard(
            self._config.backend_kind,
            model_id=model,
            need_bytes=spec.memory_bytes_estimate,
            owned_pids=self._owned_pids(),
            desktop_allowance_bytes=self._config.desktop_allowance_bytes,
            # Deliberately no `reclaimable_bytes`. An `llm` load may unload the
            # previous resident to make room for itself; an rvc job never unloads
            # somebody's model to convert a sentence, so the memory a resident
            # engine holds is not memory this job can have.
        )

    @staticmethod
    def _require_index(manifest: RvcManifest, params: RvcParams) -> None:
        """`forceIndexRate0`, said out loud.

        BookForge derives this from the ABSENCE of a `.index` file and silently
        clamps the rate to 0 (`rvc-models.ts:261`). Here the manifest states it
        and a mismatch is refused, because a clamp is a request the server
        answered with a different one: feature retrieval against an index that
        does not exist is not a preference urvc quietly ignores, and a caller who
        asked for 0.5 and got 0 has an output that sounds wrong for a reason
        nothing in it explains.
        """
        if manifest.has_index or params.index_rate == 0:
            return
        raise ApiError(
            400,
            "model_has_no_index",
            f"RVC model {manifest.id!r} ships no .index, so an index_rate of "
            f"{params.index_rate} asks for feature retrieval that cannot happen. "
            "Send index_rate 0. Note that protect_rate is then a no-op too — it "
            "blends back features that are only cloned when retrieval runs",
            {"model": manifest.id, "index_rate": params.index_rate},
        )

    # ------------------------------------------------------------------ run

    def run(self, job: Job, ctx: JobContext) -> None:
        params = RvcParams.model_validate(job.params)
        model = job.model
        if model is None:  # unreachable: resolve_model requires one
            raise JobError("model_required", f"{self.name} needs a model")

        try:
            manifest, spec, python, weights_dir = self._require_runnable(model)
            self._require_index(manifest, params)
            base = _require_base_assets(self._config)
            # The card can change between the queue and the lane, so the guard
            # runs again here against the same rules.
            state = accelerator.guard(
                self._config.backend_kind,
                model_id=model,
                need_bytes=spec.memory_bytes_estimate,
                owned_pids=self._owned_pids(),
                desktop_allowance_bytes=self._config.desktop_allowance_bytes,
            )
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None
        ctx.warming(state.detail)

        names, extension = self._inputs(ctx)
        models_dir = self._stage_models(ctx, base, weights_dir, manifest)
        output_dir = ctx.scratch / "converted"

        request = {
            "models_dir": str(models_dir),
            "model_name": manifest.model_name,
            "input_dir": str(ctx.job.inputs_dir),
            "output_dir": str(output_dir),
            "inputs": names,
            "extension": extension,
            "index_rate": params.index_rate,
            "protect_rate": params.protect_rate,
            "n_semitones": params.n_semitones,
            "batch_size": BATCH_SIZE,
        }
        # Present only when the client sent them. Their ABSENCE is what tells the
        # worker to omit the flag and leave urvc on its own tuned default, so a
        # `None` here would be a value where there must be a gap.
        if params.f0_method is not None:
            request["f0_method"] = params.f0_method
        if params.hop_length is not None:
            request["hop_length"] = params.hop_length

        total = len(names)

        def on_ready(message: dict[str, Any]) -> None:
            ctx.warming(
                f"{message['files']} file(s) through {manifest.model_name} in "
                f"{message['batches']} batch(es) of {message['batch_size']} — the "
                "batch is a memory bound, not a throughput choice"
            )

        def on_progress(message: dict[str, Any]) -> None:
            processed = int(message["processed"])
            ctx.progress(
                min(1.0, processed / total),
                f"converted {processed} of {total} file(s) "
                f"(batch {message['batch']} of {message['batches']})",
                stage=message["stage"],
                processed=processed,
                total=total,
            )

        try:
            outcome = workers.run_worker(
                python=python,
                script=WORKER_SCRIPT,
                request=request,
                log_path=self._config.logs_dir / f"rvc-{job.id}.log",
                ready_silence_timeout=READY_SILENCE_TIMEOUT_SECONDS,
                on_ready=on_ready,
                on_progress=on_progress,
                cancelled=lambda: ctx.cancelled,
                environment=dict(ENGINE_ENVIRONMENT),
            )
            results = workers.require_positional_results(outcome, total, "file")
        except workers.WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None

        # EVERY INPUT MUST PRODUCE AN OUTPUT. A run that converted 1,399 of 1,400
        # sentences and published them looks exactly like one that converted all
        # 1,400, and the book it assembles has one sentence in the wrong voice.
        missing = [
            f"{name}: {result['error']}"
            for name, result in zip(names, results)
            if "error" in result
        ]
        if missing:
            raise JobError(
                "rvc_output_missing",
                f"{len(missing)} of {total} input(s) produced no output, so nothing "
                "is published: a partial conversion assembled into a book is a book "
                "with sentences in two voices and nothing to say which. "
                + "; ".join(missing[:20])
                + (f" (and {len(missing) - 20} more)" if len(missing) > 20 else ""),
            )

        for name in names:
            ctx.artifact(name, output_dir / name)
        ctx.progress(
            1.0,
            f"{total} file(s) converted through {manifest.model_name}",
            stage="converting",
            processed=total,
            total=total,
        )
        ctx.done_extra(files=total, model_name=manifest.model_name)

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _inputs(ctx: JobContext) -> tuple[list[str], str]:
        """The input names in a stable order, and the one extension they share.

        **One extension for the whole job**, because urvc's `convert-dir` takes a
        single `--input-glob` and a single `--output-ext`, and because "one
        artifact per input, same name" is only true when the output keeps the
        input's extension. A mixed-format job is refused by name rather than half
        converted — a session's sentences are one format, and a job that has two
        is a client that sent the wrong directory.
        """
        inputs = ctx.inputs()
        if not inputs:
            raise JobError(
                "invalid_inputs", "an rvc job with no audio to convert is not a job"
            )
        names = sorted(inputs)
        extensions = {Path(name).suffix.lstrip(".").lower() for name in names}
        if "" in extensions:
            raise JobError(
                "invalid_inputs",
                "every rvc input needs a file extension; urvc selects them with a "
                "glob and writes the output under the same name",
            )
        if len(extensions) != 1:
            raise JobError(
                "invalid_inputs",
                f"this job's inputs are {sorted(extensions)}; an rvc job converts "
                "one format at a time, because urvc takes one input glob and one "
                "output extension and the artifacts keep the inputs' names",
            )
        return names, next(iter(extensions))

    @staticmethod
    def _stage_models(
        ctx: JobContext, base: Path, weights_dir: Path, manifest: RvcManifest
    ) -> Path:
        """Build this job's `URVC_MODELS_DIR`: the base assets plus ONE model.

        urvc resolves a model by name under `<URVC_MODELS_DIR>/rvc/voice_models/`,
        and the two halves of that tree live in different places here — the base
        assets are the engine's and shared, the voice model is pulled per id. So
        the job composes a root out of symlinks rather than copying either.

        One model, not all of them, and that is the point: urvc is given a NAME,
        and a root holding seven voices is a root where a name could resolve to
        the wrong one. This job can only see the model it was asked for.
        """
        root = ctx.scratch / "urvc-models" / "rvc"
        root.mkdir(parents=True, exist_ok=True)
        for name in ("embedders", "predictors", "pretraineds"):
            source = base / "rvc" / name
            if source.is_dir():
                (root / name).symlink_to(source, target_is_directory=True)
        voices = root / "voice_models"
        voices.mkdir()
        # The archive unpacks to `rvc/voice_models/<model_name>/`, so the pulled
        # weights directory is itself a models root; the model folder is one
        # level down. Verified against the published tarballs, 2026-09-13.
        model_dir = weights_dir / "rvc" / "voice_models" / manifest.model_name
        if not model_dir.is_dir():
            raise JobError(
                "model_not_installed",
                f"{manifest.id!r} is stamped as pulled but {model_dir} is not there; "
                f"{manifest.path.name} says the archive holds "
                f"rvc/voice_models/{manifest.model_name}. Re-pull it with "
                f"`crucible rvc pull {manifest.id} --force`",
            )
        (voices / manifest.model_name).symlink_to(model_dir, target_is_directory=True)
        return root.parent
