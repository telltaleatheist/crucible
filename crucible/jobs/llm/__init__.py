"""The `llm` job type: `load-model` and `unload-model`.

PHASE2-LLM.md section 5. Chat does not come through here — it is proxied straight
to the resident engine at `/v1/openai/chat/completions`, so vLLM's and mlx-lm's
own continuous batching does the work. What runs on the exclusive lane is the
*lifecycle*: putting a model on the accelerator and taking it off. That is why a
chat request can never race a load.

Every refusal in section 5 happens **before the job is queued**, in `preflight()`,
so the client gets an HTTP error naming the thing rather than a job that fails a
minute later:

    unknown_model        no manifest with that id (raised by jobs.resolve_model)
    backend_unsupported  the manifest has no block for this host's backend
    env_missing          ~/.crucible/envs/llm is not installed
    model_not_installed  no weights at the manifest's pinned revision
    accelerator_busy     somebody else's process is on the card
    insufficient_memory  free memory is below the manifest's estimate
    model_not_resident   (unload) that model is not the one that is loaded
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ... import accelerator, llmenv, weights
from ...config import Config
from ...engines import EngineError
from ...errors import ApiError, JobError
from ...manifests import ManifestError, ModelManifest, load_all_manifests
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor
from .residency import DEFAULT_READY_TIMEOUT_SECONDS, Residency, ResidentModel

__all__ = [
    "LoadModelJobType",
    "LoadParams",
    "Residency",
    "ResidentModel",
    "UnloadModelJobType",
    "model_rows",
]


class LoadParams(BaseModel):
    """`params` for a load-model job. Unknown keys are refused, not ignored."""

    model_config = ConfigDict(extra="forbid")

    timeout_s: float = Field(default=DEFAULT_READY_TIMEOUT_SECONDS, ge=30, le=7200)


class UnloadParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ helpers


def _manifests() -> dict[str, ModelManifest]:
    try:
        return load_all_manifests()
    except ManifestError as exc:
        raise ApiError(
            500,
            "manifests_unreadable",
            f"this server cannot read its model manifests: {exc}",
        ) from None


def _params(model: type[BaseModel], params: dict[str, Any], job_type: str) -> Any:
    """Validate `params` up front, as a named 400 rather than a 500."""
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


def _known(model_id: str) -> ModelManifest:
    """The manifest for this id, or `unknown_model` by name."""
    manifests = _manifests()
    manifest = manifests.get(model_id)
    if manifest is None:
        raise ApiError(
            400,
            "unknown_model",
            f"no manifest for model {model_id!r}; this build ships "
            f"{sorted(manifests)}",
        )
    return manifest


def _descriptors(
    config: Config, backend_kind: str, residency: Residency
) -> list[ModelDescriptor]:
    """`/v1/info` capabilities rows — DESIGN.md section 4's shape."""
    rows: list[ModelDescriptor] = []
    for manifest in _manifests().values():
        if manifest.supports(backend_kind):
            spec = manifest.spec(backend_kind)
            revision, source, estimate = spec.revision, spec.hf_repo, spec.memory_bytes_estimate
        else:
            revision, source, estimate = "", "", 0
        rows.append(
            ModelDescriptor(
                id=manifest.id,
                revision=revision,
                source=source,
                resident=residency.resident_id == manifest.id,
                vram_bytes=estimate,
            )
        )
    return rows


def model_rows(
    config: Config, backend: Any, residency: Residency
) -> list[dict[str, Any]]:
    """`GET /v1/models` — PHASE2-LLM.md section 5.

    `loadable` answers "is everything this host needs in place", which is a fact
    about the disk. It deliberately does **not** run nvidia-smi: the accelerator's
    state changes between a listing and a request, so the guard runs at load time
    and refuses there. A row that says `loadable: true` can still be refused with
    `accelerator_busy`.
    """
    backend_kind = backend.kind
    env = llmenv.env_status(config.home, backend_kind)
    rows: list[dict[str, Any]] = []
    for manifest in _manifests().values():
        supported = manifest.supports(backend_kind)
        estimate: int | None = None
        is_installed = False
        reason: str | None = None
        if not supported:
            reason = (
                f"{manifest.path.name} has no {backend_kind} block; it declares "
                f"{sorted(manifest.backends)}"
            )
        else:
            spec = manifest.spec(backend_kind)
            estimate = spec.memory_bytes_estimate
            is_installed = weights.installed(config, manifest, spec) is not None
            if estimate > backend.gpu.vram_bytes:
                # Not loadable here at all, so say so instead of asking for a
                # 55 GB download first.
                reason = (
                    f"needs {estimate / 1024 ** 3:.1f} GiB and "
                    f"{backend.gpu.name} has {backend.gpu.vram_bytes / 1024 ** 3:.1f}"
                    " GiB in total"
                )
            elif not env.installed:
                reason = f"the llm env is not ready: {env.detail}"
            elif not is_installed:
                reason = (
                    f"no weights at {weights.model_dir(config, manifest.id, backend_kind)}"
                    f" — run `crucible models pull {manifest.id}`"
                )
        row: dict[str, Any] = {
            "id": manifest.id,
            "family": manifest.family,
            "params_b": manifest.params_b,
            "backend_supported": supported,
            "installed": is_installed,
            "resident": residency.resident_id == manifest.id,
            "loadable": reason is None,
            "memory_bytes_estimate": estimate,
            "context_default": manifest.context_default,
        }
        if reason is not None:
            row["reason"] = reason
        rows.append(row)
    return rows


def _require_loadable(
    config: Config, backend: Any, model_id: str
) -> tuple[ModelManifest, Any, Any]:
    """Manifest, backend spec and installed weights, or the named refusal.

    The order is deliberate: what can never be fixed, then what an install or a
    pull would fix, then what the live accelerator says. So a 27B on a 24 GB card
    is refused for being a 27B on a 24 GB card, not for needing a 55 GB download
    first.
    """
    backend_kind = backend.kind
    manifest = _known(model_id)
    if not manifest.supports(backend_kind):
        raise ApiError(
            400,
            "backend_unsupported",
            f"model {model_id!r} has no {backend_kind} block; {manifest.path.name} "
            f"declares {sorted(manifest.backends)}",
            {"model": model_id, "backend": backend_kind,
             "declared": sorted(manifest.backends)},
        )
    spec = manifest.spec(backend_kind)
    accelerator.refuse_if_larger_than_host(
        model_id=model_id,
        need_bytes=spec.memory_bytes_estimate,
        host_total_bytes=backend.gpu.vram_bytes,
        host_name=backend.gpu.name,
    )
    try:
        python = llmenv.require_env(config.home, backend_kind)
    except llmenv.EnvError as exc:
        raise ApiError(
            409,
            "env_missing",
            f"cannot load {model_id!r}: {exc}",
            {"model": model_id, "env": str(llmenv.llm_env_dir(config.home))},
        ) from None
    try:
        installed = weights.require_installed(config, manifest, spec)
    except weights.WeightsError as exc:
        raise ApiError(
            409,
            "model_not_installed",
            str(exc),
            {"model": model_id, "hf_repo": spec.hf_repo, "revision": spec.revision},
        ) from None
    return manifest, spec, (python, installed)


# ----------------------------------------------------------------- load job


class LoadModelJobType:
    """`POST /v1/jobs {"type": "load-model", "model": "<id>"}`."""

    name = "load-model"

    def __init__(
        self, config: Config, backend: Any, residency: Residency
    ) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency

    @property
    def residency(self) -> Residency:
        return self._residency

    def describe_models(self) -> list[ModelDescriptor]:
        return _descriptors(self._config, self._config.backend_kind, self._residency)

    def vram_estimate(self, model: str | None) -> int:
        if model is None:
            raise JobError("model_required", f"{self.name} needs a model")
        manifest = _known(model)
        if not manifest.supports(self._config.backend_kind):
            return 0
        return manifest.spec(self._config.backend_kind).memory_bytes_estimate

    def check(self, backend: Any) -> JobTypeStatus:
        env = llmenv.env_status(self._config.home, backend.kind)
        if not env.installed:
            return JobTypeStatus(ready=False, detail=env.detail)
        rows = model_rows(self._config, backend, self._residency)
        ready = [row["id"] for row in rows if row["loadable"]]
        if not ready:
            return JobTypeStatus(
                ready=False,
                detail=(
                    f"{env.detail}; no model is installed — "
                    "`crucible models pull <id>`"
                ),
            )
        return JobTypeStatus(ready=True, detail=f"{env.detail}; loadable: {ready}")

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        if model is None:  # unreachable: resolve_model requires one
            raise ApiError(400, "model_required", f"{self.name} needs a model")
        _params(LoadParams, params, self.name)
        manifest, spec, _ = _require_loadable(self._config, self._backend, model)
        accelerator.guard(
            self._config.backend_kind,
            model_id=model,
            need_bytes=spec.memory_bytes_estimate,
            owned_pids=self._residency.owned_pids(),
            desktop_allowance_bytes=self._config.desktop_allowance_bytes,
            reclaimable_bytes=self._residency.reclaimable_bytes(excluding=model),
        )

    def run(self, job: Job, ctx: JobContext) -> None:
        params = LoadParams.model_validate(job.params)
        model = job.model
        if model is None:  # unreachable: resolve_model requires one
            raise JobError("model_required", f"{self.name} needs a model")
        try:
            manifest, spec, (python, installed) = _require_loadable(
                self._config, self._backend, model
            )
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None

        ctx.warming(f"checking the accelerator for {model}")
        try:
            # The card can change between the queue and the lane, so the guard
            # runs again here, against the same rules.
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

        ctx.progress(0.0, f"loading {model}")
        try:
            resident = self._residency.load(
                manifest,
                spec,
                installed.path,
                python,
                timeout=params.timeout_s,
                on_progress=ctx.warming,
            )
        except EngineError as exc:
            raise JobError("engine_failed", str(exc)) from None
        ctx.progress(1.0, f"{model} is resident")
        ctx.done_extra(resident=resident.model_id)


# --------------------------------------------------------------- unload job


class UnloadModelJobType:
    """`POST /v1/jobs {"type": "unload-model", "model": "<id>"}`."""

    name = "unload-model"

    def __init__(
        self, config: Config, backend: Any, residency: Residency
    ) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency

    @property
    def residency(self) -> Residency:
        return self._residency

    def describe_models(self) -> list[ModelDescriptor]:
        return _descriptors(self._config, self._config.backend_kind, self._residency)

    def vram_estimate(self, model: str | None) -> int:
        return 0

    def check(self, backend: Any) -> JobTypeStatus:
        resident = self._residency.resident_id
        return JobTypeStatus(
            ready=True,
            detail=(
                f"resident: {resident}" if resident else "nothing is resident"
            ),
        )

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        if model is None:  # unreachable: resolve_model requires one
            raise ApiError(400, "model_required", f"{self.name} needs a model")
        _params(UnloadParams, params, self.name)
        resident = self._residency.resident_id
        if resident != model:
            raise ApiError(
                409,
                "model_not_resident",
                f"{model!r} is not resident on this server; "
                + (f"{resident!r} is" if resident else "no model is"),
                {"requested": model, "resident": resident},
            )

    def run(self, job: Job, ctx: JobContext) -> None:
        UnloadParams.model_validate(job.params)
        model = job.model
        if model is None:  # unreachable: resolve_model requires one
            raise JobError("model_required", f"{self.name} needs a model")
        ctx.progress(0.0, f"unloading {model}")
        try:
            self._residency.unload(model)
        except KeyError:
            raise JobError(
                "model_not_resident",
                f"{model!r} is not resident on this server; "
                + (
                    f"{self._residency.resident_id!r} is"
                    if self._residency.resident_id
                    else "no model is"
                ),
            ) from None
        except EngineError as exc:
            raise JobError("engine_failed", str(exc)) from None
        ctx.progress(1.0, f"{model} is unloaded")
        ctx.done_extra(resident=self._residency.resident_id)
