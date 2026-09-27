from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ... import accelerator, jobenv, llamacpp, vram, weights
from ...backend import LLAMA_WINDOWS
from ...capability import (
    MIN_LOAD_CONTEXT,
    Candidate,
    available_bytes,
    check_load_context,
)
from ... import ollamastore
from ...cardfacts import card_for
from ...config import Config
from ...engines import EngineError
from ...engines import vllm as vllm_engine
from ...errors import ApiError, JobError
from ...inflight import require_act_name
from ...leases import require_ttl
from ...manifests import (
    ManifestError,
    ModelManifest,
    fingerprint,
    load_all_manifests,
)
from ...residency import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    KIND_LLM,
    Residency,
    describe_resident,
)
from .. import worker_type
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor

__all__ = [
    "LlmEngineStatus",
    "LoadModelJobType",
    "LoadParams",
    "Residency",
    "UnloadModelJobType",
    "llm_engine_status",
    "model_rows",
]


class LlmEngineStatus:
    def __init__(self, installed: bool, detail: str, executable: "Any | None") -> None:
        self.installed = installed
        self.detail = detail
        self.executable = executable


def llm_engine_status(config: Config, backend: Any) -> LlmEngineStatus:
    if backend.kind == LLAMA_WINDOWS:
        build = llamacpp.build_for(backend.gpu.vendor)
        found = llamacpp.installed(config, build)
        if found is None:
            return LlmEngineStatus(
                installed=False,
                detail=(
                    f"llama.cpp {llamacpp.LLAMA_CPP_RELEASE} ({build}) is not "
                    f"installed at {llamacpp.engine_dir(config)} — run "
                    "`crucible install llm`"
                ),
                executable=None,
            )
        return LlmEngineStatus(
            installed=True,
            detail=(
                f"llama.cpp {llamacpp.LLAMA_CPP_RELEASE} ({build}) at "
                f"{found.path} ({found.bytes / 1e6:.0f} MB)"
            ),
            executable=llamacpp.server_path(config),
        )
    env = jobenv.env_status(config.home, jobenv.llm_env(backend.kind), backend.kind)
    return LlmEngineStatus(
        installed=env.installed,
        detail=env.detail,
        executable=None if not env.installed else jobenv.env_python(config.home, jobenv.llm_env(backend.kind)),
    )


class LeaseOnLoad(BaseModel):
    model_config = ConfigDict(extra="forbid")

    act: str
    ttl_seconds: int


class LoadParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_s: float = Field(default=DEFAULT_READY_TIMEOUT_SECONDS, ge=30, le=7200)
    lease: LeaseOnLoad | None = None
    context: int | None = Field(default=None, ge=MIN_LOAD_CONTEXT, strict=True)


class UnloadParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


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
    rows: list[ModelDescriptor] = []
    for manifest in _manifests().values():
        if manifest.supports(backend_kind):
            spec = manifest.spec(backend_kind)
            revision = spec.revision
            source = spec.hf_repo
            estimate = spec.memory_bytes_estimate
            installed = weights.installed(config, manifest, spec) is not None
        else:
            revision, source, estimate, installed = "", "", 0, False
        rows.append(
            ModelDescriptor(
                id=manifest.id,
                revision=revision,
                source=source,
                installed=installed,
                resident=residency.is_resident(KIND_LLM, manifest.id),
                vram_bytes=estimate,
            )
        )
    return rows


def _in_the_ollama_store(manifest: Any, backend_kind: str) -> dict[str, Any] | None:
    if backend_kind != LLAMA_WINDOWS:
        return None
    local = manifest.local
    if local is None or getattr(local, "tag", None) is None:
        return None
    root = ollamastore.store_root()
    try:
        found = ollamastore.resident(
            root, local.tag, wants_projector="image" in manifest.serves(backend_kind)
        )
    except ollamastore.OllamaStoreError:
        return None
    return {
        "tag": found.tag,
        "bytes": found.bytes,
        "provenance": found.provenance,
        "path": str(found.model.path),
        "same_file_as_the_pin": False,
    }


def model_rows(
    config: Config, backend: Any, residency: Residency
) -> list[dict[str, Any]]:
    backend_kind = backend.kind
    env = llm_engine_status(config, backend)
    resident = residency.resident_model
    rows: list[dict[str, Any]] = []
    for manifest in _manifests().values():
        supported = manifest.supports(backend_kind)
        estimate: int | None = None
        revision: str | None = None
        max_model_len: int | None = None
        terms: Any = None
        ceiling: dict[str, Any] | None = None
        already_here: dict[str, Any] | None = None
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
            revision = spec.revision
            terms = spec.memory
            if terms is not None:
                ceiling_here = Candidate.of(manifest, backend_kind).context_ceiling(
                    available_bytes(
                        backend.gpu.vram_bytes, config.desktop_allowance_bytes
                    ),
                    1,
                )
                if ceiling_here is None:
                    raise ValueError(f"{manifest.id} has no context ceiling")
                ceiling = {
                    "tokens": ceiling_here.tokens,
                    "card_affords": ceiling_here.memory_context,
                    "max_context": ceiling_here.served_context,
                    "weights_allow": manifest.trained_context,
                    "limited_by": (
                        "card" if ceiling_here.bound_by == "memory" else "max_context"
                    ),
                    "concurrency": 1,
                    "basis": terms.basis,
                }
            max_model_len = (
                resident.max_model_len
                if resident is not None and resident.model_id == manifest.id
                else manifest.context_for(backend_kind)
            )
            is_installed = weights.installed(config, manifest, spec) is not None
            already_here = _in_the_ollama_store(manifest, backend_kind)
            if estimate > backend.gpu.vram_bytes:
                reason = (
                    f"needs {estimate / 1024 ** 3:.1f} GiB and "
                    f"{backend.gpu.name} has {backend.gpu.vram_bytes / 1024 ** 3:.1f}"
                    " GiB in total"
                )
            elif not env.installed:
                reason = f"the llm env is not ready: {env.detail}"
            elif not is_installed:
                if manifest.weights_of is not None:
                    try:
                        weights.require_installed(config, manifest, spec)
                    except weights.WeightsError as exc:
                        reason = str(exc)
                else:
                    directory = weights.subject_dir(config, manifest, backend_kind)
                    reason = (
                        f"no weights at {directory}"
                        f" — run `crucible models pull {manifest.id}`"
                    )
        row: dict[str, Any] = {
            "id": manifest.id,
            "family": manifest.family,
            "params_b": manifest.params_b,
            "revision": revision,
            "fingerprint": (
                None if revision is None else fingerprint(manifest.id, revision)
            ),
            "modalities": list(manifest.modalities),
            "serves": (
                list(manifest.serves(backend_kind)) if supported else None
            ),
            "backend_supported": supported,
            "installed": is_installed,
            "weights_of": manifest.weights_of,
            "ollama_copy": already_here,
            "resident": residency.is_resident(KIND_LLM, manifest.id),
            "loadable": reason is None,
            "memory_bytes_estimate": estimate,
            "context_default": manifest.context_for(backend_kind),
            "max_model_len": max_model_len,
            "trained_context": manifest.trained_context,
            "max_context": ceiling,
            "memory_terms": None if terms is None else terms.to_dict(),
            "defaults": (
                resident.defaults.to_dict()
                if resident is not None and resident.model_id == manifest.id
                else manifest.defaults.to_dict()
            ),
        }
        if reason is not None:
            row["reason"] = reason
        rows.append(row)
    return rows


def _model_provenance(backend_kind: str, model: str | None) -> dict[str, Any] | None:
    if model is None:
        return None
    manifest = _known(model)
    spec = manifest.backends.get(backend_kind)
    if spec is None:
        return {"id": model, "revision": None, "fingerprint": None}
    return {
        "id": model,
        "revision": spec.revision,
        "fingerprint": fingerprint(model, spec.revision),
    }


def _require_loadable(
    config: Config, backend: Any, model_id: str
) -> tuple[ModelManifest, Any, Any]:
    backend_kind = backend.kind
    manifest = _known(model_id)
    spec = worker_type.require_block(manifest, model_id, backend_kind, "model")
    worker_type.refuse_if_larger_than_host(backend, model_id, spec.memory_bytes_estimate)
    accelerator.refuse_if_card_lacks(
        model_id=model_id, spec=spec, card=card_for(config.home, backend.gpu)
    )
    if backend_kind == LLAMA_WINDOWS:
        engine = llm_engine_status(config, backend)
        if not engine.installed:
            raise ApiError(
                409,
                "env_missing",
                f"cannot load {model_id!r}: {engine.detail}",
                {
                    "model": model_id,
                    "env": str(llamacpp.engine_dir(config)),
                },
            )
        python = engine.executable
    else:
        try:
            env_spec = jobenv.llm_env(backend_kind)
            python = jobenv.require_env(config.home, env_spec, backend_kind)
        except jobenv.EnvError as exc:
            raise ApiError(
                409,
                "env_missing",
                f"cannot load {model_id!r}: {exc}",
                {
                    "model": model_id,
                    "env": str(
                        jobenv.env_dir(config.home, jobenv.llm_env(backend_kind))
                    ),
                },
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


def _load_context(
    config: Config, backend: Any, manifest: ModelManifest, params: "LoadParams"
) -> int:
    if params.context is None:
        return manifest.context_for(backend.kind)
    check_load_context(
        manifest,
        backend.kind,
        available_bytes=available_bytes(
            backend.gpu.vram_bytes, config.desktop_allowance_bytes
        ),
        context=params.context,
    )
    return params.context


class LoadModelJobType:
    name = "load-model"

    def __init__(
        self,
        config: Config,
        backend: Any,
        residency: Residency,
        leases: Any | None = None,
    ) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency
        self._leases = leases

    @property
    def residency(self) -> Residency:
        return self._residency

    def describe_models(self) -> list[ModelDescriptor]:
        return _descriptors(self._config, self._config.backend_kind, self._residency)

    def _reclaimable(self) -> int:
        return self._residency.reclaimable_bytes()

    def vram_estimate(self, model: str | None) -> int:
        if model is None:
            raise JobError("model_required", f"{self.name} needs a model")
        manifest = _known(model)
        if not manifest.supports(self._config.backend_kind):
            return 0
        return manifest.spec(self._config.backend_kind).memory_bytes_estimate

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        return _model_provenance(self._config.backend_kind, model)

    def check(self, backend: Any) -> JobTypeStatus:
        env = llm_engine_status(self._config, backend)
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
        if model is None:
            raise ApiError(400, "model_required", f"{self.name} needs a model")
        loading = _params(LoadParams, params, self.name)
        self._residency.refuse_if_claimed(f"loading {model!r}")
        manifest, spec, _ = _require_loadable(self._config, self._backend, model)
        context = _load_context(self._config, self._backend, manifest, loading)
        state = accelerator.guard(
            self._config.backend_kind,
            model_id=model,
            need_bytes=spec.memory_bytes_estimate,
            owned_pids=self._residency.owned_pids(),
            desktop_allowance_bytes=self._config.desktop_allowance_bytes,
            reclaimable_bytes=self._reclaimable(),
        )
        plan = vram.plan_vllm_memory(
            model_id=model,
            spec=spec,
            context=context,
            card=state,
            desktop_allowance_bytes=self._config.desktop_allowance_bytes,
            reclaimable_bytes=self._reclaimable(),
        )
        if plan is not None and not plan.fits:
            raise ApiError(409, "insufficient_kv_cache", plan.sentence())

    def run(self, job: Job, ctx: JobContext) -> None:
        params = LoadParams.model_validate(job.params)
        model = job.model
        if model is None:
            raise JobError("model_required", f"{self.name} needs a model")
        self._residency.begin_warming(model)
        try:
            self._load(ctx, model, params, job.client)
        finally:
            self._residency.end_warming()

    def _load(
        self,
        ctx: JobContext,
        model: str,
        params: LoadParams,
        client: str | None,
    ) -> None:
        try:
            manifest, spec, (python, installed) = _require_loadable(
                self._config, self._backend, model
            )
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None
        try:
            context = _load_context(self._config, self._backend, manifest, params)
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None

        ctx.warming(f"checking the accelerator for {model}")
        try:
            state = accelerator.guard(
                self._config.backend_kind,
                model_id=model,
                need_bytes=spec.memory_bytes_estimate,
                owned_pids=self._residency.owned_pids(),
                desktop_allowance_bytes=self._config.desktop_allowance_bytes,
                reclaimable_bytes=self._reclaimable(),
            )
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None
        ctx.warming(state.detail)

        plan = vram.plan_vllm_memory(
            model_id=model,
            spec=spec,
            context=context,
            card=state,
            desktop_allowance_bytes=self._config.desktop_allowance_bytes,
            reclaimable_bytes=self._reclaimable(),
        )
        if plan is not None:
            if not plan.fits:
                raise JobError("insufficient_kv_cache", plan.sentence())
            ctx.warming(plan.detail())

        ctx.raise_if_cancelled()
        ctx.progress(0.0, f"loading {model}")
        try:
            resident = self._residency.load(
                manifest,
                spec,
                installed.path,
                python,
                plan=plan,
                context=context,
                timeout=params.timeout_s,
                on_progress=ctx.warming,
                card_args=vllm_engine.card_args(
                    spec, card_for(self._config.home, self._backend.gpu)
                ),
            )
        except EngineError as exc:
            raise JobError("engine_failed", str(exc)) from None
        ctx.raise_if_cancelled()
        extra: dict[str, Any] = {"resident": resident.model_id}
        extra["lease_id"] = None
        if params.lease is not None:
            extra["lease_id"] = _open_lease_for_load(
                self._leases,
                kind=resident.kind,
                subject=resident.model_id,
                request=params.lease,
                client=client,
            )
        ctx.progress(1.0, f"{model} is resident")
        ctx.done_extra(**extra)


def _open_lease_for_load(
    leases: Any | None,
    *,
    kind: str,
    subject: str,
    request: LeaseOnLoad,
    client: str | None,
) -> str:
    if leases is None:
        raise JobError(
            "leases_unavailable",
            "this build has no lease register, so `params.lease` cannot be "
            "honoured. It is refused rather than ignored: a client told nothing "
            "would believe it holds the card",
        )
    try:
        act = require_act_name(request.act.strip(), "a load's `params.lease.act`")
        ttl = require_ttl(request.ttl_seconds)
    except ApiError as exc:
        raise JobError(exc.code, exc.message) from None
    try:
        lease = leases.open(
            kind=kind, subject=subject, act=act, client=client, ttl_seconds=ttl
        )
    except ApiError as exc:
        raise JobError(exc.code, exc.message) from None
    return lease.id


class UnloadModelJobType:
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

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        return _model_provenance(self._config.backend_kind, model)

    def check(self, backend: Any) -> JobTypeStatus:
        resident = self._residency.resident_id
        return JobTypeStatus(
            ready=True,
            detail=(
                f"resident: {resident}" if resident else "nothing is resident"
            ),
        )

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        if model is None:
            raise ApiError(400, "model_required", f"{self.name} needs a model")
        _params(UnloadParams, params, self.name)
        if self._residency.being_cleared(model):
            return
        self._residency.refuse_if_claimed(f"unloading {model!r}")
        if not self._residency.is_resident(KIND_LLM, model):
            raise ApiError(
                409,
                "model_not_resident",
                f"{model!r} is not resident on this server; "
                + describe_resident(self._residency, KIND_LLM, "no model is"),
                {"requested": model, "resident": self._residency.resident_id},
            )

    def run(self, job: Job, ctx: JobContext) -> None:
        UnloadParams.model_validate(job.params)
        model = job.model
        if model is None:
            raise JobError("model_required", f"{self.name} needs a model")
        ctx.progress(0.0, f"unloading {model}")
        if self._residency.await_clearance(model):
            ctx.progress(1.0, f"{model} is unloaded — the card was cleared of it")
            ctx.done_extra(resident=self._residency.resident_id)
            return
        if not self._residency.is_resident(KIND_LLM, model):
            raise JobError(
                "model_not_resident",
                f"{model!r} is not resident on this server; "
                + describe_resident(self._residency, KIND_LLM, "no model is"),
            )
        try:
            self._residency.unload(model)
        except KeyError:
            raise JobError(
                "model_not_resident",
                f"{model!r} is not resident on this server; "
                + describe_resident(self._residency, KIND_LLM, "no model is"),
            ) from None
        except EngineError as exc:
            raise JobError("engine_failed", str(exc)) from None
        ctx.progress(1.0, f"{model} is unloaded")
        ctx.done_extra(resident=self._residency.resident_id)
