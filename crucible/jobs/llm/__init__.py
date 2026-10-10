from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

from pydantic import BaseModel, ConfigDict, Field

from ... import (
    accelerator,
    engines,
    enginespec,
    hosttools,
    jobenv,
    llamacpp,
    ollamastore,
    vram,
    weights,
)
from ...backend import CUDA_LINUX, LLAMA_WINDOWS
from ...cardfacts import card_for
from ...cardkinds import KIND_LLM
from ...clock import utcnow
from ...config import Config
from ...contextceiling import MIN_LOAD_CONTEXT, check_load_context
from ...engines import EngineError, engine_load_args, find_free_port, start_engine
from ...errors import ApiError, JobError
from ...fit import Candidate
from ...jobtypes import LOAD_MODEL, UNLOAD_MODEL
from ...manifests import (
    GGUF_ENGINE,
    BackendSpec,
    ManifestError,
    ModelManifest,
    fingerprint,
    load_all_manifests,
)
from ...memorybudget import available_bytes
from ...residency import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    Occupant,
    Residency,
    ResidentModel,
    say_to,
)
from .. import worker_type
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor
from ..binding import JobTypeBinding
from ..template import (
    ManifestCatalog,
    as_job_error,
    card_guard,
    parse_params,
    require_model,
    run_model,
)
from ..unload import UnloadJobType

__all__ = [
    "JOB_TYPES",
    "LlmEngineStatus",
    "LoadModelJobType",
    "LoadParams",
    "Residency",
    "UnloadModelJobType",
    "llm_engine_status",
    "model_rows",
    "occupy_model",
]


class LlmEngineStatus:
    def __init__(
        self,
        installed: bool,
        detail: str,
        executable: "Any | None",
        library_dirs: "tuple[Path, ...]" = (),
        env: "Path | None" = None,
    ) -> None:
        self.installed = installed
        self.detail = detail
        self.executable = executable
        self.library_dirs = library_dirs
        self.env = env


def llm_engine_status(
    config: Config, backend: Any, engine: str | None = None
) -> LlmEngineStatus:
    """Whether `engine` can start here, and what it starts from. With no engine, whether
    this backend's whole llm install is here: everything `crucible install llm` places,
    which on cuda-linux is the vLLM env AND, where Crucible pins a build for the platform,
    the llama-server a GGUF block runs on."""
    if backend.kind == CUDA_LINUX:
        if engine == GGUF_ENGINE:
            return _cuda_linux_llama_status(config)
        vllm = _env_status(config, backend)
        if engine is not None or not vllm.installed or hosttools.llama_server_build() is None:
            return vllm
        llama = _cuda_linux_llama_status(config)
        if not llama.installed:
            return LlmEngineStatus(
                installed=False, detail=llama.detail, executable=None, env=llama.env
            )
        return LlmEngineStatus(
            installed=True,
            detail=f"{vllm.detail}; {llama.detail}",
            executable=vllm.executable,
            env=vllm.env,
        )
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
            env=llamacpp.engine_dir(config),
        )
    return _env_status(config, backend)


def _env_status(config: Config, backend: Any) -> LlmEngineStatus:
    spec = jobenv.llm_env(backend.kind)
    env = jobenv.env_status(config.home, spec, backend.kind)
    return LlmEngineStatus(
        installed=env.installed,
        detail=env.detail,
        executable=jobenv.env_python(config.home, spec) if env.installed else None,
        env=jobenv.env_dir(config.home, spec),
    )


def _cuda_linux_llama_status(config: Config) -> LlmEngineStatus:
    found = llamacpp.cuda_linux_engine(config.home)
    return LlmEngineStatus(
        installed=found.installed,
        detail=found.detail,
        executable=found.executable,
        library_dirs=found.library_dirs,
        env=jobenv.env_dir(config.home, jobenv.llm_env(CUDA_LINUX)),
    )


class LoadParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_s: float = Field(
        default=DEFAULT_READY_TIMEOUT_SECONDS,
        ge=30,
        le=7200,
        description="Seconds to wait for the engine to come up and answer before "
        "the load fails, 30 to 7200 (default 900).",
    )
    context: int | None = Field(
        default=None,
        ge=MIN_LOAD_CONTEXT,
        strict=True,
        description="The context length in tokens to start the engine at, at least "
        "2048; null uses the model's own default. Above this host's ceiling it is "
        "refused `context_over_limit`; loading the resident model at a new context "
        "is a reload.",
    )


MANIFESTS: ManifestCatalog[ModelManifest] = ManifestCatalog(
    lambda: load_all_manifests(),
    ManifestError,
    unreadable_code="manifests_unreadable",
    what="model manifests",
    unknown="manifest for model",
)


def _descriptors(
    config: Config, backend_kind: str, residency: Residency
) -> list[ModelDescriptor]:
    return MANIFESTS.descriptors(
        backend_kind,
        installed=lambda manifest, spec: weights.installed(config, manifest, spec)
        is not None,
        resident=lambda model_id: residency.is_resident(KIND_LLM, model_id),
    )


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
    engines_here: dict[str, LlmEngineStatus] = {}

    def engine_status(name: str) -> LlmEngineStatus:
        if name not in engines_here:
            engines_here[name] = llm_engine_status(config, backend, name)
        return engines_here[name]

    resident = residency.resident_model
    rows: list[dict[str, Any]] = []
    for manifest in MANIFESTS.all().values():
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
            elif not engine_status(spec.engine).installed:
                reason = (
                    f"the llm env is not ready for {spec.engine}: "
                    f"{engine_status(spec.engine).detail}"
                )
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


def _require_loadable(
    config: Config, backend: Any, model_id: str
) -> tuple[ModelManifest, Any, Any]:
    backend_kind = backend.kind
    manifest = MANIFESTS.known(model_id)
    spec = worker_type.require_block(manifest, model_id, backend_kind, "model")
    worker_type.refuse_if_larger_than_host(backend, model_id, spec.memory_bytes_estimate)
    accelerator.refuse_if_card_lacks(
        model_id=model_id, spec=spec, card=card_for(config.home, backend.gpu)
    )
    if backend_kind == LLAMA_WINDOWS or spec.engine == GGUF_ENGINE:
        engine = llm_engine_status(config, backend, spec.engine)
        if not engine.installed:
            raise ApiError(
                409,
                "env_missing",
                f"cannot load {model_id!r}: {engine.detail}",
                {"model": model_id, "env": str(engine.env)},
            )
        launch = EngineLaunch(engine.executable, engine.library_dirs)
    else:
        try:
            env_spec = jobenv.llm_env(backend_kind)
            launch = EngineLaunch(jobenv.require_env(config.home, env_spec, backend_kind))
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
    return manifest, spec, (launch, installed)


@dataclass(frozen=True)
class EngineLaunch:
    """What an engine is started from: its executable (the env's python, or a native
    server binary) and the directories its shared libraries load from, if any."""

    executable: Path
    library_dirs: tuple[Path, ...] = ()


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


@dataclass(frozen=True)
class LoadNeeds:
    manifest: ModelManifest
    spec: Any
    launch: EngineLaunch
    installed: Any
    context: int
    state: Any
    plan: Any


class LoadModelJobType:
    name = LOAD_MODEL.name

    def __init__(
        self,
        config: Config,
        backend: Any,
        residency: Residency,
    ) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency

    @property
    def residency(self) -> Residency:
        return self._residency

    def describe_models(self) -> list[ModelDescriptor]:
        return _descriptors(self._config, self._config.backend_kind, self._residency)

    def _reclaimable(self) -> int:
        return self._residency.reclaimable_bytes()

    def vram_estimate(self, model: str | None) -> int:
        model = run_model(model, self.name)
        return MANIFESTS.memory_estimate(model, self._config.backend_kind)

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        return MANIFESTS.provenance(self._config.backend_kind, model)

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

    def requirements(self, model: str, params: LoadParams) -> LoadNeeds:
        manifest, spec, (launch, installed) = _require_loadable(
            self._config, self._backend, model
        )
        context = _load_context(self._config, self._backend, manifest, params)
        state = card_guard(
            self._config,
            model=model,
            need_bytes=spec.memory_bytes_estimate,
            owned_pids=self._residency.owned_pids(),
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
        return LoadNeeds(manifest, spec, launch, installed, context, state, plan)

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        model = require_model(model, self.name)
        loading = parse_params(LoadParams, params, self.name)
        self._residency.refuse_if_claimed(f"loading {model!r}")
        self.requirements(model, loading)

    def run(self, job: Job, ctx: JobContext) -> None:
        params = LoadParams.model_validate(job.params)
        model = run_model(job.model, self.name)
        self._residency.begin_warming(model, KIND_LLM)
        try:
            self._load(ctx, model, params)
        finally:
            self._residency.end_warming()

    def _load(
        self,
        ctx: JobContext,
        model: str,
        params: LoadParams,
    ) -> None:
        ctx.warming(f"checking the accelerator for {model}")
        needs = as_job_error(self.requirements, model, params)
        ctx.warming(needs.state.detail)
        if needs.plan is not None:
            ctx.warming(needs.plan.detail())

        ctx.raise_if_cancelled()
        ctx.progress(0.0, f"loading {model}")
        try:
            resident = occupy_model(
                self._residency,
                needs.manifest,
                needs.spec,
                needs.installed.path,
                needs.launch.executable,
                plan=needs.plan,
                context=needs.context,
                timeout=params.timeout_s,
                on_progress=ctx.warming,
                cancelled=lambda: ctx.cancelled,
                card_args=enginespec.card_args(
                    needs.spec, card_for(self._config.home, self._backend.gpu)
                ),
                concurrency=self._config.concurrency_for(model),
                library_dirs=needs.launch.library_dirs,
            )
        except EngineError as exc:
            raise JobError("engine_failed", str(exc)) from None
        ctx.raise_if_cancelled()
        extra: dict[str, Any] = {"resident": resident.model_id}
        ctx.progress(1.0, f"{model} is resident")
        ctx.done_extra(**extra)


def occupy_model(
    residency: Residency,
    manifest: ModelManifest,
    spec: BackendSpec,
    weights_dir: Path,
    python: Path,
    *,
    plan: "vram.KvPlan | None",
    context: int,
    timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
    on_progress: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool],
    card_args: tuple[str, ...] = (),
    concurrency: int | None = None,
    library_dirs: tuple[Path, ...] = (),
) -> ResidentModel:
    say = say_to(on_progress)

    def start() -> Occupant:
        log_path = residency.log_path_for(manifest.id)
        engine = engines.build_engine(
            spec.engine, python, log_path, library_dirs=library_dirs
        )
        served = engines.engine_model_name(
            spec.engine, weights_dir, manifest.id
        )
        port = find_free_port()
        say(
            f"starting {spec.engine} for {manifest.id} on 127.0.0.1:{port} "
            f"(context {context}); log {log_path}"
        )
        args = engine_load_args(
            manifest,
            spec,
            weights_dir,
            plan,
            context=context,
            card_args=card_args,
            concurrency=concurrency,
        )
        start_engine(
            engine, weights_dir, served, port, args, say, timeout, cancelled=cancelled
        )
        resident = ResidentModel(
            model_id=manifest.id,
            backend=spec.backend,
            engine=spec.engine,
            engine_model_name=served,
            base_url=engine.base_url,
            port=port,
            revision=spec.revision,
            max_model_len=context,
            defaults=manifest.defaults,
            memory_bytes_estimate=spec.memory_bytes_estimate,
            log_path=log_path,
            loaded_at=utcnow(),
            engine_args=tuple(args),
        )
        return Occupant(resident, engine=engine, base_url=engine.base_url)

    return cast(ResidentModel, residency.occupy(KIND_LLM, manifest.id, start, say=say))


class UnloadModelJobType(UnloadJobType):
    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        super().__init__(
            UNLOAD_MODEL,
            residency,
            describe=lambda: _descriptors(config, config.backend_kind, residency),
            provenance=lambda model: MANIFESTS.provenance(config.backend_kind, model),
        )


JOB_TYPES: tuple[JobTypeBinding, ...] = (
    JobTypeBinding(
        LOAD_MODEL,
        lambda wiring: LoadModelJobType(
            wiring.config, wiring.backend, wiring.residency
        ),
    ),
    JobTypeBinding(
        UNLOAD_MODEL,
        lambda wiring: UnloadModelJobType(
            wiring.config, wiring.backend, wiring.residency
        ),
    ),
)
