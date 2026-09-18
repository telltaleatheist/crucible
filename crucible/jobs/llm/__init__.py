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

from ... import accelerator, jobenv, llamacpp, weights
from ...backend import LLAMA_WINDOWS
from ...capability import available_bytes
from ... import ollamastore
from ...config import Config
from ...engines import EngineError
from ...errors import ApiError, JobError
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


# --------------------------------------------- what serves a model, per backend
#
# THREE BACKENDS, TWO ANSWERS. On `cuda-linux` and `mlx-darwin` the thing that
# serves a model is a Python env `crucible install llm` builds, and the engine
# is a module inside it. On `llama-windows` it is `llama-server.exe` — a zip
# from a pinned llama.cpp release, the `engine` SUBJECT (PHASE15-HOST.md 3.10,
# fact 1) — and there is no env at all: `crucible/envpack.py` publishes no
# `envs/llm/llama-windows.txt` and there is nothing for one to contain.
#
# Both answers are the same THREE facts — is it here, what do we say about it,
# and what does the residency spawn — so they are one function returning them,
# rather than a `backend_kind ==` at each of the four places that used to ask
# `jobenv` directly.


class LlmEngineStatus:
    """Is this host's llm engine installed, what to say, and what to spawn."""

    def __init__(self, installed: bool, detail: str, executable: "Any | None") -> None:
        self.installed = installed
        self.detail = detail
        #: The interpreter (a Python env) or the binary (`llama-server.exe`)
        #: `Residency.load` starts. `None` when nothing is installed.
        self.executable = executable


def llm_engine_status(config: Config, backend: Any) -> LlmEngineStatus:
    """The one reader of "can this host start a model at all"."""
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
            revision = spec.revision
            source = spec.hf_repo
            estimate = spec.memory_bytes_estimate
            # The same predicate `model_rows` and `_require_loadable` read: the
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
                resident=residency.is_resident(KIND_LLM, manifest.id),
                vram_bytes=estimate,
            )
        )
    return rows


def _in_the_ollama_store(manifest: Any, backend_kind: str) -> dict[str, Any] | None:
    """The copy of this model Ollama already holds, or None.

    None for six different reasons and that is deliberate here, unlike in
    `ollamastore.resident` which raises and names each one: this runs on a
    listing route for every model on every read, and a store that is absent or a
    tag that was never pulled is the ORDINARY case, not an error. A caller that
    wants the reason asks `ollamastore` directly.
    """
    if backend_kind != LLAMA_WINDOWS:
        return None
    local = manifest.local
    if local is None or getattr(local, "tag", None) is None:
        return None
    root = ollamastore.store_root()
    try:
        found = ollamastore.resident(
            root, local.tag, wants_projector="image" in manifest.modalities
        )
    except ollamastore.OllamaStoreError:
        return None
    return {
        "tag": found.tag,
        "bytes": found.bytes,
        "provenance": found.provenance,
        "path": str(found.model.path),
        # SAY THAT IT IS NOT THE SAME FILE, on the row itself, because the id
        # beside it is the same id. Ollama's `qwen3.5:9b-bf16` and this block's
        # `unsloth/Qwen3.5-9B-GGUF` `Q8_0` are two quantizations of one model by
        # two different people, and a reader who assumes otherwise builds a cache
        # key that answers for weights that never ran.
        "same_file_as_the_pin": False,
    }


def model_rows(
    config: Config, backend: Any, residency: Residency
) -> list[dict[str, Any]]:
    """`GET /v1/models` — PHASE2-LLM.md section 5.

    These same rows are the `llm` capability's rows in `GET /v1/info`: one shape,
    one producer, so a client that has called `info()` never has to ask twice or
    reconcile two descriptions of the same model.

    `loadable` answers "is everything this host needs in place", which is a fact
    about the disk. It deliberately does **not** run nvidia-smi: the accelerator's
    state changes between a listing and a request, so the guard runs at load time
    and refuses there. A row that says `loadable: true` can still be refused with
    `accelerator_busy`.
    """
    backend_kind = backend.kind
    env = llm_engine_status(config, backend)
    # `resident_model`, not `resident`: one card holds one thing and that thing
    # may be a voice (PHASE3-TTS.md section 5). A voice on the card means no
    # model is resident, which is exactly what these rows should say — reading
    # `resident` here would ask a `ResidentVoice` for a `model_id`.
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
                # THE CEILING, run backwards out of the same terms that decide
                # whether the model fits at all (docs/FITS-AND-THE-CARD.md
                # section 6.3). At one request in flight, because that is the
                # number a client sizing ONE request needs; a client batching
                # four divides by four, and the class about to run already
                # declares its own concurrency.
                #
                # Against the card's FREE budget, not its total: vLLM's
                # `--gpu-memory-utilization` is a fraction of the total and the
                # Windows desktop is spent on top of it, which is the measured
                # defect this whole design started from (section 0a).
                afforded = terms.max_context(
                    available_bytes=available_bytes(
                        backend.gpu.vram_bytes, config.desktop_allowance_bytes
                    ),
                    concurrency=1,
                )
                # BOTH WALLS, AND THE LOWER OF THEM, rather than one number with
                # the reasoning swallowed. The card and the weights each impose a
                # limit and they are limits of different kinds: more VRAM raises
                # the first and nothing raises the second. A client reads
                # `tokens`; a person reading a refusal wants to know WHICH wall
                # they hit, because one of them is worth buying a bigger card for
                # and the other is not.
                #
                # This is also the field that caught itself: before
                # `trained_context` existed, a 64 GB Mac afforded 1_389_135
                # tokens of a checkpoint trained at 262_144, and publishing that
                # would have been Crucible doing precisely what section 6.1
                # refuses Ollama for.
                ceiling = {
                    "tokens": min(afforded, manifest.trained_context),
                    "card_affords": afforded,
                    "weights_allow": manifest.trained_context,
                    "limited_by": (
                        "card" if afforded <= manifest.trained_context else "weights"
                    ),
                    "concurrency": 1,
                    "basis": terms.basis,
                }
            # For the model that is up, the number the engine was actually
            # started with, read off the engine's own record; for everything else
            # the number this host would start it with. A manifest edited under a
            # resident engine is the case that makes the distinction real, and it
            # is the resident engine that wins, because that is the context a
            # request sent right now will be measured against.
            max_model_len = (
                resident.max_model_len
                if resident is not None and resident.model_id == manifest.id
                else manifest.context_for(backend_kind)
            )
            is_installed = weights.installed(config, manifest, spec) is not None
            already_here = _in_the_ollama_store(manifest, backend_kind)
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
                directory = weights.weights_dir(
                    config, manifest.weights_family, manifest.id, backend_kind
                )
                reason = (
                    f"no weights at {directory}"
                    f" — run `crucible models pull {manifest.id}`"
                )
        row: dict[str, Any] = {
            "id": manifest.id,
            "family": manifest.family,
            "params_b": manifest.params_b,
            # The revision this host would serve: the pin in *this* backend's
            # block, not the model's "version". A model with no block for this
            # backend has no revision here at all, and says so with null rather
            # than with an empty string that would read as a real pin.
            "revision": revision,
            # `id` and `revision` joined — exactly those two fields of this same
            # row, so it can never disagree with them, and null wherever
            # `revision` is. It is spelled out rather than left to the client to
            # assemble because it is a *record*: Foundry hashes it into the
            # cleanup cache key and BookForge stamps it into a book's OPF
            # (CLIENT-SURFACES.md section 6.5), and two clients each inventing
            # their own way of writing it down is two ways for the same weights
            # to be filed under different names.
            "fingerprint": (
                None if revision is None else fingerprint(manifest.id, revision)
            ),
            # What a client may put in a chat request's content parts. Unlike
            # `revision` and `memory_bytes_estimate` this is not a per-host fact
            # and is never null: it says what the model is offered FOR, which is
            # the same answer on a host that cannot serve it at all. A page
            # reader picks an image-capable model from this rather than knowing
            # one by name (PHASE3-VLM.md section 2).
            "modalities": list(manifest.modalities),
            "backend_supported": supported,
            "installed": is_installed,
            # ----------------------------------- a copy this machine already has
            #
            # Owen, 2026-09-16: *"if its possible to use the ollama copies that
            # already exist on disk then we should do that. i dont want to have
            # 16 copies of giant models sitting around."* A settings page cannot
            # act on that unless it can SEE the copy, and offering a 19 GB
            # download beside a file the machine already holds is the whole of
            # the complaint.
            #
            # `llama-windows` ONLY, and the null on every other backend is a fact
            # rather than an omission: Ollama stores GGUF (measured — the largest
            # blob in Owen's 64 GB store begins with the bytes `GGUF`), llama.cpp
            # reads GGUF, and vLLM and mlx-lm want safetensors. A cuda-linux host
            # saves nothing by having Ollama installed and must not be shown a
            # row suggesting otherwise.
            #
            # NOTHING LOADS FROM HERE YET. This says the bytes exist and what
            # they would be called in a record; the load path is a separate
            # change, because serving them means serving a DIFFERENT
            # quantization from the one this block pins and that has to travel
            # into provenance rather than be swapped in quietly.
            "ollama_copy": already_here,
            "resident": residency.is_resident(KIND_LLM, manifest.id),
            "loadable": reason is None,
            "memory_bytes_estimate": estimate,
            # The manifest's INTENT: the context THIS host would serve, the same
            # way `revision` and `memory_bytes_estimate` above are this host's. A
            # backend may carry its own; where it does not, this is the model's
            # own number, so a host with no block for this model still reports
            # something true.
            "context_default": manifest.context_for(backend_kind),
            # What is being served RIGHT NOW, which is a different question and
            # is why it is a different field. A client sizes a request against
            # this one: Foundry's `capFor` is
            # `max_model_len − (⌈chars/2.5⌉ + 256)` and has **no clamp at all**
            # when the server does not report the field, so the request goes out
            # unclamped and comes back a 400 (CLIENT-SURFACES.md section 6.1).
            # Null when `backend_supported` is false, for the same reason
            # `revision` and `memory_bytes_estimate` are: the number lives in a
            # backend block this manifest does not have.
            "max_model_len": max_model_len,
            # What the WEIGHTS support, which is neither of the two contexts
            # above: `context_default` is what this host chose and
            # `max_model_len` is what the engine was started with, and both are
            # decisions. This one is a property of the checkpoint and is the same
            # on every machine, which is why it is not null for an unsupported
            # backend the way `revision` is.
            "trained_context": manifest.trained_context,
            # --------------------------------------------------- the ceiling
            #
            # HOW FAR THE WALL COULD MOVE, as against `max_model_len`, which is
            # where the wall IS. A request longer than `max_model_len` is refused
            # right now; a request longer than `max_context.tokens` cannot be
            # served by this machine at all, whatever it is restarted with. The
            # gap between the two is exactly what a settings door may offer.
            #
            # Owen's Mac-versus-PC case is this field: same model, same act, two
            # numbers, and an app that reads it never has to discover the
            # difference as a 400. Null where this backend block has not been
            # taken apart into terms — null rather than a guess, because a
            # ceiling stated without arithmetic behind it is the thing Ollama
            # does (section 6.1: it clamps a million to 262144 and tells nobody).
            "max_context": ceiling,
            # The terms the ceiling came out of, so a client can show the working
            # rather than take a figure on trust — and so the `basis` travels:
            # `measured` was watched on a card, `computed` came from config.json
            # and is a FLOOR, `declared` is an allowance nobody has checked.
            "memory_terms": None if terms is None else terms.to_dict(),
            # What a chat request that states nothing will be answered with
            # (PHASE2-LLM.md section 9). For the resident model this is the
            # record the proxy is ACTUALLY applying, read off the engine's own
            # row, for `max_model_len`'s reason directly above: a manifest
            # edited under a running engine must not make this row promise a
            # temperature nothing is sending. Every key is present and `null`
            # means "this model states none, so the engine's own default".
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
    """The `model` block of a provenance sidecar (DESIGN.md section 7).

    `revision` is the sha this host's backend block pins, and that is a statement
    about bytes and not merely about a file: a load refuses weights pulled at any
    other revision (`weights.require_installed`), so the pin the manifest names is
    the pin the engine read.

    `fingerprint` is the two joined, because that is the string a client writes
    down. A finished audiobook says which server rendered it; it now also says
    which weights, which is what makes two renders at two precisions tellable
    apart in their records.
    """
    if model is None:
        return None
    manifest = _known(model)
    spec = manifest.backends.get(backend_kind)
    if spec is None:
        # Unreachable through the API — `preflight` refuses `backend_unsupported`
        # long before a job exists — but a sidecar has to say something true even
        # if it is reached some other way, and inventing a revision is not it.
        return {"id": model, "revision": None, "fingerprint": None}
    return {
        "id": model,
        "revision": spec.revision,
        "fingerprint": fingerprint(model, spec.revision),
    }


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
    if backend_kind == LLAMA_WINDOWS:
        # NO ENV. The engine is `llama-server.exe` from the pinned llama.cpp
        # release, and `env_missing` is still the right NAME for "this host
        # cannot start anything yet" — one refusal for one fact, whichever
        # form the engine takes on this backend.
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
        if model is None:  # unreachable: resolve_model requires one
            raise ApiError(400, "model_required", f"{self.name} needs a model")
        _params(LoadParams, params, self.name)
        # One card, one resident engine, and now one more thing that can hold it:
        # a `tts` streaming session is not a job and does not queue behind this
        # lane, so loading a model over it would end somebody's sentence
        # (PHASE3-TTS.md section 7).
        self._residency.refuse_if_claimed(f"loading {model!r}")
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
        # `/v1/health` says `warming` for the whole job, not only while the
        # engine's readiness is being polled: from the client's side this server
        # is warming a model from the moment the lane picks the job up.
        self._residency.begin_warming(model)
        try:
            self._load(ctx, model, params)
        finally:
            self._residency.end_warming()

    def _load(self, ctx: JobContext, model: str, params: LoadParams) -> None:
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
        if model is None:  # unreachable: resolve_model requires one
            raise ApiError(400, "model_required", f"{self.name} needs a model")
        _params(UnloadParams, params, self.name)
        if self._residency.being_cleared(model):
            # THE SAME INTENT, ALREADY UNDER WAY (T6, 2026-09-15). The
            # settlement began clearing this very model the instant the last
            # chat finished, and this request is that chat's own client tidying
            # up after itself. Admitted: `run` waits the clearance out and
            # reports the empty card it asked for. Only a holder that is USING
            # the card is a conflict, and the line below still says so.
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
        if model is None:  # unreachable: resolve_model requires one
            raise JobError("model_required", f"{self.name} needs a model")
        ctx.progress(0.0, f"unloading {model}")
        if self._residency.await_clearance(model):
            # The settlement got there first. That is the card this job asked
            # for, so it is `done` in the same shape an unload it did itself
            # would be — a client that loads, uses and unloads is never told
            # its own tidying up failed.
            ctx.progress(1.0, f"{model} is unloaded — the card was cleared of it")
            ctx.done_extra(resident=self._residency.resident_id)
            return
        if not self._residency.is_resident(KIND_LLM, model):
            # Checked before `unload()` rather than caught from it: the holder
            # unloads by id alone, and a voice sharing a model's id would be
            # taken off the card by `unload-model`.
            raise JobError(
                "model_not_resident",
                f"{model!r} is not resident on this server; "
                + describe_resident(self._residency, KIND_LLM, "no model is"),
            )
        try:
            self._residency.unload(model)
        except KeyError:  # pragma: no cover - is_resident just said it is
            raise JobError(
                "model_not_resident",
                f"{model!r} is not resident on this server; "
                + describe_resident(self._residency, KIND_LLM, "no model is"),
            ) from None
        except EngineError as exc:
            raise JobError("engine_failed", str(exc)) from None
        ctx.progress(1.0, f"{model} is unloaded")
        ctx.done_extra(resident=self._residency.resident_id)
