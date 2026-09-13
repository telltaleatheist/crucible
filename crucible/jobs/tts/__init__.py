"""The `tts` job type's lifecycle half: `load-voice` and `unload-voice`.

PHASE3-TTS.md section 5, mirroring `jobs/llm/`'s pair exactly. The render door
(`{"type": "tts"}`, section 6) and the streaming door (section 7) are not here
yet; what is here is putting a voice on the accelerator and taking it off, which
is the thing the exclusive lane serialises and the thing every other door depends
on having got right.

Every refusal happens **before the job is queued**, in `preflight()`, so the
client gets an HTTP error naming the thing rather than a job that fails a minute
later:

    unknown_voice        no manifest with that id (raised by jobs.resolve_model)
    backend_unsupported  the manifest has no block for this host's backend
    env_missing          ~/.crucible/envs/tts-<engine> is not installed
    voice_not_installed  no weights at the manifest's pinned revision
    accelerator_busy     somebody else's process is on the card
    insufficient_memory  free memory is below the manifest's estimate
    voice_not_resident   (unload) that voice is not the one that is loaded

**`model` on the wire is the voice id.** DESIGN.md's word for the thing that
produces the bytes is `model`, and for `tts` that thing is the voice — which for
Higgs is the literal truth rather than a pun: a v3 voice *is* the merged
checkpoint the engine was started on, so a voice change is a full worker restart.
No new vocabulary is invented for it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ... import accelerator, jobenv, weights
from ...config import Config
from ...engines import EngineError
from ...errors import ApiError, JobError
from ...residency import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    KIND_TTS,
    Residency,
    describe_resident,
)
from ...voices import VoiceBackendSpec, VoiceError, VoiceManifest, load_all_voices
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor

__all__ = [
    "LoadVoiceJobType",
    "LoadVoiceParams",
    "UnloadVoiceJobType",
    "voice_rows",
]


class LoadVoiceParams(BaseModel):
    """`params` for a load-voice job. Unknown keys are refused, not ignored."""

    model_config = ConfigDict(extra="forbid")

    timeout_s: float = Field(default=DEFAULT_READY_TIMEOUT_SECONDS, ge=30, le=7200)


class UnloadVoiceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ helpers


def _voices() -> dict[str, VoiceManifest]:
    try:
        return load_all_voices()
    except VoiceError as exc:
        raise ApiError(
            500,
            "voices_unreadable",
            f"this server cannot read its voice manifests: {exc}",
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


def _known(voice_id: str) -> VoiceManifest:
    """The manifest for this id, or `unknown_voice` by name."""
    manifests = _voices()
    manifest = manifests.get(voice_id)
    if manifest is None:
        raise ApiError(
            400,
            "unknown_voice",
            f"no manifest for voice {voice_id!r}; this build ships "
            f"{sorted(manifests)}",
        )
    return manifest


def _descriptors(backend_kind: str, residency: Residency) -> list[ModelDescriptor]:
    """`/v1/info` capabilities rows — DESIGN.md section 4's shape."""
    rows: list[ModelDescriptor] = []
    for manifest in _voices().values():
        if manifest.supports(backend_kind):
            spec = manifest.spec(backend_kind)
            revision, source, estimate = (
                spec.revision,
                spec.hf_repo,
                spec.memory_bytes_estimate,
            )
        else:
            revision, source, estimate = "", "", 0
        rows.append(
            ModelDescriptor(
                id=manifest.id,
                revision=revision,
                source=source,
                resident=residency.is_resident(KIND_TTS, manifest.id),
                vram_bytes=estimate,
            )
        )
    return rows


def voice_rows(
    config: Config, backend: Any, residency: Residency
) -> list[dict[str, Any]]:
    """`GET /v1/voices` — PHASE3-TTS.md section 2.

    These same rows are the `tts` capability's rows in `GET /v1/info`: one shape,
    one producer, the same rule and the same reason as `llm`'s models. One voice,
    one description; a client never reconciles two.

    `loadable` answers "is everything this host needs in place", which is a fact
    about the disk. Like `model_rows` it deliberately does **not** run
    nvidia-smi: the accelerator's state changes between a listing and a request,
    so the guard runs at load time. A row saying `loadable: true` can still be
    refused with `accelerator_busy`.

    **`sampling` is deliberately not on the row**, nor are the EOS levers, the
    token-budget formula or the engine flags. That is engine tuning, it is the
    server's, and publishing it invites a client to send it back. What a client
    gets is the shape it must pack to (`pace`, `max_chars`) and the identity it
    must record (`fingerprint`).
    """
    backend_kind = backend.kind
    rows: list[dict[str, Any]] = []
    for manifest in _voices().values():
        supported = manifest.supports(backend_kind)
        estimate: int | None = None
        basis: str | None = None
        revision: str | None = None
        fingerprint: str | None = None
        max_chars: int | None = None
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
            basis = spec.estimate_basis
            revision = spec.revision
            fingerprint = manifest.fingerprint(backend_kind)
            max_chars = spec.max_chars
            is_installed = weights.installed(config, manifest, spec) is not None
            env = jobenv.env_status(
                config.home,
                jobenv.tts_env(manifest.narrator_engine, backend_kind),
                backend_kind,
            )
            if estimate > backend.gpu.vram_bytes:
                # Not loadable here at all, so say so instead of asking for an
                # 8.5 GB download first.
                reason = (
                    f"needs {estimate / 1024 ** 3:.1f} GiB and "
                    f"{backend.gpu.name} has {backend.gpu.vram_bytes / 1024 ** 3:.1f}"
                    " GiB in total"
                )
            elif not env.installed:
                reason = (
                    f"the tts env for {manifest.narrator_engine} is not ready: "
                    f"{env.detail}"
                )
            elif not is_installed:
                directory = weights.weights_dir(
                    config, manifest.weights_family, manifest.id, backend_kind
                )
                reason = (
                    f"no weights at {directory} — run "
                    f"`crucible voices pull {manifest.id}`"
                )
        rows.append(
            {
                "id": manifest.id,
                "display": manifest.display,
                "kind": manifest.kind,
                "language": manifest.language,
                "narrator_engine": manifest.narrator_engine,
                "backend_supported": supported,
                "installed": is_installed,
                "resident": residency.is_resident(KIND_TTS, manifest.id),
                "loadable": reason is None,
                "reason": reason,
                # These four live in the backend block this host may not have,
                # and are null rather than 0 or "" when it does not: a 0 estimate
                # would read as "needs nothing" and an empty revision as a pin.
                "revision": revision,
                "fingerprint": fingerprint,
                "memory_bytes_estimate": estimate,
                # Whether somebody watched the card for that number or it came
                # off the engine's own configured reservation. On the row rather
                # than only in the manifest, so nothing downstream can mistake
                # one for the other (crucible/voices.py).
                "estimate_basis": basis,
                "max_chars": max_chars,
                "sample_rate": manifest.sample_rate,
                "takes": len(manifest.takes),
                "pace": manifest.pace.to_dict(),
            }
        )
    return rows


def _require_loadable(
    config: Config, backend: Any, voice_id: str
) -> tuple[VoiceManifest, VoiceBackendSpec, Any]:
    """Manifest, backend spec, interpreter and weights, or the named refusal.

    The order is `jobs/llm`'s, and deliberately so: what can never be fixed, then
    what an install or a pull would fix, then what the live accelerator says.
    """
    backend_kind = backend.kind
    manifest = _known(voice_id)
    if not manifest.supports(backend_kind):
        raise ApiError(
            400,
            "backend_unsupported",
            f"voice {voice_id!r} has no {backend_kind} block; {manifest.path.name} "
            f"declares {sorted(manifest.backends)}",
            {"voice": voice_id, "backend": backend_kind,
             "declared": sorted(manifest.backends)},
        )
    spec = manifest.spec(backend_kind)
    accelerator.refuse_if_larger_than_host(
        model_id=voice_id,
        need_bytes=spec.memory_bytes_estimate,
        host_total_bytes=backend.gpu.vram_bytes,
        host_name=backend.gpu.name,
    )
    env_spec = jobenv.tts_env(manifest.narrator_engine, backend_kind)
    try:
        python = jobenv.require_env(config.home, env_spec, backend_kind)
    except jobenv.EnvError as exc:
        raise ApiError(
            409,
            "env_missing",
            f"cannot load {voice_id!r}: {exc}",
            {
                "voice": voice_id,
                "narrator_engine": manifest.narrator_engine,
                "env": str(jobenv.env_dir(config.home, env_spec)),
            },
        ) from None
    try:
        installed = weights.require_installed(config, manifest, spec)
    except weights.WeightsError as exc:
        raise ApiError(
            409,
            "voice_not_installed",
            str(exc),
            {"voice": voice_id, "hf_repo": spec.hf_repo, "revision": spec.revision},
        ) from None
    return manifest, spec, (python, installed)


# ----------------------------------------------------------------- load job


def _voice_provenance(backend_kind: str, voice_id: str | None) -> dict[str, Any] | None:
    """The `model` block of a tts artifact's provenance sidecar.

    For `tts` the model IS the voice (PHASE3-TTS.md section 6), so the sidecar
    names it with the same three keys every other type uses rather than inventing
    a fourth word for the same idea. The revision is this host's backend pin,
    which is a statement about bytes: a load refuses weights pulled at any other
    revision, so the pin the manifest names is the checkpoint the engine read —
    and a finished audiobook that says which voice rendered it should also say
    which merge of that voice, because two merges of one fine-tune are two
    narrators.
    """
    if voice_id is None:
        return None
    manifest = _known(voice_id)
    spec = manifest.backends.get(backend_kind)
    if spec is None:
        # Unreachable through the API: `preflight` refuses `backend_unsupported`
        # before a job exists. A sidecar still has to say something true if it is
        # reached another way, and inventing a revision is not it.
        return {"id": voice_id, "revision": None, "fingerprint": None}
    return {
        "id": voice_id,
        "revision": spec.revision,
        "fingerprint": manifest.fingerprint(backend_kind),
    }


class LoadVoiceJobType:
    """`POST /v1/jobs {"type": "load-voice", "model": "<voice id>"}`."""

    name = "load-voice"

    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency

    @property
    def residency(self) -> Residency:
        return self._residency

    def describe_models(self) -> list[ModelDescriptor]:
        return _descriptors(self._config.backend_kind, self._residency)

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        return _voice_provenance(self._config.backend_kind, model)

    def vram_estimate(self, model: str | None) -> int:
        if model is None:
            raise JobError("model_required", f"{self.name} needs a voice")
        manifest = _known(model)
        if not manifest.supports(self._config.backend_kind):
            return 0
        return manifest.spec(self._config.backend_kind).memory_bytes_estimate

    def check(self, backend: Any) -> JobTypeStatus:
        rows = voice_rows(self._config, backend, self._residency)
        ready = [row["id"] for row in rows if row["loadable"]]
        if not ready:
            blocked = sorted(
                {
                    row["reason"]
                    for row in rows
                    if row["reason"] is not None and row["backend_supported"]
                }
            )
            return JobTypeStatus(
                ready=False,
                detail=(
                    "no voice is loadable here: "
                    + ("; ".join(blocked) if blocked else "this build ships none")
                ),
            )
        return JobTypeStatus(ready=True, detail=f"loadable: {ready}")

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        if model is None:  # unreachable: resolve_model requires one
            raise ApiError(400, "model_required", f"{self.name} needs a voice")
        _params(LoadVoiceParams, params, self.name)
        _, spec, _ = _require_loadable(self._config, self._backend, model)
        accelerator.guard(
            self._config.backend_kind,
            model_id=model,
            need_bytes=spec.memory_bytes_estimate,
            owned_pids=self._residency.owned_pids(),
            desktop_allowance_bytes=self._config.desktop_allowance_bytes,
            reclaimable_bytes=self._residency.reclaimable_bytes(excluding=model),
        )

    def run(self, job: Job, ctx: JobContext) -> None:
        params = LoadVoiceParams.model_validate(job.params)
        model = job.model
        if model is None:  # unreachable: resolve_model requires one
            raise JobError("model_required", f"{self.name} needs a voice")
        # `/v1/health` says `warming` for the whole job, not only while the
        # engine's readiness is being polled: from the client's side this server
        # is warming a voice from the moment the lane picks the job up.
        self._residency.begin_warming(model)
        try:
            self._load(ctx, model, params)
        finally:
            self._residency.end_warming()

    def _load(self, ctx: JobContext, model: str, params: LoadVoiceParams) -> None:
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
            resident = self._residency.load_voice(
                manifest,
                spec,
                installed.path,
                python,
                timeout=params.timeout_s,
                on_progress=ctx.warming,
            )
        except NotImplementedError as exc:
            # The seam, reported by name rather than as a traceback: the voice,
            # its weights, its env and the card were all in order, and the only
            # thing missing is the engine that has not been written yet.
            raise JobError("engine_not_implemented", str(exc)) from None
        except EngineError as exc:
            raise JobError("engine_failed", str(exc)) from None
        ctx.progress(1.0, f"{model} is resident")
        ctx.done_extra(resident=resident.voice_id, fingerprint=resident.fingerprint)


# --------------------------------------------------------------- unload job


class UnloadVoiceJobType:
    """`POST /v1/jobs {"type": "unload-voice", "model": "<voice id>"}`."""

    name = "unload-voice"

    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency

    @property
    def residency(self) -> Residency:
        return self._residency

    def describe_models(self) -> list[ModelDescriptor]:
        return _descriptors(self._config.backend_kind, self._residency)

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        return _voice_provenance(self._config.backend_kind, model)

    def vram_estimate(self, model: str | None) -> int:
        return 0

    def check(self, backend: Any) -> JobTypeStatus:
        voice = self._residency.resident_voice
        return JobTypeStatus(
            ready=True,
            detail=(
                f"resident: {voice.voice_id}" if voice else "no voice is resident"
            ),
        )

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        if model is None:  # unreachable: resolve_model requires one
            raise ApiError(400, "model_required", f"{self.name} needs a voice")
        _params(UnloadVoiceParams, params, self.name)
        if not self._residency.is_resident(KIND_TTS, model):
            raise ApiError(
                409,
                "voice_not_resident",
                f"{model!r} is not resident on this server; "
                + describe_resident(self._residency, KIND_TTS, "no voice is"),
                {"requested": model, "resident": self._residency.resident_id},
            )

    def run(self, job: Job, ctx: JobContext) -> None:
        UnloadVoiceParams.model_validate(job.params)
        model = job.model
        if model is None:  # unreachable: resolve_model requires one
            raise JobError("model_required", f"{self.name} needs a voice")
        if not self._residency.is_resident(KIND_TTS, model):
            # Checked before `unload()` rather than caught from it: the holder
            # unloads by id alone, and a model sharing a voice's id would be
            # taken off the card by `unload-voice`.
            raise JobError(
                "voice_not_resident",
                f"{model!r} is not resident on this server; "
                + describe_resident(self._residency, KIND_TTS, "no voice is"),
            )
        ctx.progress(0.0, f"unloading {model}")
        try:
            self._residency.unload(model)
        except EngineError as exc:
            raise JobError("engine_failed", str(exc)) from None
        ctx.progress(1.0, f"{model} is unloaded")
        ctx.done_extra(resident=self._residency.resident_id)
