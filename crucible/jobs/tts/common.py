from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from ... import accelerator, jobenv, ttsplan, weights
from ...config import Config
from ...errors import ApiError
from ...residency import KIND_TTS, Residency
from ...voicereference import ReferenceError, VoiceReference, parse_reference
from ...voices import VoiceBackendSpec, VoiceError, VoiceManifest, load_all_voices
from ..base import ModelDescriptor

__all__ = [
    "describe_voices",
    "known_voice",
    "load_voices",
    "require_loadable",
    "require_reference",
    "validated_params",
    "voice_provenance",
    "voice_rows",
]


def require_reference(
    manifest: VoiceManifest, reference: Any
) -> VoiceReference | None:
    if manifest.kind == "zeroshot":
        if reference is None:
            raise ApiError(
                400,
                "reference_required",
                f"voice {manifest.id!r} is a zeroshot voice: it is the base "
                "weights conditioned on a recording, and this load carries no "
                "`params.reference`. Send "
                '`{"data": "<base64 wav>", "transcript": "<the book-exact text '
                'spoken in it>"}`. Without one the engine would come up in the '
                "model's own voice — a different speaker at 12 % of the "
                "narrator ceiling — under this voice's id",
                {"voice": manifest.id, "kind": manifest.kind},
            )
    elif reference is not None:
        raise ApiError(
            400,
            "reference_not_allowed",
            f"voice {manifest.id!r} is a {manifest.kind} voice and this load "
            "carries a `params.reference`. A checkpoint's voice is in its "
            "weights and a token voice's is in the engine; a reference here "
            "would clone from the clip and leave the weights this load names "
            "doing nothing",
            {"voice": manifest.id, "kind": manifest.kind},
        )
    if reference is None:
        return None
    try:
        return parse_reference(reference.model_dump())
    except ReferenceError as exc:
        raise ApiError(
            400, exc.code, str(exc), {"voice": manifest.id}
        ) from None


def load_voices() -> dict[str, VoiceManifest]:
    try:
        return load_all_voices()
    except VoiceError as exc:
        raise ApiError(
            500,
            "voices_unreadable",
            f"this server cannot read its voice manifests: {exc}",
        ) from None


def validated_params(
    model: type[BaseModel], params: dict[str, Any], job_type: str
) -> Any:
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


def known_voice(voice_id: str) -> VoiceManifest:
    manifests = load_voices()
    manifest = manifests.get(voice_id)
    if manifest is None:
        raise ApiError(
            400,
            "unknown_voice",
            f"no manifest for voice {voice_id!r}; this build ships "
            f"{sorted(manifests)}",
        )
    return manifest


def describe_voices(config: Config, residency: Residency) -> list[ModelDescriptor]:
    backend_kind = config.backend_kind
    rows: list[ModelDescriptor] = []
    for manifest in load_voices().values():
        if manifest.supports(backend_kind):
            spec = manifest.spec(backend_kind)
            revision, source, estimate = (
                spec.weights_identity,
                spec.hf_repo if spec.hf_repo is not None else f"path:{spec.path}",
                spec.memory_bytes_estimate,
            )
            installed = weights.installed(config, manifest, spec) is not None
        else:
            revision, source, estimate, installed = "", "", 0, False
        rows.append(
            ModelDescriptor(
                id=manifest.id,
                revision=revision,
                source=source,
                installed=installed,
                resident=residency.is_resident(KIND_TTS, manifest.id),
                vram_bytes=estimate,
            )
        )
    return rows


def _orphan(
    voice_id: str, source: str | None, residency: Residency,
    *, leases: Any | None, store: Any | None,
) -> bool | None:
    if leases is None or store is None:
        return None
    if source != "local":
        return False
    if residency.is_resident(KIND_TTS, voice_id):
        return False
    lease = leases.current()
    if lease is not None and lease.subject == voice_id:
        return False
    running = store.running
    if running is not None and running.model == voice_id:
        return False
    return not any(job.model == voice_id for job in store.queued())


def voice_rows(
    config: Config, backend: Any, residency: Residency,
    *, leases: Any | None = None, store: Any | None = None,
) -> list[dict[str, Any]]:
    backend_kind = backend.kind
    rows: list[dict[str, Any]] = []
    for manifest in load_voices().values():
        supported = manifest.supports(backend_kind)
        estimate: int | None = None
        basis: str | None = None
        revision: str | None = None
        source: str | None = None
        identity_basis: str | None = None
        fingerprint: str | None = None
        max_chars: int | None = None
        max_chars_basis: str | None = None
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
            revision = spec.weights_identity
            source = spec.source
            identity_basis = spec.identity_basis
            fingerprint = manifest.fingerprint(backend_kind)
            max_chars = spec.max_chars
            max_chars_basis = spec.max_chars_basis
            is_installed = weights.installed(config, manifest, spec) is not None
            env = jobenv.env_status(
                config.home,
                jobenv.tts_env(manifest.narrator_engine, backend_kind),
                backend_kind,
            )
            if estimate > backend.gpu.vram_bytes:
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
            elif not is_installed and spec.source == weights.LOCAL:
                reason = (
                    f"no weights at {spec.path} — this voice names a directory on "
                    "this server, which Crucible does not fetch and cannot replace"
                )
            elif not is_installed:
                directory = weights.subject_dir(config, manifest, backend_kind)
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
                "orphan": _orphan(
                    manifest.id, source, residency, leases=leases, store=store
                ),
                "loadable": reason is None,
                "reason": reason,
                "revision": revision,
                "fingerprint": fingerprint,
                "source": source,
                "identity_basis": identity_basis,
                "memory_bytes_estimate": estimate,
                "estimate_basis": basis,
                "serving": (
                    None if manifest.serving is None else manifest.serving.to_dict()
                ),
                "max_chars": max_chars,
                "max_chars_basis": max_chars_basis,
                "pace_basis": manifest.pace_basis,
                "inherited_from": manifest.inherited_from,
                "manifest": manifest.manifest_source,
                "sample_rate": manifest.sample_rate,
                "takes": len(manifest.takes),
                "needs_reference": manifest.kind == "zeroshot",
                "pace": manifest.pace.to_dict(),
            }
        )
    from ...voices import unserved_pins

    for voice_id, (revision, why) in sorted(unserved_pins().items()):
        rows.append(
            {
                "id": voice_id, "display": voice_id, "kind": None, "language": None,
                "narrator_engine": None, "backend_supported": False,
                "installed": False, "resident": False, "orphan": False,
                "loadable": False, "reason": why, "revision": revision,
                "fingerprint": None, "source": "pinned", "identity_basis": None,
                "memory_bytes_estimate": None, "estimate_basis": None,
                "serving": None, "max_chars": None, "max_chars_basis": None,
                "pace_basis": None, "inherited_from": None, "manifest": "repo",
                "sample_rate": None, "takes": 0, "needs_reference": False,
                "pace": None,
            }
        )
    return rows


def require_loadable(
    config: Config, backend: Any, voice_id: str
) -> tuple[VoiceManifest, VoiceBackendSpec, Any]:
    backend_kind = backend.kind
    manifest = known_voice(voice_id)
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
        need_bytes=voice_load_plan(config, backend, manifest, spec).floor_bytes,
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
            {
                "voice": voice_id,
                "source": spec.source,
                "hf_repo": spec.hf_repo,
                "revision": spec.revision,
                "path": spec.path,
            },
        ) from None
    return manifest, spec, (python, installed)


def voice_provenance(backend_kind: str, voice_id: str | None) -> dict[str, Any] | None:
    if voice_id is None:
        return None
    manifest = known_voice(voice_id)
    spec = manifest.backends.get(backend_kind)
    if spec is None:
        return {
            "id": voice_id,
            "revision": None,
            "identity_basis": None,
            "fingerprint": None,
        }
    return {
        "id": voice_id,
        "revision": spec.weights_identity,
        "identity_basis": spec.identity_basis,
        "fingerprint": manifest.fingerprint(backend_kind),
    }


def voice_load_plan(
    config: Config, backend: Any, manifest: VoiceManifest, spec: VoiceBackendSpec
) -> "ttsplan.LoadPlan":
    return ttsplan.load_plan(
        manifest,
        spec,
        backend.kind,
        total_bytes=backend.gpu.vram_bytes,
        desktop_allowance_bytes=config.desktop_allowance_bytes,
    )
