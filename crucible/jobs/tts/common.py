"""What `load-voice`, `unload-voice` and `tts` all have to agree about.

These were in `crucible/jobs/tts/__init__.py` while the lifecycle pair was the
whole job type. The render door (PHASE3-TTS.md section 6) needs every one of
them — the same manifest lookup, the same backend spec, the same env and weights
refusals in the same order — and it needs them from a module the package's
`__init__` can import, so they moved here rather than being written twice.

Nothing about their behaviour changed in the move, and the refusal names are the
ones `jobs/tts/__init__.py`'s docstring already lists.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from ... import accelerator, jobenv, weights
from ...config import Config
from ...errors import ApiError
from ...residency import KIND_TTS, Residency
from ...voices import VoiceBackendSpec, VoiceError, VoiceManifest, load_all_voices
from ..base import ModelDescriptor

__all__ = [
    "describe_voices",
    "known_voice",
    "load_voices",
    "require_loadable",
    "validated_params",
    "voice_provenance",
    "voice_rows",
]


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


def known_voice(voice_id: str) -> VoiceManifest:
    """The manifest for this id, or `unknown_voice` by name."""
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


def describe_voices(backend_kind: str, residency: Residency) -> list[ModelDescriptor]:
    """`/v1/info` capabilities rows — DESIGN.md section 4's shape."""
    rows: list[ModelDescriptor] = []
    for manifest in load_voices().values():
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

    `[voice.serving].max_num_seqs` IS NOT ON THE ROW EITHER, by that same rule
    and by the division-of-knowledge ruling behind it: it is how wide the
    server admits, a Crucible-side configuration number, and a client has no
    decision to make with it. It reaches narrator through the engine's
    environment (`crucible/engines/narrator.py`) and stops there.
    """
    backend_kind = backend.kind
    rows: list[dict[str, Any]] = []
    for manifest in load_voices().values():
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


def require_loadable(
    config: Config, backend: Any, voice_id: str
) -> tuple[VoiceManifest, VoiceBackendSpec, Any]:
    """Manifest, backend spec, interpreter and weights, or the named refusal.

    The order is `jobs/llm`'s, and deliberately so: what can never be fixed, then
    what an install or a pull would fix, then what the live accelerator says.
    """
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


def voice_provenance(backend_kind: str, voice_id: str | None) -> dict[str, Any] | None:
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
    manifest = known_voice(voice_id)
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
