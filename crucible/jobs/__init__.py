"""The job-type registry.

`ALL_JOB_TYPES` is the vocabulary this build knows. `build_registry()` is the
subset this server is configured to offer. Asking for a type that exists but is not
enabled is refused differently from asking for one that does not exist — the client
is told which of the two it is.
"""

from __future__ import annotations

from typing import Any

from ..errors import ApiError
from ..residency import Residency
from .base import (
    Job,
    JobContext,
    JobType,
    JobTypeStatus,
    ModelDescriptor,
    validate_member_name,
)
from .align import AlignJobType, UnloadAlignerJobType
from .asr import AsrJobType
from .echo import EchoJobType
from .llm import LoadModelJobType, UnloadModelJobType, model_rows
from .rvc import RvcJobType
from .tts import LoadVoiceJobType, TtsJobType, UnloadVoiceJobType, voice_rows

#: The vocabulary this build knows, and which config flag turns each one on.
#: `resolve()` tells "that type does not exist" from "it exists but is off".
ALL_JOB_TYPES: dict[str, str] = {
    AlignJobType.name: "align",
    UnloadAlignerJobType.name: "align",
    AsrJobType.name: "asr",
    EchoJobType.name: "echo",
    LoadModelJobType.name: "llm",
    UnloadModelJobType.name: "llm",
    LoadVoiceJobType.name: "tts",
    TtsJobType.name: "tts",
    UnloadVoiceJobType.name: "tts",
    RvcJobType.name: "rvc",
}


def build_registry(
    config: Any, backend: Any, residency: Residency | None = None
) -> dict[str, JobType]:
    """Instantiate the job types this config enables.

    `residency` is the server instance's one-resident-engine holder, and every
    job type that touches the card is handed the SAME one — that is what makes
    "loading a voice unloads a model" true rather than aspirational. `crucible
    doctor` has no server, so it passes none and gets a fresh (empty) one — which
    is honest: a doctor run cannot see another process's resident engine.
    """
    registry: dict[str, JobType] = {}
    # Hoisted out of the `llm` branch: every job type that runs the accelerator
    # guard needs the same owned-pid set, or it would report Crucible's own
    # resident engine as somebody else's process holding the card. `tts` needs
    # the same holder for a second reason — one card holds one thing, so loading
    # a voice unloads a model and vice versa (PHASE3-TTS.md section 5).
    holder = residency if residency is not None else Residency(config)
    if config.enable_echo:
        registry[EchoJobType.name] = EchoJobType()
    if config.enable_llm:
        registry[LoadModelJobType.name] = LoadModelJobType(config, backend, holder)
        registry[UnloadModelJobType.name] = UnloadModelJobType(config, backend, holder)
    if config.enable_tts:
        registry[LoadVoiceJobType.name] = LoadVoiceJobType(config, backend, holder)
        registry[UnloadVoiceJobType.name] = UnloadVoiceJobType(config, backend, holder)
        registry[TtsJobType.name] = TtsJobType(config, backend, holder)
    if config.enable_asr:
        registry[AsrJobType.name] = AsrJobType(config, backend, holder.owned_pids)
    if config.enable_align:
        # The SAME holder: an aligner is a third kind of resident thing, so
        # loading one unloads a model or a voice exactly as those unload each
        # other (PHASE4-AUDIO.md section 2, crucible/residency.py).
        registry[AlignJobType.name] = AlignJobType(config, backend, holder)
        registry[UnloadAlignerJobType.name] = UnloadAlignerJobType(
            config, backend, holder
        )
    if config.enable_rvc:
        # Only the owned-pid set, like `asr`: nothing is ever resident for
        # `rvc`, whose whole design is a process that exits every 96 files.
        registry[RvcJobType.name] = RvcJobType(config, backend, holder.owned_pids)
    _assert_every_type_implements_the_protocol(registry)
    return registry


#: What `JobType` asks of a plugin. Read off the Protocol rather than typed out,
#: so adding a member there extends this on its own — which is the whole point,
#: since the bug this guards against was a member added in one branch and a type
#: written in another. `name` is excluded: every type sets its own, and it is the
#: key the registry is built from, so a type without one never gets this far.
#:
#: `typing.Protocol` exposes `__protocol_attrs__` only from 3.12, and this server
#: supports 3.11, so the members are taken from the class body: methods live in
#: `vars()`, annotated attributes in `__annotations__`, and a Protocol has no
#: other kind of member.
_JOB_TYPE_MEMBERS: tuple[str, ...] = tuple(
    sorted(
        {
            name
            for name in list(vars(JobType)) + list(getattr(JobType, "__annotations__", {}))
            if not name.startswith("_") and name != "name"
        }
    )
)


def _assert_every_type_implements_the_protocol(registry: dict[str, JobType]) -> None:
    """Refuse to serve a job type that does not implement all of `JobType`.

    `Protocol` is a static claim, and nothing was checking it: `asr` was written
    against a `JobType` that had six methods, `llm` added a seventh
    (`model_provenance`) in the same week, and the two met in a merge. Nothing
    complained at import, at startup, or when a job was accepted. The failure
    arrived where it could do the most damage — inside `JobStore._finish`, while
    writing a finished job's provenance, on the event loop, so the job emitted no
    terminal event and every client reading its stream waited forever.

    A missing member is a build-time fact, so it is refused at build time, by
    name, before the server answers anything. `runtime_checkable` would only
    check the members that `isinstance` knows about on the instance; this asks
    the Protocol what it declares.
    """
    for job_type, plugin in sorted(registry.items()):
        missing = [m for m in _JOB_TYPE_MEMBERS if not hasattr(plugin, m)]
        if missing:
            raise TypeError(
                f"job type {job_type!r} ({type(plugin).__name__}) does not "
                f"implement JobType: missing {sorted(missing)}. Every member of "
                "the protocol is called by the queue or the API; a type that is "
                "missing one fails somewhere far from here."
            )


def resolve(registry: dict[str, JobType], job_type: str) -> JobType:
    """Look up a job type or refuse by name."""
    plugin = registry.get(job_type)
    if plugin is not None:
        return plugin
    if job_type in ALL_JOB_TYPES:
        raise ApiError(
            400,
            "job_type_disabled",
            f"job type {job_type!r} is not enabled on this server "
            f"(set [jobs] enable_{ALL_JOB_TYPES[job_type]} = true in config.toml)",
        )
    raise ApiError(
        400,
        "unknown_job_type",
        f"unknown job type {job_type!r}; this server offers "
        f"{sorted(registry) if registry else 'no job types'}",
    )


def resolve_model(plugin: JobType, model: str | None) -> str | None:
    """Check the requested model against what the type advertises, or refuse by name."""
    offered = plugin.describe_models()
    ids = [descriptor.id for descriptor in offered]
    if model is None:
        if ids:
            raise ApiError(
                400,
                "model_required",
                f"job type {plugin.name!r} requires a model; it offers {ids}",
            )
        return None
    if model not in ids:
        raise ApiError(
            400,
            "unknown_model",
            f"job type {plugin.name!r} does not serve model {model!r}; it offers "
            f"{ids if ids else 'no models (omit `model`)'}",
        )
    return model


__all__ = [
    "ALL_JOB_TYPES",
    "AlignJobType",
    "AsrJobType",
    "EchoJobType",
    "Job",
    "JobContext",
    "JobType",
    "JobTypeStatus",
    "LoadModelJobType",
    "LoadVoiceJobType",
    "ModelDescriptor",
    "TtsJobType",
    "Residency",
    "RvcJobType",
    "UnloadAlignerJobType",
    "UnloadModelJobType",
    "UnloadVoiceJobType",
    "build_registry",
    "model_rows",
    "resolve",
    "resolve_model",
    "validate_member_name",
    "voice_rows",
]
