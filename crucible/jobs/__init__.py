from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ..capability import CLASSES, classes_for_job_type
from ..errors import ApiError
from ..jobenv import INSTALLER_FOR
from ..residency import Residency
from .base import (
    OPTIONAL_JOB_TYPE_MEMBERS,
    Job,
    JobContext,
    JobType,
    JobTypeStatus,
    ModelDescriptor,
    validate_member_name,
)
from .align import AlignJobType, UnloadAlignerJobType
from .alignlongform.jobtype import AlignLongformJobType
from .asr import AsrJobType
from .denoise import DenoiseJobType, UnloadDenoiserJobType
from .echo import EchoJobType
from .llm import LoadModelJobType, UnloadModelJobType, model_rows
from .rvc import RvcJobType
from .tts import LoadVoiceJobType, TtsJobType, UnloadVoiceJobType, voice_rows

if TYPE_CHECKING:
    from ..backend import Backend
    from ..config import Config
    from ..leases import Leases

ALL_JOB_TYPES: dict[str, str] = {
    AlignJobType.name: "align",
    UnloadAlignerJobType.name: "align",
    AlignLongformJobType.name: "align",
    AsrJobType.name: "asr",
    DenoiseJobType.name: "denoise",
    UnloadDenoiserJobType.name: "denoise",
    EchoJobType.name: "echo",
    LoadModelJobType.name: "llm",
    UnloadModelJobType.name: "llm",
    LoadVoiceJobType.name: "tts",
    TtsJobType.name: "tts",
    UnloadVoiceJobType.name: "tts",
    RvcJobType.name: "rvc",
}


def build_registry(
    config: "Config",
    backend: "Backend",
    residency: Residency | None = None,
    leases: "Leases | None" = None,
) -> dict[str, JobType]:
    registry: dict[str, JobType] = {}
    holder = residency if residency is not None else Residency(config)
    if config.enable_echo:
        registry[EchoJobType.name] = EchoJobType()
    if config.enable_llm:
        registry[LoadModelJobType.name] = LoadModelJobType(
            config, backend, holder, leases
        )
        registry[UnloadModelJobType.name] = UnloadModelJobType(config, backend, holder)
    if config.enable_tts:
        registry[LoadVoiceJobType.name] = LoadVoiceJobType(
            config, backend, holder, leases
        )
        registry[UnloadVoiceJobType.name] = UnloadVoiceJobType(config, backend, holder)
        registry[TtsJobType.name] = TtsJobType(config, backend, holder)
    if config.enable_asr:
        registry[AsrJobType.name] = AsrJobType(config, backend, holder.owned_pids)
    if config.enable_align:
        registry[AlignJobType.name] = AlignJobType(config, backend, holder)
        registry[UnloadAlignerJobType.name] = UnloadAlignerJobType(
            config, backend, holder
        )
        registry[AlignLongformJobType.name] = AlignLongformJobType(config, backend)
    if config.enable_rvc:
        registry[RvcJobType.name] = RvcJobType(config, backend, holder.owned_pids)
    if config.enable_denoise:
        registry[DenoiseJobType.name] = DenoiseJobType(config, backend, holder)
        registry[UnloadDenoiserJobType.name] = UnloadDenoiserJobType(
            config, backend, holder
        )
    _assert_every_type_implements_the_protocol(registry)
    return registry


_JOB_TYPE_MEMBERS: tuple[str, ...] = tuple(
    sorted(
        {
            name
            for name in list(vars(JobType)) + list(getattr(JobType, "__annotations__", {}))
            if not name.startswith("_")
            and name != "name"
            and name not in OPTIONAL_JOB_TYPE_MEMBERS
        }
    )
)


def _assert_every_type_implements_the_protocol(registry: dict[str, JobType]) -> None:
    for job_type, plugin in sorted(registry.items()):
        missing = [m for m in _JOB_TYPE_MEMBERS if not hasattr(plugin, m)]
        if missing:
            raise TypeError(
                f"job type {job_type!r} ({type(plugin).__name__}) does not "
                f"implement JobType: missing {sorted(missing)}. Every member of "
                "the protocol is called by the queue or the API; a type that is "
                "missing one fails somewhere far from here."
            )


_UNCOVERED = sorted(set(ALL_JOB_TYPES.values()) - {entry.job_type for entry in CLASSES})
if _UNCOVERED:
    raise TypeError(
        f"job type(s) {_UNCOVERED} have no capability class in "
        "crucible/capability.py, so nothing decides whether this host can run "
        "them and `job_type_disabled` would have no number to name"
    )


CAPABILITIES: frozenset[str] = frozenset(ALL_JOB_TYPES.values())


def disabled_error(name: str, config: Any) -> ApiError:
    if name in ALL_JOB_TYPES:
        job_type, capability = name, ALL_JOB_TYPES[name]
    elif name in CAPABILITIES:
        job_type, capability = name, name
    else:
        raise KeyError(
            f"{name!r} is neither a job type nor a capability; the job types are "
            f"{sorted(ALL_JOB_TYPES)} and the capabilities {sorted(CAPABILITIES)}"
        )
    flag = f"enable_{capability}"
    if getattr(config, flag, False):
        return ApiError(
            400,
            "job_type_disabled",
            f"job type {job_type!r} is installed and turned on in config.toml "
            f"([jobs] {flag} = true), but this running server has not taken it "
            "up. It does that by itself on the first request after config.toml "
            "changes, so this means that failed; the server's log says why.",
            {"job_type": job_type, "reason": "not_taken_up", "flag_on": True},
        )
    record = getattr(config, "capability", None)
    if record is None:
        return ApiError(
            400,
            "job_type_disabled",
            f"job type {job_type!r} is not enabled on this server, and no "
            f"capability selection has been recorded here — nothing knows whether "
            f"this host can hold the models it needs. Run `crucible capability` to "
            f"find out before turning [jobs] {flag} on; on a card that is too "
            f"small, turning it on buys an OOM instead of a server.",
            {"job_type": job_type, "capability_recorded": False, "reason": "undecided"},
        )

    rows = [record.row(entry.name) for entry in classes_for_job_type(capability)]
    known = [row for row in rows if row is not None]
    if not known:
        return ApiError(
            400,
            "job_type_disabled",
            f"job type {job_type!r} is not enabled on this server, and the "
            f"capability record in config.toml has no row for it — it was written "
            f"by an older build. Re-run `crucible capability --write`.",
            {"job_type": job_type, "capability_recorded": False, "reason": "undecided"},
        )

    fitting = [row for row in known if row.enabled]
    if fitting:
        installer = INSTALLER_FOR.get(capability, capability)
        install = {"type": "install", "job_type": installer}
        needs_engine = (
            " (with a narrator_engine: the tts env is built per engine)"
            if installer == "tts"
            else ""
        )
        return ApiError(
            400,
            "job_type_disabled",
            f"job type {job_type!r} is not installed on this server, but this "
            f"host can hold it: "
            + "; ".join(f"{row.capability} — {row.reason}" for row in fitting)
            + f". Installing {installer!r} is one request, POST /v1/tasks "
            f"{json.dumps(install)}{needs_engine}, which the Install button on "
            "this server's operator page sends; the server takes the type up "
            "when it finishes.",
            {
                "job_type": job_type,
                "capability_recorded": True,
                "fits": [row.capability for row in fitting],
                "reason": "not_installed",
                "install": install,
            },
        )

    return ApiError(
        400,
        "job_type_disabled",
        f"{job_type!r} is disabled on this server: "
        + "; ".join(f"{row.capability} — {row.reason}" for row in known)
        + f". Turning [jobs] {flag} on would not change any of those "
        "numbers; it would only move the failure to the first request.",
        {
            "job_type": job_type,
            "capability_recorded": True,
            "fits": [],
            "reason": "cannot_hold",
            "shortfall_bytes": {
                row.capability: row.shortfall_bytes for row in known
            },
        },
    )


def resolve(registry: dict[str, JobType], job_type: str, config: Any) -> JobType:
    plugin = registry.get(job_type)
    if plugin is not None:
        return plugin
    if job_type in ALL_JOB_TYPES:
        raise disabled_error(job_type, config)
    raise ApiError(
        400,
        "unknown_job_type",
        f"unknown job type {job_type!r}; this server offers "
        f"{sorted(registry) if registry else 'no job types'}",
    )


def resolve_model(plugin: JobType, model: str | None) -> str | None:
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
            {"model": model, "offered": ids},
        )
    return model


__all__ = [
    "ALL_JOB_TYPES",
    "AlignJobType",
    "AsrJobType",
    "DenoiseJobType",
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
    "UnloadDenoiserJobType",
    "UnloadModelJobType",
    "UnloadVoiceJobType",
    "build_registry",
    "disabled_error",
    "model_rows",
    "resolve",
    "resolve_model",
    "validate_member_name",
    "voice_rows",
]
