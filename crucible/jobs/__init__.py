"""The job-type registry.

`ALL_JOB_TYPES` is the vocabulary this build knows. `build_registry(config)` is the
subset this server is configured to offer. Asking for a type that exists but is not
enabled is refused differently from asking for one that does not exist — the client
is told which of the two it is.
"""

from __future__ import annotations

from typing import Any, Callable

from ..errors import ApiError
from .base import (
    Job,
    JobContext,
    JobType,
    JobTypeStatus,
    ModelDescriptor,
    validate_member_name,
)
from .echo import EchoJobType
from .llm import LoadModelJobType, Residency, UnloadModelJobType, model_rows

#: The vocabulary this build knows, and which config flag turns each one on.
#: `resolve()` tells "that type does not exist" from "it exists but is off".
ALL_JOB_TYPES: dict[str, str] = {
    EchoJobType.name: "echo",
    LoadModelJobType.name: "llm",
    UnloadModelJobType.name: "llm",
}


def build_registry(config: Any, residency: Residency | None = None) -> dict[str, JobType]:
    """Instantiate the job types this config enables.

    `residency` is the server instance's one-resident-model holder. `crucible
    doctor` has no server, so it passes none and gets a fresh (empty) one — which
    is honest: a doctor run cannot see another process's resident model.
    """
    registry: dict[str, JobType] = {}
    if config.enable_echo:
        registry[EchoJobType.name] = EchoJobType()
    if config.enable_llm:
        holder = residency if residency is not None else Residency(config)
        registry[LoadModelJobType.name] = LoadModelJobType(config, holder)
        registry[UnloadModelJobType.name] = UnloadModelJobType(config, holder)
    return registry


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
    "EchoJobType",
    "Job",
    "JobContext",
    "JobType",
    "JobTypeStatus",
    "LoadModelJobType",
    "ModelDescriptor",
    "Residency",
    "UnloadModelJobType",
    "build_registry",
    "model_rows",
    "resolve",
    "resolve_model",
    "validate_member_name",
]
