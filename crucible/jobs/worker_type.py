from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .. import accelerator, jobenv, weights
from ..config import Config
from ..errors import ApiError
from .base import JobTypeStatus


def env_or_status(
    config: Config, env_type: str, backend_kind: str, *, missing_note: str = ""
) -> "jobenv.EnvStatus | JobTypeStatus":
    try:
        env = jobenv.env_status(
            config.home, jobenv.worker_env(env_type, backend_kind), backend_kind
        )
    except jobenv.EnvError as exc:
        return JobTypeStatus(ready=False, detail=str(exc))
    if not env.installed:
        return JobTypeStatus(ready=False, detail=env.detail + missing_note)
    return env


def installed_ids(config: Config, manifests: Iterable[Any], backend_kind: str) -> list[str]:
    return [
        manifest.id
        for manifest in manifests
        if manifest.supports(backend_kind)
        and weights.installed(config, manifest, manifest.spec(backend_kind)) is not None
    ]


def require_block(manifest: Any, model_id: str, backend_kind: str, noun: str) -> Any:
    if not manifest.supports(backend_kind):
        raise ApiError(
            400,
            "backend_unsupported",
            f"{noun} {model_id!r} has no {backend_kind} block; "
            f"{manifest.path.name} declares {sorted(manifest.backends)}",
            {
                "model": model_id,
                "backend": backend_kind,
                "declared": sorted(manifest.backends),
            },
        )
    return manifest.spec(backend_kind)


def refuse_if_larger_than_host(backend: Any, model_id: str, need_bytes: int) -> None:
    accelerator.refuse_if_larger_than_host(
        model_id=model_id,
        need_bytes=need_bytes,
        host_total_bytes=backend.gpu.vram_bytes,
        host_name=backend.gpu.name,
    )


def require_worker_python(
    config: Config, env_type: str, backend_kind: str, model_id: str, *, note: str = ""
) -> Path:
    try:
        return jobenv.require_env(
            config.home, jobenv.worker_env(env_type, backend_kind), backend_kind
        )
    except jobenv.EnvError as exc:
        raise ApiError(
            409,
            "env_missing",
            f"cannot run {model_id!r}: {exc}{note}",
            {"model": model_id, "env": str(config.home / "envs" / env_type)},
        ) from None


def require_weights(config: Config, manifest: Any, spec: Any, model_id: str) -> Path:
    try:
        return weights.require_installed(config, manifest, spec).path
    except weights.WeightsError as exc:
        raise ApiError(
            409,
            "model_not_installed",
            str(exc),
            {"model": model_id, "hf_repo": spec.hf_repo, "revision": spec.revision},
        ) from None
