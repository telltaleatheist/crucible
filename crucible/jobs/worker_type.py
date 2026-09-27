"""The refusals and readiness checks every worker job type makes the same way.

`align`, `asr`, `denoise` and `rvc` each run a manifest's model in a worker
env, and each answers `check()` and refuses a submit in `llm`'s order: what no
amount of installing can fix first (`backend_unsupported`, a card too small),
then what an install or a pull would fix (`env_missing`,
`model_not_installed`), then the live accelerator. The codes, messages and
details below are those types' own; each passes the words that differ.
"""

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
    """A worker env's status when it is installed, or the not-ready status
    `check()` returns (`missing_note` appended to the env's own detail)."""
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
    """The ids among `manifests` with a block for this backend and its weights
    pulled."""
    return [
        manifest.id
        for manifest in manifests
        if manifest.supports(backend_kind)
        and weights.installed(config, manifest, manifest.spec(backend_kind)) is not None
    ]


def require_block(manifest: Any, model_id: str, backend_kind: str, noun: str) -> Any:
    """This backend's block of `manifest`, or `400 backend_unsupported`.

    `noun` names the thing in the sentence: "aligner", "ASR model", ...
    """
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
    """`accelerator.refuse_if_larger_than_host` against this backend's card."""
    accelerator.refuse_if_larger_than_host(
        model_id=model_id,
        need_bytes=need_bytes,
        host_total_bytes=backend.gpu.vram_bytes,
        host_name=backend.gpu.name,
    )


def require_worker_python(
    config: Config, env_type: str, backend_kind: str, model_id: str, *, note: str = ""
) -> Path:
    """The worker env's interpreter, or `409 env_missing` by name."""
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
    """The pulled weights' directory, or `409 model_not_installed` by name."""
    try:
        return weights.require_installed(config, manifest, spec).path
    except weights.WeightsError as exc:
        raise ApiError(
            409,
            "model_not_installed",
            str(exc),
            {"model": model_id, "hf_repo": spec.hf_repo, "revision": spec.revision},
        ) from None
