"""`[jobs] enable_<type>`: turning a job type on or off, for the CLI and Settings alike.

`crucible jobs enable|disable` and `PUT /v1/settings/jobs/{job_type}` are the same door:
both refuse an enable this host cannot honour, by the same name and sentence, and both
write through rewrite_config, so a flag never costs the token or anything else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import capabilityclasses
from .backend import Backend
from .config import CAPABILITY_FLAGS, Config, rewrite_config
from .errors import ConfigError
from .jobenv import INSTALLER_FOR
from .narratorengines import NARRATOR_ENGINE_SAMPLING

JOB_TYPE_NAMES: tuple[str, ...] = tuple(
    flag.removeprefix("enable_") for flag in CAPABILITY_FLAGS
)

UNDECIDED = "job_type_undecided"
CANNOT_HOLD = "job_type_cannot_hold"
ENV_NOT_BUILT = "env_not_built"


class EnableRefused(ConfigError):
    """An enable this host cannot honour; `code` is the refusal's name."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.sentence = message


def flag(job_type: str) -> str:
    return f"enable_{job_type}"


def require_job_type(job_type: str) -> str:
    if job_type not in JOB_TYPE_NAMES:
        raise ConfigError(
            f"{job_type!r} is not a job type with a [jobs] flag; they are "
            f"{list(JOB_TYPE_NAMES)}"
        )
    return job_type


def verdict(config: Config, job_type: str) -> tuple[bool | None, str]:
    """Whether this host's recorded capability says the type fits, and why.

    None means nothing is recorded for it, so nothing knows whether it fits.
    """
    record = config.capability
    if record is None:
        return None, "no capability selection is recorded in this config"
    rows = [
        record.row(entry.name)
        for entry in capabilityclasses.classes_for_job_type(job_type)
    ]
    known = [row for row in rows if row is not None]
    if not known:
        return None, (
            f"the capability record has no row for {job_type!r}; it was written "
            "by an older build"
        )
    reasons = "; ".join(f"{row.capability} — {row.reason}" for row in known)
    return any(row.enabled for row in known), reasons


def env_built(config: Config, backend: Backend, job_type: str) -> bool | None:
    """Whether the env this job type runs in is built here. None: it has none."""
    from .tasks.validate import env_installed

    installer = INSTALLER_FOR.get(job_type)
    if installer is None:
        return None
    if installer == "tts":
        return any(
            env_installed(config, backend, "tts", engine)
            for engine in sorted(NARRATOR_ENGINE_SAMPLING)
        )
    return env_installed(config, backend, installer, None)


def install_words(job_type: str) -> str:
    installer = INSTALLER_FOR[job_type]
    if installer == "tts":
        return " or ".join(
            f"`crucible install tts --narrator-engine {engine}`"
            for engine in sorted(NARRATOR_ENGINE_SAMPLING)
        )
    return f"`crucible install {installer}`"


def rows(config: Config, backend: Backend) -> list[dict[str, Any]]:
    """Every job type: on or off, whether it fits, whether its env is built, and why."""
    found = []
    for job_type in JOB_TYPE_NAMES:
        fits, why = verdict(config, job_type)
        found.append(
            {
                "job_type": job_type,
                "enabled": getattr(config, flag(job_type)),
                "fits": fits,
                "env_built": env_built(config, backend, job_type),
                "why": why,
            }
        )
    return found


def refuse_enable(config: Config, backend: Backend, job_type: str) -> None:
    """Raise EnableRefused when turning `job_type` on here would be a lie."""
    fits, why = verdict(config, job_type)
    if fits is None:
        raise EnableRefused(
            UNDECIDED,
            f"{why}, so nothing knows whether this host can hold {job_type!r}. Run "
            "`crucible capability --write`, which measures "
            "this card against the models and records it, then enable it again",
        )
    if not fits:
        raise EnableRefused(
            CANNOT_HOLD,
            f"{job_type!r} does not fit this host: {why}. Turning [jobs] "
            f"{flag(job_type)} on would not change any of those numbers; it would only "
            "move the failure to the first job. Nothing was written",
        )
    if env_built(config, backend, job_type) is False:
        raise EnableRefused(
            ENV_NOT_BUILT,
            f"{job_type!r} fits this host, but the env it runs in is not built here, "
            "and a flag that is on says this server offers the type. "
            f"{install_words(job_type)} builds the env and turns [jobs] "
            f"{flag(job_type)} on in one step. Nothing was written",
        )


def set_enabled(config: Config, backend: Backend, job_type: str, on: bool) -> Path | None:
    """Turn `job_type` on or off; the file written, or None when it already was."""
    require_job_type(job_type)
    if on:
        refuse_enable(config, backend, job_type)
    if getattr(config, flag(job_type)) == on:
        return None
    return rewrite_config(config, flags={flag(job_type): on})


__all__ = [
    "CANNOT_HOLD",
    "ENV_NOT_BUILT",
    "EnableRefused",
    "JOB_TYPE_NAMES",
    "UNDECIDED",
    "env_built",
    "flag",
    "install_words",
    "refuse_enable",
    "require_job_type",
    "rows",
    "set_enabled",
    "verdict",
]
