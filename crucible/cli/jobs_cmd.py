from __future__ import annotations

import argparse
import json

from .. import capabilityclasses
from ..backend import Backend
from ..config import CAPABILITY_FLAGS, Config, rewrite_config
from ..errors import ConfigError
from ..jobenv import INSTALLER_FOR
from ..narratorengines import NARRATOR_ENGINE_SAMPLING
from . import common
from .common import EXIT_OK, _fail

JOB_TYPE_NAMES: tuple[str, ...] = tuple(
    flag.removeprefix("enable_") for flag in CAPABILITY_FLAGS
)

TAKEN_UP = (
    "a running server takes this up by itself on its next request; nothing "
    "needs restarting"
)


def _flag(job_type: str) -> str:
    return f"enable_{job_type}"


def _verdict(config: Config, job_type: str) -> tuple[bool | None, str]:
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
    from ..tasks.validate import env_installed

    installer = INSTALLER_FOR.get(job_type)
    if installer is None:
        return None
    if installer == "tts":
        return any(
            env_installed(config, backend, "tts", engine)
            for engine in sorted(NARRATOR_ENGINE_SAMPLING)
        )
    return env_installed(config, backend, installer, None)


def _install_words(job_type: str) -> str:
    installer = INSTALLER_FOR[job_type]
    if installer == "tts":
        return " or ".join(
            f"`crucible install tts --narrator-engine {engine}`"
            for engine in sorted(NARRATOR_ENGINE_SAMPLING)
        )
    return f"`crucible install {installer}`"


def cmd_jobs_list(args: argparse.Namespace) -> int:
    config, backend = common.here()
    rows = []
    for job_type in JOB_TYPE_NAMES:
        fits, why = _verdict(config, job_type)
        rows.append(
            {
                "job_type": job_type,
                "enabled": getattr(config, _flag(job_type)),
                "fits": fits,
                "env_built": env_built(config, backend, job_type),
                "why": why,
            }
        )
    if args.json:
        print(json.dumps({"config": str(config.path), "job_types": rows}, indent=2))
        return EXIT_OK
    print(f"config: {config.path}")
    for row in rows:
        mark = "on " if row["enabled"] else "off"
        fits = {True: "fits", False: "does NOT fit", None: "undecided"}[row["fits"]]
        env = {True: "env built", False: "env NOT built", None: "no env"}[row["env_built"]]
        print(f"  {row['job_type']:<8} {mark}  {fits}; {env}")
    print(
        "`crucible jobs enable <type>` and `crucible jobs disable <type>` change "
        "a flag and keep the token and everything else"
    )
    return EXIT_OK


def _write(config: Config, job_type: str, on: bool) -> int:
    try:
        written = rewrite_config(config, flags={_flag(job_type): on})
    except ConfigError as exc:
        return _fail(str(exc))
    print(f"[jobs] {_flag(job_type)} = {'true' if on else 'false'} — {written}")
    return EXIT_OK


def cmd_jobs_enable(args: argparse.Namespace) -> int:
    config, backend = common.here()
    job_type = args.job_type
    fits, why = _verdict(config, job_type)
    if fits is None:
        return _fail(
            f"job_type_undecided: {why}, so nothing knows whether this host can "
            f"hold {job_type!r}. Run `crucible capability --write`, which measures "
            "this card against the models and records it, then enable it again"
        )
    if not fits:
        return _fail(
            f"job_type_cannot_hold: {job_type!r} does not fit this host: {why}. "
            f"Turning [jobs] {_flag(job_type)} on would not change any of those "
            "numbers; it would only move the failure to the first job. Nothing "
            "was written"
        )
    if env_built(config, backend, job_type) is False:
        return _fail(
            f"env_not_built: {job_type!r} fits this host, but the env it runs in is "
            f"not built here, and a flag that is on says this server offers the "
            f"type. {_install_words(job_type)} builds the env and turns "
            f"[jobs] {_flag(job_type)} on in one step. Nothing was written"
        )
    if getattr(config, _flag(job_type)):
        print(f"[jobs] {_flag(job_type)} is already true in {config.path}")
        return EXIT_OK
    refused = _write(config, job_type, True)
    if refused:
        return refused
    print(TAKEN_UP)
    return EXIT_OK


def cmd_jobs_disable(args: argparse.Namespace) -> int:
    config, _backend = common.here()
    job_type = args.job_type
    if not getattr(config, _flag(job_type)):
        print(f"[jobs] {_flag(job_type)} is already false in {config.path}")
        return EXIT_OK
    refused = _write(config, job_type, False)
    if refused:
        return refused
    print(
        f"{TAKEN_UP}. Its env and models stay installed; `crucible jobs enable "
        f"{job_type}` turns it back on"
    )
    return EXIT_OK


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    jobs = subparsers.add_parser(
        "jobs",
        help="turn a job type on or off in config.toml, keeping the token",
        description=(
            "Each job type has a [jobs] enable_<type> flag in config.toml. These "
            "verbs change one flag and write everything else back as it was: the "
            "token every app holds, the capability record, routes and the tables "
            "this build does not own. `crucible install <type>` builds a type's "
            "env and turns it on in one step; `enable` is for a type whose env is "
            "already here, and `disable` turns one off. On a Windows PC, run them "
            "in the Linux engine with `crucible guest jobs ...`."
        ),
    )
    commands = jobs.add_subparsers(dest="jobs_command", required=True)

    listing = commands.add_parser(
        "list", help="every job type: on or off, whether it fits, whether its env is built"
    )
    listing.add_argument("--json", action="store_true", help="machine-readable")
    listing.set_defaults(func=cmd_jobs_list)

    enable = commands.add_parser(
        "enable",
        help=(
            "turn on a job type whose env is built. Refused when this host's "
            "recorded capability says it does not fit, or its env is not built"
        ),
    )
    enable.add_argument("job_type", choices=JOB_TYPE_NAMES)
    enable.set_defaults(func=cmd_jobs_enable)

    disable = commands.add_parser(
        "disable", help="turn a job type off; its env and models stay installed"
    )
    disable.add_argument("job_type", choices=JOB_TYPE_NAMES)
    disable.set_defaults(func=cmd_jobs_disable)
