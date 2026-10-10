from __future__ import annotations

import argparse
import json

from .. import jobflags
from ..errors import ConfigError
from . import common
from .common import EXIT_OK, _fail

JOB_TYPE_NAMES = jobflags.JOB_TYPE_NAMES

TAKEN_UP = (
    "a running server takes this up by itself on its next request; nothing "
    "needs restarting"
)


def cmd_jobs_list(args: argparse.Namespace) -> int:
    config, backend = common.here()
    rows = jobflags.rows(config, backend)
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


def cmd_jobs_enable(args: argparse.Namespace) -> int:
    config, backend = common.here()
    job_type = args.job_type
    flag = jobflags.flag(job_type)
    try:
        written = jobflags.set_enabled(config, backend, job_type, True)
    except ConfigError as exc:
        return _fail(str(exc))
    if written is None:
        print(f"[jobs] {flag} is already true in {config.path}")
        return EXIT_OK
    print(f"[jobs] {flag} = true — {written}")
    print(TAKEN_UP)
    return EXIT_OK


def cmd_jobs_disable(args: argparse.Namespace) -> int:
    config, backend = common.here()
    job_type = args.job_type
    flag = jobflags.flag(job_type)
    try:
        written = jobflags.set_enabled(config, backend, job_type, False)
    except ConfigError as exc:
        return _fail(str(exc))
    if written is None:
        print(f"[jobs] {flag} is already false in {config.path}")
        return EXIT_OK
    print(f"[jobs] {flag} = false — {written}")
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
