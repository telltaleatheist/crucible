from __future__ import annotations

import argparse
import json
import os
import sys

from .. import service, uninstall
from ..config import crucible_home
from ..errors import CrucibleError
from .common import EXIT_OK, EXIT_REFUSED, _fail


def cmd_uninstall(args: argparse.Namespace) -> int:
    try:
        home = crucible_home()
        built = uninstall.plan(
            home=home,
            platform=sys.platform,
            env=os.environ,
            runner=service.subprocess_runner,
            purge_weights=args.purge_weights,
            wsl_too=args.wsl_too,
        )
    except CrucibleError as exc:
        return _fail(str(exc))

    if not args.dry_run:
        built = uninstall.run(built)

    if args.json:
        print(json.dumps(built.to_dict(), indent=2))
        return EXIT_OK if not built.fatal else EXIT_REFUSED

    print(f"home:      {built.home}")
    print(f"platform:  {built.platform} ({built.mechanism})")
    print(
        "backend:   "
        + (
            built.backend_kind
            if built.backend_kind is not None
            else "unrecorded — this home has no readable config.toml"
        )
    )
    print(
        "mode:      "
        + (
            "DRY RUN — nothing below has been touched"
            if built.dry_run
            else "live"
        )
    )
    print(f"weights:   {'PURGED' if built.purge_weights else 'kept unless named below'}")
    print("")
    for step in built.steps:
        size = f"  [{uninstall.gib(step.bytes)}]" if step.bytes else ""
        mark = {
            uninstall.REMOVE: "remove",
            uninstall.STOP: "stop  ",
            uninstall.KEEP: "keep  ",
        }[step.action]
        if step.refused is not None:
            mark = "SKIP  " if not step.refused.fatal else "FAILED"
        print(f"{mark}  {step.name:<26} {step.target}{size}")
        print(f"          {step.what}")
        if step.refused is not None:
            print(f"          {step.refused.code}: {step.refused.message}")
        for line in step.detail:
            print(f"          {line}")
    kept = built.kept()
    print("")
    if kept["weights_bytes"]:
        print(
            f"kept:      {uninstall.gib(kept['weights_bytes'])} of weights. "
            "`--purge-weights` is what deletes them."
        )
    if not built.dry_run:
        print(f"freed:     {uninstall.gib(built.removed_bytes())}")
    if built.fatal:
        return _fail(
            "uninstall_incomplete: "
            + "; ".join(
                f"{step.name} — {step.refused.code}"
                for step in built.fatal
                if step.refused is not None
            )
            + ". Everything else was removed"
        )
    return EXIT_OK


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    uninstall_parser = subparsers.add_parser(
        "uninstall",
        help="undo an install, in the inverse order; weights are KEPT unless "
        "--purge-weights",
        description=(
            "The exact inverse of `install.sh` / `crucible install`, step by "
            "named step: stop the server, remove the service, remove the job "
            "envs, the pairing file, the config and the working state — and "
            "then keep the weights, which are the expensive part "
            "(docs/internals/host-and-platform.md, \"Uninstall\"), unless --purge-weights says otherwise. "
            "It asks nothing: the flags decide. It removes nothing outside "
            "$CRUCIBLE_HOME and the service entry it wrote, nothing it cannot "
            "name, and never the relocatable interpreter it is running from — "
            "`install.sh --uninstall` removes that after this returns."
        ),
    )
    uninstall_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print every step and touch nothing. The plan is the SAME object "
        "the real run performs, so there is no second description of what "
        "would happen",
    )
    uninstall_parser.add_argument(
        "--purge-weights",
        action="store_true",
        help=(
            "also delete the six subject directories — models, voices, rvc, "
            "rvc-base, denoise-models, engines. Tens of gigabytes, and a "
            "reinstall re-downloads every byte"
        ),
    )
    uninstall_parser.add_argument(
        "--wsl-too",
        action="store_true",
        help=(
            f"win32 only: first run the guest's own `crucible uninstall` inside "
            f"the {uninstall.CRUCIBLE_DISTRO!r} distro, with these same flags. "
            "Refused by name when that distro is not there. The distro itself "
            "is never unregistered — every other distro on the machine is "
            "yours, and so is that decision"
        ),
    )
    uninstall_parser.add_argument(
        "--json", action="store_true", help="machine-readable; the shape an app reads"
    )
    uninstall_parser.set_defaults(func=cmd_uninstall)
