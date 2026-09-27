from __future__ import annotations

import argparse

from .. import VERSION
from ..config import CRUCIBLE_HOME_ENV
from . import (
    api_cmd,
    capability,
    common,
    doctor,
    init,
    install,
    orchestrator,
    pair,
    serve,
    service_cmd,
    token,
    uninstall_cmd,
    voices,
    weights,
)
from .capability import _capability_step, write_capability as _write_capability
from .common import EXIT_REFUSED, _backend_mismatch
from .init import carried_from
from .install import INSTALLABLE_JOB_TYPES, INSTALLER_FOR, SMOKE_IMPORT, _smoke_import
from .serve import cmd_serve
from .token import (
    PAIRING_NOT_PRINTED,
    _pairing_lines,
    _sync_pairing_file,
    _write_pairing_file,
)
from .uninstall_cmd import cmd_uninstall
from .weights import cmd_remove

__all__ = [
    "EXIT_REFUSED",
    "INSTALLABLE_JOB_TYPES",
    "INSTALLER_FOR",
    "PAIRING_NOT_PRINTED",
    "SMOKE_IMPORT",
    "_backend_mismatch",
    "_capability_step",
    "_pairing_lines",
    "_smoke_import",
    "_sync_pairing_file",
    "_write_capability",
    "_write_pairing_file",
    "build_parser",
    "carried_from",
    "cmd_remove",
    "cmd_serve",
    "cmd_uninstall",
    "common",
    "main",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crucible",
        description="One inference server, many client apps.",
        epilog=(
            f"State lives under ${CRUCIBLE_HOME_ENV} (default ~/.crucible). "
            "Crucible runs on Linux with an NVIDIA card and on Apple Silicon macOS."
        ),
    )
    parser.add_argument("--version", action="version", version=f"crucible {VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    from ..local import add_parser as add_local_parser
    add_local_parser(subparsers)

    init.add_parser(subparsers)
    install.add_parser(subparsers)
    capability.add_parser(subparsers)
    weights.add_model_parsers(subparsers)
    voices.add_parser(subparsers)
    weights.add_rvc_denoise_parsers(subparsers)
    orchestrator.add_parser(subparsers)
    serve.add_parser(subparsers)
    service_cmd.add_parser(subparsers)
    uninstall_cmd.add_parser(subparsers)
    doctor.add_parser(subparsers)
    token.add_parser(subparsers)

    from ..sharing import add_parser as add_sharing_parser
    add_sharing_parser(subparsers)
    from ..lan import add_parser as add_lan_parser
    add_lan_parser(subparsers)

    api_cmd.add_parser(subparsers)
    pair.add_pair_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except common.Refusal as exc:
        return common._fail(str(exc))
