"""The `crucible` command line.

    crucible init       mint the token, write the config, record the backend
    crucible install    build a job type's env from its recipe, then decide whether the card fits it
    crucible capability what this host can hold, and why; --write records it
    crucible serve      run the API in the foreground
    crucible service    install/start/stop the machine service that runs `serve`
    crucible orchestrator  win32 only: the tray that manages this machine's engine
    crucible models     list and pull model weights
    crucible voices     list and pull voice weights
    crucible doctor     probe the host and every job type; exit 0 only when healthy
    crucible token      print the bearer token (--show) or the pairing line (--url)
    crucible uninstall  install, run backwards; weights kept unless --purge-weights

    crucible api        THE CLIENT HALF: submit jobs, stream tts, chat, read
                        state — over HTTP, against a server that may be this
                        machine's, the one in WSL, or one across the network.
                        Every verb above acts on THIS machine's installation and
                        takes no address; these take --url and --token. See
                        crucible/apiclient.py and docs/API-CLI.md.
    crucible pair       connect to another computer's Crucible by its address
                        and keep its pairing line; then `api --server <name>`

Exit codes: 0 success, 1 refused (named reason on stderr), 2 usage.
"""

from __future__ import annotations

import argparse

from .. import VERSION
from ..config import CRUCIBLE_HOME_ENV
from . import (
    capability,
    common,
    doctor,
    init,
    install,
    orchestrator,
    serve,
    service_cmd,
    token,
    uninstall_cmd,
    voices,
    weights,
)
from .capability import _capability_step, _decide_here, _write_capability
from .common import EXIT_OK, EXIT_REFUSED, EXIT_USAGE, _backend_mismatch
from .init import carried_from, carried_reserve
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
    "EXIT_OK",
    "EXIT_REFUSED",
    "EXIT_USAGE",
    "INSTALLABLE_JOB_TYPES",
    "INSTALLER_FOR",
    "PAIRING_NOT_PRINTED",
    "SMOKE_IMPORT",
    "_backend_mismatch",
    "_capability_step",
    "_decide_here",
    "_pairing_lines",
    "_smoke_import",
    "_sync_pairing_file",
    "_write_capability",
    "_write_pairing_file",
    "build_parser",
    "carried_from",
    "carried_reserve",
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

    from ..apiclient import add_pair_parser, add_parser as add_api_parser
    add_api_parser(subparsers)
    add_pair_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
