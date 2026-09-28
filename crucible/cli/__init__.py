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
__all__ = ["build_parser", "main"]


def _run_tray_verb(verb: str) -> None:
    from ..desktop import run_tray_verb

    run_tray_verb(verb)


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
    add_local_parser(subparsers, tray_verbs=_run_tray_verb)

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
