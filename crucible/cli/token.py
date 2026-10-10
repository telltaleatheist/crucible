from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import pairing
from ..config import DEFAULT_HOST, Config, config_mode
from ..errors import ConfigError
from ..interfaces import InterfaceError
from . import common
from .common import EXIT_OK, _fail


def _write_pairing_file(home: Path, *, name: str, port: int, token: str) -> Path:
    return pairing.write_pairing_file(
        home, pairing.pairing_line(name, f"http://{DEFAULT_HOST}:{port}", token)
    )


def _sync_pairing_file(config: Config) -> None:
    written = pairing.sync_pairing_file(
        config.home, name=config.name, port=config.port, token=config.token
    )
    if written is not None:
        print(f"pairing:  {written} ({_pairing_permission(written)})")


def _pairing_permission(path: Path) -> str:
    if sys.platform == "win32":
        return "ACL: this user only"
    return f"mode {config_mode(path)}"


def _pairing_lines(
    name: str, host: str, port: int, token: str, advertise: tuple[str, ...] = ()
) -> list[str] | str:
    loopback = pairing.pairing_line(name, f"http://{DEFAULT_HOST}:{port}", token)
    try:
        urls = pairing.reachable_urls(host, port, advertise)
    except InterfaceError as exc:
        return (
            f"this host will not list its own interfaces, so there is no "
            f"pairing line for a wildcard bind: {exc}"
        )
    if not urls:
        return (
            f"bound to {host} and this host has no non-loopback IPv4 address, "
            "so nothing else can reach it yet"
        )
    lines = [loopback]
    for line in pairing.pairing_lines(name, urls, token):
        if line not in lines:
            lines.append(line)
    return lines


PAIRING_NOT_PRINTED = (
    "pairing:  the line an app pastes is in that file and is not printed here; "
    "`crucible token --url` prints it"
)


def cmd_token(args: argparse.Namespace) -> int:
    if not args.show and not args.url:
        return _fail("pass --show to print the bearer token, or --url to print "
                     "the pairing line an app's connect door takes")
    try:
        config = common.load_config()
    except ConfigError as exc:
        return _fail(str(exc))
    if args.show:
        print(config.token)
    if args.url:
        result = _pairing_lines(
            config.name, config.host, config.port, config.token,
            config.advertise + config.tailscale_advertise + config.lan_advertise
        )
        if isinstance(result, str):
            return _fail(result)
        for line in result:
            print(line)
    return EXIT_OK


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    token = subparsers.add_parser(
        "token", help="print the bearer token, or the pairing line an app takes"
    )
    token.add_argument("--show", action="store_true", help="prints the secret")
    token.add_argument(
        "--url",
        action="store_true",
        help=(
            "print the pairing line for each address this server is reachable "
            "on — crucible://<name>@<host>:<port>/#<token>. It carries the "
            "token, which is what the flag name says"
        ),
    )
    token.set_defaults(func=cmd_token)
