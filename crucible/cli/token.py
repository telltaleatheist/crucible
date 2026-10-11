from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import pairing
from ..config import (
    AUTH_MODES,
    AUTH_NONE,
    AUTH_TOKEN,
    DEFAULT_HOST,
    Config,
    check_auth_mode,
    config_mode,
    rewrite_config,
)
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
    name: str, host: str, port: int, token: str | None, advertise: tuple[str, ...] = ()
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
    if config.auth == AUTH_NONE:
        print(
            f"crucible: this server asks for no token ([auth] mode = \"{AUTH_NONE}\" in "
            f"{config.path}), so a client needs only its address"
            + ("; the pairing lines below carry no #<token>" if args.url else "")
            + (". The token printed is the stored one `crucible auth token` turns back on"
               if args.show else ""),
            file=sys.stderr,
        )
    if args.show:
        print(config.token)
    if args.url:
        result = _pairing_lines(
            config.name, config.host, config.port, pairing.handed_out_token(config),
            config.advertise + config.tailscale_advertise + config.lan_advertise
        )
        if isinstance(result, str):
            return _fail(result)
        for line in result:
            print(line)
    return EXIT_OK


AUTH_SENTENCES = {
    AUTH_TOKEN: "every route but GET /v1/ping and the operator page asks for "
    "`Authorization: Bearer <token>`",
    AUTH_NONE: "no route asks for a token, like Ollama; a bearer sent anyway is ignored, "
    "right or wrong, so a client that stored one keeps working",
}

AUTH_TAKEN_UP = (
    "a running server follows its config file and applies this from its next request; "
    "nothing needs restarting"
)


def cmd_auth(args: argparse.Namespace) -> int:
    try:
        config = common.load_config()
    except ConfigError as exc:
        return _fail(str(exc))
    if args.mode is None:
        print(f"[auth] mode = \"{config.auth}\" ({config.path}): {AUTH_SENTENCES[config.auth]}")
        return EXIT_OK
    try:
        mode = check_auth_mode(args.mode, "crucible auth")
    except ConfigError as exc:
        return _fail(str(exc))
    if mode == config.auth:
        print(f"[auth] mode is already \"{mode}\" in {config.path}: {AUTH_SENTENCES[mode]}")
        return EXIT_OK
    try:
        written = rewrite_config(config, auth=mode)
    except ConfigError as exc:
        return _fail(str(exc))
    print(f"[auth] mode = \"{mode}\" — {written}")
    print(AUTH_SENTENCES[mode])
    if mode == AUTH_TOKEN:
        print(
            "the token is the one this server already had (`crucible token --show`); "
            "a client paired before it was turned off still holds it"
        )
    print(AUTH_TAKEN_UP)
    return EXIT_OK


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    auth = subparsers.add_parser(
        "auth",
        help="say, or set, whether callers need this server's token ([auth] mode)",
        description=(
            "With no argument, print [auth] mode. `crucible auth none` lets every "
            "route answer without a token, like Ollama (a bearer sent anyway is "
            "ignored); `crucible auth token` asks for `Authorization: Bearer <token>` "
            "again, with the token this server already has. The running server picks "
            "the change up from its config file on its next request."
        ),
    )
    auth.add_argument("mode", nargs="?", choices=list(AUTH_MODES), default=None)
    auth.set_defaults(func=cmd_auth)

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
