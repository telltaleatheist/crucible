from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import pairing
from ..config import Config, DEFAULT_HOST, config_mode
from ..errors import ConfigError
from ..interfaces import InterfaceError
from . import common
from .common import EXIT_OK, _fail


def _write_pairing_file(home: Path, *, name: str, port: int, token: str) -> Path:
    """`<CRUCIBLE_HOME>/pairing`, through the ONE writer (`crucible/pairing.py`).

    PHASE15-HOST.md section 3.6. The file holds the LOOPBACK line whatever the
    server is bound to — it answers *"an app on THIS machine wants in"*, and
    the answer to that is never a LAN address — so this is where (name, port,
    token) becomes that line; `pairing.write_pairing_file` owns everything
    after it, including the Windows ACL.
    """
    return pairing.write_pairing_file(
        home, pairing.pairing_line(name, f"http://{DEFAULT_HOST}:{port}", token)
    )


def _sync_pairing_file(config: Config) -> None:
    """Write `<home>/pairing` when it is absent or does not match the config.

    PHASE15-HOST.md 3.6, as amended: `crucible serve` is the third writer,
    and it is the one that covers a server that already existed. Comparison
    is on the LINE, which is exactly the four facts an app needs — name,
    host, port, token — so there is no second notion of "matches" to keep in
    step with the writer.
    """
    wanted = pairing.pairing_line(
        config.name, f"http://{DEFAULT_HOST}:{config.port}", config.token
    )
    if pairing.read_pairing_file(config.home) == wanted:
        return
    written = pairing.write_pairing_file(config.home, wanted)
    print(f"pairing:  {written} ({_pairing_permission(written)})")


def _pairing_permission(path: Path) -> str:
    """What restricts the file, said in the platform's own vocabulary.

    A Windows file has no mode, and printing `config_mode`'s answer there
    would report a number the OS does not enforce.
    """
    if sys.platform == "win32":
        return "ACL: this user only"
    return f"mode {config_mode(path)}"


def _pairing_lines(
    name: str, host: str, port: int, token: str, advertise: tuple[str, ...] = ()
) -> list[str] | str:
    """The lines, or the sentence saying why there are none.

    PHASE13-OPERATOR.md section 3.1. A refusal is returned rather than raised
    because `token --url` has nothing else to print and exits 1. (`init` and
    `service install` called this too until #33; they no longer print the
    line at all — see `PAIRING_NOT_PRINTED`.)

    **The loopback line comes first, always** (PHASE15-HOST.md section 3.6).
    It is what `<CRUCIBLE_HOME>/pairing` holds, and *"`crucible token --url`
    prints the same"* is only true if it is printed. On a `127.0.0.1` bind it
    IS `reachable_urls`' one entry and is printed once; on a wildcard bind
    `reachable_urls` has no loopback entry at all, and without this an app on
    the server's own machine would be handed whichever interface the OS listed
    first.
    """
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


#: What `init` and `service install` say about the pairing line, INSTEAD of
#: printing it (fresh-install #33, 2026-09-26). They used to print the whole
#: `crucible://` line, token included, and both run inside every install, so
#: anything that logged an install captured the secret; the Windows move
#: already had to redact its own stream. The line is still one string an app
#: pastes (Owen, 2026-09-14: nobody types a token twice), and it is still in
#: the pairing file an app on this machine reads by itself. Printing it is
#: now the job of the one verb whose name says it prints a secret.
PAIRING_NOT_PRINTED = (
    "pairing:  the line an app pastes is in that file and is not printed here; "
    "`crucible token --url` prints it"
)


def cmd_token(args: argparse.Namespace) -> int:
    """`crucible token --show` prints the secret; `--url` prints the whole door.

    `--url` needs no `--show`, and that is not laxity: the flag's name says it
    prints a URL, and the pairing line's whole purpose is to be handed to an
    app. Requiring two flags to print one string would be a ceremony that
    protects nothing — the token is already behind a file mode 0600 and a
    terminal somebody is sitting at.
    """
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
