from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Any

from ..client.connection import SERVERS_DIR, saved_pairing_path, server_slug
from ..client.errors import ClientRefusal
from ..client.pair import pair
from ..errors import CrucibleError
from ..pairing import pairing_line, write_pairing_line
from .common import EXIT_OK, EXIT_REFUSED


def _notify(line: str) -> None:
    print(f"crucible: {line}", file=sys.stderr, flush=True)


def cmd_pair(args: argparse.Namespace) -> int:
    try:
        name, url, token = pair(
            args.address,
            client_name=f"crucible on {socket.gethostname()}",
            notify=_notify,
        )
        target = Path(args.save) if args.save else saved_pairing_path(name)
        write_pairing_line(target, pairing_line(name, url, token),
                           private_directory=not args.save)
    except ClientRefusal as exc:
        print(f"crucible: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except (CrucibleError, OSError) as exc:
        print(f"crucible: pair_save_failed: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print(json.dumps({
        "name": name,
        "url": url,
        "pairing_file": str(target),
        "use": (f"crucible api --pairing-file \"{target}\" <verb>" if args.save
                else f"crucible api --server {server_slug(name)} <verb>"),
    }, indent=2), flush=True)
    return EXIT_OK


def add_pair_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "pair",
        help="connect to the Crucible on another computer by its address; no token to copy",
    )
    parser.add_argument("address", help="that computer's address, e.g. 192.168.68.88")
    parser.add_argument(
        "--save", default=None, metavar="PATH",
        help="where to keep the pairing line (default: this machine's Crucible "
             f"home, {SERVERS_DIR}/<name>.pairing, used by `crucible api --server`)",
    )
    parser.set_defaults(func=cmd_pair)
