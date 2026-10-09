from __future__ import annotations

import argparse
import sys

from .. import API_VERSION, VERSION, pairing, reach
from ..config import Config
from . import common, token
from .common import EXIT_OK, _fail


def reach_banner(config: Config, backend_kind: str, host: str, port: int) -> str:
    if not reach.is_loopback_host(host):
        return "bound beyond loopback: the bearer token is the only lock."
    found = reach.for_server(config, place=reach.place_of(backend_kind), host=host, port=port)
    return "bound to loopback: " + " ".join(found.lines())


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        config, backend = common.here()
    except common.Refusal as exc:
        return _fail(str(exc))

    host = args.host if args.host is not None else config.host
    port = args.port if args.port is not None else config.port

    try:
        token._sync_pairing_file(config)
    except pairing.PairingFileError as exc:
        print(f"crucible: pairing file NOT written: {exc}", file=sys.stderr)

    from ..api import create_app

    app = create_app(config, backend)
    app.state.bind_host = host
    app.state.bind_port = port

    print(f"crucible {VERSION} (api {API_VERSION}) — {config.name}")
    print(f"backend: {backend.kind} ({backend.gpu.name})")
    print(f"listening on http://{host}:{port}/v1")
    print(reach_banner(config, backend.kind, host, port))

    if getattr(args, "controller_stdin", False):
        from ..host.child_lifecycle import run_owned_server
        run_owned_server(app, host=host, port=port, log_level=args.log_level)
    else:
        from ..api.serving import server_for
        server_for(app, host=host, port=port, log_level=args.log_level).run()
    return EXIT_OK


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    serve = subparsers.add_parser("serve", help="run the API in the foreground")
    serve.add_argument("--host", default=None, help="bind host (default from config)")
    serve.add_argument("--port", type=int, default=None, help="bind port (default from config)")
    serve.add_argument("--log-level", default="info", help="uvicorn log level")
    serve.add_argument("--controller-stdin", action="store_true", help=argparse.SUPPRESS)
    serve.set_defaults(func=cmd_serve)
