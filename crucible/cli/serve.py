from __future__ import annotations

import argparse
import sys

from .. import API_VERSION, KEEP_ALIVE_SECONDS, VERSION, pairing
from ..errors import ConfigError, NoViableBackend
from . import common, token
from .common import EXIT_OK, _backend_mismatch, _fail


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        config = common.load_config()
    except ConfigError as exc:
        return _fail(str(exc))
    try:
        backend = common.detect_backend()
    except NoViableBackend as exc:
        return _fail(f"no viable backend: {exc.reason}")
    if backend.kind != config.backend_kind:
        return _fail(
            _backend_mismatch(config.backend_kind, backend)
            + f" ({config.path}); re-run `crucible init --force` on this host"
        )

    host = args.host if args.host is not None else config.host
    port = args.port if args.port is not None else config.port

    # THE PAIRING FILE IS WRITTEN AT STARTUP TOO (PHASE15-HOST.md 3.6, amended
    # 2026-09-14). `init` and `service install` wrote it and nothing else did,
    # so a server that EXISTED before this phase had none and an app on its own
    # machine was told there was no engine there — measured on the Mac after
    # its upgrade. The line is written when it is absent OR when it does not
    # match what this config says, because a rotated token, a renamed server
    # or a moved port each leave a file that is worse than no file: it points
    # an app at a door with the wrong key.
    #
    # The line is the CONFIG's, not this run's `--host`/`--port` overrides:
    # 3.6's file answers "an app on THIS machine wants in", and a developer
    # running `crucible serve --port 7999` for an afternoon must not repoint
    # every app on the box at a server that is about to stop.
    try:
        token._sync_pairing_file(config)
    except pairing.PairingFileError as exc:
        # NOT fatal, and NOT silent. The server is the thing being started and
        # it works without this file; what the file changes is whether an app
        # has to be told a token by hand. Refusing to serve over it would be
        # the tail wagging the dog, and swallowing it would be a machine where
        # connect quietly stopped working.
        print(f"crucible: pairing file NOT written: {exc}", file=sys.stderr)

    from ..api import create_app  # imported here so `init`/`token` stay light

    app = create_app(config, backend)
    # WHERE IT IS REALLY LISTENING, not where the file says. `--host` and
    # `--port` override the config for this run, and `GET /v1/setup` builds its
    # pairing lines from the bind address — so a server started
    # `crucible serve --host 0.0.0.0` on a config that says `127.0.0.1` must
    # hand out its interface addresses, not a loopback nobody else can dial.
    app.state.bind_host = host
    app.state.bind_port = port

    print(f"crucible {VERSION} (api {API_VERSION}) — {config.name}")
    print(f"backend: {backend.kind} ({backend.gpu.name})")
    print(f"listening on http://{host}:{port}/v1")
    if host in ("127.0.0.1", "localhost", "::1"):
        print(
            "bound to loopback: only this host can reach it. To serve the tailnet, "
            "pass --host 0.0.0.0 (or the tailnet IP); the bearer token is the lock."
        )
    else:
        print("bound beyond loopback: the bearer token is the only lock.")

    import uvicorn

    if getattr(args, "controller_stdin", False):
        from ..host.child_lifecycle import run_owned_server
        run_owned_server(app, host=host, port=port, log_level=args.log_level)
    else:
        # `timeout_keep_alive` IS STATED, and the default is what broke.
        # uvicorn holds an idle connection 5 s; Node's undici keeps a pooled
        # one about 4 s, so a client's next request lands on a socket this
        # server is closing and reads ECONNRESET while the server is fine.
        # Four times: align's first `GET /v1/info` after a render's last
        # artifact fetch, Sep 18 and Sep 19 against the PC and 2026-09-20 00:57
        # against the Mac on 127.0.0.1. See `KEEP_ALIVE_SECONDS`.
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=args.log_level,
            timeout_keep_alive=KEEP_ALIVE_SECONDS,
        )
    return EXIT_OK


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    serve = subparsers.add_parser("serve", help="run the API in the foreground")
    serve.add_argument("--host", default=None, help="bind host (default from config)")
    serve.add_argument("--port", type=int, default=None, help="bind port (default from config)")
    serve.add_argument("--log-level", default="info", help="uvicorn log level")
    serve.add_argument("--controller-stdin", action="store_true", help=argparse.SUPPRESS)
    serve.set_defaults(func=cmd_serve)
