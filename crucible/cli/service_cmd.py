from __future__ import annotations

import argparse
import getpass
import json
import sys

from .. import service
from ..backend import Backend
from ..config import Config
from ..errors import ConfigError, NoViableBackend
from . import common
from .common import EXIT_OK, EXIT_REFUSED, _backend_mismatch, _fail
from .token import PAIRING_NOT_PRINTED, _pairing_permission, _write_pairing_file


def _service_context() -> tuple[Config, Backend, str] | int:
    """Config, backend and this host's service mechanism, or a printed refusal.

    The backend is DETECTED and compared against the config, exactly as
    `install` and `capability` do, rather than read off the config alone. A
    service is a promise that `crucible serve` will keep running on this host,
    and `serve` itself refuses when the detected backend and the recorded one
    disagree — so installing a unit in that state would install a unit that
    cannot start.
    """
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
            + f" ({config.path}); re-run `crucible init --force`"
        )
    try:
        mechanism = service.mechanism_for(backend.kind)
    except service.ServiceError as exc:
        return _fail(str(exc))
    return config, backend, mechanism


def cmd_service_install(args: argparse.Namespace) -> int:
    """`crucible service install` — PHASE5-APPS.md 6.0, PHASE11-SERVICE.md.

    Host and port come from `config.toml` AT INSTALL TIME and are baked into the
    unit's `ExecStart`, which means a config edited afterwards is not what the
    service serves until this is re-run. That is stated in the phase doc and
    printed here, because the alternative — a unit that re-reads the config —
    is a unit whose behaviour changes without anybody installing anything.
    """
    resolved = _service_context()
    if isinstance(resolved, int):
        return resolved
    config, _backend, mechanism = resolved
    try:
        lines = service.install(
            mechanism,
            home=service.user_home(),
            server_name=config.name,
            executable=sys.executable,
            crucible_home=config.home,
            host=config.host,
            port=config.port,
            runner=service.subprocess_runner,
        )
    except service.ServiceError as exc:
        return _fail(str(exc))
    print(f"mechanism: {mechanism}")
    for line in lines:
        print(line)
    print(
        f"serving:  http://{config.host}:{config.port}/v1 — read from "
        f"{config.path} now and written into the definition. Change either and "
        "re-run `crucible service install`."
    )
    # Section 3.6 again: a server installed as a service is the one an app is
    # most likely to meet without a person present, so the file it reads is
    # written here too — with the SAME token, so nothing that had paired is
    # unpaired by installing a unit.
    paired = _write_pairing_file(
        config.home, name=config.name, port=config.port, token=config.token
    )
    print(f"pairing:  {paired} ({_pairing_permission(paired)})")
    print(PAIRING_NOT_PRINTED)
    from ..local import publish_installation
    publish_installation(config.home)
    return EXIT_OK


def cmd_service_uninstall(args: argparse.Namespace) -> int:
    resolved = _service_context()
    if isinstance(resolved, int):
        return resolved
    _config, _backend, mechanism = resolved
    try:
        lines = service.uninstall(
            mechanism, home=service.user_home(), runner=service.subprocess_runner
        )
    except service.ServiceError as exc:
        return _fail(str(exc))
    for line in lines:
        print(line)
    return EXIT_OK


def cmd_service_start(args: argparse.Namespace) -> int:
    resolved = _service_context()
    if isinstance(resolved, int):
        return resolved
    _config, _backend, mechanism = resolved
    try:
        lines = service.start(
            mechanism, home=service.user_home(), runner=service.subprocess_runner
        )
    except service.ServiceError as exc:
        return _fail(str(exc))
    for line in lines:
        print(line)
    return EXIT_OK


def cmd_service_stop(args: argparse.Namespace) -> int:
    resolved = _service_context()
    if isinstance(resolved, int):
        return resolved
    _config, _backend, mechanism = resolved
    try:
        lines = service.stop(
            mechanism, home=service.user_home(), runner=service.subprocess_runner
        )
    except service.ServiceError as exc:
        return _fail(str(exc))
    for line in lines:
        print(line)
    return EXIT_OK


def cmd_service_status(args: argparse.Namespace) -> int:
    """Running or not, with the pid and the definition's path.

    **Exit 0 only when it is running**, so a script can gate on it the way it
    gates on `crucible doctor`. A service that is installed and stopped is a
    server nothing can reach, and reporting that as success would make this verb
    useless to the only thing that would automate it.
    """
    resolved = _service_context()
    if isinstance(resolved, int):
        return resolved
    _config, _backend, mechanism = resolved
    try:
        state = service.status(
            mechanism, service.user_home(), runner=service.subprocess_runner
        )
    except service.ServiceError as exc:
        return _fail(str(exc))
    if args.json:
        print(json.dumps(state.to_dict(), indent=2))
        return EXIT_OK if state.running is True else EXIT_REFUSED
    print(f"mechanism:  {state.mechanism}")
    print(
        f"definition: {state.definition} "
        f"({'present' if state.installed else 'NOT THERE'})"
    )
    # Three answers, because there are three. `None` is "no manager could be
    # asked", and reporting that as NO would be this command inventing a fact.
    if state.running is True:
        runs = "yes"
    elif state.running is False:
        runs = "NO"
    else:
        runs = "UNKNOWN - no manager could be asked"
    print(f"running:    {runs}")
    print(f"pid:        {state.pid if state.pid is not None else '-'}")
    print(f"detail:     {state.detail}")
    if state.mechanism == service.SYSTEMD:
        # REPORTED, never assumed: `loginctl enable-linger` is the operator's,
        # and without it this server dies with the session that installed it.
        if state.linger is True:
            linger = "on — this server survives a logout and starts at boot"
        elif state.linger is False:
            linger = (
                "OFF — this server stops when your last session ends. "
                f"`sudo loginctl enable-linger {getpass.getuser()}` grants it"
            )
        else:
            linger = "UNKNOWN — loginctl could not be asked"
        print(f"linger:     {linger}")
    return EXIT_OK if state.running is True else EXIT_REFUSED


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    service_parser = subparsers.add_parser(
        "service",
        help="the machine service that runs `crucible serve` (PHASE11-SERVICE.md)",
        description=(
            "A local Crucible is a service and no app owns it (Owen, 2026-09-13; "
            "PHASE5-APPS.md section 6.0). On cuda-linux that is a systemd USER "
            "unit, on mlx-darwin a launchd agent. Every verb is idempotent."
        ),
    )
    service_commands = service_parser.add_subparsers(
        dest="service_command", required=True
    )

    service_install = service_commands.add_parser(
        "install",
        help="write the unit or plist for this host, enable it and start it",
    )
    service_install.set_defaults(func=cmd_service_install)

    service_uninstall = service_commands.add_parser(
        "uninstall", help="stop the service, forget it, and remove its definition"
    )
    service_uninstall.set_defaults(func=cmd_service_uninstall)

    service_start = service_commands.add_parser(
        "start", help="make sure the installed service is running"
    )
    service_start.set_defaults(func=cmd_service_start)

    service_stop = service_commands.add_parser(
        "stop", help="stop the service without forgetting it"
    )
    service_stop.set_defaults(func=cmd_service_stop)

    service_status = service_commands.add_parser(
        "status",
        help="running or not, with the pid and the unit/plist path; exit 0 only "
        "when it is running",
    )
    service_status.add_argument(
        "--json", action="store_true", help="machine-readable"
    )
    service_status.set_defaults(func=cmd_service_status)
