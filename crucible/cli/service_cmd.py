from __future__ import annotations

import argparse
import getpass
import json
import sys
import time

from .. import service
from ..backend import Backend
from ..config import Config
from . import common
from .common import EXIT_OK, EXIT_REFUSED, _fail
from .token import PAIRING_NOT_PRINTED, _pairing_permission, _write_pairing_file


def _service_context() -> tuple[Config, Backend, str] | int:
    config, backend = common.here()
    try:
        mechanism = service.mechanism_for(backend.kind)
    except service.ServiceError as exc:
        return _fail(str(exc))
    return config, backend, mechanism


def cmd_service_install(args: argparse.Namespace) -> int:
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


RESTART_ANSWER_SECONDS = 120.0
RESTART_SAY_EVERY_SECONDS = 15.0


def wait_for_answer(
    config: Config,
    backend: Backend,
    *,
    budget_s: float = RESTART_ANSWER_SECONDS,
    say_every_s: float = RESTART_SAY_EVERY_SECONDS,
    clock=time.monotonic,
    sleep=time.sleep,
) -> float | None:
    """Wait for this config's server to answer /v1/info; the seconds it took, or None.

    A restart is not done when systemd returns (Type=simple returns as soon as the
    process is spawned) but when the server answers, so this waits, saying so."""
    started = clock()
    said = started
    while True:
        if common.server_here(config, backend) is not None:
            return clock() - started
        now = clock()
        if now - started >= budget_s:
            return None
        if now - said >= say_every_s:
            said = now
            print(f"still waiting for it to answer, {now - started:.0f} s", flush=True)
        sleep(1.0)


def cmd_service_restart(args: argparse.Namespace) -> int:
    resolved = _service_context()
    if isinstance(resolved, int):
        return resolved
    config, backend, mechanism = resolved
    try:
        lines = service.restart(
            mechanism, home=service.user_home(), runner=service.subprocess_runner
        )
    except service.ServiceError as exc:
        return _fail(str(exc))
    for line in lines:
        print(line)
    url = common.loopback_url(config)
    print(
        f"waiting up to {RESTART_ANSWER_SECONDS:.0f} s for the server to answer on "
        f"{url} (it loads its job types first)",
        flush=True,
    )
    took = wait_for_answer(config, backend)
    if took is None:
        log = (
            f"`journalctl -u {service.UNIT_NAME}`"
            if mechanism == service.SYSTEMD
            else str(service.serve_log_path(config.home))
        )
        return _fail(
            f"restart_not_answering: the service restarted, and nothing answered on "
            f"{url} within {RESTART_ANSWER_SECONDS:.0f} s. `crucible service status` "
            f"says whether it is running, and {log} why it is not answering"
        )
    print(f"answering on {url} after {took:.0f} s")
    return EXIT_OK


def cmd_service_status(args: argparse.Namespace) -> int:
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
        help="the machine service that runs `crucible serve` (docs/internals/host-and-platform.md, \"Services\")",
        description=(
            "A local Crucible is a service and no app owns it (Owen, 2026-09-13; "
            "docs/internals/host-and-platform.md, \"Standing owner rulings\"). On cuda-linux that is a systemd USER "
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

    service_restart = service_commands.add_parser(
        "restart",
        help=(
            "stop and start the installed service in one step, then wait for it to "
            "answer. Use this rather than stop then start: on a Windows PC the "
            "tray's watchdog starts a stopped engine itself"
        ),
    )
    service_restart.set_defaults(func=cmd_service_restart)

    service_status = service_commands.add_parser(
        "status",
        help="running or not, with the pid and the unit/plist path; exit 0 only "
        "when it is running",
    )
    service_status.add_argument(
        "--json", action="store_true", help="machine-readable"
    )
    service_status.set_defaults(func=cmd_service_status)
