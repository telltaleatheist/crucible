from __future__ import annotations

import argparse
import os
import sys

from .common import EXIT_OK, EXIT_REFUSED, _fail


def _orchestrator_try_again() -> int:
    from ..config import crucible_home
    from ..host import outcome as host_outcome
    from ..host.retry import try_again
    from ..platform.errors import HostError

    try:
        ended = try_again(crucible_home())
    except (HostError, OSError) as exc:
        message = f"{exc.code}: {exc.message}" if isinstance(exc, HostError) else str(exc)
        return _fail(message)
    if ended is None:
        return _fail("wsl_outcome_invalid: the move ended without recording how")
    if ended.sentence:
        print(ended.sentence)
    if ended.state == host_outcome.DONE:
        print("The Linux engine is set up and running.")
        return EXIT_OK
    return EXIT_REFUSED


def cmd_guest(args: argparse.Namespace) -> int:
    from ..host import guestcli

    return guestcli.run(args.guest_words)


def cmd_orchestrator(args: argparse.Namespace) -> int:
    from ..platform import startup as host_startup
    from ..platform.errors import HostError
    from ..platform.runner import ProcessRunner

    if sys.platform != "win32":
        return _fail(
            "host_windows_only: `crucible orchestrator` is a Windows verb. On "
            f"{sys.platform} the server runs on this machine and "
            f"{'systemd' if sys.platform == 'linux' else 'launchd'} already "
            "supervises it — `crucible service status` is the question you "
            "are asking."
        )

    if getattr(args, "try_again", False):
        return _orchestrator_try_again()

    runner = ProcessRunner(sys.platform, os.environ)
    try:
        if args.install_startup:
            outcome = host_startup.install(runner)
            print(outcome.detail)
            return EXIT_OK
        if args.remove_startup:
            outcome = host_startup.remove(runner)
            print(outcome.detail)
            return EXIT_OK
    except HostError as exc:
        return _fail(f"{exc.code}: {exc.message}")

    from ..host.app import run as run_host

    try:
        if not args.headless:
            from ..desktop import tray
            tray()
            return EXIT_OK
        return run_host(headless=args.headless)
    except HostError as exc:
        return _fail(f"{exc.code}: {exc.message}")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    host_parser = subparsers.add_parser(
        "orchestrator",
        help="win32 only: the tray that manages this machine's engine",
        description=(
            "The Windows ORCHESTRATOR (docs/internals/host-and-platform.md): a "
            "notification-area icon that boots this machine's engine at login, "
            "claims it, watches it, restarts it, and runs the move from the "
            "Windows engine to WSL2 when the operator page asks. It serves zero "
            "job types and carries no data — control is Windows's, data is the "
            "card's. Refused `host_windows_only` on Linux and macOS, where the "
            "service manager already supervises the server."
        ),
    )
    host_parser.add_argument(
        "--install-startup",
        action="store_true",
        help="write the Startup shortcut and exit (this verb OWNS that file)",
    )
    host_parser.add_argument(
        "--remove-startup",
        action="store_true",
        help="delete the Startup shortcut and exit",
    )
    host_parser.add_argument(
        "--try-again",
        action="store_true",
        help="set up the Linux engine again after it stopped, and say how it ended",
    )
    host_parser.set_defaults(func=cmd_orchestrator)
    host_parser.add_argument("--headless", action="store_true", help="Run the controller independently of the tray")

    guest_parser = subparsers.add_parser(
        "guest",
        help="win32 only: run a crucible command inside this PC's Linux engine",
        description=(
            "Forward the rest of the line to the `crucible` inside the WSL "
            "distro this PC's orchestrator manages, as the user its engine runs "
            "as: `crucible guest install rvc`, `crucible guest doctor`."
        ),
    )
    guest_parser.add_argument("guest_words", nargs=argparse.REMAINDER,
                              help="the command to run inside the Linux engine")
    guest_parser.set_defaults(func=cmd_guest)
