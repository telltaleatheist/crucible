from __future__ import annotations

import argparse
import os
import sys

from .common import EXIT_OK, EXIT_REFUSED, _fail


def _orchestrator_try_again() -> int:
    """`crucible orchestrator --try-again` (FRESH-INSTALL #16, 2026-09-26).

    The door's `POST /install`, made by the product for a person who has no
    app: the same move and the same claim as an app's Try again. It prints one
    line per step and ends with the outcome's own sentence.
    """
    from ..config import crucible_home
    from ..host import outcome as host_outcome
    from ..host.errors import HostError
    from ..host.retry import try_again

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
    """`crucible guest <words…>`: `crucible/host/guestcli.py` (fresh-install #29)."""
    from ..host import guestcli

    return guestcli.run(args.guest_words)


def cmd_orchestrator(args: argparse.Namespace) -> int:
    """`crucible orchestrator` — PHASE15 section 4, PHASE17. Windows only.

    The verb is refused `host_windows_only` everywhere else, and that is not a
    platform check standing in for a feature check: on Linux and macOS the
    server runs ON the machine and its own service manager supervises it
    (4.4, "no host on the Mac"). There is nothing for a tray to own.

    Three shapes, and the two that are not the tray exit without starting one:

      --install-startup   write the Startup item and print its path
      --remove-startup    delete it, and say whether there was one
      --try-again         PHASE19 2.5's Try again, for a machine with no app
                          (FRESH-INSTALL #16): run the Linux-engine move once
                          more, follow it, and say how it ended
      (bare)              the tray
    """
    from ..host import startup as host_startup
    from ..host.errors import HostError
    from ..host.runner import ProcessRunner

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
            "The Windows ORCHESTRATOR (PHASE15-HOST.md section 4, PHASE17): a "
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

    # FRESH-INSTALL #29 (2026-09-26): the Linux engine's own CLI, from Windows,
    # as the user the engine runs as. Nobody spells a guest path.
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
