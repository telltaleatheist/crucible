from __future__ import annotations

import os
import subprocess
import sys
from typing import Sequence

from .. import wsl

GUEST_CRUCIBLE_SH = f'exec "{wsl.GUEST_CRUCIBLE}" "$@"'


def guest_command_argv(distro: str, words: Sequence[str]) -> list[str]:
    return wsl.guest_argv(distro, ["bash", "-lc", GUEST_CRUCIBLE_SH, "crucible", *words])


def managed_distro() -> str:
    from ..config import crucible_home
    from ..platform.hostconfig import consented_distro

    return consented_distro(crucible_home()) or wsl.CRUCIBLE_DISTRO


def run(words: Sequence[str]) -> int:
    from ..platform.runner import ProcessRunner
    from .errors import HostError

    if sys.platform != "win32":
        print(
            "crucible: guest_windows_only: `crucible guest` reaches the Linux engine "
            "a Windows PC runs inside WSL. On this machine the engine is here, and "
            "this `crucible` is the one to run.",
            file=sys.stderr,
        )
        return 2
    if not words:
        print(
            "crucible: guest_needs_words: say what to run in the Linux engine, "
            "for example `crucible guest doctor`.",
            file=sys.stderr,
        )
        return 2
    try:
        distro = managed_distro()
    except HostError as exc:
        print(f"crucible: {exc.code}: {exc.message}", file=sys.stderr)
        return 1
    listed = ProcessRunner(sys.platform, os.environ).run(wsl.list_argv(), timeout_s=60.0)
    if not listed.ok or distro not in wsl.parse_distro_list(listed.stdout):
        print(
            f'crucible: guest_absent: this PC has no Linux engine yet (no "{distro}" '
            "WSL distribution). Crucible sets it up by itself: open its icon in the "
            "notification area, and `crucible guest` works once it says it is running.",
            file=sys.stderr,
        )
        return 1
    try:
        return subprocess.run(guest_command_argv(distro, list(words))).returncode
    except OSError as exc:
        print(f"crucible: guest_unreachable: wsl.exe could not be run ({exc})", file=sys.stderr)
        return 1
