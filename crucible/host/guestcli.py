"""`crucible guest …`: the Linux engine's own CLI, run from Windows.

Fresh-install #29 (2026-09-26, kylies-pc). Every hand step on a Windows PC whose
engine lives in WSL was spelled `wsl -d crucible -- ~/.crucible/server/bin/crucible
<verb>`, and `~` is whichever user the distro defaulted to at that moment: root
straight after the import, `crucible` after its first restart. The engine moved
between the two during that install, so the same line reached two different
Crucibles on the same evening. Owen's ruling for the product is that nobody
spells a guest path (PHASE19: "nobody is ever shown a command"; a hand step is a
bug), and when one IS needed, the Windows `crucible` gets there by itself:

    crucible guest install rvc
    crucible guest doctor

This forwards the words, unchanged, to the guest's own `crucible`, as the user
the engine runs as (`presence.guest_argv`: `-u crucible` in Crucible's distro,
the distro's own default user in a consented one), in the distro this machine's
orchestrator manages (`[orchestrator] distro` when a person named one, else
`crucible`). The guest's home is asked of the guest (`$CRUCIBLE_HOME`, else
`$HOME/.crucible`), never composed here. Standard input and output are the
caller's, so a long `install` streams and a prompt, if any, is answered.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Sequence

#: The guest's console script, resolved INSIDE the guest. `exec` so the exit
#: code that comes back through wsl.exe is the guest CLI's own.
GUEST_CRUCIBLE_SH = 'exec "${CRUCIBLE_HOME:-$HOME/.crucible}/server/bin/crucible" "$@"'


def guest_command_argv(distro: str, words: Sequence[str]) -> list[str]:
    """`wsl.exe -d <distro> [-u crucible] --exec bash -lc '<exec crucible "$@">' crucible <words…>`.

    The words ride as bash's positional parameters (`$0` is `crucible`), so no
    quoting rule on either side of wsl.exe can change one of them.
    """
    from .presence import guest_argv

    return guest_argv(distro, ["bash", "-lc", GUEST_CRUCIBLE_SH, "crucible", *words])


def managed_distro() -> str:
    """The distro this machine's orchestrator manages: consent, else Crucible's own."""
    from ..config import crucible_home
    from .app import consented_distro
    from .wsl_states import CRUCIBLE_DISTRO

    return consented_distro(crucible_home()) or CRUCIBLE_DISTRO


def run(words: Sequence[str]) -> int:
    """Forward `words` to the guest's `crucible`. The exit code is the guest's."""
    from .errors import HostError
    from .presence import parse_wsl_list, wsl_list_argv
    from .runner import ProcessRunner

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
    listed = ProcessRunner(sys.platform, os.environ).run(wsl_list_argv(), timeout_s=60.0)
    if not listed.ok or distro not in parse_wsl_list(listed.stdout):
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
