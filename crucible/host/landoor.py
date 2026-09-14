"""The LAN door — PHASE15-HOST.md 4.1's fifth bullet.

The WSL server binds loopback inside the guest. Windows forwards `127.0.0.1` to
the guest for free, which is why an app on the same machine works; nothing
forwards the machine's LAN addresses, which is why `/v1/setup`'s LAN pairing
lines are true on Linux and on the Mac and were not true on Windows. Somebody
had to own that forward, and the host is the only thing on Windows that knows
when the WSL engine is up.

TWO MECHANISMS, DETECTED BY NAME, NEVER GUESSED
------------------------------------------------
- **mirrored networking.** WSL >= 2.0 on Windows 11 22H2+ with
  `networkingMode=mirrored` in `.wslconfig` puts the guest ON the host's
  interfaces: a guest listener is already reachable at the machine's LAN
  address and a portproxy would be a second, redundant hop. Nothing to do, and
  saying "nothing to do" is the answer, not a silence.
- **portproxy.** Everything else: `netsh interface portproxy add v4tov4
  listenport=7100 listenaddress=0.0.0.0 connectport=7100
  connectaddress=127.0.0.1`. It needs administrator ONCE, and the host asks by
  name with the sentence that says why.

WHAT THIS MODULE DOES NOT DO
-----------------------------
It does not run `netsh` as a side effect of anything. `detect()` READS, and
`add_argv()` / `remove_argv()` spell the change; `app.py` is what runs them,
after a person has been told what it is for. Measured on Owen's PC 2026-09-14:
`netsh interface portproxy show v4tov4` listed NOTHING, WSL is 2.5.7.0 and
`.wslconfig` has no `networkingMode`, so that machine is the portproxy case
with no forward yet — read, and deliberately not changed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Mapping

from .paths import ENGINE_PORT
from .runner import Runner

#: The two mechanism names. They appear in the log, in the menu's detail and in
#: the door's events, so they are constants rather than prose.
MIRRORED = "mirrored"
PORTPROXY = "portproxy"

WSLCONFIG_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class LanDoor:
    """What this machine needs, and whether it already has it."""

    mechanism: str
    #: True when the LAN can already reach the engine: mirrored networking, or
    #: a portproxy row that is already there.
    open: bool
    detail: str


def wslconfig_path(env: Mapping[str, str]) -> PureWindowsPath | None:
    """`%USERPROFILE%\\.wslconfig`, or None when there is no profile to read."""
    profile = env.get("USERPROFILE")
    if profile is None or profile.strip() == "":
        return None
    return PureWindowsPath(profile) / ".wslconfig"


def networking_mode(wslconfig_text: str) -> str | None:
    """`networkingMode` out of `.wslconfig`, or None when it is not stated.

    None is a FACT: WSL's own default is NAT, and an absent key means the
    machine is on NAT. It is reported as None rather than as "nat" so that
    "the operator chose NAT" and "the operator said nothing" stay distinct in
    the log, which is the difference between a forward being expected and a
    forward being a surprise.
    """
    for raw in wslconfig_text.splitlines():
        line = raw.split("#", 1)[0].strip()
        match = re.match(r"^networkingMode\s*=\s*(\S+)$", line, re.IGNORECASE)
        if match is not None:
            return match.group(1).strip().lower()
    return None


def show_argv() -> list[str]:
    return ["netsh", "interface", "portproxy", "show", "v4tov4"]


def add_argv(port: int = ENGINE_PORT) -> list[str]:
    """The forward, as one argv. Run ELEVATED; `netsh` refuses otherwise."""
    return [
        "netsh",
        "interface",
        "portproxy",
        "add",
        "v4tov4",
        f"listenport={port}",
        "listenaddress=0.0.0.0",
        f"connectport={port}",
        "connectaddress=127.0.0.1",
    ]


def remove_argv(port: int = ENGINE_PORT) -> list[str]:
    return [
        "netsh",
        "interface",
        "portproxy",
        "delete",
        "v4tov4",
        f"listenport={port}",
        "listenaddress=0.0.0.0",
    ]


def has_forward(show_output: str, port: int = ENGINE_PORT) -> bool:
    """Is a v4tov4 row already listening on this port?

    Parsed by the two NUMBERS on a row rather than by column position: `netsh`
    localises its headers and pads its columns differently per locale, and a
    reader keyed on "the third word" is a reader that is wrong in German.
    """
    for raw in show_output.replace("\x00", "").splitlines():
        parts = raw.split()
        if len(parts) < 4:
            continue
        if not parts[1].isdigit():
            continue
        if int(parts[1]) == port and parts[3].isdigit() and int(parts[3]) == port:
            return True
    return False


def detect(runner: Runner, port: int = ENGINE_PORT) -> LanDoor:
    """Which mechanism this machine is on, and whether the LAN is already open.

    Reads `.wslconfig` and `netsh`. Runs nothing that changes anything.
    """
    path = wslconfig_path(runner.env)
    mode: str | None = None
    if path is not None:
        result = runner.run(
            ["cmd.exe", "/c", "type", str(path)], timeout_s=WSLCONFIG_TIMEOUT_SECONDS
        )
        if result.ok:
            mode = networking_mode(result.stdout)
    if mode == MIRRORED:
        return LanDoor(
            mechanism=MIRRORED,
            open=True,
            detail=(
                f"{path} sets networkingMode=mirrored, so the guest is already on "
                "this machine's interfaces and there is nothing to forward"
            ),
        )
    shown = runner.run(show_argv(), timeout_s=WSLCONFIG_TIMEOUT_SECONDS)
    if not shown.ok:
        return LanDoor(
            mechanism=PORTPROXY,
            open=False,
            detail=f"netsh could not be read ({shown.said()}); the LAN door is unknown",
        )
    if has_forward(shown.stdout, port):
        return LanDoor(
            mechanism=PORTPROXY,
            open=True,
            detail=f"a portproxy already forwards 0.0.0.0:{port} to 127.0.0.1:{port}",
        )
    return LanDoor(
        mechanism=PORTPROXY,
        open=False,
        detail=(
            f"nothing forwards this machine's LAN addresses on {port} to the WSL "
            "engine, so only this computer can reach it"
        ),
    )


#: What a person is told before the one UAC prompt this module causes. Named
#: rather than composed at the call site, because a consent dialog whose
#: sentence is assembled from fragments is a sentence nobody reviewed.
ELEVATION_SENTENCE = (
    "Crucible needs administrator once, to let other devices on your network "
    "reach the engine. It adds a single Windows port forward from this "
    f"machine's network addresses on port {ENGINE_PORT} to the Linux engine, and "
    "removes it when the engine stops. Nothing else on this machine changes."
)
