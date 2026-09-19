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
  listenport=7100 listenaddress=192.168.68.100 connectport=7100
  connectaddress=127.0.0.1`, ONE ROW PER ADDRESS THIS MACHINE HAS. It needs
  administrator ONCE, and the host asks by name with the sentence that says why.

`connectaddress=127.0.0.1` and not the guest's address, which is the whole
reason this is cheap: WSL's own localhost forwarding already carries
`127.0.0.1` into the guest, so the row survives the guest's DHCP address
changing on every boot. A forward aimed at `eth0`'s address would need
re-pointing each time the distro started, which is a maintenance burden this
design simply does not have.

NEVER `listenaddress=0.0.0.0`, AND THAT IS ONE RULE ABOUT THE LISTEN SET
------------------------------------------------------------------------
The wildcard is not "all the LAN addresses". It is every address this machine
answers on, `127.0.0.1` included — which is the address the row FORWARDS TO.
So the portproxy service accepted its own connection and dialled itself.
Measured on Owen's PC 2026-09-17: 15.5k of 16.4k ephemeral ports in TIME_WAIT
and localhost keepers failing at random.

The refusal is `portproxy_self_loop` and it asks one question — does what
this row would answer on include what it would dial? — rather than asking which
engine is behind the connect address. That is deliberate, because the answer is
the same either way: `llama-windows` puts the engine itself on 127.0.0.1:7100,
and the WSL engine is reached at the same loopback address by way of WSL's own
localhost forwarding. Two backends, one connect address, ONE rule. A check
keyed on the backend would have been a second rule that could disagree with
this one.

So `add_argv` takes the listen address and requires it to be one of this
machine's own; `crucible/lan.py` enumerates them and asks for a row each.
`remove_argv` takes one too and refuses nothing, because a wildcard row a
machine already carries is precisely the row that has to come back out.

THE FORWARD ALONE IS A DEAD DOOR
---------------------------------
The portproxy's listener belongs to the Windows service that owns forwarding,
not to any Crucible binary, so no program-scoped firewall rule covers it and
the default inbound block drops the connection before the forward ever sees
it. Measured on Owen's PC 2026-09-17: the only inbound rule naming 7100 is
Zoom's, scoped to Zoom's own executable. So this module owns TWO rows — the
forward and an inbound allow — and reports them separately, because a machine
with one and not the other is a machine where the door looks open from the
Windows side and is shut from the network's.

The allow is scoped to the **Private** profile. A LAN a person calls theirs is
Private; Public is the coffee shop, and a rule that admitted the coffee shop
because it was convenient here would be this file choosing an exposure on
somebody else's behalf. `detect()` therefore also reports whether any connected
network IS Private — a Private-scoped rule on a machine whose only network is
Public is precisely the silent no-op this codebase refuses to ship.

WHAT THIS MODULE DOES NOT DO
-----------------------------
It does not run `netsh` as a side effect of anything. `detect()` READS, and the
`*_argv()` builders spell the change; `crucible/lan.py` is what runs them, after
a person has been told what it is for. Reads need no elevation — measured, both
`portproxy show` and `advfirewall firewall show rule` answer an unprivileged
caller — so detection is always available and only a CHANGE ever prompts.

Measured on Owen's PC 2026-09-14 and again 2026-09-17: `netsh interface
portproxy show v4tov4` listed NOTHING, WSL is 2.5.7.0 and `.wslconfig` has no
`networkingMode`, so that machine is the portproxy case with no forward yet.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Mapping

from .errors import HostError
from .paths import ENGINE_PORT
from .runner import Runner

#: The two mechanism names. They appear in the log, in the menu's detail and in
#: the door's events, so they are constants rather than prose.
MIRRORED = "mirrored"
PORTPROXY = "portproxy"

#: The inbound rule's name. It is the HANDLE the removal uses, so it is one
#: constant rather than a sentence rebuilt at two call sites that could drift.
RULE_NAME = "Crucible engine (LAN)"

#: The firewall profile the allow is scoped to. See the module docstring.
RULE_PROFILE = "private"

#: Where every row Crucible adds points, and why one address serves both
#: backends: WSL carries loopback into the guest, and a native Windows engine
#: binds loopback directly. One constant, because the reader of the listing and
#: the builders of the argv must not disagree about what "the engine" means.
CONNECT_ADDRESS = "127.0.0.1"

#: `netsh`'s spelling of "every address this machine answers on", loopback
#: included. Named so the refusal and the reader share one word for it.
WILDCARD_ADDRESS = "0.0.0.0"

WSLCONFIG_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class LanDoor:
    """What this machine needs, and whether it already has it."""

    mechanism: str
    #: True when the LAN can already reach the engine: mirrored networking, or
    #: BOTH a portproxy row and an inbound allow.
    open: bool
    detail: str
    #: Every listen address already forwarding this port to the engine, in the
    #: order `netsh` listed them. Addresses rather than a boolean, because the
    #: wildcard row that has to come out and the per-address rows that go in are
    #: different rows on one machine and a boolean cannot tell them apart.
    #: Empty under `mirrored`, where the question does not arise.
    forwards: tuple[str, ...] = ()
    #: An inbound allow named `RULE_NAME` already exists.
    firewall: bool = False
    #: At least one connected network is in the profile the rule is scoped to.
    #: None when the question could not be asked, which is not the same as no.
    private_network: bool | None = None

    @property
    def forward(self) -> bool:
        """Is there ANY row? The half-open report's question, unchanged."""
        return bool(self.forwards)

    @property
    def self_loops(self) -> tuple[str, ...]:
        """The rows on this machine that dial themselves. Derived, never stored."""
        return tuple(a for a in self.forwards if covers_connect_address(a))


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


def covers_connect_address(
    listen_address: str, connect_address: str = CONNECT_ADDRESS
) -> bool:
    """Does a row listening here also answer at the address it forwards TO?

    `0.0.0.0` is every address this machine has, loopback included, so a
    wildcard row always covers its own target; a row that names the target
    outright is the same fact spelled shorter. Either way the forward dials
    itself, so either way it is the loop.
    """
    return listen_address in (WILDCARD_ADDRESS, connect_address)


def add_argv(listen_address: str, port: int = ENGINE_PORT) -> list[str]:
    """One forward, as one argv. Run ELEVATED; `netsh` refuses otherwise.

    `listen_address` is REQUIRED and has no default. It was `0.0.0.0` until
    2026-09-18, and a default spelling the one value this function must never
    emit is a defect waiting for its next caller.
    """
    if covers_connect_address(listen_address):
        raise HostError(
            "portproxy_self_loop",
            f"a forward listening on {listen_address} would also answer at "
            f"{CONNECT_ADDRESS}:{port}, which is where it sends what it "
            "accepts, so it would dial itself until this machine ran out of "
            "ephemeral ports. Listen on one of this machine's own LAN "
            "addresses instead",
        )
    return [
        "netsh",
        "interface",
        "portproxy",
        "add",
        "v4tov4",
        f"listenport={port}",
        f"listenaddress={listen_address}",
        f"connectport={port}",
        f"connectaddress={CONNECT_ADDRESS}",
    ]


def remove_argv(listen_address: str, port: int = ENGINE_PORT) -> list[str]:
    """One forward, taken back out.

    It refuses nothing. The wildcard row a machine already carries is exactly
    the row this has to be able to remove, so the rule that stops `add_argv`
    composing one must not also stop this deleting one.
    """
    return [
        "netsh",
        "interface",
        "portproxy",
        "delete",
        "v4tov4",
        f"listenport={port}",
        f"listenaddress={listen_address}",
    ]


def firewall_show_argv() -> list[str]:
    """Read the rule by NAME. Answers an unprivileged caller; measured."""
    return ["netsh", "advfirewall", "firewall", "show", "rule", f"name={RULE_NAME}"]


def firewall_add_argv(port: int = ENGINE_PORT) -> list[str]:
    """The inbound allow, as one argv. Run ELEVATED."""
    return [
        "netsh",
        "advfirewall",
        "firewall",
        "add",
        "rule",
        f"name={RULE_NAME}",
        "dir=in",
        "action=allow",
        "protocol=TCP",
        f"localport={port}",
        f"profile={RULE_PROFILE}",
    ]


def firewall_remove_argv(port: int = ENGINE_PORT) -> list[str]:
    return [
        "netsh",
        "advfirewall",
        "firewall",
        "delete",
        "rule",
        f"name={RULE_NAME}",
        "protocol=TCP",
        f"localport={port}",
    ]


def connection_profile_argv() -> list[str]:
    """Each connected network's firewall category, as JSON."""
    return [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        # `[string]` is not decoration: NetworkCategory serialises as its
        # INTEGER enum value, and this probe reported "unreadable" on a machine
        # whose network is Private until that was measured. PowerShell is the
        # authority on its own enum's spelling, so it does the conversion rather
        # than this file carrying a 0/1/2 table it would have had to guess.
        "@(Get-NetConnectionProfile | Select-Object InterfaceAlias,"
        "@{Name='NetworkCategory';Expression={[string]$_.NetworkCategory}})"
        " | ConvertTo-Json -Compress",
    ]


def has_private_network(profile_json: str) -> bool | None:
    """Is any connected network in `RULE_PROFILE`? None when unreadable.

    None rather than False for an unparseable or empty answer: "no Private
    network" is a claim that would send a person off to change their network
    category, and this module does not make claims it did not measure.
    """
    text = profile_json.strip()
    if text == "":
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    # PowerShell serialises one object as a scalar and none as empty.
    rows = data if isinstance(data, list) else [data]
    seen = False
    for row in rows:
        if not isinstance(row, dict):
            return None
        category = row.get("NetworkCategory")
        if isinstance(category, str):
            seen = True
            if category.strip().lower() == RULE_PROFILE:
                return True
    return False if seen else None


def forward_addresses(show_output: str, port: int = ENGINE_PORT) -> tuple[str, ...]:
    """Every listen address already forwarding this port to the engine.

    Parsed by the two NUMBERS on a row rather than by column position: `netsh`
    localises its headers and pads its columns differently per locale, and a
    reader keyed on "the third word" is a reader that is wrong in German.

    A wildcard row is REPORTED, never filtered away. It is the row `crucible
    lan` has to take out, and a reader that skipped it would leave the
    self-loop running while saying the door was correct.
    """
    found: list[str] = []
    for raw in show_output.replace("\x00", "").splitlines():
        parts = raw.split()
        if len(parts) < 4 or not parts[1].isdigit() or not parts[3].isdigit():
            continue
        if parts[2] != CONNECT_ADDRESS:
            continue
        if int(parts[1]) == port and int(parts[3]) == port and parts[0] not in found:
            found.append(parts[0])
    return tuple(found)


def detect(runner: Runner, port: int = ENGINE_PORT) -> LanDoor:
    """Which mechanism this machine is on, and whether the LAN is already open.

    Reads `.wslconfig`, `netsh` and the connection profiles. Runs nothing that
    changes anything.
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
            open=False,
            detail=(
                f"{path} requests mirrored networking. This does not prove the "
                "active WSL network mode, a non-loopback listener, or firewall access; "
                "remote reachability has not been tested"
            ),
        )
    shown = runner.run(show_argv(), timeout_s=WSLCONFIG_TIMEOUT_SECONDS)
    if not shown.ok:
        return LanDoor(
            mechanism=PORTPROXY,
            open=False,
            detail=f"netsh could not be read ({shown.said()}); the LAN door is unknown",
        )
    forwards = forward_addresses(shown.stdout, port)
    # ABSENCE IS THE EXIT CODE, not the message: `netsh` prints a LOCALISED
    # "No rules match the specified criteria." and exits non-zero. Measured
    # 2026-09-17 unelevated: exit 1 absent, exit 0 present.
    firewall = runner.run(firewall_show_argv(), timeout_s=WSLCONFIG_TIMEOUT_SECONDS).ok
    profiled = runner.run(connection_profile_argv(), timeout_s=WSLCONFIG_TIMEOUT_SECONDS)
    private = has_private_network(profiled.stdout) if profiled.ok else None
    listed = ", ".join(f"{address}:{port}" for address in forwards)
    # A row that covers its own target is named in the DETAIL as well as
    # refused at composition time: `crucible lan status` on a machine that
    # already carries one is where a person meets it.
    looping = tuple(a for a in forwards if covers_connect_address(a))
    loop = (
        "" if not looping else
        f" — and {', '.join(looping)} covers {CONNECT_ADDRESS}, so that row "
        "forwards to itself and eats this machine's ephemeral ports"
    )
    if forwards and firewall:
        shut_out = (
            "" if private is not False else
            f", but no connected network is {RULE_PROFILE}, so the rule admits "
            "nothing here"
        )
        return LanDoor(
            mechanism=PORTPROXY,
            open=private is not False and not looping,
            detail=(
                f"a portproxy forwards {listed} to {CONNECT_ADDRESS}:{port} and "
                f'"{RULE_NAME}" admits it{shut_out}{loop}'
            ),
            forwards=forwards,
            firewall=True,
            private_network=private,
        )
    if forwards:
        return LanDoor(
            mechanism=PORTPROXY,
            open=False,
            detail=(
                f"a portproxy forwards {listed}, but no inbound rule named "
                f'"{RULE_NAME}" admits it, so Windows drops the connection before '
                f"the forward sees it{loop}"
            ),
            forwards=forwards,
            firewall=False,
            private_network=private,
        )
    return LanDoor(
        mechanism=PORTPROXY,
        open=False,
        detail=(
            f"nothing forwards this machine's LAN addresses on {port} to the WSL "
            "engine, so only this computer can reach it"
        ),
        forwards=(),
        firewall=firewall,
        private_network=private,
    )


#: What a person is told before the one UAC prompt this module causes. Named
#: rather than composed at the call site, because a consent dialog whose
#: sentence is assembled from fragments is a sentence nobody reviewed.
#:
#: It no longer promises removal "when the engine stops". That promise cost a
#: UAC prompt on every start and every stop and bought nothing: a forward to a
#: port with no listener refuses a connection exactly as a machine with no
#: forward does. The rows persist until `crucible lan disable` removes them,
#: which is what the sentence now says.
ELEVATION_SENTENCE = (
    "Crucible needs administrator once, to let other devices on your network "
    "reach the engine. It adds two things to Windows: a port forward to the "
    "engine from each of this machine's own network addresses on port "
    f"{ENGINE_PORT} — never from every address, which would include this "
    "machine's loopback and make the forward dial itself — and an inbound "
    f'rule named "{RULE_NAME}" allowing TCP {ENGINE_PORT} on {RULE_PROFILE} '
    "networks. Both stay until you run `crucible lan disable`, which removes "
    "exactly these and nothing else."
)
