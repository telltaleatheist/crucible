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

`connectaddress=127.0.0.1` and not the guest's address, which is the whole
reason this is cheap: WSL's own localhost forwarding already carries
`127.0.0.1` into the guest, so the row survives the guest's DHCP address
changing on every boot. A forward aimed at `eth0`'s address would need
re-pointing each time the distro started, which is a maintenance burden this
design simply does not have.

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

"Any network" was not enough (FRESH-INSTALL #46, kylies-pc 2026-09-26): its
Tailscale adapter was Private and its Ethernet Public, so the LAN was shut while
this said Private. `read_network` / `admits` below answer it per interface, and
`crucible/lan.py` decides from those. The rule stays Private-only; a Public
network is marked Private only when a person says it is theirs.

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

WSLCONFIG_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class LanDoor:
    """What this machine needs, and whether it already has it."""

    mechanism: str
    #: True when the LAN can already reach the engine: mirrored networking, or
    #: BOTH a portproxy row and an inbound allow.
    open: bool
    detail: str
    #: A portproxy row already forwards this port. False under `mirrored`,
    #: where the question does not arise.
    forward: bool = False
    #: An inbound allow named `RULE_NAME` already exists.
    firewall: bool = False
    #: At least one connected network is in the profile the rule is scoped to.
    #: None when the question could not be asked, which is not the same as no.
    private_network: bool | None = None


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
        if (parts[0] == "0.0.0.0" and parts[2] == "127.0.0.1"
                and int(parts[1]) == port and parts[3].isdigit() and int(parts[3]) == port):
            return True
    return False


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
    forward = has_forward(shown.stdout, port)
    # ABSENCE IS THE EXIT CODE, not the message: `netsh` prints a LOCALISED
    # "No rules match the specified criteria." and exits non-zero. Measured
    # 2026-09-17 unelevated: exit 1 absent, exit 0 present.
    firewall = runner.run(firewall_show_argv(), timeout_s=WSLCONFIG_TIMEOUT_SECONDS).ok
    profiled = runner.run(connection_profile_argv(), timeout_s=WSLCONFIG_TIMEOUT_SECONDS)
    private = has_private_network(profiled.stdout) if profiled.ok else None
    if forward and firewall:
        shut_out = (
            "" if private is not False else
            f", but no connected network is {RULE_PROFILE}, so the rule admits "
            "nothing here"
        )
        return LanDoor(
            mechanism=PORTPROXY,
            open=private is not False,
            detail=(
                f"a portproxy forwards 0.0.0.0:{port} to 127.0.0.1:{port} and "
                f'"{RULE_NAME}" admits it{shut_out}'
            ),
            forward=True,
            firewall=True,
            private_network=private,
        )
    if forward:
        return LanDoor(
            mechanism=PORTPROXY,
            open=False,
            detail=(
                f"a portproxy forwards 0.0.0.0:{port}, but no inbound rule named "
                f'"{RULE_NAME}" admits it, so Windows drops the connection before '
                "the forward sees it"
            ),
            forward=True,
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
        forward=False,
        firewall=firewall,
        private_network=private,
    )


# ------------------------------------------------ which networks, and are they open
#
# FRESH-INSTALL #46 and #47 (kylies-pc, 2026-09-26). `detect()` answers "is ANY
# connected network Private", and on kylies-pc that was true and useless: the
# Tailscale adapter was Private, the Ethernet was Public, so `lan enable` said
# `"private_network": true` and "configured" while nothing on the LAN could get
# in. And the addresses it published included 192.168.96.1, the WSL vEthernet
# adapter's, which no other machine can dial. Both are one question asked per
# INTERFACE instead of per machine, so this reads everything it needs in one
# PowerShell call: each address with its interface, each interface's network
# and category, the firewall's effective settings per profile, and whether this
# process already holds an administrator token (#44).
#
# Every read here answers an unprivileged caller; measured on owens-pc
# 2026-09-26 with a non-elevated shell.

#: The Hyper-V virtual switches that exist only inside this PC. WSL's own
#: adapter normally has no connection profile at all (measured on owens-pc:
#: "vEthernet (WSL (Hyper-V firewall))" is absent from Get-NetConnectionProfile),
#: which is what excludes it; the name is checked as well because the Default
#: Switch can carry an "Unidentified network" profile and is just as private
#: to this machine. An External switch bound to the real NIC is NOT on this
#: list — that one IS the LAN address.
_INTERNAL_SWITCH = re.compile(r"^vEthernet \((WSL|Default Switch)", re.IGNORECASE)

#: `NetworkCategory` spelled as the firewall profile that governs it.
_CATEGORY_PROFILE = {
    "private": "Private",
    "public": "Public",
    "domainauthenticated": "Domain",
}


@dataclass(frozen=True)
class NetworkInterface:
    """One IPv4 address, the interface it is on, and the network behind it."""

    address: str
    alias: str
    index: int
    #: The network's own name ("PrettyFlyForAWifi"), or None when Windows has
    #: no connection profile for the interface.
    network: str | None
    #: "Private", "Public", "DomainAuthenticated", or None with `network`.
    category: str | None

    @property
    def profile(self) -> str | None:
        """The firewall profile that decides what this interface admits."""
        if self.category is None:
            return None
        return _CATEGORY_PROFILE.get(self.category.strip().lower())

    @property
    def label(self) -> str:
        """How a person recognises it: the network's name, then the adapter's."""
        if self.network and self.network != self.alias:
            return f'"{self.network}" ({self.alias})'
        return f'"{self.alias}"'


@dataclass(frozen=True)
class FirewallProfile:
    """One firewall profile's EFFECTIVE settings (policy store ActiveStore)."""

    enabled: bool
    inbound_allowed_by_default: bool
    #: False is "Block all incoming connections, including allowed apps".
    allow_inbound_rules: bool
    #: False when group policy ignores rules made on this PC.
    allow_local_rules: bool


@dataclass(frozen=True)
class NetworkFacts:
    interfaces: tuple[NetworkInterface, ...]
    firewall: Mapping[str, FirewallProfile]
    #: The profiles an ENABLED inbound allow named `RULE_NAME` covers. Empty
    #: when there is no such rule, or it is disabled.
    rule_profiles: frozenset[str]
    #: This process already holds an administrator token, so a change needs no
    #: prompt at all (an administrator's SSH session on Windows is one).
    elevated: bool


def network_argv() -> list[str]:
    """Every fact `read_network` needs, as one compact JSON document."""
    script = (
        "$ErrorActionPreference='Stop';"
        "$p=@(Get-NetConnectionProfile | ForEach-Object {[pscustomobject]@{"
        "index=[int]$_.InterfaceIndex;alias=[string]$_.InterfaceAlias;"
        "name=[string]$_.Name;category=[string]$_.NetworkCategory}});"
        "$a=@(Get-NetIPAddress -AddressFamily IPv4 -AddressState Preferred | "
        "ForEach-Object {[pscustomobject]@{address=[string]$_.IPAddress;"
        "index=[int]$_.InterfaceIndex;alias=[string]$_.InterfaceAlias}});"
        "$f=@(Get-NetFirewallProfile -PolicyStore ActiveStore | ForEach-Object "
        "{[pscustomobject]@{profile=[string]$_.Name;enabled=[string]$_.Enabled;"
        "inbound=[string]$_.DefaultInboundAction;"
        "allow_rules=[string]$_.AllowInboundRules;"
        "local_rules=[string]$_.AllowLocalFirewallRules}});"
        f"$r=@(Get-NetFirewallRule -DisplayName '{RULE_NAME}' -ErrorAction "
        "SilentlyContinue | ForEach-Object {[pscustomobject]@{"
        "enabled=[string]$_.Enabled;profile=[string]$_.Profile;"
        "action=[string]$_.Action;direction=[string]$_.Direction}});"
        "$e=([Security.Principal.WindowsPrincipal][Security.Principal."
        "WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal."
        "WindowsBuiltInRole]::Administrator);"
        "[pscustomobject]@{profiles=$p;addresses=$a;firewall=$f;rule=$r;"
        "elevated=$e} | ConvertTo-Json -Compress -Depth 4"
    )
    return ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script]


def _flag(value: object, *, unset: bool) -> bool:
    """A GpoBoolean as text: "True", "False", or "NotConfigured" (= Windows' default)."""
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return unset


def _rows(value: object) -> list[dict]:
    rows = value if isinstance(value, list) else ([] if value is None else [value])
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("expected a list of objects")
    return rows


def parse_network(text: str) -> NetworkFacts:
    """`network_argv`'s output. Raises ValueError on anything else."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("expected an object")
    profiles = {int(row["index"]): row for row in _rows(data.get("profiles"))}
    interfaces = []
    for row in _rows(data.get("addresses")):
        index = int(row["index"])
        profile = profiles.get(index)
        interfaces.append(NetworkInterface(
            address=str(row["address"]),
            alias=str(row["alias"]),
            index=index,
            network=None if profile is None else (str(profile.get("name") or "") or None),
            category=None if profile is None else (str(profile.get("category") or "") or None),
        ))
    firewall = {
        str(row["profile"]): FirewallProfile(
            enabled=_flag(row.get("enabled"), unset=True),
            inbound_allowed_by_default=str(row.get("inbound", "")).strip().lower() == "allow",
            allow_inbound_rules=_flag(row.get("allow_rules"), unset=True),
            allow_local_rules=_flag(row.get("local_rules"), unset=True),
        )
        for row in _rows(data.get("firewall"))
    }
    covered: set[str] = set()
    for row in _rows(data.get("rule")):
        if (not _flag(row.get("enabled"), unset=False)
                or str(row.get("action", "")).strip().lower() != "allow"
                or str(row.get("direction", "")).strip().lower() != "inbound"):
            continue
        for word in str(row.get("profile", "")).split(","):
            word = word.strip().lower()
            if word == "any":
                covered.update(("Domain", "Private", "Public"))
            elif word in ("domain", "private", "public"):
                covered.add(word.capitalize())
    return NetworkFacts(
        interfaces=tuple(interfaces),
        firewall=firewall,
        rule_profiles=frozenset(covered),
        elevated=data.get("elevated") is True,
    )


def read_network(runner: Runner) -> NetworkFacts:
    """Run `network_argv` and parse it. Raises ValueError, naming why."""
    result = runner.run(network_argv(), timeout_s=WSLCONFIG_TIMEOUT_SECONDS)
    if not result.ok:
        raise ValueError(f"Windows would not describe its networks ({result.said()})")
    return parse_network(result.stdout)


def offered(facts: NetworkFacts) -> list[NetworkInterface]:
    """The interfaces another machine could dial. #47.

    Out: loopback and link-local (as `crucible/interfaces.py` rules), any
    interface Windows has no network profile for, and the Hyper-V switches that
    exist only inside this PC (`_INTERNAL_SWITCH`). Order is Windows' own.
    """
    found: list[NetworkInterface] = []
    for interface in facts.interfaces:
        if interface.address.startswith(("127.", "169.254.", "0.")):
            continue
        if interface.category is None or _INTERNAL_SWITCH.match(interface.alias):
            continue
        if all(seen.address != interface.address for seen in found):
            found.append(interface)
    return found


def admits(interface: NetworkInterface, facts: NetworkFacts) -> tuple[bool, str]:
    """Would Windows Firewall let another computer's TCP connection in here?

    Decided from the effective profile settings and the rule, which is exactly
    what Windows decides from. It says nothing about a router that keeps its
    devices apart (guest Wi-Fi does) or a third-party firewall; `lan.py` says so.
    """
    profile = interface.profile
    if profile is None:
        return False, f"Windows does not know what kind of network {interface.label} is"
    settings = facts.firewall.get(profile)
    if settings is None:
        return False, f"Windows did not report its firewall settings for {profile} networks"
    if not settings.enabled:
        return True, f"Windows Firewall is off for {profile} networks"
    if not settings.allow_inbound_rules:
        return False, (
            f"Windows Firewall is set to block ALL incoming connections on "
            f"{profile} networks, which overrides every allow rule"
        )
    if settings.inbound_allowed_by_default:
        return True, f"Windows Firewall lets incoming connections in on {profile} networks"
    if profile in facts.rule_profiles:
        if not settings.allow_local_rules:
            return False, (
                "this PC's firewall is managed by an organization's policy, which "
                f'ignores the "{RULE_NAME}" rule set on this PC'
            )
        return True, f'the "{RULE_NAME}" rule lets TCP in on {profile} networks'
    if profile == "Public":
        return False, (
            f"this network {interface.label} is marked Public, and Windows keeps "
            "other computers out of a Public network"
        )
    if profile == "Domain":
        return False, (
            f"{interface.label} is an organization's (domain) network, and "
            f'"{RULE_NAME}" only covers home (Private) networks'
        )
    return False, f'no enabled "{RULE_NAME}" rule covers {profile} networks'


def make_private_argv(interface: NetworkInterface) -> list[str]:
    """Mark one network Private. Run ELEVATED; `lan.py` asks the person first.

    A nested `powershell.exe -Command` rather than the cmdlet as an argv,
    because `lan.elevated_argv` quotes every word, and a quoted `'-InterfaceIndex'`
    is a string to PowerShell, not a parameter name. The index is an int this
    module parsed, so nothing a network's name contains reaches the command.
    """
    return [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        f"Set-NetConnectionProfile -InterfaceIndex {int(interface.index)} "
        "-NetworkCategory Private",
    ]


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
    "reach the engine. It adds two things to Windows: a port forward from this "
    f"machine's network addresses on port {ENGINE_PORT} to the Linux engine, and "
    f'an inbound rule named "{RULE_NAME}" allowing TCP {ENGINE_PORT} on '
    f"{RULE_PROFILE} networks. Both stay until you run `crucible lan disable`, "
    "which removes exactly these two and nothing else."
)
