from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Mapping

from .paths import ENGINE_PORT
from .runner import Runner

MIRRORED = "mirrored"
PORTPROXY = "portproxy"

RULE_NAME = "Crucible engine (LAN)"

RULE_PROFILE = "private"

WSLCONFIG_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class LanDoor:
    mechanism: str
    open: bool
    detail: str
    forward: bool = False
    firewall: bool = False
    private_network: bool | None = None


def wslconfig_path(env: Mapping[str, str]) -> PureWindowsPath | None:
    profile = env.get("USERPROFILE")
    if profile is None or profile.strip() == "":
        return None
    return PureWindowsPath(profile) / ".wslconfig"


def networking_mode(wslconfig_text: str) -> str | None:
    for raw in wslconfig_text.splitlines():
        line = raw.split("#", 1)[0].strip()
        match = re.match(r"^networkingMode\s*=\s*(\S+)$", line, re.IGNORECASE)
        if match is not None:
            return match.group(1).strip().lower()
    return None


def show_argv() -> list[str]:
    return ["netsh", "interface", "portproxy", "show", "v4tov4"]


def add_argv(port: int = ENGINE_PORT) -> list[str]:
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
    return ["netsh", "advfirewall", "firewall", "show", "rule", f"name={RULE_NAME}"]


def firewall_add_argv(port: int = ENGINE_PORT) -> list[str]:
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
    return [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "@(Get-NetConnectionProfile | Select-Object InterfaceAlias,"
        "@{Name='NetworkCategory';Expression={[string]$_.NetworkCategory}})"
        " | ConvertTo-Json -Compress",
    ]


def has_private_network(profile_json: str) -> bool | None:
    text = profile_json.strip()
    if text == "":
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
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


_INTERNAL_SWITCH = re.compile(r"^vEthernet \((WSL|Default Switch)", re.IGNORECASE)

_CATEGORY_PROFILE = {
    "private": "Private",
    "public": "Public",
    "domainauthenticated": "Domain",
}


@dataclass(frozen=True)
class NetworkInterface:
    address: str
    alias: str
    index: int
    network: str | None
    category: str | None

    @property
    def profile(self) -> str | None:
        if self.category is None:
            return None
        return _CATEGORY_PROFILE.get(self.category.strip().lower())

    @property
    def label(self) -> str:
        if self.network and self.network != self.alias:
            return f'"{self.network}" ({self.alias})'
        return f'"{self.alias}"'


@dataclass(frozen=True)
class FirewallProfile:
    enabled: bool
    inbound_allowed_by_default: bool
    allow_inbound_rules: bool
    allow_local_rules: bool


@dataclass(frozen=True)
class NetworkFacts:
    interfaces: tuple[NetworkInterface, ...]
    firewall: Mapping[str, FirewallProfile]
    rule_profiles: frozenset[str]
    elevated: bool


def network_argv() -> list[str]:
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
    result = runner.run(network_argv(), timeout_s=WSLCONFIG_TIMEOUT_SECONDS)
    if not result.ok:
        raise ValueError(f"Windows would not describe its networks ({result.said()})")
    return parse_network(result.stdout)


def offered(facts: NetworkFacts) -> list[NetworkInterface]:
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
    return [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        f"Set-NetConnectionProfile -InterfaceIndex {int(interface.index)} "
        "-NetworkCategory Private",
    ]


ELEVATION_SENTENCE = (
    "Crucible needs administrator once, to let other devices on your network "
    "reach the engine. It adds two things to Windows: a port forward from this "
    f"machine's network addresses on port {ENGINE_PORT} to the Linux engine, and "
    f'an inbound rule named "{RULE_NAME}" allowing TCP {ENGINE_PORT} on '
    f"{RULE_PROFILE} networks. Both stay until you run `crucible lan disable`, "
    "which removes exactly these two and nothing else."
)
