from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Any, Sequence
from urllib.parse import urlsplit

from . import pairing
from .backend import LLAMA_WINDOWS
from .platform import lan_door
from .protocol import LOOPBACK

WSL_GUEST = "wsl-guest"
WINDOWS = "windows"
POSIX = "posix"

PLACES: tuple[str, ...] = (WSL_GUEST, WINDOWS, POSIX)

LAN_COMMAND = "crucible lan enable"

WINDOW_ACTION = (
    'in the Crucible window (Start Menu: Crucible), open Settings and click Share '
    'under "Share on your network"'
)

WSL_ONLY_HERE = (
    "Only this PC can reach Crucible. Its Linux engine runs inside WSL2 and "
    "answers this PC alone until Windows forwards the port to it, so a phone or "
    "another computer gets no answer at all."
)

WSL_HOW = (
    f"To let other devices on this network use it, {WINDOW_ACTION}, or run "
    f"`{LAN_COMMAND}` in PowerShell on this PC (on Windows, not inside WSL)."
)

WINDOWS_ONLY_HERE = (
    "Only this PC can reach Crucible. Its Windows engine listens on 127.0.0.1, "
    "which nothing else can reach."
)

POSIX_ONLY_HERE = (
    "Only this machine can reach Crucible. It listens on {host}, which nothing "
    "else can reach."
)


@dataclass(frozen=True)
class Reach:
    reachable: bool
    urls: tuple[str, ...]
    sentence: str
    how: str | None = None
    command: str | None = None
    changes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reachable": self.reachable,
            "urls": list(self.urls),
            "sentence": self.sentence,
            "how": self.how,
            "command": self.command,
            "changes": self.changes,
        }

    def lines(self) -> list[str]:
        return [text for text in (self.sentence, self.how, self.changes) if text]


def is_loopback_host(host: str) -> bool:
    bare = host.strip().strip("[]").lower()
    if bare == "localhost":
        return True
    try:
        return ipaddress.ip_address(bare).is_loopback
    except ValueError:
        return False


def beyond_this_machine(urls: Sequence[str]) -> tuple[str, ...]:
    return tuple(url for url in urls if not is_loopback_host(urlsplit(url).hostname or ""))


def _shared(urls: tuple[str, ...]) -> Reach:
    return Reach(
        reachable=True,
        urls=urls,
        sentence="Other devices on the network reach Crucible at " + ", ".join(urls) + ".",
    )


def wsl_guest_closed() -> Reach:
    return Reach(
        reachable=False, urls=(), sentence=WSL_ONLY_HERE, how=WSL_HOW,
        command=LAN_COMMAND, changes=lan_door.ELEVATION_SENTENCE,
    )


def native_windows_closed(config_path: str, port: int) -> Reach:
    return Reach(
        reachable=False, urls=(), sentence=WINDOWS_ONLY_HERE,
        how=(
            f'To let other devices on this network use it, set host = "0.0.0.0" '
            f"under [server] in {config_path} and restart Crucible from its tray "
            f"icon. `{LAN_COMMAND}` does not apply to this engine: it forwards into "
            "WSL2, and aimed at an engine on Windows itself it would loop back onto "
            "itself, so it refuses."
        ),
        changes=(
            f"The engine then listens on every address this PC has, with its token "
            f"as the only lock, and Windows Firewall still needs an inbound rule "
            f"allowing TCP {port} on Private networks, which takes administrator."
        ),
    )


def posix_closed(host: str, config_path: str, port: int) -> Reach:
    return Reach(
        reachable=False, urls=(), sentence=POSIX_ONLY_HERE.format(host=host),
        how=(
            f'To let other devices on this network use it, set host = "0.0.0.0" '
            f"under [server] in {config_path}, then run `crucible service restart`."
        ),
        changes=(
            "It then listens on every address this machine has, with its token as "
            f"the only lock; a firewall on this machine must also let TCP {port} in."
        ),
    )


def engine_reach(*, place: str, bind_host: str, port: int, urls: Sequence[str],
                 advertised_urls: Sequence[str], config_path: str) -> Reach:
    if place not in PLACES:
        raise ValueError(f"{place!r} is not one of {PLACES}")
    if place == WSL_GUEST:
        shared = beyond_this_machine(advertised_urls)
        return _shared(shared) if shared else wsl_guest_closed()
    shared = beyond_this_machine(urls)
    if shared:
        return _shared(shared)
    if not is_loopback_host(bind_host):
        return Reach(
            reachable=False, urls=(),
            sentence=(
                f"Crucible listens on {bind_host}, but this machine has no network "
                "address another device could use. Connect it to the network."
            ),
        )
    if place == WINDOWS:
        return native_windows_closed(config_path, port)
    return posix_closed(bind_host, config_path, port)


def place_of(backend_kind: str) -> str:
    if backend_kind == LLAMA_WINDOWS:
        return WINDOWS
    from .service import in_wsl

    return WSL_GUEST if in_wsl() else POSIX


def for_server(config: Any, *, place: str, host: str, port: int) -> Reach:
    advertised = config.advertise + config.tailscale_advertise + config.lan_advertise
    return engine_reach(
        place=place, bind_host=host, port=port,
        urls=() if place == WSL_GUEST else pairing.reachable_urls(host, port, advertised),
        advertised_urls=pairing.reachable_urls(LOOPBACK, port, advertised),
        config_path=str(config.path),
    )


def wsl_door_reach(record: dict[str, Any] | None) -> Reach:
    if record is None:
        return wsl_guest_closed()
    urls = tuple(f"http://{authority}" for authority in record.get("authorities") or [])
    if record.get("state") == "configured" and urls:
        return _shared(urls)
    return Reach(
        reachable=False, urls=urls,
        sentence=(
            "Network sharing is turned on, but Windows does not let other devices "
            "in yet (Crucible's last check said so)."
        ),
        how=(
            "Run `crucible lan status` in PowerShell on this PC; it says what is "
            "in the way and what to do."
        ),
        command="crucible lan status",
    )
