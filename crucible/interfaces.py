"""Which addresses this host actually has — asked of the OS, never guessed.

`GET /v1/setup` answers *"where can an app reach this server"*, and a server
bound to `0.0.0.0` can only answer that by listing its interfaces. There are
four ways to produce such a list and three of them produce something else
(PHASE13-OPERATOR.md section 3.1):

* `socket.gethostbyname(socket.gethostname())` is a **name lookup**, which is
  the one thing that route promises not to do. It answers with whatever this
  host's resolver believes today — `127.0.1.1` on a stock Debian, an old DHCP
  lease on a laptop that moved, nothing at all on a box with no search domain —
  and a pairing line built on it sends an app to an address that may belong to
  somebody else.
* A UDP socket `connect()`ed to a routable address and asked its
  `getsockname()` reports **one** interface, chosen by the routing table for a
  destination nobody is going to. On Owen's PC that picks the LAN address and
  silently omits the tailnet one, which is the address the Mac would use.
* `psutil.net_if_addrs()` is correct, and is a dependency added to a server
  whose whole point is to be installable, for a fact the C library on both
  supported backends already states.

So this asks **`getifaddrs(3)`** through `ctypes`, which is stdlib, is present
on `cuda-linux` (glibc) and `mlx-darwin` (libSystem), and is the same call
`psutil` would have made. Windows uses Get-NetIPAddress's structured result;
it has no getifaddrs C ABI.

WHAT IS EXCLUDED, AND WHY EACH EXCLUSION IS A FACT RATHER THAN A TASTE
----------------------------------------------------------------------
* **Anything that is not AF_INET.** IPv6 is not excluded because it is unwanted;
  it is excluded because `GET /v1/setup` says `non-loopback IPv4 interface` and
  a route that quietly returned more than it documents is a route whose readers
  disagree about its shape. Adding IPv6 is a contract change in
  PHASE13-OPERATOR.md first.
* **Loopback, `127.0.0.0/8`.** Not an address another machine can use, so not an
  address a pairing line can carry. A server bound explicitly to `127.0.0.1`
  still reports that one URL — see `crucible/pairing.py` — because there the
  operator stated it.
* **Link-local, `169.254.0.0/16`.** An address a host gives itself when DHCP
  failed. It is evidence of a network that did not come up, never a route.
* **Interfaces that are not `IFF_UP`.** A configured address on a down interface
  is a reachable address for nobody.

A HOST THAT CANNOT BE ASKED IS A REFUSAL, NOT AN EMPTY LIST
-----------------------------------------------------------
`InterfaceError` is raised when `getifaddrs` cannot be called or fails. An empty
`urls` array reads as *"this server is reachable from nowhere"*, which is a
claim, and a false one on a host where the probe simply did not run (R3:
nothing is ever told "maybe"). The caller turns it into
`503 interfaces_unreadable`.
"""

from __future__ import annotations

import ctypes
import socket
import struct
import sys

from .errors import CrucibleError

#: `struct sockaddr`'s first two bytes hold the family, and the two supported
#: platforms disagree about how. Linux has `sa_family_t sa_family` — an
#: `unsigned short` at offset 0. The BSDs, macOS included, have
#: `uint8_t sa_len; uint8_t sa_family` — so the family is the SECOND byte. Both
#: spell AF_INET 2, and on a little-endian Linux the low byte of the short is
#: also offset 0, so a naive "read byte 0" would work on Linux and read a
#: LENGTH on the Mac. Asked properly rather than accidentally.
_DARWIN = sys.platform == "darwin"

#: `IFF_UP`, 0x1 on Linux and on the BSDs alike (`net/if.h` in both).
_IFF_UP = 0x1

#: Enough bytes for a `sockaddr_in`; this module never reads past `sin_addr`.
#: Declaring the union's full 128-byte `sockaddr_storage` would be honest too
#: and buys nothing, since the pointer is the kernel's and is only ever read.
_SOCKADDR_BYTES = 16


class InterfaceError(CrucibleError):
    """This host's interface list could not be read. Carries the reason."""


class _SockAddr(ctypes.Structure):
    _fields_ = [("raw", ctypes.c_ubyte * _SOCKADDR_BYTES)]


class _IfAddrs(ctypes.Structure):
    pass


# `struct ifaddrs` has the same layout on Linux and Darwin; the union of
# broadaddr/dstaddr is one pointer either way.
_IfAddrs._fields_ = [
    ("ifa_next", ctypes.POINTER(_IfAddrs)),
    ("ifa_name", ctypes.c_char_p),
    ("ifa_flags", ctypes.c_uint),
    ("ifa_addr", ctypes.POINTER(_SockAddr)),
    ("ifa_netmask", ctypes.POINTER(_SockAddr)),
    ("ifa_ifu", ctypes.POINTER(_SockAddr)),
    ("ifa_data", ctypes.c_void_p),
]


def _libc() -> ctypes.CDLL:
    """The C library this process is already linked against.

    `CDLL(None)` rather than `find_library("c")`: the symbols are in this
    process's own namespace on both platforms, so there is no file to locate
    and no version suffix to guess at. A host where the symbol is genuinely
    absent is named rather than crashed on.
    """
    try:
        handle = ctypes.CDLL(None)
    except OSError as exc:  # pragma: no cover - no such host in this build
        raise InterfaceError(
            f"this process has no C library to ask for its interfaces: {exc}"
        ) from exc
    if not hasattr(handle, "getifaddrs"):
        raise InterfaceError(
            "this host's C library has no getifaddrs(3), so Crucible cannot say "
            "which addresses it is reachable on. Bind to a concrete address "
            "instead of 0.0.0.0 ([server] host in config.toml), which needs no "
            "interface list"
        )
    return handle


def _family(address: _SockAddr) -> int:
    raw = bytes(address.raw[:2])
    if _DARWIN:
        return raw[1]
    return int(struct.unpack("@H", raw)[0])


def ipv4_addresses() -> list[str]:
    """Every non-loopback, non-link-local IPv4 address, in the OS's own order.

    Duplicates are collapsed keeping the first sighting: an aliased interface
    can present one address twice, and two identical URLs in `/v1/setup` would
    be two identical pairing lines for a person to choose between.
    """
    if sys.platform == "win32":
        return _windows_ipv4_addresses()
    libc = _libc()
    head = ctypes.POINTER(_IfAddrs)()
    if libc.getifaddrs(ctypes.byref(head)) != 0:
        raise InterfaceError(
            "getifaddrs(3) failed; this host will not say which addresses it has"
        )
    found: list[str] = []
    try:
        node = head
        while node:
            entry = node.contents
            node = entry.ifa_next
            if not entry.ifa_addr:
                continue
            if not entry.ifa_flags & _IFF_UP:
                continue
            if _family(entry.ifa_addr.contents) != socket.AF_INET:
                continue
            # `sockaddr_in` is {family/len, port, in_addr}: the four address
            # bytes start at offset 4 on both platforms, because Darwin's
            # `sa_len` takes the byte Linux's wider `sa_family` already had.
            packed = bytes(entry.ifa_addr.contents.raw[4:8])
            text = socket.inet_ntoa(packed)
            if text.startswith("127.") or text.startswith("169.254."):
                continue
            if text not in found:
                found.append(text)
    finally:
        libc.freeifaddrs(head)
    return found


def _windows_ipv4_addresses() -> list[str]:
    import ipaddress
    import json
    import subprocess
    script = (
        "$ErrorActionPreference='Stop'; "
        "@(Get-NetIPAddress -AddressFamily IPv4 -AddressState Preferred "
        "| Select-Object -ExpandProperty IPAddress) | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            raise InterfaceError(f"Get-NetIPAddress failed: {result.stderr.strip()}")
        # PowerShell serializes a single item as a scalar, no items as empty.
        data = json.loads(result.stdout) if result.stdout.strip() else []
        if isinstance(data, str):
            data = [data]
        if not isinstance(data, list):
            raise ValueError("expected an address list")
        found = []
        for value in data:
            address = ipaddress.IPv4Address(value)
            if address.is_loopback or address.is_link_local or address.is_unspecified:
                continue
            if str(address) not in found:
                found.append(str(address))
        return found
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise InterfaceError(f"Windows network interfaces could not be read: {exc}") from exc


__all__ = ["InterfaceError", "ipv4_addresses"]
