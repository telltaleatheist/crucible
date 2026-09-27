from __future__ import annotations

import ctypes
import socket
import struct
import sys

from .errors import CrucibleError

_DARWIN = sys.platform == "darwin"

_IFF_UP = 0x1

_SOCKADDR_BYTES = 16


class InterfaceError(CrucibleError):
    ...


class _SockAddr(ctypes.Structure):
    _fields_ = [("raw", ctypes.c_ubyte * _SOCKADDR_BYTES)]


class _IfAddrs(ctypes.Structure):
    pass


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
    try:
        handle = ctypes.CDLL(None)
    except OSError as exc:
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
