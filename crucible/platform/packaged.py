from __future__ import annotations

import sys
from typing import Callable

from .errors import LocalError

APPMODEL_ERROR_NO_PACKAGE = 15700
ERROR_INSUFFICIENT_BUFFER = 122

REFUSAL_CODE = "packaged_shell"

ORDINARY_POWERSHELL = (
    "Open an ordinary PowerShell window (Start, type PowerShell, press Enter) "
    "and run it there."
)


WRITES_CONSEQUENCE = (
    "Windows quietly redirects what such a process writes under AppData into "
    "that app's own private folder, so Crucible would land where only that app "
    "can see it, and would not start when you sign in. Nothing has been "
    "written. "
)

REMOVES_CONSEQUENCE = (
    "Windows shows such a process that app's own private folder laid over the "
    "real AppData, so what it removes there is not reliably what is on the disk: "
    "an uninstall from inside it could take away part of Crucible and leave the "
    "rest, a split install neither half of which can be started or finished. "
    "Nothing has been removed. "
)


def refusal_sentence(package: str, what: str, *, removes: bool = False) -> str:
    return (
        f"{REFUSAL_CODE}: {what} is running inside the Windows app package "
        f"{package} (an app installed from the Store or as an MSIX, such as "
        "the Claude desktop app, and anything started from a terminal inside it). "
        + (REMOVES_CONSEQUENCE if removes else WRITES_CONSEQUENCE)
        + ORDINARY_POWERSHELL
    )


def _kernel32_package_name() -> str | None:
    import ctypes
    from ctypes import wintypes

    get_name = ctypes.windll.kernel32.GetCurrentPackageFullName
    get_name.argtypes = [ctypes.POINTER(wintypes.UINT), wintypes.LPWSTR]
    get_name.restype = wintypes.LONG
    length = wintypes.UINT(0)
    code = get_name(ctypes.byref(length), None)
    if code == APPMODEL_ERROR_NO_PACKAGE:
        return None
    if code != ERROR_INSUFFICIENT_BUFFER:
        raise LocalError(
            f"package_check_failed: Windows would not say whether this process runs "
            f"inside an app package (GetCurrentPackageFullName returned {code}), and "
            "a Crucible written from inside one is a Crucible nothing can start. "
            + ORDINARY_POWERSHELL
        )
    name = ctypes.create_unicode_buffer(length.value)
    code = get_name(ctypes.byref(length), name)
    if code != 0:
        raise LocalError(
            f"package_check_failed: GetCurrentPackageFullName returned {code} on its "
            "second call. " + ORDINARY_POWERSHELL
        )
    return name.value


def package_name(
    platform: str | None = None,
    probe: Callable[[], str | None] | None = None,
) -> str | None:
    if (sys.platform if platform is None else platform) != "win32":
        return None
    return (_kernel32_package_name if probe is None else probe)()


def refuse_packaged(
    what: str,
    *,
    removes: bool = False,
    platform: str | None = None,
    probe: Callable[[], str | None] | None = None,
) -> None:
    package = package_name(platform, probe)
    if package is not None:
        raise LocalError(refusal_sentence(package, what, removes=removes))
