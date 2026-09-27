from __future__ import annotations

import errno
import os
import signal
import subprocess
import sys
from typing import Any

from .errors import CrucibleError

POSIX_PLATFORMS = frozenset({"linux", "darwin"})
WIN32 = "win32"

KILL_WAIT_SECONDS = 10.0

STOP_TIMEOUT_SECONDS = 180.0

LOG_TAIL_LINES = 40


class ProcessGroupError(CrucibleError):
    ...


def platform_kind(platform: str | None = None) -> str:
    name = sys.platform if platform is None else platform
    if name in POSIX_PLATFORMS:
        return "posix"
    if name == WIN32:
        return WIN32
    raise ProcessGroupError(
        f"platform {name!r} has no process-group mechanism in Crucible: "
        f"{sorted(POSIX_PLATFORMS)} use setsid + SIGTERM and win32 uses "
        "CREATE_NEW_PROCESS_GROUP + CTRL_BREAK_EVENT. A stop that guessed would "
        "leave a child running that nothing can see"
    )


def own_group(platform: str | None = None) -> dict[str, Any]:
    if platform_kind(platform) == "posix":
        return {"start_new_session": True}
    return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}


def ask_to_stop(process: "subprocess.Popen[Any]") -> bool:
    if process.poll() is not None:
        return False
    if platform_kind() == "posix":
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            return False
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                return False
            raise ProcessGroupError(
                f"could not signal pid {process.pid}'s group: {exc}"
            ) from exc
        return True
    try:
        process.send_signal(signal.CTRL_BREAK_EVENT)
    except OSError:
        return False
    return True


def terminate_tree(process: "subprocess.Popen[Any]", what: str) -> None:
    if platform_kind() != WIN32:
        raise ProcessGroupError(
            f"{what} (pid {process.pid}) would be force-killed, and off win32 "
            "Crucible never does that: a killed CUDA process in a WSL2 GPU wait "
            "wedges the distro until Windows reboots"
        )
    if process.poll() is not None:
        return
    result = subprocess.run(
        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=KILL_WAIT_SECONDS,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        process.wait(timeout=KILL_WAIT_SECONDS)
    except subprocess.TimeoutExpired:
        raise ProcessGroupError(
            f"{what} (pid {process.pid}) survived `taskkill /T /F` "
            f"(exit {result.returncode}: "
            f"{(result.stdout + result.stderr).strip() or 'no output'})"
        ) from None


def ask_groups_to_stop(pids: "frozenset[int] | set[int]") -> frozenset[int]:
    if platform_kind() != "posix":
        raise ProcessGroupError(
            f"pids {sorted(pids)} were to be asked to stop by process group, and "
            "only POSIX has process groups to signal"
        )
    failed: set[int] = set()
    groups: dict[int, set[int]] = {}
    for pid in pids:
        try:
            groups.setdefault(os.getpgid(pid), set()).add(pid)
        except ProcessLookupError:
            continue
        except OSError:
            failed.add(pid)
    for group, members in groups.items():
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except OSError:
            failed.update(members)
    return frozenset(failed)


def stop_budget_seconds(sigterm_wait_seconds: float) -> float:
    if platform_kind() == WIN32:
        return sigterm_wait_seconds + 2 * KILL_WAIT_SECONDS
    return sigterm_wait_seconds


def stop_gracefully(
    process: "subprocess.Popen[Any]",
    what: str,
    timeout_seconds: float,
    log_path: "os.PathLike[str] | str",
) -> None:
    if process.poll() is not None:
        return
    win32 = platform_kind() == WIN32
    try:
        if not ask_to_stop(process):
            if win32:
                terminate_tree(process, what)
            return
        try:
            process.wait(timeout=timeout_seconds)
            return
        except subprocess.TimeoutExpired:
            pass
        if win32:
            terminate_tree(process, what)
            return
    except ProcessGroupError as exc:
        raise ProcessGroupError(
            f"could not stop {what} (pid {process.pid}): {exc}. Its log is {log_path}"
        ) from exc
    raise ProcessGroupError(
        f"{what} (pid {process.pid}) did not exit within {timeout_seconds:.0f}s "
        "of SIGTERM. Crucible does not SIGKILL a process holding CUDA: that "
        "wedges WSL2 until Windows reboots. Stop it with "
        f"`kill {process.pid}` (never -9), then run the request again. Its log "
        f"is {log_path}"
    )


__all__ = [
    "KILL_WAIT_SECONDS",
    "LOG_TAIL_LINES",
    "STOP_TIMEOUT_SECONDS",
    "POSIX_PLATFORMS",
    "ProcessGroupError",
    "WIN32",
    "ask_groups_to_stop",
    "ask_to_stop",
    "own_group",
    "platform_kind",
    "stop_budget_seconds",
    "stop_gracefully",
    "terminate_tree",
]
