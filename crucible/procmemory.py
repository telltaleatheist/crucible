"""Host memory, read the way each operating system Crucible serves on gives it.

Owen, 2026-11-27: "also remember some systems wont have wsl. some systems will be using a
mac, or windows". A model's footprint (crucible/footprints.py) and the room a load is
weighed against are read here, one reader per system, each with the figure that system
itself counts:

- Linux (cuda-linux, natively or inside WSL2): `/proc/meminfo`'s MemAvailable and MemTotal,
  and each process's Pss (`/proc/<pid>/smaps_rollup`), which splits a shared page between the
  processes sharing it, so an engine's API server and its EngineCore are not counted twice.
  Under WSL2 these are the guest's, which is the limit an engine there is killed at.
- macOS (mlx-darwin): `vm_stat`'s available and total (accelerator.probe_unified_memory),
  and each process's physical footprint (`proc_pid_rusage`, `ri_phys_footprint`): the figure
  Activity Monitor shows as Memory, which counts Metal's allocations, where resident size
  does not. On Apple silicon the card and host memory are one pool, so this IS the model's
  card figure there.
- Windows (llama-windows): `GlobalMemoryStatusEx`'s available and total physical memory, and
  each process's working set (`GetProcessMemoryInfo`), the physical memory Task Manager
  counts for it.

A process that has gone between being named and being read counts nothing. Anything else a
reader cannot read raises ProbeError, never a zero.
"""
from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from .accelerator import (
    ProbeError,
    expand_owned_pids,
    probe_process_table,
    probe_unified_memory,
)

PROC = Path("/proc")
MEMINFO = PROC / "meminfo"
WSL_RELEASE = PROC / "sys" / "kernel" / "osrelease"


# --- Linux ------------------------------------------------------------------------------


def _meminfo_bytes(field: str, meminfo: Path) -> int:
    with open(meminfo, encoding="ascii") as handle:
        for line in handle:
            key, _, value = line.partition(":")
            if key == field:
                number, unit = value.split()
                if unit != "kB":
                    raise ProbeError(f"{meminfo} gives {field} in {unit!r}, not kB")
                return int(number) * 1024
    raise ProbeError(f"{meminfo} has no {field}")


def linux_host_memory(meminfo: Path = MEMINFO) -> tuple[int, int]:
    try:
        return _meminfo_bytes("MemAvailable", meminfo), _meminfo_bytes("MemTotal", meminfo)
    except OSError as exc:
        raise ProbeError(f"could not read {meminfo}: {exc}") from None


def linux_pss_bytes(pids: Iterable[int], proc: Path = PROC) -> int:
    total = 0
    for pid in pids:
        path = proc / str(pid) / "smaps_rollup"
        try:
            text = path.read_text(encoding="ascii")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as exc:
            raise ProbeError(f"could not read {path}: {exc}") from None
        for line in text.splitlines():
            key, _, value = line.partition(":")
            if key == "Pss":
                number, unit = value.split()
                if unit != "kB":
                    raise ProbeError(f"{path} gives Pss in {unit!r}, not kB")
                total += int(number) * 1024
                break
        else:
            raise ProbeError(f"{path} has no Pss line")
    return total


def under_wsl(release: Path = WSL_RELEASE) -> bool:
    try:
        return "microsoft" in release.read_text(encoding="ascii").lower()
    except OSError:
        return False


# --- macOS ------------------------------------------------------------------------------


class _RusageInfoV2(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
    ]


RUSAGE_INFO_V2 = 2
ESRCH = 3


def mac_footprint_bytes(pids: Iterable[int]) -> int:
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    except OSError as exc:
        raise ProbeError(f"could not open libproc: {exc}") from None
    total = 0
    for pid in pids:
        info = _RusageInfoV2()
        if libproc.proc_pid_rusage(int(pid), RUSAGE_INFO_V2, ctypes.byref(info)) != 0:
            errno = ctypes.get_errno()
            if errno == ESRCH:
                continue
            raise ProbeError(f"proc_pid_rusage({pid}) failed with errno {errno}")
        total += int(info.ri_phys_footprint)
    return total


def mac_group_members(pids: frozenset[int]) -> frozenset[int]:
    """Every process in the process groups ``pids`` lead: an engine starts in a new session,
    so its children share its group."""
    ps = shutil.which("ps") or "/bin/ps"
    try:
        completed = subprocess.run(
            [ps, "-A", "-o", "pid=,pgid="], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProbeError(f"could not run {ps}: {exc}") from None
    if completed.returncode != 0:
        raise ProbeError(f"{ps} exited {completed.returncode}: {completed.stderr.strip()}")
    members = set(pids)
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        pid, group = int(fields[0]), int(fields[1])
        if group in pids:
            members.add(pid)
    return frozenset(members)


# --- Windows ----------------------------------------------------------------------------


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_VM_READ = 0x0010
ERROR_INVALID_PARAMETER = 87


def _kernel32() -> "ctypes.WinDLL":  # type: ignore[name-defined]
    return ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]


def windows_host_memory() -> tuple[int, int]:
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not _kernel32().GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ProbeError(
            f"GlobalMemoryStatusEx failed with error {ctypes.get_last_error()}"  # type: ignore[attr-defined]
        )
    return int(status.ullAvailPhys), int(status.ullTotalPhys)


def windows_working_set_bytes(pids: Iterable[int]) -> int:
    kernel32 = _kernel32()
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.K32GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_ProcessMemoryCounters), ctypes.c_ulong
    ]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    total = 0
    for pid in pids:
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, int(pid)
        )
        if not handle:
            error = ctypes.get_last_error()  # type: ignore[attr-defined]
            if error == ERROR_INVALID_PARAMETER:  # no such process: it has gone
                continue
            raise ProbeError(f"OpenProcess({pid}) failed with error {error}")
        try:
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
            if not kernel32.K32GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                raise ProbeError(
                    f"GetProcessMemoryInfo({pid}) failed with error "
                    f"{ctypes.get_last_error()}"  # type: ignore[attr-defined]
                )
            total += int(counters.WorkingSetSize)
        finally:
            kernel32.CloseHandle(handle)
    return total


# --- This system ------------------------------------------------------------------------


def host_memory() -> tuple[int, int]:
    """(available, total) host memory on this system, in bytes."""
    if sys.platform == "darwin":
        return probe_unified_memory()
    if sys.platform == "win32":
        return windows_host_memory()
    return linux_host_memory()


def process_bytes(pids: Iterable[int]) -> int:
    """What ``pids`` hold in host memory, summed, by this system's own count."""
    if sys.platform == "darwin":
        return mac_footprint_bytes(pids)
    if sys.platform == "win32":
        return windows_working_set_bytes(pids)
    return linux_pss_bytes(pids)


def with_children(pids: frozenset[int]) -> frozenset[int]:
    """``pids`` and every process they lead (an engine's API server leads its EngineCore)."""
    if not pids:
        return pids
    if sys.platform == "darwin":
        return mac_group_members(pids)
    if sys.platform == "win32":
        return pids
    return expand_owned_pids(pids, probe_process_table())


def memory_words() -> str:
    """What the room a load is weighed against is called on this system."""
    if sys.platform == "darwin":
        return "this Mac"
    if sys.platform == "win32":
        return "this PC"
    if under_wsl():
        return "Crucible's Linux (WSL's share of this PC)"
    return "this machine"


def more_memory_fix() -> str:
    if sys.platform == "linux" and under_wsl():
        return "raise `memory` under [wsl2] in %UserProfile%\\.wslconfig and restart WSL"
    return "run it on a machine with more memory"
