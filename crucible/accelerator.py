from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backend import (
    BACKEND_KINDS,
    CUDA_LINUX,
    LLAMA_WINDOWS,
    MLX_DARWIN,
    nvidia_smi_path,
    physical_memory_figures,
)
from .capability import available_bytes
from .engines.vllm import card_needs
from .errors import ApiError, CrucibleError, NoViableBackend

GIB = 1024 ** 3

FOREIGN_PROCESS_FLOOR_BYTES = 1 * GIB

LLAMA_SERVER_IMAGE = "llama-server"


def is_llama_server(process_name: str) -> bool:
    stem = process_name.replace("\\", "/").rsplit("/", 1)[-1].strip().lower()
    if stem.endswith(".exe"):
        stem = stem[: -len(".exe")]
    return stem == LLAMA_SERVER_IMAGE


class ProbeError(CrucibleError):
    ...


@dataclass(frozen=True)
class ComputeApp:
    pid: int
    name: str
    used_bytes: int | None

    def describe(self) -> str:
        if self.used_bytes is None:
            return f"pid {self.pid} ({self.name}, memory not reported)"
        return f"pid {self.pid} ({self.name}, {self.used_bytes / GIB:.1f} GiB)"


@dataclass(frozen=True)
class AcceleratorState:
    backend: str
    total_bytes: int
    free_bytes: int
    compute_apps: tuple[ComputeApp, ...]
    detail: str

    @property
    def used_bytes(self) -> int:
        return self.total_bytes - self.free_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
            "used_bytes": self.used_bytes,
            "compute_apps": [
                {"pid": a.pid, "name": a.name, "used_bytes": a.used_bytes}
                for a in self.compute_apps
            ],
            "detail": self.detail,
        }


def _nvidia_smi(query: str, what: str) -> list[str]:
    exe = nvidia_smi_path()
    if exe is None:
        raise ProbeError(
            "no nvidia-smi on this host, so the card cannot be inspected; Crucible "
            "will not start an engine it cannot check first"
        )
    try:
        completed = subprocess.run(
            [exe, query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except OSError as exc:
        raise ProbeError(f"could not run {exe} to read {what}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ProbeError(f"{exe} did not answer {what} within 30s") from exc
    if completed.returncode != 0:
        raise ProbeError(
            f"{exe} {query} exited {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip() or 'no output'}"
        )
    return [line for line in completed.stdout.splitlines() if line.strip()]


def probe_compute_apps() -> list[ComputeApp]:
    lines = _nvidia_smi(
        "--query-compute-apps=pid,process_name,used_gpu_memory", "the process list"
    )
    apps: list[ComputeApp] = []
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ProbeError(f"could not parse a compute-apps row: {line!r}")
        pid_text, name, used_text = parts
        try:
            pid = int(pid_text)
        except ValueError:
            raise ProbeError(f"could not parse a compute-apps pid: {line!r}") from None
        try:
            used: int | None = int(used_text) * 1024 * 1024
        except ValueError:
            used = None
        apps.append(ComputeApp(pid=pid, name=name, used_bytes=used))
    return apps


def probe_process_table() -> dict[int, tuple[int, int]]:
    table: dict[int, tuple[int, int]] = {}
    proc = Path("/proc")
    if not proc.is_dir():
        return table
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            fields = stat[stat.rindex(")") + 2 :].split()
            table[int(entry.name)] = (int(fields[2]), int(fields[3]))
        except (ValueError, IndexError):
            raise ProbeError(f"could not parse {entry / 'stat'}: {stat!r}") from None
    return table


_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_ERROR_ACCESS_DENIED = 5
_STILL_ACTIVE = 259


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    command: str
    started: str

    def same_process_as(self, other: "ProcessIdentity") -> bool:
        return (
            self.pid == other.pid
            and bool(self.command)
            and bool(self.started)
            and self.command == other.command
            and self.started == other.started
        )

    def to_dict(self) -> dict[str, Any]:
        return {"pid": self.pid, "command": self.command, "started": self.started}


def _kernel32() -> Any:
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [
        ctypes.POINTER(wintypes.FILETIME)
    ] * 4
    kernel.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    return kernel


def _windows_process(pid: int) -> tuple[bool, ProcessIdentity | None]:
    import ctypes
    from ctypes import wintypes

    kernel = _kernel32()
    handle = kernel.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ctypes.get_last_error() == _ERROR_ACCESS_DENIED, None
    try:
        code = wintypes.DWORD()
        if kernel.GetExitCodeProcess(handle, ctypes.byref(code)) and (
            code.value != _STILL_ACTIVE
        ):
            return False, None
        created, exited, kernel_time, user_time = (
            wintypes.FILETIME() for _ in range(4)
        )
        started = ""
        if kernel.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            started = str((created.dwHighDateTime << 32) | created.dwLowDateTime)
        size = wintypes.DWORD(32_768)
        buffer = ctypes.create_unicode_buffer(size.value)
        command = (
            buffer.value
            if kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size))
            else ""
        )
        return True, ProcessIdentity(pid=pid, command=command, started=started)
    finally:
        kernel.CloseHandle(handle)


def _proc_identity(pid: int) -> ProcessIdentity | None:
    entry = Path("/proc") / str(pid)
    try:
        stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
        raw = (entry / "cmdline").read_bytes()
    except OSError:
        return None
    try:
        fields = stat[stat.rindex(")") + 2 :].split()
        state, started = fields[0], fields[19]
    except (ValueError, IndexError):
        return None
    if state == "Z":
        return None
    command = " ".join(
        part.decode("utf-8", "replace") for part in raw.split(bytes(1)) if part
    )
    return ProcessIdentity(pid=pid, command=command, started=started)


def _ps_field(pid: int, field: str) -> str:
    try:
        completed = subprocess.run(
            ["ps", "-p", str(pid), "-o", f"{field}="],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _posix_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def process_identity(pid: int) -> ProcessIdentity | None:
    if pid <= 0:
        return None
    if sys.platform == "win32":
        return _windows_process(pid)[1]
    if Path("/proc").is_dir():
        return _proc_identity(pid)
    if not _posix_alive(pid):
        return None
    return ProcessIdentity(
        pid=pid, command=_ps_field(pid, "command"), started=_ps_field(pid, "lstart")
    )


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        return _windows_process(pid)[0]
    if Path("/proc").is_dir():
        return _proc_identity(pid) is not None
    return _posix_alive(pid)


def ask_pid_to_stop(pid: int) -> bool:
    if sys.platform == "win32":
        try:
            os.kill(pid, signal.CTRL_BREAK_EVENT)
        except OSError:
            return False
        return True
    try:
        group = os.getpgid(pid)
        if group == pid and group != os.getpgrp():
            os.killpg(group, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    return True


@dataclass(frozen=True)
class Leftover:
    identity: ProcessIdentity
    asked_at: str

    def sentence(self) -> str:
        pid = self.identity.pid
        command = self.identity.command or "(command unknown)"
        return (
            f"a previous Crucible left {pid} {command} running; it was asked to "
            f"stop at {self.asked_at}; stop it with `kill {pid}` (never -9)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity.to_dict(), "asked_at": self.asked_at}


_leftovers: dict[int, Leftover] = {}
_leftovers_lock = threading.Lock()


def note_leftover(identity: ProcessIdentity, asked_at: str) -> None:
    with _leftovers_lock:
        _leftovers[identity.pid] = Leftover(identity=identity, asked_at=asked_at)


def forget_leftover(pid: int) -> None:
    with _leftovers_lock:
        _leftovers.pop(pid, None)


def leftovers() -> list[Leftover]:
    with _leftovers_lock:
        noted = list(_leftovers.values())
    alive = [left for left in noted if process_alive(left.identity.pid)]
    for gone in noted:
        if gone not in alive:
            forget_leftover(gone.identity.pid)
    return alive


def _left_behind(
    message: str, details: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    alive = leftovers()
    if not alive:
        return message, details
    said = "; ".join(left.sentence() for left in alive)
    return (
        f"{message} {said[0].upper()}{said[1:]}.",
        {**details, "left_by_previous_run": [left.to_dict() for left in alive]},
    )


def expand_owned_pids(
    owned_pids: frozenset[int], table: dict[int, tuple[int, int]]
) -> frozenset[int]:
    sessions = {pid for pid in owned_pids if table.get(pid, (0, 0))[1] == pid}
    groups = {pid for pid in owned_pids if table.get(pid, (0, 0))[0] == pid}
    if not sessions and not groups:
        return owned_pids
    return owned_pids | {
        pid
        for pid, (group, session) in table.items()
        if session in sessions or group in groups
    }


def probe_vram() -> tuple[int, int]:
    lines = _nvidia_smi("--query-gpu=memory.free,memory.total", "the memory figures")
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) != 2:
        raise ProbeError(f"could not parse a memory row: {lines[0]!r}")
    try:
        free_mib, total_mib = int(parts[0]), int(parts[1])
    except ValueError:
        raise ProbeError(f"could not parse the memory figures: {lines[0]!r}") from None
    return free_mib * 1024 * 1024, total_mib * 1024 * 1024


def probe_unified_memory() -> tuple[int, int]:
    vm_stat = shutil.which("vm_stat") or "/usr/bin/vm_stat"
    try:
        completed = subprocess.run(
            [vm_stat], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProbeError(f"could not run {vm_stat}: {exc}") from exc
    if completed.returncode != 0:
        raise ProbeError(
            f"{vm_stat} exited {completed.returncode}: {completed.stderr.strip()}"
        )

    page_size: int | None = None
    counters: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        if line.startswith("Mach Virtual Memory Statistics"):
            marker = "page size of "
            index = line.find(marker)
            if index == -1:
                raise ProbeError(f"vm_stat did not state its page size: {line!r}")
            page_size = int(line[index + len(marker) :].split()[0])
            continue
        key, separator, value = line.partition(":")
        if separator != ":":
            continue
        digits = value.strip().rstrip(".")
        if digits.isdigit():
            counters[key.strip()] = int(digits)
    if page_size is None:
        raise ProbeError("vm_stat printed no page size")

    wanted = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
    missing = [key for key in wanted if key not in counters]
    if missing:
        raise ProbeError(f"vm_stat did not report {missing}")
    available = sum(counters[key] for key in wanted) * page_size

    try:
        total_text = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProbeError(f"could not read sysctl hw.memsize: {exc}") from exc
    if total_text.returncode != 0:
        raise ProbeError(
            f"sysctl -n hw.memsize exited {total_text.returncode}: "
            f"{total_text.stderr.strip()}"
        )
    return available, int(total_text.stdout.strip())


def probe_system_memory() -> tuple[int, int]:
    try:
        return physical_memory_figures()
    except NoViableBackend as exc:
        raise ProbeError(str(exc)) from exc


def read_windows_state(desktop_allowance_bytes: int) -> AcceleratorState:
    if nvidia_smi_path() is None:
        available, total = probe_system_memory()
        return AcceleratorState(
            backend=LLAMA_WINDOWS,
            total_bytes=total,
            free_bytes=available,
            compute_apps=(),
            detail=(
                f"{available / GIB:.1f} GiB of {total / GIB:.1f} GiB system "
                "memory available; no NVIDIA driver on this host, so the pool "
                "is RAM and llama.cpp runs on the CPU"
            ),
        )
    apps = tuple(probe_compute_apps())
    free, total = probe_vram()
    return AcceleratorState(
        backend=LLAMA_WINDOWS,
        total_bytes=total,
        free_bytes=free,
        compute_apps=apps,
        detail=(
            f"{free / GIB:.1f} GiB free of {total / GIB:.1f} GiB, "
            f"{len(apps)} compute app(s), desktop allowance "
            f"{desktop_allowance_bytes / GIB:.1f} GiB"
        ),
    )


def read_state(backend_kind: str, desktop_allowance_bytes: int) -> AcceleratorState:
    if backend_kind == CUDA_LINUX:
        apps = tuple(probe_compute_apps())
        free, total = probe_vram()
        return AcceleratorState(
            backend=backend_kind,
            total_bytes=total,
            free_bytes=free,
            compute_apps=apps,
            detail=(
                f"{free / GIB:.1f} GiB free of {total / GIB:.1f} GiB, "
                f"{len(apps)} compute app(s), desktop allowance "
                f"{desktop_allowance_bytes / GIB:.1f} GiB"
            ),
        )
    if backend_kind == MLX_DARWIN:
        available, total = probe_unified_memory()
        return AcceleratorState(
            backend=backend_kind,
            total_bytes=total,
            free_bytes=available,
            compute_apps=(),
            detail=(
                f"{available / GIB:.1f} GiB of {total / GIB:.1f} GiB unified memory "
                "available (free + inactive + speculative + purgeable)"
            ),
        )
    if backend_kind == LLAMA_WINDOWS:
        return read_windows_state(desktop_allowance_bytes)
    raise ProbeError(
        f"{backend_kind!r} is not a Crucible backend; the backends are "
        f"{', '.join(repr(kind) for kind in BACKEND_KINDS)}"
    )


def unattributed_bytes(
    state: AcceleratorState,
    desktop_allowance_bytes: int,
    reclaimable_bytes: int = 0,
) -> int:
    accounted = sum(
        app.used_bytes for app in state.compute_apps if app.used_bytes is not None
    )
    return max(
        0, state.used_bytes - accounted - desktop_allowance_bytes - reclaimable_bytes
    )


def refuse_if_larger_than_host(
    *, model_id: str, need_bytes: int, host_total_bytes: int, host_name: str
) -> None:
    if need_bytes <= host_total_bytes:
        return
    raise ApiError(
        409,
        "insufficient_memory",
        f"cannot load {model_id!r} on this host, ever: it needs "
        f"{need_bytes / GIB:.1f} GiB and {host_name} has "
        f"{host_total_bytes / GIB:.1f} GiB in total",
        {
            "model": model_id,
            "needed_bytes": need_bytes,
            "total_bytes": host_total_bytes,
            "free_bytes": None,
        },
    )


def refuse_if_card_lacks(*, model_id: str, spec: Any, card: Any) -> None:
    needs = card_needs(spec)
    missing = [need for need in needs if card.has(need) is False]
    if not missing:
        return
    reasons = "; ".join(
        f"{need}: {card.measured_detail.get(need) or 'measured as unavailable'}"
        for need in missing
    )
    raise ApiError(
        409,
        "card_lacks_feature",
        f"cannot load {model_id!r} on this card: Crucible measured "
        f"{card.name} on {card.measured_at or 'an earlier run'} and what this "
        f"model's engine needs did not work there ({reasons}). "
        "`crucible ladder` measures it again",
        {
            "model": model_id,
            "needs": missing,
            "compute_capability": card.compute_capability,
            "card": card.name,
            "measured_at": card.measured_at,
        },
    )


def guard(
    backend_kind: str,
    *,
    model_id: str,
    need_bytes: int,
    owned_pids: frozenset[int] = frozenset(),
    desktop_allowance_bytes: int = 0,
    reclaimable_bytes: int = 0,
) -> AcceleratorState:
    try:
        state = read_state(backend_kind, desktop_allowance_bytes)
    except ProbeError as exc:
        raise ApiError(
            409,
            "accelerator_unreadable",
            f"cannot load {model_id!r}: {exc}",
        ) from None

    if backend_kind == CUDA_LINUX and any(
        app.pid not in owned_pids for app in state.compute_apps
    ):
        try:
            owned_pids = expand_owned_pids(owned_pids, probe_process_table())
        except ProbeError as exc:
            raise ApiError(
                409,
                "accelerator_unreadable",
                f"cannot load {model_id!r}: {exc}",
            ) from None

    not_ours = [app for app in state.compute_apps if app.pid not in owned_pids]

    if backend_kind == LLAMA_WINDOWS:
        holders = [app for app in not_ours if is_llama_server(app.name)]
        opening = (
            f"cannot load {model_id!r}: a llama-server this Crucible did not "
            "start is still on the accelerator — "
        )
        closing = (
            ". It is an engine child left behind by an earlier run; stop it and "
            "load again. Crucible never evicts another process."
        )
    else:
        holders = [
            app
            for app in not_ours
            if app.used_bytes is None or app.used_bytes > FOREIGN_PROCESS_FLOOR_BYTES
        ]
        opening = f"cannot load {model_id!r}: the accelerator is held by "
        closing = ". Crucible never evicts another process."

    if holders:
        raise ApiError(
            409,
            "accelerator_busy",
            *_left_behind(
                opening + "; ".join(app.describe() for app in holders) + closing,
                {
                    "model": model_id,
                    "processes": [
                        {"pid": app.pid, "name": app.name, "used_bytes": app.used_bytes}
                        for app in holders
                    ],
                },
            ),
        )

    stray = (
        unattributed_bytes(state, desktop_allowance_bytes, reclaimable_bytes)
        if backend_kind == CUDA_LINUX
        else 0
    )
    if stray > FOREIGN_PROCESS_FLOOR_BYTES:
        raise ApiError(
            409,
            "accelerator_busy",
            *_left_behind(
                f"cannot load {model_id!r}: {stray / GIB:.1f} GiB of the "
                f"{state.total_bytes / GIB:.1f} GiB card is in use by a process "
                "this host's driver will not name (under WSL2 the compute-app "
                "list is empty even for processes inside the same VM). Crucible "
                "never evicts another process.",
                {
                    "model": model_id,
                    "unattributed_bytes": stray,
                    "used_bytes": state.used_bytes,
                    "desktop_allowance_bytes": desktop_allowance_bytes,
                },
            ),
        )

    if backend_kind == MLX_DARWIN:
        room = available_bytes(state.total_bytes, desktop_allowance_bytes)
        if need_bytes > room:
            raise ApiError(
                409,
                "insufficient_memory",
                f"cannot load {model_id!r}: it needs {need_bytes / GIB:.1f} GiB and "
                f"this Mac gives a model {room / GIB:.1f} GiB "
                f"({state.total_bytes / GIB:.1f} GiB unified memory less the "
                f"{desktop_allowance_bytes / GIB:.1f} GiB desktop allowance)",
                {
                    "model": model_id,
                    "needed_bytes": need_bytes,
                    "room_bytes": room,
                    "free_bytes": state.free_bytes,
                    "total_bytes": state.total_bytes,
                    "desktop_allowance_bytes": desktop_allowance_bytes,
                },
            )
        return state

    effective_free = state.free_bytes + reclaimable_bytes
    if effective_free < need_bytes:
        reclaim = (
            f" (plus {reclaimable_bytes / GIB:.1f} GiB Crucible's own resident "
            "engine would give back)"
            if reclaimable_bytes
            else ""
        )
        raise ApiError(
            409,
            "insufficient_memory",
            f"cannot load {model_id!r}: it needs {need_bytes / GIB:.1f} GiB and "
            f"this host has {state.free_bytes / GIB:.1f} GiB free of "
            f"{state.total_bytes / GIB:.1f} GiB{reclaim}",
            {
                "model": model_id,
                "needed_bytes": need_bytes,
                "free_bytes": state.free_bytes,
                "reclaimable_bytes": reclaimable_bytes,
                "total_bytes": state.total_bytes,
                "processes": [
                    {"pid": app.pid, "name": app.name, "used_bytes": app.used_bytes}
                    for app in not_ours
                ],
            },
        )
    return state
