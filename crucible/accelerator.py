"""The accelerator guard (PHASE2-LLM.md section 4).

Before an engine starts, Crucible looks at the card and refuses by name rather
than competing for it. There is no eviction of other people's processes, ever.

    cuda-linux   `nvidia-smi --query-compute-apps` — any process that is not one
                 of ours holding more than 1 GiB is `accelerator_busy`, named.
                 Then `nvidia-smi --query-gpu=memory.free` against the manifest's
                 estimate — short is `insufficient_memory`, naming both numbers.
    mlx-darwin   the same two questions asked of free unified memory.

Every probe is a module-level function so a test can replace it and assert on the
refusal instead of on the machine it happens to run on.

A measured limitation, stated rather than papered over
------------------------------------------------------
Under WSL2 the driver shim answers `--query-compute-apps` with an **empty list**
even while a process inside that same WSL2 VM holds 17 GB of the card (measured on
Owen's PC, 2026-09-12, against a running SGLang server). `memory.free` under WSL2
*is* accurate for the whole card. So on WSL2 the first check is blind and only the
second one protects Owen's work.

`unattributed_bytes` closes that hole without guessing: VRAM in use that no
compute app in the list accounts for, beyond `desktop_allowance_bytes` (the host
desktop's own graphics memory, a declared host fact in config.toml, not a fudge
factor), is refused as `accelerator_busy` naming the amount. On a headless Linux
box the allowance is 0 and the sum is exact.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from .backend import CUDA_LINUX, MLX_DARWIN, nvidia_smi_path
from .errors import ApiError, CrucibleError

GIB = 1024 ** 3

#: A process holding less than this is not "using the card" for our purposes —
#: a compositor or a video decoder, not somebody's job (PHASE2-LLM.md section 4).
FOREIGN_PROCESS_FLOOR_BYTES = 1 * GIB


class ProbeError(CrucibleError):
    """A probe could not answer. Never turns into "the card is free"."""


@dataclass(frozen=True)
class ComputeApp:
    pid: int
    name: str
    used_bytes: int | None  # None when the driver will not say (WDDM, permissions)

    def describe(self) -> str:
        if self.used_bytes is None:
            return f"pid {self.pid} ({self.name}, memory not reported)"
        return f"pid {self.pid} ({self.name}, {self.used_bytes / GIB:.1f} GiB)"


@dataclass(frozen=True)
class AcceleratorState:
    """What the guard saw. Reported by /v1/models's reason and by doctor."""

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


# ------------------------------------------------------------------- probes


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
    """Every CUDA compute process the driver will admit to."""
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
            # "[N/A]" / "[Insufficient Permissions]" — the driver is refusing to
            # say. That is not zero, and it is not an excuse to parse it as zero.
            used = None
        apps.append(ComputeApp(pid=pid, name=name, used_bytes=used))
    return apps


def probe_vram() -> tuple[int, int]:
    """(free bytes, total bytes) for GPU 0."""
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
    """(available bytes, total bytes) of Apple Silicon unified memory.

    "Available" is what macOS can hand a new allocation without swapping: free +
    inactive + speculative + purgeable pages. `Pages free` alone is meaningless on
    a Mac that has been up for a day — it is near zero by design.
    """
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


# -------------------------------------------------------------------- state


def read_state(backend_kind: str, desktop_allowance_bytes: int) -> AcceleratorState:
    """Look at the accelerator. Raises ProbeError; never guesses."""
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
    raise ProbeError(
        f"{backend_kind!r} is not a Crucible backend; the backends are "
        f"{CUDA_LINUX!r} and {MLX_DARWIN!r}"
    )


def unattributed_bytes(
    state: AcceleratorState,
    desktop_allowance_bytes: int,
    reclaimable_bytes: int = 0,
) -> int:
    """VRAM in use that no listed compute app accounts for, past the allowance.

    A compute app whose memory the driver will not report contributes nothing to
    the accounted total, so its usage lands here and is refused — which is the
    conservative direction.

    `reclaimable_bytes` is what Crucible's *own* resident engine holds and is
    about to give back. Under WSL2 that engine does not appear in the compute-app
    list either, so without this term Crucible's second load would refuse on its
    own first model.
    """
    accounted = sum(
        app.used_bytes for app in state.compute_apps if app.used_bytes is not None
    )
    return (
        state.used_bytes - accounted - desktop_allowance_bytes - reclaimable_bytes
    )


# -------------------------------------------------------------------- guard


def guard(
    backend_kind: str,
    *,
    model_id: str,
    need_bytes: int,
    owned_pids: frozenset[int] = frozenset(),
    desktop_allowance_bytes: int = 0,
    reclaimable_bytes: int = 0,
) -> AcceleratorState:
    """Refuse by name if this model must not start here. Returns what it saw.

    `reclaimable_bytes` is what Crucible's own resident engine will give back when
    it is unloaded to make room for this one; it counts as free.
    """
    try:
        state = read_state(backend_kind, desktop_allowance_bytes)
    except ProbeError as exc:
        raise ApiError(
            409,
            "accelerator_unreadable",
            f"cannot load {model_id!r}: {exc}",
        ) from None

    foreign = [
        app
        for app in state.compute_apps
        if app.pid not in owned_pids
        and (app.used_bytes is None or app.used_bytes > FOREIGN_PROCESS_FLOOR_BYTES)
    ]
    if foreign:
        raise ApiError(
            409,
            "accelerator_busy",
            f"cannot load {model_id!r}: the accelerator is held by "
            + "; ".join(app.describe() for app in foreign)
            + ". Crucible never evicts another process.",
            {
                "model": model_id,
                "processes": [
                    {"pid": app.pid, "name": app.name, "used_bytes": app.used_bytes}
                    for app in foreign
                ],
            },
        )

    # Only on a discrete card. On Apple Silicon "used unified memory" is the OS,
    # the browser and the editor — the machine doing its job, not a compute
    # process squatting on an accelerator. There the free figure is the whole
    # check, which is what section 4 asks for on mlx-darwin.
    stray = (
        unattributed_bytes(state, desktop_allowance_bytes, reclaimable_bytes)
        if backend_kind == CUDA_LINUX
        else 0
    )
    if stray > FOREIGN_PROCESS_FLOOR_BYTES:
        raise ApiError(
            409,
            "accelerator_busy",
            f"cannot load {model_id!r}: {stray / GIB:.1f} GiB of the "
            f"{state.total_bytes / GIB:.1f} GiB card is in use by a process this "
            "host's driver will not name (under WSL2 the compute-app list is "
            "empty even for processes inside the same VM). Crucible never evicts "
            "another process.",
            {
                "model": model_id,
                "unattributed_bytes": stray,
                "used_bytes": state.used_bytes,
                "desktop_allowance_bytes": desktop_allowance_bytes,
            },
        )

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
            },
        )
    return state
