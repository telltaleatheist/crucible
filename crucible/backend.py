"""Backend detection.

A backend is a (platform, accelerator) pair Crucible can run on. The list is explicit
and short (DESIGN.md section 2):

    cuda-linux   Linux with an NVIDIA card (on Windows: the Linux server inside WSL2)
    mlx-darwin   Apple Silicon Mac

Anything else raises NoViableBackend with the reason. There is no CPU fallback and
there is no Windows code path.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any

from .errors import NoViableBackend

CUDA_LINUX = "cuda-linux"
MLX_DARWIN = "mlx-darwin"

#: Windows, natively, with llama.cpp as the engine and GGUF as the weights.
#:
#: **Windows IS a backend** (PHASE15-HOST.md section 0's AMENDED block, Owen
#: 2026-09-14: *"the windows side should still host GPU jobs even if WSL isnt
#: present/workable … just like it runs from the mac side"*). Structurally this
#: is what `mlx-darwin` is: a per-model engine child the server spawns, leases,
#: settles and kills. What it is NOT is a second Crucible, a relay, or a
#: stopgap — it is one backend of three, and the WSL engine is still the better
#: one on any Windows machine that can run it (vLLM/SGLang, parallel page
#: reading, and the five Python job types this backend will never have).
LLAMA_WINDOWS = "llama-windows"

#: Every backend kind this build knows, in no particular order. Used where a
#: refusal has to list them, so a fourth is added in one place.
BACKEND_KINDS: tuple[str, ...] = (CUDA_LINUX, MLX_DARWIN, LLAMA_WINDOWS)

# WSL2 ships the driver shim here and does not always put it on PATH — notably not
# under `wsl.exe -d <distro> --exec bash -c ...`, which starts a non-login shell.
# This is a known location, not a guess.
WSL_NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"

#: Why a config's backend and the host it is on must agree, on every platform.
#: PHASE15-HOST.md section 3.5: *"a backend runs where its engine runs and
#: nowhere else"*. `llama-windows` off win32 is as wrong as `cuda-linux` on it.
def backend_not_here(recorded: str, detected: str, platform_name: str) -> str:
    return (
        f"this config records backend {recorded!r} and this host is "
        f"{platform_name}, which runs {detected!r}. A backend runs where its "
        "engine runs and nowhere else: llama-windows is llama.cpp on Windows, "
        "cuda-linux is vLLM/SGLang on Linux (inside WSL2 on a Windows box), "
        "mlx-darwin is mlx on Apple Silicon"
    )


#: **Superseded, and kept because the sentence is still true of ONE thing.**
#: Until 2026-09-14 this was the whole of what Crucible said about Windows, and
#: `main()` printed it before parsing a single verb. Section 0's AMENDED block
#: overrules it: `llama-windows` is a backend and every verb runs on win32.
#: What survives is the part that is still a fact — vLLM and SGLang have no
#: win32 build — so it is now the sentence a `cuda-linux` CONFIG gets when it
#: is found on a Windows host, and nothing else prints it.
WINDOWS_REFUSAL = (
    "the cuda-linux backend runs inside WSL2 on Windows and has no Windows "
    "code path (vLLM and SGLang do not run on win32). Natively, Windows runs "
    "the llama-windows backend — `crucible init` records it."
)


@dataclass(frozen=True)
class Gpu:
    vendor: str
    name: str
    vram_bytes: int


@dataclass(frozen=True)
class Backend:
    kind: str
    platform: str
    arch: str
    gpu: Gpu
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def nvidia_smi_path() -> str | None:
    """Where nvidia-smi is on this host, or None. PATH first, then the WSL location."""
    found = shutil.which("nvidia-smi")
    if found is not None:
        return found
    if os.path.exists(WSL_NVIDIA_SMI) and os.access(WSL_NVIDIA_SMI, os.X_OK):
        return WSL_NVIDIA_SMI
    return None


def probe_nvidia_smi() -> tuple[str, int]:
    """Return (gpu name, total VRAM in bytes) from nvidia-smi, or raise NoViableBackend."""
    exe = nvidia_smi_path()
    if exe is None:
        raise NoViableBackend(
            "no nvidia-smi on this Linux host (looked on PATH and at "
            f"{WSL_NVIDIA_SMI})"
        )
    try:
        completed = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except OSError as exc:
        raise NoViableBackend(f"could not execute {exe}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise NoViableBackend(f"{exe} did not answer within 20s") from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise NoViableBackend(
            f"{exe} exited {completed.returncode}: {stderr or 'no output'}"
        )

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise NoViableBackend(f"{exe} reported no GPUs")

    first = lines[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) != 2:
        raise NoViableBackend(f"could not parse nvidia-smi output line: {first!r}")
    name, mib_text = parts
    try:
        mib = int(mib_text)
    except ValueError as exc:
        raise NoViableBackend(
            f"could not parse nvidia-smi memory.total {mib_text!r} as an integer"
        ) from exc
    return name, mib * 1024 * 1024


def probe_mlx() -> str:
    """Return the installed mlx version, or raise NoViableBackend.

    The probe runs in a subprocess so that importing Metal does not happen inside the
    API server process just to answer "which backend am I".
    """
    code = (
        "import mlx.core\n"
        "import importlib.metadata as m\n"
        "print(m.version('mlx'))\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
        )
    except OSError as exc:
        raise NoViableBackend(f"could not run the mlx probe: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise NoViableBackend("the mlx probe did not finish within 60s") from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip().splitlines()
        last = stderr[-1] if stderr else "no output"
        raise NoViableBackend(f"`import mlx.core` failed in {sys.executable}: {last}")
    version = completed.stdout.strip()
    if not version:
        raise NoViableBackend("the mlx probe printed no version")
    return version


def _sysctl(name: str) -> str:
    try:
        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NoViableBackend(f"could not read sysctl {name}: {exc}") from exc
    if completed.returncode != 0:
        raise NoViableBackend(
            f"sysctl -n {name} exited {completed.returncode}: {completed.stderr.strip()}"
        )
    value = completed.stdout.strip()
    if not value:
        raise NoViableBackend(f"sysctl -n {name} returned nothing")
    return value


def physical_memory_figures() -> tuple[int, int]:
    """(available, total) bytes of this Windows machine's RAM, from the OS.

    `GlobalMemoryStatusEx` through ctypes, for `crucible/interfaces.py`'s
    reason: it is the question the OS answers, it is stdlib, and the
    alternative (`psutil`) is a dependency for a fact the C library already
    states. It is the POOL FIGURE for a CPU-only `llama-windows` host — a
    GGUF on the CPU allocates from system RAM exactly as a model on a Mac
    allocates from unified memory, so the capability arithmetic reads the same
    shape on both.

    BOTH figures, from ONE call, because two callers want different halves of
    the same answer and asking twice would let them disagree: capability sizes
    a model against the TOTAL (what this machine could ever hold) and the
    accelerator guard sizes a load against the AVAILABLE (what it can hold
    right now). `ullAvailPhys` is the free figure `nvidia-smi
    --query-gpu=memory.free` is on a card.
    """
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
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

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise NoViableBackend(
            "GlobalMemoryStatusEx would not answer, so this host cannot say "
            "how much memory it has — and a capability decided from a guessed "
            "pool is a capability nobody can trust"
        )
    return int(status.ullAvailPhys), int(status.ullTotalPhys)


def physical_memory_bytes() -> int:
    """This Windows machine's installed RAM. The TOTAL half of the figures."""
    return physical_memory_figures()[1]


def detect_windows(arch: str) -> Backend:
    """The `llama-windows` backend, with or without a card.

    PHASE15-HOST.md sections 0 and 3.10. **Nothing here refuses.** Owen: *"a
    crucible server will run on absolutely anything"* — a machine with no
    NVIDIA card runs the CPU build of llama.cpp, slowly, and the capability
    row says so in words rather than turning the class off. So the two answers
    differ only in which pool is measured and which build the engine subject
    fetches.

    `nvidia-smi` is asked, not assumed: the Windows driver is the right
    authority for a Windows-native engine (unlike the WSL question, where a
    Windows driver says nothing about whether passthrough works — PHASE13 5.5).
    """
    try:
        name, vram_bytes = probe_nvidia_smi()
    except NoViableBackend:
        # NOT an error. A CPU-only machine is a machine this backend serves,
        # and the pool it serves from is RAM.
        return Backend(
            kind=LLAMA_WINDOWS,
            platform="windows",
            arch=arch,
            gpu=Gpu(vendor="cpu", name="cpu", vram_bytes=physical_memory_bytes()),
            detail="llama.cpp cpu build; no NVIDIA card answered nvidia-smi",
        )
    return Backend(
        kind=LLAMA_WINDOWS,
        platform="windows",
        arch=arch,
        gpu=Gpu(vendor="nvidia", name=name, vram_bytes=vram_bytes),
        detail=f"llama.cpp cuda build; nvidia-smi at {nvidia_smi_path()}",
    )


def detect_backend() -> Backend:
    """Detect this host's backend or raise NoViableBackend(reason)."""
    system = sys.platform
    arch = platform.machine()

    if system == "win32":
        return detect_windows(arch)

    if system == "linux":
        name, vram_bytes = probe_nvidia_smi()
        return Backend(
            kind=CUDA_LINUX,
            platform="linux",
            arch=arch,
            gpu=Gpu(vendor="nvidia", name=name, vram_bytes=vram_bytes),
            detail=f"nvidia-smi at {nvidia_smi_path()}",
        )

    if system == "darwin":
        if arch != "arm64":
            raise NoViableBackend(
                f"macOS on {arch} is not a Crucible backend; Apple Silicon (arm64) only"
            )
        mlx_version = probe_mlx()
        chip = _sysctl("machdep.cpu.brand_string")
        memsize = int(_sysctl("hw.memsize"))
        return Backend(
            kind=MLX_DARWIN,
            platform="darwin",
            arch=arch,
            # Apple Silicon has unified memory: vram_bytes is the whole machine's RAM,
            # not a dedicated pool.
            gpu=Gpu(vendor="apple", name=chip, vram_bytes=memsize),
            detail=f"mlx {mlx_version}, unified memory",
        )

    raise NoViableBackend(
        f"{system} is not a Crucible backend; supported hosts are Linux with an "
        "NVIDIA card (cuda-linux) and Apple Silicon macOS (mlx-darwin)"
    )
