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

# WSL2 ships the driver shim here and does not always put it on PATH — notably not
# under `wsl.exe -d <distro> --exec bash -c ...`, which starts a non-login shell.
# This is a known location, not a guess.
WSL_NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"

WINDOWS_REFUSAL = (
    "Crucible runs inside WSL2 on Windows; it has no Windows code path "
    "(vLLM and SGLang do not run on win32)."
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


def detect_backend() -> Backend:
    """Detect this host's backend or raise NoViableBackend(reason)."""
    system = sys.platform
    arch = platform.machine()

    if system == "win32":
        raise NoViableBackend(WINDOWS_REFUSAL)

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
