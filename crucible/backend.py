from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import NoViableBackend

CUDA_LINUX = "cuda-linux"
MLX_DARWIN = "mlx-darwin"

LLAMA_WINDOWS = "llama-windows"

BACKEND_KINDS: tuple[str, ...] = (CUDA_LINUX, MLX_DARWIN, LLAMA_WINDOWS)

WSL_NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"

def backend_not_here(recorded: str, detected: str, platform_name: str) -> str:
    return (
        f"this config records backend {recorded!r} and this host is "
        f"{platform_name}, which runs {detected!r}. A backend runs where its "
        "engine runs and nowhere else: llama-windows is llama.cpp on Windows, "
        "cuda-linux is vLLM/SGLang on Linux (inside WSL2 on a Windows box), "
        "mlx-darwin is mlx on Apple Silicon"
    )


WINDOWS_REFUSAL = (
    "the cuda-linux backend runs inside WSL2 on Windows and has no Windows "
    "code path (vLLM and SGLang do not run on win32). Natively, Windows runs "
    "the llama-windows backend — `crucible init` records it."
)


BF16 = "bf16"
FLASH_ATTENTION_2 = "flash_attention_2"
FP8 = "fp8"

FEATURE_FLOORS: tuple[tuple[str, tuple[int, int], str], ...] = (
    (BF16, (8, 0), "bfloat16 arithmetic"),
    (FLASH_ATTENTION_2, (8, 0), "FlashAttention 2 kernels"),
    (FP8, (8, 9), "fp8 weights and KV cache"),
)


def parse_compute_capability(text: str) -> tuple[int, int] | None:
    major, dot, minor = text.strip().partition(".")
    if dot != "." or not major.isdigit() or not minor.isdigit():
        return None
    return int(major), int(minor)


def sm_name(compute_capability: str) -> str:
    parsed = parse_compute_capability(compute_capability)
    if parsed is None:
        return compute_capability
    return f"sm_{parsed[0]}{parsed[1]}"


def card_features(compute_capability: str | None) -> dict[str, bool] | None:
    if compute_capability is None:
        return None
    parsed = parse_compute_capability(compute_capability)
    if parsed is None:
        return None
    return {name: parsed >= floor for name, floor, _what in FEATURE_FLOORS}


def feature_floor(feature: str) -> str:
    for name, (major, minor), _what in FEATURE_FLOORS:
        if name == feature:
            return f"{major}.{minor}"
    raise KeyError(
        f"{feature!r} is not a card feature this build knows; it knows "
        f"{[name for name, _floor, _what in FEATURE_FLOORS]}"
    )


@dataclass(frozen=True)
class Gpu:
    vendor: str
    name: str
    vram_bytes: int
    compute_capability: str | None = None

    def features(self) -> dict[str, bool] | None:
        return card_features(self.compute_capability)


TENSOR_CORES = "tensor_cores"
TENSOR_CORE_FLOOR = (7, 0)
NO_TENSOR_CORE_PARTS: tuple[str, ...] = ("GTX 16",)


def has_tensor_cores(compute_capability: str | None, name: str) -> bool | None:
    if compute_capability is None:
        return None
    parsed = parse_compute_capability(compute_capability)
    if parsed is None:
        return None
    if parsed < TENSOR_CORE_FLOOR:
        return False
    return not any(part in name for part in NO_TENSOR_CORE_PARTS)


CUDA_GRAPHS = "cuda_graphs"
VLLM_STARTS = "vllm"
MEASURED_FEATURES: tuple[str, ...] = (CUDA_GRAPHS, VLLM_STARTS)


@dataclass(frozen=True)
class CardFacts:
    name: str
    compute_capability: str | None
    measured: dict[str, bool] = field(default_factory=dict)
    measured_detail: dict[str, str] = field(default_factory=dict)
    measured_at: str | None = None

    def has(self, feature: str) -> bool | None:
        if feature in MEASURED_FEATURES:
            return self.measured.get(feature)
        if feature == TENSOR_CORES:
            return has_tensor_cores(self.compute_capability, self.name)
        declared = card_features(self.compute_capability)
        if declared is None:
            return None
        if feature not in declared:
            raise KeyError(
                f"{feature!r} is not a card feature this build knows; it knows "
                f"{[n for n, _f, _w in FEATURE_FLOORS]}, {TENSOR_CORES!r} and "
                f"{list(MEASURED_FEATURES)}"
            )
        return declared[feature]

    def features(self) -> dict[str, bool | None]:
        names = [n for n, _f, _w in FEATURE_FLOORS] + [TENSOR_CORES, *MEASURED_FEATURES]
        return {name: self.has(name) for name in names}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "compute_capability": self.compute_capability,
            "sm": (
                None
                if self.compute_capability is None
                else sm_name(self.compute_capability)
            ),
            "features": self.features(),
            "floors": {n: feature_floor(n) for n, _f, _w in FEATURE_FLOORS},
            "measured_at": self.measured_at,
            "measured_detail": dict(self.measured_detail),
        }


def declared_card(gpu: Gpu) -> CardFacts:
    return CardFacts(name=gpu.name, compute_capability=gpu.compute_capability)


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
    found = shutil.which("nvidia-smi")
    if found is not None:
        return found
    if os.path.exists(WSL_NVIDIA_SMI) and os.access(WSL_NVIDIA_SMI, os.X_OK):
        return WSL_NVIDIA_SMI
    return None


def probe_nvidia_smi() -> tuple[str, int]:
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


def probe_compute_capability() -> str | None:
    exe = nvidia_smi_path()
    if exe is None:
        return None
    try:
        completed = subprocess.run(
            [exe, "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines or parse_compute_capability(lines[0]) is None:
        return None
    return lines[0]


def probe_mlx() -> str:
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
    return physical_memory_figures()[1]


def detect_windows(arch: str) -> Backend:
    try:
        name, vram_bytes = probe_nvidia_smi()
    except NoViableBackend:
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
        gpu=Gpu(
            vendor="nvidia",
            name=name,
            vram_bytes=vram_bytes,
            compute_capability=probe_compute_capability(),
        ),
        detail=f"llama.cpp cuda build; nvidia-smi at {nvidia_smi_path()}",
    )


def detect_backend() -> Backend:
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
            gpu=Gpu(
                vendor="nvidia",
                name=name,
                vram_bytes=vram_bytes,
                compute_capability=probe_compute_capability(),
            ),
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
            gpu=Gpu(vendor="apple", name=chip, vram_bytes=memsize),
            detail=f"mlx {mlx_version}, unified memory",
        )

    raise NoViableBackend(
        f"{system} is not a Crucible backend; supported hosts are Linux with an "
        "NVIDIA card (cuda-linux) and Apple Silicon macOS (mlx-darwin)"
    )
