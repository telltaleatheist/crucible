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
from dataclasses import asdict, dataclass, field
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


#: WHAT A CARD'S COMPUTE CAPABILITY SAYS IT CAN RUN (fresh-install #48,
#: 2026-09-26). Owen, on kylies-pc's GTX 1660 SUPER: *"this gpu is also not
#: capable of most things, so itll be useful to figure out how we can know what
#: its capable of without direct measurements."* Memory was the only axis until
#: then, and a card with room and without bf16 would have been told "yes" and
#: then failed at the engine's first line.
#:
#: Each floor is the ENGINE'S OWN, read in the pinned llm env (vLLM 0.29.0,
#: torch 2.13.0+cu130) rather than recalled, so a card is judged by the number
#: the code that runs on it will judge it by:
#:
#:   bf16                8.0   vLLM `platforms/cuda.py` L237-246 (`supported_dtypes`:
#:                             "Pascal, Volta and Turing NVIDIA GPUs, BF16 is not
#:                             supported") and L622-640 (`check_if_supports_dtype`
#:                             raises below 80); torch `cuda/__init__.py` L244
#:                             (`is_bf16_supported`: native only at major >= 8,
#:                             emulated below)
#:   flash_attention_2   8.0   vLLM `v1/attention/backends/flash_attn.py` L198-199
#:                             (`capability >= DeviceCapability(8, 0)`)
#:   fp8                 8.9   vLLM `platforms/cuda.py` L571-572 (`supports_fp8`:
#:                             `has_device_capability(89)`)
#:
#: A FEATURE NOTHING NEEDS IS STILL REPORTED. Nothing in this build's catalog
#: needs FlashAttention 2 (vLLM falls back to its Triton attention, which
#: `triton_attn.py` L373-374 allows on any capability) or fp8 (no cuda-linux
#: block runs fp8 weights or fp8 KV). They are on `crucible capability` and
#: `crucible doctor` because they are what a person asking "what can this card
#: do" is asking, and the day a manifest needs one the fact is already here.
BF16 = "bf16"
FLASH_ATTENTION_2 = "flash_attention_2"
FP8 = "fp8"

#: `(feature, (major, minor), what it is)`, in report order. The one owner of
#: every floor above: `card_features` reads it and nothing else types one.
FEATURE_FLOORS: tuple[tuple[str, tuple[int, int], str], ...] = (
    (BF16, (8, 0), "bfloat16 arithmetic"),
    (FLASH_ATTENTION_2, (8, 0), "FlashAttention 2 kernels"),
    (FP8, (8, 9), "fp8 weights and KV cache"),
)


def parse_compute_capability(text: str) -> tuple[int, int] | None:
    """`"7.5"` as `(7, 5)`, or None for anything that is not `<major>.<minor>`."""
    major, dot, minor = text.strip().partition(".")
    if dot != "." or not major.isdigit() or not minor.isdigit():
        return None
    return int(major), int(minor)


def sm_name(compute_capability: str) -> str:
    """`"7.5"` as `"sm_75"`, the name a person reads on a spec sheet and in a
    kernel's error, so a refusal can use the same words."""
    parsed = parse_compute_capability(compute_capability)
    if parsed is None:
        return compute_capability
    return f"sm_{parsed[0]}{parsed[1]}"


def card_features(compute_capability: str | None) -> dict[str, bool] | None:
    """Every feature in `FEATURE_FLOORS`, true where this card meets its floor.

    None when the capability is unknown — a card nvidia-smi would not report it
    for, or no card at all (a Mac, a CPU-only Windows box) — and None is not
    "none of them": a caller that cannot tell the two apart would refuse bf16
    on an M1 Ultra.
    """
    if compute_capability is None:
        return None
    parsed = parse_compute_capability(compute_capability)
    if parsed is None:
        return None
    return {name: parsed >= floor for name, floor, _what in FEATURE_FLOORS}


def feature_floor(feature: str) -> str:
    """The floor a feature needs, as `"8.0"`. A name not in the table is refused."""
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
    #: `nvidia-smi --query-gpu=compute_cap`, as the driver prints it (`"7.5"`),
    #: or None where there is no NVIDIA card or the driver would not say. The
    #: one owner of "what generation is this card"; `card_features` is what it
    #: means.
    compute_capability: str | None = None

    def features(self) -> dict[str, bool] | None:
        return card_features(self.compute_capability)


#: TENSOR CORES, which the compute capability alone does NOT answer. Volta
#: (7.0) is where they arrive, and every NVIDIA part at or above it has them
#: EXCEPT the Turing GTX 16 family (TU116/TU117: GTX 1650, 1660, 1660 SUPER,
#: 1660 Ti), which NVIDIA shipped at sm_75 with the tensor cores removed and
#: dedicated FP16 units in their place. kylies-pc's card is exactly that part,
#: so "sm_70 and up" would have said yes about the one card #48 is about.
#: Reported, never required: nothing in this build refuses a card for lacking
#: them; they are the difference between fp16 matmuls that are fast and ones
#: that merely work.
TENSOR_CORES = "tensor_cores"
TENSOR_CORE_FLOOR = (7, 0)
#: The marketing prefix NVIDIA gives the TU116/TU117 parts, as nvidia-smi
#: prints the name (`NVIDIA GeForce GTX 1660 SUPER`).
NO_TENSOR_CORE_PARTS: tuple[str, ...] = ("GTX 16",)


def has_tensor_cores(compute_capability: str | None, name: str) -> bool | None:
    """True, False, or None where the generation is unknown."""
    if compute_capability is None:
        return None
    parsed = parse_compute_capability(compute_capability)
    if parsed is None:
        return None
    if parsed < TENSOR_CORE_FLOOR:
        return False
    return not any(part in name for part in NO_TENSOR_CORE_PARTS)


#: FACTS THE LADDER MEASURES (docs/PROPOSAL-GPU-LADDER.md, built 2026-09-26):
#: what no spec sheet answers and only a run on this card, in this env, can.
#: Owen, 2026-09-26: *"our measurement tool should determine how much space is
#: available, whether tensors are available, cuda graphs, vllm, etc. and
#: install the best the user can use"*.
#:
#:   cuda_graphs   a CUDA graph captured and replayed in the llm env's torch.
#:                 False means vLLM is started `--enforce-eager` (slower, and
#:                 it runs), never that vLLM is refused.
#:   vllm          vLLM started a model in the llm env on this card and
#:                 answered one request. False REFUSES every vLLM block here,
#:                 with the measured first error line.
CUDA_GRAPHS = "cuda_graphs"
VLLM_STARTS = "vllm"
MEASURED_FEATURES: tuple[str, ...] = (CUDA_GRAPHS, VLLM_STARTS)


@dataclass(frozen=True)
class CardFacts:
    """Everything capability and the engines may ask about THIS card.

    Two sources, one answer. DECLARED facts come off the compute capability
    (`FEATURE_FLOORS`, `has_tensor_cores`) and need nobody to run anything;
    MEASURED facts come off the ladder's record (`crucible/ladder.py`) and
    exist only once it has run. A fact nobody knows is None — "unknown" — and
    never False: an unmeasured card is not a card that failed.
    """

    name: str
    compute_capability: str | None
    #: `{feature: passed}` for each `MEASURED_FEATURES` entry the ladder has
    #: an answer for on this card; absent means not measured.
    measured: dict[str, bool] = field(default_factory=dict)
    #: The measured facts' own sentences — the first error line of a failed
    #: rung — so a refusal built on one can quote it.
    measured_detail: dict[str, str] = field(default_factory=dict)
    #: When the ladder's record was taken, or None when there is none.
    measured_at: str | None = None

    def has(self, feature: str) -> bool | None:
        """True, False, or None (unknown) for one feature."""
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
        """Every feature this build knows, in report order, None where unknown."""
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
    """The card as the probe alone describes it: no ladder record consulted."""
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


def probe_compute_capability() -> str | None:
    """GPU 0's compute capability (`"7.5"`), or None if the driver will not say.

    A QUERY OF ITS OWN, not a third column on `probe_nvidia_smi`'s: a driver
    that predates the `compute_cap` field rejects the WHOLE query as an invalid
    field, and a detection that failed on it would turn "this card's generation
    is unknown" into "this host has no backend". Unknown is reported as unknown
    (`card_features` returns None) and never as a card that lacks everything.

    The bootstrap installer asks the same field in shell before any Python
    exists (`sdk/bootstrap/scripts/install.sh`'s compute-capability floor);
    this is the Python side's one reading of it.
    """
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
        gpu=Gpu(
            vendor="nvidia",
            name=name,
            vram_bytes=vram_bytes,
            compute_capability=probe_compute_capability(),
        ),
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
            # Apple Silicon has unified memory: vram_bytes is the whole machine's RAM,
            # not a dedicated pool.
            gpu=Gpu(vendor="apple", name=chip, vram_bytes=memsize),
            detail=f"mlx {mlx_version}, unified memory",
        )

    raise NoViableBackend(
        f"{system} is not a Crucible backend; supported hosts are Linux with an "
        "NVIDIA card (cuda-linux) and Apple Silicon macOS (mlx-darwin)"
    )
