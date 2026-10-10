"""What an audio model keeps in host memory while it serves, weighed against this machine's.

Victoria's RTX 3070 laptop, 2026-10-10: a 15-track album on YuE2 under `[audio] low_vram`,
in a WSL guest of 15.8 GB, and the kernel OOM-killed the worker on track 12. The card
fit was weighed (crucible/lowvram.py) and the host never was, yet YuE2 keeps its whole
backbone in host memory while it serves: yue2-infer parks it there while the VAE decodes,
and low_vram keeps the half a stage does not use there all the time. That growth was a
worker fault, fixed at its owner (yue2_worker.HostHomes); this module is the part that
stays true after it: a model that declares `host_memory_bytes_estimate` says what it
keeps in host memory, and doctor names the model and both figures when this machine has
less than that in all.

Only on cuda-linux: the server and its workers run in the same Linux (on Windows, the
WSL guest), so /proc/meminfo's MemTotal is the memory the worker has. On Apple silicon
the card and host memory are one pool, which the card's own figure already weighs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .backend import CUDA_LINUX
from .memorybudget import gib_text

MEMINFO = "/proc/meminfo"

# Where the memory of a WSL guest is set; the server on Windows runs in one.
WSL_MEMORY_FIX = (
    "give this machine more memory - on Windows, raise `memory` under [wsl2] in "
    "%UserProfile%\\.wslconfig and restart WSL - or make songs on a machine with more"
)


def memory_total_bytes(meminfo: str = MEMINFO) -> int:
    """MemTotal from /proc/meminfo, in bytes."""
    with open(meminfo, encoding="ascii") as handle:
        for line in handle:
            key, _, value = line.partition(":")
            if key == "MemTotal":
                number, unit = value.split()
                if unit != "kB":
                    raise ValueError(f"{meminfo} gives MemTotal in {unit!r}, not kB")
                return int(number) * 1024
    raise ValueError(f"{meminfo} has no MemTotal line")


@dataclass(frozen=True)
class HostShort:
    """One audio model that keeps more in host memory than this machine has."""

    id: str
    host_bytes: int
    total_bytes: int
    note: str

    @property
    def words(self) -> str:
        return (
            f"{self.id} keeps {gib_text(self.host_bytes)} in host memory while it makes "
            f"audio, and this machine has {gib_text(self.total_bytes)} in all"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "host_bytes": self.host_bytes,
            "total_bytes": self.total_bytes,
            "words": self.words,
        }


def short_of_host(
    manifests: Iterable[Any], backend_kind: str, total_bytes: int
) -> tuple[HostShort, ...]:
    """The audio models on this backend whose declared host need is above `total_bytes`."""
    found = []
    for manifest in manifests:
        if not manifest.supports(backend_kind):
            continue
        spec = manifest.spec(backend_kind)
        need = spec.host_memory_bytes_estimate
        if need is not None and need > total_bytes:
            found.append(HostShort(manifest.id, need, total_bytes, spec.host_memory_note))
    return tuple(sorted(found, key=lambda short: short.id))


def weighed_here(backend_kind: str) -> bool:
    return backend_kind == CUDA_LINUX


__all__ = [
    "HostShort",
    "WSL_MEMORY_FIX",
    "memory_total_bytes",
    "short_of_host",
    "weighed_here",
]
