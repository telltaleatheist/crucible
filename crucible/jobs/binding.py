from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from ..jobtypes import JobTypeSpec

if TYPE_CHECKING:
    from ..residency import Residency
    from .base import JobType


@dataclass(frozen=True)
class Wiring:
    config: Any
    backend: Any
    residency: "Residency"
    leases: Any | None


@dataclass(frozen=True)
class JobTypeBinding:
    spec: JobTypeSpec
    build: Callable[[Wiring], "JobType"]


__all__ = ["JobTypeBinding", "Wiring"]
