from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .memorybudget import gib_text

DESKTOP_BASIS_MEASURED = "measured"
DESKTOP_BASIS_DECLARED = "declared"
DESKTOP_BASIS_STATED = "stated"
DESKTOP_BASES: tuple[str, ...] = (
    DESKTOP_BASIS_MEASURED,
    DESKTOP_BASIS_DECLARED,
    DESKTOP_BASIS_STATED,
)


def desktop_reserve_words(allowance_bytes: int, basis: str) -> str:
    said = {
        DESKTOP_BASIS_MEASURED: "measured",
        DESKTOP_BASIS_DECLARED: "not measured; Crucible's default",
        DESKTOP_BASIS_STATED: "as set for this machine",
    }.get(basis, basis)
    return f"kept {gib_text(allowance_bytes)} for this PC's desktop ({said})"


@dataclass(frozen=True)
class CapabilityRow:

    capability: str
    enabled: bool
    selected: str
    reason: str
    shortfall_bytes: int
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "enabled": self.enabled,
            "selected": self.selected,
            "reason": self.reason,
            "summary": self.summary,
            "shortfall_bytes": self.shortfall_bytes,
        }


@dataclass(frozen=True)
class CapabilityRecord:

    backend_kind: str
    total_bytes: int
    desktop_allowance_bytes: int
    rows: tuple[CapabilityRow, ...]

    def row(self, capability: str) -> CapabilityRow | None:
        for entry in self.rows:
            if entry.capability == capability:
                return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_kind": self.backend_kind,
            "total_bytes": self.total_bytes,
            "desktop_allowance_bytes": self.desktop_allowance_bytes,
            "classes": [entry.to_dict() for entry in self.rows],
        }
