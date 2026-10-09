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
    # A class whose requests may carry images (CapabilityClass.takes_images), granted: the
    # model that serves a request with images, "" when none fits, and why. None on every
    # other row, and on a record written before 2026-10-09.
    with_images: str | None = None
    with_images_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "capability": self.capability,
            "enabled": self.enabled,
            "selected": self.selected,
            "reason": self.reason,
            "summary": self.summary,
            "shortfall_bytes": self.shortfall_bytes,
        }
        if self.with_images is not None:
            document["with_images"] = self.with_images
            document["with_images_reason"] = self.with_images_reason
        return document

    def to_wire(self) -> dict[str, Any]:
        """The row as `/v1/capability` answers it: `with_images` and its reason on every
        row, null where the class takes no images (to_dict leaves them out for TOML)."""
        return {
            **self.to_dict(),
            "with_images": self.with_images,
            "with_images_reason": self.with_images_reason,
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
