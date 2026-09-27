from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PASSAGE_UNIT = "passage"


@dataclass(frozen=True)
class ServingVariant:
    bits: int
    width: int
    need_bytes: int
    available: bool
    basis: str
    unit: str = PASSAGE_UNIT

    @property
    def full_precision(self) -> bool:
        return self.bits >= 16

    def label(self) -> str:
        if self.full_precision:
            precision = (
                "full quality (bf16)" if self.unit == PASSAGE_UNIT else "full precision"
            )
        else:
            precision = f"{self.bits}-bit"
        pace = f"one {self.unit} at a time" if self.width == 1 else f"{self.width} {self.unit}s at a time"
        return f"{precision}, {pace}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "width": self.width,
            "need_bytes": self.need_bytes,
            "available": self.available,
            "basis": self.basis,
        }


__all__ = ["PASSAGE_UNIT", "ServingVariant"]
