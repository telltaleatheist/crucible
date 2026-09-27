from __future__ import annotations

MIB = 1024 ** 2
GIB = 1024 ** 3


def gib_text(value: int | float, places: int = 1) -> str:
    return f"{value / GIB:.{places}f} GiB"


def available_bytes(total_bytes: int, desktop_allowance_bytes: int) -> int:
    return max(0, total_bytes - desktop_allowance_bytes)


def engine_budget_bytes(
    total_bytes: int, desktop_allowance_bytes: int, free_bytes: int
) -> int:
    return min(available_bytes(total_bytes, desktop_allowance_bytes), max(0, free_bytes))


__all__ = [
    "GIB",
    "MIB",
    "available_bytes",
    "engine_budget_bytes",
    "gib_text",
]
