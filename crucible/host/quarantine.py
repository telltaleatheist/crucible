from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

BAD_SUFFIX = ".bad-"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def quarantine(path: Path, stamp: Callable[[], str] = _stamp) -> Path:
    path = Path(path)
    aside = path.with_name(f"{path.name}{BAD_SUFFIX}{stamp()}")
    counter = 1
    while aside.exists():
        counter += 1
        aside = path.with_name(f"{path.name}{BAD_SUFFIX}{stamp()}-{counter}")
    path.replace(aside)
    return aside


def is_quarantined(path: Path) -> bool:
    return BAD_SUFFIX in Path(path).name
