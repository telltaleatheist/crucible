from __future__ import annotations

from datetime import datetime, timezone


def now() -> datetime:
    return datetime.now(timezone.utc)


def utcnow() -> str:
    return now().isoformat()


def utcnow_to_the_second() -> str:
    return now().replace(microsecond=0).isoformat()


__all__ = ["now", "utcnow", "utcnow_to_the_second"]
