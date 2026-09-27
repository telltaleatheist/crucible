from __future__ import annotations

from pathlib import Path

MARKER_NAME = "engine.stopped"
REASON = "Stopped by the operator"


def marker(home: Path) -> Path:
    return Path(home) / MARKER_NAME


def is_stopped(home: Path) -> bool:
    return marker(home).exists()


def record(home: Path) -> None:
    Path(home).mkdir(parents=True, exist_ok=True)
    marker(home).write_text(REASON + "\n", encoding="utf-8")


def clear(home: Path) -> None:
    marker(home).unlink(missing_ok=True)


def restore(home: Path, stopped: bool) -> None:
    if stopped:
        record(home)
    else:
        clear(home)
