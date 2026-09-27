from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable

from ..errors import CrucibleError
from ..jobs.base import CANCELLED, DONE, FAILED, RUNNING
from ..settle import Held

TERMINAL_STATES = frozenset({DONE, FAILED, CANCELLED})

TASK_TYPES: tuple[str, ...] = (
    "pull",
    "install",
    "module",
    "engine",
    "engine-restart",
)

HISTORY = 50


class TaskCancelled(CrucibleError):
    ...


class TaskFailedByHost(CrucibleError):

    def __init__(self, task: "Task") -> None:
        super().__init__(f"task {task.id} failed on the host")


class ReloadRefused(CrucibleError):

    def __init__(self, held: Held) -> None:
        super().__init__(str(held))
        self.held = held


@dataclass
class Task:

    id: str
    type: str
    request: dict[str, Any]
    created: str
    started: str
    state: str = RUNNING
    finished: str | None = None
    error: dict[str, str] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    unmet: list[dict[str, str]] = field(default_factory=list)
    cancel_requested: bool = False
    on_submit: bool = False
    reason: str | None = None
    describe: Callable[["Task"], str | None] | None = None
    process: subprocess.Popen[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.id,
            "type": self.type,
            "request": self.request,
            "state": self.state,
            "error": self.error,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "unmet": self.unmet,
            "message": None if self.describe is None else self.describe(self),
        }

    def raise_if_cancelled(self) -> None:
        if self.cancel_requested:
            raise TaskCancelled(f"task {self.id} was cancelled")

    def reason_first(self) -> str:
        return "" if self.reason is None else f"{self.reason}. "


__all__ = [
    "CANCELLED",
    "DONE",
    "FAILED",
    "HISTORY",
    "RUNNING",
    "TASK_TYPES",
    "TERMINAL_STATES",
    "ReloadRefused",
    "Task",
    "TaskCancelled",
    "TaskFailedByHost",
]
