from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .screens import Progress, size_text

TERMINAL = ("done", "failed", "cancelled")

LINES_KEPT = 4


def request_words(task: Mapping[str, Any]) -> str:
    request = task.get("request") or {}
    kind = task.get("type")
    if kind == "pull":
        return f"Downloading {request.get('id')}"
    if kind == "install":
        engine = request.get("narrator_engine")
        return f"Installing {request.get('job_type')}" + (f" ({engine})" if engine else "")
    if kind == "module":
        module = request.get("module") or {}
        return f"Installing module {module.get('name')}"
    return str(kind).replace("-", " ").capitalize()


@dataclass
class TaskWatch:
    task_id: str
    title: str
    step: str = ""
    bytes_done: int | None = None
    bytes_total: int | None = None
    lines: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    ended: str | None = None
    error: str | None = None

    @classmethod
    def of(cls, task: Mapping[str, Any]) -> "TaskWatch":
        return cls(task_id=str(task.get("task_id")), title=request_words(task))

    def _step(self, data: Mapping[str, Any]) -> None:
        total = data.get("total")
        count = f" (step {data.get('index')} of {total})" if total and total > 1 else ""
        self.step = f"{data.get('name')}{count}"
        self.bytes_done = self.bytes_total = None

    def _progress(self, data: Mapping[str, Any]) -> None:
        if "line" in data:
            self.lines = (self.lines + [str(data["line"])])[-LINES_KEPT:]
            return
        self.bytes_done = data.get("bytes_done")
        self.bytes_total = data.get("bytes_total")

    def apply(self, event: str, data: Mapping[str, Any]) -> None:
        if event in TERMINAL:
            self.finish(event, data)
            return
        handlers = {
            "step": self._step,
            "progress": self._progress,
            "skipped": lambda said: self.skipped.append(str(said.get("reason"))),
        }
        handler = handlers.get(event)
        if handler is not None:
            handler(data)

    def finish(self, state: str, error: Mapping[str, Any] | None) -> None:
        self.ended = state
        if state == "failed":
            code = (error or {}).get("code")
            message = (error or {}).get("message")
            self.error = f"{message} ({code})" if message else (
                "It failed on the server without a reason. Its log is under Settings, "
                "Open logs folder")

    def settle(self, task: Mapping[str, Any]) -> None:
        state = str(task.get("state"))
        if self.ended is None and state in TERMINAL:
            self.finish(state, task.get("error"))

    def fraction(self) -> float | None:
        if self.ended == "done":
            return 1.0
        if self.bytes_done is None or not self.bytes_total:
            return None
        return max(0.0, min(1.0, self.bytes_done / self.bytes_total))

    def detail(self) -> str:
        if self.ended == "done":
            return "Finished"
        if self.ended == "cancelled":
            return "Cancelled"
        if self.error is not None:
            return self.error
        if self.bytes_total:
            return f"{size_text(self.bytes_done) or '0 MB'} of {size_text(self.bytes_total)}"
        return self.lines[-1] if self.lines else self.step or "Starting"

    def view(self) -> Progress:
        return Progress(title=self.title, detail=self.detail(), fraction=self.fraction(),
                        cancel=None if self.ended else self.task_id, target="task")
