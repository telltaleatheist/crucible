from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import StreamingResponse

from .. import sse
from ..context import AppContext, Routers
from ..responses import BUSY_RESPONSES
from ..schemas import TaskCreate


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private

    @private.post("/tasks", status_code=202, responses=BUSY_RESPONSES)
    async def create_task(body: TaskCreate) -> dict[str, str]:
        """Admit one operator task (pull, install, module, engine or engine-restart), or
        refuse by name.
        """
        return {"task_id": ctx.tasks.submit(body.request()).id}

    @private.get("/tasks")
    async def list_tasks() -> dict[str, Any]:
        """The last few tasks, newest first. In memory; a restart forgets them."""
        return {"tasks": [task.to_dict() for task in ctx.tasks.recent()]}

    @private.get("/tasks/{task_id}")
    async def get_task(task_id: str) -> dict[str, Any]:
        """One task's state: type, status, progress, and its error when it failed."""
        return ctx.tasks.get(task_id).to_dict()

    @private.delete("/tasks/{task_id}")
    async def cancel_task(task_id: str) -> dict[str, str]:
        """Cancel a task. Answers `cancelling`; the stream's `cancelled` event says when
        it has stopped.
        """
        tasks = ctx.tasks
        task = tasks.get(task_id)
        return {"task_id": task.id, "status": tasks.cancel(task)}

    @private.get("/tasks/{task_id}/events")
    async def task_events(request: Request, task_id: str) -> StreamingResponse:
        """The task's events as SSE, ending with `done`, `failed` or `cancelled`; the same
        shape and resume rules as a job's."""
        tasks = ctx.tasks
        task = tasks.get(task_id)
        return sse.job_events(request, tasks, task, sse.last_event_id(request))
