from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import StreamingResponse

from ...tasks import TaskStore
from ..context import AppContext, Routers
from ..schemas import TaskCreate
from ..sse import _event_stream, _last_event_id


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private

    @private.post("/tasks", status_code=202)
    async def create_task(request: Request, body: TaskCreate) -> dict[str, str]:
        """Admit one operator task (pull, install, module, engine or engine-restart), or
        refuse by name.
        """
        tasks: TaskStore = request.app.state.tasks
        return {"task_id": tasks.submit(body.request()).id}

    @private.get("/tasks")
    async def list_tasks(request: Request) -> dict[str, Any]:
        """The last few tasks, newest first. In memory; a restart forgets them."""
        tasks: TaskStore = request.app.state.tasks
        return {"tasks": [task.to_dict() for task in tasks.recent()]}

    @private.get("/tasks/{task_id}")
    async def get_task(request: Request, task_id: str) -> dict[str, Any]:
        tasks: TaskStore = request.app.state.tasks
        return tasks.get(task_id).to_dict()

    @private.delete("/tasks/{task_id}")
    async def cancel_task(request: Request, task_id: str) -> dict[str, str]:
        """Cancel a task. Answers `cancelling`; the stream's `cancelled` event says when
        it has stopped.
        """
        tasks: TaskStore = request.app.state.tasks
        task = tasks.get(task_id)
        return {"task_id": task.id, "status": tasks.cancel(task)}

    @private.get("/tasks/{task_id}/events")
    async def task_events(request: Request, task_id: str) -> StreamingResponse:
        tasks: TaskStore = request.app.state.tasks
        task = tasks.get(task_id)
        delivered = _last_event_id(request)
        return StreamingResponse(
            _event_stream(request, tasks, task, delivered),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )
