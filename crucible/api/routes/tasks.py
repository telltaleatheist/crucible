from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import StreamingResponse

from ...tasks import TaskStore
from ..schemas import TaskCreate
from ..sse import _last_event_id, _task_event_stream
from ..context import AppContext, Routers


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private

    # ----------------------------------------------------------------- tasks
    #
    # PHASE13-OPERATOR.md section 3.3. Five routes with the shapes the job
    # routes have — a 202 with an id, a status read, an SSE stream, a DELETE
    # that cancels, a list — because a page that already knows how to watch a
    # job should not have to learn a second protocol to watch an install. What
    # they are NOT is `POST /v1/jobs`: a job is work a client wants done with
    # this server's card, a task is work done to the server itself, and
    # `crucible/tasks.py` is where that difference is written down.

    @private.post("/tasks", status_code=202)
    async def create_task(request: Request, body: TaskCreate) -> dict[str, str]:
        """Admit one operator task, or refuse by name.

        Every refusal is made here, before the 202, and in the order the job
        door uses: what is wrong with the REQUEST first (`unknown_subject`,
        `unknown_job_type`, `narrator_engine_required`, `invalid_module`), then
        what is already true (`already_installed`, `job_type_installed`), then
        what this server is doing (`task_busy`, and for anything that reloads
        the registry, `server_busy`). A client with a misspelled id told "busy"
        would come back in ten minutes to be told about the typo.
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
        """Cancel. A pull stops at its next chunk and its partial bytes go.

        `cancelling` and not `cancelled`, exactly as the job door answers:
        the flag is set here and the runner ends when it sees it, which for a
        pull is the next progress callback and for an install is the SIGTERM
        landing. Watch the stream for the `cancelled` event — telling a caller
        "cancelled" before the download thread has stopped would be the
        ambiguous answer R3 forbids.
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
            _task_event_stream(request, tasks, task, delivered),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )
