from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import StreamingResponse

from ... import clock
from ...jobs.line import OPERATOR, limits
from ...queuesessions import OPEN, is_session_id
from .. import sse
from ..caller import client_agent
from ..context import AppContext, Routers
from ..responses import NOT_FOUND, QueueList, QueueRemoved


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private

    @private.get("/queue", response_model=QueueList)
    async def list_queue(request: Request) -> dict[str, Any]:
        """What waits for the lane, in the order it will be offered it: the open queue
        session's items first, then first come, first served. A queue session waiting
        to open is a row of kind `session`.
        """
        line = ctx.line
        line.touch(client=client_agent(request))
        return {"items": line.rows(), "depth": len(line), "limits": limits()}

    @private.delete("/queue/{job_id}", response_model=QueueRemoved, responses=NOT_FOUND)
    async def remove_from_queue(job_id: str) -> dict[str, Any]:
        """Take a waiting job, call or queue session out of the queue (reason
        `operator`), or end the open queue session; a job that has started is cancelled
        with DELETE /v1/jobs/{id}.
        """
        if is_session_id(job_id):
            session = ctx.sessions.get(job_id)
            if session.status == OPEN:
                await ctx.session_closer.end(
                    session, OPERATOR,
                    "an operator ended it (DELETE /v1/queue/{id}, or the desktop Queue)",
                )
                return {"job_id": session.id, "status": session.status,
                        "reason": OPERATOR}
        item = ctx.line.remove(
            job_id, OPERATOR, "an operator removed it from the queue (DELETE /v1/queue)"
        )
        return {"job_id": item.job.id, "status": item.job.status, "reason": OPERATOR}

    @private.post("/queue/{job_id}/heartbeat", responses=NOT_FOUND)
    async def heartbeat_queued(job_id: str) -> dict[str, Any]:
        """Say the client that queued this job is still there. Only needed by a client
        that neither follows the job's events nor polls it.
        """
        line = ctx.line
        item = line.get(job_id)
        if item is None:
            raise line.not_waiting(job_id)
        line.touch(job_id=job_id)
        return {
            "job_id": job_id,
            "position": item.position,
            "waited_s": item.waited_s(clock.now()),
            "expires_at": item.expires_at.isoformat(),
        }

    @private.get("/queue/events")
    async def queue_events(request: Request) -> StreamingResponse:
        """Every change to the queue, for a dashboard: a `snapshot` first, then `added`,
        `moved`, `started` and `removed` as they happen.
        """
        return sse.queue_events(request, ctx.line)
