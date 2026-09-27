from __future__ import annotations

from typing import Any

from fastapi import Request

from ...jobs.queue import JobStore
from ..context import AppContext, Routers


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private

    # ------------------------------------------------------------- resumable

    @private.get("/resumable")
    async def list_resumable(request: Request) -> dict[str, Any]:
        """Every journal this server keeps, newest first (Owen, 2026-09-27:
        *"maybe we could even have a call that shows what's available to
        resume?"*).

        One row per journal: its `resume_id` (send it as `params.resume`), the
        job type, the model and revision, the inputs by name and sha256, the
        output-affecting `params` it was written under, `units_done` of
        `units_total` and a `progress` sentence, `last_saved`, `expires_at`
        (`[jobs] retention_days` after the last save), the job that started
        it (`job_id`), and the job that last wrote it with how that ended
        (`last_job_id`, `state`: queued, running, done, failed, cancelled or
        interrupted). Nothing is resumed by reading this: resuming is the
        app's decision (docs/RESUMABLE-JOBS.md).
        """
        store: JobStore = request.app.state.store
        return {"resumable": store.journals.list()}

    @private.get("/resumable/{resume_id}")
    async def get_resumable(request: Request, resume_id: str) -> dict[str, Any]:
        """One journal, as `GET /v1/resumable` lists it."""
        store: JobStore = request.app.state.store
        return store.journals.entry(resume_id)

    @private.delete("/resumable/{resume_id}")
    async def discard_resumable(request: Request, resume_id: str) -> dict[str, Any]:
        """Discard a journal now. Refused `resume_in_use` while a job writes it.

        The id then answers `resume_expired` rather than `unknown_resume_id`,
        so a client resuming it later is told what happened.
        """
        store: JobStore = request.app.state.store
        return store.journals.discard(resume_id)
