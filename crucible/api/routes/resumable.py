from __future__ import annotations

from typing import Any

from ..context import AppContext, Routers


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private

    @private.get("/resumable")
    async def list_resumable() -> dict[str, Any]:
        """Every resume journal this server keeps, newest first, with progress, inputs
        and expiry. Send a row's `resume_id` as `params.resume` to continue it.
        """
        return {"resumable": ctx.store.journals.list()}

    @private.get("/resumable/{resume_id}")
    async def get_resumable(resume_id: str) -> dict[str, Any]:
        """One journal, as `GET /v1/resumable` lists it."""
        return ctx.store.journals.entry(resume_id)

    @private.delete("/resumable/{resume_id}")
    async def discard_resumable(resume_id: str) -> dict[str, Any]:
        """Discard a journal now; refused `resume_in_use` while a job writes it."""
        return ctx.store.journals.discard(resume_id)
