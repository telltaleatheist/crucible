from __future__ import annotations

from typing import Any

from ... import playground
from ..context import AppContext, Routers


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private

    @private.get("/playground")
    async def playground_pages() -> dict[str, Any]:
        """One page per image, video and audio model this build declares: the params its
        form shows, with defaults and limits from its manifest, and its `standing`:
        `ready`; `download`, which its first job fetches by itself (409 `installing`
        names the task; `download_bytes` when the size is known); or `unavailable`,
        with `reason` saying why.
        """
        return {"pages": playground.pages(ctx.config, ctx.backend, ctx.store.registry)}
