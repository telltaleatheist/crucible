from __future__ import annotations

from typing import Any

from ... import playground
from ..context import AppContext, Routers


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private

    @private.get("/playground")
    async def playground_pages() -> dict[str, Any]:
        """One page per image, video and audio model this build declares: the params its
        form shows, with defaults and limits from its manifest, and whether this server
        can run it now. `reason` says why not when `available` is false.
        """
        return {"pages": playground.pages(ctx.config, ctx.backend, ctx.store.registry)}
