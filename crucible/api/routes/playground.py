from __future__ import annotations

from typing import Any

import asyncio

from fastapi import Body

from ... import playground
from ...playgroundpresets import Presets
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

    presets = Presets(ctx.config.home)

    @private.get("/playground/presets/{model}")
    async def playground_presets(model: str) -> dict[str, Any]:
        """The presets saved for `model` on this server, by name: each a form's params
        (no seed) and when it was saved."""
        return {"model": model, "presets": await asyncio.to_thread(presets.of, model)}

    @private.put("/playground/presets/{model}/{name}")
    async def save_playground_preset(
        model: str, name: str, params: dict[str, Any] = Body(..., embed=True)
    ) -> dict[str, Any]:
        """Save (or replace) the preset `name` for `model`: `{"params": {...}}` with the
        form's own fields - text, numbers and true/false; a seed is refused by name."""
        return {"model": model, "preset": await asyncio.to_thread(presets.save, model, name, params)}

    @private.delete("/playground/presets/{model}/{name}")
    async def delete_playground_preset(model: str, name: str) -> dict[str, Any]:
        """Remove the preset `name` of `model`; 404 when there is none."""
        await asyncio.to_thread(presets.delete, model, name)
        return {"model": model, "deleted": name}
