from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from fastapi import Request

from ... import catalog, ladder, upstreams
from ... import settings as settings_module
from ...config import Config
from ...errors import ApiError
from ...inflight import read_act
from ..caller import client_agent
from ..context import AppContext, Routers


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    config, backend, residency = ctx.config, ctx.backend, ctx.residency

    def _installed_subjects(live: Config) -> dict[str, bool]:
        return {row["id"]: row["installed"] for row in catalog.rows(live, backend, residency)}

    @private.get("/settings")
    async def get_settings(request: Request) -> dict[str, Any]:
        """Where each class's work runs and which upstreams are configured. A key is
        never returned; `key_hint` shows its last four characters.
        """
        live: Config = request.app.state.config
        return settings_module.document(live, installed=_installed_subjects(live))

    @private.put("/settings")
    async def put_settings(request: Request) -> dict[str, Any]:
        """Apply a partial settings patch, whole or not at all, live without a restart.
        Answers the full settings document after the write.
        """
        live: Config = request.app.state.config
        act = read_act(request.headers)
        try:
            patch = json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ApiError(
                400, "invalid_request", f"the settings body is not JSON: {exc}"
            ) from None
        resolved = settings_module.resolve(live, patch)
        await asyncio.to_thread(
            lambda: settings_module.apply(
                live,
                resolved,
                gpu_vendor=backend.gpu.vendor,
                card=ladder.card_for(config.home, backend.gpu),
            )
        )
        if resolved.changed:
            request.app.state.settings_history.record(
                act=act,
                client=client_agent(request),
                changed=resolved.changed,
            )
        return settings_module.document(live, installed=_installed_subjects(live))

    @private.post("/settings/upstreams/{name}/test")
    async def test_upstream(request: Request, name: str) -> dict[str, Any]:
        """List what an upstream serves, using the body's `key` or `url` when given,
        else the stored record. Never cached.
        """
        live: Config = request.app.state.config
        upstreams.require_name(name, "the path")
        raw = await request.body()
        if raw.strip() == b"":
            probe = None
        else:
            try:
                probe = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ApiError(
                    400, "invalid_request", f"the test body is not JSON: {exc}"
                ) from None
        if probe is None or probe == {}:
            record = live.upstream(name)
            if record is None:
                raise ApiError(
                    400,
                    "upstream_unconfigured",
                    f"{name} is not configured on this server and the request "
                    f"carried no {upstreams.UPSTREAM_FIELD[name]!r} to test "
                    "with. Send one to check it before saving it",
                    {
                        "field": f"upstreams.{name}."
                        f"{upstreams.UPSTREAM_FIELD[name]}",
                        "upstream": name,
                    },
                )
        else:
            record = upstreams.record_from_patch(
                name, probe, f"the test body for {name}"
            )
        client: httpx.AsyncClient = request.app.state.http
        return {"models": await upstreams.list_models(client, record)}
