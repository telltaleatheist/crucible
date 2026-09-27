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

    # -------------------------------------------------------------- settings

    def _installed_subjects(live: Config) -> dict[str, bool]:
        """Which subjects are on this disk, by id, straight from the catalog.

        The settings document needs `installed` for every model an app may
        choose, and `GET /v1/catalog` is ALREADY the one owner of that fact.
        Asking it here rather than re-deriving is what keeps the chooser and
        the catalog from disagreeing about what is on the disk
        (ARCHITECTURE.md R1) — every kind, not only `model`, because a voice
        and an aligner are chosen the same way.
        """
        return {row["id"]: row["installed"] for row in catalog.rows(live, backend, residency)}

    @private.get("/settings")
    async def get_settings(request: Request) -> dict[str, Any]:
        """Where each class's work runs, and which upstreams are configured.

        PHASE15-HOST.md section 3.1. Owen, 2026-09-14: *"Settings live in the
        engine and nowhere else."* An app draws this document and writes
        through `PUT`; it holds no key, no route and no model list of its own.

        **A key is never in this answer.** `key_hint` is its last four
        characters, which is enough to recognise WHICH key is there — the
        question a person with two accounts asks — and nothing else. There is
        no route on this server that returns one.
        """
        live: Config = request.app.state.config
        return settings_module.document(live, installed=_installed_subjects(live))

    @private.put("/settings")
    async def put_settings(request: Request) -> dict[str, Any]:
        """A partial patch, applied whole or not at all, live without a restart.

        PHASE15-HOST.md section 3.2. The order inside one request is the
        contract's — upstreams, then routes, then the whole validated — which
        is what lets an app configure an upstream AND route a class to it in
        one call, the way section 5.2 tells it to.

        **A refusal applies nothing.** `settings.resolve` builds the candidate
        document in memory and raises before `settings.apply` writes a byte, so
        a request refused for its routes does not leave a key behind on a
        server whose operator believes it failed.

        The answer is the whole `GET /v1/settings` document AFTER the write, so
        a window never has to guess what took.
        """
        live: Config = request.app.state.config
        # Read BEFORE the work, like the chat door: an unknown act is a 400
        # rather than a write that happened and was then recorded under a name
        # nobody knows.
        act = read_act(request.headers)
        try:
            patch = json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ApiError(
                400, "invalid_request", f"the settings body is not JSON: {exc}"
            ) from None
        resolved = settings_module.resolve(live, patch)
        # Off the event loop: this writes a file and re-reads it, and a settings
        # write must not stall a job's event stream.
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
        """Ask an upstream what it serves, with a key that may not be saved yet.

        PHASE15-HOST.md section 3.2. The body is optional and carries
        `{"key": …}` or `{"url": …}` to test BEFORE saving, which is the order a
        person actually works in: paste, check it works, then save. With no
        body the stored record is used.

        **Unbilled, and never cached.** The answer is somebody else's and
        changes without telling us; a stale list shown beside a key the
        operator pasted ten seconds ago is exactly the moment they would
        believe it.

        `POST` and not `GET` because it takes a body carrying a secret, and a
        secret in a query string is a secret in a log.
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
