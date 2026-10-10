from __future__ import annotations

import asyncio
from typing import Any

from fastapi import Request

from ... import capabilityclasses, capabilityquery, installplan
from ...capabilitystore import rerecord
from ...cardfacts import card_for
from ...config import load_config
from ...errors import ApiError
from ...inflight import read_act
from ...installonsubmit import live_decisions
from ...jobenv import INSTALLER_FOR
from ...narratorengines import NARRATOR_ENGINE_SAMPLING
from ..caller import client_agent
from ..context import AppContext, Routers


def installable_job_type_rows() -> list[dict[str, Any]]:
    ordered: list[str] = []
    classes_of: dict[str, list[str]] = {}
    for entry in capabilityclasses.CLASSES:
        if entry.job_type not in classes_of:
            ordered.append(entry.job_type)
            classes_of[entry.job_type] = []
        classes_of[entry.job_type].append(entry.name)
    return [
        {
            "job_type": job_type,
            "classes": classes_of[job_type],
            "installer": INSTALLER_FOR.get(job_type),
            "narrator_engines": (
                sorted(NARRATOR_ENGINE_SAMPLING) if job_type == "tts" else []
            ),
        }
        for job_type in ordered
    ]


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    config, backend = ctx.config, ctx.backend

    @private.get("/capability/plan")
    async def capability_plan(request: Request) -> dict[str, Any]:
        """What an install (`?job_type=`) or a pull (`?subject=`) would give this card,
        decided live and writing nothing.
        """
        query = request.query_params
        job_type, subject = query.get("job_type"), query.get("subject")
        if (job_type is None) == (subject is None):
            raise ApiError(
                400,
                "invalid_request",
                "name exactly one of ?job_type= (an install) or ?subject= (a pull)",
            )
        decisions, card, pool = live_decisions(config, backend)
        if job_type is not None:
            return installplan.install_plan(
                job_type,
                decisions,
                card=card,
                total_bytes=backend.gpu.vram_bytes,
                pool=pool,
                desktop_allowance_bytes=config.desktop_allowance_bytes,
                desktop_basis=config.desktop_allowance_basis,
            )
        return installplan.subject_plan(
            subject, decisions, card=card, total_bytes=backend.gpu.vram_bytes, pool=pool
        )

    @private.post("/capability/record")
    async def record_capability(request: Request) -> dict[str, Any]:
        """Decide this card again and record it, as `crucible capability --write` does:
        the operator's "re-measure", never run by itself. A job type the card can no
        longer hold is turned off (`turned_off`); none is turned on, which is an
        install. Answers what was recorded and `low_vram_change`, the sentence when
        `[audio] low_vram` moved with it; read `GET /v1/capability` for the rows.
        """
        done = await asyncio.to_thread(lambda: rerecord(config, backend))
        changed = ["[capability] recorded again"] + [
            f"[jobs] {flag} = false (this card cannot hold it)"
            for flag in sorted(done.turned_off)
        ]
        if done.recorded.low_vram_change is not None:
            changed.append(done.recorded.low_vram_change)
        ctx.settings_history.record(
            act=read_act(request.headers), client=client_agent(request), changed=changed
        )
        # Adopted by the config follower before the next request, which also takes up
        # the record; the answer is read from the file just written.
        written = load_config(config.home)
        assert written.capability is not None
        return {
            "recorded": str(done.recorded.path),
            "total_bytes": written.capability.total_bytes,
            "desktop_allowance_bytes": written.capability.desktop_allowance_bytes,
            "turned_off": sorted(done.turned_off),
            "low_vram_change": done.recorded.low_vram_change,
        }

    @private.get("/capability")
    async def capability(request: Request) -> dict[str, Any]:
        """What this server can hold, per capability class, and why not; `enabled:
        false` is an answer, not an error. A row's `goal` is the size its automatic pick
        aims at and never exceeds (`params_b`, and the ruling it comes from), or null for
        a class that has none. A row's `with_images` is the model that serves a request
        of that class carrying images (decide's: the vision form of `selected` when it
        fits, else the largest model that reads images and fits at or below the goal; ""
        when none fits, and `with_images_reason` says what would), or null for a class
        that takes no images. A client-sized class may be sized with
        `?class=&context_tokens=&concurrency=`.
        """
        record = config.capability
        if record is None:
            raise ApiError(
                503,
                "capability_undecided",
                "this server has no capability record; nothing has probed the card "
                "on this host yet. Run `crucible capability --write` (or reinstall) "
                "to decide, and read `GET /v1/info` for what it offers meanwhile",
            )
        query = request.query_params
        document = record.to_dict()
        document["classes"] = capabilityquery.served_rows(
            record,
            gpu_vendor=backend.gpu.vendor,
            chosen={entry.capability: entry.model for entry in config.local_models},
            routes={entry.capability: entry.model for entry in config.routes},
            capability_class=query.get("class"),
            context_tokens=query.get(capabilityquery.CONTEXT_TOKENS_PARAM),
            concurrency=query.get(capabilityquery.CONCURRENCY_PARAM),
            audio_low_vram=config.audio_low_vram,
            card=card_for(config.home, backend.gpu),
        )
        for row in document["classes"]:
            row["route"] = (
                "upstream"
                if config.route_model(row["capability"]) is not None
                else "local"
            )
        return {**document, "job_types": installable_job_type_rows()}
