from __future__ import annotations

from typing import Any

from fastapi import Request

from ... import capabilityclasses, capabilityquery, installplan
from ...cardfacts import card_for
from ...errors import ApiError
from ...installonsubmit import live_decisions
from ...jobenv import INSTALLER_FOR
from ...narratorengines import NARRATOR_ENGINE_SAMPLING
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

    @private.get("/capability")
    async def capability(request: Request) -> dict[str, Any]:
        """What this server can hold, per capability class, and why not; `enabled:
        false` is an answer, not an error. A row's `goal` is the size its automatic pick
        aims at and never exceeds (`params_b`, and the ruling it comes from), or null for
        a class that has none. A client-sized class may be sized with
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
