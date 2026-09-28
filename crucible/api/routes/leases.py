from __future__ import annotations

import asyncio
from typing import Any

from fastapi import Request, Response

from ...admission import refuse_lease_on_an_upstream
from ...cardkinds import KIND_NOUNS
from ...errors import ApiError
from ...inflight import require_act_name
from ...leases import require_ttl
from ..caller import client_agent
from ..context import AppContext, Routers
from ..schemas import LeaseOpen


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    residency = ctx.residency

    @private.post("/models/{subject_id:path}/lease", status_code=201)
    async def open_lease(
        request: Request, subject_id: str, body: LeaseOpen
    ) -> dict[str, Any]:
        """Hold whatever is resident (model, voice or aligner) on the card for a run;
        jobs that would move it are refused `409 leased`. A lease never loads anything.
        """
        refuse_lease_on_an_upstream(subject_id)
        ttl_seconds = require_ttl(body.ttl_seconds)
        act = require_act_name(body.act.strip(), "a lease's `act`")
        async with residency.settled_for(f"a lease on {subject_id!r}"):
            resident = residency.resident
            if resident is None or resident.id != subject_id:
                raise ApiError(
                    409,
                    "not_resident",
                    f"{subject_id!r} is not resident on this server; "
                    + (
                        f"the resident {KIND_NOUNS[resident.kind]} is "
                        f"{resident.id!r}. "
                        if resident is not None
                        else "nothing is. "
                    )
                    + "A lease promises not to move what is on the card; it "
                    "never loads anything — load it first (load-model, "
                    "load-voice, or an align job for an aligner), then lease "
                    "what that left resident.",
                    {
                        "requested": subject_id,
                        "resident": None if resident is None else resident.id,
                        "resident_kind": (
                            None if resident is None else resident.kind
                        ),
                    },
                )
            lease = ctx.leases.open(
                kind=resident.kind,
                subject=subject_id,
                act=act,
                client=client_agent(request),
                ttl_seconds=ttl_seconds,
            )
        ctx.settlement.arm_for_lease_expiry()
        return lease.receipt()

    @private.post("/leases/{lease_id}/heartbeat")
    async def heartbeat_lease(lease_id: str) -> dict[str, Any]:
        """Push the lease's deadline out by its own ttl. A 404 means the lease was
        released or expired and the run is no longer protected.
        """
        extended = ctx.leases.heartbeat(lease_id)
        ctx.settlement.arm_for_lease_expiry()
        return {"expires_at": extended.expires_at.isoformat()}

    @private.delete("/leases/{lease_id}", status_code=204)
    async def release_lease(lease_id: str) -> Response:
        """Release the lease; if nothing else holds the card, it is cleared before this
        answers.
        """
        settlement = ctx.settlement
        ctx.leases.release(lease_id)
        settlement.arm_for_lease_expiry()
        await asyncio.to_thread(settlement.settle_quietly, "the lease was released")
        return Response(status_code=204)
