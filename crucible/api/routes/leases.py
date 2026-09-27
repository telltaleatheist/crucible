from __future__ import annotations

import asyncio
from typing import Any

from fastapi import Request, Response

from ...errors import ApiError
from ...inflight import require_act_name
from ...leases import Leases, require_ttl
from ...residency import KIND_NOUNS
from ...settle import Settlement
from ..caller import client_agent
from ..schemas import LeaseOpen
from ..upstream import _refuse_lease_on_an_upstream
from ..context import AppContext, Routers


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    residency = ctx.residency

    # ---------------------------------------------------------------- leases
    #
    # PHASE7-LANES.md section 5.2. Three routes and no state worth the name: a
    # client says it intends a run on the resident thing — model, voice or
    # aligner — heartbeats while the run is alive, and releases when it is done.
    # What that buys it is one refusal — `409 leased` at the job door for
    # anything that would take that thing off the card. Everything else about
    # this server is unchanged.

    @private.post("/models/{subject_id:path}/lease", status_code=201)
    async def open_lease(
        request: Request, subject_id: str, body: LeaseOpen
    ) -> dict[str, Any]:
        """Take the one lease this server holds at a time, on ANY resident kind.

        The order of the checks is their specificity, which is the job door's
        rule: a bad ttl and an unknown act are true of the request whatever this
        server is doing, so a client with a typo is told about the typo rather
        than about somebody else's lease. Residency comes next, because leasing a
        thing that is not here is a different mistake from being too late for
        one that is.

        **The id may name a model, a voice or an aligner** (PHASE7-LANES.md
        section 5.2, extended 2026-09-14). The route keeps its `/models/` path
        and its one route family, because the question it asks does not change
        with the kind: *is this the thing on the card?* The card holds ONE thing,
        so the kind is read off the residency rather than sent — and the
        namespaces being separate (a voice may be called `qwen3.5-9b`) cannot
        produce an ambiguity here, since only one of two colliding ids can be
        resident at a time and a lease is only ever on the resident one.

        Without this a book rendered chapter by chapter paid a narrator load per
        chapter and a book aligned chapter by chapter paid an aligner load per
        chapter, because the unload ruling clears the card the moment nothing
        holds it and the lease — the one thing that can hold it — could only name
        a model.
        """
        leases: Leases = request.app.state.leases
        _refuse_lease_on_an_upstream(subject_id)
        ttl_seconds = require_ttl(body.ttl_seconds)
        act = require_act_name(body.act.strip(), "a lease's `act`")
        # A CLEARANCE IS WAITED OUT, AND THE LEASE IS TAKEN ATOMICALLY AGAINST
        # ONE BEGINNING (2026-09-24, Briefcase). Briefcase's second run got a
        # 201 here on a model the settlement was already clearing, and its
        # first chat under that lease was `model_not_resident`. Now the door
        # waits the clearance out and answers from the settled card, and the
        # residency check and `leases.open` are made under the lock the
        # settlement's check-and-claim takes — so a lease granted here is one
        # the settlement will see, and never one on a thing already leaving.
        async with residency.settled_for(f"a lease on {subject_id!r}"):
            # Whatever is on the card, of any kind. A lease NEVER loads anything
            # — it is the promise not to move what is already there — so the
            # honest answer to an id that is not resident is the same one an
            # empty card gets, and it names what IS there so the client is not
            # left guessing which of the two mistakes it made.
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
            lease = leases.open(
                kind=resident.kind,
                subject=subject_id,
                act=act,
                client=client_agent(request),
                ttl_seconds=ttl_seconds,
            )
        # The lease is now the thing holding the card, and its deadline is the
        # one moment a holder lets go that this server would otherwise never
        # see. Armed at the client's own `expires_at` (crucible/settle.py).
        request.app.state.settlement.arm_for_lease_expiry()
        return lease.receipt()

    @private.post("/leases/{lease_id}/heartbeat")
    async def heartbeat_lease(request: Request, lease_id: str) -> dict[str, Any]:
        """I am still here. Pushes the deadline out by the lease's own ttl.

        A 404 here is not an error to log and continue past: it means this
        client's run is no longer protected, and the card may move under it at
        any moment. The body says whether the lease was released or expired,
        which is the difference between "somebody took it from me" and "I stopped
        talking for too long".
        """
        leases: Leases = request.app.state.leases
        extended = leases.heartbeat(lease_id)
        # The deadline moved, so the one-shot that watches it moves with it.
        request.app.state.settlement.arm_for_lease_expiry()
        return {"expires_at": extended.expires_at.isoformat()}

    @private.delete("/leases/{lease_id}", status_code=204)
    async def release_lease(request: Request, lease_id: str) -> Response:
        """Give the card back before the ttl does it for you.

        The usual end of a lease, and the one that matters: expiry is the
        backstop for a client that died, not the way a finished run ends. A run
        that releases frees the next client immediately instead of after up to an
        hour of nothing happening.
        """
        leases: Leases = request.app.state.leases
        settlement: Settlement = request.app.state.settlement
        leases.release(lease_id)
        # The lease is gone, so the deadline it was watched by is too.
        settlement.arm_for_lease_expiry()
        # OWEN'S RULING, 2026-09-14: a released lease is a holder letting go, so
        # if the lane, the claim and the chats are also clear the card is cleared
        # before this 204 is written. That is Foundry's *"down when the queue
        # drains"* (FROM-FOUNDRY-WSL-VLLM.md section 3) with the drain stated by
        # the client instead of guessed at. Off the loop, because stopping an
        # engine waits on a process.
        await asyncio.to_thread(settlement.settle_quietly, "the lease was released")
        return Response(status_code=204)
