from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from ..errors import ApiError, JobError
from ..inflight import require_act_name
from ..leases import require_ttl


class LeaseOnLoad(BaseModel):
    model_config = ConfigDict(extra="forbid")

    act: str
    ttl_seconds: int


def open_lease_for_load(
    leases: Any | None,
    *,
    kind: str,
    subject: str,
    request: LeaseOnLoad,
    client: str | None,
) -> str:
    if leases is None:
        raise JobError(
            "leases_unavailable",
            "this build has no lease register, so `params.lease` cannot be "
            "honoured. It is refused rather than ignored: a client told nothing "
            "would believe it holds the card",
        )
    try:
        act = require_act_name(request.act.strip(), "a load's `params.lease.act`")
        ttl = require_ttl(request.ttl_seconds)
        lease = leases.open(
            kind=kind, subject=subject, act=act, client=client, ttl_seconds=ttl
        )
    except ApiError as exc:
        raise JobError(exc.code, exc.message) from None
    return lease.id


__all__ = ["LeaseOnLoad", "open_lease_for_load"]
