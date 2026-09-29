from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..errors import ApiError, JobError
from ..inflight import require_act_name
from ..leases import require_ttl


class LeaseOnLoad(BaseModel):
    model_config = ConfigDict(extra="forbid")

    act: str
    ttl_seconds: int


@dataclass(frozen=True)
class HeldForLoad:
    lease_id: str
    opened: bool


def require_lease_request(request: LeaseOnLoad, *, act: str | None = None) -> None:
    named = require_act_name(request.act.strip(), "a load's `params.lease.act`")
    require_ttl(request.ttl_seconds)
    if act is not None and named != act:
        raise ApiError(
            400,
            "lease_act_mismatch",
            f"`params.lease.act` is {named!r}, and a lease taken by this job holds "
            f"{act!r} work, so it must say {act!r}. Send "
            f'{{"lease": {{"act": "{act}", "ttl_seconds": {request.ttl_seconds}}}}}',
            {"act": named, "expected": act},
        )


def _held_by_the_same_caller(
    leases: Any, *, kind: str, subject: str, client: str | None
) -> str | None:
    held = leases.current()
    if held is None:
        return None
    if (held.kind, held.subject, held.client) != (kind, subject, client):
        return None
    return str(leases.heartbeat(held.id).id)


def hold_for_load(
    leases: Any | None,
    *,
    kind: str,
    subject: str,
    request: LeaseOnLoad,
    client: str | None,
) -> HeldForLoad:
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
        already = _held_by_the_same_caller(
            leases, kind=kind, subject=subject, client=client
        )
        if already is not None:
            return HeldForLoad(already, opened=False)
        lease = leases.open(
            kind=kind, subject=subject, act=act, client=client, ttl_seconds=ttl
        )
    except ApiError as exc:
        raise JobError(exc.code, exc.message) from None
    return HeldForLoad(lease.id, opened=True)


def open_lease_for_load(
    leases: Any | None,
    *,
    kind: str,
    subject: str,
    request: LeaseOnLoad,
    client: str | None,
) -> str:
    return hold_for_load(
        leases, kind=kind, subject=subject, request=request, client=client
    ).lease_id


def let_go_of(leases: Any | None, held: HeldForLoad | None) -> None:
    if leases is None or held is None or not held.opened:
        return
    try:
        leases.release(held.lease_id)
    except ApiError:
        pass


__all__ = [
    "HeldForLoad",
    "LeaseOnLoad",
    "hold_for_load",
    "let_go_of",
    "open_lease_for_load",
    "require_lease_request",
]
