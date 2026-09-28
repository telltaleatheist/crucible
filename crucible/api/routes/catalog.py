from __future__ import annotations

import sys
from typing import Any

from fastapi import Request, Response

from ... import catalog, weights
from ...cardkinds import KIND_NOUNS
from ...errors import ApiError, CrucibleError
from ...inflight import read_act
from ...jobs import disabled_error, model_rows
from ..caller import client_agent
from ..context import AppContext, Routers


def _entry_names(entry: Any, kind: str, subject_id: str) -> bool:
    return (
        isinstance(entry, dict)
        and entry.get("kind") == kind
        and entry.get("id") == subject_id
    )


def _task_names(task: Any, kind: str, subject_id: str) -> bool:
    body = task.request
    if body.get("kind") == kind and body.get("id") == subject_id:
        return True
    module = body.get("module")
    if not isinstance(module, dict):
        return False
    return any(
        _entry_names(entry, kind, subject_id)
        for entry in module.get("subjects", []) or []
    )


def _through(subject: catalog.Subject, holder_id: str) -> str:
    if holder_id == subject.id:
        return ""
    return f" ({holder_id}, which shares its weights)"


def _resident_holder(ctx: AppContext, subject: catalog.Subject, readers: Any) -> dict | None:
    resident = ctx.residency.resident
    if resident is None or resident.id not in readers:
        return None
    return {
        "kind": subject.kind,
        "id": subject.id,
        "who": (
            f"it is the {KIND_NOUNS[resident.kind]} on the card right "
            f"now{_through(subject, resident.id)}; unload it first"
        ),
        "fact": "resident",
    }


def _lease_holder(ctx: AppContext, subject: catalog.Subject, readers: Any) -> dict | None:
    lease = ctx.leases.current()
    if lease is None or lease.subject not in readers:
        return None
    return {
        "kind": subject.kind,
        "id": subject.id,
        "who": (
            f"{lease.client or 'a client'} holds a lease on it"
            f"{_through(subject, lease.subject)} "
            f"until {lease.expires_at.isoformat()}"
        ),
        "fact": "lease",
        **lease.receipt(),
    }


def _task_holder(ctx: AppContext, subject: catalog.Subject, readers: Any) -> dict | None:
    running = ctx.tasks.running
    if running is None:
        return None
    if not any(_task_names(running, subject.kind, reader) for reader in readers):
        return None
    return {
        "kind": subject.kind,
        "id": subject.id,
        "who": f"task {running.id} ({running.type}) names it",
        "fact": "task",
        "task_id": running.id,
    }


def _subject_holder(ctx: AppContext, subject: catalog.Subject) -> dict | None:
    readers = catalog.ids_reading(subject)
    for holder in (_resident_holder, _lease_holder, _task_holder):
        found = holder(ctx, subject, readers)
        if found is not None:
            return found
    return None


def _installed_subject(ctx: AppContext, kind: str, subject_id: str) -> tuple[Any, Any]:
    ids = {"kind": kind, "id": subject_id}
    if kind not in catalog.KINDS:
        raise ApiError(
            404,
            "subject_unknown",
            f"{kind!r} is not a subject kind; they are {list(catalog.KINDS)}",
            ids,
        )
    subject = catalog.find(ctx.config, ctx.backend, kind, subject_id)
    if subject is None:
        raise ApiError(
            404,
            "subject_unknown",
            f"this server has no {kind} called {subject_id!r} for "
            f"{ctx.backend.kind}. GET /v1/catalog lists every subject it can hold",
            ids,
        )
    found = subject.installed()
    if found is None:
        raise ApiError(
            409,
            "subject_not_installed",
            f"{kind} {subject_id!r} is not installed on this server, so "
            "there is nothing to remove. Refused rather than answered 204: "
            "a caller told 'done' about a subject that was never there "
            "would believe a migration had deleted something",
            ids,
        )
    return subject, found


def _refuse_if_held(ctx: AppContext, subject: catalog.Subject, kind: str, subject_id: str) -> None:
    who = _subject_holder(ctx, subject)
    if who is None:
        return
    raise ApiError(
        409,
        "subject_in_use",
        f"{kind} {subject_id!r} cannot be removed: {who['who']}. "
        "Deleting the files under a running engine would leave it "
        "serving a model that is no longer on the disk",
        who,
    )


def _remove_files(subject: catalog.Subject, found: Any, kind: str, subject_id: str) -> Any:
    ids = {"kind": kind, "id": subject_id}
    try:
        return subject.remove()
    except weights.WeightsShared as exc:
        raise ApiError(
            409,
            "weights_shared",
            str(exc),
            {**ids, "backend": exc.backend, "aliases": list(exc.aliases)},
        ) from None
    except weights.RemoveFailed as exc:
        raise ApiError(
            500, "subject_remove_failed", str(exc), {**ids, "path": str(exc.path)}
        ) from None
    except (CrucibleError, OSError) as exc:
        raise ApiError(
            500,
            "subject_remove_failed",
            f"removing {kind} {subject_id!r} failed: {type(exc).__name__}: {exc}",
            {**ids, "path": str(found.path)},
        ) from None


def _catalog_route_handler(ctx: AppContext):
    async def catalog_route() -> dict[str, Any]:
        """Every subject this backend can hold, installed or not."""
        return {"rows": catalog.rows(ctx.config, ctx.backend, ctx.residency),
                "backend_kind": ctx.backend.kind}

    return catalog_route


def _catalog_remove_handler(ctx: AppContext):
    async def catalog_remove(
        request: Request, kind: str, subject_id: str
    ) -> Response:
        """Delete an installed subject's files. Refused while the subject is resident,
        leased or named by a running task.
        """
        subject, found = _installed_subject(ctx, kind, subject_id)
        _refuse_if_held(ctx, subject, kind, subject_id)
        gone = _remove_files(subject, found, kind, subject_id)
        ctx.removals.record(
            kind=kind,
            subject_id=subject_id,
            bytes_freed=found.bytes,
            act=read_act(request.headers),
            client=client_agent(request),
        )
        print(
            f"crucible: removed {kind} {subject_id} ({found.bytes / 1e9:.2f} GB) "
            f"from {gone}",
            file=sys.stderr,
        )
        return Response(status_code=204)

    return catalog_remove


def _models_handler(ctx: AppContext):
    async def models() -> list[dict[str, Any]]:
        """Every model this build has a manifest for, and where it stands here."""
        if not ctx.config.enable_llm:
            raise disabled_error("load-model", ctx.config)
        return model_rows(ctx.config, ctx.backend, ctx.residency)

    return models


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    private.get("/catalog")(_catalog_route_handler(ctx))
    private.delete("/catalog/{kind}/{subject_id}", status_code=204)(
        _catalog_remove_handler(ctx)
    )
    private.get("/models")(_models_handler(ctx))
