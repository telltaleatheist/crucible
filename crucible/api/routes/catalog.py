from __future__ import annotations

import sys
from typing import Any

from fastapi import Request, Response

from ... import catalog, weights
from ...errors import ApiError, CrucibleError
from ...inflight import read_act
from ...jobs import disabled_error, model_rows
from ...residency import KIND_NOUNS
from ..caller import client_agent
from ..context import AppContext, Routers


def _task_names(task: Any, kind: str, subject_id: str) -> bool:
    body = task.request
    if body.get("kind") == kind and body.get("id") == subject_id:
        return True
    module = body.get("module")
    if isinstance(module, dict):
        for entry in module.get("subjects", []) or []:
            if (
                isinstance(entry, dict)
                and entry.get("kind") == kind
                and entry.get("id") == subject_id
            ):
                return True
    return False


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    config, backend, residency = ctx.config, ctx.backend, ctx.residency

    @private.get("/catalog")
    async def catalog_route() -> dict[str, Any]:
        """Every subject this backend can hold, installed or not."""
        return {"rows": catalog.rows(config, backend, residency),
                "backend_kind": backend.kind}

    @private.delete("/catalog/{kind}/{subject_id}", status_code=204)
    async def catalog_remove(
        request: Request, kind: str, subject_id: str
    ) -> Response:
        """Delete an installed subject's files. Refused while the subject is resident,
        leased or named by a running task.
        """
        if kind not in catalog.KINDS:
            raise ApiError(
                404,
                "subject_unknown",
                f"{kind!r} is not a subject kind; they are {list(catalog.KINDS)}",
                {"kind": kind, "id": subject_id},
            )
        subject = catalog.find(config, backend, kind, subject_id)
        if subject is None:
            raise ApiError(
                404,
                "subject_unknown",
                f"this server has no {kind} called {subject_id!r} for "
                f"{backend.kind}. GET /v1/catalog lists every subject it can hold",
                {"kind": kind, "id": subject_id},
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
                {"kind": kind, "id": subject_id},
            )
        who = _subject_holder(subject)
        if who is not None:
            raise ApiError(
                409,
                "subject_in_use",
                f"{kind} {subject_id!r} cannot be removed: {who['who']}. "
                "Deleting the files under a running engine would leave it "
                "serving a model that is no longer on the disk",
                who,
            )
        try:
            gone = subject.remove()
        except weights.WeightsShared as exc:
            raise ApiError(
                409,
                "weights_shared",
                str(exc),
                {
                    "kind": kind,
                    "id": subject_id,
                    "backend": exc.backend,
                    "aliases": list(exc.aliases),
                },
            ) from None
        except weights.RemoveFailed as exc:
            raise ApiError(
                500,
                "subject_remove_failed",
                str(exc),
                {"kind": kind, "id": subject_id, "path": str(exc.path)},
            ) from None
        except (CrucibleError, OSError) as exc:
            raise ApiError(
                500,
                "subject_remove_failed",
                f"removing {kind} {subject_id!r} failed: "
                f"{type(exc).__name__}: {exc}",
                {"kind": kind, "id": subject_id, "path": str(found.path)},
            ) from None
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

    def _subject_holder(subject: catalog.Subject) -> dict | None:
        readers = catalog.ids_reading(subject)

        def through(holder_id: str) -> str:
            return (
                ""
                if holder_id == subject.id
                else f" ({holder_id}, which shares its weights)"
            )

        resident = residency.resident
        if resident is not None and resident.id in readers:
            return {
                "kind": subject.kind,
                "id": subject.id,
                "who": (
                    f"it is the {KIND_NOUNS[resident.kind]} on the card right "
                    f"now{through(resident.id)}; unload it first"
                ),
                "fact": "resident",
            }
        lease = ctx.leases.current()
        if lease is not None and lease.subject in readers:
            return {
                "kind": subject.kind,
                "id": subject.id,
                "who": (
                    f"{lease.client or 'a client'} holds a lease on it"
                    f"{through(lease.subject)} "
                    f"until {lease.expires_at.isoformat()}"
                ),
                "fact": "lease",
                **lease.receipt(),
            }
        running = ctx.tasks.running
        if running is not None and any(
            _task_names(running, subject.kind, reader) for reader in readers
        ):
            return {
                "kind": subject.kind,
                "id": subject.id,
                "who": f"task {running.id} ({running.type}) names it",
                "fact": "task",
                "task_id": running.id,
            }
        return None

    @private.get("/models")
    async def models() -> list[dict[str, Any]]:
        """Every model this build has a manifest for, and where it stands here."""
        if not config.enable_llm:
            raise disabled_error("load-model", config)
        return model_rows(config, backend, residency)
