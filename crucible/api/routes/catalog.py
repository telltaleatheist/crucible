from __future__ import annotations

import sys
from typing import Any

from fastapi import Request, Response

from ... import catalog, weights
from ...config import Config
from ...errors import ApiError, CrucibleError
from ...inflight import read_act
from ...jobs import disabled_error, model_rows
from ...residency import KIND_NOUNS
from ..context import AppContext, Routers


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    config, backend, residency = ctx.config, ctx.backend, ctx.residency

    # --------------------------------------------------------------- catalog

    @private.get("/catalog")
    async def catalog_route(request: Request) -> dict[str, Any]:
        """Every subject this backend can hold, installed or not.

        PHASE13-OPERATOR.md section 3.2. Every field is derived from something
        this server already owns and no row is authored here — see
        `crucible/catalog.py`, which is the whole of it.
        """
        live: Config = request.app.state.config
        # `backend_kind` on every backend, not only where the list is empty
        # (PHASE15-HOST.md section 3.5). A reader that had to infer "there are
        # no rows because there is no card" from the emptiness would be
        # guessing, and the same key on cuda-linux is what makes this a field
        # rather than a marker for one mode.
        return {"rows": catalog.rows(live, backend, residency),
                "backend_kind": backend.kind}

    @private.delete("/catalog/{kind}/{subject_id}", status_code=204)
    async def catalog_remove(
        request: Request, kind: str, subject_id: str
    ) -> Response:
        """Delete an installed subject's files. PHASE15-HOST.md 3.5a.

        THE DOOR THE WEIGHTS RULE NEEDS. 3.5: a subject is never stored twice
        on one machine, so when the guest has its own copy the Windows one
        goes — and the host must never reach into `crucible/weights.py`'s
        layout from outside to do it, because a layout with two owners is the
        shape ARCHITECTURE.md R1 is about. So the server that owns the disk
        owns the deletion, and this is how it is asked.

        THE ORDER OF THE REFUSALS IS THE JOB DOOR'S: what is wrong with the
        REQUEST first (an unknown kind or id is true whatever this server is
        doing), then what is wrong with this server's STATE. A caller who
        misspelled a subject id and was told "it is in use" would fix the
        wrong thing.
        """
        live: Config = request.app.state.config
        if kind not in catalog.KINDS:
            raise ApiError(
                404,
                "subject_unknown",
                f"{kind!r} is not a subject kind; they are {list(catalog.KINDS)}",
                {"kind": kind, "id": subject_id},
            )
        subject = catalog.find(live, backend, kind, subject_id)
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
        who = _subject_holder(request, subject)
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
            # PHASE22 section 2.9: a base's folder is also an alias's. 409 and
            # not 500 — nothing is broken, the subject is HELD, by models this
            # refusal names so a caller can remove them first.
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
        request.app.state.removals.record(
            kind=kind,
            subject_id=subject_id,
            bytes_freed=found.bytes,
            act=read_act(request.headers),
            client=request.headers.get("user-agent"),
        )
        print(
            f"crucible: removed {kind} {subject_id} ({found.bytes / 1e9:.2f} GB) "
            f"from {gone}",
            file=sys.stderr,
        )
        return Response(status_code=204)

    def _subject_holder(request: Request, subject: catalog.Subject) -> dict | None:
        """Why this subject may not be deleted right now, or None.

        THREE HOLDS, and each is read from the thing that owns it rather than
        inferred: the RESIDENCY (this subject is the thing on the card), the
        LEASES (a client has said it intends a run on it), and the TASK STORE
        (a pull or a module naming it is in flight). The four facts'
        `Settlement.holder` is deliberately NOT what is asked — it answers
        "is the card busy at all", and a `denoise` separator on disk is not
        made undeletable by a `tts` render.
        """
        # WHO READS THESE FILES (PHASE22 section 2.9): the subject, and every
        # alias whose weights are its download. A `qwen3.5-9b-vl` on the card
        # is serving `qwen3.5-9b`'s folder, so it holds the base exactly as
        # the base itself would — whether or not the alias was ever pulled as
        # itself (on cuda-linux it is installed by the base's pull alone).
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
        lease = request.app.state.leases.current()
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
        running = request.app.state.tasks.running
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

    def _task_names(task: Any, kind: str, subject_id: str) -> bool:
        """Does this running task name this subject? Read off its REQUEST.

        The request is the task's own echo of what was asked (`Task.request`),
        so this needs no second table of which task types touch which
        subjects — a `pull` names one, a `module` names a list, and an
        `install` names none.
        """
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

    # ---------------------------------------------------------------- models

    @private.get("/models")
    async def models(request: Request) -> list[dict[str, Any]]:
        """Every model this build has a manifest for, and where it stands here."""
        if not config.enable_llm:
            # The same sentence the job door refuses with, from the same
            # producer: a client told "llm is off" by /v1/models and something
            # else by POST /v1/jobs would have two stories about one server
            # (PHASE9-CAPABILITY.md section 2.1).
            raise disabled_error("load-model", config)
        return model_rows(config, backend, residency)
