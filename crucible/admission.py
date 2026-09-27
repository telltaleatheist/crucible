from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .config import Config
from .errors import ApiError
from .inputs import (
    InputSource,
    input_digests,
    journal_identity,
    materialise_inputs,
    refuse_resume_without_a_journal,
)
from .installonsubmit import INSTALLABLE_REFUSALS, InstallOnSubmit
from .jobs import resolve, resolve_model
from .jobs.base import Job
from .jobs.queue import JobStore
from .journal import InputDigest
from .leases import CARD_EFFECTS, Leases
from .residency import Residency
from .upstreams import split_model


@dataclass(frozen=True)
class JobRequest:
    type: str
    model: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    inputs: Mapping[str, InputSource] = field(default_factory=dict)
    client: str | None = None
    client_ref: str | None = None
    hold: bool = False


@dataclass(frozen=True)
class AdmissionContext:
    config: Config
    store: JobStore
    residency: Residency
    leases: Leases
    installs: InstallOnSubmit
    decide_here: Callable[[str], ApiError]


@dataclass(frozen=True)
class AdmittedJob:
    job: Job

    def receipt(self) -> dict[str, Any]:
        return {"job_id": self.job.id, "resume_id": self.job.resume_id}


@dataclass(frozen=True)
class Refusal:
    error: ApiError


async def admit(request: JobRequest, ctx: AdmissionContext) -> AdmittedJob | Refusal:
    try:
        return AdmittedJob(await _admit(request, ctx))
    except ApiError as error:
        return Refusal(error)


def refuse_lease_on_an_upstream(subject_id: str) -> None:
    if split_model(subject_id) is None:
        return
    raise ApiError(
        409,
        "lease_not_needed",
        "an upstream model is never resident; send the chat",
        {"model": subject_id},
    )


def _installing_or_disabled(request: JobRequest, ctx: AdmissionContext, refusal: ApiError) -> ApiError:
    installs = ctx.installs
    if (refusal.details or {}).get("reason") == "undecided" and installs.installable(
        request.type
    ):
        refusal = ctx.decide_here(request.type)
    if not ctx.config.install_on_submit:
        return refusal
    return installs.start(installs.plan(request.type, request.model, refusal))


def _resolved_plugin(request: JobRequest, ctx: AdmissionContext) -> Any:
    try:
        return resolve(ctx.store.registry, request.type, ctx.config)
    except ApiError as refusal:
        if refusal.code != "job_type_disabled":
            raise
        raise _installing_or_disabled(request, ctx, refusal) from None


async def _admit(request: JobRequest, ctx: AdmissionContext) -> Job:
    plugin = _resolved_plugin(request, ctx)
    if request.model is not None:
        refuse_lease_on_an_upstream(request.model)
    model = resolve_model(plugin, request.model)
    refuse_resume_without_a_journal(plugin, request.type, request.params)

    def unloads_what_is_being_cleared() -> bool:
        return (
            model is not None
            and ctx.residency.being_cleared(model)
            and CARD_EFFECTS[request.type].takes_off is not None
        )

    async with ctx.residency.settled_for(
        f"a {request.type} job", same_intent=unloads_what_is_being_cleared
    ):
        ctx.leases.refuse_if_leased(request.type, model)
        ctx.store.refuse_if_busy()
        try:
            plugin.preflight(model, request.params)
        except ApiError as refusal:
            if not (
                ctx.config.install_on_submit and refusal.code in INSTALLABLE_REFUSALS
            ):
                raise
            not_installed = refusal
        else:
            return _created(request, ctx, plugin, model)
    need = ctx.installs.need_for(request.type, model, not_installed)
    if need is None:
        raise not_installed
    raise ctx.installs.start(need)


def _created(
    request: JobRequest, ctx: AdmissionContext, plugin: Any, model: str | None
) -> Job:
    store = ctx.store
    identity = journal_identity(plugin, model, request.params)
    digests: list[InputDigest] = []
    resuming: Any = None
    if identity is not None:
        digests = input_digests(ctx.config, store, request.inputs)
        resume = request.params.get("resume")
        if resume is not None:
            resuming = store.journals.verify(resume, identity, digests)
    job = store.create(
        request.type, model, request.params,
        client=request.client, client_ref=request.client_ref, hold=request.hold,
    )
    fresh: Any = None
    try:
        materialise_inputs(ctx.config, store, job, request.inputs)
        if identity is not None and resuming is None:
            fresh = store.journals.create(identity, digests, job.id)
        store.enqueue(job)
    except ApiError:
        store.discard(job)
        if fresh is not None:
            store.journals.forget_new(fresh)
        raise
    if fresh is not None:
        store.attach_journal(job, fresh.id, resumed=False)
    elif resuming is not None:
        store.journals.adopt(resuming, job.id)
        store.attach_journal(job, resuming.id, resumed=True)
    return job
