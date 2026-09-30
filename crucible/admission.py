from __future__ import annotations

from dataclasses import dataclass, field, replace
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
from .upstreamrecord import split_model

KEEPS_WAITING = frozenset({"server_busy", "leased", "engine_in_use"})


@dataclass(frozen=True)
class JobRequest:
    type: str
    model: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    inputs: Mapping[str, InputSource] = field(default_factory=dict)
    client: str | None = None
    client_ref: str | None = None
    hold: bool = False
    queue: int | None = None


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
    queued: bool | None = None
    position: int | None = None

    def receipt(self) -> dict[str, Any]:
        receipt: dict[str, Any] = {"job_id": self.job.id, "resume_id": self.job.resume_id}
        if self.queued is not None:
            receipt.update(queued=self.queued, position=self.position)
        return receipt


@dataclass(frozen=True)
class Refusal:
    error: ApiError


async def admit(request: JobRequest, ctx: AdmissionContext) -> AdmittedJob | Refusal:
    try:
        job = await _admit(request, ctx)
    except ApiError as error:
        return Refusal(error)
    if request.queue is None:
        return AdmittedJob(job)
    return AdmittedJob(
        job, queued=job.waiting is not None, position=ctx.store.position(job)
    )


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


class _JoinTheLine(Exception):
    pass


def _unloads_what_is_being_cleared(
    request: JobRequest, ctx: AdmissionContext, model: str | None
) -> Callable[[], bool]:
    def same_intent() -> bool:
        return (
            model is not None
            and ctx.residency.being_cleared(model)
            and CARD_EFFECTS[request.type].takes_off is not None
        )

    return same_intent


def _preflight(
    request: JobRequest, ctx: AdmissionContext, plugin: Any, model: str | None
) -> ApiError | None:
    try:
        plugin.preflight(model, request.params)
    except ApiError as refusal:
        if not (ctx.config.install_on_submit and refusal.code in INSTALLABLE_REFUSALS):
            raise
        return refusal
    return None


def _install_for(
    request: JobRequest, ctx: AdmissionContext, model: str | None, missing: ApiError
) -> ApiError:
    need = ctx.installs.need_for(request.type, model, missing)
    if need is None:
        return missing
    return ctx.installs.start(need)


def _lease_holder(ctx: AdmissionContext) -> str | None:
    lease = ctx.leases.current()
    return None if lease is None else lease.client


def refuse_if_line_ahead(request: JobRequest, ctx: AdmissionContext) -> None:
    """Jobs already waiting in the queue go first; only the lease holder's go ahead."""
    line = ctx.store.line
    if line is None or len(line) == 0:
        return
    holder = _lease_holder(ctx)
    if holder is not None and request.client == holder:
        return
    front = line.ordered()[0].job
    depth = len(line)
    raise ApiError(
        409,
        "server_busy",
        f"this server has {depth} job(s) waiting in its queue, the first job "
        f"{front.id} ({front.type}), and a new job goes behind them. Submit with "
        '"queue": {} to take a place in the line, or read GET /v1/queue',
        {**front.busy_details(), "queue_depth": depth},
    )


def _waiting_instead(request: JobRequest, refusal: ApiError) -> bool:
    return request.queue is not None and refusal.code in KEEPS_WAITING


def _refuse_if_busy(request: JobRequest, ctx: AdmissionContext, model: str | None) -> None:
    try:
        ctx.leases.refuse_if_leased(request.type, model)
        ctx.store.refuse_if_busy()
        refuse_if_line_ahead(request, ctx)
    except ApiError as busy:
        if not _waiting_instead(request, busy):
            raise
        raise _JoinTheLine() from busy


def _preflight_or_wait(
    request: JobRequest, ctx: AdmissionContext, plugin: Any, model: str | None
) -> ApiError | None:
    try:
        return _preflight(request, ctx, plugin, model)
    except ApiError as refusal:
        if not _waiting_instead(request, refusal):
            raise
        raise _JoinTheLine() from refusal


async def _admit(request: JobRequest, ctx: AdmissionContext) -> Job:
    plugin = _resolved_plugin(request, ctx)
    if request.model is not None:
        refuse_lease_on_an_upstream(request.model)
    model = resolve_model(plugin, request.model)
    refuse_resume_without_a_journal(plugin, request.type, request.params)
    try:
        async with ctx.residency.settled_for(
            f"a {request.type} job",
            same_intent=_unloads_what_is_being_cleared(request, ctx, model),
        ):
            _refuse_if_busy(request, ctx, model)
            not_installed = _preflight_or_wait(request, ctx, plugin, model)
            if not_installed is None:
                return _created(request, ctx, plugin, model)
    except _JoinTheLine:
        return _created(request, ctx, plugin, model, wait=True)
    except ApiError as refusal:
        if not _waiting_instead(request, refusal):
            raise
        return _created(request, ctx, plugin, model, wait=True)
    raise _install_for(request, ctx, model, not_installed)


async def admit_waiting(waiting: Any, ctx: AdmissionContext) -> ApiError | None:
    """Admit one queued job through the same checks a fresh submit meets.

    None means it is on the lane (or was removed while it was being admitted); a
    refusal says why it is not.
    """
    request = waiting.request
    try:
        plugin = _resolved_plugin(request, ctx)
        model = resolve_model(plugin, request.model)
        async with ctx.residency.settled_for(
            f"a queued {request.type} job",
            same_intent=_unloads_what_is_being_cleared(request, ctx, model),
        ):
            ctx.leases.refuse_if_leased(request.type, model)
            ctx.store.refuse_if_busy()
            not_installed = _preflight(request, ctx, plugin, model)
            if not_installed is None:
                _onto_the_lane(waiting, ctx)
                return None
        return _install_for(request, ctx, model, not_installed)
    except ApiError as refusal:
        return refusal


def _onto_the_lane(waiting: Any, ctx: AdmissionContext) -> None:
    line = ctx.store.line
    if waiting.gone or line is None:
        return
    ctx.store.enqueue(waiting.job, announce=False)
    line.started(waiting)


def _join(
    request: JobRequest, ctx: AdmissionContext, job: Job, fresh: Any
) -> None:
    line = ctx.store.line
    assert line is not None and request.queue is not None
    line.join(job, replace(request, inputs={}), request.queue, fresh)


def _created(
    request: JobRequest,
    ctx: AdmissionContext,
    plugin: Any,
    model: str | None,
    *,
    wait: bool = False,
) -> Job:
    store = ctx.store
    if wait:
        if store.line is None:
            raise ApiError(500, "queue_missing", "this server has no queue attached")
        store.line.refuse_if_full(request.client)
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
        if not wait:
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
    if wait:
        _join(request, ctx, job, fresh)
    elif request.queue is not None:
        store.append_event(job, "started", {"waited_s": 0.0})
    return job
