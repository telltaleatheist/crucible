from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

from .accelerator import WAITS_FOR_THE_CARD
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
from .jobtypes import CARD_EFFECTS
from .journal import InputDigest
from .queuesessions import QueueSessions
from .residency import Residency
from .upstreamrecord import split_model

BUSY = frozenset({"server_busy", "engine_in_use"})
KEEPS_WAITING = BUSY | WAITS_FOR_THE_CARD
"""Refusals that mean "not yet": a request that may wait keeps (or takes) its place in
the line instead of ending. ``BUSY`` waits for the lane or the engine; the card's
``accelerator_busy`` waits for a process Crucible does not own to let go of memory, and
is recorded on the waiting item and re-checked on a pace (WaitingLine.not_yet)."""


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
    from_the_line: bool = False
    session: str | None = None


@dataclass(frozen=True)
class AdmissionContext:
    config: Config
    store: JobStore
    residency: Residency
    sessions: QueueSessions
    installs: InstallOnSubmit
    decide_here: Callable[[str], ApiError]
    chats_in_flight: Callable[[], int] = lambda: 0
    streaming: Callable[[], bool] = lambda: False


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


def refuse_an_upstream_model(subject_id: str) -> None:
    if split_model(subject_id) is None:
        return
    raise ApiError(
        409,
        "upstream_never_resident",
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
    def __init__(self, why: ApiError | None = None) -> None:
        super().__init__()
        self.why = why


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


async def _preflight(
    request: JobRequest, ctx: AdmissionContext, plugin: Any, model: str | None
) -> ApiError | None:
    """A job type's preflight, run off the event loop. It reads the card (`nvidia-smi`,
    over a second under WSL2) and the env (`pip list` on its first read), and every
    route, SSE stream and chat proxy shares the loop (crucible/loopwatch.py). It runs
    outside ``settled_for``: that holds the residency's claim lock on the loop's
    thread, and a preflight that takes the same lock from another thread would wait on
    it forever. The checks that decide a place on the lane are made again inside it."""
    try:
        await asyncio.to_thread(plugin.preflight, model, request.params)
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


def refuse_if_line_ahead(request: JobRequest, ctx: AdmissionContext) -> None:
    """Jobs already waiting in the queue go first. An item of the open session waits only
    behind the session's own items."""
    line = ctx.store.line
    if line is None or len(line) == 0:
        return
    ahead = line.ordered()
    if request.session is not None:
        ahead = [item for item in ahead if item.session == request.session]
        if not ahead:
            return
    front = ahead[0].job
    depth = len(ahead)
    raise ApiError(
        409,
        "server_busy",
        f"this server has {depth} job(s) waiting in its queue, the first job "
        f"{front.id} ({front.type}), and a new job goes behind them. Leave out "
        '"queue": false to take a place in the line, or read GET /v1/queue',
        {**front.busy_details(), "queue_depth": depth},
    )


def chats_hold_the_card(job_type: str, chats_in_flight: int) -> bool:
    """A job that would change what is on the card, while completions are in flight
    on what is there now."""
    if chats_in_flight == 0:
        return False
    effect = CARD_EFFECTS.get(job_type)
    return effect is None or (
        effect.makes_resident is not None or effect.takes_off is not None
    )


def _waiting_instead(request: JobRequest, refusal: ApiError) -> bool:
    return request.queue is not None and refusal.code in KEEPS_WAITING


def _refuse_if_busy(request: JobRequest, ctx: AdmissionContext, model: str | None) -> None:
    try:
        ctx.sessions.refuse_if_held(request.session, f"a {request.type} job")
        ctx.store.refuse_if_busy()
        if not request.from_the_line:
            refuse_if_line_ahead(request, ctx)
    except ApiError as busy:
        if not _waiting_instead(request, busy):
            raise
        raise _JoinTheLine(busy) from busy
    if request.queue is not None and chats_hold_the_card(
        request.type, ctx.chats_in_flight()
    ):
        raise _JoinTheLine()


async def _preflight_or_wait(
    request: JobRequest, ctx: AdmissionContext, plugin: Any, model: str | None
) -> ApiError | None:
    try:
        return await _preflight(request, ctx, plugin, model)
    except ApiError as refusal:
        if not _waiting_instead(request, refusal):
            raise
        raise _JoinTheLine(refusal) from refusal


async def _admit(request: JobRequest, ctx: AdmissionContext) -> Job:
    plugin = _resolved_plugin(request, ctx)
    if request.model is not None:
        refuse_an_upstream_model(request.model)
    model = resolve_model(plugin, request.model)
    refuse_resume_without_a_journal(plugin, request.type, request.params)
    same_intent = _unloads_what_is_being_cleared(request, ctx, model)
    try:
        async with ctx.residency.settled_for(f"a {request.type} job", same_intent=same_intent):
            _refuse_if_busy(request, ctx, model)
        not_installed = await _preflight_or_wait(request, ctx, plugin, model)
        async with ctx.residency.settled_for(f"a {request.type} job", same_intent=same_intent):
            _refuse_if_busy(request, ctx, model)
            if not_installed is None:
                return _created(request, ctx, plugin, model)
    except _JoinTheLine as joining:
        return _created(request, ctx, plugin, model, wait=True, why=joining.why)
    except ApiError as refusal:
        if not _waiting_instead(request, refusal):
            raise
        return _created(request, ctx, plugin, model, wait=True, why=refusal)
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
        what = f"a queued {request.type} job"
        same_intent = _unloads_what_is_being_cleared(request, ctx, model)
        async with ctx.residency.settled_for(what, same_intent=same_intent):
            ctx.sessions.refuse_if_held(request.session, what)
            ctx.store.refuse_if_busy()
        not_installed = await _preflight(request, ctx, plugin, model)
        async with ctx.residency.settled_for(what, same_intent=same_intent):
            ctx.sessions.refuse_if_held(request.session, what)
            ctx.store.refuse_if_busy()
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
    request: JobRequest, ctx: AdmissionContext, job: Job, fresh: Any,
    why: ApiError | None,
) -> None:
    line = ctx.store.line
    assert line is not None and request.queue is not None
    item = line.join(job, replace(request, inputs={}), request.queue, fresh)
    if why is not None:
        line.not_yet(item, why)


def _created(
    request: JobRequest,
    ctx: AdmissionContext,
    plugin: Any,
    model: str | None,
    *,
    wait: bool = False,
    why: ApiError | None = None,
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
        session=request.session,
    )
    if request.session is not None:
        ctx.sessions.adopt_job(request.session, job.id)
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
        _join(request, ctx, job, fresh, why)
    elif request.queue is not None:
        store.append_event(job, "started", {"waited_s": 0.0})
    return job
