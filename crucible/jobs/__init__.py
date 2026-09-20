"""The job-type registry.

`ALL_JOB_TYPES` is the vocabulary this build knows. `build_registry()` is the
subset this server is configured to offer. Asking for a type that exists but is not
enabled is refused differently from asking for one that does not exist — the client
is told which of the two it is.

Since phase 9 the first of those two refusals also carries the NUMBER that turned
the type off, read off the `[capability]` record `crucible install` wrote — see
`disabled_error`, which is the one producer of that sentence for all four doors
that say it.
"""

from __future__ import annotations

from typing import Any

from ..capability import CLASSES, classes_for_job_type
from ..errors import ApiError
from ..residency import Residency
from .base import (
    Job,
    JobContext,
    JobType,
    JobTypeStatus,
    ModelDescriptor,
    validate_member_name,
)
from .align import AlignJobType, UnloadAlignerJobType
from .alignlongform.jobtype import AlignLongformJobType
from .asr import AsrJobType
from .denoise import DenoiseJobType, UnloadDenoiserJobType
from .echo import EchoJobType
from .llm import LoadModelJobType, UnloadModelJobType, model_rows
from .rvc import RvcJobType
from .tts import LoadVoiceJobType, TtsJobType, UnloadVoiceJobType, voice_rows

#: The vocabulary this build knows, and which config flag turns each one on.
#: `resolve()` tells "that type does not exist" from "it exists but is off".
ALL_JOB_TYPES: dict[str, str] = {
    AlignJobType.name: "align",
    UnloadAlignerJobType.name: "align",
    # SHARES `align`'s FLAG and needs `asr`'s env too. Its own `check()` reports
    # which half is missing rather than claiming ready on the aligner alone — a
    # job type that says ready and fails in its first stage is the shape this
    # server spends its refusals avoiding.
    AlignLongformJobType.name: "align",
    AsrJobType.name: "asr",
    DenoiseJobType.name: "denoise",
    UnloadDenoiserJobType.name: "denoise",
    EchoJobType.name: "echo",
    LoadModelJobType.name: "llm",
    UnloadModelJobType.name: "llm",
    LoadVoiceJobType.name: "tts",
    TtsJobType.name: "tts",
    UnloadVoiceJobType.name: "tts",
    RvcJobType.name: "rvc",
}


def build_registry(
    config: Any,
    backend: Any,
    residency: Residency | None = None,
    leases: Any | None = None,
) -> dict[str, JobType]:
    """Instantiate the job types this config enables.

    `residency` is the server instance's one-resident-engine holder, and every
    job type that touches the card is handed the SAME one — that is what makes
    "loading a voice unloads a model" true rather than aspirational. `crucible
    doctor` has no server, so it passes none and gets a fresh (empty) one — which
    is honest: a doctor run cannot see another process's resident engine.
    """
    registry: dict[str, JobType] = {}
    # Hoisted out of the `llm` branch: every job type that runs the accelerator
    # guard needs the same owned-pid set, or it would report Crucible's own
    # resident engine as somebody else's process holding the card. `tts` needs
    # the same holder for a second reason — one card holds one thing, so loading
    # a voice unloads a model and vice versa (PHASE3-TTS.md section 5).
    holder = residency if residency is not None else Residency(config)
    if config.enable_echo:
        registry[EchoJobType.name] = EchoJobType()
    if config.enable_llm:
        # `leases` reaches the LOADERS and nothing else: a load can be asked to
        # hold what it made resident (`params.lease`), which is the one place a
        # job opens a lease rather than refusing against one. `crucible doctor`
        # passes none and the loaders refuse `params.lease` by name.
        registry[LoadModelJobType.name] = LoadModelJobType(
            config, backend, holder, leases
        )
        registry[UnloadModelJobType.name] = UnloadModelJobType(config, backend, holder)
    if config.enable_tts:
        registry[LoadVoiceJobType.name] = LoadVoiceJobType(
            config, backend, holder, leases
        )
        registry[UnloadVoiceJobType.name] = UnloadVoiceJobType(config, backend, holder)
        registry[TtsJobType.name] = TtsJobType(config, backend, holder)
    if config.enable_asr:
        registry[AsrJobType.name] = AsrJobType(config, backend, holder.owned_pids)
    if config.enable_align:
        # The SAME holder: an aligner is a third kind of resident thing, so
        # loading one unloads a model or a voice exactly as those unload each
        # other (PHASE4-AUDIO.md section 2, crucible/residency.py).
        registry[AlignJobType.name] = AlignJobType(config, backend, holder)
        registry[UnloadAlignerJobType.name] = UnloadAlignerJobType(
            config, backend, holder
        )
        # NO HOLDER. align-longform drives the asr and align WORKERS directly
        # (`alignlongform/stages.py`): nested jobs would deadlock, because a
        # Crucible takes one job at a time and a job waiting on a job waits on a
        # lane it is holding itself. It starts and stops its own aligner session
        # rather than borrowing the resident one, so it never evicts what a
        # client has loaded.
        registry[AlignLongformJobType.name] = AlignLongformJobType(config, backend)
    if config.enable_rvc:
        # Only the owned-pid set, like `asr`: nothing is ever resident for
        # `rvc`, whose whole design is a process that exits every 96 files.
        registry[RvcJobType.name] = RvcJobType(config, backend, holder.owned_pids)
    if config.enable_denoise:
        # THE SAME HOLDER, since 2026-09-15: a separator is the FOURTH kind of
        # resident thing, so loading one unloads a model, a voice or an aligner
        # exactly as those unload each other. It used to take only the
        # owned-pid set, on the reasoning "one job, one model load, one exit" —
        # true of one job, and false of the ~44 a book sends
        # (`jobs/denoise/__init__.py`, Owen's ruling).
        #
        # `denoise` shares `rvc`'s ENV but not its flag — a host can have the
        # env and no separator checkpoint.
        registry[DenoiseJobType.name] = DenoiseJobType(config, backend, holder)
        registry[UnloadDenoiserJobType.name] = UnloadDenoiserJobType(
            config, backend, holder
        )
    _assert_every_type_implements_the_protocol(registry)
    return registry


#: What `JobType` asks of a plugin. Read off the Protocol rather than typed out,
#: so adding a member there extends this on its own — which is the whole point,
#: since the bug this guards against was a member added in one branch and a type
#: written in another. `name` is excluded: every type sets its own, and it is the
#: key the registry is built from, so a type without one never gets this far.
#:
#: `typing.Protocol` exposes `__protocol_attrs__` only from 3.12, and this server
#: supports 3.11, so the members are taken from the class body: methods live in
#: `vars()`, annotated attributes in `__annotations__`, and a Protocol has no
#: other kind of member.
_JOB_TYPE_MEMBERS: tuple[str, ...] = tuple(
    sorted(
        {
            name
            for name in list(vars(JobType)) + list(getattr(JobType, "__annotations__", {}))
            if not name.startswith("_") and name != "name"
        }
    )
)


def _assert_every_type_implements_the_protocol(registry: dict[str, JobType]) -> None:
    """Refuse to serve a job type that does not implement all of `JobType`.

    `Protocol` is a static claim, and nothing was checking it: `asr` was written
    against a `JobType` that had six methods, `llm` added a seventh
    (`model_provenance`) in the same week, and the two met in a merge. Nothing
    complained at import, at startup, or when a job was accepted. The failure
    arrived where it could do the most damage — inside `JobStore._finish`, while
    writing a finished job's provenance, on the event loop, so the job emitted no
    terminal event and every client reading its stream waited forever.

    A missing member is a build-time fact, so it is refused at build time, by
    name, before the server answers anything. `runtime_checkable` would only
    check the members that `isinstance` knows about on the instance; this asks
    the Protocol what it declares.
    """
    for job_type, plugin in sorted(registry.items()):
        missing = [m for m in _JOB_TYPE_MEMBERS if not hasattr(plugin, m)]
        if missing:
            raise TypeError(
                f"job type {job_type!r} ({type(plugin).__name__}) does not "
                f"implement JobType: missing {sorted(missing)}. Every member of "
                "the protocol is called by the queue or the API; a type that is "
                "missing one fails somewhere far from here."
            )


#: Every job type must be reachable from some capability class, or `crucible
#: capability` would decide nothing about it and a refusal would have nothing to
#: name. The two tables are in two files because they answer two questions — what
#: can be POSTed, and what has to fit on the card — and this is the check that
#: keeps them from drifting apart the way a job type added in one branch and a
#: class added in another would.
_UNCOVERED = sorted(set(ALL_JOB_TYPES.values()) - {entry.job_type for entry in CLASSES})
if _UNCOVERED:  # pragma: no cover - a build-time impossibility, asserted anyway
    raise TypeError(
        f"job type(s) {_UNCOVERED} have no capability class in "
        "crucible/capability.py, so nothing decides whether this host can run "
        "them and `job_type_disabled` would have no number to name"
    )


#: The capability names — `llm`, `tts`, … — as distinct from the POSTable job
#: types that operate them. `/v1/models` and `/v1/voices` refuse in terms of the
#: capability, `POST /v1/jobs` in terms of the type it was sent, and
#: `disabled_error` takes either so both doors produce one sentence.
CAPABILITIES: frozenset[str] = frozenset(ALL_JOB_TYPES.values())


def disabled_error(name: str, config: Any) -> ApiError:
    """`400 job_type_disabled`, carrying the number that turned the type off.

    `name` is either a job type (`tts`, `load-model`) or the capability behind it
    (`llm`), because the doors that refuse are of both kinds.

    PHASE9-CAPABILITY.md section 2.1. Until 2026-09-13 this said *"set [jobs]
    enable_tts = true in config.toml"*, and on a card that cannot hold Higgs that
    sentence is not merely unhelpful, it is **harmful**: it tells the operator to
    flip a flag whose next effect is an OOM in the middle of somebody's book. A
    refusal that recommends a fix which cannot work is worse than one that just
    says no — the no-band-aids rule applied to an error message, and the whole
    reason `crucible capability` records a reason rather than a boolean.

    There are three genuinely different reasons a type is off, and a reader must
    be able to tell them apart:

    1. **Nothing has been decided here.** No `[capability]` record: a config
       written by `crucible init` alone. Nothing knows whether the host could run
       it, so nothing is claimed — the operator is sent to the probe.
    2. **The card cannot hold it.** Every class behind this flag is recorded
       disabled, and their reasons carry the shortfall. Flipping the flag is
       named as the thing NOT to do.
    3. **The card can hold it and it is simply not installed.** Recorded enabled,
       flag off. This is the one case with an action that works, so it is the one
       case that gets an action.

    This lives here rather than at each raise site because the same sentence was
    being written in four places (`resolve`, `/v1/models`, `/v1/voices` and the
    streaming door) and three of them would have been left behind.
    """
    if name in ALL_JOB_TYPES:
        job_type, capability = name, ALL_JOB_TYPES[name]
    elif name in CAPABILITIES:
        job_type, capability = name, name
    else:
        raise KeyError(
            f"{name!r} is neither a job type nor a capability; the job types are "
            f"{sorted(ALL_JOB_TYPES)} and the capabilities {sorted(CAPABILITIES)}"
        )
    flag = f"enable_{capability}"
    record = getattr(config, "capability", None)
    if record is None:
        return ApiError(
            400,
            "job_type_disabled",
            f"job type {job_type!r} is not enabled on this server, and no "
            f"capability selection has been recorded here — nothing knows whether "
            f"this host can hold the models it needs. Run `crucible capability` to "
            f"find out before turning [jobs] {flag} on; on a card that is too "
            f"small, turning it on buys an OOM instead of a server.",
            {"job_type": job_type, "capability_recorded": False},
        )

    rows = [record.row(entry.name) for entry in classes_for_job_type(capability)]
    known = [row for row in rows if row is not None]
    if not known:
        return ApiError(
            400,
            "job_type_disabled",
            f"job type {job_type!r} is not enabled on this server, and the "
            f"capability record in config.toml has no row for it — it was written "
            f"by an older build. Re-run `crucible capability --write`.",
            {"job_type": job_type, "capability_recorded": False},
        )

    fitting = [row for row in known if row.enabled]
    if fitting:
        return ApiError(
            400,
            "job_type_disabled",
            f"job type {job_type!r} is not enabled on this server, but this host "
            f"can hold it: "
            + "; ".join(f"{row.capability} — {row.reason}" for row in fitting)
            + f". Install it with `crucible install {capability}`.",
            {
                "job_type": job_type,
                "capability_recorded": True,
                "fits": [row.capability for row in fitting],
            },
        )

    return ApiError(
        400,
        "job_type_disabled",
        f"{job_type!r} is disabled on this server: "
        + "; ".join(f"{row.capability} — {row.reason}" for row in known)
        + f". Turning [jobs] {flag} on would not change any of those "
        "numbers; it would only move the failure to the first request.",
        {
            "job_type": job_type,
            "capability_recorded": True,
            "fits": [],
            "shortfall_bytes": {
                row.capability: row.shortfall_bytes for row in known
            },
        },
    )


def resolve(registry: dict[str, JobType], job_type: str, config: Any) -> JobType:
    """Look up a job type or refuse by name."""
    plugin = registry.get(job_type)
    if plugin is not None:
        return plugin
    if job_type in ALL_JOB_TYPES:
        raise disabled_error(job_type, config)
    raise ApiError(
        400,
        "unknown_job_type",
        f"unknown job type {job_type!r}; this server offers "
        f"{sorted(registry) if registry else 'no job types'}",
    )


def resolve_model(plugin: JobType, model: str | None) -> str | None:
    """Check the requested model against what the type advertises, or refuse by name."""
    offered = plugin.describe_models()
    ids = [descriptor.id for descriptor in offered]
    if model is None:
        if ids:
            raise ApiError(
                400,
                "model_required",
                f"job type {plugin.name!r} requires a model; it offers {ids}",
            )
        return None
    if model not in ids:
        raise ApiError(
            400,
            "unknown_model",
            f"job type {plugin.name!r} does not serve model {model!r}; it offers "
            f"{ids if ids else 'no models (omit `model`)'}",
        )
    return model


__all__ = [
    "ALL_JOB_TYPES",
    "AlignJobType",
    "AsrJobType",
    "DenoiseJobType",
    "EchoJobType",
    "Job",
    "JobContext",
    "JobType",
    "JobTypeStatus",
    "LoadModelJobType",
    "LoadVoiceJobType",
    "ModelDescriptor",
    "TtsJobType",
    "Residency",
    "RvcJobType",
    "UnloadAlignerJobType",
    "UnloadDenoiserJobType",
    "UnloadModelJobType",
    "UnloadVoiceJobType",
    "build_registry",
    "disabled_error",
    "model_rows",
    "resolve",
    "resolve_model",
    "validate_member_name",
    "voice_rows",
]
