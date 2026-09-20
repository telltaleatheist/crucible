"""The `tts` job type's lifecycle half: `load-voice` and `unload-voice`.

PHASE3-TTS.md section 5, mirroring `jobs/llm/`'s pair exactly. What is here is
putting a voice on the accelerator and taking it off, which is the thing the
exclusive lane serialises and the thing every other door depends on having got
right. The render door (`{"type": "tts"}`, section 6) is `render.py` beside this
file and the helpers both halves share are `common.py`; the streaming door
(section 7) is still unbuilt.

Every refusal happens **before the job is queued**, in `preflight()`, so the
client gets an HTTP error naming the thing rather than a job that fails a minute
later:

    unknown_voice        no manifest with that id (raised by jobs.resolve_model)
    backend_unsupported  the manifest has no block for this host's backend
    env_missing          ~/.crucible/envs/tts-<engine> is not installed
    voice_not_installed  no weights at the manifest's pinned revision
    accelerator_busy     somebody else's process is on the card
    insufficient_memory  free memory is below the manifest's estimate
    voice_not_resident   (unload) that voice is not the one that is loaded

And three this door added on 2026-09-14, when a zero-shot voice became loadable
(PHASE3-TTS.md section 5's amendment; before it, `kind = "zeroshot"` was
refused outright because narrator's load message carried no clip):

    reference_required     the voice is `kind = "zeroshot"` and the load
                           carries no `params.reference`
    reference_not_allowed  a checkpoint or token voice carries one
    reference_malformed    it is not base64, not a readable WAV, has no
                           transcript, or is over narrator's 30-second budget

**`model` on the wire is the voice id.** DESIGN.md's word for the thing that
produces the bytes is `model`, and for `tts` that thing is the voice — which for
Higgs is the literal truth rather than a pun: a v3 voice *is* the merged
checkpoint the engine was started on, so a voice change is a full worker restart.
No new vocabulary is invented for it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ... import accelerator
from ...config import Config
from ...engines import EngineError
from ...errors import ApiError, JobError
from ..llm import LeaseOnLoad, _open_lease_for_load
from ...residency import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    KIND_TTS,
    Residency,
    describe_resident,
)
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor
from .common import (
    describe_voices,
    known_voice,
    require_loadable,
    require_reference,
    validated_params,
    voice_provenance,
    voice_rows,
)
from .render import TtsJobType, TtsParams

__all__ = [
    "LoadVoiceJobType",
    "LoadVoiceParams",
    "TtsJobType",
    "TtsParams",
    "UnloadVoiceJobType",
    "voice_rows",
]


class ReferenceInput(BaseModel):
    """`params.reference` — the clip a zero-shot voice is loaded with.

    Validated for SHAPE here and for CONTENT in
    `crucible/voicereference.py:parse_reference`, which is where the base64,
    the RIFF header and the 30-second budget are checked and where the
    duration is measured. The split is the same one every other door makes:
    pydantic says what the object is, the module says whether it is usable.

    No `seconds`: the server is holding the bytes and reads the duration off
    the header, and a number the client states about audio the server has is a
    fact with two owners.
    """

    model_config = ConfigDict(extra="forbid")

    #: The wav's bytes, base64, no `data:` prefix.
    data: str
    #: The BOOK-EXACT text spoken in the clip — never an ASR guess. narrator
    #: refuses a clip without one at construction.
    transcript: str
    #: A short label, for whoever reads `/v1/info` and wants to know which of
    #: their clips is resident. Optional; the digest is always there.
    name: str | None = None


class LoadVoiceParams(BaseModel):
    """`params` for a load-voice job. Unknown keys are refused, not ignored."""

    model_config = ConfigDict(extra="forbid")

    timeout_s: float = Field(default=DEFAULT_READY_TIMEOUT_SECONDS, ge=30, le=7200)
    #: Required when the voice's kind is `zeroshot` (`reference_required`),
    #: refused on any other kind (`reference_not_allowed`). Not a pydantic
    #: rule because which it is depends on the MANIFEST rather than on the
    #: body, and a refusal that cannot name the voice is a refusal a client
    #: has to guess at.
    reference: ReferenceInput | None = None
    #: Hold the voice this load makes resident, from the instant it exists.
    #: Same field, same reason and same validators as `load-model`'s — see
    #: `crucible/jobs/llm/__init__.py`'s `LeaseOnLoad`. Absent means today's
    #: behaviour exactly: loaded, and held by nothing.
    #:
    #: It matters MORE here than for a model. `settle.py`'s own "RULING OWED"
    #: says the streaming door is safe only because `load-voice` is: the gap
    #: between `load-voice` finishing and `POST /v1/tts/stream` opening is held
    #: by nothing, and survives today only because a load is not a settlement
    #: trigger. This is that gap closed.
    lease: LeaseOnLoad | None = None


class UnloadVoiceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------- load job


class LoadVoiceJobType:
    """`POST /v1/jobs {"type": "load-voice", "model": "<voice id>"}`."""

    name = "load-voice"

    def __init__(
        self,
        config: Config,
        backend: Any,
        residency: Residency,
        leases: Any | None = None,
    ) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency
        #: See `LoadModelJobType`: the one register, or None where there is no
        #: server, in which case `params.lease` is refused rather than ignored.
        self._leases = leases

    @property
    def residency(self) -> Residency:
        return self._residency

    def describe_models(self) -> list[ModelDescriptor]:
        return describe_voices(self._config, self._residency)

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        return voice_provenance(self._config.backend_kind, model)

    def vram_estimate(self, model: str | None) -> int:
        if model is None:
            raise JobError("model_required", f"{self.name} needs a voice")
        manifest = known_voice(model)
        if not manifest.supports(self._config.backend_kind):
            return 0
        return manifest.spec(self._config.backend_kind).memory_bytes_estimate

    def check(self, backend: Any) -> JobTypeStatus:
        rows = voice_rows(self._config, backend, self._residency)
        ready = [row["id"] for row in rows if row["loadable"]]
        if not ready:
            blocked = sorted(
                {
                    row["reason"]
                    for row in rows
                    if row["reason"] is not None and row["backend_supported"]
                }
            )
            return JobTypeStatus(
                ready=False,
                detail=(
                    "no voice is loadable here: "
                    + ("; ".join(blocked) if blocked else "this build ships none")
                ),
            )
        return JobTypeStatus(ready=True, detail=f"loadable: {ready}")

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        if model is None:  # unreachable: resolve_model requires one
            raise ApiError(400, "model_required", f"{self.name} needs a voice")
        validated = validated_params(LoadVoiceParams, params, self.name)
        # A streaming session holds the resident engine, and loading over it
        # would SIGTERM narrator mid-sentence (PHASE3-TTS.md section 7).
        self._residency.refuse_if_claimed(f"loading {model!r}")
        manifest, spec, _ = require_loadable(self._config, self._backend, model)
        # Before the card is asked about and before anything is queued: whether
        # this load carries what this KIND of voice needs is a fact about the
        # request, and the client can fix it without waiting for a job.
        require_reference(manifest, validated.reference)
        accelerator.guard(
            self._config.backend_kind,
            model_id=model,
            need_bytes=spec.memory_bytes_estimate,
            owned_pids=self._residency.owned_pids(),
            desktop_allowance_bytes=self._config.desktop_allowance_bytes,
            reclaimable_bytes=self._residency.reclaimable_bytes(excluding=model),
        )

    def run(self, job: Job, ctx: JobContext) -> None:
        params = LoadVoiceParams.model_validate(job.params)
        model = job.model
        if model is None:  # unreachable: resolve_model requires one
            raise JobError("model_required", f"{self.name} needs a voice")
        # `/v1/health` says `warming` for the whole job, not only while the
        # engine's readiness is being polled: from the client's side this server
        # is warming a voice from the moment the lane picks the job up.
        self._residency.begin_warming(model)
        try:
            self._load(ctx, model, params, job.client)
        finally:
            self._residency.end_warming()

    def _load(
        self,
        ctx: JobContext,
        model: str,
        params: LoadVoiceParams,
        client: str | None,
    ) -> None:
        """`client` is `job.client`, carried in because a lease records who
        holds it and this helper is the only place with the resident thing in
        hand. Threaded rather than read off a second source: `Job.client` is
        what every other holder is recorded under."""
        try:
            manifest, spec, (python, installed) = require_loadable(
                self._config, self._backend, model
            )
            # Parsed again here rather than carried from `preflight`: a job is
            # a document that survives the process, and re-deriving the clip
            # from the body it holds is what makes the job the one source. The
            # bytes are not decoded twice in any way that matters — one wav,
            # once, as the lane picks the job up.
            reference = require_reference(manifest, params.reference)
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None

        ctx.warming(f"checking the accelerator for {model}")
        try:
            # The card can change between the queue and the lane, so the guard
            # runs again here, against the same rules.
            state = accelerator.guard(
                self._config.backend_kind,
                model_id=model,
                need_bytes=spec.memory_bytes_estimate,
                owned_pids=self._residency.owned_pids(),
                desktop_allowance_bytes=self._config.desktop_allowance_bytes,
                reclaimable_bytes=self._residency.reclaimable_bytes(excluding=model),
            )
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None
        ctx.warming(state.detail)

        # Read on both sides of the load, for `jobs/llm/__init__.py`'s reason:
        # narrator's start is not interruptible, so the only two moments a
        # cancel can be honoured are before the worker is spawned and after it
        # answers. This side refuses to start a voice nobody wants any more.
        ctx.raise_if_cancelled()
        ctx.progress(0.0, f"loading {model}")
        try:
            resident = self._residency.load_voice(
                manifest,
                spec,
                installed.path,
                python,
                reference=reference,
                timeout=params.timeout_s,
                on_progress=ctx.warming,
            )
        except EngineError as exc:
            raise JobError("engine_failed", str(exc)) from None
        # And on the far side, with no teardown of its own: the settlement
        # takes the voice off the card because a load that ended `cancelled` is
        # no longer exempt from it (crucible/settle.py). A stranded VOICE is
        # the same 21 GB as a stranded model, and the streaming door's safety
        # rests on `load-voice` behaving (that module docstring's RULING OWED).
        ctx.raise_if_cancelled()
        ctx.progress(1.0, f"{model} is resident")
        # `reference` on `done` for the same reason it is on the residency
        # report: two clients loading `zeroshot` see one voice id, and the
        # digest is the only thing that says whose clip won.
        extra: dict[str, Any] = {
            "resident": resident.voice_id,
            "fingerprint": resident.fingerprint,
            "reference": resident.reference,
        }
        # STATED EVEN WHEN THERE IS NONE. `null` here is "this load was not
        # asked to hold anything"; an ABSENT key would mean "this server does
        # not speak leases on a load", and a client cannot tell those apart
        # from a hole. The SDK's own rule, applied on the server side of it.
        extra["lease_id"] = None
        if params.lease is not None:
            # AFTER the cancel check above, for `load-model`'s reason: a lease
            # opened before it would be held by a job about to raise
            # `JobCancelled`, and the settlement would find a holder and leave
            # the card stranded behind our own lease until its ttl ran out.
            extra["lease_id"] = _open_lease_for_load(
                self._leases,
                kind=resident.kind,
                subject=resident.voice_id,
                request=params.lease,
                client=client,
            )
        ctx.done_extra(**extra)


# --------------------------------------------------------------- unload job


class UnloadVoiceJobType:
    """`POST /v1/jobs {"type": "unload-voice", "model": "<voice id>"}`."""

    name = "unload-voice"

    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency

    @property
    def residency(self) -> Residency:
        return self._residency

    def describe_models(self) -> list[ModelDescriptor]:
        return describe_voices(self._config, self._residency)

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        return voice_provenance(self._config.backend_kind, model)

    def vram_estimate(self, model: str | None) -> int:
        return 0

    def check(self, backend: Any) -> JobTypeStatus:
        voice = self._residency.resident_voice
        return JobTypeStatus(
            ready=True,
            detail=(
                f"resident: {voice.voice_id}" if voice else "no voice is resident"
            ),
        )

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        if model is None:  # unreachable: resolve_model requires one
            raise ApiError(400, "model_required", f"{self.name} needs a voice")
        validated_params(UnloadVoiceParams, params, self.name)
        if self._residency.being_cleared(model):
            # The same intent, already under way — `unload-model`'s finding
            # (T6, 2026-09-15), and a render's client hits it the same way: the
            # settlement clears the voice the moment the render job ends, and
            # the client's own `unload-voice` lands inside that moment.
            return
        self._residency.refuse_if_claimed(f"unloading {model!r}")
        if not self._residency.is_resident(KIND_TTS, model):
            raise ApiError(
                409,
                "voice_not_resident",
                f"{model!r} is not resident on this server; "
                + describe_resident(self._residency, KIND_TTS, "no voice is"),
                {"requested": model, "resident": self._residency.resident_id},
            )

    def run(self, job: Job, ctx: JobContext) -> None:
        UnloadVoiceParams.model_validate(job.params)
        model = job.model
        if model is None:  # unreachable: resolve_model requires one
            raise JobError("model_required", f"{self.name} needs a voice")
        if self._residency.await_clearance(model):
            # The settlement got there first, which is the card this job asked
            # for. Same terminal shape as an unload this job did itself.
            ctx.progress(0.0, f"unloading {model}")
            ctx.progress(1.0, f"{model} is unloaded — the card was cleared of it")
            ctx.done_extra(resident=self._residency.resident_id)
            return
        if not self._residency.is_resident(KIND_TTS, model):
            # Checked before `unload()` rather than caught from it: the holder
            # unloads by id alone, and a model sharing a voice's id would be
            # taken off the card by `unload-voice`.
            raise JobError(
                "voice_not_resident",
                f"{model!r} is not resident on this server; "
                + describe_resident(self._residency, KIND_TTS, "no voice is"),
            )
        ctx.progress(0.0, f"unloading {model}")
        try:
            self._residency.unload(model)
        except EngineError as exc:
            raise JobError("engine_failed", str(exc)) from None
        ctx.progress(1.0, f"{model} is unloaded")
        ctx.done_extra(resident=self._residency.resident_id)
