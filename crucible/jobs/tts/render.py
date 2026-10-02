from __future__ import annotations

import base64
import binascii
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ... import hosttools, jobenv
from ...cardkinds import KIND_TTS
from ...config import Config
from ...engines import EngineError, EngineWouldNotStop, NarratorEngine
from ...errors import ApiError, JobCancelled, JobError
from ...jobtypes import TTS_JOB
from ...narratorvoices import take_sampling
from ...residency import Residency, describe_resident
from ...voices import VoiceManifest
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor
from ..template import as_job_error, card_guard, parse_params, require_model, run_model
from .common import (
    describe_voices,
    known_voice,
    occupy_voice,
    require_loadable,
    voice_load_plan,
    voice_provenance,
    voice_rows,
)

__all__ = ["TtsChunk", "TtsJobType", "TtsParams"]

JOB_TYPE = TTS_JOB.name

RENDER_SILENCE_TIMEOUT_SECONDS = 600.0

ENCODE_TIMEOUT_SECONDS = 120.0

DURATION_TOLERANCE_SECONDS = 0.05


class TtsChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    text: str

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if value.strip() == "":
            raise ValueError(
                "a chunk's text must not be blank; narrator refuses an empty "
                "generate with a whole-request error, which would end the batch"
            )
        return value


class TtsParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str
    take: int = Field(ge=0)
    chunks: list[TtsChunk] = Field(min_length=1)
    retake: bool = False
    band: dict[str, Any] | None = None
    width: int | None = Field(default=None, ge=1)

    @field_validator("language")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if value.strip() == "":
            raise ValueError("language must not be blank")
        return value

    @model_validator(mode="after")
    def indices_are_unique(self) -> "TtsParams":
        seen: set[int] = set()
        duplicates: set[int] = set()
        for chunk in self.chunks:
            if chunk.index in seen:
                duplicates.add(chunk.index)
            seen.add(chunk.index)
        if duplicates:
            raise ValueError(
                f"chunk index(es) {sorted(duplicates)} appear more than once. An index is "
                "an artifact name, so two chunks sharing one would be two renders "
                "writing the same FLAC and one of them winning silently"
            )
        return self


FFMPEG_WHY = (
    "encodes every chunk through it: narrator returns base64 PCM16 and the "
    "artifact BookForge's assembly expects is a mono FLAC. The alternative is "
    "linking libsndfile into the server's own interpreter, which is a "
    "compiled audio dependency in a process that deliberately imports no "
    "engine at all."
)


_BAND_ON_THE_WIRE: dict[str, str] = {
    "pace_chars_per_sec": "paceCharsPerSec",
    "max_chars_per_sec": "maxCharsPerSec",
    "min_chars_per_sec": "minCharsPerSec",
}


def _require_band(params: TtsParams) -> dict[str, float] | None:
    band = params.band
    if band is None:
        if params.retake:
            raise _retake_without_band()
        return None
    _require_band_keys(band)
    rates = {key: _band_rate(band, key) for key in _BAND_ON_THE_WIRE}
    _require_band_order(band, rates)
    return {wire: rates[key] for key, wire in _BAND_ON_THE_WIRE.items()}


def _retake_without_band() -> ApiError:
    return ApiError(
        400,
        "retake_without_band",
        "retake is true and this request states no band. The guarded "
        "arm re-rolls a chunk against a pace band, and the band is the "
        "caller's to state: Crucible does not read one off the voice, "
        "because an inherited band is indistinguishable from a measured "
        "one at the point of use (deathstalker carried pace 16.64 onto "
        "weights that measured 15.91). Send `band` "
        f"{{{', '.join(_BAND_ON_THE_WIRE)}}}, or render bare",
        {"retake": True},
    )


def _require_band_keys(band: dict[str, Any]) -> None:
    missing = sorted(set(_BAND_ON_THE_WIRE) - set(band))
    unknown = sorted(set(band) - set(_BAND_ON_THE_WIRE))
    if missing or unknown:
        raise ApiError(
            400,
            "band_malformed",
            f"band is {band!r}. A band is the measured pace and the two edges "
            f"derived from it, and it travels as one statement: "
            f"{sorted(_BAND_ON_THE_WIRE)}, all three"
            + (f". Missing {missing}" if missing else "")
            + (f". Unknown {unknown}" if unknown else ""),
            {"band": band},
        )


def _band_rate(band: dict[str, Any], key: str) -> float:
    value = band[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError(
            400,
            "band_malformed",
            f"band.{key} is {value!r}, which is not a rate in characters "
            "per second",
            {"band": band, "field": key},
        )
    if value <= 0:
        raise ApiError(
            400,
            "band_malformed",
            f"band.{key} is {value}. Every rate is characters per second "
            "and positive; a zero or a negative is a band nobody finished "
            "writing, not a band with no floor",
            {"band": band, "field": key},
        )
    return float(value)


def _require_band_order(band: dict[str, Any], rates: dict[str, float]) -> None:
    if not (
        rates["min_chars_per_sec"]
        < rates["pace_chars_per_sec"]
        < rates["max_chars_per_sec"]
    ):
        raise ApiError(
            400,
            "band_malformed",
            f"band is min {rates['min_chars_per_sec']}, pace "
            f"{rates['pace_chars_per_sec']}, max {rates['max_chars_per_sec']}, "
            "which are out of order; the band is min < pace < max, because "
            "narrator keeps only the band's RATIOS and re-centres them on the "
            "book's running median",
            {"band": band},
        )


def _require_width(
    manifest: VoiceManifest, spec: Any, params: TtsParams
) -> int | None:
    if params.width is None:
        return None
    serving = manifest.serving
    if serving is None:
        return params.width
    if jobenv.tts_env(manifest.narrator_engine, spec.backend).serving_stack is None:
        return params.width
    if params.width > serving.max_num_seqs:
        raise ApiError(
            400,
            "width_over_serving",
            f"this job asks for {params.width} chunk(s) in flight and "
            f"{manifest.id!r} was configured for {serving.max_num_seqs} "
            f"([voice.serving].max_num_seqs, which is the server's admission "
            f"width and the width of narrator's own batch). A job cannot run "
            f"wider than the engine was started; ask for "
            f"{serving.max_num_seqs} or fewer, or change the voice's serving "
            f"width and reload it. Never clamped: a job that thought it was "
            f"running {params.width} wide and was not would report a number "
            f"nobody can reproduce",
            {"width": params.width, "max_num_seqs": serving.max_num_seqs},
        )
    return params.width


def _require_renderable(
    config: Config, backend: Any, voice_id: str, params: TtsParams, resident: bool
) -> tuple[VoiceManifest, Any, Any, dict[str, float] | None, int | None]:
    manifest, spec, interpreter = require_loadable(config, backend, voice_id)

    if manifest.kind == "zeroshot" and not resident:
        raise ApiError(
            400,
            "voice_kind_unsupported",
            f"voice {voice_id!r} is a zeroshot voice and is not resident. A "
            "render job loads its own voice, and a zero-shot load needs the "
            "reference clip only `load-voice` carries (`params.reference`): "
            "rendering without it would be a whole book in the base model's "
            "voice, reported as success. Load it first, then render",
            {"voice": voice_id, "kind": manifest.kind, "resident": False},
        )


    return (
        manifest, spec, interpreter,
        _require_band(params), _require_width(manifest, spec, params),
    )


def _require_item_take(engine: NarratorEngine, take: int, sampling: Any) -> None:
    if take > 0 and not engine.announces_item_take():
        raise JobError(
            "sampling_not_wired",
            f"take {take} resolves to sampling {sampling}, and the "
            f"narrator serving this voice did not announce `itemTake` "
            f"on its ready line — it has no per-item rung channel, so it "
            f"would render take 0, in take 0's seed lane, and this job "
            f"would report take {take}. Re-resolve the tts env's "
            f"narrator pin (envs/tts/*.txt) to a bookforge commit that "
            f"carries narrator/engine/item_sampling.py, reinstall the env, "
            f"and reload the voice. Take 0 renders on this narrator as it "
            f"is.",
        )


def _batch_request(
    params: TtsParams,
    sampling: Any,
    band: dict[str, float] | None,
    width: int | None,
) -> dict[str, Any]:
    return {
        "action": "generate_batch",
        "language": params.language,
        "retake": params.retake,
        **({} if band is None else {"band": band}),
        **({} if width is None else {"width": width}),
        "items": [
            {"i": chunk.index, "text": chunk.text, "take": params.take}
            | ({} if sampling is None else {"sampling": sampling})
            for chunk in params.chunks
        ],
    }


_QUIET_BATCH_MESSAGES = frozenset({"batch_done", "stopped"})


class _BatchTally:

    def __init__(self, expected: set[int], total: int) -> None:
        self.expected = expected
        self.total = total
        self.answered: set[int] = set()
        self.failures: list[dict[str, Any]] = []
        self.rendered = 0

    def counts(self) -> dict[str, int]:
        return {"rendered": self.rendered, "failed": len(self.failures), "total": self.total}

    def claim(self, message: dict[str, Any]) -> int | None:
        kind = message["type"]
        if kind in _QUIET_BATCH_MESSAGES:
            return None
        if kind != "batch_item":
            raise JobError(
                "narrator_protocol",
                f"narrator sent a {kind!r} message during a non-streamed "
                "generate_batch; this door asked for whole rows and knows "
                "only batch_item and batch_done",
            )
        try:
            index = _row_index(message, self.expected)
        except EngineError as exc:
            raise JobError("narrator_protocol", str(exc)) from None
        if index in self.answered:
            raise JobError(
                "narrator_protocol",
                f"narrator answered row {index} twice. One answer per row is "
                "narrator's own guarantee, and two would mean one FLAC "
                "overwriting another",
            )
        self.answered.add(index)
        return index

    def record(self, index: int, failed: str | None) -> None:
        if failed is None:
            self.rendered += 1
        else:
            self.failures.append({"index": index, "message": failed})

    def row_line(self, index: int, failed: str | None) -> str:
        head = (
            f"chunk {index} failed: {failed}"
            if failed is not None
            else f"{self.rendered} of {self.total} chunk(s) rendered"
        )
        return head + (f"; {len(self.failures)} failed so far" if self.failures else "")

    def require_complete(self) -> None:
        missing = sorted(self.expected - self.answered)
        if missing:
            raise JobError(
                "narrator_protocol",
                f"narrator said batch_done with {len(missing)} row(s) unanswered: "
                f"{missing[:20]}. One answer per row is its own guarantee, so a "
                "short batch is not a short answer",
            )

    def final_line(self) -> str:
        return f"{self.rendered} of {self.total} chunk(s) rendered" + (
            f"; {len(self.failures)} failed" if self.failures else ""
        )


def encode_flac(ffmpeg: str, pcm: bytes, sample_rate: int, destination: Path) -> None:
    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-f", "s16le",
            "-ar", str(sample_rate),
            "-ac", "1",
            "-i", "pipe:0",
            "-c:a", "flac",
            "-y",
            str(destination),
        ],
        input=pcm,
        capture_output=True,
        timeout=ENCODE_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise JobError(
            "flac_encode_failed",
            f"ffmpeg exited {completed.returncode} encoding {destination.name}: "
            + (completed.stderr.decode("utf-8", "replace").strip() or "no output"),
        )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise JobError(
            "flac_encode_failed",
            f"ffmpeg exited 0 but wrote no {destination.name}",
        )


class _RowFailure(Exception):
    ...


def _row_index(row: dict[str, Any], expected: set[int]) -> int:
    index = row.get("i")
    if not isinstance(index, int) or isinstance(index, bool):
        raise EngineError(
            f"narrator sent a batch_item whose `i` is {index!r}, which is not a "
            "row index. Rows retire out of order, so `i` is the only thing that "
            "says which chunk a reply is about"
        )
    if index not in expected:
        raise EngineError(
            f"narrator sent a batch_item for row {index}, which this job did not "
            f"ask for. It asked for {len(expected)} chunk(s)"
        )
    return index


def _pcm_of(row: dict[str, Any], index: int, sample_rate: int) -> tuple[bytes, float]:
    if row.get("format") != "pcm16":
        raise _RowFailure(
            f"narrator sent format {row.get('format')!r}, not 'pcm16'; Crucible "
            "encodes signed 16-bit little-endian mono and will not guess at "
            "another layout"
        )
    reported_rate = row.get("sampleRate")
    if reported_rate != sample_rate:
        raise _RowFailure(
            f"narrator rendered this row at {reported_rate!r} Hz while the voice "
            f"was loaded at {sample_rate}. The bytes are at one rate and the FLAC "
            "header would claim the other; Crucible refuses rather than resamples"
        )
    payload = row.get("data")
    if not isinstance(payload, str):
        raise _RowFailure(f"narrator sent no base64 audio for row {index}")
    try:
        pcm = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _RowFailure(f"narrator's audio for row {index} is not base64: {exc}")
    if not pcm:
        raise _RowFailure(f"narrator returned no audio for row {index}")
    if len(pcm) % 2:
        raise _RowFailure(
            f"narrator returned {len(pcm)} bytes for row {index}, which is not a "
            "whole number of 16-bit samples"
        )
    measured = len(pcm) / 2 / sample_rate
    reported = row.get("duration")
    if not isinstance(reported, (int, float)) or isinstance(reported, bool):
        raise _RowFailure(
            f"narrator reported duration {reported!r} for row {index}, which is "
            "not a duration"
        )
    if abs(measured - float(reported)) > DURATION_TOLERANCE_SECONDS:
        raise _RowFailure(
            f"narrator reported {float(reported):.3f}s for row {index} but sent "
            f"{measured:.3f}s of audio. A reply that describes audio other than "
            "the audio attached to it is not a measurement"
        )
    return pcm, measured


def _optional_int(row: dict[str, Any], key: str) -> int | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise _RowFailure(f"narrator sent {key}={value!r}, which is not a count")
    return value


def _optional_bool(row: dict[str, Any], key: str) -> bool | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise _RowFailure(f"narrator sent {key}={value!r}, which is not a flag")
    return value


def _seconds_of(cut: dict[str, Any], key: str, at: int) -> float:
    value = cut.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise _RowFailure(
            f"narrator sent pauseCuts[{at}].{key}={value!r}, which is not a number "
            "of seconds"
        )
    return float(value)


def _pause_cuts_of(row: dict[str, Any]) -> list[dict[str, float]] | None:
    """The interior pauses narrator cut down in this chunk ({atS, fromS, toS}, seconds;
    [] when it cut none), or None when narrator did not say (an older narrator)."""
    cuts = row.get("pauseCuts")
    if cuts is None:
        return None
    if not isinstance(cuts, list):
        raise _RowFailure(f"narrator sent pauseCuts={cuts!r}, which is not a list")
    read = []
    for at, cut in enumerate(cuts):
        if not isinstance(cut, dict):
            raise _RowFailure(f"narrator sent pauseCuts[{at}]={cut!r}, which is not an object")
        read.append({
            "at_s": _seconds_of(cut, "atS", at),
            "from_s": _seconds_of(cut, "fromS", at),
            "to_s": _seconds_of(cut, "toS", at),
        })
    return read


def _guard_of(row: dict[str, Any]) -> dict[str, Any] | None:
    guard = row.get("guard")
    if guard is None:
        return None
    if not isinstance(guard, dict):
        raise _RowFailure(
            f"narrator sent guard={guard!r}, which is not an object. Crucible "
            "forwards the verdict verbatim and reads nothing inside it, but the "
            "`chunk` event's field is an object or null and this is neither"
        )
    return guard


class TtsJobType:
    name = JOB_TYPE

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
        manifest = known_voice(run_model(model, self.name, "a voice"))
        if not manifest.supports(self._config.backend_kind):
            return 0
        return manifest.spec(self._config.backend_kind).memory_bytes_estimate

    def check(self, backend: Any) -> JobTypeStatus:
        if hosttools.ffmpeg_path() is None:
            return JobTypeStatus(
                ready=False,
                detail=(
                    "there is no ffmpeg on PATH, and tts encodes every chunk "
                    "through it. " + hosttools.searched_note()
                ),
            )
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
                    "no voice is renderable here: "
                    + ("; ".join(blocked) if blocked else "this build ships none")
                ),
            )
        return JobTypeStatus(ready=True, detail=f"renderable: {ready}")


    def requirements(
        self, voice_id: str, params: TtsParams, resident: bool
    ) -> tuple[str, VoiceManifest, Any, Any, dict[str, float] | None, int | None]:
        ffmpeg = hosttools.require_ffmpeg(JOB_TYPE, FFMPEG_WHY)
        return (
            ffmpeg,
            *_require_renderable(self._config, self._backend, voice_id, params, resident),
        )

    def _guard(self, manifest: VoiceManifest, spec: Any) -> tuple[Any, Any]:
        plan = voice_load_plan(self._config, self._backend, manifest, spec)
        state = card_guard(
            self._config,
            model=manifest.id,
            need_bytes=plan.need_bytes,
            owned_pids=self._residency.owned_pids(),
            reclaimable_bytes=self._residency.reclaimable_bytes(excluding=manifest.id),
        )
        return plan, state

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        model = require_model(model, self.name, "a voice")
        checked = parse_params(TtsParams, params, self.name)
        self._residency.refuse_if_claimed("a tts render")
        resident = self._residency.is_resident(KIND_TTS, model)
        _, manifest, spec, _, _, _ = self.requirements(model, checked, resident)
        if not resident:
            self._guard(manifest, spec)

    def run(self, job: Job, ctx: JobContext) -> None:
        params = TtsParams.model_validate(job.params)
        voice_id = run_model(job.model, self.name, "a voice")
        ffmpeg, manifest, spec, (python, installed), band, width = as_job_error(
            self.requirements,
            voice_id,
            params,
            self._residency.is_resident(KIND_TTS, voice_id),
        )

        with self._residency.claimed(f"tts job {job.id}", may_mutate=True):
            engine = self._make_resident(ctx, manifest, spec, installed.path, python)
            resident = self._residency.resident_voice
            if resident is None:
                raise JobError(
                    "voice_not_resident",
                    f"{voice_id!r} was loaded but nothing is resident; "
                    + describe_resident(self._residency, KIND_TTS, "no voice is"),
                )
            try:
                self._render(
                    ctx, params, engine, resident.sample_rate, ffmpeg,
                    manifest, spec, band, width,
                )
            except EngineWouldNotStop as exc:
                ctx.note(str(exc))
                self._residency.unload(voice_id)
                raise JobCancelled(
                    f"job {job.id} was cancelled and {voice_id!r} had to be "
                    "taken off the card to make it stop"
                ) from None

    def _make_resident(
        self,
        ctx: JobContext,
        manifest: VoiceManifest,
        spec: Any,
        weights_dir: Path,
        python: Path,
    ) -> NarratorEngine:
        if self._residency.is_resident(KIND_TTS, manifest.id):
            engine = self._residency.voice_engine
            if engine is None:
                raise JobError(
                    "voice_not_resident",
                    f"{manifest.id!r} is recorded as resident but there is no "
                    "narrator process serving it",
                )
            ctx.progress(0.0, f"{manifest.id} is already resident")
            return engine

        ctx.warming(f"checking the accelerator for {manifest.id}")
        plan, state = as_job_error(self._guard, manifest, spec)
        ctx.warming(state.detail)

        try:
            occupy_voice(
                self._residency,
                manifest,
                spec,
                weights_dir,
                python,
                on_progress=ctx.warming,
                serving_width=plan.width,
            )
        except EngineError as exc:
            raise JobError("engine_failed", str(exc)) from None
        engine = self._residency.voice_engine
        if engine is None:
            raise JobError(
                "engine_failed", f"{manifest.id} loaded but published no engine"
            )
        ctx.progress(0.0, f"{manifest.id} is resident")
        return engine

    def _render(
        self,
        ctx: JobContext,
        params: TtsParams,
        engine: NarratorEngine,
        sample_rate: int,
        ffmpeg: str,
        manifest: VoiceManifest,
        spec: Any,
        band: dict[str, float] | None,
        width: int | None,
    ) -> None:
        sampling = take_sampling(manifest, params.take)
        _require_item_take(engine, params.take, sampling)
        by_index = {chunk.index: chunk for chunk in params.chunks}
        tally = _BatchTally(expected=set(by_index), total=len(params.chunks))
        request = _batch_request(params, sampling, band, width)

        ctx.expect_chunks(tally.total)
        ctx.progress(
            0.0,
            f"rendering {tally.total} chunk(s) at take {params.take}",
            **tally.counts(),
        )
        for message in engine.converse(
            request,
            terminal=frozenset({"batch_done"}),
            silence_timeout=RENDER_SILENCE_TIMEOUT_SECONDS,
            cancelled=lambda: ctx.cancelled,
        ):
            index = tally.claim(message)
            if index is None:
                continue
            failed = self._one_row(
                ctx, by_index[index], message, sample_rate, ffmpeg, params.take
            )
            tally.record(index, failed)
            ctx.progress(
                len(tally.answered) / tally.total,
                tally.row_line(index, failed),
                **tally.counts(),
            )

        tally.require_complete()
        ctx.progress(1.0, tally.final_line(), **tally.counts())
        ctx.done_extra(
            rendered=tally.rendered,
            failed=tally.failures,
            take=params.take,
            sample_rate=sample_rate,
            sampling=manifest.applied_sampling(spec.backend, params.take),
            voice={
                "id": manifest.id,
                "identity": spec.weights_identity,
                "identity_basis": spec.identity_basis,
            },
            width=width,
        )

    def _one_row(
        self,
        ctx: JobContext,
        chunk: TtsChunk,
        row: dict[str, Any],
        sample_rate: int,
        ffmpeg: str,
        take: int,
    ) -> str | None:
        if "message" in row:
            return str(row["message"])

        try:
            pcm, seconds = _pcm_of(row, chunk.index, sample_rate)
            tokens = _optional_int(row, "tokens")
            capped = _optional_bool(row, "capped")
            guard = _guard_of(row)
            pause_cuts = _pause_cuts_of(row)
        except _RowFailure as exc:
            return str(exc)

        destination = ctx.scratch / f"{chunk.index}.flac"
        encode_flac(ffmpeg, pcm, sample_rate, destination)
        ctx.artifact(destination.name, destination, index=chunk.index)

        chars = len(chunk.text)
        ctx.chunk(
            index=chunk.index,
            seconds=seconds,
            chars=chars,
            chars_per_sec=chars / seconds,
            tokens=tokens,
            capped=capped,
            take=take,
            guard=guard,
            pause_cuts=pause_cuts,
        )
        return None
