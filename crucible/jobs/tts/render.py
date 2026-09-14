"""The render door — job type `tts`. PHASE3-TTS.md section 6.

Text in, one FLAC per chunk out, a measurement per chunk, and the verdict the
ENGINE reached about it. A normal job on the exclusive lane: it owns the card for
its whole duration, it is the thing the queue was built to serialise, and it is
an operator's explicit order rather than an unattended request.

What this job does that no other job does
-----------------------------------------
**It may load its own model.** The one asymmetry with `llm`, and section 6 says
why: a chat request is fine-grained and unattended, so two clients alternating
would thrash the card and the proxy therefore never loads. A render job is not
like that. So if the wrong voice — or none — is resident when this reaches the
front of the lane, it loads the right one and emits `warming` exactly as
`load-voice` does. (The streaming door, being a connection rather than a job,
goes back to behaving like chat.)

**The model judges, this server forwards, the client orders.** `chunk {index,
seconds, chars, chars_per_sec, tokens, capped, take, guard}`. The first seven are
Crucible's own measurements of the bytes that arrived — it still measures, and it
still never retakes, never re-splits and never substitutes. `guard` is the
verdict narrator's own retake ladder reached about that chunk, forwarded
**verbatim** and `null` when narrator did not send one. Crucible does not read
inside it, does not validate its contents beyond "it is an object", and never
acts on it: that is `model_provenance`'s discipline, and it is what keeps
`api_version` at 1 while the ladder's vocabulary is free to grow.

*What this replaced, and why.* Until Owen's ruling of 2026-09-13 this paragraph
said "it measures and reports and decides nothing", called `chunk {index,
seconds, chars, chars_per_sec, tokens, capped, take}` "the whole guard
interface", and said BookForge's PaceTracker would read those numbers and decide
what to do about them. That was never true of this door. narrator had **two
rendering worlds** and only the audiobook one was guarded: the serve world —
which is the door this file drives — reached a Higgs engine through one bare
`render_audio()` per sentence, with no PaceTracker, no re-roll and no split
ladder, while `convert_many` ran the same model through all three. And
BookForge's PaceTracker was not in this path either: the bridge scraped guard
decisions off narrator's stderr, and a render driven through Crucible has no
stderr for it to scrape. So the numbers that arm of the design forwarded
described an unguarded single take, and nobody judged them.

The ruling fixes that by construction — the guard belongs to the model and its
inference, so it runs wherever the model runs, and the conclusion travels with
the audio it is about. narrator grew `render_many` (the guarded driver with the
file-writing sink removed) and puts its verdict on `batch_item`; this file's job
is to carry it across unopened. `crucible/docs/PHASE6-REMOTE-RENDER.md` sections
0, 2, 3 and 4; it amends PHASE3-TTS.md sections 1, 3 and 6.

**One artifact per requested index, always** (PHASE6 section 5). The ladder may
decide a chunk is unsalvageable whole and render it as two halves, but
`truncation.join_parts` joins them before the chunk retires, so a split arrives
here as ONE `batch_item` for the requested index and shows up only as `parts: 2`
inside the verdict. A client that asked for chunk 12 gets `12.flac`. This file
therefore needs no re-indexing and has none.

**A failed chunk is reported and the run continues.** The same rule `asr`'s
sibling types have, and the opposite of `asr`'s own: a transcript with a hole in
the middle is invisible in the output, while a missing `<index>.flac` is a file
that is not there and BookForge's resume already knows how to ask for it again.
So one bad sentence never sinks the other 1,399, the failures are named as they
happen and again in `done`, and no artifact is invented for a row that did not
render.

Why ffmpeg encodes the FLAC
---------------------------
narrator hands back base64 PCM16 and something has to turn it into a FLAC.
That something is **not** `soundfile`: `libsndfile` is a compiled dependency on
every platform Crucible installs on, and growing the server's own interpreter a
compiled audio library so it can re-encode audio it did not decode is the wrong
shape — the server process deliberately imports no torch, no vLLM and no
narrator, and this would be the first crack in that. ffmpeg is already a hard
requirement of this server (`asr` refuses `ffmpeg_missing` before it queues a
job), so `tts` refuses the same way, by the same name, through the same probe.

    ffmpeg -f s16le -ar <rate> -ac 1 -i pipe:0 -c:a flac <index>.flac

**The sample rate in that command is not a constant.** It is the rate narrator
reported on its `loaded` line, which `Residency._load_the_voice` has already
compared against the voice manifest and refused on a disagreement — it is
therefore both the engine's truth and the manifest's, which is the only state in
which writing a header is honest. Every `batch_item` also carries a
`sampleRate`, and a row whose rate is not the loaded one is a **failed row**, not
a resample: the bytes are at one rate and the file would claim another.

What narrator does not report, and what `null` means
----------------------------------------------------
`capped` and `tokens` are on section 6's `chunk` event and are **not on
narrator's wire** at the pinned sha. `serve/worker.py` sends `{i, format, data,
duration, sampleRate}` for a retiring row, plus `guard` when the engine guarded
its own batch, and nothing else; the frame cap it computed
(`HiggsBudget.cap_frames`, clamped by `sgl_served.frame_cap`) never leaves the
engine. Crucible cannot derive either — the cap is narrator's own arithmetic over
the text, and the server does not see a frame count at all. (The cap-hit is now
one input to a verdict the engine has already reached rather than a number a
client has to reason from, which is why this gap stopped being urgent without
being closed.)

So both are `None` when narrator does not say, and `None` is published as JSON
`null` and **means "narrator did not say"**. It is never to be read as `false`:
a runaway reported as "not capped" is exactly the failure the field exists to
prevent. PHASE3-TTS.md section 6 records the owed change on narrator's side.

`guard` obeys the same rule one level up. A `null` guard means narrator sent no
verdict — an engine with no `render_many` to offer (Orpheus, whose older guard is
a different mechanism), or a row that failed before the ladder reached a decision
— and it is never to be read as "the take was clean". `clean` is a key INSIDE a
verdict that exists; the absence of a verdict says nothing about the take.

**The pace state does NOT round-trip yet, and this door does not pretend it
does.** PHASE6 section 4 has the client carry the tracker's state between
chapters, because the guard re-centres on the running median of the book's own
shipped takes and a book rendered as 40 cold-started chapters would guard
measurably worse than the same book rendered as one run. narrator has no wire for
it at the pinned sha: `generate_batch` accepts no `pace`, `batch_done` carries
only `count`, and `truncation.PaceTracker` has no state to export or import. So
`tts` params gain no `pace` and `done_extra` carries none. A half-built
round-trip would silently drop the state and degrade a long book with nothing
failing, which is the opposite of what this server does with a fact it cannot
get: the field arrives when narrator can answer it.

`chars` is deliberately **the server's own count of the text it sent**, not a
number read off the reply, for the reason `crucible/workers.py` gives about
positional results: a number a subprocess echoes back is a number a subprocess
can get wrong, and this one is already known exactly.
"""

from __future__ import annotations

import base64
import binascii
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ... import accelerator, hosttools
from ...config import Config
from ...engines import EngineError, NarratorEngine
from ...errors import ApiError, JobError
from ...residency import KIND_TTS, Residency, describe_resident
from ...voices import VoiceError, VoiceManifest
from .. import asr
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor
from .common import (
    describe_voices,
    known_voice,
    require_loadable,
    validated_params,
    voice_provenance,
    voice_rows,
)

__all__ = ["TtsChunk", "TtsJobType", "TtsParams"]

JOB_TYPE = "tts"

#: How long narrator may go without saying anything at all during a render. A
#: SILENCE timeout, reset by every line, not a deadline on the batch: a
#: 1,400-chunk book is hours of legitimate work and any clock over the whole of
#: it would kill a real render. Ten minutes is Owen's standing number for a
#: single Higgs chunk (`higgs-first-inapp-render-findings`), which is the longest
#: narrator is ever legitimately quiet for between two rows.
RENDER_SILENCE_TIMEOUT_SECONDS = 600.0

#: How long one ffmpeg encode may take. It is a local CPU transcode of at most a
#: minute of PCM, so this is a wedge detector rather than a budget.
ENCODE_TIMEOUT_SECONDS = 120.0

#: How far narrator's reported `duration` may sit from the duration of the PCM it
#: actually sent before the row is refused. One Higgs frame is 40 ms at 25 fps,
#: so a disagreement of more than this is not rounding — it is a reply describing
#: audio other than the audio attached to it.
DURATION_TOLERANCE_SECONDS = 0.05


class TtsChunk(BaseModel):
    """One unit of work: the client's own index, and the text to speak."""

    model_config = ConfigDict(extra="forbid")

    #: The client's number for this chunk, and the artifact's name. Crucible
    #: neither assigns it nor renumbers: `<index>.flac` is where BookForge's
    #: assembly and resume already look, so the index is the client's and travels
    #: unchanged through narrator's batch `i` and back out again.
    index: int = Field(ge=0)
    text: str

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if value.strip() == "":
            # narrator answers `{"type": "error", "message": "No text provided"}`
            # to an empty generate, which is a WHOLE-REQUEST refusal and would
            # take the other rows with it. Refused here, by name, before the job
            # exists.
            raise ValueError(
                "a chunk's text must not be blank; narrator refuses an empty "
                "generate with a whole-request error, which would end the batch"
            )
        return value


class TtsParams(BaseModel):
    """`params` for a tts job. Unknown keys are refused, not ignored.

    Nothing has a default. `language` and `take` are both decisions — a book
    rendered in the wrong language, or at a take the client did not choose, is a
    silent substitution — and a default here would let a client make neither and
    still get audio.
    """

    model_config = ConfigDict(extra="forbid")

    language: str
    take: int = Field(ge=0)
    chunks: list[TtsChunk] = Field(min_length=1)

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


# ------------------------------------------------------------------ refusals


def _require_ffmpeg() -> str:
    """ffmpeg's path, or `ffmpeg_missing` by name before the job is queued.

    The same probe `asr` uses, deliberately: one question about this host asked
    in one place, so a test that replaces it covers both types and a host without
    ffmpeg refuses both by the same name. The reason differs — `asr` decodes
    through it, `tts` encodes through it — and so the message does.
    """
    found = asr.ffmpeg_path()
    if found is None:
        raise ApiError(
            409,
            "ffmpeg_missing",
            "there is no ffmpeg on this server's PATH, and tts encodes every "
            "chunk through it: narrator returns base64 PCM16 and the artifact "
            "BookForge's assembly expects is a mono FLAC. The alternative is "
            "linking libsndfile into the server's own interpreter, which is a "
            "compiled audio dependency in a process that deliberately imports "
            "no engine at all. " + hosttools.searched_note(),
            {"path": hosttools.search_path()},
        )
    return found


def _require_renderable(
    config: Config, backend: Any, voice_id: str, params: TtsParams
) -> tuple[VoiceManifest, Any, Any]:
    """Everything a render needs, or the first refusal, by name.

    The order is `require_loadable`'s and then this door's own three: what the
    voice IS, what the ladder can honour, and whether the text fits the cap
    certificate. Each of them is cheap and none of them needs the card.
    """
    manifest, spec, interpreter = require_loadable(config, backend, voice_id)

    if manifest.kind == "zeroshot":
        # narrator's `load` message carries `voice`, `modelDir`, `adapterDir`,
        # `baseDir`, `caps` and `warm` — and no reference clips. There is
        # therefore no channel on this wire for the thing a zero-shot voice IS,
        # and rendering one would mean conditioning on nothing: a whole book in
        # the base model's voice, reported as success. PHASE3-TTS.md section 6
        # records what narrator owes before this can be lifted.
        raise ApiError(
            400,
            "voice_kind_unsupported",
            f"voice {voice_id!r} is a zeroshot voice, and narrator's serve wire "
            "carries no reference clips on its load message. Crucible will not "
            "render one rather than render it in the base model's voice and call "
            "that success",
            {"voice": voice_id, "kind": manifest.kind},
        )

    # TAKE 0's SAMPLING IS THE MANIFEST'S, AND IT REACHES NARRATOR. Not through
    # `caps` on the load message — that channel is `register_voice_caps`, whose
    # vocabulary is Orpheus's — but through the NARRATOR_HIGGS_VOICES document
    # `crucible/narratorvoices.py` writes at every load, whose `sampling` key
    # narrator's `load_voices` reads onto the voice and both arms apply as the
    # engine's override. So a voice that deviates from the boson default (with
    # the written reason `crucible/voices.py` requires) renders at what it
    # declares, and there is nothing here to refuse. (Until 2026-09-14 this was
    # a `sampling_not_wired` refusal; it was unreachable with the shipped
    # manifests and, once the document existed, false.)

    try:
        manifest.take(params.take)
    except VoiceError as exc:
        # Never clamped to the last rung: a silent clamp is a retake ladder that
        # stops climbing without telling anyone, and the client would keep asking
        # for take 4 and keep getting take 2's draw.
        raise ApiError(
            400,
            "unknown_take",
            str(exc),
            {"voice": voice_id, "take": params.take, "takes": len(manifest.takes)},
        ) from None
    if params.take != 0:
        # Reachable only for a voice that declares a ladder. No manifest in this
        # build does, so this is not dead code so much as a refusal waiting for
        # its manifest: a rung is PER RENDER, the voices document is written
        # PER LOAD, and narrator has no per-request sampling channel on
        # `generate_batch` — so a take above 0 has no way to reach the engine
        # without a reload, and a reload per take is not a ladder.
        raise ApiError(
            409,
            "sampling_not_wired",
            f"take {params.take} of {voice_id!r} deviates from take 0's sampling "
            "and Crucible has no per-request channel to narrator for it: the "
            "voices document carries one sampling per load, and narrator's "
            "generate_batch takes none (PHASE3-TTS.md section 4)",
            {"voice": voice_id, "take": params.take},
        )

    over = [
        {"index": chunk.index, "chars": len(chunk.text)}
        for chunk in params.chunks
        if len(chunk.text) > spec.max_chars
    ]
    if over:
        # The cap certificate doing the one job it exists for. It is per (voice,
        # backend) and refused rather than re-split: chunking is BookForge's
        # (section 1), and a server that quietly cut a chunk in half would be
        # returning two files where a client asked for one.
        raise ApiError(
            400,
            "chunk_too_long",
            f"{len(over)} chunk(s) are longer than the {spec.max_chars}-character "
            f"cap for {voice_id!r} on {spec.backend}: "
            + ", ".join(f"index {row['index']} is {row['chars']}" for row in over[:8])
            + ("" if len(over) <= 8 else f", and {len(over) - 8} more")
            + ". Chunking is the client's, so this is a refusal and not a re-split",
            {"voice": voice_id, "max_chars": spec.max_chars, "chunks": over},
        )
    return manifest, spec, interpreter


# ------------------------------------------------------------------ encoding


def encode_flac(ffmpeg: str, pcm: bytes, sample_rate: int, destination: Path) -> None:
    """Raw mono PCM16 into a FLAC at `destination`. Raises JobError by name.

    `-y` overwrites, because the destination is inside the job's own scratch
    directory and a stale file there could only come from this job.
    """
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


# --------------------------------------------------------------- the replies


class _RowFailure(Exception):
    """This row did not render. Its neighbours still can."""


def _row_index(row: dict[str, Any], expected: set[int]) -> int:
    """The `i` narrator echoed, checked against what was asked for.

    narrator keys a `batch_item` by the caller's own `i`, so this is the one
    identifier that crosses the wire in both directions — and therefore the one
    that has to be checked. A row for an index nobody asked for is a protocol
    error rather than an extra file.
    """
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
    """The row's audio and the duration Crucible measured in it.

    The duration is computed from the bytes that arrived and then compared with
    the one narrator reported. The server measures — that is the rule the whole
    job type is built on, and applying it to itself is what makes a `chunk`
    event's `seconds` a fact about a file rather than a claim about a request.
    """
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


def _guard_of(row: dict[str, Any]) -> dict[str, Any] | None:
    """narrator's verdict for this row, UNOPENED, or None when it sent none.

    The whole of Crucible's interest in a guard. It is not parsed, not
    summarised, not re-keyed and not checked against any vocabulary — the one
    thing asked of it is that it be a JSON object, because that is what the
    `chunk` event's field is declared to be and an event has to serialise.

    **Reading inside it would be the bug.** The verdict's own words are the
    ladder's (`clean`, `short`, `long`, `hole`, `rerolled`, `resplit`,
    `accepted-off-length` today), the take records are whatever
    `truncation._LadderTask` put in them, and both are free to grow the day
    somebody adds a rung. A server that validated any of that would refuse a
    chunk it rendered perfectly well, at the first guard fire on a real book,
    for saying a true thing this file had not heard of. PHASE6-REMOTE-RENDER.md
    section 3, and the same discipline `model_provenance` already has.

    A `guard` that is not an object fails ITS ROW and not the batch, like every
    other malformed field here: the audio may be fine, but a reply whose verdict
    is a string is a reply this door cannot describe, and shipping the FLAC with
    the verdict thrown away would publish a measurement nobody can trace.
    """
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


# ------------------------------------------------------------------ job type


class TtsJobType:
    """`POST /v1/jobs {"type": "tts", "model": "<voice id>", "params": {...}}`."""

    name = JOB_TYPE

    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency

    @property
    def residency(self) -> Residency:
        return self._residency

    # ----------------------------------------------------------- describing

    def describe_models(self) -> list[ModelDescriptor]:
        """The voices. `model` on the wire is the voice id (section 6)."""
        return describe_voices(self._config.backend_kind, self._residency)

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
        if asr.ffmpeg_path() is None:
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

    # ------------------------------------------------------------ preflight

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        if model is None:  # unreachable: resolve_model requires one
            raise ApiError(400, "model_required", f"{self.name} needs a voice")
        checked = validated_params(TtsParams, params, self.name)
        # Before anything else about this host: a streaming session holds
        # narrator's one wire, and a render that queued behind it would fail in
        # the lane instead of being refused here (PHASE3-TTS.md section 7).
        self._residency.refuse_if_claimed("a tts render")
        _require_ffmpeg()
        _, spec, _ = _require_renderable(self._config, self._backend, model, checked)
        accelerator.guard(
            self._config.backend_kind,
            model_id=model,
            need_bytes=spec.memory_bytes_estimate,
            owned_pids=self._residency.owned_pids(),
            desktop_allowance_bytes=self._config.desktop_allowance_bytes,
            # A render job MAY unload the current resident to load its own voice
            # — section 6's one asymmetry with `llm` — so what is on the card now
            # is memory this job can have. `asr` passes none for the opposite
            # reason: it never unloads somebody's model to run a transcript.
            reclaimable_bytes=self._residency.reclaimable_bytes(excluding=model),
        )

    # ------------------------------------------------------------------ run

    def run(self, job: Job, ctx: JobContext) -> None:
        params = TtsParams.model_validate(job.params)
        voice_id = job.model
        if voice_id is None:  # unreachable: resolve_model requires one
            raise JobError("model_required", f"{self.name} needs a voice")

        try:
            ffmpeg = _require_ffmpeg()
            manifest, spec, (python, installed) = _require_renderable(
                self._config, self._backend, voice_id, params
            )
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None

        # The card and narrator's wire, held for the whole job. `preflight`
        # already refused a session that was open when this was submitted; this
        # is the half that matters when one opened while the job sat in the
        # queue. Two conversations on one stdin do not collide loudly — they read
        # each other's `batch_item` lines — so this is a claim rather than a
        # check (PHASE3-TTS.md section 7, `crucible/residency.py`).
        with self._residency.claimed(f"tts job {job.id}", may_mutate=True):
            engine = self._make_resident(ctx, manifest, spec, installed.path, python)
            resident = self._residency.resident_voice
            if resident is None:  # unreachable: the load above worked or raised
                raise JobError(
                    "voice_not_resident",
                    f"{voice_id!r} was loaded but nothing is resident; "
                    + describe_resident(self._residency, KIND_TTS, "no voice is"),
                )
            self._render(ctx, params, engine, resident.sample_rate, ffmpeg)

    def _make_resident(
        self,
        ctx: JobContext,
        manifest: VoiceManifest,
        spec: Any,
        weights_dir: Path,
        python: Path,
    ) -> NarratorEngine:
        """The engine serving this job's voice, loading it if it is not on the card."""
        if self._residency.is_resident(KIND_TTS, manifest.id):
            engine = self._residency.voice_engine
            if engine is None:  # unreachable
                raise JobError(
                    "voice_not_resident",
                    f"{manifest.id!r} is recorded as resident but there is no "
                    "narrator process serving it",
                )
            ctx.progress(0.0, f"{manifest.id} is already resident")
            return engine

        ctx.warming(f"checking the accelerator for {manifest.id}")
        try:
            # The card can change between the queue and the lane, so the guard
            # runs again here against the same rules.
            state = accelerator.guard(
                self._config.backend_kind,
                model_id=manifest.id,
                need_bytes=spec.memory_bytes_estimate,
                owned_pids=self._residency.owned_pids(),
                desktop_allowance_bytes=self._config.desktop_allowance_bytes,
                reclaimable_bytes=self._residency.reclaimable_bytes(
                    excluding=manifest.id
                ),
            )
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None
        ctx.warming(state.detail)

        try:
            self._residency.load_voice(
                manifest, spec, weights_dir, python, on_progress=ctx.warming
            )
        except EngineError as exc:
            raise JobError("engine_failed", str(exc)) from None
        engine = self._residency.voice_engine
        if engine is None:  # unreachable: load_voice publishes or raises
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
    ) -> None:
        """One `generate_batch`, and one FLAC per row as the row retires.

        **The whole job is one batch.** How many rows the engine runs at once is
        engine tuning and belongs to the server (section 7 says so about the
        streaming door, and it is the same rule here) — but "the server" in that
        sentence is narrator, not this file: `generate_batch` does its own
        scheduling, grouping consecutive rows on MLX and dispatching singly on
        vLLM, and cutting the list up here would be Crucible second-guessing a
        read-ahead window it cannot see. Rows come back **out of order**, which
        is why nothing below indexes by position.
        """
        by_index = {chunk.index: chunk for chunk in params.chunks}
        expected = set(by_index)
        answered: set[int] = set()
        failures: list[dict[str, Any]] = []
        rendered = 0
        total = len(params.chunks)

        request = {
            "action": "generate_batch",
            "language": params.language,
            # No `stream` flag anywhere in the list. narrator's own docstring:
            # "A generate_batch with no `stream` flag anywhere takes the
            # pre-existing code path, byte for byte." The render door wants whole
            # rows; sub-sentence chunks are the streaming door's, and asking for
            # them here would mean reassembling audio the engine already had in
            # one piece.
            "items": [
                {"i": chunk.index, "text": chunk.text} for chunk in params.chunks
            ],
        }

        ctx.progress(
            0.0,
            f"rendering {total} chunk(s) at take {params.take}",
            rendered=0,
            failed=0,
            total=total,
        )

        for message in engine.converse(
            request,
            terminal=frozenset({"batch_done"}),
            silence_timeout=RENDER_SILENCE_TIMEOUT_SECONDS,
            cancelled=lambda: ctx.cancelled,
        ):
            kind = message["type"]
            if kind == "batch_done":
                # `continue`, not `break`. The iterator ends itself on the
                # terminal message — and if the job was cancelled it raises
                # `JobCancelled` at that point instead, which is how a cancelled
                # render is reported as cancelled rather than as a short success.
                # Breaking out here would abandon the generator before it could.
                continue
            if kind != "batch_item":
                # `batch_chunk` is the streaming door's and cannot arrive here —
                # nothing asked for a streamed row. Anything else is narrator
                # saying something this door has no meaning for, and skipping it
                # is how a protocol change becomes a silent behaviour change.
                raise JobError(
                    "narrator_protocol",
                    f"narrator sent a {kind!r} message during a non-streamed "
                    "generate_batch; this door asked for whole rows and knows "
                    "only batch_item and batch_done",
                )

            try:
                index = _row_index(message, expected)
            except EngineError as exc:
                raise JobError("narrator_protocol", str(exc)) from None
            if index in answered:
                raise JobError(
                    "narrator_protocol",
                    f"narrator answered row {index} twice. One answer per row is "
                    "narrator's own guarantee, and two would mean one FLAC "
                    "overwriting another",
                )
            answered.add(index)

            failed = self._one_row(
                ctx, by_index[index], message, sample_rate, ffmpeg, params.take
            )
            if failed is None:
                rendered += 1
            else:
                failures.append({"index": index, "message": failed})
            ctx.progress(
                len(answered) / total,
                (
                    f"chunk {index} failed: {failed}"
                    if failed is not None
                    else f"{rendered} of {total} chunk(s) rendered"
                )
                + (f"; {len(failures)} failed so far" if failures else ""),
                rendered=rendered,
                failed=len(failures),
                total=total,
            )

        missing = sorted(expected - answered)
        if missing:
            # narrator guarantees one answer per item — "a row with no message
            # hangs its sentence until the 180s timeout taints the worker" is its
            # own comment on why. A batch that ended short is a protocol failure,
            # not a partial answer.
            raise JobError(
                "narrator_protocol",
                f"narrator said batch_done with {len(missing)} row(s) unanswered: "
                f"{missing[:20]}. One answer per row is its own guarantee, so a "
                "short batch is not a short answer",
            )

        ctx.progress(
            1.0,
            f"{rendered} of {total} chunk(s) rendered"
            + (f"; {len(failures)} failed" if failures else ""),
            rendered=rendered,
            failed=len(failures),
            total=total,
        )
        # The authoritative list, in `done`, as well as the `progress` line each
        # failure produced when it happened: a client reading only the terminal
        # event still learns exactly which indices it has to ask for again.
        ctx.done_extra(
            rendered=rendered,
            failed=failures,
            take=params.take,
            sample_rate=sample_rate,
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
        """One retiring row. Returns None when it rendered, else why it did not.

        A failure is a **return value and not an exception** on purpose: it is
        the ordinary outcome for one chunk of a book, the caller counts it and
        names it in the same progress line it would have written anyway, and the
        other 1,399 rows carry on.
        """
        if "message" in row:
            # narrator's per-row failure shape (`serve/worker.py`: `{'i': ...,
            # 'message': ...}`) — 'No audio generated', 'cancelled', or the
            # exception text from one row's generate.
            return str(row["message"])

        try:
            pcm, seconds = _pcm_of(row, chunk.index, sample_rate)
            tokens = _optional_int(row, "tokens")
            capped = _optional_bool(row, "capped")
            guard = _guard_of(row)
        except _RowFailure as exc:
            return str(exc)

        destination = ctx.scratch / f"{chunk.index}.flac"
        encode_flac(ffmpeg, pcm, sample_rate, destination)
        ctx.artifact(destination.name, destination)

        chars = len(chunk.text)
        ctx.chunk(
            index=chunk.index,
            seconds=seconds,
            chars=chars,
            # seconds cannot be zero here: `_pcm_of` refuses an empty payload.
            chars_per_sec=chars / seconds,
            tokens=tokens,
            capped=capped,
            take=take,
            # The object narrator sent, handed straight through. Not copied, not
            # normalised, not inspected — see `_guard_of`.
            guard=guard,
        )
        return None
