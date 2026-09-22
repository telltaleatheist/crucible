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

**And the accelerator guard belongs to that load and to nothing else.** A
render of the voice that is already on the card loads nothing, so neither
`preflight` nor `_make_resident` asks the card whether there is room for it;
asking would refuse the resident voice on the strength of its own VRAM.

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

**And WHICH arm judges is the request's to say** (PHASE18-UNCERTIFIED.md
sections 4 and 6, Owen's ruling of 2026-09-19). `retake: true` runs the
paragraph above — narrator's guarded driver, measured against the `band` the
same request states — and `retake: false`, which is what absence means, runs
the bare arm: every chunk rendered once as sent, at the requested take's
sampling, nothing judged and nothing retaken. The measurements come back on
both arms, because they are this server's own count of bytes that arrived;
`guard` on a bare row is `null`, and here that `null` has its most exact
meaning yet — nobody judged it, because nobody was asked to.

The bare arm is the default because it is the primitive, and because the first
client that needs it is measuring the very failures a guard hides: the
fine-tuning ladder renders 512 matched cells per checkpoint and scores the
failure count, and a re-roll that succeeds is indistinguishable from a good
first draw. A guarded screen under-counts, invisibly, in the direction that
makes a good checkpoint look bad — narrator falls back to its ENGINE's default
band for a voice that states none, centred at 15.0 against a real book pace
nearer 17.2, and re-rolls healthy chunks to MAX_DEPTH. Hence the band being
the caller's to state and `retake_without_band` rather than a lookup.

**This door does not refuse a chunk by length.** `chunk_too_long` was retired
on 2026-09-19 — see `_require_renderable`.

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

**`capped` is being closed from narrator's end, and this file needs no change
for it.** The 2026-09-19 narrator work that reads `retake` and `band` also puts
`"capped": bool` on every retiring row, so against a narrator that carries it
the `chunk` event's `capped` is a real boolean. `_optional_bool` already reads
it exactly that way and has since this door was built — present and a bool, it
travels; absent, it is `None`. There is no branch here to add and there must
not be one: a reader that turned an absent key into `false` the day the wire
was supposed to carry it would report every runaway as a finished sentence.
Against the currently PINNED narrator (`crucible/envs/tts/*.txt`, not yet
moved) it is still `null` on every row.

`guard` obeys the same rule one level up. A `null` guard means narrator sent no
verdict — a bare render (`retake: false`, which is what nobody asking for a
guard looks like), an engine with no `render_many` to offer, or a row that
failed before the ladder reached a decision
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

from ... import accelerator, hosttools, jobenv
from ...config import Config
from ...engines import EngineError, EngineWouldNotStop, NarratorEngine
from ...errors import ApiError, JobCancelled, JobError
from ...narratorvoices import take_sampling
from ...residency import KIND_TTS, Residency, describe_resident
from ...voices import VoiceManifest
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

    `language`, `take` and `chunks` have no default. All three are decisions —
    a book rendered in the wrong language, or at a take the client did not
    choose, is a silent substitution — and a default here would let a client
    make neither and still get audio.

    `retake` and `band` DO have an absent meaning, and that is the 2026-09-19
    ruling rather than a lapse in the same rule (PHASE18-UNCERTIFIED.md
    section 4). A default is the server inventing a value for a fact that has a
    right answer. An absent constraint is nobody having asked for one — and
    "nobody asked me to judge this render" is a real, sayable state, which is
    what a bare render IS.
    """

    model_config = ConfigDict(extra="forbid")

    language: str
    take: int = Field(ge=0)
    chunks: list[TtsChunk] = Field(min_length=1)
    #: WHICH ARM RENDERS THIS BATCH, and the whole of Crucible's say in it.
    #:
    #: `true` is narrator's guarded driver — `render_many`, the PaceTracker,
    #: the re-roll on truncation/runaway/loop and the split ladder — measured
    #: against the `band` this request states. `false` (and absent) is the bare
    #: arm: every chunk rendered ONCE as sent, at the requested take's
    #: sampling, nothing judged and nothing retaken.
    #:
    #: Bare is the default because it is the primitive. The guarded arm is a
    #: narration behaviour a client asks for; a server that guarded by default
    #: would be deciding, for the screening ladder that is measuring a failure
    #: curve, to hide the failures — a re-roll that succeeds looks exactly like
    #: a good first draw, and the numbers come out systematically clean in
    #: precisely the way the instrument exists to detect.
    retake: bool = False
    #: THE BAND THE GUARD MEASURES AGAINST, stated by the caller, in the
    #: manifest's own spelling: `{pace_chars_per_sec, max_chars_per_sec,
    #: min_chars_per_sec}`, all three positive with min < pace < max.
    #:
    #: **Not read off the voice.** BookForge echoes back the row it read from
    #: `/v1/voices`; the screening ladder sends nothing and cannot be guarded.
    #: The alternative — `retake: true` and the server finds a band — is the
    #: shape that produced the 2026-09-18 defect twice over: `higgs-default`
    #: satisfied a mandatory triple with narrator's own `CHARS_PER_SEC` 15.0,
    #: which is the frame cap's DIVISOR and not a narration rate, and
    #: deathstalker inherited 16.64 onto weights that measured 15.91. A band
    #: nobody stated is a band nobody can be held to.
    #:
    #: `dict` and not a sub-model on purpose: every way a band can be wrong is
    #: ONE refusal, `band_malformed`, raised by `_check_band` below. Pydantic
    #: would sort the same mistake into `invalid_params` by which key it
    #: landed on, which is two names for one thing.
    band: dict[str, Any] | None = None
    #: HOW MANY OF THIS JOB'S CHUNKS MAY BE IN FLIGHT AT ONCE, or absent for
    #: THE WIDTH THE ENGINE WAS STARTED WITH — which is narrator's to know and
    #: not this file's to restate (Owen's ruling of 2026-09-20).
    #:
    #: Absent means NO `width` ON THE BATCH, so the running engine keeps the
    #: width it was started at. On the served arm that is `HIGGS_MAX_NUM_SEQS`,
    #: which Crucible did state, out of `[voice.serving].max_num_seqs`; on
    #: `mlx-darwin` it is `NARRATOR_HIGGS3_MLX_BATCH`, which comes off
    #: `engines/narrator.py:MLX_TIERS` — a different number, out of a different
    #: table, measured on a different machine.
    #:
    #: THE SUBSTITUTION IS DELETED, AND IT COST A MEASURED 2.3x. Between
    #: 2026-09-19 and 2026-09-20 an absent width resolved HERE to
    #: `[voice.serving].max_num_seqs`, which is 16 in every packaged manifest
    #: and whose own note says it was measured on a 3090 Ti as vllm-omni's
    #: stage-0 admission width. narrator's MLX backend honours a batch `width`
    #: as an in-flight ceiling, so every Mac batch logged "MLX batch narrowed
    #: 64 rows -> 16" against an engine started at 64 out of its own tier row.
    #: Measured on mistborn/Shift Book 2, chunk lengths equal and zero retakes:
    #: 12.9x realtime and 189 sentences/min at 64, 5.5x and 78 at 16. One fact,
    #: two owners, nothing comparing them — `docs/ARCHITECTURE.md`'s shape.
    #:
    #: A width above the engine's is `width_over_serving`, never clamped: a job
    #: that believed it was running 16 wide while narrator ran 4 would report a
    #: throughput nobody can reproduce. WHO refuses it depends on the arm, and
    #: `_require_width` below says why.
    #:
    #: Why a screening job needs it (measured by the ladder's author,
    #: 2026-09-19): 0.60 mem fraction at 16 wide summed to 24.2 GB on a 24 GB
    #: card, and WDDM then pages to host RAM 4-10x slower with no error at all.
    #: The ladder's baseline is 4 wide, on voices whose manifests say 16.
    #: Narrowing needs no restart and gets none: SGLang's
    #: `--max-running-requests` and `cuda_graph_max_bs` stay whatever the voice
    #: was loaded with, and this only limits what narrator keeps in flight.
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


#: The three rates in narrator's spelling, in the order a message prints them.
#: One map, so the request's key names and the wire's key names cannot drift
#: apart — the shape `docs/ARCHITECTURE.md`'s audit found seven times.
_BAND_ON_THE_WIRE: dict[str, str] = {
    "pace_chars_per_sec": "paceCharsPerSec",
    "max_chars_per_sec": "maxCharsPerSec",
    "min_chars_per_sec": "minCharsPerSec",
}


def _require_band(params: TtsParams) -> dict[str, float] | None:
    """The stated band in narrator's spelling, or None, or a refusal by name.

    Two refusals and they are both about the REQUEST rather than about a
    manifest, which is why they are codes here and not `voice_invalid`
    messages (PHASE18-UNCERTIFIED.md section 9):

    `retake_without_band` — the guarded arm was asked for and no band was
    stated. Not filled in from the voice: see `TtsParams.band` for the two
    measured incidents that shape is responsible for. Not silently downgraded
    to the bare arm either — a client that asked to be guarded and was not
    would read every clean row as a verdict.

    `band_malformed` — a band was stated and is not one: a missing rate, a
    value that is not a number, a rate at or below zero, or an order other
    than min < pace < max. It refuses the WHOLE request, because a band is one
    statement and there is no row it belongs to.

    **A band with `retake` false or absent is accepted and does nothing.**
    Owen, 2026-09-19: *"it won't do anything with the number because it wasn't
    asked to."* It is still CHECKED — a malformed band is a client mistake
    whether or not this run would have used it, and reporting it only on the
    runs that happen to be guarded would hide it until the day it matters.
    """
    band = params.band
    if band is None:
        if params.retake:
            raise ApiError(
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
        return None

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
    rates: dict[str, float] = {}
    for key in _BAND_ON_THE_WIRE:
        value = band[key]
        # A bool is an int in Python, and `True` as a rate would read as 1.0.
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
        rates[key] = float(value)
    if not (
        rates["min_chars_per_sec"]
        < rates["pace_chars_per_sec"]
        < rates["max_chars_per_sec"]
    ):
        # narrator's own rule (`engine/higgs/config.py:_length_band`) and
        # `voices.py:_check_pace`'s, asked here so a request that could only be
        # refused at the engine, on somebody's book, is refused at the door.
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
    return {wire: rates[key] for key, wire in _BAND_ON_THE_WIRE.items()}


def _require_width(
    manifest: VoiceManifest, spec: Any, params: TtsParams
) -> int | None:
    """The width the CALLER stated, verbatim, or `width_over_serving` by name.

    **There is one owner of the width and it is not this file** (Owen's ruling
    of 2026-09-20). A width the client stated is forwarded exactly as stated; a
    width the client did not state is ABSENT from the batch, and the engine
    then keeps the width it was started with. Nothing here substitutes.

    *What this replaced, and what it cost.* From 2026-09-19 this function
    answered an absent width with `[voice.serving].max_num_seqs`, on the
    reasoning that it is "the width the engine was STARTED with" — and on the
    served arm it is: `crucible/residency.py` hands it to the engine and
    `engines/narrator.py:environment()` emits it as `HIGGS_MAX_NUM_SEQS`.
    On `mlx-darwin` it is not. narrator starts no server
    there, reads no `HIGGS_*` variable, and takes its width from
    `engines/narrator.py:MLX_TIERS` by the machine's own memory — 64 on the
    64 GB Mac Studio, emitted as `NARRATOR_HIGGS3_MLX_BATCH`. So every Mac
    batch since 1.0.7 carried 16, a vllm-omni stage-0 number measured on a
    3090 Ti, and narrator's MLX backend (which honours a batch width as an
    in-flight ceiling) logged "MLX batch narrowed 64 rows -> 16 (depth 2957
    positions, cap 16, budget 42 GB)" on every one. Measured on mistborn/Shift
    Book 2 with chunk lengths equal and zero retakes: **12.9x realtime and 189
    sentences/min before, 5.5x and 78 after.** The width was the whole
    difference.

    **A width above the engine's is still a refusal and never a clamp** — a
    clamp would be a job reporting a width it did not run at, and the
    throughput of a screening run is one of the things the run exists to
    measure. What changed is WHO refuses it. The ceiling is the width the
    ENGINE was started at, and Crucible knows that number on exactly one arm:
    the SERVED one, where the env recipe installs a serving stack and this
    server itself stated `max_num_seqs`. There this door refuses early, before
    the job exists and before the card is touched. On `mlx-darwin` Crucible
    knows no such number — `max_num_seqs` describes a stack that is not
    running — so the stated width is forwarded and narrator's own
    `width_over_serving` (added on the narrator side 2026-09-19) answers for
    the width it actually has.

    `jobenv.tts_env(...).serving_stack` is the question asked, rather than
    `spec.backend == "mlx-darwin"`, because that is the same authority
    `crucible/engines/narrator.py:environment()` uses to decide whether to emit
    `HIGGS_MAX_NUM_SEQS` at all. Two files reading one fact, not two files
    stating it.
    """
    if params.width is None:
        return None
    serving = manifest.serving
    if serving is None:
        # A voice whose engine reads no `HIGGS_*` variable — the shape the next
        # narrator engine will have and no manifest has today. No configured
        # width to compare against, so the stated one travels and the engine
        # answers for it.
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
    """Everything a render needs, or the first refusal, by name.

    The order is `require_loadable`'s and then this door's own two: what the
    voice IS, and what this request asks the guard to do. Both are cheap and
    neither needs the card. Returns the band in narrator's spelling beside the
    three things it always returned, so the rule that validates it has one
    reader and `_render` does not re-derive it.

    **This door no longer asks whether the text fits.** `chunk_too_long` was
    the third check here until 2026-09-19 and it is retired, not relaxed.
    Owen: *"I don't think it's crucible's place to refuse chunks outside the
    band… especially if we add a different tts engine."* Chunking and packing
    are the client's (PHASE3-TTS.md section 1) — that has never changed — and
    the cap a voice carries is advertised on `/v1/voices` so the client can
    pack to it. What changed is who acts on it: an oversize chunk now surfaces
    as whatever the engine does with it, reported honestly on the `chunk` row,
    rather than as a server refusing a render it was never asked to judge. The
    two facts that made the refusal untenable arrived together: a screening
    checkpoint has no measured cap to be refused against, and a second TTS
    engine would have its own frame arithmetic that this number describes
    nothing about.

    `resident` is whether this voice is the one already on the card. It bears
    on exactly one refusal — see below.
    """
    manifest, spec, interpreter = require_loadable(config, backend, voice_id)

    if manifest.kind == "zeroshot" and not resident:
        # A RENDER JOB LOADS ITS OWN VOICE (section 6's one asymmetry with
        # `llm`), and since 2026-09-14 a zero-shot load needs the reference
        # clip that `load-voice` carries in `params.reference`. This job has
        # no such field — `language`, `take` and `chunks` are the whole of its
        # params — and inventing a second channel for clips here would be two
        # doors owning one fact. So this refusal NARROWED rather than being
        # deleted: a zero-shot voice that is already resident was loaded with
        # its clip and renders like any other, and one that is not is refused
        # because this door cannot load it. (It used to refuse the KIND
        # outright, on the true-at-the-time grounds that narrator's load
        # message carried no clips at all.)
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

    # TAKE 0's SAMPLING IS THE MANIFEST'S, AND IT REACHES NARRATOR. Not through
    # `caps` on the load message — that channel is `register_voice_caps`, whose
    # vocabulary is narrator's older engine's — but through the
    # NARRATOR_HIGGS_VOICES document
    # `crucible/narratorvoices.py` writes at every load, whose `sampling` key
    # narrator's `load_voices` reads onto the voice and both arms apply as the
    # engine's override. So a voice that deviates from the boson default (with
    # the written reason `crucible/voices.py` requires) renders at what it
    # declares, and there is nothing here to refuse. (Until 2026-09-14 this was
    # a `sampling_not_wired` refusal; it was unreachable with the shipped
    # manifests and, once the document existed, false.)

    # AND A TAKE ABOVE 0 NOW REACHES IT TOO, per item, as BOTH HALVES OF A
    # RUNG. narrator's `generate_batch` items carry `sampling` since 2026-09-14
    # (`narrator/engine/item_sampling.py`), which the engine lays over the
    # voice's loaded numbers key by key, and `take` since 2026-09-15, which
    # moves that row's SEED into the take's own lane
    # (`engine/higgs/truncation.py:in_take_lane`). Both are needed and neither
    # implies the other: a rung that declares no sampling override is still a
    # different draw because the lane moved, and until the seed half landed
    # such a rung rendered take 0 byte for byte.
    #
    # `sampling_not_wired` SURVIVES, with a different subject. It used to mean
    # "the contract has no channel", and that is what stopped being true. It now
    # means "the narrator on this wire has no channel", which is asked of the
    # live process in `_render` rather than assumed here — because the tts env
    # pins narrator by commit and a pin is allowed to be older than the channel.
    # On 2026-09-15 it was, and two takes of one sentence came back byte-
    # identical. The check needs the engine, so it is not in this function.
    # AND A TAKE PAST THE END OF THE LADDER IS NO LONGER REFUSED (2026-09-19).
    # `unknown_take` lived here and is retired with the rule it enforced: a
    # take at or past the last declared rung is a SEED LANE at the voice's own
    # sampling, which `VoiceManifest.take` now answers with directly. It is
    # still not a clamp — the numbers are take 0's, never take 2's under take
    # 4's name — and `take` still rides on every item, so the draw moves.
    # Screening names takes 0..N on a voice that declares no ladder at all, and
    # that is the case the refusal made inexpressible.

    return (
        manifest, spec, interpreter,
        _require_band(params), _require_width(manifest, spec, params),
    )


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
        resident = self._residency.is_resident(KIND_TTS, model)
        _, spec, _, _, _ = _require_renderable(
            self._config, self._backend, model, checked, resident
        )
        if resident:
            # A RENDER ON THE VOICE THAT IS ALREADY ON THE CARD LOADS NOTHING,
            # so there is nothing for the LOAD guard to guard. `_make_resident`
            # has always known this — it returns the resident engine before it
            # reaches the guard — and this door had not caught up, so the
            # cheapest render there is was the one refused.
            #
            # It was not theoretical. Measured on the PC, 2026-09-15 07:30:
            # `load-voice zeroshot` succeeded, `/v1/activity` reported it
            # resident, and the render of that same voice came back
            # `409 accelerator_busy: cannot load 'zeroshot': 18.1 GiB of the
            # 24.0 GiB card is in use by a process this host's driver will not
            # name`. The 18.1 GiB was narrator SERVING `zeroshot` — the guard
            # asked whether there was room to load a voice that was already
            # loaded, `reclaimable_bytes(excluding=model)` is 0 for exactly
            # that voice (unloading it frees nothing you are about to need),
            # and under WSL2 our own engine is unattributed. The same sequence
            # rendered on mlx-darwin, where nothing is unattributed. A cap the
            # card cannot hold is still refused — by `_make_resident`, on the
            # load it would actually perform.
            return
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
            manifest, spec, (python, installed), band, width = _require_renderable(
                self._config, self._backend, voice_id, params,
                self._residency.is_resident(KIND_TTS, voice_id),
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
            try:
                self._render(
                    ctx, params, engine, resident.sample_rate, ffmpeg,
                    manifest, spec, band, width,
                )
            except EngineWouldNotStop as exc:
                # THE CANCEL WAS REAL AND THE ENGINE IS NOT. `_until` already
                # sent the cancel and waited `CANCEL_GRACE_SECONDS` for it to be
                # acted on; reaching here means narrator is still rendering a
                # book nobody is going to read, on a card nobody can have back
                # while it does. So the voice comes off.
                #
                # THROUGH THE ONE UNLOAD DOOR, not a second teardown.
                # `Residency.unload` is what `unload-voice` calls and what
                # `crucible/settle.py` calls when the last holder lets go; this
                # job holds a `may_mutate=True` claim on THIS thread, which is
                # exactly the claim that door lets through. What is different
                # here is only the TRIGGER: the settlement clears a card nobody
                # holds, and this clears one whose engine broke the wire — so it
                # happens even when a lease would have kept the voice resident,
                # because a lease is a promise about the next render and this
                # engine can no longer serve one.
                ctx.note(str(exc))
                self._residency.unload(voice_id)
                # Cancelled, not failed. The operator asked for a stop and got
                # one; the artifacts already written stay fetchable, which is
                # what a client draining after a stop reads.
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
        manifest: VoiceManifest,
        spec: Any,
        band: dict[str, float] | None,
        width: int | None,
    ) -> None:
        """One `generate_batch`, and one FLAC per row as the row retires.

        `manifest` and `spec` are the voice this job resolved to. They are here
        rather than a pre-computed pair of dicts because this method owes the
        job's terminal event two facts about them — see `done_extra` at the
        end — and resolving the rung is the same lookup.

        `band` is the request's own band in narrator's spelling, already
        checked by `_require_band`, or None because none was stated. `width` is
        the in-flight cap THE CALLER STATED, already checked by
        `_require_width`, or None because the caller stated none — in which
        case no width leaves this server and the engine keeps its own.

        **The whole job is one batch, and how wide it runs is now sayable.**
        The list is never cut up here — `generate_batch` does its own
        scheduling, grouping consecutive rows on MLX and dispatching singly on
        vLLM, and slicing it in this file would be Crucible second-guessing a
        read-ahead window it cannot see. `width` is not that: it is a NUMBER on
        the envelope that narrator applies to its own in-flight set, so the
        scheduling stays the engine's and only the ceiling is the job's. Rows
        come back **out of order**, which is why nothing below indexes by
        position.
        """
        sampling = take_sampling(manifest, params.take)
        # A RUNG NEEDS A NARRATOR THAT HAS THE CHANNEL, and that is asked, not
        # assumed. `sampling_not_wired` came back on 2026-09-15 with a new
        # meaning. It used to say "narrator's `generate_batch` takes no sampling"
        # — a statement about the CONTRACT, which stopped being true when
        # `narrator/engine/item_sampling.py` landed, so it was deleted. It now
        # says "the narrator ON THIS WIRE has no such channel", which is a
        # statement about a PROCESS and can never stop being possible: the tts
        # env pins narrator by commit, and a pin is allowed to be old.
        #
        # It was old. Two render jobs that night — voice `owen`, one sentence,
        # take 0 and take 1 (`temperature = 0.7`) — returned byte-identical
        # 264,174-byte artifacts, because the env's narrator (bookforge
        # 0eeb0267) read `item['voice']` and dropped `item['sampling']` without
        # a word. Crucible built the item right and reported a take that had
        # not happened. A wrong take delivered as a success is the failure this
        # job type exists to make impossible, so it refuses instead.
        #
        # THE TEST IS THE TAKE, NOT THE SAMPLING (corrected 2026-09-15 when the
        # seed half landed). It used to be `sampling is not None`, and that
        # asked the wrong question about the right thing. What the client asked
        # for is take N; what the handshake answers is "do you read a rung";
        # `sampling is not None` stood in for both and was equal to neither.
        # It happened to be equivalent only because `voices.py:_check_takes`
        # refuses a rung above 0 that declares no numbers — a rule in a
        # different file, written when a different DRAW was the one thing a
        # rung could not ask for, and now the only thing holding the old gate
        # up. That is a fact with two owners (docs/ARCHITECTURE.md); asking
        # about the take directly has one.
        #
        # Only above take 0. Take 0 sends the numbers nobody (no `sampling`
        # key) and the lane every narrator ever built already draws in, so an
        # old narrator keeps serving the takes it can serve.
        if params.take > 0 and not engine.announces_item_take():
            raise JobError(
                "sampling_not_wired",
                f"take {params.take} resolves to sampling {sampling}, and the "
                f"narrator serving this voice did not announce `itemTake` "
                f"on its ready line — it has no per-item rung channel, so it "
                f"would render take 0, in take 0's seed lane, and this job "
                f"would report take {params.take}. Re-resolve the tts env's "
                f"narrator pin (envs/tts/*.txt) to a bookforge commit that "
                f"carries narrator/engine/item_sampling.py, reinstall the env, "
                f"and reload the voice. Take 0 renders on this narrator as it "
                f"is.",
            )

        by_index = {chunk.index: chunk for chunk in params.chunks}

        expected = set(by_index)
        answered: set[int] = set()
        failures: list[dict[str, Any]] = []
        rendered = 0
        total = len(params.chunks)

        request = {
            "action": "generate_batch",
            "language": params.language,
            # WHICH ARM, SAID OUT LOUD, ON EVERY BATCH (2026-09-19). Until
            # this key existed the arm was chosen by a CAPABILITY PROBE inside
            # narrator — `serve/worker.py`'s `_guards_its_own_batch`, which is
            # `callable(getattr(engine, 'render_many', None))` — so what
            # happened to a book depended on what the engine offered and the
            # caller had no say. The request says now, both ways: `false` takes
            # the plain `_generate_audio_batch` branch that ran for everything
            # before the 2026-09-13 ruling.
            #
            # BATCH-LEVEL AND NOT PER ITEM, and the asymmetry with `sampling`
            # is real rather than an oversight. A rung is per row because a
            # batch may legitimately mix rungs; the guard is a DRIVER — one
            # `render_many` call with one PaceTracker running a median across
            # the batch — and a batch half of whose rows were judged and half
            # not is not a thing narrator can be asked for.
            "retake": params.retake,
            # THE BAND THE GUARD MEASURES AGAINST, forwarded exactly as stated
            # and absent when nothing was stated. Sent even on a bare render
            # (Owen: "it won't do anything with the number because it wasn't
            # asked to"), because this door's job is to carry what the caller
            # said, not to decide which of its statements narrator will find
            # useful.
            **({} if band is None else {"band": band}),
            # HOW MANY ROWS MAY BE IN FLIGHT, batch-level beside the other two,
            # forwarded exactly as stated and ABSENT when the caller stated
            # none — `band`'s discipline, and for the same reason. An absent
            # width means the engine keeps the width it was STARTED at, which
            # narrator knows on both arms and Crucible knows on one
            # (`HIGGS_MAX_NUM_SEQS` on the served arm, `MLX_TIERS` on darwin).
            # Substituting `max_num_seqs` here is what narrowed every Mac batch
            # from 64 to 16 between 1.0.7 and 2026-09-20 — 12.9x realtime to
            # 5.5x, 189 sentences/min to 78, measured on mistborn/Shift Book 2.
            **({} if width is None else {"width": width}),
            # No `stream` flag anywhere in the list. narrator's own docstring:
            # "A generate_batch with no `stream` flag anywhere takes the
            # pre-existing code path, byte for byte." The render door wants whole
            # rows; sub-sentence chunks are the streaming door's, and asking for
            # them here would mean reassembling audio the engine already had in
            # one piece.
            # THE RUNG RIDES ON EVERY ITEM, or on none of them. It is one take
            # per job (`take` is a job-level parameter), so every row carries
            # the same numbers; per ITEM rather than per REQUEST because that
            # is where narrator's channel is — `generate_batch` itself takes no
            # sampling, and a batch may legitimately mix rungs, which is what
            # Correct Sentences will do when it spreads N candidates across the
            # ladder. At take 0 the key is ABSENT, not `{}`: absent means "the
            # voice's loaded sampling", which is what take 0 is.
            #
            # `take` RIDES ON EVERY ITEM INCLUDING TAKE 0, and the asymmetry
            # with `sampling` is deliberate. `{}` is not a sampling — narrator
            # refuses it as `sampling_malformed`, correctly — so absence is the
            # only way to say "no override". But 0 IS a take: it is the
            # documented bottom rung and narrator's `parse_item_take` reads an
            # absent key and an explicit 0 as the same number. Sending it makes
            # the wire say which take produced each artifact, which is the one
            # fact the 2026-09-15 incident had nowhere to write down.
            "items": [
                {"i": chunk.index, "text": chunk.text, "take": params.take}
                | ({} if sampling is None else {"sampling": sampling})
                for chunk in params.chunks
            ],
        }

        # On the RECORD, not only on the event: `chunks_total` beside
        # `chunks_done` is what makes done/total one read after the stream, or
        # the process, is gone.
        ctx.expect_chunks(total)
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
            if kind == "stopped":
                # NARRATOR'S ANSWER TO THE CANCEL THIS DOOR SENT, and it carries
                # nothing. Refused by name until 2026-09-15, which turned every
                # cancelled render into `narrator_protocol: narrator sent a
                # 'stopped' message` — a FAILED job where the operator had asked
                # for a cancelled one, and a failure blamed on the engine for
                # answering a question Crucible asked it.
                #
                # WHEREVER IT LANDS. narrator's main loop acknowledges the cancel
                # after the in-flight batch ends, so on the real wire it arrives
                # AFTER `batch_done` — which means it is still sitting in the
                # inbox when the NEXT conversation on this engine starts, and
                # that conversation would refuse it too. The streaming door has
                # always ignored it (`ttsstream.py:_on_message`); this door is
                # the one that had not caught up.
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
            # WHAT THE ENGINE ACTUALLY SAMPLED WITH, once per job (2026-09-19).
            # The FULL triple as applied — the voice's take-0 numbers with this
            # take's rung laid over them — and never just the override, because
            # the override is meaningless without what it deviates from.
            #
            # It is here because SAMPLING LIVES ON THE MANIFEST AND NOT ON THE
            # REQUEST, which is the right shape (PHASE3-TTS.md section 3: the
            # client asks for take N, the server says what N means) and leaves
            # exactly one hole: a manifest edited between two runs makes two
            # incomparable records that both say "take 0". That is not
            # hypothetical — every Higgs measurement before 2026-09-06 was
            # rendered at temperature 1.0 and is comparable to nothing since,
            # and the entire prior ladder record had to be marked "at the wrong
            # temperature" once already. Pinned into the result, a mismatch is
            # visible instead of silent.
            sampling=manifest.applied_sampling(spec.backend, params.take),
            # AND WHICH WEIGHTS RAN, in the `/v1/voices` row's own three words.
            # `identity` is the pin's sha or the local block's asserted string,
            # and `identity_basis` says which — `verified` or `asserted` — so a
            # ladder's record is self-describing and a reader cannot mistake a
            # directory somebody pointed at for a commit somebody fetched.
            voice={
                "id": manifest.id,
                "identity": spec.weights_identity,
                "identity_basis": spec.identity_basis,
            },
            # THE WIDTH THIS JOB STATED, because it is the number a throughput
            # figure is only comparable against — and `null` when it stated
            # none, which is a fact and not a gap. The engine's own started
            # width is narrator's to know; reporting it here would mean this
            # server inventing a number on the arm where it does not have one,
            # which is exactly the 2026-09-19 defect this is the fix for.
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
        # THE CHUNK INDEX travels with the artifact, so an interrupted job can
        # report which chunks it finished and a resume is a set difference.
        ctx.artifact(destination.name, destination, index=chunk.index)

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
