"""What is on the accelerator right now, whatever kind of thing it is.

PHASE2-LLM.md section 3 said **one resident model at a time**. PHASE3-TTS.md
section 5 generalises it, and the reason is arithmetic rather than architecture:
the accelerator does not care what kind of thing is on it, and a card holding a
Higgs checkpoint has no room for a 9B. So this holds **at most one resident
engine, of either kind**, and loading a voice unloads a model exactly as loading
a model unloads a voice.

This file used to be `crucible/jobs/llm/residency.py`. It moved out from under
`jobs/llm/` because it is no longer the llm's: `jobs/tts/` mutates it too, and a
`tts` job reaching into another job type's package for the thing that owns the
card would make the one-at-a-time rule look like a courtesy between two modules
rather than a property of the server.

Only the exclusive job lane mutates this (the `load-model` / `unload-model` and
`load-voice` / `unload-voice` jobs), so the proxy, `/v1/health` and `/v1/voices`
read a value that is never half-written: an engine is published as resident only
once it has proved it is up, and it is unpublished before it is signalled.

**Since PHASE3-TTS.md section 7 that is no longer the whole story**, and the
claim below is the part that is new. A streaming session is a connection rather
than a job, so it does not queue behind the lane; it holds the resident engine
directly, for as long as somebody is listening. So the card now has a named
owner, and the mutators refuse by name while somebody else has it.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from .alignmodels import AlignBackendSpec, AlignManifest
from .config import Config
from .engines import (
    STOP_TIMEOUT_SECONDS,
    EngineError,
    NarratorEngine,
    SubprocessEngine,
    build_engine,
    build_voice_engine,
    engine_log_path,
    engine_model_name,
    find_free_port,
)
from .errors import ApiError, JobError
from .jobenv import tts_env
from .manifests import (
    NO_DEFAULTS,
    BackendSpec,
    ModelDefaults,
    ModelManifest,
    fingerprint,
)
from .narratorvoices import DOCUMENT_READERS, write_document
from .voicereference import VoiceReference
from .voices import VoiceBackendSpec, VoiceManifest
from .workers import WorkerError, WorkerSession

#: The kinds of thing that can hold the card, and what `/v1/health` reports as
#: `resident_kind` so a client can tell which door to knock on.
#:
#: `align` is the third, and it is a different *shape* of resident thing rather
#: than a third engine: an LLM and a voice are both HTTP-or-stdio servers behind
#: `SubprocessEngine`, while the aligner is a `workers.WorkerSession` — the same
#: JSON-lines worker every phase 4 type speaks to, held open instead of run once
#: (PHASE4-AUDIO.md section 2). It gets its own resident dataclass and its own
#: holder slot rather than being dressed up as an engine, because an aligner has
#: no `base_url`, nothing to proxy to, and no readiness route; pretending
#: otherwise would put three lies in a row on one row of `/v1/health`.
KIND_LLM = "llm"
KIND_TTS = "tts"
KIND_ALIGN = "align"

#: How long a load waits for the engine to prove it is up. vLLM on a 19 GB model
#: spends most of it reading weights and capturing CUDA graphs; narrator on
#: `cuda-linux` spends it starting SGLang-Omni, measured at about 110 s.
DEFAULT_READY_TIMEOUT_SECONDS = 900.0

#: How long an unload job waits out a clearance of its own subject that is
#: already under way (`Residency.await_clearance`).
#:
#: The engine's OWN SIGTERM deadline plus a margin, and derived rather than
#: chosen: the waiter is waiting for the settlement to release the card, and the
#: settlement cannot release it until `SubprocessEngine.stop()` either returns or
#: gives up at `STOP_TIMEOUT_SECONDS`. A number equal to that deadline would race
#: the `EngineError` it raises and report a wedge that was about to resolve.
CLEARANCE_TIMEOUT_SECONDS = STOP_TIMEOUT_SECONDS + 30.0


@dataclass(frozen=True)
class ResidentModel:
    kind = KIND_LLM

    model_id: str
    backend: str
    engine: str
    engine_model_name: str
    base_url: str
    port: int
    revision: str
    #: The context this engine was actually started with — vLLM's
    #: `--max-model-len`, mlx-lm's own config. It comes from
    #: `ModelManifest.context_for(backend)` at load time, which is why it is not
    #: simply read back off the manifest: a manifest edited while this engine is
    #: up would then describe a context nothing is serving.
    max_model_len: int
    memory_bytes_estimate: int
    log_path: Path
    loaded_at: str
    #: `[defaults]` as the manifest read at LOAD time, for `max_model_len`'s
    #: reason one field up: a manifest edited while this engine is up must not
    #: change what a request in flight is answered with, and the record is what
    #: the door reports. `crucible/sampling.py` applies it; `NO_DEFAULTS` is a
    #: model that states none, which is not the same as a field nobody set.
    defaults: ModelDefaults = NO_DEFAULTS

    @property
    def id(self) -> str:
        return self.model_id

    @property
    def fingerprint(self) -> str:
        """`<id>@<revision>` for the weights this engine actually read.

        A property here and a field on `ResidentVoice`, because the two are
        pinned differently: a model's revision is the one its manifest names,
        while a voice's comes off the manifest at load time through
        `VoiceManifest.fingerprint(backend)`. Both spell it with the same helper.
        """
        return fingerprint(self.model_id, self.revision)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "backend": self.backend,
            "engine": self.engine,
            "engine_model_name": self.engine_model_name,
            "base_url": self.base_url,
            "revision": self.revision,
            "fingerprint": self.fingerprint,
            "max_model_len": self.max_model_len,
            # What a request that states nothing will be answered with, read off
            # the record rather than the file, so it cannot disagree with what
            # the proxy is actually applying.
            "defaults": self.defaults.to_dict(),
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "log_path": str(self.log_path),
            "loaded_at": self.loaded_at,
        }


@dataclass(frozen=True)
class ResidentVoice:
    """A voice narrator is currently serving.

    No `base_url`: narrator's wire is newline-delimited JSON over stdin and
    stdout, not HTTP, so there is nothing for a client to be proxied to and the
    field would be a lie if it were here to make the two shapes symmetrical.
    `fingerprint` is what a render records as the voice it used, and it is bound
    to the revision rather than to the id, because two merges of one run are two
    sets of weights under one name.
    """

    kind = KIND_TTS

    voice_id: str
    backend: str
    narrator_engine: str
    revision: str
    fingerprint: str
    sample_rate: int
    max_chars: int
    memory_bytes_estimate: int
    log_path: Path
    loaded_at: str
    #: What a zero-shot voice is conditioned on, `{name, sha256, seconds}`, or
    #: None for every other kind. A `zeroshot` row that said only `zeroshot`
    #: would be two clients looking at one word and each assuming it was their
    #: clip; the digest is what lets either of them tell.
    reference: dict[str, Any] | None = None

    @property
    def id(self) -> str:
        return self.voice_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "voice": self.voice_id,
            "backend": self.backend,
            "narrator_engine": self.narrator_engine,
            "revision": self.revision,
            "fingerprint": self.fingerprint,
            "sample_rate": self.sample_rate,
            "max_chars": self.max_chars,
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "log_path": str(self.log_path),
            "loaded_at": self.loaded_at,
            # Always present, `null` for a voice that is not conditioned on a
            # clip — an absent key would mean "this build does not say".
            "reference": self.reference,
        }


@dataclass(frozen=True)
class ResidentAligner:
    """The forced aligner Crucible currently has loaded.

    No `base_url` and no `engine`, for `ResidentVoice`'s reason and one more: the
    aligner is not a server at all. It is `crucible/jobs/align/worker.py` held
    open by a `workers.WorkerSession`, and what makes it *resident* is that the
    1.7 GB checkpoint stays on the card between jobs — hundreds of chunks and one
    load, which is the whole point (PHASE4-AUDIO.md section 2).

    `device` and `dtype` are on the row because they are what the model was
    actually loaded with, not what a manifest says it prefers: `bfloat16` is what
    the bake-off measured, and a row that did not name it could not tell a reader
    whether the timings they are looking at came from that arrangement.
    """

    kind = KIND_ALIGN

    aligner_id: str
    backend: str
    revision: str
    fingerprint: str
    device: str
    dtype: str
    max_audio_s: float
    memory_bytes_estimate: int
    log_path: Path
    loaded_at: str

    @property
    def id(self) -> str:
        return self.aligner_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "aligner": self.aligner_id,
            "backend": self.backend,
            "revision": self.revision,
            "fingerprint": self.fingerprint,
            "device": self.device,
            "dtype": self.dtype,
            "max_audio_s": self.max_audio_s,
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "log_path": str(self.log_path),
            "loaded_at": self.loaded_at,
        }


Resident = ResidentModel | ResidentVoice | ResidentAligner


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


#: What to call each kind in a refusal. A mapping and not a two-way conditional,
#: because there are three of them now and "the resident model is
#: 'qwen3-aligner'" would be a sentence that sends its reader to `unload-model`.
KIND_NOUNS: dict[str, str] = {
    KIND_LLM: "model",
    KIND_TTS: "voice",
    KIND_ALIGN: "aligner",
}


def describe_resident(residency: "Residency", kind: str, absent: str) -> str:
    """What holds the card, as the tail of a `*_not_resident` refusal.

    One wording for every door. `absent` is what to say when nothing is resident,
    in the vocabulary of the door the reader came through — "no model is" for
    `unload-model`, "no voice is" for `unload-voice`, "no aligner is" for
    `unload-aligner` — and when something of ANOTHER kind is resident the message
    says so by name, because "no model is resident" while narrator holds the
    whole card is true and useless.
    """
    resident = residency.resident
    if resident is None:
        return absent
    if resident.kind == kind:
        return f"{resident.id!r} is"
    return f"the resident {KIND_NOUNS[resident.kind]} is {resident.id!r}"


class Residency:
    """The one-resident-engine holder for a server instance."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._resident: Resident | None = None
        self._engine: SubprocessEngine | None = None
        # The held thing when the resident is an aligner. A second slot rather
        # than one `_held` of a union type, because the two are stopped
        # differently and nothing good comes of a `hasattr` deciding which: an
        # engine is stopped through `EngineError`, a session through
        # `WorkerError`, and each caller catches the one its own door raises.
        self._session: WorkerSession | None = None
        self._warming: str | None = None
        self._claim: str | None = None
        #: The thread a mutating claimant took the card on, or None when the
        #: claimant declared it will not mutate. See `claim`.
        self._claim_thread: int | None = None
        #: True while the current claim is held in order to TAKE THE RESIDENT
        #: THING OFF the card, rather than to use it. Only the settlement claims
        #: that way (`crucible/settle.py`). See `being_cleared`.
        self._claim_clears = False
        #: The subject a clearing claim took off the card, while the card has
        #: stayed empty since — *why* the card is empty, which is a different
        #: fact from *that* it is. Written at the one door everything comes off
        #: the card through (`unload`). See `being_cleared`.
        self._cleared: str | None = None
        #: A Condition and not a Lock, because a claim is now something a thread
        #: can WAIT OUT (`await_clearance`) as well as something it reads.
        self._claim_lock = threading.Condition()

    # ------------------------------------------------- the exclusive claim
    #
    # PHASE3-TTS.md section 7 put a second user on the card. Until it, every
    # user of the resident engine was a job, the exclusive lane serialised them,
    # and "one at a time" needed no mechanism. A streaming session is **not a
    # job** — it is a connection that lives for as long as somebody is listening
    # — so it runs beside the lane, and narrator has exactly one stdin, one
    # stdout and one `_inbox`. Two conversations on that wire do not collide
    # loudly; they steal each other's `batch_item` lines, and the symptom is a
    # row of audio delivered under another row's id. Worse, `load_voice` would
    # SIGTERM the engine out from under a session mid-sentence.
    #
    # So the card has a named owner. A streaming session claims it for its
    # lifetime; the render door claims it for the duration of its batch; and
    # everything that mutates residency refuses by name while somebody else
    # holds it, rather than proceeding and corrupting the wire.

    @property
    def claimed_by(self) -> str | None:
        """Who holds the resident engine's exclusive attention, or None."""
        return self._claim

    @contextmanager
    def claimed(self, holder: str, *, may_mutate: bool) -> Iterator[None]:
        """Hold the card for the duration of the block, or refuse by name.

        Deliberately **not** blocking. A second claimant is told who has it and
        goes away; a claimant that waited would turn "the card is busy" into a
        hang with no event to explain it, which is the one thing DESIGN.md
        section 10 will not have.
        """
        self.claim(holder, may_mutate=may_mutate)
        try:
            yield
        finally:
            self.release(holder)

    def claim(self, holder: str, *, may_mutate: bool, clears: bool = False) -> None:
        """Take the card. `may_mutate` is a promise about what will be done to it.

        `clears` is a promise of a different kind — *"this claim exists in order
        to take the resident thing OFF the card"* — and only the settlement
        makes it (`crucible/settle.py`). It is what lets `being_cleared` tell a
        holder that is using the card from one that is emptying it, so an
        `unload-...` for the very thing being cleared is answered as the same
        intent instead of refused `engine_in_use`.

        The two claimants are not alike, and the flag is what keeps the guard
        honest for both rather than being loosened until it fits the looser one:

        - A **render job** claims with `may_mutate=True`, because loading its own
          voice is the first thing it does (PHASE3-TTS.md section 6's one
          asymmetry with `llm`). So the thread it claimed on may load and unload,
          and every other thread is still refused.
        - A **streaming session** claims with `may_mutate=False`. It never loads
          — the streaming door never loads, exactly as chat never loads — so no
          thread is exempted and every mutation there is refuses by name,
          including one from whichever thread happened to open the session.

        An exemption is a thread identity rather than a name, because the thing
        being prevented is a *second conversation*, and the claimant's own thread
        is by definition not one.
        """
        with self._claim_lock:
            if self._claim is not None:
                raise JobError(
                    "engine_in_use",
                    f"the resident engine is held by {self._claim!r} and "
                    f"{holder!r} cannot have it at the same time. narrator has "
                    "one stdin and one stdout, so two conversations on it read "
                    "each other's replies",
                )
            self._claim = holder
            self._claim_thread = threading.get_ident() if may_mutate else None
            self._claim_clears = clears

    def release(self, holder: str) -> None:
        with self._claim_lock:
            if self._claim != holder:
                # Not a silent no-op: releasing somebody else's claim would free
                # the wire under a conversation that is still on it.
                raise JobError(
                    "engine_in_use",
                    f"{holder!r} tried to release the card, which is held by "
                    f"{self._claim!r}",
                )
            self._claim = None
            self._claim_thread = None
            self._claim_clears = False
            # Whoever is waiting out a clearance is waiting for exactly this.
            self._claim_lock.notify_all()

    # ------------------------------------------ the card being cleared of it
    #
    # A CLEARING CLAIM IS NOT A FOREIGN HOLDER. Found by PHASE15-HOST.md section
    # 8's T6 on a live card, 2026-09-15: the settlement started clearing
    # `dots-ocr` the instant the last chat completion finished, the same client's
    # own `unload-model dots-ocr` landed a few milliseconds later, and it was
    # refused `engine_in_use` — held by *"the settlement clearing the card"*. The
    # refusal's whole meaning is *"somebody else is using this and taking it off
    # the card would end their conversation mid-sentence"*, and none of that is
    # true of a settlement: it is doing the very thing the request asked for.
    #
    # So the three unload doors ask this first. Everything else the claim
    # refuses, it still refuses by name.

    def being_cleared(self, subject_id: str) -> bool:
        """`subject_id` is on its way off the card, or is off it because it was.

        True in two states, because a client's `unload-...` can arrive in
        either — the window is milliseconds wide and both halves of it are the
        same answer:

        - a clearing claim is up and `subject_id` is what is on the card, or
        - the card is empty, and it is empty BECAUSE a clearing claim took
          `subject_id` off it and nothing has been on it since.

        False for everything else, including a claim held to USE the card. That
        one is a genuinely different holder and `refuse_if_claimed` says so.
        """
        with self._claim_lock:
            return self._being_cleared(subject_id)

    def _being_cleared(self, subject_id: str) -> bool:
        """`being_cleared`, for a caller already holding `_claim_lock`."""
        resident = self._resident
        if resident is None:
            return self._cleared == subject_id
        return self._claim_clears and resident.id == subject_id

    def await_clearance(
        self, subject_id: str, *, timeout: float = CLEARANCE_TIMEOUT_SECONDS
    ) -> bool:
        """Wait out a clearance of `subject_id`. **Never the event loop.**

        True when the card is clear of `subject_id` because the clearance did
        it: the unload job that asked has nothing left to do and is `done`.
        False when no clearance of `subject_id` is under way at all — the
        ordinary case, and the caller unloads it itself. False too when a
        clearance was under way and DECLINED to unload (a holder appeared under
        its claim), because then the thing is still on the card and unloading it
        is still this job's work.

        A CLEARANCE THAT NEVER FINISHES IS NOT WAITED OUT QUIETLY. The timeout
        is longer than the engine's own SIGTERM deadline, so reaching it means
        something is wedged, and that is raised rather than turned into a second
        unload on top of the first.
        """
        with self._claim_lock:
            if not self._being_cleared(subject_id):
                return False
            deadline = time.monotonic() + timeout
            while self._claim_clears:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise JobError(
                        "engine_in_use",
                        f"waited {timeout:.0f}s for the card to be cleared of "
                        f"{subject_id!r} and it has not been. It is still held "
                        f"by {self._claim!r}, which is longer than the engine's "
                        "own SIGTERM deadline: something is wedged, and "
                        "unloading on top of it would make it worse",
                    )
                self._claim_lock.wait(remaining)
            return self._resident is None and self._cleared == subject_id

    def refuse_if_claimed(self, what: str) -> None:
        """The same refusal as an HTTP 409, for a preflight to make before queuing.

        WHICH JOB TYPES ASK THIS, AND WHY THE REST DO NOT. Admission is the
        server's one scheduling answer (ARCHITECTURE.md section 3) and the lane
        is only half of it: a streaming session holds the resident engine
        *without* occupying the lane, so `JobStore.refuse_if_busy` can say the
        server is free while the card is not. This is the other half, and it is
        asked by each type rather than centrally because the claim is about
        **narrator's one stdin and one stdout**, not about VRAM in general — so
        the answer genuinely differs per type:

        - `load-model`, `unload-model`, `load-voice`, `unload-voice`,
          `unload-aligner` **ask**: they move what is on the card, and doing that
          under a session ends its conversation mid-sentence.
        - `tts` (render) **asks**: it talks to the resident narrator directly, and
          two conversations on that one pipe read each other's replies.
        - `align` **asks** (added with the admission ruling): it is the third
          mutator of residency, and without this it was accepted and then failed a
          minute later at `_refuse_mutation_if_claimed`.
        - `asr` and `rvc` **do not, deliberately**. Neither touches the resident
          engine — they spawn their own worker and are handed only `owned_pids` —
          so what they contend for is memory, not the wire. That contention is
          already answered, by name, by `accelerator.guard` in their preflights,
          which counts a resident engine's bytes as taken. Making them ask here
          would quietly redefine the claim from "narrator's wire" to "the card",
          which is a different rule and would need to be ruled as one.
        - `echo` **does not**: it takes no accelerator at all. The lane is the
          only thing it can be busy with.
        """
        holder = self._claim
        if holder is None:
            return
        raise ApiError(
            409,
            "engine_in_use",
            f"{what} needs the card, which is held by {holder!r}. The card has "
            "one holder at a time, and both ways past that are damage: narrator "
            "has one stdin and one stdout, so a second conversation reads the "
            "first one's replies, and a job that loads or unloads would take the "
            "engine off the card mid-sentence",
            {"held_by": holder},
        )

    def _refuse_mutation_if_claimed(self, what: str) -> None:
        """The backstop, at the three places that actually move the weights.

        `refuse_if_claimed` answers the client before a job is queued; this
        answers the lane if a session opened in between. Both say
        `engine_in_use`, because it is the same fact.
        """
        holder = self._claim
        if holder is not None and self._claim_thread != threading.get_ident():
            raise JobError(
                "engine_in_use",
                f"cannot {what}: the resident engine is held by {holder!r}. "
                "Taking it off the card now would end that conversation "
                "mid-sentence",
            )

    # -------------------------------------------------------------- reading

    @property
    def resident(self) -> Resident | None:
        return self._resident

    @property
    def resident_id(self) -> str | None:
        return None if self._resident is None else self._resident.id

    @property
    def resident_kind(self) -> str | None:
        """`"llm"`, `"tts"`, or None — `/v1/health`'s `resident_kind`."""
        return None if self._resident is None else self._resident.kind

    @property
    def resident_model(self) -> ResidentModel | None:
        """The resident, if it is a model. None when a voice holds the card.

        The OpenAI proxy asks for this rather than for `resident`: with a voice
        resident there is no `base_url` to forward a chat request to, and the
        proxy's `model_not_resident` is the honest answer.
        """
        return self._resident if isinstance(self._resident, ResidentModel) else None

    @property
    def resident_voice(self) -> ResidentVoice | None:
        """The resident, if it is a voice. None when a model holds the card."""
        return self._resident if isinstance(self._resident, ResidentVoice) else None

    @property
    def voice_engine(self) -> NarratorEngine | None:
        """The narrator process serving the resident voice, or None.

        `resident_model` carries a `base_url` and that is all the proxy needs;
        there is no such string for a voice, because narrator answers no HTTP
        route. So the render door is handed the engine OBJECT — it is the
        channel — and it is published only while a voice is resident, which is
        exactly the window in which sending narrator a `generate_batch` means
        anything.
        """
        if not isinstance(self._resident, ResidentVoice):
            return None
        engine = self._engine
        if not isinstance(engine, NarratorEngine):  # unreachable
            raise EngineError(
                f"a voice is resident but the engine holding the card is "
                f"{type(engine).__name__}, not a narrator"
            )
        return engine

    @property
    def resident_aligner(self) -> ResidentAligner | None:
        """The resident, if it is an aligner. None otherwise."""
        return self._resident if isinstance(self._resident, ResidentAligner) else None

    @property
    def aligner_session(self) -> "WorkerSession | None":
        """The held worker behind the resident aligner, or None.

        This is what `align` sends each job's chunks down. It is exposed rather
        than wrapped because the job type owns the *vocabulary* of an align
        request and this module owns the *lifetime* of the process — and a
        `Residency.align(...)` would be this module learning what a chunk is.
        """
        return None if self.resident_aligner is None else self._session

    @property
    def warming(self) -> str | None:
        """The id a load job is currently warming, or None."""
        return self._warming

    def ids(self) -> list[str]:
        return [] if self._resident is None else [self._resident.id]

    def is_resident(self, kind: str, subject_id: str) -> bool:
        """Is exactly this thing on the card?

        Kind **and** id. Model ids and voice ids are separate namespaces, and
        nothing stops a voice being called `qwen3.5-9b`; asking on the id alone
        would let a resident voice light up a model's `/v1/models` row.
        """
        return (
            self._resident is not None
            and self._resident.kind == kind
            and self._resident.id == subject_id
        )

    def begin_warming(self, subject_id: str) -> None:
        """Mark a load as in progress, so `/v1/health` says `warming`.

        Set for the whole load job, not just the engine's readiness poll: from
        the client's side, "this server is warming something" is true from the
        moment the lane picks the job up.
        """
        self._warming = subject_id

    def end_warming(self) -> None:
        self._warming = None

    def owned_pids(self) -> frozenset[int]:
        """Every pid Crucible itself has on the card.

        Both slots, unioned, so the accelerator guard never reports Crucible's own
        resident aligner as somebody else's process holding the card. Only one of
        them is ever non-empty — one card holds one thing — but reading only the
        engine slot was the bug waiting to happen the moment a second shape of
        resident thing existed.
        """
        pids: frozenset[int] = frozenset()
        if self._engine is not None:
            pids |= self._engine.pids
        if self._session is not None:
            pids |= self._session.pids
        return pids

    def reclaimable_bytes(self, excluding: str | None = None) -> int:
        """What unloading the current resident would give back.

        Zero when the resident *is* `excluding` — reloading a model does not free
        its own memory before it needs it again. Across kinds it is never zero:
        a voice's bytes are as reclaimable as a model's, which is the whole point
        of one holder for both.
        """
        if self._resident is None or self._resident.id == excluding:
            return 0
        return self._resident.memory_bytes_estimate

    # -------------------------------------------------------------- writing

    def _evict(self, say: Callable[[str], None], incoming: str) -> None:
        """Unload whatever is there, naming what it was, before `incoming` loads."""
        if self._resident is None:
            return
        previous = self._resident
        say(
            f"unloading {previous.id} (the resident {previous.kind}) to make room "
            f"for {incoming} — one resident engine at a time, of either kind"
        )
        self.unload(previous.id)

    def load(
        self,
        manifest: ModelManifest,
        spec: BackendSpec,
        weights_dir: Path,
        python: Path,
        *,
        timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
        on_progress: Callable[[str], None] | None = None,
    ) -> ResidentModel:
        """Make this model the resident one, unloading whatever was there."""
        self._refuse_mutation_if_claimed(f"load {manifest.id}")

        def say(message: str) -> None:
            if on_progress is not None:
                on_progress(message)

        self._evict(say, manifest.id)

        log_path = engine_log_path(self._config.home, manifest.id)
        # WHICH SERVER CLASS comes off the manifest's own `engine`, which is
        # all that changed when `BACKEND_ENGINES` became one engine per
        # (backend, class family) on 2026-09-14: the loader has already refused
        # every pairing that is not allowed, so the name arriving here is the
        # right one for this model's family and the residency never learns
        # what a family is.
        engine = build_engine(spec.engine, python, log_path)
        served = engine_model_name(spec.engine, weights_dir, manifest.id)
        port = find_free_port()

        context = manifest.context_for(spec.backend)
        self.begin_warming(manifest.id)
        say(
            f"starting {spec.engine} for {manifest.id} on 127.0.0.1:{port} "
            f"(context {context}); log {log_path}"
        )
        try:
            self._start(
                engine,
                weights_dir,
                served,
                port,
                self._engine_args(manifest, spec, weights_dir),
                say,
                timeout,
            )
        finally:
            self.end_warming()

        self._engine = engine
        self._resident = ResidentModel(
            model_id=manifest.id,
            backend=spec.backend,
            engine=spec.engine,
            engine_model_name=served,
            base_url=engine.base_url,
            port=port,
            revision=spec.revision,
            max_model_len=context,
            defaults=manifest.defaults,
            memory_bytes_estimate=spec.memory_bytes_estimate,
            log_path=log_path,
            loaded_at=_now(),
        )
        say(f"{manifest.id} is resident at {engine.base_url}")
        return self._resident

    def load_voice(
        self,
        manifest: VoiceManifest,
        spec: VoiceBackendSpec,
        weights_dir: Path,
        python: Path,
        *,
        reference: VoiceReference | None = None,
        timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
        on_progress: Callable[[str], None] | None = None,
    ) -> ResidentVoice:
        """Make this voice the resident one, unloading whatever was there.

        A Higgs v3 voice change IS a full worker restart — the voice is the merged
        checkpoint the engine was started on — so there is no cheaper path here
        than the one a model takes, and none is pretended at.

        `reference` is the clip a zero-shot voice is conditioned on: required of
        one, refused on any other kind, and refused by `voice_entry` below
        rather than here, so there is one statement of that rule and the load
        door's own `reference_required` is the same rule made earlier.
        """
        self._refuse_mutation_if_claimed(f"load {manifest.id}")

        def say(message: str) -> None:
            if on_progress is not None:
                on_progress(message)

        self._evict(say, manifest.id)

        log_path = engine_log_path(self._config.home, manifest.id)
        # THE SERVER'S OWN CONFIGURATION, from its three owners: the env recipe
        # says which serving stack narrator will start (None where it starts
        # none), the voice manifest says how wide it admits, and the voices
        # DOCUMENT — written here, now, from that manifest and the pulled
        # directory — says which weights, which cap and which sampling the
        # voice is. All three are stated here rather than left to the engine
        # to find, because narrator refuses each of them BY NAME: the first
        # real render died on the stack before `ready` (HIGGS_STACK is not
        # set, exit 3), and the next one died at the load on both arms
        # (`Higgs v3 load carried modelDir=...`), because a Higgs voice is a
        # NAME in the NARRATOR_HIGGS_VOICES document and never a directory on
        # the message. Regenerated at every load so it can never name a voice
        # whose stamp has since moved. An engine outside `DOCUMENT_READERS`
        # reads no document and is handed none.
        env_spec = tts_env(manifest.narrator_engine, spec.backend)
        voices = (
            write_document(
                self._config.home, manifest, spec, weights_dir, reference
            )
            if manifest.narrator_engine in DOCUMENT_READERS
            else None
        )
        engine = build_voice_engine(
            manifest.narrator_engine,
            python,
            log_path,
            serving_stack=env_spec.serving_stack,
            max_num_seqs=(
                None if manifest.serving is None
                else manifest.serving.max_num_seqs
            ),
            voices=voices,
        )
        # narrator answers no HTTP route, so this port is not a proxy target; it
        # is found and passed for the same reason every other engine's is, so
        # that an engine which does decide to bind something has a free one.
        port = find_free_port()

        self.begin_warming(manifest.id)
        say(
            f"starting narrator ({manifest.narrator_engine}) for {manifest.id} "
            f"on {spec.backend}; log {log_path}"
        )
        try:
            # `ready` says narrator is listening; it does not say a voice is in
            # memory. A `load-voice` job that stopped at `ready` would report a
            # resident voice while the card was empty, and the first render would
            # be the thing that found out. So the load message is part of the
            # load, and a failure in it tears the engine down exactly as a
            # readiness failure does — which is what `_start`'s `confirm`
            # argument is: the proof that comes after the announcement.
            self._start(
                engine,
                weights_dir,
                manifest.id,
                port,
                [],
                say,
                timeout,
                confirm=lambda: self._load_the_voice(
                    engine, manifest, weights_dir, say
                ),
            )
        finally:
            self.end_warming()

        self._engine = engine
        self._resident = ResidentVoice(
            voice_id=manifest.id,
            backend=spec.backend,
            narrator_engine=manifest.narrator_engine,
            revision=spec.revision,
            fingerprint=manifest.fingerprint(spec.backend),
            sample_rate=manifest.sample_rate,
            max_chars=spec.max_chars,
            memory_bytes_estimate=spec.memory_bytes_estimate,
            log_path=log_path,
            loaded_at=_now(),
            reference=None if reference is None else reference.to_report(),
        )
        say(f"{manifest.id} is resident")
        return self._resident

    def load_aligner(
        self,
        manifest: AlignManifest,
        spec: AlignBackendSpec,
        weights_dir: Path,
        python: Path,
        script: Path,
        *,
        device: str,
        dtype: str,
        max_audio_s: float,
        timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
        on_progress: Callable[[str], None] | None = None,
    ) -> ResidentAligner:
        """Make this aligner the resident thing, unloading whatever was there.

        The load is the session's FIRST exchange — a `{"op": "load"}` request the
        worker answers with `ready` once the checkpoint is on the device. It is a
        real exchange and not a bare spawn on purpose: a process that has started
        has proved only that python runs, while a `ready` to a load has proved
        that 1.7 GB of weights are where the next job expects them. That is the
        same bar `engine.ready()` sets for vLLM, met the way this worker can meet
        it.
        """
        self._refuse_mutation_if_claimed(f"load {manifest.id}")

        def say(message: str) -> None:
            if on_progress is not None:
                on_progress(message)

        self._evict(say, manifest.id)

        log_path = engine_log_path(self._config.home, manifest.id)
        session = WorkerSession(python=python, script=script, log_path=log_path)

        self.begin_warming(manifest.id)
        say(
            f"loading {manifest.id} ({spec.engine}) on {device} at {dtype}; "
            f"log {log_path}"
        )
        try:
            outcome = session.start(
                {
                    "op": "load",
                    "model_dir": str(weights_dir),
                    "device": device,
                    "dtype": dtype,
                },
                ready_silence_timeout=timeout,
                on_ready=lambda message: say(
                    f"{manifest.id} loaded in {message['seconds']:.1f}s on "
                    f"{message['device']} at {message['dtype']}"
                ),
                on_progress=lambda message: say(str(message["message"])),
            )
        finally:
            self.end_warming()
        if outcome.results:
            # `start` already stopped nothing — the worker is alive and holding
            # the card — so this refuses loudly rather than letting a worker that
            # answered a load with chunk results go on to answer a book.
            session.stop()
            raise WorkerError(
                f"{script.name} answered a load request with "
                f"{len(outcome.results)} result(s); a load produces none"
            )

        self._session = session
        self._resident = ResidentAligner(
            aligner_id=manifest.id,
            backend=spec.backend,
            revision=spec.revision,
            fingerprint=fingerprint(manifest.id, spec.revision),
            device=device,
            dtype=dtype,
            max_audio_s=max_audio_s,
            memory_bytes_estimate=spec.memory_bytes_estimate,
            log_path=log_path,
            loaded_at=_now(),
        )
        say(f"{manifest.id} is resident")
        return self._resident

    @staticmethod
    def _load_the_voice(
        engine: NarratorEngine,
        manifest: VoiceManifest,
        weights_dir: Path,
        say: Callable[[str], None],
    ) -> dict[str, Any]:
        """Send narrator its `load`, and refuse a sample rate that disagrees.

        **The sample rate is narrator's, not Crucible's.** `/v1/voices` publishes
        `sample_rate` off the manifest and a client writes FLACs at it; narrator
        reports on its `loaded` line the rate the engine it actually built
        renders at. Those two being 24000 for every voice in the catalog is the
        kind of coincidence that becomes a hard-coded constant, so they are
        compared, and a disagreement is a **refusal naming both numbers**. It is
        deliberately not a resample: audio resampled to match a manifest is audio
        that no longer matches the engine, and nothing downstream would say so.
        """
        say(f"loading {manifest.id} into narrator from {weights_dir}")
        loaded = engine.load(
            voice=manifest.id, weights_dir=weights_dir, warm=True, on_progress=say
        )
        reported = loaded.get("sampleRate")
        if not isinstance(reported, int) or isinstance(reported, bool):
            raise EngineError(
                f"{engine.name} loaded {manifest.id} and reported sampleRate "
                f"{reported!r}, which is not a sample rate. Every duration and "
                "every byte count downstream is derived from it"
            )
        if reported != manifest.sample_rate:
            raise EngineError(
                f"{engine.name} renders {manifest.id} at {reported} Hz, but "
                f"{manifest.path.name} declares {manifest.sample_rate}. Crucible "
                "refuses rather than resampling: a FLAC written at the manifest's "
                "rate from bytes generated at the engine's is a chunk of the "
                "wrong length, and nothing in the file would say so. Fix the "
                "manifest, or find out why the engine changed"
            )
        say(
            f"narrator loaded {manifest.id}: engine {loaded.get('engine')!r}, "
            f"backend {loaded.get('backend')!r}, {reported} Hz"
        )
        return loaded

    @staticmethod
    def _start(
        engine: SubprocessEngine,
        weights_dir: Path,
        served: str,
        port: int,
        args: list[str],
        say: Callable[[str], None],
        timeout: float,
        confirm: Callable[[], Any] | None = None,
    ) -> None:
        """Spawn and wait, tidying up a half-started engine without hiding why.

        `confirm` is whatever else must be true before this engine counts as
        loaded. A model's engine has nothing there — a 200 from `/v1/models`
        means the weights are on the card. A voice's has narrator's own `load`,
        because `ready` only means the process is listening.
        """
        try:
            engine.start(weights_dir, served, port, args)
            engine.ready(timeout, on_progress=say)
            if confirm is not None:
                confirm()
        except EngineError as start_failure:
            # Tidy up the half-started engine, but report the *start* failure —
            # that is the one that explains the load. A stop failure on top of it
            # is appended, never substituted.
            try:
                engine.stop()
            except EngineError as stop_failure:
                raise EngineError(
                    f"{start_failure}\n...and stopping it also failed: {stop_failure}"
                ) from start_failure
            raise

    @staticmethod
    def _engine_args(
        manifest: ModelManifest, spec: BackendSpec, weights_dir: Path
    ) -> list[str]:
        """The manifest's args plus what Crucible always sets.

        `--max-model-len` only goes to vLLM; mlx-lm takes the context from the
        model's own config and has no such flag (see engines/mlx_lm.py).

        `llama-server` is the one engine that has to be told WHERE THE FILES
        ARE, because its weights are named files inside a directory rather
        than the directory itself: `-m <dir>/<file>`, `--mmproj <dir>/<mmproj>`
        for a vision model, and `-c <context>`. The manifest carries
        `--parallel 1` and nothing else — PHASE15-HOST.md 7.4 item 3: *"only
        the server knows where it put the weights"*, so composing these in a
        manifest would be a path with two owners.
        """
        args = list(spec.engine_args)
        if spec.engine == "vllm":
            args += ["--max-model-len", str(manifest.context_for(spec.backend))]
        if spec.engine == "llama-server":
            if spec.file is None:  # pragma: no cover - the loader requires it
                raise EngineError(
                    f"{manifest.path.name}'s {spec.backend} block names no "
                    "`file`, and llama-server serves one GGUF. A block for "
                    "this backend without a file is a block for nothing"
                )
            args = ["-m", str(weights_dir / spec.file)] + args
            if spec.mmproj is not None:
                args += ["--mmproj", str(weights_dir / spec.mmproj)]
            args += ["-c", str(manifest.context_for(spec.backend))]
        return args

    def unload(self, subject_id: str) -> Resident:
        """Stop whatever is serving `subject_id`. Raises KeyError if not resident.

        One door for all three kinds, because "one card holds one thing" is only
        true if there is one place that takes it off. Which of the two holder
        slots is occupied decides how — `engine.stop()` raises `EngineError`,
        `session.stop()` raises `WorkerError` — and both are let out, because a
        thing that would not stop is the one fact a caller must not be told a
        soothing version of.
        """
        self._refuse_mutation_if_claimed(f"unload {subject_id}")
        resident = self._resident
        if resident is None or resident.id != subject_id:
            raise KeyError(subject_id)
        engine, session = self._engine, self._session
        # Unpublish first: from here on nothing new is proxied to a dying engine,
        # and no align job is handed a session that is being stopped.
        self._resident = None
        self._engine = None
        self._session = None
        with self._claim_lock:
            # WHY the card is now empty, recorded here because this is the one
            # door everything comes off it through. A clearance leaves the
            # subject's name behind so an `unload-...` that arrives a moment
            # late is answered as the same intent (`being_cleared`); any other
            # unload wipes it, because the emptiness now has a different cause.
            self._cleared = subject_id if self._claim_clears else None
        if engine is not None:
            engine.stop()
        if session is not None:
            session.stop()
        return resident

    def shutdown(self) -> None:
        """Stop whatever is resident. Called when the server exits."""
        if self._resident is not None:
            self.unload(self._resident.id)
