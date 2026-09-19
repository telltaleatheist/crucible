"""Voice manifests — `voices/<id>.toml` (PHASE3-TTS.md section 2).

A voice is to `tts` what a model is to `llm`, and this module is `manifests.py`
with the same strictness for the same reason: one file per Crucible voice id, the
id stable across backends, one `[voice.backends.<kind>]` block per backend the
voice can be served on, and an unknown key refused rather than warned about.

The numbers here are not this repo's inventions. They are a translation of
BookForge's `electron/data/higgs-models.json`, which is where the caps, the pace
bands and the sampling live today and where every one of them was measured. Each
manifest cites its provenance in prose the way the model manifests do, and where
a value does not exist over there this build does not make one up — it refuses.

Three things in this schema are not in PHASE3-TTS.md as written, and each is here
because the file as specified could not be filled in truthfully.

`max_chars`, not `cap_tokens`
-----------------------------
Section 2 calls the per-backend cap certificate `cap_tokens` and gives it the
catalog's 600 / 800 / 1000 / 1100. **Those numbers are CHARACTERS.** In BookForge
they are `backends.<arm>.maxChars`, the length of text a voice may be handed; the
token cap is a different quantity that narrator derives per chunk from the text it
is actually given (`HiggsBudget.cap_frames`: `int(len(text) / 15.0 * 25 * 1.8) +
100`, then clamped against the stack's context window by `sgl_served.frame_cap`).
Writing 800 into a field named `cap_tokens` would hand the engine a frame ceiling
eight times too small and cut every chunk mid-sentence while the request reported
success. So the field carries the catalog's name and the catalog's meaning, and
the token budget stays where it is computed.

`estimate_basis`
----------------
Section 2 says a voice with no measured `memory_bytes_estimate` carries no block
for that backend. Correct in principle and unusable tonight: neither of Owen's
accelerators is free, so no voice could carry a measured number and the whole job
type would be untestable. Every block therefore states where its number came
from — `"measured"` (somebody watched the card) or `"declared"` (the engine's own
configured reservation, e.g. SGLang's `--mem-fraction-static 0.6`), the latter
owing an `estimate_note` — and the basis rides on the `/v1/voices` row so nothing
downstream can mistake one for the other. The model manifests have exactly the
same problem and do **not** have this field: `models/qwen3.5-9b.toml` carries the
word MEASURED in a comment no protocol reads. That asymmetry is deliberate for
now — those numbers really were measured, and changing the model schema is not
this change — but it is the reason a reader finds provenance in two shapes.

`sampling_reason`
-----------------
Owen's standing rule (`higgs-sampling-default-no-deviation`): 0.8 / 0.95 / 50 is
*the boson default*, one engine-level number for every Higgs voice on both arms,
and a per-voice deviation needs a written reason. BookForge enforces this in
`higgsVoiceCapsForModel`; so does this loader. A backend block whose `sampling`
differs from its narrator engine's default and does not say why is refused.

Where the voices live
---------------------
`voices/` sits beside the `crucible` package in the checkout, exactly as
`models/` does, and `$CRUCIBLE_VOICES_DIR` overrides it so a test can point at a
fixture directory. If neither exists the loader refuses by name; it never falls
back to "no voices".

`load_all_voices()` reads FOUR sources now, and this file is still the owner of
what a voice MEANS in all four (PHASE21-VOICES-FROM-HF.md): a PIN, whose
`crucible-voice.toml` comes out of the weights' own repo at the pinned revision
(`crucible/voicerepo.py`, which translates it into this module's document and
hands it to the same `_parse`); the ENGINE's own base rows
(`crucible/engines/<engine>/base.toml`, section 2.6); the packaged set; and this
machine's overlay. `load_all_voices` documents the precedence and is its one
owner.

One consequence for the two blocks below. For a voice that comes out of a REPO
manifest, `memory_bytes_estimate`, `estimate_basis`, `estimate_note` and
`[voice.serving]` are not in the file at all — a manifest cannot make a claim
about a box it has never run on, and the repo schema refuses each of them by
name. They come from that server's `config.toml` `[tts.<engine>]` table
(section 2.3), and `voicerepo.merge` fills them in before `_parse` ever sees
them. The fields, the rules and the refusals here are unchanged; what moved is
who states the numbers.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import tomli_w

from .backend import CUDA_LINUX, MLX_DARWIN
from .errors import CrucibleError
from .manifests import check_table
from .weights import LOCAL, PINNED

VOICES_DIR_ENV = "CRUCIBLE_VOICES_DIR"

#: The narrator engines a voice may name, and the sampling each one renders at
#: when a voice says nothing. A deviation from these values owes a
#: `sampling_reason` — see the module docstring.
#:
#: THIS IS THE ONE LIST. `NARRATOR_ENGINES`, the `--narrator-engine` choices,
#: `/v1/capability`'s `narrator_engines` row and the task door's refusal all
#: read it, so an engine that is not here cannot be named anywhere.
#:
#: A TABLE OF ONE, ON PURPOSE. Owen's ruling of 2026-09-14: "orpheus is
#: deprecated too but hasn't been removed yet. higgs is the frontier" — "I
#: guess we can remove it now." Crucible had listed `orpheus` as a servable
#: narrator engine since PHASE3-TTS.md and would never serve it, which is the
#: same defect as any other fact with two owners: an operator page drew an
#: engine picker from this table and offered a choice with no future. It is
#: still a TABLE and not a constant because a second engine WILL come, and the
#: shape a new one has to fill is: a row here (its own sampling defaults), a
#: row in `ttsstream.STREAM_BATCH_WIDTH` (a MEASURED streaming width, never a
#: guess), a recipe per backend under `envs/tts/`, a row in
#: `jobenv.CUDA_LINUX_SERVING_STACK` if it starts a server underneath narrator,
#: and membership of `narratorvoices.DOCUMENT_READERS` if it resolves a voice
#: by name in a document. `tests/test_narrator_engine.py`'s drift guard
#: asserts the first three agree.
#:
#: higgs-v3: Owen's ruling of 2026-09-12, "Let's set temp to 0.8 across the board
#: for Higgs in Bookforge. Streaming and rendering both." Recorded with its A/B
#: in `higgs-models.json`'s `_samplingNote`.
NARRATOR_ENGINE_SAMPLING: dict[str, dict[str, float]] = {
    "higgs-v3": {"temperature": 0.8, "top_p": 0.95, "top_k": 50},
}

#: What a voice can BE. `checkpoint` is a merged fine-tune the engine is started
#: on and prompted text-only; `zeroshot` is base weights conditioned on reference
#: clips; `token` is a built-in speaker the engine already holds. The three are
#: not interchangeable and the wire never guesses between them.
VOICE_KINDS = frozenset({"checkpoint", "zeroshot", "token"})

#: The backends a voice may declare a block for. Unlike a model, a voice does not
#: name an engine per backend: `narrator_engine` is a property of the VOICE (a
#: Higgs checkpoint is a Higgs checkpoint on either card), and which stack serves
#: it on this host — SGLang-Omni or mlx-audio — is narrator's business, not the
#: manifest's.
VOICE_BACKENDS = frozenset({CUDA_LINUX, MLX_DARWIN})

#: A `clips` value meaning "this voice id exists so an operator can hand over a
#: clip that is not published yet; the job must carry them in its `inputs`".
#: PHASE3-TTS.md section 2's one named exception to clips belonging to the voice.
CLIPS_FROM_REQUEST = "from-request"

ESTIMATE_BASES = frozenset({"measured", "declared"})

#: How much a backend block's `identity` is worth, and it rides on the
#: `/v1/voices` row for `estimate_basis`'s reason: a reader must not be able to
#: mistake one for the other (PHASE18-UNCERTIFIED.md section 3.1). `PINNED` and
#: `LOCAL` — where the bytes come from — live in `crucible/weights.py`, which is
#: the module that acts on the difference.
VERIFIED = "verified"
ASSERTED = "asserted"

#: WHICH KIND OF FILE A VOICE CAME OUT OF — the `manifest` column on the
#: `/v1/voices` row (PHASE21-VOICES-FROM-HF.md sections 2.4 and 2.6).
#:
#: FOUR WORDS AND NOT THE CONTRACT'S THREE. Sections 2.4 and 2.6 name `repo`,
#: `override` and `engine`, and they are the three that survive section 8: at
#: the end of the migration every fine-tune is a pin, the two base rows are the
#: engine's, and a person's own file is an override. `packaged` is the FIFTH
#: state that exists only while section 8.1 is true — the five
#: `crucible/voices/*.toml` fine-tunes this build still ships and still prefers
#: — and it is a word rather than a silence because "this row's numbers come
#: from a file inside the install" is a real and temporary fact a reader has to
#: be able to see. Section 8.3 deletes those five files and this word with them.
MANIFEST_REPO = "repo"
MANIFEST_OVERRIDE = "override"
MANIFEST_ENGINE = "engine"
MANIFEST_PACKAGED = "packaged"

#: HOW A `[voice.pace]` BAND WAS GOT, and how a `max_chars` was. Both are
#: certificates the manifest states about its own numbers, both are REQUIRED by
#: the repo schema, and both ride on the `/v1/voices` row — see
#: `VoiceManifest.pace_basis` and `VoiceBackendSpec.max_chars_basis` for why
#: the states they name are real rather than pedantic.
PACE_BASES = frozenset({"measured", "inherited"})
MAX_CHARS_BASES = frozenset({"measured", "placeholder"})

#: `pins.toml` IS NOT A VOICE, and it lives in the directory voices are read
#: from (PHASE21 section 2.2), so the glob that finds `<id>.toml` would find it
#: and try to load a voice called `pins`. Reserved here, in the module that owns
#: what a voice id is, rather than filtered at each of the two globs.
PINS_FILE = "pins.toml"
RESERVED_VOICE_IDS = frozenset({"pins"})

_VOICE_REQUIRED: dict[str, type] = {
    "id": str,
    "display": str,
    "kind": str,
    "narrator_engine": str,
    "language": str,
    # Not in section 2's example, and required here because the `/v1/voices` row
    # carries it and a client writing FLACs cannot be handed a null sample rate.
    # It is 24000 for every voice in the catalog, which is exactly the kind of
    # coincidence that turns into a hard-coded constant if it is not written down
    # per voice.
    "sample_rate": int,
}

#: The three rates are a property of the VOICE, measured per voice by BookForge
#: from clean renders, and they travel as ONE statement: all three or none.
#: That is narrator's own rule in `engine/higgs/config.py`'s `_length_band`,
#: which refuses a partial triple by name, and this is the same rule rather
#: than a second copy of it — the band is a measured pace and the two edges
#: derived from it, so a subset is a band nobody finished writing.
#:
#: OPTIONAL AS A GROUP SINCE 2026-09-18, and the reason is the point of the
#: block. They were required, and the two voices in this build that are the
#: BASE WEIGHTS rather than a fine-tune — `higgs-default` and `zeroshot`, which
#: no ladder has ever been run on — met the requirement by copying narrator's
#: Higgs v3 defaults out of its source: `HiggsDefaults.CHARS_PER_SEC` 15.0 as
#: the pace, with `HiggsV3Defaults.MAX_CHARS_PER_SEC` 20.0 and
#: `MIN_CHARS_PER_SEC` 14.5 as the edges. That is not one fact: 15.0 is the
#: DIVISOR `cap_frames()` sizes the frame cap against and nothing was ever
#: measured speaking at it, while the edges were written around a real book
#: pace nearer 17.2. narrator keeps a band's RATIOS and re-centres them on the
#: book's running median, and those ratios are 1.333 on the short side against
#: 1.034 on the long — so after warm-up healthy chunks fell under
#: `median x 0.967`, were judged run-ons, and went re-roll -> split -> re-roll
#: to MAX_DEPTH. A manifest states what was measured; with nothing stated
#: narrator uses its own default band and derives the centre as the geometric
#: mean of the edges (`truncation.tracker_for`), and that derivation keeps its
#: one owner. Crucible does not compute a centre.
#:
#: `object` rather than `float` because TOML's 16 is an int and its 16.0 is a
#: float, and a pace that happens to land on a whole number is still a pace.
#: `_number()` does the real check and refuses a bool, which `isinstance` would
#: not.
_PACE_RATES: dict[str, type] = {
    "pace_chars_per_sec": object,
    "max_chars_per_sec": object,
    "min_chars_per_sec": object,
}
#: How the client packs to this voice, when the voice has something tighter to
#: say than its backend's `max_chars`. The catalog has both shapes and never
#: mixes them: a fine-tune declares a safe band — its training corpus's
#: interquartile range, measured 2026-09-09 — and packs between its two edges,
#: while a zero-shot voice declares a single `targetChars` the packer packs to.
#: A voice declaring neither packs to the arm's `max_chars`, which is what
#: BookForge does today and is why these are optional: section 2 lists all four
#: as required, and not one voice in the catalog declares all four.
_PACE_OPTIONAL: dict[str, type] = {
    "target_chars": int,
    "safe_min_chars": int,
    "safe_max_chars": int,
}
#: HOW THE TWO EDGES WERE GOT, stated only when it is not the usual way.
#:
#: Every ladder run in this build wrote `max = pace x 1.3` and `min = pace /
#: 1.3`, so a triple whose two ratios disagree is edges that were not derived
#: from that pace — the defect of 2026-09-18, where narrator's `CHARS_PER_SEC`
#: 15.0 sat between `HiggsV3Defaults`' 20.0/14.5 and gave 1.333 long against
#: 1.034 short. `_check_pace` therefore refuses a lopsided triple by default,
#: and this key is the manifest saying the lopsidedness is real: a band read
#: off a distribution's percentiles is lopsided because the distribution is.
#:
#: It does NOT reach the wire. Nothing downstream branches on how the edges
#: were got — narrator keeps the RATIOS whatever produced them — so this is a
#: statement to this loader and stays here, rather than a seventh `Pace` field
#: every client must learn to ignore.
_PACE_EDGES = "edges"
#: The one word `edges` takes. A closed set, so a typo is refused rather than
#: read as "not percentile, therefore check the symmetry".
_PACE_EDGES_WORDS = ("percentile",)
#: HALF THE LAST PLACE OF A MANIFEST NUMBER. Every rate in this catalog is
#: written to two decimals (`deathstalker.toml` 15.91 / 20.68 / 12.24), so a
#: stated rate stands for a real one up to 0.005 either side, and the two
#: ratios computed from three such numbers cannot be compared for exact
#: equality. The tolerance in `_check_pace` is this propagated through the two
#: divisions rather than a round number chosen to make the catalog pass.
_PACE_HALF_ULP = 0.005

#: `[voice.serving]` — WHAT THE SERVER narrator STARTS IS CONFIGURED WITH.
#:
#: REQUIRED of every voice in this build, because every voice in this build
#: names `higgs-v3` and narrator reads it on that path: `HIGGS_MAX_NUM_SEQS` is
#: `v3_served.serve_concurrency()`, which refuses BY NAME when it is unset and
#: is BOTH stage 0's admission width and the width of narrator's own batch.
#: (It was REFUSED on an `orpheus` voice until 2026-09-14, when that engine
#: left `NARRATOR_ENGINE_SAMPLING` — see the ruling there. A second engine that
#: reads no `HIGGS_*` variable brings that refusal back with it, rather than
#: inheriting a required table it configures nothing with.)
#:
#: The NOTE is required with the number for the reason `estimate_note` is: 16
#: is not an obvious value and it is CONTESTED — the deathstalker cap
#: certificate was measured at width 64 — so the next person to touch it has to
#: be able to find out where it came from without a git archaeology session.
#: (The number itself stays OFF `/v1/voices`: it is engine tuning, the server's
#: business, and a client has no decision to make with it — see
#: `crucible/jobs/tts/common.py`'s `voice_rows`.)
#:
#: WHO WRITES IT DEPENDS ON WHERE THE VOICE CAME FROM (PHASE21 section 2.3). A
#: packaged `voices/<id>.toml` and a `PUT` override state it themselves, as they
#: always have. A voice that comes out of its own repo does NOT — the repo
#: schema refuses `[voice.serving]` by name, because the width sizes the server
#: narrator starts on a particular box — and `voicerepo.merge` fills this table
#: from that machine's `config.toml` `[tts.<engine>]` before `_parse` runs. The
#: rules below are the same either way.
_SERVING_REQUIRED: dict[str, type] = {
    "max_num_seqs": int,
    "max_num_seqs_note": str,
}

#: THE SOURCE KEYS, and a block declares EXACTLY ONE of the two shapes
#: (PHASE18-UNCERTIFIED.md section 3). They are optional here and checked as a
#: pair below, because "one of these two groups" is not a thing `check_table`
#: can say.
#:
#:     hf_repo + revision    a PIN. Crucible fetches it, stamps it, and the
#:                           catalog owns the bytes.
#:     path + identity       a DIRECTORY somebody else put there. Crucible
#:                           never fetches it, never stamps it, never deletes
#:                           it, and tolerates it vanishing between jobs.
#:
#: The second shape is why a voice can exist at all while Owen's HuggingFace
#: private storage is full (HIGGS_FIELD_NOTES.md 4n.74 open item (a)) and is
#: what a screening checkpoint uses, its 8 GiB merge being scratch that is
#: deleted minutes later.
_SOURCE_KEYS: dict[str, type] = {
    "hf_repo": str,
    "revision": str,
    "path": str,
    # WHAT A LOCAL BLOCK CLAIMS TO BE, and the reason it is required of one.
    # A pin's identity is VERIFIED — the sha is what was fetched — and a
    # directory's cannot be, so this is the registrant's ASSERTION and the
    # `/v1/voices` row says so. It is what `fingerprint()` records in place of
    # a revision, so two screened checkpoints can be told apart in a client's
    # own output; without it every merge at a reused path would render as the
    # same voice.
    "identity": str,
}

_BACKEND_REQUIRED: dict[str, type] = {
    "memory_bytes_estimate": int,
    "estimate_basis": str,
    # The cap certificate for (voice, backend), in CHARACTERS. Per backend and it
    # must stay per backend: every voice's two blocks carry identical numbers
    # today, and that is a coincidence of the current catalog rather than a
    # property of the world — a cap is produced by RENDERING, and the two arms
    # sample through different implementations of top-k/top-p over different
    # runtimes.
    "max_chars": int,
    "sampling": dict,
}
_BACKEND_OPTIONAL: dict[str, type] = {
    **_SOURCE_KEYS,
    "estimate_note": str,
    "sampling_reason": str,
    # A list of clip tables, or the literal CLIPS_FROM_REQUEST. Required of a
    # zeroshot voice and refused on any other kind — a checkpoint's voice is in
    # its weights, and a token voice's is in the engine.
    "clips": object,
}

_CLIP_REQUIRED: dict[str, type] = {
    "file": str,
    "transcript": str,
    # `object` for the same reason the pace rates are: a clip that is exactly 15
    # seconds is written `15` in TOML and is still a duration. `_number()` checks
    # it, and refuses a bool.
    "seconds": object,
}

_TAKE_OPTIONAL: dict[str, type] = {"reason": str}

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_VOICE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_HF_REPO = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


class VoiceError(CrucibleError):
    """A voice manifest is missing, unreadable, or does not say what it must say."""


# ------------------------------------------------------------------ the shapes


@dataclass(frozen=True)
class ReferenceClip:
    """One reference recording and the book-exact text spoken in it."""

    file: str
    transcript: str
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "transcript": self.transcript,
            "seconds": self.seconds,
        }


@dataclass(frozen=True)
class Pace:
    """The band a client packs to, advertised so it can (section 2).

    The server states the shape; the client does the packing. At most one of
    `target_chars` and the `safe_*` pair is set; with neither, the client packs
    to the backend's `max_chars` — see `_PACE_OPTIONAL`.

    THE THREE RATES ARE ALL THREE OR ALL NONE (`_PACE_RATES`). `None` is a
    voice nobody measured, and it means exactly that rather than a default
    standing in for one: a client reading it derives nothing here, and narrator
    reaches for its engine's own band. A caller may test any one of the three
    to know which it has.
    """

    pace_chars_per_sec: float | None
    max_chars_per_sec: float | None
    min_chars_per_sec: float | None
    target_chars: int | None
    safe_min_chars: int | None
    safe_max_chars: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pace_chars_per_sec": self.pace_chars_per_sec,
            "max_chars_per_sec": self.max_chars_per_sec,
            "min_chars_per_sec": self.min_chars_per_sec,
            "target_chars": self.target_chars,
            "safe_min_chars": self.safe_min_chars,
            "safe_max_chars": self.safe_max_chars,
        }


@dataclass(frozen=True)
class VoiceBackendSpec:
    """One `[voice.backends.<kind>]` block."""

    backend: str
    #: Set together, and None on a local block. See `_SOURCE_KEYS`.
    hf_repo: str | None
    revision: str | None
    #: Set together, and None on a pinned block.
    path: str | None
    identity: str | None
    memory_bytes_estimate: int
    estimate_basis: str
    estimate_note: str | None
    max_chars: int
    sampling: dict[str, float]
    sampling_reason: str | None
    #: The clips this voice is conditioned on, `CLIPS_FROM_REQUEST`, or None for
    #: a voice that carries none.
    clips: tuple[ReferenceClip, ...] | str | None
    #: HOW `max_chars` WAS GOT — `"measured"` (a sweep was run on these weights
    #: on this arm) or `"placeholder"` (a number somebody wrote down so the arm
    #: could be served at all). PHASE21 section 2.1.
    #:
    #: `None` means THIS MANIFEST'S SCHEMA CANNOT SAY, which is a different
    #: statement from either word and is what every manifest written before the
    #: repo schema reports: `voices/*.toml` has no such key, so a value here
    #: would be this loader deciding which of the two a number was. It rides on
    #: the `/v1/voices` row for `estimate_basis`'s reason — thirdreich shipped
    #: `higgs_max_chars_mlx: 900`, a placeholder nobody measured, and a schema
    #: that cannot say so ships it as a measured fact.
    max_chars_basis: str | None = None

    @property
    def clips_from_request(self) -> bool:
        return self.clips == CLIPS_FROM_REQUEST

    @property
    def source(self) -> str:
        """`"pinned"` or `"local"`. `_parse` has already refused everything else."""
        return PINNED if self.hf_repo is not None else LOCAL

    @property
    def identity_basis(self) -> str:
        """How much the `identity` on the row is worth.

        `"verified"` for a pin — the sha is what `snapshot_download` fetched and
        what the stamp records. `"asserted"` for a path — the registrant said so
        and nothing checked. The distinction rides on the row for
        `estimate_basis`'s reason: a reader must not be able to mistake one for
        the other, and asserted identity is the honest shape for a directory
        whose bytes this server did not fetch.
        """
        return VERIFIED if self.hf_repo is not None else ASSERTED

    @property
    def weights_identity(self) -> str:
        """WHAT THIS BLOCK SAYS ITS WEIGHTS ARE — the pin's sha, or the local
        block's asserted `identity`.

        One reader for one fact. Four places want it — `fingerprint()`, the
        `/v1/voices` row, the render's provenance sidecar and the resident
        record `residency.ResidentVoice` — and a copy of `revision if revision
        is not None else identity` in each is four chances to leave one of them
        reporting `None` for a voice whose identity was stated. Two of the four
        were left reading `spec.revision` when the source axis first landed, and
        both published a `fingerprint` naming a checkpoint beside a `revision`
        of null: one record, two answers, which is the failure `identity_basis`
        exists to make impossible.

        Never None: `_check_source` has already refused a block that sets
        neither.
        """
        return self.revision if self.revision is not None else self.identity

    @property
    def local_path(self) -> Path | None:
        """The directory this block names, or None for a pin.

        `crucible/weights.py` asks every spec this through `getattr`, because
        only a VOICE can be local today: a model is a catalog subject with a
        download, an installer and a host migration behind it, and none of
        those have been designed for bytes Crucible does not own.
        """
        return None if self.path is None else Path(self.path)

    @property
    def files(self) -> tuple[str, ...]:
        """Empty: this backend fetches the WHOLE repo.

        `crucible/weights.py`'s `WeightsSource` asks every spec this, and the
        empty tuple is a real answer and not a gap — it is what "there is no
        file to choose, the repository IS the weights" reads as. Only
        `llama-windows` names files (one GGUF, and a projector beside it for a
        vision model), because a GGUF repo holds twenty quantizations and
        pulling all of them is hundreds of gigabytes.
        """
        return ()

    def to_dict(self) -> dict[str, Any]:
        clips: Any
        if self.clips is None or isinstance(self.clips, str):
            clips = self.clips
        else:
            clips = [clip.to_dict() for clip in self.clips]
        return {
            "backend": self.backend,
            "source": self.source,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "path": self.path,
            "identity": self.identity,
            "identity_basis": self.identity_basis,
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "estimate_basis": self.estimate_basis,
            "estimate_note": self.estimate_note,
            "max_chars": self.max_chars,
            "sampling": dict(self.sampling),
            "sampling_reason": self.sampling_reason,
            "clips": clips,
            "max_chars_basis": self.max_chars_basis,
        }


@dataclass(frozen=True)
class Serving:
    """`[voice.serving]` — the one number the SERVER under narrator is sized by.

    Not `[voice.pace]`'s neighbour by accident, and not its twin either: pace
    is what a CLIENT packs to and is published on `/v1/voices`; this is what
    the ENGINE admits and is never published. It is per VOICE rather than per
    backend because the value is a property of the stack narrator starts for
    Higgs v3 and the card it starts it on, and every Higgs voice on a given
    host shares both.
    """

    max_num_seqs: int
    max_num_seqs_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_num_seqs": self.max_num_seqs,
            "max_num_seqs_note": self.max_num_seqs_note,
        }


@dataclass(frozen=True)
class Take:
    """One rung of the retake ladder: what take N means for this voice.

    A job carries `take: N` and nothing else about sampling. The client decides
    *that* a row needs another take; the server decides what take N *is*
    (PHASE3-TTS.md section 3).
    """

    index: int
    #: Sampling keys this take overrides. Empty for take 0, the no-deviation
    #: default.
    overrides: dict[str, float]
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "overrides": dict(self.overrides),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class VoiceManifest:
    id: str
    display: str
    kind: str
    narrator_engine: str
    language: str
    sample_rate: int
    pace: Pace
    #: `[voice.serving]`, or None for a voice whose engine reads no `HIGGS_*`
    #: variable. Required of every `higgs-v3` voice — see `_SERVING_REQUIRED`.
    #: `higgs-v3` is the only engine in this build, so None is what the next
    #: engine will need rather than a shape any manifest has today.
    serving: Serving | None
    backends: dict[str, VoiceBackendSpec]
    takes: tuple[Take, ...]
    path: Path
    #: WHICH KIND OF FILE THIS VOICE CAME OUT OF (PHASE21 section 2.4), on the
    #: `/v1/voices` row as `manifest`. `MANIFEST_REPO` is a `crucible-voice.toml`
    #: in the weights' own repo at the pinned revision — the shape Phase 21
    #: exists to make ordinary; `MANIFEST_OVERRIDE` is a whole manifest written
    #: to this machine through `PUT /v1/voices/{id}`; `MANIFEST_ENGINE` is the
    #: engine's own base behaviour, which is not a voice anybody trains
    #: (section 2.6); `MANIFEST_PACKAGED` is a `crucible/voices/*.toml` this
    #: build still ships, and it is TRANSITIONAL — section 8.3 deletes the last
    #: five and the word goes with them.
    manifest_source: str = MANIFEST_PACKAGED
    #: HOW THE PACE BAND WAS GOT — `"measured"` or `"inherited"` — or None
    #: because this manifest's schema cannot say (`voices/*.toml` has no such
    #: key) or because there is no band. deathstalker's 16.64 survived onto
    #: weights that measured 15.91 precisely because an inherited pace is
    #: indistinguishable from a measured one at the point of use; this is the
    #: field that tells them apart, and it rides on the row.
    pace_basis: str | None = None
    #: WHERE AN INHERITED PACE CAME FROM, in prose — the run and checkpoint the
    #: number was measured on, and why it was not measured on these weights. None
    #: unless `pace_basis` is `"inherited"`.
    #:
    #: RULED 2026-09-19, and it mirrors `estimate_basis` exactly: a `declared`
    #: estimate REQUIRES its note and a `measured` one refuses it, because each
    #: basis owes its own sentence and no other. The reason it matters here is
    #: that "inherited" covers two situations a reader must be able to tell
    #: apart. An inherited pace from a SIBLING checkpoint of the same corpus is
    #: near enough — mistborn measured 13.29, 13.33 and 13.76 across three
    #: retrains. An inherited pace from a DIFFERENT corpus two versions back is
    #: the deathstalker defect: 16.64 carried from `ds_v5_prod` onto weights
    #: that measured 15.91, 4.4% fast, enough to mis-size narrator's duration
    #: guard from the first chunk. The word alone cannot separate them; the
    #: sentence can, so the sentence is required and rides on the row.
    inherited_from: str | None = None

    #: Which subtree of `~/.crucible/` this thing's weights live under. A voice id
    #: and a model id are separate namespaces and must not be able to collide on
    #: disk — see `crucible/weights.py`.
    weights_family = "voices"

    def supports(self, backend_kind: str) -> bool:
        return backend_kind in self.backends

    def spec(self, backend_kind: str) -> VoiceBackendSpec:
        """The block for `backend_kind`, or a named refusal."""
        found = self.backends.get(backend_kind)
        if found is None:
            raise VoiceError(
                f"voice {self.id!r} has no {backend_kind} block; {self.path.name} "
                f"declares {sorted(self.backends)}"
            )
        return found

    def take(self, index: int) -> Take:
        """Take `index`, or `unknown_take` by name.

        Never clamped to the last rung: a silent clamp is a retake ladder that
        stops climbing without telling anyone, and the client would keep asking
        for take 4 and keep getting take 2's draw.
        """
        if index < 0 or index >= len(self.takes):
            raise VoiceError(
                f"voice {self.id!r} has no take {index}; {self.path.name} declares "
                f"{len(self.takes)} take(s), 0 to {len(self.takes) - 1}"
            )
        return self.takes[index]

    def fingerprint(self, backend_kind: str) -> str:
        """`<id>@<identity>` — what a render records as the voice it used.

        The identity is the PIN's sha for a pinned block and the block's own
        asserted `identity` for a local one. Same shape either way, and
        deliberately: a client comparing two renders is asking "were these the
        same weights", and that question has an answer in both cases. How much
        the answer is worth is `identity_basis` on the row, not a second
        spelling here — two fingerprint formats would make every consumer
        parse before it could compare.
        """
        return f"{self.id}@{self.spec(backend_kind).weights_identity}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display": self.display,
            "kind": self.kind,
            "narrator_engine": self.narrator_engine,
            "language": self.language,
            "sample_rate": self.sample_rate,
            "pace": self.pace.to_dict(),
            "serving": None if self.serving is None else self.serving.to_dict(),
            "backends": {k: v.to_dict() for k, v in sorted(self.backends.items())},
            "takes": [take.to_dict() for take in self.takes],
            "manifest": self.manifest_source,
            "pace_basis": self.pace_basis,
            "inherited_from": self.inherited_from,
        }


# ------------------------------------------------------------------ locating


def home_voices_dir() -> Path:
    """`<CRUCIBLE_HOME>/voices` — this machine's own voices. May not exist."""
    from .config import crucible_home

    return crucible_home() / "voices"


def voices_dir_is_overridden() -> bool:
    """Is `$CRUCIBLE_VOICES_DIR` set, i.e. does it REPLACE the whole catalog?

    Asked by three readers now rather than one, so it is stated once. The
    variable's meaning has always been *"run this exact set and nothing else"*,
    and PHASE21 gives a host two more sources of a voice — the pins and the
    engine's base rows — that "nothing else" has to cover, or the escape hatch
    would quietly stop being one.
    """
    override = os.environ.get(VOICES_DIR_ENV)
    return override is not None and override != ""


def voice_dirs() -> tuple[Path, ...]:
    """Every directory voices are read from, LOWEST precedence first.

    TWO DIRECTORIES, AND THE SECOND IS WHY THIS FUNCTION EXISTS.

    Voice manifests began as package data, which made *deploying a voice* mean
    *cutting a release*: a new fine-tune could not be served until a version was
    tagged, its packs rebuilt on CI and the result installed on every machine.
    Owen, 2026-09-16: *"we dont have to cut a new release every time we deploy a
    model do we? ... i train models all the time. nearly every night."* No.

    So `<CRUCIBLE_HOME>/voices/*.toml` is read after the packaged set and WINS on
    a shared id. Drop a file in, restart the engine, and it serves — no release,
    no pack, no version. Delete the file and the packaged voice is back, which
    is what makes overriding a shipped voice safe to try.

    `CRUCIBLE_VOICES_DIR` still REPLACES both, unchanged. That is the right
    shape for "run this exact set and nothing else" and the wrong shape for
    "add one", which is the mistake this overlay corrects: with only the
    override, adding a single voice meant copying all seven shipped manifests
    into a directory and maintaining the set by hand forever.
    """
    override = os.environ.get(VOICES_DIR_ENV)
    if override is not None and override != "":
        return (voices_dir(),)
    home = home_voices_dir()
    packaged = voices_dir()
    return (packaged, home) if home.is_dir() and home != packaged else (packaged,)


def engine_voices_dir() -> Path:
    """`crucible/engines/` — where an ENGINE's own base rows live.

    ONE PLACE, ON PURPOSE (PHASE21 section 2.6, ruling 1 still Owen's).
    `higgs-default` and `zeroshot` sit on `bosonai/higgs-tts-3-4b`, which is not
    ours and cannot carry a `crucible-voice.toml`; they are not voices anybody
    trains but the engine's own base behaviour — the token default narrator
    renders with on the mlx arm, and "clips from the request". So they stay
    packaged, and they stay packaged HERE rather than among the voices, so that
    "Crucible ships no voices" is exactly true of voices.

    If Owen takes the alternative — a manifest-only repo of ours pointing at
    Boson's weights — this function and `engine_voices_path` are the whole of
    what moves.
    """
    return Path(__file__).resolve().parent / "engines"


def engine_voices_path(narrator_engine: str) -> Path:
    """The `base.toml` for one narrator engine. May not exist."""
    return engine_voices_dir() / narrator_engine / "base.toml"


def voices_dir() -> Path:
    """The PACKAGED voices, the set every install ships. Refuses if absent.

    `CRUCIBLE_VOICES_DIR` replaces it wholesale. For the ordinary read — shipped
    plus this machine's own — callers want `voice_dirs()`.
    """
    override = os.environ.get(VOICES_DIR_ENV)
    if override is not None and override != "":
        path = Path(override).expanduser()
        if not path.is_dir():
            raise VoiceError(f"{VOICES_DIR_ENV}={override!r} is not a directory")
        return path
    # voices/ sits beside the crucible package in the checkout.
    path = Path(__file__).resolve().parent / "voices"
    if not path.is_dir():
        raise VoiceError(
            f"no voice manifests at {path}; they are package data and this "
            f"install has lost them, or ${VOICES_DIR_ENV} must point at them"
        )
    return path


# ------------------------------------------------------------------ checking


def _number(where: str, key: str, value: Any) -> float:
    """A TOML int or float as a float. A bool is not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VoiceError(
            f"{where}: {key} must be a number, got {type(value).__name__}"
        )
    return float(value)


def _check_pace(where: str, table: dict[str, Any]) -> Pace:
    check_table(
        where,
        table,
        {},
        {**_PACE_RATES, **_PACE_OPTIONAL, _PACE_EDGES: str},
        error=VoiceError,
    )
    edges = table.get(_PACE_EDGES)
    if edges is not None and edges not in _PACE_EDGES_WORDS:
        raise VoiceError(
            f"{where}: {_PACE_EDGES} {edges!r} is not one of "
            f"{sorted(_PACE_EDGES_WORDS)}; the key says how the two edges were "
            "got, and a word this loader does not know would silently read as "
            "'derived from the pace'"
        )
    # ALL THREE OR NONE, refused by name on the subset — narrator's `_length_band`
    # refuses the same subset with the same sentence, and a manifest that got past
    # this door would only be refused later, at the engine, on somebody's book.
    stated = set(_PACE_RATES) & set(table)
    if stated and stated != set(_PACE_RATES):
        raise VoiceError(
            f"{where}: declares only part of its rate band, missing "
            f"{sorted(set(_PACE_RATES) - stated)}. The band is a measured pace "
            "and the two edges derived from it; write all three or none"
        )
    rates: dict[str, float | None] = dict.fromkeys(_PACE_RATES)
    if stated:
        rates = {key: _number(where, key, table[key]) for key in _PACE_RATES}
        for key, value in rates.items():
            if value <= 0:
                raise VoiceError(f"{where}: {key} must be positive, got {value}")
        # min < pace < max, narrator's own rule (`engine/higgs/config.py`
        # `_length_band`): the band is the measured pace and the two edges DERIVED
        # from it, so a pace outside its own edges is a band nobody finished
        # writing. narrator keeps only the band's RATIOS and re-centres them on the
        # running median of the book's own shipped takes, which it cannot do
        # without knowing what the edges were centred on.
        if not (
            rates["min_chars_per_sec"]
            < rates["pace_chars_per_sec"]
            < rates["max_chars_per_sec"]
        ):
            raise VoiceError(
                f"{where}: min_chars_per_sec {rates['min_chars_per_sec']}, "
                f"pace_chars_per_sec {rates['pace_chars_per_sec']}, "
                f"max_chars_per_sec {rates['max_chars_per_sec']} are out of order; "
                "the band is min < pace < max"
            )

        # THE TWO EDGES ARE DERIVED FROM THE PACE, so the band is symmetric in
        # ratio — every ladder run in this build wrote `max = pace x 1.3` and
        # `min = pace / 1.3`, and the five fine-tunes here all measure 1.30 on
        # both sides. A triple whose ratios disagree is edges that came from
        # somewhere else: the 15.0 / 20.0 / 14.5 that shipped until 2026-09-18
        # passed `min < pace < max` and was still a splice of two different
        # centres, 1.333 long against 1.034 short. narrator keeps only the
        # RATIOS (`engine/higgs/truncation.PaceTracker`), so a lopsided pair
        # re-centred on the book's running median judged healthy chunks run-ons
        # and re-rolled them to MAX_DEPTH — a band nobody can read as a band.
        #
        # The tolerance is the rounding, not a fudge: each rate is written to
        # two decimals, so it stands for a real number within `_PACE_HALF_ULP`,
        # and that uncertainty propagates through each division as
        # `half_ulp x (1 + ratio) / divisor` — the divisor's own rounding
        # scaled by the ratio, plus the numerator's. Nothing wider.
        long_side = rates["max_chars_per_sec"] / rates["pace_chars_per_sec"]
        short_side = rates["pace_chars_per_sec"] / rates["min_chars_per_sec"]
        rounding = _PACE_HALF_ULP * (1 + long_side) / rates[
            "pace_chars_per_sec"
        ] + _PACE_HALF_ULP * (1 + short_side) / rates["min_chars_per_sec"]
        if edges is None and abs(long_side - short_side) > rounding:
            raise VoiceError(
                f"{where}: the band is not symmetric — max_chars_per_sec is "
                f"{long_side:.3f} x pace_chars_per_sec but pace_chars_per_sec "
                f"is only {short_side:.3f} x min_chars_per_sec, further apart "
                f"than two-decimal rounding allows ({rounding:.4f}). The two "
                "edges are derived from the measured pace, so both ratios are "
                "the same number; a band whose edges came off a distribution "
                f'instead says so with {_PACE_EDGES} = "percentile"'
            )
    elif edges is not None:
        # An `edges` with no edges to describe. It is the leftover of a triple
        # somebody deleted, and left alone it reads as a band this loader
        # checked and passed.
        raise VoiceError(
            f"{where}: states {_PACE_EDGES} = {edges!r} but states no rate "
            "band for it to describe; the key says how max_chars_per_sec and "
            "min_chars_per_sec were got, and there are none"
        )

    target = table.get("target_chars")
    floor = table.get("safe_min_chars")
    ceiling = table.get("safe_max_chars")
    band = floor is not None or ceiling is not None
    if target is not None and band:
        raise VoiceError(
            f"{where}: declares both target_chars and a safe band. A voice packs "
            "one way or the other — a fine-tune to its measured band, a zero-shot "
            "voice to a single target — and two answers would let the packer pick"
        )
    if band and (floor is None or ceiling is None):
        missing = "safe_min_chars" if floor is None else "safe_max_chars"
        raise VoiceError(
            f"{where}: a safe band needs both edges and is missing {missing}"
        )
    for key, value in (
        ("target_chars", target),
        ("safe_min_chars", floor),
        ("safe_max_chars", ceiling),
    ):
        if value is not None and value <= 0:
            raise VoiceError(f"{where}: {key} must be positive, got {value}")
    if band and floor >= ceiling:
        raise VoiceError(
            f"{where}: safe_min_chars {floor} is not below safe_max_chars "
            f"{ceiling}; the floor is what lets two short paragraphs merge, and a "
            "floor at the ceiling is how a 400-character chunk ships alone"
        )
    return Pace(
        pace_chars_per_sec=rates["pace_chars_per_sec"],
        max_chars_per_sec=rates["max_chars_per_sec"],
        min_chars_per_sec=rates["min_chars_per_sec"],
        target_chars=target,
        safe_min_chars=floor,
        safe_max_chars=ceiling,
    )


@dataclass(frozen=True)
class _Source:
    """The four source fields after `_check_source` has settled which shape a
    block is. Exactly one pair is set."""

    hf_repo: str | None
    revision: str | None
    path: str | None
    identity: str | None


def _blank(value: Any) -> bool:
    """True for a key that is absent or is whitespace.

    An EMPTY STRING IS NOT A DECLARATION. `path = ""` reads as "this block
    declares a path" to `in`, and would then be refused for not being
    absolute — a confusing second-order message about a block that really
    declared no source at all. Treated as absent so the refusal names the
    actual problem.
    """
    return value is None or (isinstance(value, str) and value.strip() == "")


def _is_absolute(value: str) -> bool:
    """Absolute in EITHER flavour, and that is deliberate.

    A `cuda-linux` block names a POSIX path and an `mlx-darwin` block names
    one too, but the loader that reads them may be running on Windows — a
    test, `crucible voices show`, or an operator checking a manifest before
    sending it to the machine that will serve it. `Path('/home/x')
    .is_absolute()` is FALSE on Windows (no drive letter), so asking the host
    would refuse a perfectly good Linux manifest for being on the wrong
    machine.
    """
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _check_source(where: str, block: dict[str, Any]) -> _Source:
    """Which of `_SOURCE_KEYS`' two shapes this block is, or a named refusal.

    EXACTLY ONE, and both halves of the refusal are named rather than sharing
    a "bad source" message: a block with both has two answers to "where are
    these weights" and a block with neither has none, and they are different
    mistakes to have made.
    """
    pinned = not _blank(block.get("hf_repo"))
    local = not _blank(block.get("path"))

    if pinned and local:
        raise VoiceError(
            f"{where}: declares both hf_repo {block['hf_repo']!r} and path "
            f"{block['path']!r}. A backend block names ONE source — a pin Crucible "
            "fetches and owns, or a directory somebody else put there and still "
            "owns — and a block with two would let the loader pick which weights "
            "the voice is"
        )
    if not pinned and not local:
        raise VoiceError(
            f"{where}: names no weights. A backend block declares either "
            "hf_repo + revision (a pin) or path + identity (a directory on the "
            "machine that serves it); see PHASE18-UNCERTIFIED.md section 3"
        )

    if pinned:
        for key in ("path", "identity"):
            if not _blank(block.get(key)):
                raise VoiceError(
                    f"{where}: is a pinned block and also carries {key}. A pin's "
                    "identity is its revision, which is VERIFIED — the sha is what "
                    f"was fetched — so a second {key} beside it would be a fact with "
                    "two owners"
                )
        if not _HF_REPO.match(block["hf_repo"]):
            raise VoiceError(
                f"{where}: hf_repo {block['hf_repo']!r} is not an <owner>/<name> "
                "HuggingFace repo id"
            )
        if _blank(block.get("revision")):
            raise VoiceError(
                f"{where}: declares hf_repo {block['hf_repo']!r} and no revision. A "
                "pin is a repo AND a commit; `PUT /v1/voices/{id}` is the door that "
                "may omit one, and it resolves it before the manifest is written"
            )
        if not _REVISION.match(block["revision"]):
            raise VoiceError(
                f"{where}: revision {block['revision']!r} must be a full 40-character "
                "commit sha, so a pull is reproducible; branch names are not pins"
            )
        return _Source(
            hf_repo=block["hf_repo"],
            revision=block["revision"],
            path=None,
            identity=None,
        )

    for key in ("hf_repo", "revision"):
        if not _blank(block.get(key)):
            raise VoiceError(
                f"{where}: is a local block and also carries {key}. Crucible does not "
                "fetch these bytes, does not stamp them and cannot check them against "
                f"a pin, so a {key} here would describe a download that never happens"
            )
    if not _is_absolute(block["path"]):
        raise VoiceError(
            f"{where}: path {block['path']!r} is not absolute. The SERVER resolves it, "
            "so a relative path would resolve against whatever directory that process "
            "happens to have been started in"
        )
    if _blank(block.get("identity")):
        raise VoiceError(
            f"{where}: declares path {block['path']!r} and no identity. A directory "
            "cannot say what weights it holds, so the registrant states it and the "
            "row marks it ASSERTED. Without one, every checkpoint served from a "
            "reused path would render under the same fingerprint and no client could "
            "tell two of them apart"
        )
    return _Source(
        hf_repo=None,
        revision=None,
        path=block["path"],
        identity=block["identity"],
    )


def _check_sampling(
    where: str, block: dict[str, Any], narrator_engine: str
) -> tuple[dict[str, float], str | None]:
    """The block's sampling, and its reason when it deviates from the engine's."""
    default = NARRATOR_ENGINE_SAMPLING[narrator_engine]
    table = block["sampling"]
    # `sampling` REPLACES the engine's table rather than merging into it: narrator
    # warns about exactly this, because a block stating only `repetition_penalty`
    # would otherwise leave the rest to be filled in by something, and on SGLang
    # an unfilled top_k samples the untruncated 1026-way codebook tail (measured:
    # one chunk running to the cap with 80 s of silence). So every key the engine
    # takes must be present, and no key it does not take may be.
    check_table(
        f"{where} sampling",
        table,
        {key: object for key in default},
        {},
        error=VoiceError,
    )
    values = {
        key: _number(f"{where} sampling", key, table[key]) for key in default
    }
    deviations = sorted(
        f"{key} {values[key]} (engine default {default[key]})"
        for key in default
        if values[key] != default[key]
    )
    reason = block.get("sampling_reason")
    if deviations and (reason is None or reason.strip() == ""):
        raise VoiceError(
            f"{where}: sampling deviates from the {narrator_engine} default — "
            + "; ".join(deviations)
            + " — and carries no sampling_reason. One engine-level number is the "
            "rule (Owen, 2026-09-06: 'we shouldnt deviate from the default unless "
            "we have a very good reason'); a deviation owes that reason in writing"
        )
    if not deviations and reason is not None:
        raise VoiceError(
            f"{where}: carries a sampling_reason but its sampling is the "
            f"{narrator_engine} default. A reason with nothing to explain is a "
            "reason a reader will trust the next time the numbers do differ"
        )
    return values, reason


def _check_clips(where: str, block: dict[str, Any], kind: str) -> Any:
    declared = block.get("clips")
    if kind != "zeroshot":
        if declared is not None:
            raise VoiceError(
                f"{where}: a {kind} voice declares clips. A checkpoint's voice is "
                "in its weights and a token voice's is in the engine; reference "
                "clips would be conditioning nothing reads"
            )
        return None
    if declared is None:
        raise VoiceError(
            f"{where}: a zeroshot voice must declare its reference clips, or the "
            f"literal {CLIPS_FROM_REQUEST!r} if the job is to carry them"
        )
    if isinstance(declared, str):
        if declared != CLIPS_FROM_REQUEST:
            raise VoiceError(
                f"{where}: clips is the string {declared!r}; the only string it may "
                f"be is {CLIPS_FROM_REQUEST!r}"
            )
        return CLIPS_FROM_REQUEST
    if not isinstance(declared, list) or not declared:
        raise VoiceError(
            f"{where}: clips must be a non-empty list of "
            "{file, transcript, seconds} tables, or the literal "
            f"{CLIPS_FROM_REQUEST!r}"
        )
    clips: list[ReferenceClip] = []
    for index, entry in enumerate(declared):
        at = f"{where} clips[{index}]"
        if not isinstance(entry, dict):
            raise VoiceError(f"{at}: must be a table, got {type(entry).__name__}")
        check_table(at, entry, _CLIP_REQUIRED, {}, error=VoiceError)
        seconds = _number(at, "seconds", entry["seconds"])
        if seconds <= 0:
            raise VoiceError(
                f"{at}: seconds must be positive, got {seconds}. narrator reads a "
                "clip's declared duration rather than opening the file "
                "(v3_served.reference_seconds), so a missing one is a render that "
                "dies after the server has already spent five minutes coming up"
            )
        if entry["transcript"].strip() == "":
            # narrator refuses this at construction too
            # (`narrator/engine/protocol.py`, ReferenceClip), and for the reason
            # written there: the transcript is the BOOK-EXACT text the clip was cut
            # from and never an ASR guess, because a zero-shot clone conditioned on
            # a wrong or absent transcript is a whole book in a subtly wrong voice,
            # reported as success.
            raise VoiceError(
                f"{at}: has no transcript. A reference clip is only usable with the "
                "book-exact text spoken in it — the corpus row, or the narration "
                "copy after the narration-text pass — never a transcription, and "
                "never nothing"
            )
        clips.append(
            ReferenceClip(
                file=entry["file"], transcript=entry["transcript"], seconds=seconds
            )
        )
    return tuple(clips)


def _check_serving(
    path: Path, voice: dict[str, Any], narrator_engine: str
) -> Serving | None:
    """`[voice.serving]`: required for `higgs-v3`, refused for any other engine.

    THE NUMBER narrator REFUSES TO RENDER WITHOUT. `HIGGS_MAX_NUM_SEQS` is
    stage 0's `max_num_seqs` on the vllm-omni stack AND the width of narrator's
    own batch (`v3_served.serve_concurrency()`, which raises by name when it is
    unset — "a guessed width is either a server idling at 1 or a queue the
    render never asked for"). Crucible states it from here.

    REFUSED ON AN ENGINE THAT READS NO `HIGGS_*` VARIABLE rather than ignored,
    because a number in that manifest would be a lever that reports success —
    the exact shape of the defect the whole BookForge serving block was until
    2026-09-05, when it declared a configuration nothing applied. `higgs-v3` is
    the only engine `NARRATOR_ENGINE_SAMPLING` names today, so this refusal is
    the rule a second engine arrives into rather than one any manifest trips.
    """
    where = f"{path.name} [voice.serving]"
    block = voice.get("serving")
    if narrator_engine != "higgs-v3":
        if block is not None:
            raise VoiceError(
                f"{where}: narrator_engine is {narrator_engine!r}, which reads no "
                "HIGGS_* variable, so a [voice.serving] table here configures "
                "nothing. Delete it rather than leaving a lever that reports "
                "success"
            )
        return None
    if block is None:
        raise VoiceError(
            f"{path.name}: a higgs-v3 voice needs a [voice.serving] table with "
            "max_num_seqs and max_num_seqs_note. narrator refuses to render "
            "without HIGGS_MAX_NUM_SEQS (v3_served.serve_concurrency): it is the "
            "server's admission width AND the width of narrator's own batch, and "
            "there is no default"
        )
    if not isinstance(block, dict):
        raise VoiceError(f"{where}: must be a table")
    check_table(where, block, _SERVING_REQUIRED, {}, error=VoiceError)
    if block["max_num_seqs"] < 1:
        raise VoiceError(
            f"{where}: max_num_seqs must be at least 1, got "
            f"{block['max_num_seqs']}"
        )
    if block["max_num_seqs_note"].strip() == "":
        raise VoiceError(
            f"{where}: max_num_seqs carries no note. The number is contested — "
            "the deathstalker cap certificate was measured at 64 while the "
            "shipped width is 16 — so a reader of a /v1/voices row has to be "
            "able to find out where it came from"
        )
    return Serving(
        max_num_seqs=block["max_num_seqs"],
        max_num_seqs_note=block["max_num_seqs_note"],
    )


def _check_takes(
    path: Path, document: dict[str, Any], narrator_engine: str
) -> tuple[Take, ...]:
    """The retake ladder. Take 0 exists whether or not the file declares it."""
    declared = document.get("takes")
    default = NARRATOR_ENGINE_SAMPLING[narrator_engine]
    if declared is None:
        return (Take(index=0, overrides={}, reason=None),)
    if not isinstance(declared, list) or not declared:
        raise VoiceError(
            f"{path.name}: [[voice.takes]] must be a non-empty list of tables; a "
            "voice with no ladder simply omits it and gets take 0"
        )
    takes: list[Take] = []
    for index, entry in enumerate(declared):
        at = f"{path.name} [[voice.takes]][{index}]"
        if not isinstance(entry, dict):
            raise VoiceError(f"{at}: must be a table, got {type(entry).__name__}")
        check_table(
            at,
            entry,
            {},
            {**{key: object for key in default}, **_TAKE_OPTIONAL},
            error=VoiceError,
        )
        overrides = {
            key: _number(at, key, entry[key]) for key in default if key in entry
        }
        reason = entry.get("reason")
        if index == 0 and overrides:
            raise VoiceError(
                f"{at}: take 0 is the engine default and may not deviate. It is the "
                "draw every render starts from; a ladder whose first rung is "
                "already a deviation has no baseline to climb from"
            )
        if index > 0 and not overrides:
            raise VoiceError(
                f"{at}: take {index} changes nothing. A rung that is the same "
                "sampling as the one below it is a different DRAW, which is what "
                "a re-roll is for — say so with a reason and a value, or drop it"
            )
        if overrides and (reason is None or reason.strip() == ""):
            raise VoiceError(
                f"{at}: deviates from the {narrator_engine} default "
                + "("
                + ", ".join(f"{k} {v}" for k, v in sorted(overrides.items()))
                + ") and carries no reason. The ladder's steps are server config and "
                "each one owes the measurement that chose it"
            )
        takes.append(Take(index=index, overrides=overrides, reason=reason))
    return tuple(takes)


def _parse(document: dict[str, Any], path: Path, expected_id: str) -> VoiceManifest:
    unknown = sorted(set(document) - {"voice"})
    if unknown:
        raise VoiceError(
            f"{path.name}: unknown top-level table(s) {unknown}; a voice manifest "
            "has exactly [voice], and everything else hangs off it"
        )
    if "voice" not in document:
        raise VoiceError(f"{path.name}: missing the [voice] table")
    voice = document["voice"]
    if not isinstance(voice, dict):
        raise VoiceError(f"{path.name}: [voice] must be a table")

    scalars = {
        key: value for key, value in voice.items()
        if key not in ("pace", "serving", "backends", "takes")
    }
    check_table(f"{path.name} [voice]", scalars, _VOICE_REQUIRED, {}, error=VoiceError)

    voice_id = voice["id"]
    if not _VOICE_ID.match(voice_id):
        raise VoiceError(
            f"{path.name}: voice.id {voice_id!r} must be lower-case and start with "
            "a letter or digit ([a-z0-9][a-z0-9._-]*)"
        )
    if voice_id != expected_id:
        raise VoiceError(
            f"{path.name}: voice.id is {voice_id!r} but the file is named "
            f"{expected_id!r}; the id and the filename are the same thing"
        )
    if voice["kind"] not in VOICE_KINDS:
        raise VoiceError(
            f"{path.name}: voice.kind {voice['kind']!r} is not a voice kind; the "
            f"kinds are {sorted(VOICE_KINDS)}"
        )
    narrator_engine = voice["narrator_engine"]
    if narrator_engine not in NARRATOR_ENGINE_SAMPLING:
        raise VoiceError(
            f"{path.name}: voice.narrator_engine {narrator_engine!r} is not one of "
            f"narrator's engines; they are {sorted(NARRATOR_ENGINE_SAMPLING)}"
        )
    if voice["language"].strip() == "":
        raise VoiceError(f"{path.name}: voice.language must not be empty")
    if voice["sample_rate"] <= 0:
        raise VoiceError(
            f"{path.name}: voice.sample_rate must be positive, got "
            f"{voice['sample_rate']}"
        )

    if "pace" not in voice:
        raise VoiceError(f"{path.name}: missing the [voice.pace] table")
    if not isinstance(voice["pace"], dict):
        raise VoiceError(f"{path.name}: [voice.pace] must be a table")
    pace = _check_pace(f"{path.name} [voice.pace]", voice["pace"])
    serving = _check_serving(path, voice, narrator_engine)

    if "backends" not in voice:
        raise VoiceError(f"{path.name}: missing every [voice.backends.<kind>] table")
    backends_table = voice["backends"]
    if not isinstance(backends_table, dict):
        raise VoiceError(
            f"{path.name}: [voice.backends] must hold one table per backend"
        )
    if not backends_table:
        raise VoiceError(
            f"{path.name}: no backend blocks; a voice nothing can serve is not a voice"
        )

    backends: dict[str, VoiceBackendSpec] = {}
    for kind, block in backends_table.items():
        where = f"{path.name} [voice.backends.{kind}]"
        if kind not in VOICE_BACKENDS:
            raise VoiceError(
                f"{where}: {kind!r} is not a Crucible backend; the backends are "
                f"{sorted(VOICE_BACKENDS)}"
            )
        if not isinstance(block, dict):
            raise VoiceError(f"{where}: must be a table")
        check_table(
            where, block, _BACKEND_REQUIRED, _BACKEND_OPTIONAL, error=VoiceError
        )

        source = _check_source(where, block)
        if block["memory_bytes_estimate"] <= 0:
            raise VoiceError(
                f"{where}: memory_bytes_estimate must be positive, got "
                f"{block['memory_bytes_estimate']}"
            )
        basis = block["estimate_basis"]
        if basis not in ESTIMATE_BASES:
            raise VoiceError(
                f"{where}: estimate_basis {basis!r} is not one of "
                f"{sorted(ESTIMATE_BASES)}"
            )
        note = block.get("estimate_note")
        if basis == "declared" and (note is None or note.strip() == ""):
            raise VoiceError(
                f"{where}: estimate_basis is 'declared' and there is no "
                "estimate_note. A declared number came from somewhere — an engine's "
                "configured reservation, a sibling voice's measurement — and the "
                "reader of a `/v1/voices` row has to be able to find out where"
            )
        if basis == "measured" and note is not None:
            raise VoiceError(
                f"{where}: estimate_basis is 'measured' and it also carries an "
                "estimate_note. Put the measurement in a comment beside the number, "
                "the way the model manifests do; estimate_note is what a DECLARED "
                "number owes, and a row carrying one for a measured number would "
                "read as an excuse"
            )
        if block["max_chars"] <= 0:
            raise VoiceError(
                f"{where}: max_chars must be positive, got {block['max_chars']}"
            )
        if pace.safe_max_chars is not None and pace.safe_max_chars > block["max_chars"]:
            raise VoiceError(
                f"{where}: this backend caps the voice at {block['max_chars']} "
                f"characters, but [voice.pace] packs up to safe_max_chars "
                f"{pace.safe_max_chars}. The band may never exceed the arm's cap — "
                "the same rule BookForge and narrator both refuse on"
            )
        if pace.target_chars is not None and pace.target_chars > block["max_chars"]:
            raise VoiceError(
                f"{where}: this backend caps the voice at {block['max_chars']} "
                f"characters, but [voice.pace] packs to target_chars "
                f"{pace.target_chars}"
            )
        if not isinstance(block["sampling"], dict):
            raise VoiceError(f"{where}: sampling must be a table")
        sampling, reason = _check_sampling(where, block, narrator_engine)
        clips = _check_clips(where, block, voice["kind"])

        backends[kind] = VoiceBackendSpec(
            backend=kind,
            hf_repo=source.hf_repo,
            revision=source.revision,
            path=source.path,
            identity=source.identity,
            memory_bytes_estimate=block["memory_bytes_estimate"],
            estimate_basis=basis,
            estimate_note=note,
            max_chars=block["max_chars"],
            sampling=sampling,
            sampling_reason=reason,
            clips=clips,
        )

    takes = _check_takes(path, voice, narrator_engine)

    return VoiceManifest(
        id=voice_id,
        display=voice["display"],
        kind=voice["kind"],
        narrator_engine=narrator_engine,
        language=voice["language"],
        sample_rate=voice["sample_rate"],
        pace=pace,
        serving=serving,
        backends=backends,
        takes=takes,
        path=path,
    )


# ------------------------------------------------------------------- loading


def parse_voice(text: str, path: Path, expected_id: str) -> VoiceManifest:
    """Parse and validate one voice manifest's text. Raises VoiceError by name."""
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise VoiceError(f"{path.name}: not valid TOML: {exc}") from exc
    return _parse(document, path, expected_id)


def load_voice(voice_id: str, directory: Path | None = None) -> VoiceManifest:
    """The manifest for `voice_id`, from whichever source this host has for it.

    ONE PRECEDENCE, ONE OWNER. With no `directory` this is a lookup in
    `load_all_voices()` rather than a second search of its own: the two used to
    walk `voice_dirs()` from opposite ends and agreed only because there were
    two directories, and PHASE21 makes four sources of a voice (a pin, the
    engine's base rows, the packaged set, this machine's overlay). A second
    hand-written order over four sources is two answers to "which manifest is
    this voice", which is the whole of ARCHITECTURE.md section 1.

    `directory` still reads exactly that directory and nothing else, because
    that is what `_voices_in` and `--voices-dir` mean.
    """
    if directory is not None:
        return _load_voice_file(directory / f"{voice_id}.toml", voice_id)
    served = load_all_voices()
    found = served.get(voice_id)
    if found is None:
        where = ", ".join(str(r) for r in voice_dirs())
        raise VoiceError(
            f"no manifest for voice {voice_id!r} in {where}; this host serves "
            f"{sorted(served)}"
        )
    return found


def _load_voice_file(path: Path, voice_id: str) -> VoiceManifest:
    """One `<id>.toml` off the disk, refused by name if it is not there."""
    if not path.is_file():
        known = (
            sorted(p.stem for p in path.parent.glob("*.toml"))
            if path.parent.is_dir()
            else []
        )
        raise VoiceError(
            f"no manifest for voice {voice_id!r} at {path}; that directory holds "
            f"{known}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VoiceError(f"could not read {path}: {exc}") from exc
    return parse_voice(text, path, voice_id)


def load_all_voices(directory: Path | None = None) -> dict[str, VoiceManifest]:
    """Every voice this host serves, by id, in id order.

    FOUR SOURCES, LOWEST PRECEDENCE FIRST, and the order is the whole of what
    PHASE21 section 8.1 asks for -- *"the loader accepts pins; the five packaged
    manifests STILL ship and still win"*:

        1. the PINS (`crucible/voices/pins.toml` + `<home>/voices/pins.toml`,
           home winning per id) -- each one a `crucible-voice.toml` read out of
           the weights' own repo at the pinned revision;
        2. the ENGINE's own base rows (`crucible/engines/<engine>/base.toml`,
           section 2.6);
        3. the PACKAGED voices (`crucible/voices/*.toml`) -- the five fine-tunes
           this build still ships, which section 8.3 deletes;
        4. this machine's OVERLAY (`<CRUCIBLE_HOME>/voices/*.toml`), which is
           what `PUT /v1/voices/{id}` with a `voice` body writes.

    So a packaged manifest beats a pin for the same id while both exist, which
    is what makes section 8's order safe: adding the pins regresses nothing, and
    deleting the packaged files is the step that hands the id over.

    THE PINS ARE LOADED EVEN WHERE THEY ARE SHADOWED. Skipping a shadowed pin
    would save a file read and hide a broken one until the day the packaged
    manifest went away, which is exactly the failure section 8's order exists to
    prevent.

    Passing `directory` reads exactly that one -- no pins, no engine rows --
    which is what the tests and `--voices-dir` mean by it.
    """
    if directory is not None:
        return dict(sorted(_voices_in(directory).items()))
    from . import voicerepo

    voices: dict[str, VoiceManifest] = {}
    voices.update(voicerepo.pinned_voices())
    if not voices_dir_is_overridden():
        # THE ENGINE'S BASE ROWS ARE PART OF THE INSTALL, so `CRUCIBLE_VOICES_DIR`
        # replaces them along with everything else: the variable means "run this
        # exact set", and a catalog that still carried two rows the caller did not
        # put in that directory would not be that set.
        voices.update(_engine_voices())
    for root in voice_dirs():
        voices.update(_voices_in(root))
    # Re-sorted because the merge is by SOURCE and the ORDER is by id: a home
    # voice inserted in the middle of the shipped set must list in the middle,
    # not at the end. `/v1/voices` lists in this order and it is documented.
    return {vid: voices[vid] for vid in sorted(voices)}


def _engine_voices() -> dict[str, VoiceManifest]:
    """The base rows every narrator engine in this build declares.

    Keyed by id out of `crucible/engines/<engine>/base.toml`'s `[voices.<id>]`
    tables, each of which is exactly the `[voice]` table a `voices/*.toml`
    holds -- the SAME `_parse`, so the base rows are held to every rule a voice
    is and cannot drift into a schema of their own.

    An engine with no such file contributes nothing and is not an error: a
    second narrator engine will arrive before its base rows do.
    """
    found: dict[str, VoiceManifest] = {}
    for engine in sorted(NARRATOR_ENGINE_SAMPLING):
        path = engine_voices_path(engine)
        if not path.is_file():
            continue
        try:
            document = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise VoiceError(f"{path}: not valid TOML: {exc}") from exc
        unknown = sorted(set(document) - {"voices"})
        if unknown:
            raise VoiceError(
                f"{path.name}: unknown top-level table(s) {unknown}; an engine's "
                "base rows are exactly [voices.<id>], one table per row"
            )
        table = document.get("voices")
        if not isinstance(table, dict) or not table:
            raise VoiceError(
                f"{path.name}: declares no [voices.<id>] table. A base file with "
                "no rows in it is a file nothing reads; delete it instead"
            )
        for voice_id in sorted(table):
            block = table[voice_id]
            if not isinstance(block, dict):
                raise VoiceError(f"{path.name}: [voices.{voice_id}] must be a table")
            manifest = _parse({"voice": block}, path, voice_id)
            if manifest.narrator_engine != engine:
                raise VoiceError(
                    f"{path.name}: [voices.{voice_id}] names narrator_engine "
                    f"{manifest.narrator_engine!r} but sits under {engine!r}. A "
                    "base row is the engine's own behaviour, so the directory it "
                    "is in and the engine it names are one fact"
                )
            found[voice_id] = replace(manifest, manifest_source=MANIFEST_ENGINE)
    return found


#: The id a caller may write. Same shape a file stem has to have, checked here
#: because a request is not a filename until this says so: an id with a slash or
#: a `..` in it is a path, and a path is how a write to `voices/` becomes a write
#: to anywhere.
_VOICE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def home_voice_path(voice_id: str) -> Path:
    """Where a voice this machine owns is written. NOT where one is read from.

    Reading searches both directories (`voice_dirs`); writing has exactly one
    destination, because "which of the two did that land in" is not a question
    anybody should have to ask about their own machine.
    """
    if not _VOICE_ID.match(voice_id):
        raise VoiceError(
            f"voice id {voice_id!r} is not usable as a manifest name: lower-case "
            "letters, digits, dot, dash and underscore, starting with a letter or "
            "digit, at most 64 characters. The id becomes a FILENAME, so anything "
            "else is a path rather than a name"
        )
    if voice_id in RESERVED_VOICE_IDS:
        raise VoiceError(
            f"voice id {voice_id!r} is reserved: {PINS_FILE} in this directory is "
            "this machine's pin list (PHASE21 section 2.2), so a voice of that "
            "name would be written over it"
        )
    return home_voices_dir() / f"{voice_id}.toml"


def write_home_voice(voice_id: str, document: dict[str, Any]) -> VoiceManifest:
    """Validate a voice document and store it in this machine's own overlay.

    ── Why this is here and not in the API layer ───────────────────────────────

    `voices/*.toml` has one owner, and it is this module: `_parse` decides what a
    manifest means and refuses by name, and now `tomli_w` writes back the same
    shape. An HTTP handler building TOML with an f-string would be a second
    author of the format, and the two would disagree the first time a field grew
    a type — silently, because the file would still parse.

    ── VALIDATED BEFORE IT IS WRITTEN, AND VALIDATED AS A FILE ────────────────

    The document is parsed by the SAME `_parse` every packaged manifest goes
    through, at the path it is about to occupy, so a caller is refused by the
    reader's own sentence rather than by a second opinion invented for the wire.
    Then it is round-tripped: serialise, re-parse, and compare what comes back.
    That catches the one class of bug a pre-write check cannot — a value this
    validator accepts and `tomli_w` cannot represent — and it catches it before
    anything reaches the disk rather than at the next render.

    ── AND WRITTEN ATOMICALLY ────────────────────────────────────────────────

    To a temporary file in the same directory, then replaced. A half-written
    manifest is not a broken voice; it is a broken SERVER, because
    `load_all_voices` reads the whole directory and one unparseable file raises
    for every caller of it.

    Returns the manifest as it will be read back.
    """
    path = home_voice_path(voice_id)
    manifest = _parse(document, path, voice_id)

    try:
        text = tomli_w.dumps(document)
    except (TypeError, ValueError) as exc:
        raise VoiceError(
            f"{path.name}: this manifest cannot be written as TOML ({exc}). "
            "Every value must be a string, number, boolean, array or table"
        ) from exc
    # The round trip, for the reason above: what a reader will see, compared with
    # what this call meant.
    written = parse_voice(text, path, voice_id)
    if written != manifest:
        raise VoiceError(
            f"{path.name}: writing this manifest and reading it back did not give "
            "the same voice, so it was not written. This is a defect in Crucible "
            "rather than in the request; please report it"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise VoiceError(f"could not write {path}: {exc}") from exc
    return written


def remove_home_voice(voice_id: str) -> bool:
    """Delete this machine's own manifest for `voice_id`. True if one went.

    A PACKAGED voice is untouched and unreachable from here — the packaged set
    is the install, and deleting out of it would make the next upgrade the thing
    that "restored" a voice somebody meant to be rid of. Removing an overlay that
    SHADOWED a packaged voice brings the packaged one back, which is exactly what
    makes overriding a shipped voice safe to try.
    """
    path = home_voice_path(voice_id)
    if not path.is_file():
        return False
    try:
        path.unlink()
    except OSError as exc:
        raise VoiceError(f"could not remove {path}: {exc}") from exc
    return True


def is_home_voice(voice_id: str) -> bool:
    """Does this machine's own overlay hold a manifest for this id?"""
    try:
        return home_voice_path(voice_id).is_file()
    except VoiceError:
        return False


def _voices_in(root: Path) -> dict[str, VoiceManifest]:
    """The manifests in one directory, by id.

    `pins.toml` IS SKIPPED. It lives here by section 2.2's design -- a machine's
    pins belong beside that machine's voices -- and it is not one, so the glob
    that finds `<id>.toml` would otherwise try to load a voice called `pins` and
    refuse the whole directory over a file that is doing its job.
    """
    voices: dict[str, VoiceManifest] = {}
    source = (
        MANIFEST_OVERRIDE
        if root == home_voices_dir() and root != voices_dir()
        else MANIFEST_PACKAGED
    )
    # By id -- `path.stem` -- and not by path, for the reason `load_all_manifests`
    # gives: the two orders differ whenever one id is a prefix of another, because
    # the extension gets in the way ('-' is 0x2D, '.' is 0x2E). Here that is not
    # hypothetical -- `zeroshot` and `zeroshot-deathstalker` are exactly that pair.
    # This function's order is what `/v1/voices` lists in, so it is the documented
    # one.
    for path in sorted(root.glob("*.toml"), key=lambda p: p.stem):
        if path.name == PINS_FILE:
            continue
        voices[path.stem] = replace(
            _load_voice_file(path, path.stem), manifest_source=source
        )
    return voices
