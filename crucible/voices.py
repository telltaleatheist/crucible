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
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w

from .backend import CUDA_LINUX, MLX_DARWIN
from .errors import CrucibleError
from .manifests import check_table

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
_SERVING_REQUIRED: dict[str, type] = {
    "max_num_seqs": int,
    "max_num_seqs_note": str,
}

_BACKEND_REQUIRED: dict[str, type] = {
    "hf_repo": str,
    "revision": str,
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
    hf_repo: str
    revision: str
    memory_bytes_estimate: int
    estimate_basis: str
    estimate_note: str | None
    max_chars: int
    sampling: dict[str, float]
    sampling_reason: str | None
    #: The clips this voice is conditioned on, `CLIPS_FROM_REQUEST`, or None for
    #: a voice that carries none.
    clips: tuple[ReferenceClip, ...] | str | None

    @property
    def clips_from_request(self) -> bool:
        return self.clips == CLIPS_FROM_REQUEST

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
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "estimate_basis": self.estimate_basis,
            "estimate_note": self.estimate_note,
            "max_chars": self.max_chars,
            "sampling": dict(self.sampling),
            "sampling_reason": self.sampling_reason,
            "clips": clips,
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
        """`<id>@<revision>` — what a render records as the voice it used."""
        return f"{self.id}@{self.spec(backend_kind).revision}"

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
        }


# ------------------------------------------------------------------ locating


def home_voices_dir() -> Path:
    """`<CRUCIBLE_HOME>/voices` — this machine's own voices. May not exist."""
    from .config import crucible_home

    return crucible_home() / "voices"


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
        where, table, {}, {**_PACE_RATES, **_PACE_OPTIONAL}, error=VoiceError
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

        if not _HF_REPO.match(block["hf_repo"]):
            raise VoiceError(
                f"{where}: hf_repo {block['hf_repo']!r} is not an <owner>/<name> "
                "HuggingFace repo id"
            )
        if not _REVISION.match(block["revision"]):
            raise VoiceError(
                f"{where}: revision {block['revision']!r} must be a full 40-character "
                "commit sha, so a pull is reproducible; branch names are not pins"
            )
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
            hf_repo=block["hf_repo"],
            revision=block["revision"],
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
    """Load `<voice_id>.toml`. Raises VoiceError if no directory holds it.

    Searched HIGHEST precedence first, so `<CRUCIBLE_HOME>/voices` answers
    before the packaged set — the same order `load_all_voices` merges in, read
    from the other end.
    """
    roots = (directory,) if directory is not None else tuple(reversed(voice_dirs()))
    for root in roots:
        path = root / f"{voice_id}.toml"
        if path.is_file():
            break
    else:
        known = sorted({p.stem for root in roots for p in root.glob("*.toml")})
        where = ", ".join(str(r) for r in roots)
        raise VoiceError(
            f"no manifest for voice {voice_id!r} in {where}; this host serves {known}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VoiceError(f"could not read {path}: {exc}") from exc
    return parse_voice(text, path, voice_id)


def load_all_voices(directory: Path | None = None) -> dict[str, VoiceManifest]:
    """Every voice this host serves, by id, in id order.

    Packaged first, then `<CRUCIBLE_HOME>/voices`, so a home manifest sharing an
    id REPLACES the shipped one — see `voice_dirs()`. Passing `directory`
    reads exactly that one, which is what the tests and `--voices-dir` want.
    """
    roots = (directory,) if directory is not None else voice_dirs()
    voices: dict[str, VoiceManifest] = {}
    for root in roots:
        voices.update(_voices_in(root))
    # Re-sorted because the merge is by directory and the ORDER is by id: a home
    # voice inserted in the middle of the shipped set must list in the middle,
    # not at the end. `/v1/voices` lists in this order and it is documented.
    return {vid: voices[vid] for vid in sorted(voices)}


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
    """The manifests in one directory, by id."""
    voices: dict[str, VoiceManifest] = {}
    # By id — `path.stem` — and not by path, for the reason `load_all_manifests`
    # gives: the two orders differ whenever one id is a prefix of another, because
    # the extension gets in the way ('-' is 0x2D, '.' is 0x2E). Here that is not
    # hypothetical — `zeroshot` and `zeroshot-deathstalker` are exactly that pair.
    # This function's order is what `/v1/voices` lists in, so it is the documented
    # one.
    for path in sorted(root.glob("*.toml"), key=lambda p: p.stem):
        voices[path.stem] = load_voice(path.stem, root)
    return voices
