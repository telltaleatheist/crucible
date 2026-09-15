"""The reference clip a zero-shot voice is loaded with.

PHASE3-TTS.md section 5's amendment. A `kind = "zeroshot"` voice is the base
weights plus a recording of somebody: the weights are Crucible's (pulled at the
manifest's pin, like any other voice's) and **the clip is the client's**. Owen,
2026-09-14: *"zero shot uses a voice reference and the base model i believe. it
should effectively be treated as a model, for all intents and purposes, except
the route it takes to retrieve and return the audio."* So it loads through the
same `load-voice` job as every fine-tune, with one extra field.

WHY A `reference` AND NOT A CATALOG ENTRY. BookForge's four `zeroshot-*` voices
name wavs that live in `<userData>/runtime/higgs-models/refs/` and are published
nowhere; the browser extension's live in `chrome.storage.local`. `voices/
zeroshot.toml` says at length why writing four manifests naming files no server
has would be worse than useless. The clip is a per-client CHOICE, like the voice
pick itself, so it travels with the load — which is the one moment it is needed,
because a Higgs v3 voice change is a worker restart anyway.

THE SHAPE IS NARRATOR'S, TRANSLATED ONCE. narrator reads reference clips out of
the `NARRATOR_HIGGS_VOICES` document as `{"path", "transcript", "seconds"}`
(`engine/higgs/config.py:load_voices`), where `path` is a FILE ON THE SERVER'S
OWN DISK — it calls `os.path.isfile` on it, the served arm base64s that file
itself into vllm-omni's `references[].data`, and the MLX arm hands it to
`encode_reference_audio`. A client across a network has no such path, so the
wire carries the BYTES and this module is the one place they become a file:

    {"data": "<base64 of a RIFF/WAVE file>",     -> the file this writes
     "transcript": "<the book-exact text>",      -> narrator's `transcript`
     "name": "<short label>"}                    -> the residency report only

`seconds` IS NOT ON THE WIRE, deliberately, and this is the one place the two
shapes differ by more than a spelling. narrator needs it — `reference_seconds`
RAISES on a clip that has none, because the 30 s reference budget is checked
before the request is built — but Crucible is holding the bytes and can read the
duration out of the header. A duration the client states is a second owner of a
fact the server already has, and the day the two disagree the refusal would name
the honest one as the liar (`docs/ARCHITECTURE.md`, R1). So it is measured here.

THE TRANSCRIPT IS REQUIRED, and section 4b of BookForge's
`docs/EXTENSION-TO-CRUCIBLE-PLAN.md` — which sketched this field as
`{data: <base64 wav>}` alone — is wrong about that. narrator refuses a
`ReferenceClip` with an empty transcript AT CONSTRUCTION, and says why in as
many words: *"a zero-shot clone conditioned on a wrong or absent transcript is a
whole book in a subtly wrong voice, reported as success"*. It is the same law
the training corpora are held to (`orpheus-training-text-doctrine`): the
book-exact text the clip was cut from, never an ASR guess. A load with no
transcript is refused here rather than accepted and turned into a refusal inside
narrator after the process is up — and the extension's clip picker therefore
needs a transcript field beside its file input.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import CrucibleError

#: Total reference audio narrator will accept, from `v3_served
#: .MAX_REFERENCE_SECONDS`: vllm-omni answers HTTP 400 "Reference audio too
#: long" above it, and `check_reference_budget` refuses it client-side first.
#: Mirrored here for `crucible/narratorpatches.py`'s reason — a Crucible server
#: must not need a BookForge checkout to answer "will this clip load" — and
#: checked at the door so the refusal names the clip rather than arriving from
#: inside an engine that has already started.
MAX_REFERENCE_SECONDS = 30.0

#: The decoded ceiling. 30 s of 96 kHz stereo 32-bit PCM is 23.0 MB, which is
#: the largest a wav inside the duration cap can honestly be, so 32 MiB is a
#: ceiling no legitimate clip reaches and a bound on what this process will
#: decode. Checked against the ENCODED length first: base64 of a gigabyte is a
#: gigabyte and a third, and refusing it after decoding it is refusing it too
#: late.
MAX_REFERENCE_BYTES = 32 * 1024 * 1024

#: One file per server, beside the voices document and overwritten at every
#: load, for the same reason: the previous load's clip is exactly the stale
#: thing a post-mortem must not find. A server holds one resident voice, so
#: there is never a second live clip to collide with.
REFERENCE_NAME = "narrator-reference.wav"


class ReferenceError(CrucibleError):
    """The reference on a `load-voice` is not a usable reference clip.

    Carries the refusal code the door reports it under, so the door does not
    re-derive from the message which of the three it was.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ClipEntry:
    """One row of narrator's `clips` list, spelled as narrator reads it.

    A file on this host's disk, the book-exact text spoken in it, and how long
    it is. Exactly the three keys `config.load_voices` reads and nothing else.
    """

    path: Path
    transcript: str
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "transcript": self.transcript,
            "seconds": self.seconds,
        }


@dataclass(frozen=True)
class VoiceReference:
    """A validated reference clip: the bytes, what is said in them, how long.

    `sha256` is over the DECODED audio, so two clients sending the same wav
    (however each of them encoded it) agree about which clip is resident, and a
    client that re-encoded a different take of the same line does not.
    """

    audio: bytes
    transcript: str
    name: str | None
    seconds: float
    sha256: str

    @property
    def short_hash(self) -> str:
        """The first 12 hex of the digest — what the residency report shows."""
        return self.sha256[:12]

    def to_report(self) -> dict[str, Any]:
        """What `/v1/info` and `/v1/activity` say about the resident clip.

        The clip itself is never published: it is somebody's voice, it is
        megabytes, and nothing downstream needs it back. What two clients need
        is to be able to tell whether the clip THEY chose is the one that is
        loaded, which the digest answers and the label makes readable.
        `name` is null when the client sent none; it is a label and is never
        derived from the hash, because a made-up name is one a client would
        then look for.
        """
        return {
            "name": self.name,
            "sha256": self.sha256,
            "seconds": self.seconds,
        }


def _refuse(message: str) -> ReferenceError:
    return ReferenceError("reference_malformed", message)


def parse_reference(raw: Any) -> VoiceReference:
    """One `params.reference` off the wire, or `reference_malformed` by name.

    Every check names the field it failed on. Nothing here has a default and
    nothing is repaired: a clip that is not what it says it is conditions a
    whole book on the wrong thing and reports success.
    """
    if not isinstance(raw, dict):
        raise _refuse(
            f"reference is {type(raw).__name__}, not an object with `data` (a "
            "base64 WAV) and `transcript` (the book-exact text spoken in it)"
        )
    data = raw.get("data")
    if not isinstance(data, str) or data == "":
        raise _refuse("reference.data is missing or empty; it is a base64 WAV")
    transcript = raw.get("transcript")
    if not isinstance(transcript, str) or transcript.strip() == "":
        raise _refuse(
            "reference.transcript is missing or empty. A reference clip is only "
            "usable with the BOOK-EXACT text spoken in it — narrator refuses a "
            "clip without one at construction, because a zero-shot clone "
            "conditioned on an absent transcript is a whole book in a subtly "
            "wrong voice, reported as success"
        )
    name = raw.get("name")
    if name is not None and (not isinstance(name, str) or name.strip() == ""):
        raise _refuse(
            "reference.name is present and is not a label. Omit it or send a "
            "short string; an empty one would show on /v1/info as a clip that "
            "has a name and will not say it"
        )

    # The encoded length first: 4 characters per 3 bytes, so this bounds the
    # decode rather than following it.
    ceiling = 4 * math.ceil(MAX_REFERENCE_BYTES / 3)
    if len(data) > ceiling:
        raise _refuse(
            f"reference.data is {len(data)} base64 characters, over this "
            f"server's {MAX_REFERENCE_BYTES / 1024 ** 2:.0f} MiB ceiling. "
            f"narrator caps a reference at {MAX_REFERENCE_SECONDS:.0f} seconds "
            "and no wav that short is this big"
        )
    try:
        # `validate=True`: without it base64 SKIPS characters it does not
        # recognise, so a data: URI prefix or a pasted newline would decode to
        # something that is not what was sent. narrator's served arm has the
        # same rule from the other side — `load_base64` fails on a prefixed
        # string — and a silent skip here would be Crucible deciding which
        # bytes the client meant.
        audio = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _refuse(
            f"reference.data is not base64 ({exc}). Send the wav's bytes "
            "base64-encoded with no `data:` prefix and no whitespace"
        ) from None
    if len(audio) > MAX_REFERENCE_BYTES:
        raise _refuse(
            f"reference.data decodes to {len(audio)} bytes, over this server's "
            f"{MAX_REFERENCE_BYTES / 1024 ** 2:.0f} MiB ceiling"
        )

    seconds = _wav_seconds(audio)
    if seconds > MAX_REFERENCE_SECONDS:
        raise _refuse(
            f"reference.data is {seconds:.1f} s of audio and narrator caps a "
            f"reference at {MAX_REFERENCE_SECONDS:.0f} s "
            "(`v3_served.check_reference_budget`; vllm-omni answers HTTP 400 "
            '"Reference audio too long" above it). Two ~14 s clips joined is '
            "the practical maximum, and the second clip is worth +0.012 speaker "
            "cosine — a same-BOOK clip is worth +0.076"
        )
    return VoiceReference(
        audio=audio,
        transcript=transcript,
        name=None if name is None else name.strip(),
        seconds=seconds,
        sha256=hashlib.sha256(audio).hexdigest(),
    )


def _wav_seconds(audio: bytes) -> float:
    """The clip's duration, read off its own header, or `reference_malformed`.

    `wave` is the standard library and parses the RIFF container, so this both
    proves the bytes ARE a wav and measures them in one pass — no `soundfile`,
    no ffmpeg subprocess, and no second opinion about a number the header
    already states. A wav is what both arms want: the served one declares
    `audio/wav` as the media type of the reference it posts, and the MLX one
    hands the path to mlx-audio's loader.
    """
    try:
        with wave.open(io.BytesIO(audio), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
    except (wave.Error, EOFError) as exc:
        raise _refuse(
            f"reference.data is not a readable WAV file ({exc}). narrator's "
            "served arm posts it as audio/wav and the MLX arm loads it as a "
            "file; a container neither can read would fail after the engine "
            "was already up"
        ) from None
    if rate <= 0 or frames <= 0:
        raise _refuse(
            f"reference.data is a WAV with {frames} frames at {rate} Hz, which "
            "is no audio at all"
        )
    return frames / float(rate)


def reference_path(home: Path) -> Path:
    return home / REFERENCE_NAME


def place(home: Path, reference: VoiceReference) -> ClipEntry:
    """Write the clip where narrator can read it, and describe it as narrator does.

    Called from `write_document`, at the same moment and for the same reason:
    narrator checks `os.path.isfile` on every clip path in the document, so the
    file and the document that names it are written together or the load dies
    inside narrator instead of here.
    """
    path = reference_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(reference.audio)
    return ClipEntry(
        path=path, transcript=reference.transcript, seconds=reference.seconds
    )


__all__ = [
    "MAX_REFERENCE_BYTES",
    "MAX_REFERENCE_SECONDS",
    "REFERENCE_NAME",
    "ClipEntry",
    "ReferenceError",
    "VoiceReference",
    "parse_reference",
    "place",
    "reference_path",
]
