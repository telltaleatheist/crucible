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

MAX_REFERENCE_SECONDS = 30.0

MAX_REFERENCE_BYTES = 32 * 1024 * 1024

REFERENCE_NAME = "narrator-reference.wav"


class ReferenceError(CrucibleError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ClipEntry:
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
    audio: bytes
    transcript: str
    name: str | None
    seconds: float
    sha256: str

    @property
    def short_hash(self) -> str:
        return self.sha256[:12]

    def to_report(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sha256": self.sha256,
            "seconds": self.seconds,
        }


def _refuse(message: str) -> ReferenceError:
    return ReferenceError("reference_malformed", message)


def parse_reference(raw: Any) -> VoiceReference:
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

    ceiling = 4 * math.ceil(MAX_REFERENCE_BYTES / 3)
    if len(data) > ceiling:
        raise _refuse(
            f"reference.data is {len(data)} base64 characters, over this "
            f"server's {MAX_REFERENCE_BYTES / 1024 ** 2:.0f} MiB ceiling. "
            f"narrator caps a reference at {MAX_REFERENCE_SECONDS:.0f} seconds "
            "and no wav that short is this big"
        )
    try:
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
