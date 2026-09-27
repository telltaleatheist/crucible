from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from importlib import import_module
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import tomli_w

from .backend import CUDA_LINUX, MLX_DARWIN
from .config import crucible_home
from .errors import CrucibleError
from .narratorengines import ESTIMATE_BASES, HIGGS_V3, NARRATOR_ENGINE_SAMPLING
from .tomltable import (
    HF_REPO_PATTERN,
    REVISION_PATTERN,
    VOICE_ID_PATTERN,
    check_table,
)
from .weights import LOCAL, PINNED

VOICES_DIR_ENV = "CRUCIBLE_VOICES_DIR"

VOICES_PULL_COMMAND = "crucible voices pull"

VOICE_KINDS = frozenset({"checkpoint", "zeroshot", "token"})

VOICE_BACKENDS = frozenset({CUDA_LINUX, MLX_DARWIN})

CLIPS_FROM_REQUEST = "from-request"

VERIFIED = "verified"
ASSERTED = "asserted"

MANIFEST_REPO = "repo"
MANIFEST_OVERRIDE = "override"
MANIFEST_ENGINE = "engine"

PACE_BASES = frozenset({"measured", "inherited"})
MAX_CHARS_BASES = frozenset({"measured", "placeholder"})

PINS_FILE = "pins.toml"
RESERVED_VOICE_IDS = frozenset({"pins"})

_VOICE_REQUIRED: dict[str, type] = {
    "id": str,
    "display": str,
    "kind": str,
    "narrator_engine": str,
    "language": str,
    "sample_rate": int,
}

_PACE_RATES: dict[str, type] = {
    "pace_chars_per_sec": object,
    "max_chars_per_sec": object,
    "min_chars_per_sec": object,
}
_PACE_OPTIONAL: dict[str, type] = {
    "target_chars": int,
    "safe_min_chars": int,
    "safe_max_chars": int,
}
_PACE_EDGES = "edges"
_PACE_EDGES_WORDS = ("percentile",)
_PACE_HALF_ULP = 0.005

_SERVING_REQUIRED: dict[str, type] = {
    "max_num_seqs": int,
    "max_num_seqs_note": str,
}
_SERVING_OPTIONAL: dict[str, type] = {
    "mem_fraction": object,
    "mem_fraction_note": str,
    "context_length": int,
    "context_length_note": str,
}
SERVING_KEYS = frozenset({*_SERVING_REQUIRED, *_SERVING_OPTIONAL})

_SOURCE_KEYS: dict[str, type] = {
    "hf_repo": str,
    "revision": str,
    "path": str,
    "identity": str,
}

_BACKEND_REQUIRED: dict[str, type] = {
    "memory_bytes_estimate": int,
    "estimate_basis": str,
    "sampling": dict,
}
_BACKEND_OPTIONAL: dict[str, type] = {
    **_SOURCE_KEYS,
    "estimate_note": str,
    "sampling_reason": str,
    "max_chars": int,
    "clips": object,
}

_CLIP_REQUIRED: dict[str, type] = {
    "file": str,
    "transcript": str,
    "seconds": object,
}

_TAKE_OPTIONAL: dict[str, type] = {"reason": str}


class VoiceError(CrucibleError):
    ...


@dataclass(frozen=True)
class ReferenceClip:
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
    backend: str
    hf_repo: str | None
    revision: str | None
    path: str | None
    identity: str | None
    memory_bytes_estimate: int
    estimate_basis: str
    estimate_note: str | None
    max_chars: int | None
    sampling: dict[str, float]
    sampling_reason: str | None
    clips: tuple[ReferenceClip, ...] | str | None
    max_chars_basis: str | None = None

    @property
    def clips_from_request(self) -> bool:
        return self.clips == CLIPS_FROM_REQUEST

    @property
    def source(self) -> str:
        return PINNED if self.hf_repo is not None else LOCAL

    @property
    def identity_basis(self) -> str:
        return VERIFIED if self.hf_repo is not None else ASSERTED

    @property
    def weights_identity(self) -> str:
        return self.revision if self.revision is not None else self.identity

    @property
    def local_path(self) -> Path | None:
        return None if self.path is None else Path(self.path)

    @property
    def files(self) -> tuple[str, ...]:
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
    max_num_seqs: int
    max_num_seqs_note: str
    mem_fraction: float | None = None
    mem_fraction_note: str | None = None
    context_length: int | None = None
    context_length_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_num_seqs": self.max_num_seqs,
            "max_num_seqs_note": self.max_num_seqs_note,
            "mem_fraction": self.mem_fraction,
            "mem_fraction_note": self.mem_fraction_note,
            "context_length": self.context_length,
            "context_length_note": self.context_length_note,
        }


@dataclass(frozen=True)
class Take:
    index: int
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
    serving: Serving | None
    backends: dict[str, VoiceBackendSpec]
    takes: tuple[Take, ...]
    path: Path
    manifest_source: str = MANIFEST_OVERRIDE
    pace_basis: str | None = None
    measured_from: str | None = None
    inherited_from: str | None = None
    weights_of: str | None = None
    weights_base: "VoiceManifest | None" = field(default=None, compare=False, repr=False)
    weights_aliases: "tuple[VoiceManifest, ...]" = field(
        default=(), compare=False, repr=False
    )

    weights_family = "voices"

    @property
    def pull_command(self) -> str:
        return f"{VOICES_PULL_COMMAND} {self.id}"

    def aliases(self) -> "tuple[VoiceManifest, ...]":
        return self.weights_aliases

    def extra_files(self, backend_kind: str) -> tuple[str, ...]:
        return ()

    def supports(self, backend_kind: str) -> bool:
        return backend_kind in self.backends

    def spec(self, backend_kind: str) -> VoiceBackendSpec:
        found = self.backends.get(backend_kind)
        if found is None:
            raise VoiceError(
                f"voice {self.id!r} has no {backend_kind} block; {self.path.name} "
                f"declares {sorted(self.backends)}"
            )
        return found

    def take(self, index: int) -> Take:
        if index < 0:
            raise VoiceError(
                f"voice {self.id!r}: take {index} is below take 0. A take names "
                "a rung of the ladder and a seed lane, and counts up from 0"
            )
        if index >= len(self.takes):
            return Take(index=index, overrides={}, reason=None)
        return self.takes[index]

    def applied_sampling(self, backend_kind: str, take: int) -> dict[str, float]:
        return {**self.spec(backend_kind).sampling, **self.take(take).overrides}

    def fingerprint(self, backend_kind: str) -> str:
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
            "measured_from": self.measured_from,
            "inherited_from": self.inherited_from,
        }


def home_voices_dir() -> Path:
    return crucible_home() / "voices"


def voices_dir_is_overridden() -> bool:
    override = os.environ.get(VOICES_DIR_ENV)
    return override is not None and override != ""


def voice_dirs() -> tuple[Path, ...]:
    override = os.environ.get(VOICES_DIR_ENV)
    if override is not None and override != "":
        return (voices_dir(),)
    home = home_voices_dir()
    return (home,) if home.is_dir() else ()


def engine_voices_dir() -> Path:
    return Path(__file__).resolve().parent / "engines"


def engine_voices_path(narrator_engine: str) -> Path:
    return engine_voices_dir() / narrator_engine / "base.toml"


def voices_dir() -> Path:
    override = os.environ.get(VOICES_DIR_ENV)
    if override is not None and override != "":
        path = Path(override).expanduser()
        if not path.is_dir():
            raise VoiceError(f"{VOICES_DIR_ENV}={override!r} is not a directory")
        return path
    path = Path(__file__).resolve().parent / "voices"
    if not path.is_dir():
        raise VoiceError(
            f"no {PINS_FILE} directory at {path}; it is package data and this "
            f"install has lost it, or ${VOICES_DIR_ENV} must point at one"
        )
    return path


def _number(where: str, key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VoiceError(
            f"{where}: {key} must be a number, got {type(value).__name__}"
        )
    return float(value)


def check_pace(where: str, table: dict[str, Any]) -> Pace:
    check_table(
        where,
        table,
        {},
        {**_PACE_RATES, **_PACE_OPTIONAL, _PACE_EDGES: str},
        error=VoiceError,
    )
    edges = _check_edges_word(where, table.get(_PACE_EDGES))
    rates = _check_rate_band(where, table, edges)
    target, floor, ceiling = _check_packing(where, table)
    return Pace(
        pace_chars_per_sec=rates["pace_chars_per_sec"],
        max_chars_per_sec=rates["max_chars_per_sec"],
        min_chars_per_sec=rates["min_chars_per_sec"],
        target_chars=target,
        safe_min_chars=floor,
        safe_max_chars=ceiling,
    )


def _check_edges_word(where: str, edges: Any) -> str | None:
    if edges is not None and edges not in _PACE_EDGES_WORDS:
        raise VoiceError(
            f"{where}: {_PACE_EDGES} {edges!r} is not one of "
            f"{sorted(_PACE_EDGES_WORDS)}; the key says how the two edges were "
            "got, and a word this loader does not know would silently read as "
            "'derived from the pace'"
        )
    return edges


def _check_rate_band(
    where: str, table: dict[str, Any], edges: str | None
) -> dict[str, float | None]:
    stated = set(_PACE_RATES) & set(table)
    if stated and stated != set(_PACE_RATES):
        raise VoiceError(
            f"{where}: declares only part of its rate band, missing "
            f"{sorted(set(_PACE_RATES) - stated)}. The band is a measured pace "
            "and the two edges derived from it; write all three or none"
        )
    if not stated:
        if edges is not None:
            raise VoiceError(
                f"{where}: states {_PACE_EDGES} = {edges!r} but states no rate "
                "band for it to describe; the key says how max_chars_per_sec and "
                "min_chars_per_sec were got, and there are none"
            )
        return dict.fromkeys(_PACE_RATES)
    rates = {key: _number(where, key, table[key]) for key in _PACE_RATES}
    for key, value in rates.items():
        if value <= 0:
            raise VoiceError(f"{where}: {key} must be positive, got {value}")
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
    if edges is None:
        _check_symmetric(where, rates)
    return dict(rates)


def _check_symmetric(where: str, rates: dict[str, float]) -> None:
    long_side = rates["max_chars_per_sec"] / rates["pace_chars_per_sec"]
    short_side = rates["pace_chars_per_sec"] / rates["min_chars_per_sec"]
    rounding = _PACE_HALF_ULP * (1 + long_side) / rates[
        "pace_chars_per_sec"
    ] + _PACE_HALF_ULP * (1 + short_side) / rates["min_chars_per_sec"]
    if abs(long_side - short_side) > rounding:
        raise VoiceError(
            f"{where}: the band is not symmetric — max_chars_per_sec is "
            f"{long_side:.3f} x pace_chars_per_sec but pace_chars_per_sec "
            f"is only {short_side:.3f} x min_chars_per_sec, further apart "
            f"than two-decimal rounding allows ({rounding:.4f}). The two "
            "edges are derived from the measured pace, so both ratios are "
            "the same number; a band whose edges came off a distribution "
            f'instead says so with {_PACE_EDGES} = "percentile"'
        )


def _check_packing(
    where: str, table: dict[str, Any]
) -> tuple[int | None, int | None, int | None]:
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
    return target, floor, ceiling


@dataclass(frozen=True)
class _Source:
    hf_repo: str | None
    revision: str | None
    path: str | None
    identity: str | None


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _is_absolute(value: str) -> bool:
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _check_source(where: str, block: dict[str, Any]) -> _Source:
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
            "machine that serves it); see docs/internals/voices.md, \"The voice schema\""
        )
    return _pinned_source(where, block) if pinned else _local_source(where, block)


def _pinned_source(where: str, block: dict[str, Any]) -> _Source:
    for key in ("path", "identity"):
        if not _blank(block.get(key)):
            raise VoiceError(
                f"{where}: is a pinned block and also carries {key}. A pin's "
                "identity is its revision, which is VERIFIED — the sha is what "
                f"was fetched — so a second {key} beside it would be a fact with "
                "two owners"
            )
    if not HF_REPO_PATTERN.match(block["hf_repo"]):
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
    if not REVISION_PATTERN.match(block["revision"]):
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


def _local_source(where: str, block: dict[str, Any]) -> _Source:
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
    default = NARRATOR_ENGINE_SAMPLING[narrator_engine]
    table = block["sampling"]
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
    where = f"{path.name} [voice.serving]"
    block = voice.get("serving")
    if narrator_engine != HIGGS_V3:
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
    check_table(
        where, block, _SERVING_REQUIRED, _SERVING_OPTIONAL, error=VoiceError
    )
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
        mem_fraction=_check_mem_fraction(where, block),
        mem_fraction_note=block.get("mem_fraction_note"),
        context_length=_check_context_length(where, block),
        context_length_note=block.get("context_length_note"),
    )


def _check_mem_fraction(where: str, block: dict[str, Any]) -> float | None:
    mem_fraction = _check_serving_extra(
        where, block, "mem_fraction",
        "the fraction is preallocated as KV on top of the weights whatever the "
        "width is, and 0.55 measured 24.0-24.1 GB on a 24 GB card, where WDDM "
        "pages to host RAM 4-10x slower and says nothing",
    )
    if mem_fraction is not None:
        mem_fraction = _number(where, "mem_fraction", mem_fraction)
        if not 0 < mem_fraction < 1:
            raise VoiceError(
                f"{where}: mem_fraction must be a fraction in (0, 1), got "
                f"{mem_fraction}. It is SGLang's --mem-fraction-static, and "
                "narrator's launcher refuses anything else by name "
                "(serve_higgs_sgl.sh)"
            )
    return mem_fraction


def _check_context_length(where: str, block: dict[str, Any]) -> int | None:
    context_length = _check_serving_extra(
        where, block, "context_length",
        "4096 is the engine builder's class attribute and holds about 2,000 "
        "characters, so a bank whose longest prompt is 2,008 truncates on the "
        "CONTEXT and the run records it as the voice's length wall",
    )
    if context_length is not None and context_length <= 0:
        raise VoiceError(
            f"{where}: context_length must be positive, got {context_length}"
        )
    return context_length


def _check_serving_extra(
    where: str, block: dict[str, Any], key: str, why: str
) -> Any:
    note_key = f"{key}_note"
    value = block.get(key)
    note = block.get(note_key)
    if value is None:
        if note is not None:
            raise VoiceError(
                f"{where}: states {note_key} and no {key}. The note says where a "
                "number came from and there is no number; drop it, or state the "
                "number it describes"
            )
        return None
    if note is None or note.strip() == "":
        raise VoiceError(
            f"{where}: {key} carries no note. It reconfigures the server "
            f"narrator starts — {why} — so a reader of a /v1/voices row has to "
            "be able to find out where the number came from"
        )
    return value


def _check_takes(
    path: Path, document: dict[str, Any], narrator_engine: str
) -> tuple[Take, ...]:
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


def parse_document(
    document: dict[str, Any], path: Path, expected_id: str
) -> VoiceManifest:
    voice = _voice_table(document, path)
    weights_of = _check_weights_of(path, voice)
    _check_scalars(path, voice, expected_id)
    narrator_engine = voice["narrator_engine"]
    if "pace" in voice and not isinstance(voice["pace"], dict):
        raise VoiceError(f"{path.name}: [voice.pace] must be a table")
    pace = check_pace(f"{path.name} [voice.pace]", voice.get("pace", {}))
    serving = _check_serving(path, voice, narrator_engine)
    backends = _check_backends(path, voice, pace)
    takes = _check_takes(path, voice, narrator_engine)
    return VoiceManifest(
        id=voice["id"],
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
        weights_of=weights_of,
    )


_parse = parse_document


def _voice_table(document: dict[str, Any], path: Path) -> dict[str, Any]:
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
    return voice


def _check_weights_of(path: Path, voice: dict[str, Any]) -> str | None:
    weights_of = voice.get("weights_of")
    if weights_of is not None and not (
        isinstance(weights_of, str) and VOICE_ID_PATTERN.match(weights_of)
    ):
        raise VoiceError(f"{path.name}: voice.weights_of {weights_of!r} is not a voice id")
    return weights_of


def _check_scalars(path: Path, voice: dict[str, Any], expected_id: str) -> None:
    scalars = {
        key: value for key, value in voice.items()
        if key not in ("pace", "serving", "backends", "takes", "weights_of")
    }
    check_table(f"{path.name} [voice]", scalars, _VOICE_REQUIRED, {}, error=VoiceError)
    _check_voice_id(path, voice["id"], expected_id)
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


def _check_voice_id(path: Path, voice_id: str, expected_id: str) -> None:
    if not VOICE_ID_PATTERN.match(voice_id):
        raise VoiceError(
            f"{path.name}: voice.id {voice_id!r} must be lower-case, start with "
            "a letter or digit and be at most 64 characters "
            "([a-z0-9][a-z0-9._-]{0,63})"
        )
    if voice_id != expected_id:
        raise VoiceError(
            f"{path.name}: voice.id is {voice_id!r} but the file is named "
            f"{expected_id!r}; the id and the filename are the same thing"
        )


def _check_backends(
    path: Path, voice: dict[str, Any], pace: Pace
) -> dict[str, VoiceBackendSpec]:
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
    return {
        kind: _check_backend(
            f"{path.name} [voice.backends.{kind}]", kind, block, voice, pace
        )
        for kind, block in backends_table.items()
    }


def _check_backend(
    where: str, kind: str, block: Any, voice: dict[str, Any], pace: Pace
) -> VoiceBackendSpec:
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
    basis, note = _check_estimate(where, block)
    max_chars = _check_max_chars(where, block, pace)
    if not isinstance(block["sampling"], dict):
        raise VoiceError(f"{where}: sampling must be a table")
    sampling, reason = _check_sampling(where, block, voice["narrator_engine"])
    return VoiceBackendSpec(
        backend=kind,
        hf_repo=source.hf_repo,
        revision=source.revision,
        path=source.path,
        identity=source.identity,
        memory_bytes_estimate=block["memory_bytes_estimate"],
        estimate_basis=basis,
        estimate_note=note,
        max_chars=max_chars,
        sampling=sampling,
        sampling_reason=reason,
        clips=_check_clips(where, block, voice["kind"]),
    )


def _check_estimate(where: str, block: dict[str, Any]) -> tuple[str, str | None]:
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
    return basis, note


def _check_max_chars(where: str, block: dict[str, Any], pace: Pace) -> int | None:
    max_chars = block.get("max_chars")
    if max_chars is None:
        return None
    if max_chars <= 0:
        raise VoiceError(
            f"{where}: max_chars must be positive, got {max_chars}"
        )
    if pace.safe_max_chars is not None and pace.safe_max_chars > max_chars:
        raise VoiceError(
            f"{where}: this backend caps the voice at {max_chars} "
            f"characters, but [voice.pace] packs up to safe_max_chars "
            f"{pace.safe_max_chars}. The band may never exceed the arm's "
            "cap — the same rule BookForge and narrator both refuse on"
        )
    if pace.target_chars is not None and pace.target_chars > max_chars:
        raise VoiceError(
            f"{where}: this backend caps the voice at {max_chars} "
            f"characters, but [voice.pace] packs to target_chars "
            f"{pace.target_chars}"
        )
    return max_chars


def voice_document(manifest: VoiceManifest) -> tuple[dict[str, Any], list[str]]:
    voice: dict[str, Any] = {
        "id": manifest.id,
        "display": manifest.display,
        "kind": manifest.kind,
        "narrator_engine": manifest.narrator_engine,
        "language": manifest.language,
        "sample_rate": manifest.sample_rate,
    }
    if manifest.weights_of is not None:
        voice["weights_of"] = manifest.weights_of

    pace = {key: value for key, value in manifest.pace.to_dict().items() if value is not None}
    if pace:
        voice["pace"] = pace

    if manifest.serving is not None:
        voice["serving"] = {
            key: value
            for key, value in manifest.serving.to_dict().items()
            if value is not None
        }

    voice["backends"] = {
        kind: _backend_document(spec) for kind, spec in manifest.backends.items()
    }
    if manifest.takes != (Take(index=0, overrides={}, reason=None),):
        voice["takes"] = [_take_document(take) for take in manifest.takes]
    return {"voice": voice}, _not_carried(manifest)


def _backend_document(spec: VoiceBackendSpec) -> dict[str, Any]:
    block: dict[str, Any] = {}
    if spec.hf_repo is not None:
        block["hf_repo"] = spec.hf_repo
        block["revision"] = spec.revision
    else:
        block["path"] = spec.path
        block["identity"] = spec.identity
    block["memory_bytes_estimate"] = spec.memory_bytes_estimate
    block["estimate_basis"] = spec.estimate_basis
    if spec.estimate_note is not None:
        block["estimate_note"] = spec.estimate_note
    if spec.max_chars is not None:
        block["max_chars"] = spec.max_chars
    block["sampling"] = dict(spec.sampling)
    if spec.sampling_reason is not None:
        block["sampling_reason"] = spec.sampling_reason
    if spec.clips is not None:
        block["clips"] = (
            spec.clips
            if isinstance(spec.clips, str)
            else [clip.to_dict() for clip in spec.clips]
        )
    return block


def _take_document(take: Take) -> dict[str, Any]:
    rung: dict[str, Any] = dict(take.overrides)
    if take.reason is not None:
        rung["reason"] = take.reason
    return rung


def _not_carried(manifest: VoiceManifest) -> list[str]:
    not_carried: list[str] = []
    if manifest.pace_basis is not None:
        not_carried.append(f"pace_basis = {manifest.pace_basis!r}")
    if manifest.measured_from is not None:
        not_carried.append(f"measured_from = {manifest.measured_from!r}")
    if manifest.inherited_from is not None:
        not_carried.append(f"inherited_from = {manifest.inherited_from!r}")
    for kind, spec in manifest.backends.items():
        if spec.max_chars_basis is not None:
            not_carried.append(f"backends.{kind}.max_chars_basis = {spec.max_chars_basis!r}")
    return not_carried


def parse_voice(text: str, path: Path, expected_id: str) -> VoiceManifest:
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise VoiceError(f"{path.name}: not valid TOML: {exc}") from exc
    return parse_document(document, path, expected_id)


def parse_engine_base(text: str, path: Path, engine: str) -> dict[str, VoiceManifest]:
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
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
    found: dict[str, VoiceManifest] = {}
    for voice_id in sorted(table):
        block = table[voice_id]
        if not isinstance(block, dict):
            raise VoiceError(f"{path.name}: [voices.{voice_id}] must be a table")
        manifest = parse_document({"voice": block}, path, voice_id)
        if manifest.narrator_engine != engine:
            raise VoiceError(
                f"{path.name}: [voices.{voice_id}] names narrator_engine "
                f"{manifest.narrator_engine!r} but sits under {engine!r}. A "
                "base row is the engine's own behaviour, so the directory it "
                "is in and the engine it names are one fact"
            )
        found[voice_id] = replace(manifest, manifest_source=MANIFEST_ENGINE)
    return found


def home_voice_path(voice_id: str) -> Path:
    if not VOICE_ID_PATTERN.match(voice_id):
        raise VoiceError(
            f"voice id {voice_id!r} is not usable as a manifest name: lower-case "
            "letters, digits, dot, dash and underscore, starting with a letter or "
            "digit, at most 64 characters. The id becomes a FILENAME, so anything "
            "else is a path rather than a name"
        )
    if voice_id in RESERVED_VOICE_IDS:
        raise VoiceError(
            f"voice id {voice_id!r} is reserved: {PINS_FILE} in this directory is "
            "this machine's pin list (docs/internals/voices.md, \"Pins\"), so a voice of that "
            "name would be written over it"
        )
    return home_voices_dir() / f"{voice_id}.toml"


def write_home_voice(voice_id: str, document: dict[str, Any]) -> VoiceManifest:
    path = home_voice_path(voice_id)
    manifest = parse_document(document, path, voice_id)

    try:
        text = tomli_w.dumps(document)
    except (TypeError, ValueError) as exc:
        raise VoiceError(
            f"{path.name}: this manifest cannot be written as TOML ({exc}). "
            "Every value must be a string, number, boolean, array or table"
        ) from exc
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
    path = home_voice_path(voice_id)
    if not path.is_file():
        return False
    try:
        path.unlink()
    except OSError as exc:
        raise VoiceError(f"could not remove {path}: {exc}") from exc
    return True


_MOVED_TO_VOICECATALOG: dict[str, str] = {
    "load_all_voices": "load_all_voices",
    "load_voice": "load_voice",
    "unserved_pins": "unserved_pins",
    "voice_aliases_of": "voice_aliases_of",
    "_resolve_weights_of": "resolve_weights_of",
    "_engine_voices": "engine_voices",
    "_voices_in": "voices_in",
    "_load_voice_file": "load_voice_file",
}


def __getattr__(name: str) -> Any:
    moved = _MOVED_TO_VOICECATALOG.get(name)
    if moved is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__package__}.voicecatalog"), moved)
