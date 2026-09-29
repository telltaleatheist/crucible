from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .errors import CrucibleError
from .tomltable import check_table
from .voicereference import MAX_REFERENCE_SECONDS

CHUNK_GAP = "chunk_gap"
EDGE_FADE_MS = "edge_fade_ms"
REFERENCE_SECONDS_CAP = "reference_seconds_cap"
ALLOWED_CONTROLS = "allowed_controls"

ARM_FACT_KEYS: dict[str, type] = {
    EDGE_FADE_MS: dict,
    REFERENCE_SECONDS_CAP: object,
    ALLOWED_CONTROLS: list,
}

_CHUNK_GAP_SECONDS = ("inject_s", "target_join_s", "model_self_tail_s")
_CHUNK_GAP_OPTIONAL_SECONDS = ("reader_sentence_gap_s", "model_internal_gap_s")
_CHUNK_GAP_PROSE = ("rule", "method", "source", "measured_on")

_CHUNK_GAP_REQUIRED: dict[str, type] = {
    **{key: object for key in _CHUNK_GAP_SECONDS},
    **{key: str for key in _CHUNK_GAP_PROSE},
}
_CHUNK_GAP_OPTIONAL: dict[str, type] = {key: object for key in _CHUNK_GAP_OPTIONAL_SECONDS}

CHUNK_GAP_SUM_TOLERANCE_S = 0.011

_EDGE_FADE_REQUIRED: dict[str, type] = {"in": object, "out": object}

CONTROL_TOKEN_PATTERN = re.compile(r"^<\|[a-z_]+:[a-z_]+\|>$")


class VoiceFactError(CrucibleError):
    ...


def _seconds(where: str, key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VoiceFactError(f"{where}: {key} must be a number, got {type(value).__name__}")
    if not math.isfinite(value) or value < 0:
        raise VoiceFactError(f"{where}: {key} must be a finite number >= 0, got {value}")
    return float(value)


@dataclass(frozen=True)
class ChunkGap:
    inject_s: float
    target_join_s: float
    model_self_tail_s: float
    reader_sentence_gap_s: float | None
    model_internal_gap_s: float | None
    rule: str
    method: str
    source: str
    measured_on: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "inject_s": self.inject_s,
            "target_join_s": self.target_join_s,
            "model_self_tail_s": self.model_self_tail_s,
            "reader_sentence_gap_s": self.reader_sentence_gap_s,
            "model_internal_gap_s": self.model_internal_gap_s,
            "rule": self.rule,
            "method": self.method,
            "source": self.source,
            "measured_on": self.measured_on,
        }

    def to_document(self) -> dict[str, Any]:
        return {key: value for key, value in self.to_dict().items() if value is not None}


def check_chunk_gap(where: str, table: Any) -> ChunkGap:
    if not isinstance(table, dict):
        raise VoiceFactError(f"{where}: must be a table")
    check_table(where, table, _CHUNK_GAP_REQUIRED, _CHUNK_GAP_OPTIONAL, error=VoiceFactError)
    seconds = {key: _seconds(where, key, table[key]) for key in _CHUNK_GAP_SECONDS}
    optional = {
        key: None if table.get(key) is None else _seconds(where, key, table[key])
        for key in _CHUNK_GAP_OPTIONAL_SECONDS
    }
    for key in _CHUNK_GAP_PROSE:
        if table[key].strip() == "":
            raise VoiceFactError(
                f"{where}: {key} is empty. A chunk gap is measured, and rule, method, "
                "source and measured_on say how, so two gaps can be compared"
            )
    joined = seconds["inject_s"] + seconds["model_self_tail_s"]
    if abs(joined - seconds["target_join_s"]) > CHUNK_GAP_SUM_TOLERANCE_S:
        raise VoiceFactError(
            f"{where}: inject_s {seconds['inject_s']} + model_self_tail_s "
            f"{seconds['model_self_tail_s']} = {joined:.3f}, which is not its "
            f"target_join_s {seconds['target_join_s']}. inject_s is the silence "
            "added NET of the tail the model already emits, not the target join"
        )
    return ChunkGap(
        **seconds,
        **optional,
        **{key: table[key] for key in _CHUNK_GAP_PROSE},
    )


def check_edge_fade(where: str, table: Any) -> dict[str, float]:
    at = f"{where} {EDGE_FADE_MS}"
    if not isinstance(table, dict):
        raise VoiceFactError(f"{at}: must be a table of `in` and `out` milliseconds")
    check_table(at, table, _EDGE_FADE_REQUIRED, {}, error=VoiceFactError)
    return {key: _seconds(at, key, table[key]) for key in _EDGE_FADE_REQUIRED}


def check_reference_cap(where: str, value: Any) -> float:
    cap = _seconds(where, REFERENCE_SECONDS_CAP, value)
    if cap == 0 or cap > MAX_REFERENCE_SECONDS:
        raise VoiceFactError(
            f"{where}: {REFERENCE_SECONDS_CAP} {cap} must be above 0 and at most "
            f"{MAX_REFERENCE_SECONDS:.0f}, the longest reference narrator accepts"
        )
    return cap


def check_allowed_controls(where: str, value: Any) -> tuple[str, ...]:
    at = f"{where} {ALLOWED_CONTROLS}"
    if not isinstance(value, list):
        raise VoiceFactError(f"{at}: must be a list of control tokens, got {type(value).__name__}")
    for token in value:
        if not isinstance(token, str) or not CONTROL_TOKEN_PATTERN.match(token):
            raise VoiceFactError(
                f"{at}: {token!r} is not a control token shaped `<|group:name|>`, "
                "for example `<|prosody:long_pause|>`. An empty list allows none"
            )
    if len(set(value)) != len(value):
        raise VoiceFactError(f"{at}: lists a token twice")
    return tuple(value)


@dataclass(frozen=True)
class ArmFacts:
    edge_fade_ms: dict[str, float] | None = None
    reference_seconds_cap: float | None = None
    allowed_controls: tuple[str, ...] | None = None

    def to_document(self) -> dict[str, Any]:
        found: dict[str, Any] = {}
        if self.edge_fade_ms is not None:
            found[EDGE_FADE_MS] = dict(self.edge_fade_ms)
        if self.reference_seconds_cap is not None:
            found[REFERENCE_SECONDS_CAP] = self.reference_seconds_cap
        if self.allowed_controls is not None:
            found[ALLOWED_CONTROLS] = list(self.allowed_controls)
        return found


def check_arm_facts(where: str, block: dict[str, Any]) -> ArmFacts:
    fade = block.get(EDGE_FADE_MS)
    cap = block.get(REFERENCE_SECONDS_CAP)
    controls = block.get(ALLOWED_CONTROLS)
    return ArmFacts(
        edge_fade_ms=None if fade is None else check_edge_fade(where, fade),
        reference_seconds_cap=None if cap is None else check_reference_cap(where, cap),
        allowed_controls=None if controls is None else check_allowed_controls(where, controls),
    )


__all__ = [
    "ALLOWED_CONTROLS",
    "ARM_FACT_KEYS",
    "CHUNK_GAP",
    "CHUNK_GAP_SUM_TOLERANCE_S",
    "CONTROL_TOKEN_PATTERN",
    "EDGE_FADE_MS",
    "REFERENCE_SECONDS_CAP",
    "ArmFacts",
    "ChunkGap",
    "VoiceFactError",
    "check_allowed_controls",
    "check_arm_facts",
    "check_chunk_gap",
    "check_edge_fade",
    "check_reference_cap",
]
