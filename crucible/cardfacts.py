from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import VERSION
from .backend import CUDA_GRAPHS, VLLM_STARTS, CardFacts, Gpu, declared_card

LADDER_SCHEMA = 1
RECORD_NAME = "card.json"

CARD = "card"
ENV = "env"
GRAPHS = "cuda_graphs"
VLLM = "vllm"
RUNGS: tuple[str, ...] = (CARD, ENV, GRAPHS, VLLM)
GPU_RUNGS: frozenset[str] = frozenset({ENV, GRAPHS, VLLM})

PASSED = "passed"
FAILED = "failed"
INTERRUPTED = "interrupted"
WAITING = "waiting"
SKIPPED = "skipped"


@dataclass
class RungResult:
    rung: str
    outcome: str
    measured_at: str
    facts: dict[str, Any] = field(default_factory=dict)
    detail: str = ""
    contention: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rung": self.rung,
            "outcome": self.outcome,
            "basis": "measured",
            "measured_at": self.measured_at,
            "facts": self.facts,
            "detail": self.detail,
            "contention": self.contention,
        }


def record_path(home: Path) -> Path:
    return home / "ladder" / RECORD_NAME


def card_key(gpu: Gpu) -> dict[str, Any]:
    return {
        "name": gpu.name,
        "compute_capability": gpu.compute_capability,
        "vram_bytes": gpu.vram_bytes,
        "crucible_version": VERSION,
    }


def load_record(home: Path) -> dict[str, Any] | None:
    path = record_path(home)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict) or document.get("schema") != LADDER_SCHEMA:
        return None
    return document


def stale_reason(home: Path, gpu: Gpu) -> str | None:
    document = load_record(home)
    if document is None:
        return None
    recorded = document.get("key", {})
    current = card_key(gpu)
    changed = [name for name in current if recorded.get(name) != current[name]]
    if not changed:
        return None
    return (
        "the measurement record was taken with a different "
        + ", ".join(changed)
        + " than this host has now; `crucible ladder` measures again"
    )


def results_from(document: dict[str, Any] | None) -> dict[str, RungResult]:
    found: dict[str, RungResult] = {}
    if document is None:
        return found
    for name, row in (document.get("rungs") or {}).items():
        if not isinstance(row, dict) or name not in RUNGS:
            continue
        found[name] = RungResult(
            rung=name,
            outcome=str(row.get("outcome", "")),
            measured_at=str(row.get("measured_at", "")),
            facts=dict(row.get("facts") or {}),
            detail=str(row.get("detail", "")),
            contention=row.get("contention"),
        )
    return found


_FEATURE_OF_RUNG: dict[str, str] = {GRAPHS: CUDA_GRAPHS, VLLM: VLLM_STARTS}


def card_for(home: Path, gpu: Gpu) -> CardFacts:
    declared = declared_card(gpu)
    if stale_reason(home, gpu) is not None:
        return declared
    results = results_from(load_record(home))
    measured: dict[str, bool] = {}
    detail: dict[str, str] = {}
    latest: str | None = None
    for rung, feature in _FEATURE_OF_RUNG.items():
        result = results.get(rung)
        if result is None or result.outcome not in (PASSED, FAILED):
            continue
        measured[feature] = result.outcome == PASSED
        if result.outcome == FAILED and result.detail:
            detail[feature] = result.detail
        if latest is None or result.measured_at > latest:
            latest = result.measured_at
    return CardFacts(
        name=declared.name,
        compute_capability=declared.compute_capability,
        measured=measured,
        measured_detail=detail,
        measured_at=latest,
    )


__all__ = [
    "CARD",
    "ENV",
    "FAILED",
    "GPU_RUNGS",
    "GRAPHS",
    "INTERRUPTED",
    "LADDER_SCHEMA",
    "PASSED",
    "RECORD_NAME",
    "RUNGS",
    "RungResult",
    "SKIPPED",
    "VLLM",
    "WAITING",
    "card_for",
    "card_key",
    "load_record",
    "record_path",
    "results_from",
    "stale_reason",
]
