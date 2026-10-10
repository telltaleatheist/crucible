from __future__ import annotations

from typing import Any, Mapping

from .backend import CardFacts
from .capabilityclasses import BY_NAME, CLASSES, CapabilityClass
from .capabilityrecord import CapabilityRecord
from .contextceiling import check_ceiling, context_ceilings
from .errors import ApiError
from .fit import WorkingContext
from .memorybudget import available_bytes
from .verdict import decide_capabilities, routed_row

WORK_FROM_DEFAULT = "default"
WORK_FROM_REQUEST = "request"

CONTEXT_TOKENS_PARAM = "context_tokens"
CONCURRENCY_PARAM = "concurrency"


def _client_sized_names() -> list[str]:
    return [c.name for c in CLASSES if c.client_sized]


def _positive_int(name: str, raw: str) -> int:
    text = raw.strip()
    if not text.isdigit() or int(text) < 1:
        unit = "tokens" if name == CONTEXT_TOKENS_PARAM else "requests in flight"
        raise ApiError(
            400,
            "invalid_working_context",
            f"{name} is {raw!r}; it must be a positive whole number of {unit}",
            {"field": name, "value": raw},
        )
    return int(text)


def sized_work(
    entry: CapabilityClass,
    *,
    context_tokens: str | None,
    concurrency: str | None,
) -> "WorkingContext | None":
    if context_tokens is None and concurrency is None:
        return None
    if not entry.client_sized:
        client_sized = _client_sized_names()
        ruling = entry.work.source if entry.work is not None else "it has none"
        raise ApiError(
            400,
            "capability_not_client_sized",
            f"{entry.name}'s working context is not the client's to state: it "
            f"is a ruling about the act ({ruling}). Only {client_sized} take "
            f"{CONTEXT_TOKENS_PARAM} and {CONCURRENCY_PARAM}",
            {"capability": entry.name, "client_sized": client_sized},
        )
    default = entry.work
    if default is None:
        raise ValueError(f"{entry.name} is client-sized and declares no default work")
    tokens = (
        default.tokens
        if context_tokens is None
        else _positive_int(CONTEXT_TOKENS_PARAM, context_tokens)
    )
    width = (
        default.concurrency
        if concurrency is None
        else _positive_int(CONCURRENCY_PARAM, concurrency)
    )
    stated = []
    if context_tokens is not None:
        stated.append(f"{CONTEXT_TOKENS_PARAM}={tokens}")
    if concurrency is not None:
        stated.append(f"{CONCURRENCY_PARAM}={width}")
    rest = (
        ""
        if context_tokens is not None and concurrency is not None
        else f"; the rest is the class default ({default.source})"
    )
    return WorkingContext(
        tokens=tokens,
        concurrency=width,
        source="stated by the client: " + ", ".join(stated) + rest,
    )


def _named_class(
    capability_class: str | None, sizing: bool
) -> "CapabilityClass | None":
    if sizing and capability_class is None:
        raise ApiError(
            400,
            "capability_class_required",
            f"{CONTEXT_TOKENS_PARAM} and {CONCURRENCY_PARAM} size ONE class; "
            "name it with ?class=. Only "
            f"{_client_sized_names()} may be sized",
        )
    if capability_class is None:
        return None
    entry = BY_NAME.get(capability_class)
    if entry is None:
        raise ApiError(
            400,
            "unknown_capability",
            f"{capability_class!r} is not a capability class; this build "
            f"knows {sorted(BY_NAME)}",
            {"capability": capability_class, "known": sorted(BY_NAME)},
        )
    return entry


def _check_requested(
    record: CapabilityRecord,
    entry: CapabilityClass,
    requested: WorkingContext,
    *,
    budget: int,
    chosen: Mapping[str, str],
    routes: Mapping[str, str],
    card: "CardFacts | None",
) -> None:
    if not any(row.capability == entry.name for row in record.rows):
        raise ApiError(
            503,
            "capability_undecided",
            f"this server's capability record predates the {entry.name!r} "
            "class and has decided nothing about it. Run `crucible "
            "capability --write` to decide it",
        )
    if routes.get(entry.name) is None:
        check_ceiling(
            entry,
            record.backend_kind,
            available_bytes=budget,
            work=requested,
            chosen=chosen.get(entry.name),
            card=card,
        )


def _redecided_row(
    record: CapabilityRecord,
    entry: CapabilityClass,
    requested: WorkingContext,
    *,
    gpu_vendor: str,
    chosen: Mapping[str, str],
    routes: Mapping[str, str],
    audio_low_vram: bool,
    card: "CardFacts | None",
    packages: frozenset[str],
) -> dict[str, Any]:
    decision = decide_capabilities(
        entry,
        record.backend_kind,
        total_bytes=record.total_bytes,
        desktop_allowance_bytes=record.desktop_allowance_bytes,
        gpu_vendor=gpu_vendor,
        chosen=chosen.get(entry.name),
        work=requested,
        audio_low_vram=audio_low_vram,
        card=card,
        packages=packages,
    )
    fresh = decision.row()
    model = routes.get(entry.name)
    return (fresh if model is None else routed_row(fresh, model)).to_wire()


def served_rows(
    record: CapabilityRecord,
    *,
    gpu_vendor: str,
    chosen: Mapping[str, str],
    routes: Mapping[str, str],
    capability_class: str | None,
    context_tokens: str | None,
    concurrency: str | None,
    audio_low_vram: bool,
    card: "CardFacts | None" = None,
    packages: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    entry = _named_class(
        capability_class, context_tokens is not None or concurrency is not None
    )
    requested = (
        None
        if entry is None
        else sized_work(entry, context_tokens=context_tokens, concurrency=concurrency)
    )
    budget = available_bytes(record.total_bytes, record.desktop_allowance_bytes)
    if entry is not None and requested is not None:
        _check_requested(
            record,
            entry,
            requested,
            budget=budget,
            chosen=chosen,
            routes=routes,
            card=card,
        )

    rows: list[dict[str, Any]] = []
    for stored in record.rows:
        row = stored.to_wire()
        found = BY_NAME.get(stored.capability)
        if found is None:
            row["work"] = None
            row["goal"] = None
            row["context_ceilings"] = None
            rows.append(row)
            continue
        work = found.work
        basis = WORK_FROM_DEFAULT
        if entry is not None and found.name == entry.name and requested is not None:
            work, basis = requested, WORK_FROM_REQUEST
            row = _redecided_row(
                record,
                found,
                requested,
                gpu_vendor=gpu_vendor,
                chosen=chosen,
                routes=routes,
                audio_low_vram=audio_low_vram,
                card=card,
                packages=packages,
            )
        row["work"] = None if work is None else {**work.to_dict(), "from": basis}
        row["goal"] = None if found.goal is None else found.goal.to_dict()
        row["context_ceilings"] = None
        if found.client_sized and work is not None:
            row["context_ceilings"] = [
                ceiling.to_dict()
                for ceiling in context_ceilings(
                    found,
                    record.backend_kind,
                    available_bytes=budget,
                    concurrency=work.concurrency,
                    card=card,
                )
            ]
        rows.append(row)
    return rows


__all__ = [
    "CONCURRENCY_PARAM",
    "CONTEXT_TOKENS_PARAM",
    "WORK_FROM_DEFAULT",
    "WORK_FROM_REQUEST",
    "served_rows",
    "sized_work",
]
