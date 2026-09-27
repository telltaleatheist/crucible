from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .backend import CardFacts
from .errors import ApiError
from .fit import Candidate, ContextCeiling, WorkingContext

if TYPE_CHECKING:
    from .capabilityclasses import CapabilityClass

MIN_LOAD_CONTEXT = 2048


def context_ceilings(
    entry: "CapabilityClass",
    backend_kind: str,
    *,
    available_bytes: int,
    concurrency: int,
    card: "CardFacts | None" = None,
) -> tuple[ContextCeiling, ...]:
    if entry.candidates is None:
        return ()
    found = tuple(
        c
        for c in entry.candidates(backend_kind)
        if not c.lacks(card)
    )
    ceilings = (c.context_ceiling(available_bytes, concurrency) for c in found)
    return tuple(ceiling for ceiling in ceilings if ceiling is not None)


def check_ceiling(
    entry: "CapabilityClass",
    backend_kind: str,
    *,
    available_bytes: int,
    work: WorkingContext,
    chosen: str | None,
    card: "CardFacts | None" = None,
) -> tuple[ContextCeiling, ...]:
    ceilings = context_ceilings(
        entry,
        backend_kind,
        available_bytes=available_bytes,
        concurrency=work.concurrency,
        card=card,
    )
    if not ceilings or entry.candidates is None:
        return ceilings
    by_model = {ceiling.model: ceiling for ceiling in ceilings}
    if chosen is not None:
        if chosen not in by_model:
            return ceilings
        governing = by_model[chosen]
    else:
        governing = max(ceilings, key=lambda ceiling: ceiling.tokens)
    holds_weights = [
        c
        for c in entry.candidates(backend_kind)
        if not c.lacks(card)
        and (c.memory is None or c.memory.fixed_bytes < available_bytes)
    ]
    if not holds_weights or work.tokens <= governing.tokens:
        return ceilings
    whose = (
        " (the model chosen for this class)"
        if chosen is not None
        else " (the highest of this class's candidates on this host)"
    )
    raise over_limit(
        f"{work.tokens} tokens x {work.concurrency} in flight is more than "
        f"{entry.name} can serve here",
        backend_kind,
        work=work,
        governing=governing,
        whose=whose,
        details={"capability": entry.name},
        ceilings=ceilings,
    )


def over_limit(
    opening: str,
    backend_kind: str,
    *,
    work: WorkingContext,
    governing: ContextCeiling,
    whose: str,
    details: dict[str, Any],
    ceilings: tuple[ContextCeiling, ...],
) -> ApiError:
    memory_half = (
        f"{governing.memory_context} that this host's memory affords at "
        f"{work.concurrency} in flight"
        if governing.memory_context is not None
        else "no memory figure (this model's block is not taken apart into terms)"
    )
    return ApiError(
        400,
        "context_over_limit",
        f"{opening}: the ceiling is {governing.tokens} tokens, computed for "
        f"{governing.model}{whose} — the smaller of {governing.served_context} "
        f"served (the most its manifest ever starts an engine with on "
        f"{backend_kind}: max_context, or context_default where none is "
        f"stated) and {memory_half}. Ask for {governing.tokens} or fewer; "
        "nothing is clamped",
        {
            **details,
            "requested": {"tokens": work.tokens, "concurrency": work.concurrency},
            "ceiling": governing.to_dict(),
            "ceilings": [ceiling.to_dict() for ceiling in ceilings],
        },
    )


def check_load_context(
    manifest: Any,
    backend_kind: str,
    *,
    available_bytes: int,
    context: int,
) -> ContextCeiling:
    candidate = Candidate.of(manifest, backend_kind)
    ceiling = candidate.context_ceiling(available_bytes, 1)
    if ceiling is None:
        raise ValueError(f"{manifest.id} is not token-shaped")
    holds_weights = (
        candidate.memory is None or candidate.memory.fixed_bytes < available_bytes
    )
    if not holds_weights or context <= ceiling.tokens:
        return ceiling
    raise over_limit(
        f"a context of {context} tokens is more than {manifest.id} can be "
        "loaded at here",
        backend_kind,
        work=WorkingContext(
            tokens=context, concurrency=1, source="load-model params.context"
        ),
        governing=ceiling,
        whose="",
        details={"model": manifest.id},
        ceilings=(ceiling,),
    )


__all__ = [
    "MIN_LOAD_CONTEXT",
    "check_ceiling",
    "check_load_context",
    "context_ceilings",
    "over_limit",
]
