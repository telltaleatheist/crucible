from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .backend import CUDA_LINUX, LLAMA_WINDOWS, MLX_DARWIN, CardFacts
from .capabilityclasses import CLASSES, CapabilityClass
from .capabilityrecord import CapabilityRecord, CapabilityRow
from .capabilitywords import (
    CPU_BUILD_REASON,
    LOCAL_ANSWER_PREFIX,
    NEEDS_WSL_REASON,
    UPSTREAM_OFFER,
    barred_note,
    feature_order,
    needs_phrase,
    precision_note,
    serving_note,
    serving_refusal_note,
    serving_refusal_summary,
    serving_summary,
    skipped_ladder_note,
    spell_chosen,
    spell_floor,
    too_old_phrase,
    with_notes,
)
from .fit import Candidate, WorkingContext
from .memorybudget import available_bytes, gib_text

POOL_NAME: dict[str, str] = {
    CUDA_LINUX: "card",
    MLX_DARWIN: "unified memory",
    LLAMA_WINDOWS: "card",
}

CPU_VENDOR = "cpu"
CPU_POOL_NAME = "system memory"

WSL_ONLY_JOB_TYPES: frozenset[str] = frozenset(
    {"tts", "asr", "align", "rvc", "denoise", "image", "audio"}
)


@dataclass(frozen=True)
class Decision:
    capability: str
    job_type: str
    enabled: bool
    selected: str
    reason: str
    summary: str
    shortfall_bytes: int
    available_bytes: int
    candidates: tuple[Candidate, ...]
    fit_count: int
    lacking_features: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "job_type": self.job_type,
            "enabled": self.enabled,
            "selected": self.selected,
            "reason": self.reason,
            "summary": self.summary,
            "shortfall_bytes": self.shortfall_bytes,
            "lacking_features": list(self.lacking_features),
            "available_bytes": self.available_bytes,
            "fit_count": self.fit_count,
            "candidates": [c.to_dict() for c in self.candidates],
        }

    def row(self) -> CapabilityRow:
        return CapabilityRow(
            capability=self.capability,
            enabled=self.enabled,
            selected=self.selected,
            reason=self.reason,
            summary=self.summary,
            shortfall_bytes=self.shortfall_bytes,
        )


def pool_name(backend_kind: str, gpu_vendor: str) -> str:
    if gpu_vendor == CPU_VENDOR:
        return CPU_POOL_NAME
    pool = POOL_NAME.get(backend_kind)
    if pool is None:
        raise ValueError(
            f"{backend_kind!r} is not a Crucible backend; the backends are "
            f"{sorted(POOL_NAME)}"
        )
    return pool


def _over_served(
    entry: CapabilityClass, candidate: Candidate, work: "WorkingContext | None"
) -> bool:
    return (
        entry.client_sized
        and work is not None
        and candidate.served_context is not None
        and work.tokens > candidate.served_context
    )


def _fits(
    entry: CapabilityClass,
    candidate: Candidate,
    work: "WorkingContext | None",
    budget: int,
) -> bool:
    return candidate.holds(work, budget) and not _over_served(entry, candidate, work)


@dataclass(frozen=True)
class _Weighing:
    entry: CapabilityClass
    backend_kind: str
    card: "CardFacts | None"
    work: "WorkingContext | None"
    budget: int
    arithmetic: str
    found: tuple[Candidate, ...]
    usable: tuple[Candidate, ...]
    barred: tuple[Candidate, ...]
    fitting: tuple[Candidate, ...]
    cpu_note: str

    @property
    def offer(self) -> str:
        return UPSTREAM_OFFER if self.entry.routable else ""


def _decision(
    entry: CapabilityClass,
    budget: int,
    found: tuple[Candidate, ...],
    fitting: tuple[Candidate, ...],
    *,
    enabled: bool,
    selected: str,
    reason: str,
    summary: str,
    shortfall_bytes: int = 0,
    lacking_features: tuple[str, ...] = (),
) -> Decision:
    return Decision(
        capability=entry.name,
        job_type=entry.job_type,
        enabled=enabled,
        selected=selected,
        reason=reason,
        summary=summary,
        shortfall_bytes=shortfall_bytes,
        available_bytes=budget,
        candidates=found,
        fit_count=len(fitting),
        lacking_features=lacking_features,
    )


def _refuse(
    entry: CapabilityClass,
    budget: int,
    found: tuple[Candidate, ...],
    fitting: tuple[Candidate, ...],
    reason: str,
    summary: str,
    **extra: Any,
) -> Decision:
    return _decision(
        entry,
        budget,
        found,
        fitting,
        enabled=False,
        selected="",
        reason=reason,
        summary=summary,
        **extra,
    )


def _grant(w: _Weighing, picked: Candidate, reason: str) -> Decision:
    return _decision(
        w.entry,
        w.budget,
        w.found,
        w.fitting,
        enabled=True,
        selected=picked.id,
        reason=reason,
        summary=f"can {w.entry.plainly}, using {picked.id}"
        + serving_summary(picked, w.budget),
    )


def _needs_wsl(entry: CapabilityClass, budget: int) -> Decision:
    return _refuse(
        entry,
        budget,
        (),
        (),
        NEEDS_WSL_REASON,
        f"cannot {entry.plainly} — this machine has no Linux engine "
        "installed yet. Finish setting it up, or use another server",
    )


def _always_available(entry: CapabilityClass, budget: int) -> Decision:
    return _decision(
        entry,
        budget,
        (),
        (),
        enabled=True,
        selected="",
        reason=f"always available: {entry.purpose}",
        summary=f"can {entry.plainly}",
    )


def _nothing_shipped(entry: CapabilityClass, backend_kind: str, budget: int) -> Decision:
    return _refuse(
        entry,
        budget,
        (),
        (),
        f"disabled: {entry.purpose} needs {entry.noun}, and this build "
        f"ships none with a {backend_kind} block",
        f"cannot {entry.plainly} — nothing that can do it runs on this "
        "machine's hardware. Another server has to take this work",
    )


def _chosen_not_shipped(w: _Weighing, chosen: str) -> Decision:
    entry = w.entry
    return _refuse(
        entry,
        w.budget,
        w.found,
        w.fitting,
        f"disabled: {chosen} was chosen for {entry.name}, and it is "
        f"not among the {len(w.found)} {entry.noun} this build ships "
        f"with a {w.backend_kind} block",
        f"cannot {entry.plainly} — it is set to use {chosen}, which "
        "this machine cannot run. Choose another in Settings",
    )


def _chosen_cannot_start(w: _Weighing, picked: Candidate, missing: tuple[str, ...]) -> Decision:
    entry = w.entry
    return _refuse(
        entry,
        w.budget,
        w.found,
        w.fitting,
        f"disabled: {picked.id} was chosen for {entry.name}; its "
        f"engine needs {needs_phrase(missing, w.card)}. "
        "It refuses to start on this card whatever the memory."
        + w.offer,
        f"cannot {entry.plainly} — it is set to use {picked.id}, and "
        f"{too_old_phrase(missing, w.card)}. Choose "
        "another in Settings",
        lacking_features=missing,
    )


def _chosen_over_served(w: _Weighing, picked: Candidate) -> Decision:
    entry, work = w.entry, w.work
    return _refuse(
        entry,
        w.budget,
        w.found,
        w.fitting,
        f"disabled: {picked.id} was chosen for {entry.name} and is "
        f"never served past {picked.served_context} tokens on "
        f"{w.backend_kind} (its manifest's max_context), which is "
        f"less than the {work.tokens} tokens this work asks for",
        f"cannot {entry.plainly} — {picked.id} cannot take requests "
        f"of {work.tokens} tokens on this machine",
    )


def _chosen_too_big(w: _Weighing, picked: Candidate) -> Decision:
    entry = w.entry
    shortfall = picked.floor_bytes(w.work) - w.budget
    return _refuse(
        entry,
        w.budget,
        w.found,
        w.fitting,
        f"disabled: {picked.id} was chosen for {entry.name} and needs "
        f"{spell_floor(picked, w.work)}, and there is only "
        f"{w.arithmetic} — short by {gib_text(shortfall)}. This choice fit "
        f"the machine it was made on{w.cpu_note}."
        + w.offer,
        f"cannot {entry.plainly} — {picked.id} needs "
        f"{gib_text(shortfall)} more memory than this machine has free. "
        "A smaller choice, or another server",
        shortfall_bytes=shortfall,
    )


def _chosen_granted(w: _Weighing, picked: Candidate) -> Decision:
    entry = w.entry
    return _grant(
        w,
        picked,
        with_notes(
            f"{picked.id} was chosen for {entry.name}: it needs "
            f"{spell_chosen(picked, w.work, w.budget)} and there is {w.arithmetic}; "
            f"{len(w.fitting)} of {len(w.found)} {entry.noun} fit{w.cpu_note}",
            precision_note(picked, w.card),
            serving_note(picked, w.budget),
        ),
    )


def _decide_chosen(w: _Weighing, chosen: str) -> Decision:
    picked = next((c for c in w.found if c.id == chosen), None)
    if picked is None:
        return _chosen_not_shipped(w, chosen)
    missing = picked.lacks(w.card)
    if missing:
        return _chosen_cannot_start(w, picked, missing)
    if _over_served(w.entry, picked, w.work):
        return _chosen_over_served(w, picked)
    if not picked.holds(w.work, w.budget):
        return _chosen_too_big(w, picked)
    return _chosen_granted(w, picked)


def _best_fit(w: _Weighing) -> Decision:
    best = w.fitting[0]
    return _grant(
        w,
        best,
        with_notes(
            f"{best.id} fits: it needs {spell_chosen(best, w.work, w.budget)} and "
            f"there is {w.arithmetic}; {len(w.fitting)} of {len(w.found)} "
            f"{w.entry.noun} fit{w.cpu_note}",
            barred_note(w.barred, w.card),
            skipped_ladder_note(w.usable, best, w.budget),
            precision_note(best, w.card),
            serving_note(best, w.budget),
        ),
    )


def _none_can_start(w: _Weighing) -> Decision:
    entry, found = w.entry, w.found
    missing = feature_order({f for c in w.barred for f in c.lacks(w.card)})
    return _refuse(
        entry,
        w.budget,
        found,
        w.fitting,
        "disabled: "
        + ("the only candidate" if len(found) == 1 else f"all {len(found)} candidates")
        + f" this build ships for {entry.name} on {w.backend_kind} — "
        f"{', '.join(c.id for c in found)} — "
        f"{'needs' if len(found) == 1 else 'need'} "
        f"{needs_phrase(missing, w.card)}. The engine "
        "refuses to start on this card whatever the memory, so this "
        "is not a sizing choice."
        + w.offer,
        f"cannot {entry.plainly} — {too_old_phrase(missing, w.card)}",
        lacking_features=missing,
    )


def _none_fits(w: _Weighing) -> Decision:
    entry = w.entry
    smallest = w.usable[-1]
    shortfall = smallest.floor_bytes(w.work) - w.budget
    note = f" {entry.binary_note}" if entry.binary_note else ""
    note = serving_refusal_note(smallest, w.budget) + note
    of_these = (
        f"{len(w.found)} {entry.noun}"
        if not w.barred
        else f"{len(w.usable)} {entry.noun} this card can start"
    )
    return _refuse(
        entry,
        w.budget,
        w.found,
        w.fitting,
        f"disabled: the smallest of {of_these} is {smallest.id} "
        f"at {spell_floor(smallest, w.work)} and there is only "
        f"{w.arithmetic} — short by {gib_text(shortfall)}."
        f"{barred_note(w.barred, w.card)}{note}"
        + w.offer,
        serving_refusal_summary(entry, smallest, w.budget)
        or (
            f"cannot {entry.plainly} — the smallest option needs "
            f"{gib_text(shortfall)} more memory than this machine has free"
        ),
        shortfall_bytes=shortfall,
    )


def _work_for(
    entry: CapabilityClass, work: "WorkingContext | None"
) -> "WorkingContext | None":
    if work is None:
        return entry.work
    if not entry.client_sized:
        raise ValueError(
            f"{entry.name} is not client-sized; its working context is its own "
            f"ruling ({entry.work.source if entry.work else 'none'}), and a "
            "caller may not restate it"
        )
    return work


def _weigh(
    entry: CapabilityClass,
    backend_kind: str,
    found: tuple[Candidate, ...],
    *,
    total_bytes: int,
    desktop_allowance_bytes: int,
    gpu_vendor: str,
    pool: str,
    work: "WorkingContext | None",
    card: "CardFacts | None",
) -> _Weighing:
    budget = available_bytes(total_bytes, desktop_allowance_bytes)
    usable = tuple(c for c in found if not c.lacks(card))
    return _Weighing(
        entry=entry,
        backend_kind=backend_kind,
        card=card,
        work=work,
        budget=budget,
        arithmetic=(
            f"{gib_text(budget)} available ({gib_text(total_bytes)} {pool} less a "
            f"{gib_text(desktop_allowance_bytes)} desktop allowance)"
        ),
        found=found,
        usable=usable,
        barred=tuple(c for c in found if c.lacks(card)),
        fitting=tuple(c for c in usable if _fits(entry, c, work, budget)),
        cpu_note=(
            f" {CPU_BUILD_REASON}."
            if backend_kind == LLAMA_WINDOWS and gpu_vendor == CPU_VENDOR
            else ""
        ),
    )


def decide_capabilities(
    entry: CapabilityClass,
    backend_kind: str,
    *,
    total_bytes: int,
    desktop_allowance_bytes: int,
    gpu_vendor: str,
    chosen: str | None,
    work: "WorkingContext | None" = None,
    card: "CardFacts | None" = None,
) -> Decision:
    work = _work_for(entry, work)
    budget = available_bytes(total_bytes, desktop_allowance_bytes)
    if backend_kind == LLAMA_WINDOWS and entry.job_type in WSL_ONLY_JOB_TYPES:
        return _needs_wsl(entry, budget)
    pool = pool_name(backend_kind, gpu_vendor)
    if entry.candidates is None:
        return _always_available(entry, budget)
    found = entry.candidates(backend_kind)
    if not found:
        return _nothing_shipped(entry, backend_kind, budget)
    weighing = _weigh(
        entry,
        backend_kind,
        found,
        total_bytes=total_bytes,
        desktop_allowance_bytes=desktop_allowance_bytes,
        gpu_vendor=gpu_vendor,
        pool=pool,
        work=work,
        card=card,
    )
    if chosen is not None:
        return _decide_chosen(weighing, chosen)
    if weighing.fitting:
        return _best_fit(weighing)
    if not weighing.usable:
        return _none_can_start(weighing)
    return _none_fits(weighing)


decide = decide_capabilities


def decide_all(
    backend_kind: str,
    *,
    total_bytes: int,
    desktop_allowance_bytes: int,
    gpu_vendor: str,
    chosen: Mapping[str, str],
    card: "CardFacts | None" = None,
) -> tuple[Decision, ...]:
    return tuple(
        decide_capabilities(
            entry,
            backend_kind,
            total_bytes=total_bytes,
            desktop_allowance_bytes=desktop_allowance_bytes,
            gpu_vendor=gpu_vendor,
            chosen=chosen.get(entry.name),
            card=card,
        )
        for entry in CLASSES
    )


def routed_row(row: CapabilityRow, model: str) -> CapabilityRow:
    return CapabilityRow(
        capability=row.capability,
        enabled=True,
        selected=model,
        reason=(
            f"routed to {model.partition('/')[0]}; "
            f"{LOCAL_ANSWER_PREFIX}{row.reason}"
        ),
        summary=f"sends this work to {model.partition('/')[0]}",
        shortfall_bytes=0,
    )


def record(
    backend_kind: str,
    *,
    total_bytes: int,
    desktop_allowance_bytes: int,
    decisions: tuple[Decision, ...],
    routes: dict[str, str],
) -> CapabilityRecord:
    rows = []
    for decision in decisions:
        row = decision.row()
        model = routes.get(decision.capability)
        rows.append(row if model is None else routed_row(row, model))
    return CapabilityRecord(
        backend_kind=backend_kind,
        total_bytes=total_bytes,
        desktop_allowance_bytes=desktop_allowance_bytes,
        rows=tuple(rows),
    )


def job_type_enabled(job_type: str, decisions: tuple[Decision, ...]) -> bool:
    mine = [d for d in decisions if d.job_type == job_type]
    if not mine:
        raise ValueError(
            f"no capability class feeds {job_type!r}; CLASSES covers "
            f"{sorted({entry.job_type for entry in CLASSES})}"
        )
    return any(d.enabled for d in mine)


__all__ = [
    "CPU_POOL_NAME",
    "CPU_VENDOR",
    "Decision",
    "POOL_NAME",
    "WSL_ONLY_JOB_TYPES",
    "decide",
    "decide_all",
    "decide_capabilities",
    "job_type_enabled",
    "pool_name",
    "record",
    "routed_row",
]
