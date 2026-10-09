from __future__ import annotations

from typing import Any

from . import asrplan
from .backend import CardFacts
from .capabilityclasses import BY_NAME, CLASSES, CapabilityClass, classes_for_job_type
from .capabilityrecord import desktop_reserve_words
from .capabilitywords import (
    describe_card,
    goal_phrase,
    needs_phrase,
    serving_refusal_note,
    serving_summary,
    shown_precision,
    spell_floor,
)
from .errors import ApiError
from .fit import Candidate, WorkingContext
from .memorybudget import available_bytes, gib_text
from .verdict import Decision


def _why_not_best(
    decision: Decision,
    best: Candidate,
    work: "WorkingContext | None",
    card: "CardFacts | None",
) -> str:
    missing = best.lacks(card)
    if missing:
        return f" The best, {best.id}, cannot start here: it needs {needs_phrase(missing, card)}."
    if not best.holds(work, decision.available_bytes):
        fewer = (
            " Fewer pieces at once was tried first; a smaller model is taken "
            "only when even one at a time does not fit."
            if asrplan.is_ladder(best.serving)
            else ""
        )
        return (
            f" The best, {best.id}{shown_precision(best, card) or ''}, needs "
            f"{spell_floor(best, work)} and this card gives a job "
            f"{gib_text(decision.available_bytes)}.{fewer}"
        )
    return f" {decision.selected} is the one chosen in Settings; {best.id} would also fit."


def _best(entry: CapabilityClass, decision: Decision) -> "Candidate | None":
    """The candidate the class would take on a card with room for all of them: the first
    of its pick order (CapabilityClass.pick_order), so never one above its goal."""
    ranked = entry.pick_order(decision.candidates)
    return ranked[0] if ranked else None


def _goal_words(
    entry: CapabilityClass, decision: Decision, picked: Candidate, card: "CardFacts | None"
) -> str:
    if entry.goal is None or decision.chosen:
        return ""
    usable = tuple(c for c in decision.candidates if not c.lacks(card))
    phrase = goal_phrase(
        entry, picked, entry.pick_order(usable), entry.work, decision.available_bytes, card
    )
    return f" ({phrase})"


def _class_line(
    entry: CapabilityClass, decision: Decision, card: "CardFacts | None"
) -> str:
    if not decision.enabled:
        return decision.summary[0].upper() + decision.summary[1:] + "."
    if not decision.candidates:
        return f"Can {entry.plainly}."
    by_id = {c.id: c for c in decision.candidates}
    picked = by_id.get(decision.selected)
    if picked is None:
        return f"Can {entry.plainly}, using {decision.selected}."
    line = (
        f"Will {entry.plainly} with {picked.id}{shown_precision(picked, card)}"
        f"{_goal_words(entry, decision, picked, card)}"
        f"{serving_summary(picked, decision.available_bytes)}."
    )
    best = _best(entry, decision)
    if best is not None and best.id != picked.id:
        line += _why_not_best(decision, best, entry.work, card)
    return line


def _class_row(
    entry: CapabilityClass, decision: Decision, card: "CardFacts | None"
) -> dict[str, Any]:
    picked = next(
        (c for c in decision.candidates if c.id == decision.selected), None
    )
    best = _best(entry, decision)
    return {
        "capability": entry.name,
        "enabled": decision.enabled,
        "selected": decision.selected,
        "precision": None if picked is None else picked.precision_on(card),
        "reduced_precision": False if picked is None else picked.degraded_on(card),
        "best": None if best is None else best.id,
        "goal": None if entry.goal is None else entry.goal.to_dict(),
        "lacking_features": list(decision.lacking_features),
        "line": _class_line(entry, decision, card),
    }


def _reserve_line(
    reserve_words: str | None, total_bytes: int, desktop_allowance_bytes: int | None
) -> str:
    if reserve_words is None:
        return ""
    budget = available_bytes(total_bytes, desktop_allowance_bytes or 0)
    return (
        f"{reserve_words[0].upper()}{reserve_words[1:]}, so a job gets "
        f"{gib_text(budget)}.\n"
    )


def install_plan(
    job_type: str,
    decisions: tuple[Decision, ...],
    *,
    card: "CardFacts | None",
    total_bytes: int,
    pool: str,
    desktop_allowance_bytes: int | None = None,
    desktop_basis: str | None = None,
) -> dict[str, Any]:
    entries = classes_for_job_type(job_type)
    if not entries:
        raise ApiError(
            400,
            "unknown_job_type",
            f"{job_type!r} has no capability classes; this build knows "
            f"{sorted({entry.job_type for entry in CLASSES})}",
        )
    by_name = {d.capability: d for d in decisions}
    rows = [
        _class_row(entry, by_name[entry.name], card)
        for entry in entries
        if entry.name in by_name
    ]
    usable = any(row["enabled"] for row in rows)
    card_words = describe_card(card, total_bytes, pool)
    reserve_words = (
        None
        if desktop_allowance_bytes is None or desktop_basis is None
        else desktop_reserve_words(desktop_allowance_bytes, desktop_basis)
    )
    reserve_line = _reserve_line(reserve_words, total_bytes, desktop_allowance_bytes)
    lines = "\n".join(f"- {row['line']}" for row in rows)
    closing = (
        "Install it?"
        if usable
        else (
            "Nothing it offers can run on this card. Install it anyway? It stays "
            "off until this server has a card that can run it."
        )
    )
    return {
        "job_type": job_type,
        "card": None if card is None else card.to_dict(),
        "card_words": card_words,
        "desktop_reserve": reserve_words,
        "desktop_allowance_basis": desktop_basis,
        "usable": usable,
        "classes": rows,
        "confirm": (
            f"Install {job_type}.\n\nYour card ({card_words}):\n{reserve_line}"
            f"{lines}\n\n{closing}"
        ),
    }


def _subject_line(
    entry: CapabilityClass,
    decision: Decision,
    candidate: Candidate,
    subject_id: str,
    card: "CardFacts | None",
) -> tuple[str, bool]:
    missing = candidate.lacks(card)
    if missing:
        return f"Cannot {entry.plainly} with it: it needs {needs_phrase(missing, card)}.", False
    if not candidate.holds(entry.work, decision.available_bytes):
        return (
            f"Cannot {entry.plainly} with it: it needs "
            f"{spell_floor(candidate, entry.work)} and this card gives a job "
            f"{gib_text(decision.available_bytes)}."
            + serving_refusal_note(candidate, decision.available_bytes)
        ), False
    if decision.selected == subject_id:
        return (
            f"Will {entry.plainly} with it{shown_precision(candidate, card)}"
            f"{serving_summary(candidate, decision.available_bytes)}."
        ), True
    return (
        f"Can {entry.plainly} with it{shown_precision(candidate, card)}; "
        f"{decision.selected or 'nothing'} is what this server uses for "
        "that unless it is chosen in Settings."
    ), True


def subject_plan(
    subject_id: str,
    decisions: tuple[Decision, ...],
    *,
    card: "CardFacts | None",
    total_bytes: int,
    pool: str,
) -> dict[str, Any]:
    card_words = describe_card(card, total_bytes, pool)
    lines: list[str] = []
    runs = False
    for decision in decisions:
        entry = BY_NAME.get(decision.capability)
        candidate = next((c for c in decision.candidates if c.id == subject_id), None)
        if entry is None or candidate is None:
            continue
        line, can = _subject_line(entry, decision, candidate, subject_id, card)
        lines.append(line)
        runs = runs or can
    if not lines:
        raise ApiError(
            404,
            "unknown_subject",
            f"{subject_id!r} is not offered by any capability class on this backend",
        )
    closing = "Download it?" if runs else "It cannot run on this card. Download it anyway?"
    body = "\n".join(f"- {line}" for line in lines)
    return {
        "subject": subject_id,
        "card": None if card is None else card.to_dict(),
        "card_words": card_words,
        "usable": runs,
        "lines": lines,
        "confirm": f"Download {subject_id}.\n\nYour card ({card_words}):\n{body}\n\n{closing}",
    }


__all__ = ["describe_card", "install_plan", "subject_plan"]
