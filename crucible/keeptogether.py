"""A model that keeps its queued calls together (Owen, 2026-10-10: "keep yue's calls
together").

On Victoria's laptop B-Sides queued a batch of YuE2 songs, and a CLI ``load-model`` sent
mid-batch took its first-come place between two of them: YuE2 (minutes to load) would
have come off the card for the load and gone back on for the next song. A model whose
manifest says ``keep_calls_together = true`` (the resident carries it as
``keeps_calls_together``) is kept instead:

- **Back to back.** While it is resident, the queued calls that run on it (jobs that make
  it resident, chats for it when it is a model) go ahead of the first queued item that
  would take it off the card: a load or a job of anything else, a chat or a session for
  another model, an unload of its kind. Items that never touch the card (``echo``, the
  unload of another kind) keep their place and take nothing off it.
- **The settlement keeps it.** A queued call on it that runs before anything that would
  take it off holds the card (crucible/settle.py), so it is not unloaded between songs.
- **The bound.** The item kept waiting takes its *turn* once: the first time it would be
  next (nothing that arrived before it still waits) and the lane is free. The calls on
  the model waiting at that moment are its ``Kept.behind``, and only they go ahead of it;
  a call that arrives later waits behind it. Until that turn every call on the model goes
  ahead of it (they would all have run while it waited for the items before it anyway).
  So it waits for at most the calls queued when its turn came, each run once: never for
  a stream of new ones.
- **Only while it is resident.** If the model leaves the card some other way, nothing is
  kept and the line is first come, first served again.

The open queue session's own items are left in the order its client sent them: the
session already runs them back to back with nothing from anyone else in between.

The item kept waiting says why (``waiting_for`` with code ``keeping_calls_together``, a
``waiting`` event; crucible/jobs/line.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .cardkinds import KIND_LLM
from .jobtypes import CARD_EFFECTS

KEEPING_CALLS_TOGETHER = "keeping_calls_together"

NEVER_TOUCH_THE_CARD = frozenset({"echo"})
"""Job types that neither load nor unload anything nor use the card's memory."""


@dataclass(frozen=True)
class Kept:
    """The turn an item that would take ``model`` off the card took: the calls on the
    model that were waiting then (``behind``, by id) go ahead of it, and no others."""

    model: str
    behind: frozenset[str]


@dataclass(frozen=True)
class Hold:
    """An item kept waiting behind the calls on the resident model that go first."""

    item: Any
    model: str
    ahead: tuple[Any, ...]
    turn_taken: bool


def kept_resident(resident: Any) -> Any | None:
    """The resident when its manifest asked to keep its calls together, else None."""
    if resident is None or not resident.keeps_calls_together:
        return None
    return resident


def runs_on(item: Any, resident: Any) -> bool:
    """Whether a waiting item runs on the resident as it is, without changing the card."""
    job = item.job
    if item.is_session:
        return False
    if item.is_call:
        return resident.kind == KIND_LLM and job.model == resident.id
    effect = CARD_EFFECTS.get(job.type)
    return (
        effect is not None
        and effect.makes_resident == resident.kind
        and job.model == resident.id
    )


def takes_it_off(item: Any, resident: Any) -> bool:
    """Whether a waiting item, run now, would take the resident off the card: by loading
    something else over it, unloading it, or needing the card's memory for itself."""
    if runs_on(item, resident):
        return False
    job = item.job
    if item.is_session:
        # A session that names a model opens by loading it; one that names none opens
        # on whatever is resident.
        return job.model is not None and not (
            resident.kind == KIND_LLM and job.model == resident.id
        )
    if item.is_call:
        return True
    if job.type in NEVER_TOUCH_THE_CARD:
        return False
    effect = CARD_EFFECTS.get(job.type)
    if effect is None:
        return True
    if effect.takes_off is not None:
        return effect.takes_off == resident.kind
    return True


def _turn(item: Any, resident: Any) -> Kept | None:
    kept = item.kept
    return kept if kept is not None and kept.model == resident.id else None


def order(items: Sequence[Any], resident: Any) -> tuple[list[Any], Hold | None]:
    """``items`` (first come, first served) with the calls on a resident that keeps its
    calls together moved ahead of the first item that would take it off the card, and
    that item's hold, if any call goes ahead of it."""
    resident = kept_resident(resident)
    listed = list(items)
    if resident is None:
        return listed, None
    for index, item in enumerate(listed):
        if takes_it_off(item, resident):
            break
    else:
        return listed, None
    turn = _turn(item, resident)
    behind = listed[index + 1:]
    ahead = [
        other for other in behind
        if runs_on(other, resident) and (turn is None or other.job.id in turn.behind)
    ]
    if not ahead:
        return listed, None
    going = {id(other) for other in ahead}
    rest = [other for other in behind if id(other) not in going]
    hold = Hold(item=item, model=resident.id, ahead=tuple(ahead), turn_taken=turn is not None)
    return listed[:index] + ahead + [item] + rest, hold


def take_turn(items: Sequence[Any], resident: Any) -> bool:
    """Called when the lane is free. When the first item to arrive would take a resident
    that keeps its calls together off the card and has not had its turn, it takes it
    now: the calls on the model waiting at this moment are the only ones that go ahead
    of it. True when a turn was taken (the line's order may have changed)."""
    resident = kept_resident(resident)
    if resident is None or not items:
        return False
    first = items[0]
    if not takes_it_off(first, resident) or _turn(first, resident) is not None:
        return False
    first.kept = Kept(
        model=resident.id,
        behind=frozenset(
            other.job.id for other in items[1:] if runs_on(other, resident)
        ),
    )
    return True


def runs_next_on(ordered: Sequence[Any], resident: Any) -> int:
    """How many queued calls run on a resident that keeps its calls together before
    anything takes it off the card, in the line's order: what the settlement keeps it
    for. Zero when the next item to change the card would take it off."""
    resident = kept_resident(resident)
    if resident is None:
        return 0
    count = 0
    for item in ordered:
        if runs_on(item, resident):
            count += 1
        elif takes_it_off(item, resident):
            break
    return count
