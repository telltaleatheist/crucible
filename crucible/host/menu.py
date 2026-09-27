from __future__ import annotations

from dataclasses import dataclass

from .state import Distro, Engine, MoveState, Owner

TRY_AGAIN = "try-again"
RESTART_OWED = "restart-owed"

TRY_AGAIN_STATES = frozenset({MoveState.CANNOT, MoveState.FAILED})

TRY_AGAIN_LABEL = "Try again: set up the Linux engine"
RESTART_OWED_LABEL = "Needs Update and restart (to install WSL), then sign in"


@dataclass(frozen=True)
class MenuItem:
    item_id: str
    label: str
    enabled: bool


def outcome_items(outcome_state: MoveState | str | None, *, busy: bool) -> list[MenuItem]:
    if outcome_state == MoveState.REBOOT_PENDING:
        return [MenuItem(RESTART_OWED, RESTART_OWED_LABEL, False)]
    if outcome_state in TRY_AGAIN_STATES:
        return [MenuItem(TRY_AGAIN, TRY_AGAIN_LABEL, not busy)]
    return []


__all__ = [
    "Distro", "Engine", "MenuItem", "Owner", "RESTART_OWED", "RESTART_OWED_LABEL",
    "TRY_AGAIN", "TRY_AGAIN_LABEL", "TRY_AGAIN_STATES", "outcome_items",
]
