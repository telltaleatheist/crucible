from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..atomicjson import write_json
from ..platform.errors import HostError
from ..platform.wsl_table import WSL_OUTCOME_NAME, WSL_STATES
from .state import MoveState

OUTCOME_NAME = WSL_OUTCOME_NAME

DONE = MoveState.DONE
REBOOT_PENDING = MoveState.REBOOT_PENDING
CANNOT = MoveState.CANNOT
FAILED = MoveState.FAILED
DECLINED = MoveState.DECLINED
STATES: tuple[str, ...] = tuple(state.value for state in MoveState)

FAILED_ATTEMPT_CEILING = 2

REBOOT_REQUIRED_CODE = "wsl_reboot_required"

REBOOT_STILL_OWED_CODE = "wsl_reboot_still_owed"

REBOOT_BUDGET_SPENT_CODE = "wsl_reboot_again"

REBOOT_CODES: frozenset[str] = frozenset({REBOOT_REQUIRED_CODE, REBOOT_STILL_OWED_CODE})

TRANSIENT_CANNOT_CODES: frozenset[str] = frozenset({REBOOT_BUDGET_SPENT_CODE})

RESTART_BANNER_CODES: frozenset[str] = REBOOT_CODES | TRANSIENT_CANNOT_CODES

FIRMWARE_CANNOT_CODES: frozenset[str] = frozenset({"virtualization_disabled"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class Outcome:
    state: MoveState
    code: str | None
    sentence: str | None
    at: str
    release: str
    attempts: int
    restarts: int

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "code": self.code,
            "sentence": self.sentence,
            "at": self.at,
            "release": self.release,
            "attempts": self.attempts,
            "restarts": self.restarts,
        }

    def at_epoch(self) -> float | None:
        try:
            return datetime.fromisoformat(self.at).timestamp()
        except ValueError:
            return None


def path(home: Path) -> Path:
    return Path(home) / OUTCOME_NAME


def classify(code: str) -> MoveState:
    if code in REBOOT_CODES:
        return REBOOT_PENDING
    if code in TRANSIENT_CANNOT_CODES:
        return CANNOT
    for row in WSL_STATES:
        if row.code == code:
            return CANNOT if not row.automatic else FAILED
    return FAILED


def read(home: Path) -> Outcome | None:
    file = path(home)
    if not file.is_file():
        return None
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HostError(
            "wsl_outcome_invalid",
            f"{file} exists and is not the JSON document docs/internals/host-and-platform.md, \"Outcome file\", describes "
            f"({exc}). It records what happened to this machine's move.",
        ) from exc
    if not isinstance(raw, dict):
        raise HostError(
            "wsl_outcome_invalid",
            f"{file} holds a {type(raw).__name__} and not an object.",
        )
    state = raw.get("state")
    if state not in STATES:
        raise HostError(
            "wsl_outcome_invalid",
            f"{file} says state={state!r}; the states are {list(STATES)}.",
        )
    attempts = raw.get("attempts")
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 0:
        raise HostError(
            "wsl_outcome_invalid",
            f"{file} says attempts={attempts!r}, which is not a count. The tray "
            "retries a `failed` once and reads that number to know which try "
            "this is.",
        )
    release = raw.get("release")
    if not isinstance(release, str) or release == "":
        raise HostError(
            "wsl_outcome_invalid",
            f"{file} names no release. An outcome is about the move to one "
            "release, and one that names none cannot be compared with this "
            "host's.",
        )
    at = raw.get("at")
    if not isinstance(at, str) or at == "":
        raise HostError("wsl_outcome_invalid", f"{file} has no `at` timestamp.")
    code = raw.get("code")
    sentence = raw.get("sentence")
    if code is not None and not isinstance(code, str):
        raise HostError("wsl_outcome_invalid", f"{file} says code={code!r}.")
    if sentence is not None and not isinstance(sentence, str):
        raise HostError("wsl_outcome_invalid", f"{file} says sentence={sentence!r}.")
    restarts = raw.get("restarts")
    if not isinstance(restarts, int) or isinstance(restarts, bool) or restarts < 0:
        raise HostError("wsl_outcome_invalid", f"{file} says restarts={restarts!r}.")
    return Outcome(
        state=MoveState(state),
        code=code,
        sentence=sentence,
        at=at,
        release=release,
        attempts=attempts,
        restarts=restarts,
    )


def read_or_quarantine(home: Path, log: Callable[[str], None]) -> Outcome | None:
    try:
        return read(home)
    except HostError as exc:
        from ..platform.quarantine import quarantine

        aside = quarantine(path(home))
        log(
            f"outcome: {exc.message} It was moved to {aside} and the controller "
            "decides again from nothing, as if no move had been recorded."
        )
        return None


def write(
    home: Path,
    *,
    state: MoveState | str,
    release: str,
    code: str | None = None,
    sentence: str | None = None,
    attempts: int,
    restarts: int = 0,
    now: Callable[[], str] = _utc_now,
) -> Outcome:
    if state not in STATES:
        raise HostError(
            "wsl_outcome_invalid",
            f"{state!r} is not one of {list(STATES)}; a state nobody defined is "
            "not a thing to write into the file the tray decides from.",
        )
    if release == "":
        raise HostError(
            "wsl_outcome_invalid",
            "an outcome names the release its move was for, and this one names none.",
        )
    outcome = Outcome(
        state=MoveState(state),
        code=code,
        sentence=sentence,
        at=now(),
        release=release,
        attempts=attempts,
        restarts=restarts,
    )
    write_json(path(home), outcome.to_dict())
    return outcome
