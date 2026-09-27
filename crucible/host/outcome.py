from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .errors import HostError
from .wsl_states import WSL_OUTCOME_NAME, WSL_STATES

OUTCOME_NAME = WSL_OUTCOME_NAME

DONE = "done"
REBOOT_PENDING = "reboot-pending"
CANNOT = "cannot"
FAILED = "failed"
DECLINED = "declined"
STATES: tuple[str, ...] = (DONE, REBOOT_PENDING, CANNOT, FAILED, DECLINED)

FAILED_ATTEMPT_CEILING = 2

CANNOT_CODES: frozenset[str] = frozenset({"wsl_reboot_again"})

REBOOT_CODE = "wsl_reboot_required"

REBOOT_AGAIN_CODE = "wsl_reboot_still_owed"

REBOOT_CODES: frozenset[str] = frozenset({REBOOT_CODE, REBOOT_AGAIN_CODE})

TRANSIENT_CANNOT_CODES: frozenset[str] = frozenset({"wsl_reboot_again"})

FIRMWARE_CANNOT_CODES: frozenset[str] = frozenset({"virtualization_disabled"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class Outcome:
    state: str
    code: str | None
    sentence: str | None
    at: str
    release: str
    attempts: int
    restarts: int

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
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


def classify(code: str) -> str:
    if code in REBOOT_CODES:
        return REBOOT_PENDING
    if code in CANNOT_CODES:
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
            f"{file} exists and is not the JSON document PHASE19 2.2 describes "
            f"({exc}). It records what happened to this machine's move; delete "
            "it to let the orchestrator decide again from nothing.",
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
        state=state,
        code=code,
        sentence=sentence,
        at=at,
        release=release,
        attempts=attempts,
        restarts=restarts,
    )


def write(
    home: Path,
    *,
    state: str,
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
        state=state,
        code=code,
        sentence=sentence,
        at=now(),
        release=release,
        attempts=attempts,
        restarts=restarts,
    )
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    staged = path(home).with_suffix(".tmp")
    staged.write_text(json.dumps(outcome.to_dict(), indent=2) + "\n", encoding="utf-8")
    staged.replace(path(home))
    return outcome
