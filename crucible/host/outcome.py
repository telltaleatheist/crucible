"""`wsl-outcome.json` — what happened to the move on this machine.

PHASE19-AUTOMATIC-WSL.md 2.2. ONE OWNER of the sentence *"what happened to the
move on this machine"*: this module writes the file, reads it back, and is the
only place its five states are spelled. The tray reads it at start to decide
whether to move (2.3), the door answers `GET /install` out of it (2.6), and
`install.ps1` reads it to choose its closing sentence (2.7).

IT REPLACES THE `wsl-reboot-pending` MARKER, which was a zero-byte fact with a
release string in it and no room for the other four endings. A marker that says
"we stopped for a reboot" and a file that says "we stopped, here is why and
when and for which release" are the same fact at two levels of detail, and
ARCHITECTURE.md R1 does not let a fact have two owners — so the marker is gone
rather than kept beside this.

WHY A FILE AND NOT THE PRESENCE. `owner: wsl-unit` on the presence is the LIVE
fact and stays the live fact; this is the HISTORY, and the two questions it
answers are ones no presence can: "did this machine try and find out it
cannot", and "did it already fail once". Both have to survive a reboot and a
logout, which is what makes them a file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .errors import HostError
from .wsl_states import WSL_OUTCOME_NAME, WSL_STATES

#: In the host home, beside `config.toml` and `pairing` (PHASE15 3.6).
#:
#: GENERATED, from `sdk/bootstrap/src/distro.ts`, because `install.ps1` reads
#: the same file for its closing sentence (2.7) and that script is generated
#: from the same place. A name spelled here and again in a shell script is a
#: file two programs can disagree about the location of.
OUTCOME_NAME = WSL_OUTCOME_NAME

#: The five endings of 2.2, and there is no sixth. `done` and `declined` carry
#: no code and no sentence; the other three always carry both.
DONE = "done"
REBOOT_PENDING = "reboot-pending"
CANNOT = "cannot"
FAILED = "failed"
DECLINED = "declined"
STATES: tuple[str, ...] = (DONE, REBOOT_PENDING, CANNOT, FAILED, DECLINED)

#: 2.2: a `failed` is retried by the tray at its NEXT start, once. A second
#: consecutive `failed` stays `failed` until a person presses Try again (2.5).
#: The number is the plan's — "retried by the tray at its next start, once" —
#: and it is a ceiling on ATTEMPTS, so the second attempt is the last.
FAILED_ATTEMPT_CEILING = 2

#: Codes that are `cannot` without being a row of the 4c table.
#:
#: `wsl_reboot_again` is 2.4's: `wsl --install` was run, Windows was restarted,
#: and `wsl --status` still asks for a restart. There is no probe for that —
#: it is the same state seen twice — so it is a code of its own rather than a
#: row, and it is terminal for the tray because a machine that asks twice is
#: one a person has to look at.
CANNOT_CODES: frozenset[str] = frozenset({"wsl_reboot_again"})

#: The code `installer.py` raises when it stops for the restart `wsl --install`
#: demands. Named here because the classifier is what turns it into a state.
REBOOT_CODE = "wsl_reboot_required"


def _utc_now() -> str:
    """ISO-8601, UTC, to the second. The `at` of 2.2's shape."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class Outcome:
    """2.2's document, field for field."""

    state: str
    code: str | None
    sentence: str | None
    at: str
    release: str
    attempts: int

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "code": self.code,
            "sentence": self.sentence,
            "at": self.at,
            "release": self.release,
            "attempts": self.attempts,
        }


def path(home: Path) -> Path:
    return Path(home) / OUTCOME_NAME


def classify(code: str) -> str:
    """Which of 2.2's states a refusal code IS.

    DERIVED FROM THE TABLE, not from a list of codes kept beside it. Section 1
    partitions the 4c rows into *can* and *cannot*, `wsl-states.ts` carries that
    partition as `automatic` (2.1), and a refusal naming a row the tray cannot
    carry is exactly a `cannot`. Everything else — a download that died, an
    import that timed out, `install.sh` exiting non-zero, a repair that ran and
    changed nothing — is a `failed`, which is the one the tray retries.
    """
    if code == REBOOT_CODE:
        return REBOOT_PENDING
    if code in CANNOT_CODES:
        return CANNOT
    for row in WSL_STATES:
        if row.code == code:
            return CANNOT if not row.automatic else FAILED
    return FAILED


def read(home: Path) -> Outcome | None:
    """The outcome, or None when this machine has never recorded one.

    A file that is PRESENT and unreadable is refused by name and never read as
    absent: "nothing has happened here yet" and "something happened and this
    build cannot tell what" lead to opposite decisions in 2.3, and guessing the
    first would start a move on a machine that already said it cannot.
    """
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
    return Outcome(
        state=state,
        code=code,
        sentence=sentence,
        at=at,
        release=release,
        attempts=attempts,
    )


def write(
    home: Path,
    *,
    state: str,
    release: str,
    code: str | None = None,
    sentence: str | None = None,
    attempts: int,
    now: Callable[[], str] = _utc_now,
) -> Outcome:
    """Record one terminal point. Written whole, then moved into place.

    `attempts` is the CALLER's, because the caller is the thing that knows
    which try this is — 2.3 reads the previous outcome to decide whether to run
    at all, and a writer that incremented for itself would be counting a
    different number from the one the decision was made on.
    """
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
    )
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    staged = path(home).with_suffix(".tmp")
    staged.write_text(json.dumps(outcome.to_dict(), indent=2) + "\n", encoding="utf-8")
    staged.replace(path(home))
    return outcome
