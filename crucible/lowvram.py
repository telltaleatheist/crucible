"""`[audio] low_vram`: what this card makes of it, and the one rule that turns it on.

A friend's 8 GiB RTX 3070 laptop, 2026-10-08: songs needed `[audio] low_vram`, and it had
to be turned on in a config file inside a WSL distro its owner did not know existed; nothing
offered it. Owen, the same day: it is "a configuration for systems with low ram, not for
high ram systems like this pc". So Crucible decides it from the card, and only where the
card needs it:

- `needed`: a model that declares a low-VRAM figure does not fit whole in what this card
  gives a job, and does fit at that figure. Crucible turns it on.
- `not_needed`: every such model fits whole. Crucible leaves it off (Owen's 3090 Ti).
- `too_small`: none fits even at its low-VRAM figure, so the setting would change nothing.
- `not_offered`: no model on this backend declares a low-VRAM figure (the Mac).

Who decided it is recorded beside it (config.AudioLowVram): a person's on or off is
never changed here. Crucible's own is decided again every time the card is, so a bigger
card turns it back off. This module is the one place that rule lives; the capability
store applies it whenever it decides or records the card, and every other reader acts
on the value it wrote.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .audiomodels import LOW_VRAM_SETTING
from .capabilityclasses import CLASSES
from .config import AudioLowVram, Config
from .memorybudget import available_bytes, gib_text

NEEDED = "needed"
NOT_NEEDED = "not_needed"
TOO_SMALL = "too_small"
NOT_OFFERED = "not_offered"

ON = "on"
OFF = "off"
AUTO = "auto"
STATES: tuple[str, ...] = (ON, OFF, AUTO)

TURN_OFF = "crucible audio low-vram off"
TURN_ON = "crucible audio low-vram on"
LET_CRUCIBLE = "crucible audio low-vram auto"
RECORD = "crucible capability --write"


@dataclass(frozen=True)
class Splittable:
    """One audio model on this backend that declares a low-VRAM figure."""

    id: str
    whole_bytes: int
    low_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "whole_bytes": self.whole_bytes, "low_bytes": self.low_bytes}


@dataclass(frozen=True)
class CardNeed:
    """What this card makes of `[audio] low_vram`."""

    verdict: str
    budget: int
    models: tuple[Splittable, ...]
    needing: tuple[Splittable, ...]

    @property
    def words(self) -> str:
        if self.verdict == NOT_OFFERED:
            return "no audio model on this machine can be held part at a time"
        if self.verdict == NEEDED:
            return "this card needs it: " + "; ".join(
                f"{m.id} needs {gib_text(m.whole_bytes)} whole and "
                f"{gib_text(m.low_bytes)} with it, and this card gives a job "
                f"{gib_text(self.budget)}"
                for m in self.needing
            )
        if self.verdict == NOT_NEEDED:
            return "this card does not need it: " + "; ".join(
                f"{m.id} fits whole ({gib_text(m.whole_bytes)} of "
                f"{gib_text(self.budget)})"
                for m in self.models
            )
        return "it would not help on this card: " + "; ".join(
            f"{m.id} needs {gib_text(m.low_bytes)} even with it, and this card "
            f"gives a job {gib_text(self.budget)}"
            for m in self.models
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "available_bytes": self.budget,
            "models": [m.to_dict() for m in self.models],
            "needing": [m.id for m in self.needing],
            "words": self.words,
        }


def splittable(backend_kind: str) -> tuple[Splittable, ...]:
    """The audio models this build ships for this backend that declare a low-VRAM figure."""
    found: dict[str, Splittable] = {}
    for entry in CLASSES:
        if entry.job_type != "audio" or entry.candidates is None:
            continue
        for candidate in entry.candidates(backend_kind):
            if candidate.low_vram_bytes is None or candidate.id in found:
                continue
            found[candidate.id] = Splittable(
                candidate.id, candidate.memory_bytes_estimate, candidate.low_vram_bytes
            )
    return tuple(found[key] for key in sorted(found))


def card_need(
    backend_kind: str, *, total_bytes: int, desktop_allowance_bytes: int
) -> CardNeed:
    """Memory only: a card that cannot start a model at all is the verdict's to say."""
    budget = available_bytes(total_bytes, desktop_allowance_bytes)
    models = splittable(backend_kind)
    needing = tuple(m for m in models if m.whole_bytes > budget >= m.low_bytes)
    if not models:
        verdict = NOT_OFFERED
    elif needing:
        verdict = NEEDED
    elif any(m.whole_bytes <= budget for m in models):
        verdict = NOT_NEEDED
    else:
        verdict = TOO_SMALL
    return CardNeed(verdict, budget, models, needing)


@dataclass(frozen=True)
class LowVram:
    """`[audio] low_vram` on this host: the value to decide and record with, who owns
    it, what the card makes of it, and what the config file says now."""

    on: bool
    auto: bool
    need: CardNeed
    recorded: AudioLowVram

    @property
    def setting(self) -> AudioLowVram:
        return AudioLowVram(on=self.on, auto=self.auto)

    @property
    def changed(self) -> bool:
        """Crucible's own decision differs from what the file says: writing the card's
        record writes this too."""
        return self.setting != self.recorded

    @property
    def state(self) -> str:
        return AUTO if self.auto else (ON if self.on else OFF)

    @property
    def change_sentence(self) -> str:
        """The one plain sentence for a change Crucible made itself."""
        assert self.changed and self.auto
        if self.on:
            needs = "; ".join(
                f"{m.id} needs {gib_text(m.whole_bytes)} whole and this card gives a "
                f"job {gib_text(self.need.budget)}, so it now holds only the part each "
                f"stage uses ({gib_text(m.low_bytes)})"
                for m in self.need.needing
            )
            return (
                f"Crucible turned {LOW_VRAM_SETTING} on: {needs}; the audio is the "
                f"same, and `{TURN_OFF}` turns it off"
            )
        return (
            f"Crucible turned {LOW_VRAM_SETTING} off, which it had turned on itself: "
            f"{self.need.words}"
        )

    @property
    def words(self) -> str:
        """The state in one line, for doctor, the CLI and Settings."""
        if self.changed and self.auto:
            return (
                f"{LOW_VRAM_SETTING} is recorded {'on' if self.recorded.on else 'off'}, "
                f"and Crucible would turn it {'on' if self.on else 'off'} "
                f"({self.need.words}); `{RECORD}` does that"
            )
        if self.auto and self.on:
            return (
                f"{LOW_VRAM_SETTING} is on, because {self.need.words}. Crucible turned "
                f"it on and decides it again with the card; `{TURN_OFF}` turns it off "
                "for good"
            )
        if self.auto:
            return (
                f"{LOW_VRAM_SETTING} is off, and Crucible decides it: {self.need.words}"
            )
        said = "on" if self.on else "off"
        nudge = ""
        if not self.on and self.need.verdict == NEEDED:
            nudge = f"; `{TURN_ON}` or `{LET_CRUCIBLE}` turns it back on"
        elif self.on and self.need.verdict == NOT_NEEDED:
            nudge = (
                f"; this card can hold it whole, and `{LET_CRUCIBLE}` lets Crucible "
                "turn it off"
            )
        return (
            f"{LOW_VRAM_SETTING} is {said}, set by a person, and Crucible leaves it as it "
            f"is ({self.need.words}){nudge}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            # What the audio job loads with now; `pending` says Crucible's own decision
            # for this card is not written yet, and `words` what writing it would do.
            "on": self.recorded.on,
            "state": self.state,
            "set_by": "crucible" if self.auto else "person",
            "recorded": {"on": self.recorded.on, "auto": self.recorded.auto},
            "pending": self.changed,
            "card": self.need.to_dict(),
            "words": self.words,
        }


def decide(recorded: AudioLowVram, need: CardNeed) -> LowVram:
    """The rule. A person's setting stands; Crucible's is on exactly where the card
    needs it."""
    if not recorded.auto:
        return LowVram(on=recorded.on, auto=False, need=need, recorded=recorded)
    return LowVram(on=need.verdict == NEEDED, auto=True, need=need, recorded=recorded)


def recorded_of(config: Config) -> AudioLowVram:
    return AudioLowVram(on=config.audio_low_vram, auto=config.audio_low_vram_auto)


def on_card(
    config: Config, backend_kind: str, *, total_bytes: int, desktop_allowance_bytes: int
) -> LowVram:
    return decide(
        recorded_of(config),
        card_need(
            backend_kind,
            total_bytes=total_bytes,
            desktop_allowance_bytes=desktop_allowance_bytes,
        ),
    )


def state_setting(state: str) -> AudioLowVram:
    """What `crucible audio low-vram <state>` and Settings write. `auto` is written off
    and decided by the capability step that always follows it."""
    if state not in STATES:
        raise ValueError(f"{state!r} is not one of {list(STATES)}")
    return AudioLowVram(on=state == ON, auto=state == AUTO)


__all__ = [
    "AUTO",
    "LET_CRUCIBLE",
    "NEEDED",
    "NOT_NEEDED",
    "NOT_OFFERED",
    "OFF",
    "ON",
    "STATES",
    "TOO_SMALL",
    "TURN_OFF",
    "TURN_ON",
    "CardNeed",
    "LowVram",
    "Splittable",
    "card_need",
    "decide",
    "on_card",
    "recorded_of",
    "splittable",
    "state_setting",
]
