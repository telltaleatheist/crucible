"""SheetSage2's ABC made YuE2-native: its section labels mapped onto YuE2's set, every
decision recorded, and the result held to YuE2's own score tools.

SheetSage2 and YuE2 write the same two-voice ABC dialect (abc_tools.py says so), with one
difference that matters: the `% label` comments. SheetSage2 labels sections from its own
structure vocabulary (labels_sheetsage2.STRUCTURE_LABELS, 23 labels: `silence`,
`post-chorus`, `rap`, `fade-out`, `solo`, `pre-outro`, `intro and verse`, ...) and can open
one measure group with several of them; YuE2 knows eight (compile_score.SECTIONS) and its
validator refuses an unknown one and the same one twice in a row. So:

1. Every label is mapped by the table below. `silence` alone depends on where it stands:
   the first label in the score is the song's `intro`, the last its `outro`, any other an
   `interlude`. A label not in the table fails the job by name: a vocabulary this build has
   not seen is not guessed at.
2. Where one group opens with several labels, the last is kept: SheetSage2 starts a group
   at the measure where a section begins, and the last label it wrote there is the one in
   force through the group's music.
3. A label equal to the section already in force is dropped: the two stretches are one
   section to YuE2, which refuses a repeated label.

Then the checks, all YuE2's own code (yue2music/, vendored unchanged): the relabelled score
must parse (`abc_tools.parse_abc`, the strict two-voice parser), and its instrumental form -
the sung melody moved to the instrument voice by `instrumentalize.convert_score`, as YuE's
cover skill does before YuE2 plays a cover - must pass `instrumental.validate_score`. That
validator is written for the instrumental score (it refuses Vocal notes), which is why it
is applied to the instrumental form and the sung score is held to the parser and the label
rule. A score that fails any of it is not repaired by guessing; the job fails by name and
keeps SheetSage2's own score.

Standard library only, plus the vendored tools: the score worker loads this as a sibling.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

YUE2_MUSIC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio", "yue2music"
)

LABEL_PREFIX = "% "

# YuE2's section labels (compile_score.SECTIONS), checked against the vendored set at use.
YUE2_SECTIONS = frozenset(
    {"intro", "verse", "pre-chorus", "chorus", "bridge", "interlude", "outro", "instrumental"}
)

# Every SheetSage2 structure label (labels_sheetsage2.STRUCTURE_LABELS at 32c7c747) but
# `silence`, to its YuE2 section and why. Where the MIREX reduction SheetSage2 ships
# (MIREX_STRUCTURE_LABEL_REDUCTION) has an answer YuE2 can say, it is that answer.
MAPPING: dict[str, tuple[str, str]] = {
    "intro": ("intro", "the same section"),
    "verse": ("verse", "the same section"),
    "pre-chorus": ("pre-chorus", "the same section"),
    "chorus": ("chorus", "the same section"),
    "bridge": ("bridge", "the same section"),
    "interlude": ("interlude", "the same section"),
    "outro": ("outro", "the same section"),
    "instrumental": ("instrumental", "the same section"),
    "post-chorus": ("chorus", "the hook's tail; YuE2 has no post-chorus, and it joins the chorus it follows"),
    "fade-out": ("outro", "the song's ending (the MIREX reduction: outro)"),
    "pre-outro": ("outro", "the run into the ending; YuE2 has no pre-outro, and it joins the outro"),
    "loop": ("chorus", "a repeated hook (the MIREX reduction: chorus)"),
    "rap": ("verse", "a spoken or rapped verse (the MIREX reduction: verse)"),
    "irregular": ("verse", "an irregular sung section (the MIREX reduction: verse)"),
    "preshot": ("interlude", "a short instrumental pickup (the MIREX reduction: inst)"),
    "solo": ("instrumental", "an instrumental solo"),
    "intro and verse": ("intro", "a compound label takes its first section, where it begins"),
    "pre-chorus and chorus": ("pre-chorus", "a compound label takes its first section, where it begins"),
    "verse and pre-chorus": ("verse", "a compound label takes its first section, where it begins"),
    "theme": ("verse", "the main material of a theme-and-variations form"),
    "variation": ("verse", "a restatement of the main material"),
    "development": ("bridge", "the contrasting middle of the form"),
}

SILENCE = "silence"

SILENCE_AT = {
    "first": ("intro", "silence that opens the song is its intro"),
    "last": ("outro", "silence that ends the song is its outro"),
    "middle": ("interlude", "silence between sections is an interlude"),
}

# What the job fails with, by name, when the score cannot be made native.
UNKNOWN_LABEL = "score_label_unknown"
NOT_NATIVE = "score_not_yue2_native"


class NotNative(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class Native:
    """The outcome, success or not: `error` is None when both scores are native."""

    score: str | None
    instrumental: str | None
    labels: list[dict[str, Any]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    transfer: dict[str, Any] | None = None
    error: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.error is None,
            "labels": self.labels,
            "checks": self.checks,
            "instrumental_transfer": self.transfer,
            "error": self.error,
        }


def tools():
    """The vendored yue2-music tools; they import one another by bare name."""
    if YUE2_MUSIC not in sys.path:
        sys.path.insert(0, YUE2_MUSIC)
    import abc_tools
    import compile_score
    import instrumental
    import instrumentalize

    if set(compile_score.SECTIONS) != YUE2_SECTIONS:
        raise RuntimeError(
            f"the vendored yue2music SECTIONS are {sorted(compile_score.SECTIONS)} and "
            f"this mapping targets {sorted(YUE2_SECTIONS)}; re-vendoring changed YuE2's "
            "labels, so crucible/jobs/score/yue2native.py's table must follow"
        )
    return abc_tools, instrumental, instrumentalize


def _groups(lines: list[str]) -> list[tuple[int, list[int]]]:
    """The runs of consecutive label lines, as (first index, [line indexes])."""
    runs: list[tuple[int, list[int]]] = []
    for index, line in enumerate(lines):
        if not line.startswith(LABEL_PREFIX):
            continue
        if runs and runs[-1][1][-1] == index - 1:
            runs[-1][1].append(index)
        else:
            runs.append((index, [index]))
    return runs


def _mapped(raw: str, position: str) -> tuple[str, str]:
    if raw == SILENCE:
        return SILENCE_AT[position]
    found = MAPPING.get(raw)
    if found is None:
        raise NotNative(
            UNKNOWN_LABEL,
            f"SheetSage2 labelled a section {raw!r}, which is not in the vocabulary this "
            f"build maps onto YuE2's ({sorted(YUE2_SECTIONS)}); the score is kept as "
            "SheetSage2 wrote it, and crucible/jobs/score/yue2native.py's MAPPING is "
            "where a new label gets its ruling",
        )
    return found


def relabel(text: str) -> tuple[str, list[dict[str, Any]]]:
    """SheetSage2's labels as YuE2's, with one record per label line it read."""
    lines = text.split("\n")
    runs = _groups(lines)
    labelled = [index for _, run in runs for index in run]
    records: list[dict[str, Any]] = []
    drop: set[int] = set()
    rewrite: dict[int, str] = {}
    active: str | None = None
    for run_number, (_, run) in enumerate(runs):
        kept_index = run[-1]
        for index in run:
            raw = lines[index][len(LABEL_PREFIX):].strip()
            order = labelled.index(index)
            position = (
                "first" if order == 0 else "last" if order == len(labelled) - 1 else "middle"
            )
            section, why = _mapped(raw, position)
            record: dict[str, Any] = {
                "line": index + 1,
                "group": run_number + 1,
                "sheetsage2": raw,
                "yue2": section,
                "why": why,
            }
            if index != kept_index:
                record["kept"] = False
                record["dropped_because"] = (
                    "another label opens the same group after it; the last one there "
                    "is the section in force through the group"
                )
                drop.add(index)
            elif section == active:
                record["kept"] = False
                record["dropped_because"] = (
                    f"{section!r} is already the section in force; YuE2 refuses the "
                    "same label twice in a row, and the two stretches are one section"
                )
                drop.add(index)
            else:
                record["kept"] = True
                rewrite[index] = LABEL_PREFIX + section
                active = section
            records.append(record)
    out = [rewrite.get(index, line) for index, line in enumerate(lines) if index not in drop]
    return "\n".join(out), records


def _labels_are_native(text: str) -> None:
    previous = None
    for line in text.splitlines():
        if line.startswith(LABEL_PREFIX):
            label = line[len(LABEL_PREFIX):]
            if label not in YUE2_SECTIONS or label == previous:
                raise NotNative(
                    NOT_NATIVE,
                    f"after relabelling, the score still has the label {label!r} "
                    f"(after {previous!r}); this is a bug in the relabelling",
                )
            previous = label


def _check(checks: list[dict[str, Any]], name: str, what: str, run) -> Any:
    try:
        found = run()
    except NotNative:
        raise
    except (ValueError, KeyError, TypeError) as exc:
        checks.append({"check": name, "what": what, "passed": False, "said": str(exc)})
        raise NotNative(
            NOT_NATIVE,
            f"the relabelled score fails YuE2's {what}: {exc}",
        ) from None
    checks.append({"check": name, "what": what, "passed": True})
    return found


def to_yue2(raw: str, *, keep_harmony: bool) -> Native:
    """The YuE2-native forms of SheetSage2's score, or why there are none."""
    native = Native(score=None, instrumental=None)
    try:
        abc_tools, instrumental, instrumentalize = tools()
        text, native.labels = relabel(raw)
        _labels_are_native(text)
        native.checks.append(
            {"check": "labels", "what": "section labels (YuE2's eight, never twice in a row)", "passed": True}
        )
        _check(native.checks, "parse_abc", "two-voice parser (abc_tools.parse_abc)",
               lambda: abc_tools.parse_abc(text))
        native.score = text
        converted, transfer = _check(
            native.checks,
            "convert_score",
            "voice transfer to the instrumental form (instrumentalize.convert_score)",
            lambda: instrumentalize.convert_score(text, keep_chords=keep_harmony),
        )
        _check(native.checks, "validate_score", "instrumental validator (instrumental.validate_score)",
               lambda: instrumental.validate_score(converted))
        native.instrumental = converted
        native.transfer = transfer
    except NotNative as exc:
        native.error = {"code": exc.code, "message": str(exc)}
    return native


__all__ = [
    "MAPPING",
    "NOT_NATIVE",
    "Native",
    "NotNative",
    "SILENCE_AT",
    "UNKNOWN_LABEL",
    "YUE2_SECTIONS",
    "relabel",
    "to_yue2",
    "tools",
]
