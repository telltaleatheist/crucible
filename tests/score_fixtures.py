"""Scores written the way SheetSage2 writes them (notation_sheetsage2.score_to_abc): the
native two-voice header, measure groups of one to four bars, and its own structure labels
as `% label` lines before a group - several before one group where sections change inside
its first bar. Shared by the score tests and the fake score worker."""

from __future__ import annotations

HEADER = (
    "X:1\n"
    "T:\n"
    "M:4/4\n"
    "L:1/32\n"
    "Q:1/4=120\n"
    'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
    'V: Ins clef=treble name="Ins Melody" snm="Inst."\n'
    "K:C\n"
)


def group(labels: list[str], vocal: str, ins: str) -> str:
    return "".join(f"% {label}\n" for label in labels) + f"V: Vocal\n{vocal}\nV: Ins\n{ins}\n"


def song(groups: list[tuple[list[str], str, str]]) -> str:
    return HEADER + "".join(group(*each) for each in groups)


# A sung song in SheetSage2's vocabulary: `silence` at both ends, a verse labelled twice in
# a row, a `post-chorus` after the chorus, and one group opened by two labels.
SHEETSAGE2_SONG = song([
    (["silence"], "Z|", "C,32|"),
    (["verse"], "C8D8E8F8|G16z16|C8D8E8F8|G16z16|", "Z4|"),
    (["verse"], "E8F8G16|z32|", "C,32|G,,32|"),
    (["chorus"], '"C"c8B8A8G8|"G"G16z16|', "Z2|"),
    (["post-chorus"], '"F"A8G8F8E8|"C"C32|', "Z2|"),
    (["bridge", "rap"], "E4E4E4E4D16|C32|", "Z2|"),
    (["fade-out"], "z32|", "C,32|"),
    (["silence"], "Z|", "Z|"),
])

# The same song with the chords SheetSage2's `full` task writes (they are already in the
# Vocal voice above); melody-full writes none.
MELODY_ONLY_SONG = SHEETSAGE2_SONG.replace('"C"', "").replace('"G"', "").replace('"F"', "")

UNKNOWN_LABEL_SONG = song([
    (["intro"], "Z|", "C,32|"),
    (["kazoo break"], "C8D8E8F8|", "Z|"),
])

# Parses, but has no sounding note anywhere: YuE2's voice transfer refuses it.
SILENT_SONG = song([
    (["intro"], "Z|", "Z|"),
    (["verse"], "Z2|", "Z2|"),
])


__all__ = [
    "HEADER",
    "MELODY_ONLY_SONG",
    "SHEETSAGE2_SONG",
    "SILENT_SONG",
    "UNKNOWN_LABEL_SONG",
    "group",
    "song",
]
