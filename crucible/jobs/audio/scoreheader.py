"""A YuE2 score's key, meter and tempo: what a client may fix before YuE2 plans its score
(`key`, `meter`, `bpm`), the header lines that fixes, and the three read back from the
score as written (docs/AUDIO.md "Key, meter and tempo").

YuE2 writes every score in one header layout, the yue2-music skill's native dialect
(yue2music/abc_tools.py `parse` refuses any other), line by line:

    0 X:1
    1 T:
    2 M:<meter>
    3 L:1/<unit>
    4 Q:1/4=<bpm>
    5 V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
    6 V: Ins clef=treble name="Ins Melody" snm="Inst."
    7 K:<key>

so a fixed field is a whole header line at a known index (`LINE`), written in for the
model as that line begins, and every other line - the unit length, the voices, then the
whole body - stays the model's (yue2_worker.forced_header).

What may be fixed is what the skill's tools read: a key is one of its 30 standard major
and minor keys (abc_tools.KEYS; no modes - the dialect writes none and the skill refuses
them), a meter an explicit fraction over a power of two (abc_tools.meter_value), and a
tempo an integer quarter-note rate (`Q:1/4=<BPM>`, the only tempo the dialect writes).

Standard library only: the worker loads it as a sibling.
"""

from __future__ import annotations

import re

# abc_tools.KEYS, in its order (tests/test_audio_header.py holds the two equal): fifteen
# major keys from seven flats to seven sharps, then their relative minors.
KEYS: tuple[str, ...] = (
    "Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#",
    "Abm", "Ebm", "Bbm", "Fm", "Cm", "Gm", "Dm", "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m", "A#m",
)

# A meter a song may be written in: abc_tools.meter_value's explicit fraction, held to
# what a song is played in - 1 to 16 beats of a half, quarter, eighth or sixteenth note.
METER = re.compile(r"([1-9][0-9]?)/(2|4|8|16)")
MAX_BEATS = 16

# Quarter notes a minute. YuE2's own scores run 66 to 118 (tests/fixtures/yue2-scores);
# this range is what a song's tags ask for, slow ballad to fast dance.
MIN_BPM = 40
MAX_BPM = 240

# Which header line each field is (the layout above).
LINE: dict[str, int] = {"meter": 2, "bpm": 4, "key": 7}
FIELDS: tuple[str, ...] = ("key", "meter", "bpm")

TEMPO = re.compile(r"1/4=([1-9][0-9]*)")
FIELD = re.compile(r"^([A-Za-z]):(.*)$")


def check_key(value: str) -> str:
    if value not in KEYS:
        raise ValueError(
            f"{value!r} is not a key YuE2's score writes; send one of the 15 major keys "
            f"({', '.join(KEYS[:15])}) or 15 minor keys ({', '.join(KEYS[15:])}), spelled "
            "as ABC's K: field spells them (sharps #, flats b, minor m; no modes)"
        )
    return value


def check_meter(value: str) -> str:
    match = METER.fullmatch(value)
    if match is None or int(match.group(1)) > MAX_BEATS:
        raise ValueError(
            f"{value!r} is not a meter YuE2's score writes; send a fraction such as \"4/4\", "
            f"\"3/4\" or \"6/8\": 1 to {MAX_BEATS} beats over 2, 4, 8 or 16"
        )
    return value


def lines(header: dict | None) -> dict[int, str]:
    """The header lines a song's fixed fields write, by line index; {} when none is fixed."""
    if header is None:
        return {}
    unknown = sorted(set(header) - set(FIELDS))
    if unknown:
        raise ValueError(f"a score header fixes {list(FIELDS)}, not {unknown}")
    written = {}
    if header.get("meter") is not None:
        written[LINE["meter"]] = f"M:{check_meter(header['meter'])}"
    if header.get("bpm") is not None:
        bpm = header["bpm"]
        if type(bpm) is not int or not MIN_BPM <= bpm <= MAX_BPM:
            raise ValueError(f"bpm {bpm!r} is not an integer from {MIN_BPM} to {MAX_BPM}")
        written[LINE["bpm"]] = f"Q:1/4={bpm}"
    if header.get("key") is not None:
        written[LINE["key"]] = f"K:{check_key(header['key'])}"
    return written


def read(abc: str | None) -> dict | None:
    """The key, meter and tempo a score's header states: its first K:, M: and Q: lines
    before the body (the first line that is not a field), as written. `bpm` is the integer
    of `Q:1/4=<BPM>` and null for any other tempo line; a field the header lacks is null.
    None when there is no score."""
    if not abc:
        return None
    found: dict = {"key": None, "meter": None, "bpm": None}
    seen: set = set()
    for line in abc.splitlines():
        field = FIELD.match(line.strip())
        if field is None:
            break
        name, value = field.group(1), field.group(2).strip()
        if name in seen or name not in "KMQ":
            continue
        seen.add(name)
        if name == "K":
            found["key"] = value
        elif name == "M":
            found["meter"] = value
        else:
            tempo = TEMPO.fullmatch(value)
            found["bpm"] = None if tempo is None else int(tempo.group(1))
    return found


__all__ = ["FIELDS", "KEYS", "LINE", "MAX_BPM", "MIN_BPM", "check_key", "check_meter", "lines", "read"]
