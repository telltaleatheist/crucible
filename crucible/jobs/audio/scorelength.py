"""How long a YuE2 score lasts, read from the score as written.

YuE2 has no length input: it writes a score (ABC) first, in 10 to 30 s, and then composes
and synthesizes the song from it, which is the expensive part. The score's nominal length
- its bars at its tempo - predicted the finished audio closely on the PC's first nine
planning-lyrics instrumentals (2026-10-10): actual over nominal 0.943 to 1.069, median
0.987 (113.7 s of score made 113.3 s of audio, 213.6 made 212.2, 154.3 made 165.0). So a
song's length can be checked against what a client asked for after the score and before
any composing (yue2_worker, docs/AUDIO.md "Song length").

This reader is deliberately NOT the yue2-music skill's strict parser (yue2music/abc_tools.py
`parse_abc`). That parser exists to guard the instrument transfer and refuses a score
whose bars do not fill their meter exactly - and YuE2 writes such scores: the first
planning-lyrics song on the PC has a 4-quarter bar in 3/4 (lantern, seed 1000). A length
check that depended on it would refuse to measure a score the model really wrote. Here
every bar lasts what its notes and rests add up to, at the unit length (L:) and tempo (Q:)
in force where they are written, and the meter (M:) is read only for a whole-bar rest
(`Z`, `X`), which lasts a bar of it. Each voice is summed on its own and the score lasts
as long as its longest voice, since the voices play together. For a score the strict
parser accepts this is exactly its `nominal_duration_seconds` (every bar then fills its
meter, and the voices agree): tests/test_audio_song_length.py holds the two equal on the
nine real scores.

What it does not read, it refuses by name rather than guesses: repeats (`|:`, `:|`, which
would play a passage twice), tuplets, and a score with no tempo. YuE2's dialect writes
none of them. Standard library only: the worker loads it as a sibling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

FIELD = re.compile(r"^([A-Za-z]):(.*)$")
NOTE = re.compile(
    r"(\^\^|__|\^|_|=)?([A-Ga-g])([,']*)(\d*)(/*)(\d*)"
)
REST = re.compile(r"([zx])(\d*)(/*)(\d*)")
MEASURE_REST = re.compile(r"([ZX])(\d*)")
TEMPO = re.compile(r"((?:\d+/\d+\s*)+)=\s*(\d+(?:\.\d+)?)")
FRACTION = re.compile(r"(\d+)/(\d+)")


class ScoreLengthError(ValueError):
    """A score whose length this reader cannot read; the message says what stopped it."""


@dataclass(frozen=True)
class ScoreLength:
    seconds: float
    # The longest voice's bars and quarter notes, and the tempo (quarter notes a minute)
    # and meter the score's header states.
    bars: int
    quarters: float
    bpm: float
    meter: str | None
    voices: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "seconds": round(self.seconds, 2),
            "bars": self.bars,
            "quarters": round(self.quarters, 3),
            "bpm": self.bpm,
            "meter": self.meter,
        }


def _fraction(text: str, what: str) -> Fraction:
    match = FRACTION.fullmatch(text.strip())
    if match is None or int(match.group(2)) == 0:
        raise ScoreLengthError(f"{what} {text.strip()!r} is not a fraction such as 1/32")
    return Fraction(int(match.group(1)), int(match.group(2)))


def _meter(text: str) -> Fraction | None:
    value = text.strip()
    if value in ("", "none"):
        return None
    if value == "C":
        return Fraction(1)
    if value == "C|":
        return Fraction(1)
    return _fraction(value, "the meter")


def _tempo(text: str) -> Fraction:
    """Quarter notes a minute, from `Q:1/4=95` (and `Q:"Allegro" 3/8=60`, `Q:1/4 1/8=60`)."""
    match = TEMPO.search(text)
    if match is None:
        raise ScoreLengthError(
            f"the tempo {text.strip()!r} does not say a beat and a rate, such as Q:1/4=95"
        )
    beat = sum((_fraction(part, "the tempo's beat") for part in match.group(1).split()), Fraction(0))
    return Fraction(match.group(2)) * beat * 4


def _length(digits: str, slashes: str, divisor: str) -> Fraction:
    """An ABC note length multiplier: `2` is 2, `/2` and `/` are 1/2, `//` 1/4, `3/2` 3/2."""
    top = Fraction(int(digits)) if digits else Fraction(1)
    if not slashes:
        return top
    if divisor:
        if len(slashes) != 1 or int(divisor) == 0:
            raise ScoreLengthError(f"the note length {digits}{slashes}{divisor} is not one ABC writes")
        return top / int(divisor)
    return top / (2 ** len(slashes))


@dataclass
class _Voice:
    unit: Fraction
    meter: Fraction | None
    tempo: Fraction | None
    seconds: Fraction = Fraction(0)
    quarters: Fraction = Fraction(0)
    bars: int = 0
    in_bar: Fraction = Fraction(0)
    started: bool = False

    def sound(self, whole_notes: Fraction) -> None:
        if self.tempo is None:
            raise ScoreLengthError("the score has no tempo (Q:) before its first note")
        quarters = whole_notes * 4
        self.quarters += quarters
        self.seconds += quarters * 60 / self.tempo
        self.in_bar += quarters
        self.started = True

    def bar(self) -> None:
        if self.started:
            self.bars += 1
        self.in_bar = Fraction(0)
        self.started = False


def _default_unit(meter: Fraction | None) -> Fraction:
    # ABC 2.1, 3.1.7: without L:, the unit is 1/16 when the meter is below 3/4, else 1/8.
    if meter is not None and meter < Fraction(3, 4):
        return Fraction(1, 16)
    return Fraction(1, 8)


def read(text: str) -> ScoreLength:
    """The score's nominal length: every voice's notes and rests at the unit and tempo in
    force where they stand, the longest voice being the score's."""
    defaults: dict = {"unit": None, "meter": None, "tempo": None}
    header_meter: str | None = None
    header_tempo: Fraction | None = None
    in_header = True
    voices: dict[str, _Voice] = {}
    current: _Voice | None = None

    def voice(name: str) -> _Voice:
        if name not in voices:
            unit = defaults["unit"] or _default_unit(defaults["meter"])
            voices[name] = _Voice(unit=unit, meter=defaults["meter"], tempo=defaults["tempo"])
        return voices[name]

    def apply(field: str, value: str, target: _Voice | None) -> None:
        nonlocal header_meter, header_tempo
        if field == "M":
            meter = _meter(value)
            if target is None:
                defaults["meter"] = meter
                header_meter = header_meter or value.strip()
            else:
                target.meter = meter
        elif field == "L":
            unit = _fraction(value, "the unit length")
            if target is None:
                defaults["unit"] = unit
            else:
                target.unit = unit
        elif field == "Q":
            tempo = _tempo(value)
            if target is None:
                defaults["tempo"] = tempo
                header_tempo = header_tempo or tempo
            else:
                target.tempo = tempo

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        field = FIELD.match(line)
        if field and not line.startswith(("|", "[")):
            name, value = field.group(1), field.group(2)
            if name == "V":
                words = value.split()
                if not words:
                    raise ScoreLengthError("a V: line names no voice")
                current = voice(words[0])
            elif name == "K":
                # K: ends the header (ABC 2.1, 3.1.14); a key changes no length.
                in_header = False
            elif name in ("M", "L", "Q"):
                if in_header or current is None:
                    # A header field sets every voice's starting value, including a
                    # voice the header has already declared.
                    apply(name, value, None)
                    for declared in voices.values():
                        apply(name, value, declared)
                else:
                    apply(name, value, current)
            continue
        if current is None:
            current = voice("")
        _music(line, current, apply)

    sounding = {name: v for name, v in voices.items() if v.quarters > 0}
    if not sounding:
        raise ScoreLengthError("the score has no notes or rests to measure")
    for v in sounding.values():
        if v.started:
            v.bar()
    longest = max(sounding.values(), key=lambda v: v.seconds)
    if header_tempo is None:
        header_tempo = longest.tempo
    return ScoreLength(
        seconds=float(longest.seconds),
        bars=longest.bars,
        quarters=float(longest.quarters),
        bpm=float(header_tempo) if header_tempo is not None else 0.0,
        meter=header_meter,
        voices={name: round(float(v.seconds), 3) for name, v in sounding.items()},
    )


def _music(line: str, voice: _Voice, apply) -> None:
    at = 0
    end = len(line)
    while at < end:
        char = line[at]
        if char.isspace() or char in "-.~()\\y":
            if char == "(" and at + 1 < end and line[at + 1].isdigit():
                raise ScoreLengthError(
                    f"the score has a tuplet ({line[at:at + 3]!r}), which this reader does not time"
                )
            at += 1
            continue
        if char == '"':
            close = line.find('"', at + 1)
            if close < 0:
                raise ScoreLengthError(f"an annotation opens at {line[at:at + 16]!r} and never closes")
            at = close + 1
            continue
        if char in "!+":
            close = line.find(char, at + 1)
            if close < 0:
                raise ScoreLengthError(f"a decoration opens at {line[at:at + 16]!r} and never closes")
            at = close + 1
            continue
        if char == "{":
            close = line.find("}", at + 1)
            if close < 0:
                raise ScoreLengthError(f"a grace group opens at {line[at:at + 16]!r} and never closes")
            at = close + 1
            continue
        if char in "<>":
            # Broken rhythm moves time between two notes; their sum is unchanged.
            at += 1
            continue
        if char == "|" or char == ":" or (char == "[" and at + 1 < end and line[at + 1] == "|"):
            barline = re.match(r"\[?\|*:*\|*\]?:*\d*", line[at:])
            token = barline.group(0) if barline and barline.group(0) else char
            if ":" in token:
                raise ScoreLengthError(
                    f"the score repeats ({token!r}), which would play a passage twice; this "
                    "reader measures a score as written, once through"
                )
            if any(c.isdigit() for c in token):
                raise ScoreLengthError(f"the score has a numbered ending ({token!r})")
            voice.bar()
            at += len(token)
            continue
        if char == "[" and at + 2 < end and line[at + 2] == ":" and line[at + 1].isalpha():
            close = line.find("]", at)
            if close < 0:
                raise ScoreLengthError(f"an inline field opens at {line[at:at + 16]!r} and never closes")
            name = line[at + 1]
            if name == "V":
                raise ScoreLengthError(
                    f"the score changes voice inside a line ({line[at:close + 1]!r}); this "
                    "reader follows voices given on their own V: lines"
                )
            if name in ("M", "L", "Q"):
                apply(name, line[at + 3:close], voice)
            at = close + 1
            continue
        if char == "[":
            close = line.find("]", at)
            if close < 0:
                raise ScoreLengthError(f"a chord opens at {line[at:at + 16]!r} and never closes")
            first = NOTE.search(line, at + 1, close)
            if first is None:
                raise ScoreLengthError(f"the chord {line[at:close + 1]!r} has no note")
            inner = _length(first.group(4), first.group(5), first.group(6))
            after = re.match(r"(\d*)(/*)(\d*)", line[close + 1:])
            outer = _length(after.group(1), after.group(2), after.group(3))
            voice.sound(voice.unit * inner * outer)
            at = close + 1 + len(after.group(0))
            continue
        measure = MEASURE_REST.match(line, at)
        if measure:
            if voice.meter is None:
                raise ScoreLengthError("a whole-bar rest (Z) stands where no meter (M:) is set")
            count = int(measure.group(2) or "1")
            for index in range(count):
                if index:
                    voice.bar()
                voice.sound(voice.meter)
            at = measure.end()
            continue
        rest = REST.match(line, at)
        if rest:
            voice.sound(voice.unit * _length(rest.group(2), rest.group(3), rest.group(4)))
            at = rest.end()
            continue
        note = NOTE.match(line, at)
        if note:
            voice.sound(voice.unit * _length(note.group(4), note.group(5), note.group(6)))
            at = note.end()
            continue
        raise ScoreLengthError(f"the score has {line[at:at + 16]!r}, which this reader does not time")


__all__ = ["ScoreLength", "ScoreLengthError", "read"]
