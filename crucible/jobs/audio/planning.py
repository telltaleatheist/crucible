"""Planning lyrics: the words a YuE2 instrumental is planned from and never sings.

An instrumental's score is planned like a sung song's, from lyric lines, so its vocal melody
has a sung song's bounded phrase structure; the worker then moves that melody to the
instrument and YuE2 re-plans from the fixed score with section tags only
(yue2_worker._instrumental_plan). Planned from empty sections, nothing bounded the score:
4 of 13 instrumentals on Victoria's laptop ran it to its 4096-token cap (2026-10-10).

The words come from the client's `planning_lyrics`, or else from the engine's pool
(crucible/audio/planning/<engine>.toml), one set picked from the job's seed. The pick is
made here, on the server, before the job runs, so the kept request and the done record
both say which set it was; the worker is handed only the text.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The section labels YuE2 writes into a score and the yue2-music skill accepts back
# (yue2music/compile_score.py SECTIONS; tests/test_audio_planning_lyrics.py holds the two
# equal). A tag outside them would come back as a score label the skill refuses.
SECTIONS = frozenset(
    {"intro", "verse", "pre-chorus", "chorus", "bridge", "interlude", "outro", "instrumental"}
)

# What planning lyrics may hold. Sung lyrics have no limit of Crucible's own: YuE2's caps
# bound them (4096 score tokens, 9000 song tokens - about 6 minutes). Planning lyrics exist
# to keep the score inside the first of those, and a sung song of 16 lines scores 1800 to
# 2600 tokens, so 36 lines is past what any score can hold; 2000 characters is 36 lines of
# 55. Past either, the plan could only run to its cap.
MAX_LINES = 36
MAX_CHARS = 2000

POOL = "pool"
REQUEST = "request"

PLANNING_DIR = Path(__file__).resolve().parents[2] / "audio" / "planning"


class PlanningLyricsError(ValueError):
    """Planning lyrics that cannot plan an instrumental; the message says why."""


def check(text: str) -> list[tuple[str, list[str]]]:
    """The sections of `text` as (label, lines), or PlanningLyricsError naming the first
    rule it breaks. The same rules hold for a client's `planning_lyrics` and the pool."""
    sections: list[tuple[str, list[str]]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]") and line.count("[") == line.count("]") == 1:
            label = line[1:-1].strip().casefold()
            if label not in SECTIONS:
                raise PlanningLyricsError(
                    f"{line!r} is not one of YuE2's section tags; use "
                    + ", ".join(f"[{name.title()}]" for name in sorted(SECTIONS))
                )
            if sections and sections[-1][0] == label:
                raise PlanningLyricsError(
                    f"{line!r} follows another {line!r}; YuE2's score cannot carry the same "
                    "section twice in a row, so put the lines in one section or another tag "
                    "between them"
                )
            sections.append((label, []))
            continue
        if "[" in line or "]" in line:
            raise PlanningLyricsError(
                f"{line!r} mixes a section tag with words; a tag such as [Verse] goes on a "
                "line of its own"
            )
        if not sections:
            raise PlanningLyricsError(
                f"{line!r} comes before any section tag; start with one such as [Verse]"
            )
        sections[-1][1].append(line)
    lines = sum(len(words) for _, words in sections)
    if not lines:
        raise PlanningLyricsError(
            "has no lines of words, only section tags; planning lyrics are words the score "
            "is planned from (never sung). To shape an instrumental by its sections alone, "
            "send the tags as `lyrics`"
        )
    if lines > MAX_LINES:
        raise PlanningLyricsError(
            f"has {lines} lines of words; at most {MAX_LINES} fit in a score (a sung song "
            "of 16 lines scores 1800 to 2600 of its 4096 tokens)"
        )
    if len(text) > MAX_CHARS:
        raise PlanningLyricsError(
            f"is {len(text)} characters; at most {MAX_CHARS} (about {MAX_LINES} lines of 55)"
        )
    return sections


@dataclass(frozen=True)
class PlanningSet:
    id: str
    shape: str
    lyrics: str


def pool_path(engine: str) -> Path:
    return PLANNING_DIR / f"{engine}.toml"


def load_pool(engine: str) -> list[PlanningSet]:
    """The engine's planning sets, in file order; every one passes `check`. A missing or
    broken file is this build's fault and is refused by name."""
    path = pool_path(engine)
    if not path.is_file():
        raise PlanningLyricsError(
            f"no planning lyrics for the {engine} engine at {path}; every engine that makes "
            "instrumentals ships a pool there"
        )
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    entries = document.get("set")
    if not isinstance(entries, list) or not entries:
        raise PlanningLyricsError(f"{path} has no [[set]] tables")
    pool: list[PlanningSet] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        fields = {key: entry.get(key) for key in ("id", "shape", "lyrics")}
        if set(entry) != set(fields) or not all(isinstance(v, str) and v.strip() for v in fields.values()):
            raise PlanningLyricsError(
                f"{path}: set {index} needs exactly a non-empty id, shape and lyrics"
            )
        if fields["id"] in seen:
            raise PlanningLyricsError(f"{path}: set id {fields['id']!r} is used twice")
        seen.add(fields["id"])
        try:
            check(fields["lyrics"])
        except PlanningLyricsError as exc:
            raise PlanningLyricsError(f"{path}: set {fields['id']!r} {exc}") from None
        pool.append(PlanningSet(**fields))
    return pool


def pick(pool: list[PlanningSet], seed: int) -> PlanningSet:
    """The set a seed plans from: number (seed mod len(pool)), in file order."""
    return pool[seed % len(pool)]


def named(pool: list[PlanningSet], set_id: str) -> PlanningSet:
    """The set a client named as `planning_set`, or PlanningLyricsError naming every id."""
    for entry in pool:
        if entry.id == set_id:
            return entry
    raise PlanningLyricsError(
        f"there is no planning set {set_id!r}; the pool's sets are "
        + ", ".join(entry.id for entry in pool)
    )


def record(source: str, lyrics: str, set_id: str | None, requested: bool | None = None) -> dict[str, Any]:
    """What the job keeps of its planning lyrics (the kept request's `settled` and the done
    record's `audio`): where they came from, the pool set's id (null for the client's
    own), whether the client named that set (`planning_set`; null for the client's own
    words), and the text, which sent back as `planning_lyrics` plans the same song even
    after the pool has changed."""
    return {"source": source, "id": set_id, "requested": requested, "lyrics": lyrics}


# --- sizing a planning structure to a length (docs/AUDIO.md "Song length") -------------
#
# An instrumental asked for a length range is planned from its pool set made longer or
# shorter by whole sections: the set's body (every section between a leading [Intro] and a
# trailing [Outro]) is cut short from its end, or carried on past it by starting the body
# over, so a longer song repeats the set's own verses and choruses in its own order. Each
# size is a count of body sections, so sizes are few, ordered and the same every time.

# Seconds of score a planning line makes, before this request has scored anything:
# the median of the PC's first nine pool instrumentals (2026-10-10, one per set, tags at
# 66 to 112 BPM), whose scores ran 5.7 to 12.0 s a line. The tempo the model picks and how
# many bars it gives a line both move it, so the first score is aimed with it and every
# later one with what this request's own scores measured.
PRIOR_SECONDS_PER_LINE = 8.6

# How far inside the range a size is aimed: a score lands within about 7% of where its
# lines put it once a request's own rate is known (the nine scores' audio ran 0.943 to
# 1.069 of their nominal length), so aiming 10% in from each end leaves room for that.
AIM_MARGIN = 0.10

# Scores an instrumental may plan to land in its range: the first, and two re-plans. A
# score is 10 to 30 s of the card; composing is the expensive part and never repeats.
MAX_LENGTH_ATTEMPTS = 3

FRAME_OPENING = "intro"
FRAME_CLOSING = "outro"


@dataclass(frozen=True)
class Section:
    tag: str
    label: str
    lines: tuple[str, ...]

    def text(self) -> str:
        return "\n".join((self.tag, *self.lines))


def sections(text: str) -> list[Section]:
    """`text`'s sections as written (each tag spelt as the text spells it), after `check`
    has passed them."""
    check(text)
    found: list[Section] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            found.append(Section(line, line[1:-1].strip().casefold(), ()))
        else:
            last = found[-1]
            found[-1] = Section(last.tag, last.label, (*last.lines, line))
    return found


def render(parts: list[Section]) -> str:
    return "\n\n".join(part.text() for part in parts) + "\n"


@dataclass(frozen=True)
class Sizing:
    """The sizes one planning text comes in: `text(n)` has n body sections, for every n in
    `sizes` (each one passes `check`, so none is longer than a score can hold)."""

    base: str
    opening: tuple[Section, ...]
    body: tuple[Section, ...]
    closing: tuple[Section, ...]
    sizes: tuple[int, ...] = ()

    @classmethod
    def of(cls, text: str) -> "Sizing":
        parts = sections(text)
        start = 0
        while start < len(parts) and parts[start].label == FRAME_OPENING:
            start += 1
        stop = len(parts)
        while stop > start and parts[stop - 1].label == FRAME_CLOSING:
            stop -= 1
        opening, body, closing = tuple(parts[:start]), tuple(parts[start:stop]), tuple(parts[stop:])
        if not body:
            raise PlanningLyricsError("has no sections between its intro and outro to size")
        unsized = cls(text, opening, body, closing)
        sizes = []
        count = 1
        while True:
            try:
                check(render(unsized._parts(count)))
            except PlanningLyricsError:
                if count >= len(body):
                    break
            else:
                sizes.append(count)
            count += 1
        return cls(text, opening, body, closing, tuple(sizes))

    @property
    def written(self) -> int:
        """The body sections the text has as written."""
        return len(self.body)

    def _carried(self, count: int) -> list[Section]:
        if count <= len(self.body):
            return list(self.body[:count])
        if len({part.label for part in self.body}) < 2:
            raise PlanningLyricsError("has one kind of section in its body, so it cannot grow")
        carried = list(self.body)
        index = 0
        while len(carried) < count:
            part = self.body[index % len(self.body)]
            index += 1
            # Starting the body over must not put one section after another of its kind,
            # which YuE2's score cannot carry (check).
            if part.label != carried[-1].label:
                carried.append(part)
        return carried

    def _parts(self, count: int) -> list[Section]:
        return [*self.opening, *self._carried(count), *self.closing]

    def text(self, count: int) -> str:
        """The planning text with `count` body sections; the text as written for its own
        count, so a song not resized plans from exactly the words it always did."""
        if count not in self.sizes:
            raise ValueError(f"{count} body sections is not one of this text's sizes {self.sizes}")
        if count == self.written:
            return self.base
        return render(self._parts(count))

    def lines(self, count: int) -> int:
        return sum(len(part.lines) for part in self._parts(count))

    def structure(self, count: int) -> list[str]:
        """The section labels of size `count`, in order (what a job's attempts record)."""
        return [part.label for part in self._parts(count)]


@dataclass(frozen=True)
class LengthRange:
    """The range a client asked a song's length to land in; either end may be absent.
    `ceiling` is the model's own longest song, the far end when no maximum was sent."""

    minimum: float | None
    maximum: float | None
    ceiling: float

    def holds(self, seconds: float) -> bool:
        return (self.minimum is None or seconds >= self.minimum) and (
            self.maximum is None or seconds <= self.maximum
        )

    def window(self) -> tuple[float, float]:
        """Where a size is aimed: AIM_MARGIN in from each end, or the middle of a range
        too narrow for that."""
        low = self.minimum if self.minimum is not None else 0.0
        high = self.maximum if self.maximum is not None else self.ceiling
        inner = (low * (1 + AIM_MARGIN), high * (1 - AIM_MARGIN))
        if inner[0] > inner[1]:
            middle = (low + high) / 2
            return middle, middle
        return inner

    def to_dict(self) -> dict[str, float | None]:
        return {"min_duration_s": self.minimum, "max_duration_s": self.maximum}


def aim(sizing: Sizing, wanted: LengthRange, seconds_per_line: float, tried: set[int]) -> int | None:
    """The body-section count to plan next: of the sizes not yet tried, the one whose
    estimate (its lines at `seconds_per_line`) lands in the range's window, the fewest
    sections from the text as written first (a set that already fits is left as it is);
    if none lands there, the one nearest the window. None when every size was tried."""
    low, high = wanted.window()

    def miss(count: int) -> float:
        estimate = sizing.lines(count) * seconds_per_line
        return max(low - estimate, estimate - high, 0.0)

    left = [count for count in sizing.sizes if count not in tried]
    if not left:
        return None
    return min(left, key=lambda count: (miss(count), abs(count - sizing.written), count))


__all__ = [
    "AIM_MARGIN",
    "MAX_CHARS",
    "MAX_LENGTH_ATTEMPTS",
    "MAX_LINES",
    "POOL",
    "PRIOR_SECONDS_PER_LINE",
    "REQUEST",
    "SECTIONS",
    "LengthRange",
    "PlanningLyricsError",
    "PlanningSet",
    "Section",
    "Sizing",
    "aim",
    "check",
    "load_pool",
    "named",
    "pick",
    "pool_path",
    "record",
    "render",
    "sections",
]
