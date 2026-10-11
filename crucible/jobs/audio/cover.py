"""Covers: a YuE2 song rendered from a score the client gives instead of one YuE2 plans.

YuE2 writes a score (ABC) and then composes and synthesizes the song from it. Given a
score, `pipe.plan(tags, lyrics, abc=..., cot=...)` decodes nothing and returns that score
as the plan (yue2-infer's SongRequest: external ABC needs cot "melody" or "full"), so the
song follows the given melody. Two covers come from it (Owen, 2026-10-11):

- an INSTRUMENTAL cover (`instrumental: true`): the score's Vocal melody is moved note for
  note to the instrument, by the same vendored skill the instrumental workflow uses
  (yue2music/instrumentalize.py convert_score, instrumental.py validate_score), and it is
  rendered with lyrics that are only the score's own section tags (lyric_tags), so
  nothing is sung;
- a SUNG cover: the score's Vocal melody is sung with the client's `lyrics`, tagged
  sections the client wrote to the score's phrases (upstream's "cover from a score: a
  validated melody-only ABC and the desired lyrics", skills/yue2-music at the pinned
  commit).

Everything that needs no model is decided here, on the server, before the worker is asked
for anything: a score the skill's parser refuses, or one longer than the model composes,
fails the job by name without loading YuE2. The one check that needs YuE2's tokenizer -
the score fits the 4096-token planning budget, which yue2-infer does not enforce on a
given score - is the worker's (yue2_worker._given_plan), and its refusal comes back as a
failed job with its own code without taking the worker down.

The vendored tools import one another by bare module name; `_tools` loads them for the
server with their directory on sys.path only while they import, and refuses a clash with
another module of the same name.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...errors import JobError
from . import scorelength

# The score as the client sent it, kept beside the rendered `score.abc` (what YuE2 was
# given after any transfer or chord removal).
GIVEN_ARTIFACT = "given-score.abc"

INSTRUMENTAL = "instrumental"
SUNG = "sung"

KEEP = "keep"
STRIP = "strip"

# Where a score came from, for the done record: bytes the client sent (a blob or inline)
# or another job's artifact named by reference.
UPLOAD = "upload"
REFERENCE = "reference"

TOOLS_DIR = Path(__file__).resolve().parent / "yue2music"
TOOL_MODULES = ("common", "abc_tools", "compile_score", "instrumentalize", "instrumental")


@dataclass(frozen=True)
class Tools:
    parse_abc: Any
    strip_chords: Any
    convert_score: Any
    validate_score: Any
    lyric_tags: Any


_LOADED: list[Tools] = []


def _tools() -> Tools:
    if _LOADED:
        return _LOADED[0]
    for name in TOOL_MODULES:
        # The tools stay registered under their bare names: compile_score imports
        # abc_tools again inside compile_events, at call time. Another module of the same
        # name would be a clash this build cannot render through, so it is refused.
        found = sys.modules.get(name)
        where = getattr(found, "__file__", None)
        if found is not None and (where is None or Path(where).resolve().parent != TOOLS_DIR):
            raise RuntimeError(
                f"a module named {name!r} from {getattr(found, '__file__', None)} is already "
                f"loaded; YuE2's score tools ({TOOLS_DIR}) import theirs by that bare name"
            )
    sys.path.insert(0, str(TOOLS_DIR))
    try:
        modules = {name: importlib.import_module(name) for name in TOOL_MODULES}
    finally:
        sys.path.remove(str(TOOLS_DIR))
    tools = Tools(
        parse_abc=modules["abc_tools"].parse_abc,
        strip_chords=modules["abc_tools"].strip_chords,
        convert_score=modules["instrumentalize"].convert_score,
        validate_score=modules["instrumental"].validate_score,
        lyric_tags=modules["instrumental"].lyric_tags,
    )
    _LOADED.append(tools)
    return tools


@dataclass(frozen=True)
class Cover:
    kind: str
    # What YuE2 renders: the given score after the instrumental transfer or chord removal.
    abc: str
    cot: str
    chords: str
    chords_chosen_by: str
    # The [Lyrics] the plan is given: the client's words (sung) or the score's section
    # tags (instrumental).
    lyrics: str
    seconds: float
    reading: dict[str, Any]
    # convert_score's own account of the transfer (instrumental only).
    transfer: dict[str, Any] | None
    given: str

    def request(self) -> dict[str, Any]:
        """What the worker is handed (yue2_worker._given_plan)."""
        return {"kind": self.kind, "abc": self.abc, "cot": self.cot, "lyrics": self.lyrics}

    def record(self, source: dict[str, Any], score_tokens: int | None) -> dict[str, Any]:
        """The done record's `audio.cover` (docs/AUDIO.md "Covers"); `score_tokens` is
        the worker's count, None in a record kept before it rendered."""
        return {
            "kind": self.kind,
            **source,
            "given": GIVEN_ARTIFACT,
            "cot": self.cot,
            "chords": "kept" if self.chords == KEEP else "stripped",
            "chords_chosen_by": self.chords_chosen_by,
            "score_tokens": score_tokens,
            "score_seconds": round(self.seconds, 2),
            "score": self.reading,
        }


def _refused(code: str, message: str) -> JobError:
    return JobError(code, message)


def read_text(path: Path) -> str:
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _refused(
            "score_invalid",
            f"the score {path.name!r} is not UTF-8 text ({exc}); send the ABC itself",
        ) from None
    # A score saved on Windows: the dialect is line-based, and YuE2 is given the exact
    # text, so its lines end the way the model writes them.
    return text.replace("\r\n", "\n")


def _invalid(name: str, exc: Exception) -> JobError:
    return _refused(
        "score_invalid",
        f"the score {name!r} is not one YuE2's score tools accept "
        f"({type(exc).__name__}: {exc}). It must be YuE2's own two-voice ABC dialect, as "
        "YuE2 or the `score` job writes it",
    )


def prepare(
    text: str,
    name: str,
    *,
    instrumental: bool,
    lyrics: str | None,
    chords: str | None,
    longest_s: float,
) -> Cover:
    """The score a cover renders, checked by the vendored skill; raises JobError
    `score_invalid` or `score_too_long` by name."""
    tools = _tools()
    sung_lines = [line for line in text.splitlines() if line.lower().startswith("w:")]
    if sung_lines:
        raise _refused(
            "score_invalid",
            f"the score {name!r} carries words in the ABC ({sung_lines[0]!r} first); "
            "YuE2's dialect has none. Send the words as `lyrics` (a sung cover) and the "
            "score without its w: lines",
        )
    chosen_by = "request" if chords is not None else "default"
    # The defaults are upstream's own: the instrumental conversion keeps the harmony
    # (convert_score keep_chords=True); a sung cover is rendered from a melody-only score
    # (skills/yue2-music "Cover from a score").
    chords = chords if chords is not None else (KEEP if instrumental else STRIP)
    transfer = None
    try:
        tools.parse_abc(text)
        if instrumental:
            abc, transfer = tools.convert_score(text, keep_chords=chords == KEEP)
            tools.validate_score(abc)
        else:
            abc = text if chords == KEEP else tools.strip_chords(text)
        score = tools.parse_abc(abc)
    except Exception as exc:
        # The skill's checks are the refusal: any of them failing is a score it will not
        # render, and its own words say why.
        raise _invalid(name, exc) from None
    if not instrumental and not score.voices["Vocal"].notes:
        raise _refused(
            "score_invalid",
            f"the score {name!r} has no notes in its Vocal voice, so a sung cover has no "
            "melody to sing; send `instrumental: true` to play it, or a score with a "
            "Vocal melody",
        )
    try:
        reading = scorelength.read(abc)
    except scorelength.ScoreLengthError as exc:
        raise _invalid(name, exc) from None
    if reading.seconds > longest_s:
        raise _refused(
            "score_too_long",
            f"the score {name!r} lasts {reading.seconds:.1f} s ({reading.bars} bars at "
            f"{reading.bpm:g} BPM); YuE2 composes at most {longest_s:g} s. Send a shorter "
            "score",
        )
    return Cover(
        kind=INSTRUMENTAL if instrumental else SUNG,
        abc=abc,
        cot="full" if score.voices["Vocal"].chords else "melody",
        chords=chords,
        chords_chosen_by=chosen_by,
        lyrics=tools.lyric_tags(abc) if instrumental else lyrics,
        seconds=reading.seconds,
        reading=reading.to_dict(),
        transfer=transfer,
        given=text,
    )


def source_of(name: str, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """`{"source": "upload"}` or `{"source": "reference", "reference": {job_id, name}}`,
    from what the server recorded as it took the job's inputs."""
    declared = sources.get(name)
    if declared is None:
        raise JobError(
            "invalid_inputs",
            f"the server recorded no source for input {name!r}; it records one for every "
            "input it takes",
        )
    if declared["from"] == "artifact":
        return {"source": REFERENCE, "input": name,
                "reference": {"job_id": declared["job_id"], "name": declared["name"]}}
    return {"source": UPLOAD, "input": name, "reference": None}


__all__ = [
    "Cover",
    "GIVEN_ARTIFACT",
    "INSTRUMENTAL",
    "KEEP",
    "STRIP",
    "SUNG",
    "prepare",
    "read_text",
    "source_of",
]
