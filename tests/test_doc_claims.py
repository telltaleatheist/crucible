from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CLIENT_TS = ROOT / "sdk" / "ts" / "src" / "client.ts"
PHASE15 = ROOT / "docs" / "history" / "PHASE15-HOST.md"
MODEL_CHOICE = ROOT / "docs" / "MODEL-CHOICE.md"
CAPABILITY = ROOT / "crucible" / "capability.py"
ENVPACKS_DOC = ROOT / "docs" / "history" / "PHASE14-ENVPACKS.md"


def text(path: Path) -> str:
    assert path.is_file(), f"{path} is gone; this test names a file that moved"
    return path.read_text(encoding="utf-8")


def flowed(path: Path) -> str:
    body = text(path)
    for furniture in ("> ", " *", "* ", "   ", chr(10), chr(13)):
        body = body.replace(furniture, " ")
    return " ".join(body.split())


WITHDRAWN_DIVISION = re.compile(
    r"neither\s+BookForge\s+nor\s+Foundry\s+calls\s+it", re.IGNORECASE
)


def test_the_sdk_does_not_still_say_no_app_may_delete() -> None:
    body = flowed(CLIENT_TS)
    hit = WITHDRAWN_DIVISION.search(body)
    if hit is None:
        return
    around = body[max(0, hit.start() - 220) : hit.end() + 120]
    assert "used to" in around or "WHAT CHANGED" in around, (
        f"removeSubject asserts the withdrawn division as live text: ...{around}..."
    )


def test_the_condition_that_survived_the_reversal_is_still_stated() -> None:
    assert "without saying so on" in flowed(CLIENT_TS)
    assert "without saying so on screen" in flowed(PHASE15)


def test_the_phase_doc_records_the_supersession_rather_than_being_edited_away() -> None:
    doc = flowed(PHASE15)
    assert "SUPERSEDED IN PART, 2026-09-16" in doc
    assert "MODEL-CHOICE.md" in doc
    assert WITHDRAWN_DIVISION.search(doc) is None
    assert "the host is the only caller for now" in doc


def test_no_live_prose_still_says_translation_needs_a_27b() -> None:
    source = text(CAPABILITY)
    for phrase in (
        "it needs a 27B and the smallest",
        "so this host cannot translate",
    ):
        assert phrase not in source, f"capability.py still asserts: {phrase!r}"


def test_the_reversal_is_written_down_where_a_reader_will_look() -> None:
    assert "cant pick smaller than 9b" in flowed(MODEL_CHOICE)
    assert "docs/MODEL-CHOICE.md" in text(CAPABILITY)


COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
}


@pytest.mark.parametrize(
    "sentence,actual",
    [
        ("spell_out", 4),
    ],
)
def test_a_comment_that_counts_is_counted(sentence: str, actual: int) -> None:
    source = text(CAPABILITY)
    body = source[source.index(f"def {sentence}") : source.index(f"def {sentence}") + 1600]
    named = re.findall(r"\b(" + "|".join(COUNT_WORDS) + r")\s+terms\b", body, re.I)
    assert named, f"{sentence} no longer counts its terms; drop this row or fix it"
    for word in named:
        assert COUNT_WORDS[word.lower()] == actual, (
            f"{sentence} says {word!r} terms and emits {actual}"
        )


def test_the_weakened_invariant_is_written_down_where_it_changed() -> None:
    doc = flowed(ENVPACKS_DOC)
    assert "carried by reference" in doc
    assert "resolves" in doc
    for measured in ("thirteen packs", "three that genuinely changed"):
        assert measured in doc, f"the doc no longer carries the measurement: {measured!r}"


def test_every_uvicorn_this_repo_starts_states_the_one_keep_alive() -> None:
    starts = re.compile(r"uvicorn\.(?:run|Config)\s*\(")
    found: list[tuple[Path, str]] = []
    for path in sorted((ROOT / "crucible").rglob("*.py")):
        source = text(path)
        for match in starts.finditer(source):
            depth, index = 0, match.end() - 1
            while index < len(source):
                if source[index] == "(":
                    depth += 1
                elif source[index] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                index += 1
            found.append((path.relative_to(ROOT), source[match.start() : index + 1]))
    assert len(found) == 2, f"a uvicorn door was added or removed: {[p for p, _ in found]}"
    for path, call in found:
        assert "timeout_keep_alive=KEEP_ALIVE_SECONDS" in call, (
            f"{path} starts a uvicorn on the 5 s default keep-alive; "
            "state KEEP_ALIVE_SECONDS, which is why it exists"
        )
