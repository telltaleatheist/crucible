"""Prose that makes a CHECKABLE claim is checked against the thing it describes.

Four data points in one evening, found by two sessions independently:

  1. `SubjectKind`'s comment said "five things" over a union of six — nearly
     shipping a Windows engine that could not fetch llama.cpp. Caught by the
     Foundry session; `tests/test_sdk_subject_kinds.py` is its keeper and is the
     model for this file.
  2. `readCapabilityRow`'s `route = 'local'` shim outlived the document version
     it existed for.
  3. `removeSubject`'s comment said 3.5a is explicit that neither app calls it —
     while the Foundry session was building exactly that button on Owen's
     ruling. Caught by that session reading both.
  4. After the 2026-09-16 floor reversal, five keepers across three files still
     ASSERTED the withdrawn ruling, because only the new test files were re-run
     and the suite stayed green.

The pattern is one thing: **signatures and unions move with a ruling and the
prose above them does not.** The generated-file checks in `release.sh`
(`gen-modules --check`, `gen-api-docs --check`) close the same gap one level up,
where the artifact is generated. These close it where the artifact is a sentence.

WHAT IS AND IS NOT TESTABLE HERE. A comment explaining WHY cannot be checked and
must not be. What can is a comment that COUNTS something, NAMES a set, or states
a DIVISION that some other file settles — those have a referent in the tree, and
a test can hold them to it. Each is a keeper of its own rather than a framework;
there is no rule that finds them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CLIENT_TS = ROOT / "sdk" / "ts" / "src" / "client.ts"
PHASE15 = ROOT / "docs" / "PHASE15-HOST.md"
MODEL_CHOICE = ROOT / "docs" / "MODEL-CHOICE.md"
CAPABILITY = ROOT / "crucible" / "capability.py"
ENVPACKS_DOC = ROOT / "docs" / "PHASE14-ENVPACKS.md"


def text(path: Path) -> str:
    assert path.is_file(), f"{path} is gone; this test names a file that moved"
    return path.read_text(encoding="utf-8")


def flowed(path: Path) -> str:
    """One long line, with wrapping and comment furniture removed.

    THE FIRST THREE ASSERTIONS IN THIS FILE FAILED ON THIS and they were right
    to: prose is wrapped, and a quoted sentence in a markdown blockquote carries
    `> ` at every line start and `*` at its ends. An exact-string match against
    a file that a human formatted is a test about formatting, not about the
    claim — and it would go red on a reflow that changed nothing.
    """
    body = text(path)
    for furniture in ("> ", " *", "* ", "   ", chr(10), chr(13)):
        body = body.replace(furniture, " ")
    return " ".join(body.split())


# ------------------------------------------------- claim 3: who may delete


#: The division `removeSubject` used to assert and that Owen withdrew on
#: 2026-09-16. Matched loosely on purpose — a reworded restatement is the same
#: mistake, and an exact-string test would pass on a paraphrase.
WITHDRAWN_DIVISION = re.compile(
    r"neither\s+BookForge\s+nor\s+Foundry\s+calls\s+it", re.IGNORECASE
)


def test_the_sdk_does_not_still_say_no_app_may_delete() -> None:
    """Both apps call `DELETE /v1/catalog/{kind}/{id}` since Owen's ruling.

    Foundry built its delete control (foundry 2a789da) while this comment still
    said the opposite. The cost of leaving it is not a broken build — it is a
    later reader concluding that a shipped button was a mistake.
    """
    body = flowed(CLIENT_TS)
    hit = WITHDRAWN_DIVISION.search(body)
    if hit is None:
        return
    # It MAY appear inside the block that says it was withdrawn — that block
    # quotes the old words on purpose, so a reader arriving with them in mind
    # finds them and is told what replaced them. What must not happen is the
    # claim standing on its own.
    around = body[max(0, hit.start() - 220) : hit.end() + 120]
    assert "used to" in around or "WHAT CHANGED" in around, (
        f"removeSubject asserts the withdrawn division as live text: ...{around}..."
    )


def test_the_condition_that_survived_the_reversal_is_still_stated() -> None:
    """THE HALF THAT IS NOT SUPERSEDED, and the reason this is not a deletion.

    "An app does not call this without saying so on screen" was always the rule
    and survives a change in WHO calls. Deleting it along with the division it
    was attached to would have thrown away the part that still governs — which
    is how a correction becomes a second defect.
    """
    assert "without saying so on" in flowed(CLIENT_TS)
    assert "without saying so on screen" in flowed(PHASE15)


def test_the_phase_doc_records_the_supersession_rather_than_being_edited_away() -> None:
    """The source, not just the echo.

    The SDK comment CITES PHASE15-HOST.md 3.5a. Correcting the citation and
    leaving the cited text saying the old thing moves the defect rather than
    fixing it, and leaves the next reader trusting the more authoritative of the
    two.
    """
    doc = flowed(PHASE15)
    assert "SUPERSEDED IN PART, 2026-09-16" in doc
    assert "MODEL-CHOICE.md" in doc
    # The withdrawn words are QUOTED in the supersession block, which is the
    # point: a reader who arrives with the old sentence in mind finds it and is
    # told what replaced it, instead of finding nothing and wondering.
    assert WITHDRAWN_DIVISION.search(doc) is None
    assert "the host is the only caller for now" in doc


# ---------------------------------------- claim: the withdrawn 27B floor


def test_no_live_prose_still_says_translation_needs_a_27b() -> None:
    """Owen's 2026-09-13 ruling, withdrawn on 2026-09-16.

    `capability.py` is the file that DECIDES, so a sentence there claiming the
    27B floor is the one most likely to be believed. Historical mentions are
    fine and necessary — MODEL-CHOICE.md quotes the old ruling in full to
    explain what was reversed — so this holds only the deciding file.
    """
    source = text(CAPABILITY)
    for phrase in (
        "it needs a 27B and the smallest",
        "so this host cannot translate",
    ):
        assert phrase not in source, f"capability.py still asserts: {phrase!r}"


def test_the_reversal_is_written_down_where_a_reader_will_look() -> None:
    """A ruling that lives only in a commit message is a ruling nobody finds."""
    assert "cant pick smaller than 9b" in flowed(MODEL_CHOICE)
    assert "docs/MODEL-CHOICE.md" in text(CAPABILITY)


# ------------------------------------------------- the counting claims


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
        # `decide()`'s refusal names four terms, and the docstring says four.
        ("spell_out", 4),
    ],
)
def test_a_comment_that_counts_is_counted(sentence: str, actual: int) -> None:
    """The `SubjectKind` failure, generalised as far as it honestly goes.

    A comment saying "all four terms" beside code that emits three is the exact
    shape that put "five" over a union of six. Only the claims with a countable
    referent are here; a sentence saying "several" is not a defect.
    """
    source = text(CAPABILITY)
    body = source[source.index(f"def {sentence}") : source.index(f"def {sentence}") + 1600]
    named = re.findall(r"\b(" + "|".join(COUNT_WORDS) + r")\s+terms\b", body, re.I)
    assert named, f"{sentence} no longer counts its terms; drop this row or fix it"
    for word in named:
        assert COUNT_WORDS[word.lower()] == actual, (
            f"{sentence} says {word!r} terms and emits {actual}"
        )



# ------------------------------------------- the schema the doc puts in a code block

def test_the_envpacks_doc_shows_the_schema_the_code_emits() -> None:
    """A JSON sample in a document is a CLAIM, and this one is copied by hand.

    Section 2 of PHASE14-ENVPACKS.md prints an `envpacks.json` and calls itself
    "the single owner of what packs exist". A sample showing a schema the code
    no longer writes is worse than no sample: it is authoritative-looking and
    wrong, and the reader has no reason to doubt it.
    """
    from crucible import envpack

    doc = text(ENVPACKS_DOC)
    assert f'"schema": {envpack.PACK_SCHEMA},' in doc, (
        f"the doc's example manifest does not show schema {envpack.PACK_SCHEMA}"
    )


def test_every_row_field_the_code_requires_appears_in_the_example() -> None:
    """The eight-then-nine fields, held to the code's own list.

    `_ENTRY_FIELDS` is what `parse_manifest` demands. A field the code requires
    and the example omits is a document that teaches somebody to write a
    manifest this build refuses.
    """
    from crucible import envpack

    doc = text(ENVPACKS_DOC)
    required = envpack._ENTRY_FIELDS + envpack._ENTRY_FIELDS_V2
    for field, _kind in required:
        assert f'"{field}"' in doc, f"the example manifest has no {field!r}"


def test_the_weakened_invariant_is_written_down_where_it_changed() -> None:
    """NOT just in a commit message, and not only in the code.

    Section 4 promised a release's manifest names only assets of that release.
    It does not any more, and the replacement — every pack RESOLVES, the row
    says where, and the manifest job checks — is the sort of thing somebody
    needs to find when they wonder why a v0.6.8 manifest points at v0.6.6.
    """
    doc = flowed(ENVPACKS_DOC)
    assert "carried by reference" in doc
    assert "resolves" in doc
    for measured in ("thirteen packs", "three that genuinely changed"):
        assert measured in doc, f"the doc no longer carries the measurement: {measured!r}"
