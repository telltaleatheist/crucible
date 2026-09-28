from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CLIENT_TS = ROOT / "sdk" / "ts" / "src" / "client.ts"
PHASE15 = ROOT / "docs" / "history" / "PHASE15-HOST.md"
MODEL_CHOICE = ROOT / "docs" / "MODEL-CHOICE.md"
CAPABILITY_PROSE = tuple(
    ROOT / "crucible" / f"{name}.py"
    for name in ("capabilityclasses", "capabilitywords", "verdict", "installplan")
)
CAPABILITY_CLASSES = ROOT / "crucible" / "capabilityclasses.py"
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
    assert "without saying so on screen" in flowed(PHASE15)


def test_the_phase_doc_records_the_supersession_rather_than_being_edited_away() -> None:
    doc = flowed(PHASE15)
    assert "SUPERSEDED IN PART, 2026-09-16" in doc
    assert "MODEL-CHOICE.md" in doc
    assert WITHDRAWN_DIVISION.search(doc) is None
    assert "the host is the only caller for now" in doc


def test_no_live_prose_still_says_translation_needs_a_27b() -> None:
    for path in CAPABILITY_PROSE:
        source = text(path)
        for phrase in (
            "it needs a 27B and the smallest",
            "so this host cannot translate",
        ):
            assert phrase not in source, f"{path.name} still asserts: {phrase!r}"


def test_the_reversal_is_written_down_where_a_reader_will_look() -> None:
    assert "cant pick smaller than 9b" in flowed(MODEL_CHOICE)
    assert "docs/MODEL-CHOICE.md" in text(CAPABILITY_CLASSES)


def test_spell_out_names_the_total_and_each_of_its_three_terms() -> None:
    from crucible.fit import Candidate, WorkingContext
    from crucible.capabilitywords import spell_out
    from crucible.manifests import MemoryTerms

    gib = 1024 ** 3
    candidate = Candidate(
        id="probe",
        memory_bytes_estimate=20 * gib,
        memory=MemoryTerms(
            weights_bytes=16 * gib,
            overhead_bytes=2 * gib,
            kv_bytes_per_token=1024,
            basis="measured",
            measured_at_context=16384,
        ),
    )
    work = WorkingContext(tokens=4096, concurrency=4, source="a test")
    said = spell_out(candidate, work)
    total, terms = said.split(" — ")
    assert total == f"{candidate.need_bytes(work) / gib:.1f} GiB"
    assert terms.split(" + ") == [
        "16.0 GiB weights",
        "2.0 GiB overhead",
        "0.0 GiB KV for 4096 tokens x 4 in flight",
    ]


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
