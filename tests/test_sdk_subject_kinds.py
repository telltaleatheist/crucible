"""The SDK's `SubjectKind` union and `catalog.KINDS` are ONE list, compared.

WHY THIS FILE EXISTS. They are two hand-maintained spellings of the same fact —
the server's tuple and the TypeScript union apps write against — and until
2026-09-16 nothing compared them. The union's own sentence said "The five
things a subject can be" while six were listed: `engine` was added for the
llama.cpp binaries a `llama-windows` server runs its GGUF models with
(PHASE15-HOST.md 3.10) and the count above it did not move.

Foundry read the SENTENCE, wrote a five-word mirror of the union, and was saved
only by its own compiler. A mirror built by hand from the prose would have
shipped a Windows-native engine that could not fetch its own llama.cpp.

`docs/ARCHITECTURE.md` R1: a fact with two owners and nothing comparing them.
This is the comparison.
"""

from __future__ import annotations

import re
from pathlib import Path

from crucible import catalog

TYPES_TS = Path(__file__).resolve().parent.parent / "sdk" / "ts" / "src" / "types.ts"


def union_members() -> list[str]:
    """The quoted members of `export type SubjectKind = …`, in order.

    THE COMMENTS COME OUT FIRST. A member of this union carries a doc comment
    (`engine` does), and that prose contains a semicolon — "a weights pull;
    what makes it different is…" — so slicing to the first `;` stops halfway
    through the union and silently reports five members when there are six.
    Which is the very mistake this file exists to catch, made again by the
    file catching it.
    """
    source = TYPES_TS.read_text(encoding="utf-8")
    start = source.index("export type SubjectKind =")
    tail = re.sub(r"/\*.*?\*/", "", source[start:], flags=re.DOTALL)
    return re.findall(r"\|\s*'([a-z-]+)'", tail[: tail.index(";")])


def test_the_sdk_union_is_exactly_the_servers_kinds() -> None:
    assert union_members() == list(catalog.KINDS), (
        "sdk/ts/src/types.ts's SubjectKind and crucible/catalog.py's KINDS have "
        "drifted. The server's tuple is the owner; the union mirrors it, and an "
        "app that writes a kind this server does not know gets a 400 three "
        "layers down in a vocabulary it cannot see."
    )


def test_the_sentence_above_the_union_counts_it() -> None:
    """The drift that actually happened was in the PROSE, not the code."""
    source = TYPES_TS.read_text(encoding="utf-8")
    head = source[: source.index("export type SubjectKind =")]
    paragraph = head[head.rindex("/**") :]
    words = {
        3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
    }
    expected = words[len(catalog.KINDS)]
    assert expected.upper() in paragraph.upper(), (
        f"the sentence above SubjectKind does not say {expected!r}, and there "
        f"are {len(catalog.KINDS)} kinds. A reader who counts the sentence "
        "instead of the union writes the wrong mirror — which is exactly how "
        "this was found."
    )
