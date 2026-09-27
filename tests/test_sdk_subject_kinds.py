from __future__ import annotations

import re
from pathlib import Path

from crucible import catalog

TYPES_TS = Path(__file__).resolve().parent.parent / "sdk" / "ts" / "src" / "types.ts"


def union_members() -> list[str]:
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
