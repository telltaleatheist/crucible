"""The port must keep agreeing with the original it was taken from.

`crucible/jobs/alignlongform/coarse.py` is a transcription of bookforge's
`electron/scripts/align_audiobook.py:coarse_align`. The tests beside this one pin
the BEHAVIOURS that specific books put there; this one pins the stronger and
duller property — that the two functions return the same thing.

WHY A DIFFERENTIAL TEST AND NOT MORE EXAMPLES. Every constant in that function
was set by a book going wrong, and the failure mode of a bad port is not a crash:
it aligns, produces cues, and is quietly wrong. Examples catch the cases somebody
thought of. This catches the ones nobody did, by running both implementations
over randomised books and comparing all six return values.

It also catches drift in the OTHER direction. If BookForge's aligner is improved
and this port is not, the suite goes red and names the divergence, which is the
only thing that keeps "port, don't re-derive" true over time rather than on the
day it was written. This session paid for that lesson once already: the MLX
backend ran seven times slow because a width was re-derived instead of carried.

SKIPPED, LOUDLY, when the BookForge checkout is not beside this one. The original
lives in another repository, so this cannot be a hard dependency of Crucible's
suite — but a silent pass would be worse than a skip, because the whole point is
the comparison.
"""

from __future__ import annotations

import ast
import bisect
import random
import re
import unicodedata
from pathlib import Path

import pytest

from crucible.jobs.alignlongform import coarse as port

#: Where the original lives. A sibling checkout, which is how both repos sit on
#: Owen's machines.
ORIGINAL = (
    Path(__file__).resolve().parents[2]
    / "bookforge"
    / "electron"
    / "scripts"
    / "align_audiobook.py"
)

pytestmark = pytest.mark.skipif(
    not ORIGINAL.is_file(),
    reason=(
        f"the BookForge aligner is not at {ORIGINAL}. This suite compares the "
        "port against the function it was taken from, so without that checkout "
        "there is nothing to compare and a pass would mean nothing."
    ),
)


def _norm(s: str) -> str:
    """The original's own normaliser, verbatim — the port must not supply it."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^0-9a-z]+", "", s.lower())


def _toks(s: str) -> list[str]:
    return [t for t in (_norm(w) for w in s.split()) if t]


def load_original():
    """Lift `coarse_align` out of the script without importing it.

    The script imports faster-whisper, torch and friends at module scope and is
    written to be run, not imported. Extracting the one function by AST keeps
    this suite free of that entire stack — and keeps it honest, because what is
    executed is the original's own bytes rather than a copy kept here.
    """
    tree = ast.parse(ORIGINAL.read_text(encoding="utf-8", errors="replace"))
    fn = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "coarse_align"
        ),
        None,
    )
    assert fn is not None, (
        f"{ORIGINAL} no longer defines coarse_align. Either it was renamed — in "
        "which case this suite must follow it — or the aligner was restructured "
        "and the port needs re-reading against whatever replaced it."
    )
    namespace: dict = {
        "bisect": bisect,
        "toks": _toks,
        "_norm": _norm,
        "log": lambda *a, **k: None,
    }
    module = ast.Module(body=[fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ORIGINAL), "exec"), namespace)
    return namespace["coarse_align"]


def stream(text: str, step: float = 0.5) -> list[tuple[str, float]]:
    return [(_norm(w), i * step) for i, w in enumerate(text.split())]


def books() -> list[tuple[list[str], list[tuple[str, float]]]]:
    """Two shaped cases and twenty-five randomised ones."""
    cases: list[tuple[list[str], list[tuple[str, float]]]] = []

    prose = (
        "It was the best of times it was the worst of times. "
        "We had everything before us we had nothing before us. "
        "The year was seventeen seventy five and the age was one of wisdom."
    )
    cases.append(
        ([s.strip() + "." for s in prose.split(". ") if s.strip()], stream(prose))
    )

    # Well of Ascension's shape: text nobody read aloud, between two spoken lines.
    a = "It was the best of times it was the worst of times."
    b = "The year was seventeen seventy five and the age was wisdom."
    unspoken = [
        "All rights reserved under international copyright conventions here."
        for _ in range(12)
    ]
    cases.append(([a, *unspoken, b], stream(f"{a} {b}")))

    rng = random.Random(7)
    vocabulary = "time year house river wisdom shadow morning iron silver ember".split()
    for _ in range(25):
        sentences = [
            " ".join(rng.choice(vocabulary) for _ in range(rng.randint(3, 12))) + "."
            for _ in range(rng.randint(4, 14))
        ]
        spoken = " ".join(rng.sample(sentences, k=max(1, len(sentences) // 2)))
        cases.append((sentences, stream(spoken)))
    return cases


def test_the_port_returns_exactly_what_the_original_returns() -> None:
    original = load_original()
    divergences: list[str] = []

    for index, (sentences, words) in enumerate(books()):
        rough, first, last, dropped, rate, direct = original(
            list(sentences), list(words)
        )
        ported = port.coarse_align(list(sentences), list(words))

        if list(rough) != list(ported.rough):
            divergences.append(f"case {index}: rough times differ")
        if first != ported.first_index or last != ported.last_index:
            divergences.append(f"case {index}: matched range differs")
        if dropped != ported.dropped:
            divergences.append(
                f"case {index}: dropped {dropped} vs {ported.dropped} — the "
                "narrated/unnarrated judgement moved, which is the Well of "
                "Ascension rule"
            )
        if abs(rate - ported.rate) > 1e-9:
            divergences.append(
                f"case {index}: rate {rate} vs {ported.rate} — the recap-poisoning "
                "guard moved"
            )
        if list(direct) != list(ported.direct):
            divergences.append(
                f"case {index}: `direct` differs, so the align stage would trust "
                "a different set of times over the forced aligner"
            )

    assert not divergences, (
        "the port and the original no longer agree:\n  "
        + "\n  ".join(divergences[:10])
    )
