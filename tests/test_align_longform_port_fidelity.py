from __future__ import annotations

import ast
import bisect
import random
import re
import unicodedata
from pathlib import Path

import pytest

from crucible.jobs.alignlongform import coarse as port

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
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^0-9a-z]+", "", s.lower())


def _toks(s: str) -> list[str]:
    return [t for t in (_norm(w) for w in s.split()) if t]


def load_original(name: str = "coarse_align"):
    tree = ast.parse(ORIGINAL.read_text(encoding="utf-8", errors="replace"))
    fn = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )
    assert fn is not None, (
        f"{ORIGINAL} no longer defines {name}. Either it was renamed — in "
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
    return namespace[name]


def stream(text: str, step: float = 0.5) -> list[tuple[str, float]]:
    return [(_norm(w), i * step) for i, w in enumerate(text.split())]


def books() -> list[tuple[list[str], list[tuple[str, float]]]]:
    cases: list[tuple[list[str], list[tuple[str, float]]]] = []

    prose = (
        "It was the best of times it was the worst of times. "
        "We had everything before us we had nothing before us. "
        "The year was seventeen seventy five and the age was one of wisdom."
    )
    cases.append(
        ([s.strip() + "." for s in prose.split(". ") if s.strip()], stream(prose))
    )

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


def test_snap_boundaries_matches_the_original_too() -> None:
    original = load_original("snap_boundaries")
    from crucible.jobs.alignlongform import cues as ported

    rng = random.Random(11)
    divergences: list[str] = []
    for case in range(200):
        n = rng.randint(2, 8)
        starts = [0.0]
        ends: list[float] = []
        t = 0.0
        for _ in range(n):
            t += rng.uniform(1.0, 6.0)
            ends.append(round(t, 3))
            starts.append(round(t, 3))
        starts = starts[:n]
        silences = []
        for e in ends[:-1]:
            if rng.random() < 0.7:
                a = round(e + rng.uniform(-1.5, 1.0), 3)
                silences.append((a, round(a + rng.uniform(0.05, 3.0), 3)))
        silences.sort()
        window = rng.choice([0.25, 0.5, 1.0, 2.0])

        o_starts, o_ends, o_stats = original(
            list(starts), list(ends), list(silences), window
        )
        p_starts, p_ends, p_stats = ported.snap_boundaries(
            list(starts), list(ends), list(silences), window
        )
        if [round(x, 9) for x in o_starts] != [round(x, 9) for x in p_starts]:
            divergences.append(f"case {case}: starts differ")
        if [round(x, 9) for x in o_ends] != [round(x, 9) for x in p_ends]:
            divergences.append(f"case {case}: ends differ")
        if o_stats["snapped"] != p_stats.snapped:
            divergences.append(
                f"case {case}: snapped {o_stats['snapped']} vs {p_stats.snapped}"
            )
        if o_stats["considered"] != p_stats.considered:
            divergences.append(f"case {case}: considered differs")

    assert not divergences, (
        "snap_boundaries no longer matches the original:\n  "
        + "\n  ".join(divergences[:10])
    )
