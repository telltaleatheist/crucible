"""Every line of every shipped env recipe is something pip can read.

1.0.59 shipped a cuda-linux audio recipe with a stray line of lock-tool log
output in it ("round 0 adding {...}"). pip refused the whole file on the PC and
the audio env could not be built. This scan catches that class of damage
before a release rather than on a user's machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement

ENVS = Path(__file__).resolve().parents[1] / "crucible" / "envs"
RECIPES = sorted(ENVS.rglob("*.txt"))


def _requirement_lines(path: Path) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        lines.append((number, line))
    return lines


def test_there_are_recipes_to_scan() -> None:
    assert RECIPES, f"no recipes under {ENVS}"


@pytest.mark.parametrize("path", RECIPES, ids=lambda p: str(p.relative_to(ENVS)))
def test_every_recipe_line_is_a_requirement(path: Path) -> None:
    bad = []
    for number, line in _requirement_lines(path):
        try:
            Requirement(line)
        except InvalidRequirement as exc:
            bad.append(f"line {number}: {line[:80]!r} ({exc})")
    assert not bad, f"{path.name} has lines pip cannot read:\n" + "\n".join(bad)
