"""docs/CHANGELOG.md: what changed in each release.

Every change adds a line under `## Unreleased` as it lands. `ship.sh` refuses to cut a
release whose Unreleased section is empty (`check`), renames that section to the version
it cuts (`cut <version>`), and release.sh puts the version's section in the GitHub
release notes (`notes <version>`). Owen, 2026-10-09: "record what changed in each update".

  python scripts/changelog.py check
  python scripts/changelog.py cut 1.0.119
  python scripts/changelog.py notes 1.0.119
"""

from __future__ import annotations

import datetime
import re
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "docs" / "CHANGELOG.md"
UNRELEASED = "## Unreleased"
HEADING = re.compile(r"^## (\S+)", re.MULTILINE)


def _sections(text: str) -> list[tuple[str, int, int]]:
    """(name, start of heading, end of section) for every `## ` section, in order."""
    found = list(HEADING.finditer(text))
    return [
        (match.group(1), match.start(), found[i + 1].start() if i + 1 < len(found) else len(text))
        for i, match in enumerate(found)
    ]


def _body(text: str, start: int, end: int) -> str:
    return text[start:end].split("\n", 1)[1].strip() if "\n" in text[start:end] else ""


def _unreleased(text: str) -> tuple[int, int]:
    for name, start, end in _sections(text):
        if name == "Unreleased":
            return start, end
    sys.exit(f"changelog: {PATH} has no '{UNRELEASED}' section; add one above the newest release")


def check() -> None:
    text = PATH.read_text(encoding="utf-8")
    start, end = _unreleased(text)
    if not _body(text, start, end):
        sys.exit(
            f"changelog: nothing under '{UNRELEASED}' in {PATH}. Say what this release "
            "changes there (one line per change, written for the people who use Crucible), "
            "commit it, and ship again"
        )
    print("changelog: Unreleased says what this release changes")


def cut(version: str) -> None:
    text = PATH.read_text(encoding="utf-8")
    if any(name == version for name, _, _ in _sections(text)):
        sys.exit(f"changelog: {PATH} already has a section for {version}")
    start, end = _unreleased(text)
    body = _body(text, start, end)
    if not body:
        sys.exit(f"changelog: nothing under '{UNRELEASED}' to cut as {version}")
    today = datetime.date.today().isoformat()
    section = f"{UNRELEASED}\n\n## {version} — {today}\n\n{body}\n\n"
    PATH.write_text(text[:start] + section + text[end:].lstrip("\n"), encoding="utf-8", newline="\n")
    print(f"changelog: Unreleased is now {version}")


def notes(version: str) -> None:
    text = PATH.read_text(encoding="utf-8")
    for name, start, end in _sections(text):
        if name == version:
            print(_body(text, start, end))
            return
    sys.exit(f"changelog: {PATH} has no section for {version}")


def main(argv: list[str]) -> None:
    # The notes go into a pipe on Windows too, where the default encoding is not UTF-8.
    sys.stdout.reconfigure(encoding="utf-8")
    if argv == ["check"]:
        check()
    elif len(argv) == 2 and argv[0] in ("cut", "notes"):
        (cut if argv[0] == "cut" else notes)(argv[1])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
