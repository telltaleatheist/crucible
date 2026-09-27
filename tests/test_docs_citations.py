from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

SCANNED = ("crucible", "scripts", "sdk", "docs/internals")
BARE_NAMES_SCANNED = ("crucible", "scripts", "docs/internals")
SKIPPED_PARTS = {"__pycache__", "node_modules", "dist", ".git"}
TEXT_SUFFIXES = {
    ".py", ".sh", ".ps1", ".md", ".txt", ".toml", ".js", ".ts", ".html", ".css", ".json",
}

DOCS_PATH = re.compile(r"(?<![\w./-])docs/((?:[\w-]+/)*[\w.-]+\.md)\b")
BARE_DOC = re.compile(r"(?<![\w./-])([A-Z][A-Z0-9]*(?:[-_][A-Z0-9.]+)*\.md)\b")
MARKDOWN_LINK = re.compile(r"\]\(([^)#\s:]+\.md)(?:#[^)]*)?\)")
SECTION = re.compile(
    r"(?<![\w./-])docs/(internals/[\w-]+\.md)(?:'s|,)?\s+\\?[\"']([^\"'\\]+)\\?[\"']"
)
HEADING = re.compile(r"^#+\s+(.*)$", re.MULTILINE)


def _files(top: str) -> list[Path]:
    found = []
    for path in sorted((ROOT / top).rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if SKIPPED_PARTS.intersection(path.relative_to(ROOT).parts):
            continue
        found.append(path)
    return found


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _where(path: Path, text: str, index: int) -> str:
    return f"{path.relative_to(ROOT).as_posix()}:{text.count(chr(10), 0, index) + 1}"


def _docs_path_citations() -> list[tuple[str, str]]:
    cited = []
    for top in SCANNED:
        for path in _files(top):
            text = _read(path)
            for match in DOCS_PATH.finditer(text):
                cited.append((_where(path, text, match.start()), match.group(1)))
    return cited


def _bare_citations() -> list[tuple[str, Path, str]]:
    cited = []
    for top in BARE_NAMES_SCANNED:
        for path in _files(top):
            text = _read(path)
            for match in BARE_DOC.finditer(text):
                cited.append((_where(path, text, match.start()), path, match.group(1)))
    return cited


def _markdown_links() -> list[tuple[str, Path, str]]:
    cited = []
    for path in _files("docs/internals"):
        text = _read(path)
        for match in MARKDOWN_LINK.finditer(text):
            cited.append((_where(path, text, match.start()), path, match.group(1)))
    return cited


def test_the_scan_finds_the_citations_it_is_meant_to_guard() -> None:
    assert len(_docs_path_citations()) > 20
    assert len(_bare_citations()) > 5
    assert len(_markdown_links()) > 0


def test_every_docs_path_named_in_code_and_internals_exists() -> None:
    missing = [
        f"{where} cites docs/{name}"
        for where, name in _docs_path_citations()
        if not (DOCS / name).is_file()
    ]
    assert not missing, (
        "these cite a doc that is not there; point each at the docs/internals/*.md "
        "section that now holds the rule, or at docs/history/ if it is history:\n"
        + "\n".join(missing)
    )


def test_every_bare_doc_name_in_code_and_internals_exists() -> None:
    missing = []
    for where, path, name in _bare_citations():
        places = (DOCS / name, ROOT / name, path.parent / name)
        if not any(place.is_file() for place in places):
            missing.append(f"{where} names {name}")
    assert not missing, (
        "these name a doc that is neither in docs/, at the repo root nor beside the file; "
        "point each at the docs/internals/*.md section that now holds the rule:\n"
        + "\n".join(missing)
    )


def test_every_markdown_link_in_the_internals_resolves() -> None:
    missing = [
        f"{where} links {target}"
        for where, path, target in _markdown_links()
        if not (path.parent / target).resolve().is_file()
    ]
    assert not missing, "\n".join(missing)


def _section_citations() -> list[tuple[str, str, str]]:
    cited = []
    for top in SCANNED:
        for path in _files(top):
            text = _read(path)
            for match in SECTION.finditer(text):
                cited.append((_where(path, text, match.start()), match.group(1), match.group(2)))
    return cited


def test_every_quoted_internals_section_is_a_heading_of_that_file() -> None:
    cited = _section_citations()
    assert len(cited) > 20
    missing = []
    for where, name, section in cited:
        target = DOCS / name
        headings = HEADING.findall(target.read_text(encoding="utf-8")) if target.is_file() else []
        wanted = section.strip().strip("`").lower()
        if not any(heading.strip("`").lower().startswith(wanted) for heading in headings):
            missing.append(f'{where} cites docs/{name}, "{section}", which is not one of its headings')
    assert not missing, "\n".join(missing)


@pytest.mark.parametrize("name", ["README.md"])
def test_the_history_folder_says_it_is_not_maintained(name: str) -> None:
    text = (DOCS / "history" / name).read_text(encoding="utf-8")
    assert "not maintained" in text
