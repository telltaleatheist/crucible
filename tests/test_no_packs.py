from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from crucible import cli

ROOT = Path(__file__).resolve().parents[1]

GONE_ASSETS = ("envpacks.json", "crucible-env-", "crucible-rootfs-")

GENERATED = (
    "sdk/bootstrap/scripts/install.sh",
    "sdk/bootstrap/scripts/install.ps1",
    "crucible/platform/wsl_table.py",
)


def test_the_deleted_files_are_deleted() -> None:
    for relative in (
        "crucible/envpack.py",
        "scripts/plan_packs.py",
        "scripts/release_packs.py",
        ".github/workflows/envpacks.yml",
        "sdk/bootstrap/scripts/build-rootfs.sh",
        "sdk/bootstrap/src/envpacks.ts",
        "sdk/bootstrap/src/pack.ts",
    ):
        assert not (ROOT / relative).exists(), f"{relative} is back"


def python_sources() -> list[Path]:
    found = [path for path in (ROOT / "crucible").rglob("*.py")]
    found += [path for path in (ROOT / "tests").rglob("*.py")]
    found += [path for path in (ROOT / "scripts").glob("*.py")]
    return found


def test_no_python_module_imports_the_deleted_one() -> None:
    pattern = re.compile(
        r"^\s*(?:from\s+\.?\s*envpack\s+import|"
        r"from\s+crucible\.envpack\s+import|"
        r"from\s+\.\s+import\s+[^\n]*\benvpack\b|"
        r"from\s+crucible\s+import\s+[^\n]*\benvpack\b|"
        r"import\s+crucible\.envpack)",
        re.MULTILINE,
    )
    offenders = []
    for path in python_sources():
        relative = path.relative_to(ROOT).as_posix()
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(relative)
    assert offenders == [], f"these still import crucible.envpack: {offenders}"


def typescript_sources() -> list[Path]:
    root = ROOT / "sdk" / "bootstrap"
    return [
        path
        for directory in ("src", "test", "scripts")
        for path in (root / directory).rglob("*.ts")
    ]


def test_no_typescript_module_imports_the_deleted_ones() -> None:
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in typescript_sources()
        if "from './envpacks.js'" in path.read_text(encoding="utf-8")
        or "from './pack.js'" in path.read_text(encoding="utf-8")
        or "from '../src/envpacks.js'" in path.read_text(encoding="utf-8")
        or "from '../src/pack.js'" in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"these still import the deleted modules: {offenders}"


@pytest.mark.parametrize("relative", GENERATED)
def test_nothing_generated_names_an_asset_no_release_carries(relative: str) -> None:
    text = (ROOT / relative).read_text(encoding="utf-8")
    body = "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith("#")
    )
    for gone in GONE_ASSETS:
        assert gone not in body, f"{relative} still names {gone!r}"


def test_the_cli_offers_neither_the_verb_nor_the_flag() -> None:
    parser = cli.build_parser()
    verbs = {
        name
        for action in parser._subparsers._group_actions
        for name in action.choices
    }
    assert "envpack" not in verbs, f"`crucible envpack` is back: {sorted(verbs)}"
    assert "install" in verbs

    install = next(
        action.choices["install"]
        for action in parser._subparsers._group_actions
        if "install" in action.choices
    )
    flags = {
        option
        for action in install._actions
        for option in action.option_strings
    }
    assert "--build" not in flags, f"`crucible install --build` is back: {sorted(flags)}"
    assert "--manifest-url" not in flags, "there is no manifest to point at"
    assert "--force" in flags, "and --force is still the one thing that deletes an env"


def test_the_install_parser_still_refuses_an_unknown_flag() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["install", "llm", "--build"])
    parsed = parser.parse_args(["install", "llm", "--force"])
    assert isinstance(parsed, argparse.Namespace)
    assert parsed.force is True
