"""The pack system is GONE, and this is what keeps it gone.

PHASE20-CODE-NOT-ENVIRONMENTS.md section 6 lists what was deleted: the pack
build, the manifest, the download, the carry-by-reference, `crucible envpack`,
`scripts/plan_packs.py`, `scripts/release_packs.py`,
`.github/workflows/envpacks.yml`, `envpacks.json`, the disk guards, the part
splitting, `build-rootfs.sh` and `crucible install --build`.

WHY THIS IS NOT A GREP FOR THE WORD
------------------------------------
The obvious keeper is "the string `envpack` appears nowhere", and it is the
wrong one: half this repo's comments earn their keep by saying what USED to be
there and why it went — `crucible/interpreter.py` explains that its table used
to live in `crucible/envpack.py`, `sdk/bootstrap/src/distro.ts` explains what
`build-rootfs.sh` baked in — and a check that forbade those would be a check
that forces history to be deleted along with the code. A phase doc describing
a superseded design is the same shape, on purpose.

So what this file proves is MECHANISM: nothing IMPORTS the deleted modules,
nothing GENERATED names the deleted assets, the CLI offers neither the verb nor
the flag, and the files themselves are not on disk. Every one of those fails if
somebody brings a piece of it back, and none of them fails because a comment
remembers it.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from crucible import cli

ROOT = Path(__file__).resolve().parents[1]

#: The assets a release used to carry, which no GENERATED file may name. The
#: installers and `crucible/host/wsl_states.py` are written from
#: `sdk/bootstrap/`, so a stale constant there reaches every machine.
GONE_ASSETS = ("envpacks.json", "crucible-env-", "crucible-rootfs-")

GENERATED = (
    "sdk/bootstrap/scripts/install.sh",
    "sdk/bootstrap/scripts/install.ps1",
    "crucible/host/wsl_states.py",
)


def test_the_deleted_files_are_deleted() -> None:
    """The files themselves, named, so a revert is a failing test rather than a
    quiet return of the thing that cost two hours per deploy."""
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
    """Every `.py` this repo ships or tests with. Not `build/`, which is a
    `python -m build` leftover, and not `node_modules`."""
    found = [path for path in (ROOT / "crucible").rglob("*.py")]
    found += [path for path in (ROOT / "tests").rglob("*.py")]
    found += [path for path in (ROOT / "scripts").glob("*.py")]
    return found


def test_no_python_module_imports_the_deleted_one() -> None:
    """An import is the only thing that can RUN it, so an import is the check."""
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
    """THE ONES THAT REACH A MACHINE. These three are written by
    `sdk/bootstrap/scripts/gen-install-scripts.ts` and then run on somebody's
    computer, so a stale asset name here is not a stale comment — it is a
    download of a file that does not exist."""
    text = (ROOT / relative).read_text(encoding="utf-8")
    body = "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith("#")
    )
    for gone in GONE_ASSETS:
        assert gone not in body, f"{relative} still names {gone!r}"


def test_the_cli_offers_neither_the_verb_nor_the_flag() -> None:
    """`crucible envpack` and `crucible install --build`, asked of the PARSER.

    A grep would find the words in a comment; this asks the thing a person
    types. `--build` was the developer's way past the download, and with no
    download left it is a flag whose name would lie.
    """
    parser = cli.build_parser()
    verbs = {
        name
        for action in parser._subparsers._group_actions  # type: ignore[union-attr]
        for name in action.choices
    }
    assert "envpack" not in verbs, f"`crucible envpack` is back: {sorted(verbs)}"
    assert "install" in verbs

    install = next(
        action.choices["install"]
        for action in parser._subparsers._group_actions  # type: ignore[union-attr]
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
    """The deletion did not loosen the parser: a flag it does not take is a
    usage error, not a silently ignored word."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["install", "llm", "--build"])
    parsed = parser.parse_args(["install", "llm", "--force"])
    assert isinstance(parsed, argparse.Namespace)
    assert parsed.force is True
