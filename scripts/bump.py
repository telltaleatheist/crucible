from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]

SEMVER = r"(\d+\.\d+\.\d+)"

PLACES: list[tuple[str, str, str]] = [
    ("crucible/__init__.py", rf'^VERSION = "{SEMVER}"$',
     "the server's own version, and the one every other place is checked against"),
    ("pyproject.toml", rf'^version = "{SEMVER}"$',
     "the wheel's version (this is what nearly shipped 0.1.0 bytes as v0.2.0)"),
    ("sdk/ts/package.json", rf'^  "version": "{SEMVER}",$',
     "the TypeScript client's package version"),
    ("sdk/ts/src/version.ts", rf"^export const SDK_VERSION = '{SEMVER}';$",
     "the version the SDK reports in User-Agent"),
    ("sdk/bootstrap/package.json", rf'^  "version": "{SEMVER}",$',
     "the bootstrapper's package version"),
    ("sdk/bootstrap/package.json", rf'^    "@crucible/client": "{SEMVER}"$',
     "the bootstrapper's peer pin on the client cut beside it"),
    ("sdk/bootstrap/src/version.ts", rf"^export const BOOTSTRAP_VERSION = '{SEMVER}';$",
     "the version the bootstrapper names itself"),
]

GENERATORS: list[tuple[list[str], str]] = [
    ([sys.executable, "scripts/gen-modules.py"], "modules/*.module.json name this release"),
    ([sys.executable, "scripts/gen-api-docs.py"], "docs/API.md matches this app"),
]


def fail(message: str) -> None:
    print(f"bump: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_current() -> str:
    found: dict[str, str] = {}
    for relative, pattern, description in PLACES:
        text = (REPO / relative).read_text(encoding="utf-8")
        matches = re.findall(pattern, text, re.MULTILINE)
        if len(matches) != 1:
            fail(f"{relative} has {len(matches)} places matching {pattern!r}, expected exactly 1 "
                 f"({description}); the file changed shape, so fix this script rather than "
                 f"letting it edit by guesswork")
        found[f"{relative} ({description})"] = matches[0]
    distinct = set(found.values())
    if len(distinct) != 1:
        listed = "\n".join(f"  {value}  {where}" for where, value in sorted(found.items()))
        fail(f"the version places do not agree, so there is no version to bump FROM:\n{listed}")
    return distinct.pop()


def next_version(current: str, wanted: str) -> str:
    major, minor, patch = (int(part) for part in current.split("."))
    if wanted == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if wanted == "minor":
        return f"{major}.{minor + 1}.0"
    if wanted == "major":
        return f"{major + 1}.0.0"
    if not re.fullmatch(SEMVER, wanted):
        fail(f"{wanted!r} is neither major, minor, patch nor a x.y.z version")
    if tuple(int(p) for p in wanted.split(".")) <= (major, minor, patch):
        fail(f"{wanted} is not after {current}; a version is cut once and never goes backwards")
    return wanted


def write_places(current: str, version: str) -> None:
    for relative, pattern, _ in PLACES:
        path = REPO / relative
        text = path.read_text(encoding="utf-8")
        updated, count = re.subn(pattern,
                                 lambda match: match.group(0).replace(current, version),
                                 text, flags=re.MULTILINE)
        assert count == 1, f"{relative}: {count} substitutions after read_current checked for 1"
        path.write_text(updated, encoding="utf-8")
        print(f"  {relative}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set this release's version in all seven places and regenerate what derives from it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("version", help="major, minor, patch, or an explicit x.y.z")
    parser.add_argument("--commit", action="store_true",
                        help="commit the bump (the message names the version and nothing else)")
    args = parser.parse_args()

    current = read_current()
    version = next_version(current, args.version)
    print(f"bump: {current} -> {version}")
    write_places(current, version)

    for command, what in GENERATORS:
        print(f"bump: regenerating so {what}")
        result = subprocess.run(command, cwd=REPO)
        if result.returncode != 0:
            fail(f"{' '.join(command)} failed; the tree is now half-bumped — "
                 f"fix the generator and re-run it, or `git checkout .` to undo")

    changed = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    print("bump: changed\n" + "".join(f"  {line}\n" for line in changed.splitlines()))

    if args.commit:
        subprocess.run(["git", "add", "-A"], cwd=REPO, check=True)
        subprocess.run(["git", "commit", "-m", f"Cut {version}"], cwd=REPO, check=True)
        print(f"bump: committed {version}")
    else:
        print(f"bump: not committed. Verify with ./scripts/release.sh --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
