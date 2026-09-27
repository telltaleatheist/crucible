from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

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


CANONICAL = PLACES[0][0]


def read_places() -> dict[str, str]:
    found: dict[str, str] = {}
    for relative, pattern, description in PLACES:
        text = (REPO / relative).read_text(encoding="utf-8")
        matches = re.findall(pattern, text, re.MULTILINE)
        if len(matches) != 1:
            fail(f"{relative} has {len(matches)} places matching {pattern!r}, expected exactly 1 "
                 f"({description}); the file changed shape, so fix PLACES in scripts/bump.py "
                 f"rather than letting it edit by guesswork")
        found[f"{relative} ({description})"] = matches[0]
    return found


def read_current() -> str:
    found = read_places()
    distinct = set(found.values())
    if len(distinct) != 1:
        listed = "\n".join(f"  {value}  {where}" for where, value in found.items())
        fail(f"the version places do not agree:\n{listed}\n"
             f"Set every place to the version {CANONICAL} names with: "
             f"python scripts/bump.py --align")
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


def write_places(version: str) -> None:
    for relative, pattern, _ in PLACES:
        path = REPO / relative
        text = path.read_text(encoding="utf-8")
        updated, count = re.subn(pattern,
                                 lambda match: match.group(0).replace(match.group(1), version),
                                 text, flags=re.MULTILINE)
        assert count == 1, f"{relative}: {count} substitutions after read_places checked for 1"
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(updated)
        print(f"  {relative}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set this release's version in all seven places and regenerate what derives from it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("version", nargs="?", help="major, minor, patch, or an explicit x.y.z")
    parser.add_argument("--commit", action="store_true",
                        help="commit the bump (the message names the version and nothing else)")
    parser.add_argument("--check", action="store_true",
                        help="print the version all seven places agree on, or name each place "
                             "and exit 1 if they do not; change nothing")
    parser.add_argument("--align", action="store_true",
                        help=f"set every place to the version {CANONICAL} names")
    args = parser.parse_args()

    if (args.check or args.align) and (args.version or args.commit or (args.check and args.align)):
        parser.error("--check and --align each stand alone: "
                     "python scripts/bump.py --check, or python scripts/bump.py --align")
    if args.check:
        print(read_current())
        return 0
    if args.align:
        version = next(iter(read_places().values()))
        print(f"bump: every place -> {version}, the version {CANONICAL} names")
    elif args.version:
        current = read_current()
        version = next_version(current, args.version)
        print(f"bump: {current} -> {version}")
    else:
        parser.error("say what to bump to: python scripts/bump.py patch "
                     "(or minor, major, an explicit x.y.z; --check only reads)")
    write_places(version)

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
        print("bump: not committed. Verify with ./scripts/release.sh --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
