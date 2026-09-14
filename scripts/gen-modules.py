#!/usr/bin/env python3
"""Write each app's `<app>.module.json` from its declaration, or check it.

WHY THIS EXISTS. Foundry, reviewing PHASE13-OPERATOR.md on 2026-09-14: a module
file typed beside `foundry-lineup.json` would restate the same model ids with
nothing comparing them, which is ARCHITECTURE.md R1's defect introduced on
purpose in the one file whose job is to be correct about ids. So an app
declares what it needs in `modules/<app>.toml` — job types, capability classes,
and any subject it names outright — and this resolves every one of those
against the manifests beside it.

    python scripts/gen-modules.py            # write modules/*.module.json
    python scripts/gen-modules.py --check    # exit 1 if a file has drifted

`--check` is what CI runs, beside `gen-foundry-lineup.py --check`, and
`tests/test_modules.py` asserts the same equality — so a manifest edited
without regenerating is red twice rather than shipped. Unlike the lineup's
check nothing is ignored: a module carries no provenance key and its version is
a hash of its own content, so a drifted version IS a drifted module.

This script deliberately does NOT touch `foundry-lineup.json`. That file has
its own generator and its own guard, and folding them together would mean one
command whose two halves fail for unrelated reasons.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# THIS checkout's package, not whichever one the interpreter has installed —
# `gen-foundry-lineup.py`'s rule and its reason: the manifests are found
# relative to the package, so a script in one checkout importing the package
# from another would write the OTHER checkout's catalog into this one's files.
sys.path.insert(0, str(REPO_ROOT))

import crucible  # noqa: E402
from crucible import modules  # noqa: E402
from crucible.errors import CrucibleError  # noqa: E402

_IMPORTED_FROM = Path(crucible.__file__).resolve().parent.parent
if _IMPORTED_FROM != REPO_ROOT:
    raise SystemExit(
        f"refused: `crucible` was imported from {_IMPORTED_FROM}, not from this "
        f"checkout ({REPO_ROOT}). A module must be generated from the manifests "
        "beside it; run this with an interpreter whose `crucible` is this "
        "checkout (pip install -e .)."
    )


def declarations(directory: Path) -> list[Path]:
    """Every `<app>.toml` in `modules/`, in name order. Never zero.

    An empty run is a broken checkout, not a clean one — the same rule
    `sdk/ts/scripts/unit.mjs` states about finding no tests.
    """
    found = sorted(directory.glob("*.toml"))
    if not found:
        raise SystemExit(
            f"refused: no declarations in {directory}. A run that generates "
            "nothing is a broken checkout, not a clean one"
        )
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the checked-in files with a fresh build; write nothing",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=REPO_ROOT / modules.DIR_NAME,
        help=f"where declarations and modules live (default: <repo>/{modules.DIR_NAME})",
    )
    args = parser.parse_args()

    failed = False
    for declaration_path in declarations(args.directory):
        try:
            declaration = modules.read_declaration(declaration_path)
            fresh = modules.build(declaration, declaration_path.name)
        except CrucibleError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 1

        target = args.directory / modules.file_name(fresh["name"])
        if args.check:
            if not target.is_file():
                print(
                    f"no {target}; run scripts/gen-modules.py to write it",
                    file=sys.stderr,
                )
                failed = True
                continue
            problems = modules.check(target.read_text(encoding="utf-8"), fresh)
            if problems:
                failed = True
                print(f"{target.name} HAS DRIFTED FROM THE MANIFESTS:\n", file=sys.stderr)
                for problem in problems:
                    print(f"  {problem}", file=sys.stderr)
                print(
                    "\nThe manifests win: they are the catalog of record. "
                    "Regenerate with\n  python scripts/gen-modules.py\nand commit "
                    "the file alongside the manifest that moved.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"{target.name} matches the manifests "
                    f"({len(fresh['job_types'])} job type(s), "
                    f"{len(fresh['subjects'])} subject(s))."
                )
            continue

        # LF on every platform: the file is vendored into another repo and
        # compared by content, and a CRLF copy would differ in every line.
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(modules.render(fresh))
        print(
            f"wrote {target} ({len(fresh['job_types'])} job type(s), "
            f"{len(fresh['subjects'])} subject(s), version {fresh['version']})"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
