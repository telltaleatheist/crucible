#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))

import crucible
from crucible import modules
from crucible.errors import CrucibleError

_IMPORTED_FROM = Path(crucible.__file__).resolve().parent.parent
if _IMPORTED_FROM != REPO_ROOT:
    raise SystemExit(
        f"refused: `crucible` was imported from {_IMPORTED_FROM}, not from this "
        f"checkout ({REPO_ROOT}). A module must be generated from the manifests "
        "beside it; run this with an interpreter whose `crucible` is this "
        "checkout (pip install -e .)."
    )


def declarations(directory: Path) -> list[Path]:
    found = sorted(directory.glob("*.toml"))
    if not found:
        raise SystemExit(
            f"refused: no declarations in {directory}. A run that generates "
            "nothing is a broken checkout, not a clean one"
        )
    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write each app's <app>.module.json from modules/<app>.toml, or check it.",
    )
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
