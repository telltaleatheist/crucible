#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))

import crucible
from crucible import lineup
from crucible.errors import CrucibleError

_IMPORTED_FROM = Path(crucible.__file__).resolve().parent.parent
if _IMPORTED_FROM != REPO_ROOT:
    raise SystemExit(
        f"refused: `crucible` was imported from {_IMPORTED_FROM}, not from this "
        f"checkout ({REPO_ROOT}). The lineup must be generated from the manifests "
        "beside it; run this with an interpreter whose `crucible` is this checkout "
        "(pip install -e .)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write foundry-lineup.json from the model manifests, or check that it is current.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the checked-in file with a fresh build; write nothing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="name the models omitted for having no [local] table",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / lineup.FILE_NAME,
        help=f"where the file lives (default: <repo>/{lineup.FILE_NAME})",
    )
    args = parser.parse_args()

    try:
        rows, omitted = lineup.build()
        fresh = lineup.document(rows, lineup.git_head(REPO_ROOT))
    except CrucibleError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        for model_id in omitted:
            print(f"omitted {model_id}: no [local] table, so no machine without Crucible runs it")
        print(f"{len(rows)} model(s) with a local form, {len(omitted)} omitted")

    if args.check:
        if not args.output.is_file():
            print(
                f"no {args.output}; run scripts/gen-foundry-lineup.py to write it",
                file=sys.stderr,
            )
            return 1
        problems = lineup.check(args.output.read_text(encoding="utf-8"), fresh)
        if problems:
            print(f"{args.output.name} HAS DRIFTED FROM THE MANIFESTS:\n", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            print(
                "\nThe manifests win: they are the catalog of record. Regenerate the "
                "file with\n  python scripts/gen-foundry-lineup.py\nand commit it "
                "alongside the manifest that moved.",
                file=sys.stderr,
            )
            return 1
        print(f"{args.output.name} matches the manifests ({len(rows)} model(s)).")
        return 0

    text = lineup.render(fresh)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(f"wrote {args.output} ({len(rows)} model(s), from {fresh[lineup.PROVENANCE_KEY][:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
