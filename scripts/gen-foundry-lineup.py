#!/usr/bin/env python3
"""Write `foundry-lineup.json` from the model manifests, or check that it is current.

WHY THIS EXISTS. Owen, 2026-09-13, via Foundry: Crucible's model manifests are the
catalog of record for Foundry's LOCAL lineup too — the Ollama / llama.cpp fallback
its app runs when no Crucible is present — so that "what can this machine run" has
one owner. Before this, Foundry carried its own table (`app/electron/llm-catalog.ts`,
sizes read off ollama.com by hand) and its page reader pinned its own GGUF pair,
which is one fact with two owners and nothing comparing them — the exact shape of
every defect in ARCHITECTURE.md section 1.

Foundry vendors the file this writes and compares it by content. Nothing here
parses TOML: every row comes through `crucible/manifests.py` and every `classes`
list through `crucible/capability.py`, so the file cannot say anything the server
does not (`crucible/lineup.py` is the whole of the logic; this is its door).

    python scripts/gen-foundry-lineup.py            # write foundry-lineup.json
    python scripts/gen-foundry-lineup.py --verbose  # ...and say which models were omitted
    python scripts/gen-foundry-lineup.py --check    # exit 1 if the file has drifted

`--check` is what CI runs, and `tests/test_lineup.py` asserts the same equality, so
a manifest edited without regenerating the file is red twice rather than shipped.
The comparison ignores `generated_from` — the commit the generator ran on, which is
always the parent of the commit carrying the file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# THIS checkout's package, not whichever one the interpreter has installed. The
# manifests are found relative to the package (`manifests_dir()` walks up from
# `crucible/__init__.py`), so a script in one checkout importing the package from
# another would read the OTHER checkout's models/ and write them into this one's
# file — two owners of the catalog, silently, which is the defect this whole
# script exists to prevent. Put this checkout first, then prove it won.
sys.path.insert(0, str(REPO_ROOT))

import crucible  # noqa: E402
from crucible import lineup  # noqa: E402
from crucible.errors import CrucibleError  # noqa: E402

_IMPORTED_FROM = Path(crucible.__file__).resolve().parent.parent
if _IMPORTED_FROM != REPO_ROOT:
    raise SystemExit(
        f"refused: `crucible` was imported from {_IMPORTED_FROM}, not from this "
        f"checkout ({REPO_ROOT}). The lineup must be generated from the manifests "
        "beside it; run this with an interpreter whose `crucible` is this checkout "
        "(pip install -e .)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    # LF on every platform: the file is vendored into another repo and compared
    # by content, and a CRLF copy would differ from an LF one in every line.
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(f"wrote {args.output} ({len(rows)} model(s), from {fresh[lineup.PROVENANCE_KEY][:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
