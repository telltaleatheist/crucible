#!/usr/bin/env python3
"""One field out of a task document, for `scripts/testrun-phase15.sh`.

`python task_field.py <file.json> state` prints the state;
`python task_field.py <file.json> error.code` prints the refusal's name, or an
empty line when there is no error. Dotted paths only, no defaults, no jq: the
test runner is POSIX sh on Windows and `jq` is not a thing this machine has.

A document that will not parse prints NOTHING and exits 0, because the caller
polls: an answer that has not arrived yet is not an error, and the caller
decides when it has waited long enough. Every OTHER kind of wrongness — a
missing key, a path through something that is not an object — also prints
nothing, and the caller's own assertion is what names it. This script makes no
judgements; it reads one field.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {Path(sys.argv[0]).name} <file.json> <dotted.path>", file=sys.stderr)
        return 2
    try:
        document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print("")
        return 0
    value = document
    for segment in sys.argv[2].split("."):
        if not isinstance(value, dict):
            print("")
            return 0
        value = value.get(segment)
        if value is None:
            print("")
            return 0
    print(value if isinstance(value, str) else json.dumps(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
