#!/usr/bin/env python3

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
