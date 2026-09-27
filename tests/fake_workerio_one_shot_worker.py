from __future__ import annotations

import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "crucible", "jobs"),
)
import workerio

sys.path.pop(0)
workerio.claim_stdout()


def main() -> int:
    request = workerio.read_request("fake")
    if request is None:
        return 1
    workerio.send("result", text=request["text"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
