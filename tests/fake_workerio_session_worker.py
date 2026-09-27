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


def echo(request: dict) -> None:
    workerio.send("result", text=workerio.require(request, "text", str, "fake", "it is echoed"))
    workerio.send("done")


def missing(request: dict) -> None:
    workerio.require(request, "text", str, "fake", "it is echoed")


def explode(request: dict) -> None:
    raise RuntimeError("boom")


OPS = {"echo": echo, "explode": explode, "missing": missing}

if __name__ == "__main__":
    sys.exit(workerio.serve("fake", OPS))
