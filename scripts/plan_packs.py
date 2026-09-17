"""Decide which packs this release must BUILD, and which it can point at.

An environment pack is a function of its RECIPE, not of the version beside it.
When a recipe and its standalone-Python pin are unchanged, the pack the previous
release built is byte-for-byte the pack this one would build, and building it
again costs CI minutes to arrive back where we started. Owen, 2026-09-17:
*"We don't need to rebuild the env every time we change something."*

MEASURED on v0.6.6 -> 0.6.7: of thirteen packs, three needed rebuilding and ten
did not. The three took 0, 0 and 1 minute; the ten included a `tts-higgs-v3`
build that ran for over forty. That is the whole argument.

WHAT THIS DOES NOT DO IS COPY. `scripts/release_packs.py` can download a pack
from the old release and upload it to the new one, and at 17 GB of unchanged
packs per release that is neither fast nor free — it is the same bytes stored
again under every tag, forever. A carried row instead keeps naming the release
that already holds it (`PackEntry.release`), so an unchanged pack is built once
and stored once.

The decision itself is `release_packs.plan()`, which this does not duplicate:
one owner for "is this pack still the same pack". This turns its answer into the
two things a workflow needs — a matrix to build, and rows to carry.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crucible import VERSION, envpack  # noqa: E402

#: How much free disk each pack's build needs, in GB, and which runner builds
#: it. The matrix in `.github/workflows/envpacks.yml` used to hold this inline;
#: it is here now because the matrix is COMPUTED, and a list that decides what
#: runs must sit beside the code that decides whether it runs at all.
RUNNERS: dict[tuple[str, str], dict[str, object]] = {
    ("server", "cuda-linux"): {"runner": "ubuntu-latest", "needs_gb": 2},
    ("asr", "cuda-linux"): {"runner": "ubuntu-latest", "needs_gb": 6},
    ("align", "cuda-linux"): {"runner": "ubuntu-latest", "needs_gb": 20},
    ("rvc", "cuda-linux"): {"runner": "ubuntu-latest", "needs_gb": 24},
    ("llm", "cuda-linux"): {"runner": "ubuntu-latest", "needs_gb": 30},
    ("tts-higgs-v3", "cuda-linux"): {"runner": "ubuntu-latest", "needs_gb": 30},
    ("server", "mlx-darwin"): {"runner": "macos-14", "needs_gb": 2},
    ("rvc", "mlx-darwin"): {"runner": "macos-14", "needs_gb": 20},
    ("align", "mlx-darwin"): {"runner": "macos-14", "needs_gb": 12},
    ("asr", "mlx-darwin"): {"runner": "macos-14", "needs_gb": 12},
    ("llm", "mlx-darwin"): {"runner": "macos-14", "needs_gb": 10},
    ("tts", "mlx-darwin"): {"runner": "macos-14", "needs_gb": 25},
    # NOT IN THE MATRIX, AND IT IS STILL A PACK. `host/llama-windows` has a row
    # in `envpacks.json` and `install.ps1` fetches it by name, but it builds in
    # a job of its own because every step in the matrix is written in `sh` and
    # reclaims disk with `sudo rm -rf`, which is not a thing a Windows runner
    # does. It is here because THIS map answers "can this pack be skipped", and
    # a pack missing from it raised KeyError the first time this ran — which is
    # the right failure: a pack nobody planned must not quietly not be built.
    ("host", "llama-windows"): {"runner": "windows-latest", "needs_gb": 8},
}

#: The packs that build outside the matrix, by the job that builds them. The
#: workflow needs to know whether to run that job at all.
OWN_JOB = {("host", "llama-windows"): "host"}


def _planner():
    """`scripts/release_packs.py`, imported by path — it is a script, not a module."""
    path = Path(__file__).resolve().parent / "release_packs.py"
    spec = importlib.util.spec_from_file_location("release_packs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decide(previous: str | None, version: str) -> dict:
    """The build matrix and the carried rows, for `version`.

    `previous` is a manifest LOCATION — a release URL or a file — or None for a
    release with nothing to compare against, where every pack is built. A
    previous release that cannot be read is also "build everything": a slow
    release is a fine answer to a missing manifest, and guessing is not.
    """
    every = envpack.every_pack()
    if previous is None:
        return _all_of_them(every, version, "no previous release to compare against")
    try:
        source = envpack.read_manifest(previous)
    except envpack.PackError as exc:
        return _all_of_them(every, version, f"previous manifest unreadable: {exc}")

    plan = _planner().plan(source, version)
    build: list[dict] = []
    carry: list[dict] = []
    for row in plan["packs"]:
        key = (row["name"], row["backend"])
        if row["action"] == "reuse":
            entry = source.require(row["name"], row["backend"])
            # The row KEEPS the release it already names, which is not always
            # the source release: a pack carried from 0.6.5 into 0.6.6 arrives
            # here still naming 0.6.5, and carrying it again must not relabel it
            # as 0.6.6 — the bytes never moved.
            carry.append(entry.to_dict())
        else:
            build.append({"name": row["name"], "backend": row["backend"],
                          "reason": row["reason"], **RUNNERS[key]})
    return {"version": version, "previous": source.version,
            "build": build, "carry": carry}


def _all_of_them(every, version: str, why: str) -> dict:
    return {
        "version": version,
        "previous": None,
        "why": why,
        "build": [{"name": n, "backend": b, "reason": why, **RUNNERS[(n, b)]}
                  for n, b in every],
        "carry": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", default=None,
                        help="manifest location of the release to compare against")
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--github-output", action="store_true",
                        help="also write build/carry to $GITHUB_OUTPUT")
    args = parser.parse_args()

    plan = decide(args.previous, args.version)
    for row in plan["build"]:
        print(f"BUILD  {row['name']}/{row['backend']}  — {row['reason']}")
    for row in plan["carry"]:
        print(f"CARRY  {row['name']}/{row['backend']}  — already in v{row['release']}")
    print(f"{len(plan['build'])} to build, {len(plan['carry'])} carried")

    if args.github_output:
        out = os.environ["GITHUB_OUTPUT"]
        with open(out, "a", encoding="utf-8") as handle:
            # THE MATRIX ROWS ONLY. `host/llama-windows` builds in a job of
            # its own -- every step in the matrix is written in `sh` and
            # reclaims disk with `sudo rm -rf`, neither of which a Windows
            # runner does -- so it is planned like any other pack and
            # answered with a boolean instead of a matrix row.
            matrix = [row for row in plan["build"]
                      if (row["name"], row["backend"]) not in OWN_JOB]
            build_host = any((row["name"], row["backend"]) == ("host", "llama-windows")
                             for row in plan["build"])
            handle.write(f"build={json.dumps(matrix)}\n")
            handle.write(f"carry={json.dumps(plan['carry'])}\n")
            handle.write(f"any_to_build={'true' if matrix else 'false'}\n")
            handle.write(f"build_host={'true' if build_host else 'false'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
