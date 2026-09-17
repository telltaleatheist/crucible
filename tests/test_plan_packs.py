"""Which packs a release must BUILD, and which it can point at.

Owen, 2026-09-17: *"We don't need to rebuild the env every time we change
something."*

MEASURED, on the real v0.6.6 and v0.6.7 manifests: thirteen packs, three that
genuinely had to be rebuilt, ten that did not. The three took 0, 0 and 1 minute.
One of the ten ran for over forty. The release's wall-clock was that one pack.

These tests hold the DECISION and the two shapes it is turned into — a matrix
and a boolean — not the arithmetic of `release_packs.plan()`, which
`tests/test_release_packs.py` already owns. One owner per question.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from crucible import envpack

ROOT = Path(__file__).resolve().parent.parent


def planner():
    spec = importlib.util.spec_from_file_location(
        "plan_packs", ROOT / "scripts" / "plan_packs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def a_manifest(tmp_path: Path, version: str) -> str:
    """A manifest claiming every declared pack, built from THIS checkout's recipes.

    Built from the recipes rather than from fixture strings on purpose: the
    question "has this recipe changed" is only meaningful against the real ones,
    and a hand-written digest would make every row look changed forever.
    """
    rows = []
    for name, backend in envpack.every_pack():
        target = envpack.pack_target(name, backend)
        part = envpack.part_filename(target.archive_name(version), 0)
        rows.append(
            envpack.PackEntry(
                name=name, backend=backend,
                python=envpack.python_for_target(target).python_version,
                bytes=10, sha256="a" * 64, parts=(part,),
                recipe_sha256=envpack.recipe_digest(target.recipe),
                unpacked_bytes=20, release=version,
            ).to_dict()
        )
    path = tmp_path / "envpacks.json"
    path.write_text(
        json.dumps({"schema": envpack.PACK_SCHEMA, "version": version, "packs": rows}),
        encoding="utf-8",
    )
    return path.resolve().as_uri()


def test_an_unchanged_recipe_is_carried_and_not_built(tmp_path: Path) -> None:
    """THE POINT OF THE WHOLE THING."""
    plan = planner().decide(a_manifest(tmp_path, "0.6.7"), "0.6.8")
    carried = {(row["name"], row["backend"]) for row in plan["carry"]}
    built = {(row["name"], row["backend"]) for row in plan["build"]}
    assert carried, "nothing was carried from a manifest built off these very recipes"
    assert not (carried & built), "a pack cannot be both carried and rebuilt"
    assert carried | built == set(envpack.every_pack()), (
        "every declared pack must be accounted for exactly once"
    )


def test_a_carried_row_keeps_pointing_at_the_release_that_has_the_bytes(
    tmp_path: Path,
) -> None:
    """NOT relabelled to the release being cut, which would be a lie about where
    the parts are — and the parts are named for the old release, so the lie
    would be a 404 rather than a subtle wrong."""
    plan = planner().decide(a_manifest(tmp_path, "0.6.7"), "0.6.8")
    for row in plan["carry"]:
        assert row["release"] == "0.6.7", row
        assert "0.6.7" in row["parts"][0], row["parts"]


def test_the_runtime_packs_are_always_rebuilt(tmp_path: Path) -> None:
    """They embed the Crucible source, so their bytes change with the version
    even when the recipe does not. `release_packs.plan()` decides this; the
    check is here because it is the reason a release is never zero jobs."""
    plan = planner().decide(a_manifest(tmp_path, "0.6.7"), "0.6.8")
    built = {(row["name"], row["backend"]) for row in plan["build"]}
    assert ("server", "cuda-linux") in built
    assert ("server", "mlx-darwin") in built
    assert ("host", "llama-windows") in built


def test_the_windows_host_is_planned_but_never_in_the_matrix(tmp_path: Path) -> None:
    """Every step in the matrix is written in `sh` and reclaims disk with
    `sudo rm -rf`. The host pack builds in a job of its own and is answered with
    a boolean — but it IS planned, because a pack nobody plans is a pack nobody
    notices is missing."""
    module = planner()
    plan = module.decide(a_manifest(tmp_path, "0.6.7"), "0.6.8")
    matrix = [r for r in plan["build"] if (r["name"], r["backend"]) not in module.OWN_JOB]
    assert all(r["backend"] != "llama-windows" for r in matrix)
    assert ("host", "llama-windows") in module.RUNNERS


def test_every_declared_pack_has_a_runner(tmp_path: Path) -> None:
    """THIS FAILED THE FIRST TIME IT RAN, with KeyError on
    `('host', 'llama-windows')`, because that pack is not in the workflow matrix
    and the table was copied from the matrix. A pack with no runner row cannot
    be built and must not be silently skipped, so the absence is an error and
    this is the check that says so before a release finds out."""
    module = planner()
    for key in envpack.every_pack():
        assert key in module.RUNNERS, f"{key} has no runner or disk figure"


def test_no_previous_release_builds_everything(tmp_path: Path) -> None:
    """A slow release is a fine answer to a missing manifest. Guessing is not."""
    plan = planner().decide(None, "0.6.8")
    assert not plan["carry"]
    assert {(r["name"], r["backend"]) for r in plan["build"]} == set(envpack.every_pack())


def test_an_unreadable_previous_release_builds_everything(tmp_path: Path) -> None:
    """Same rule, and it is the one that matters in practice: a network blip
    must cost CI minutes, never a release that quietly ships no packs."""
    missing = (tmp_path / "not-there.json").resolve().as_uri()
    plan = planner().decide(missing, "0.6.8")
    assert not plan["carry"]
    assert len(plan["build"]) == len(envpack.every_pack())
    assert "unreadable" in plan["why"]
