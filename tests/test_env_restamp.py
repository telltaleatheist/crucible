"""Correcting a stamp whose recipe moved, and refusing when that is not provable.

`crucible doctor` has always been able to say an env's recipe MOVED, and never
what moved. The remedy it printed - "re-run `crucible install`" - reached
`install_env`, found the env installed, and returned it unchanged, so the only
thing that ever cleared the warning was `--force`: a multi-GB rebuild of a
working env to correct a line of JSON. These tests hold the line between the
edits that can be re-stamped and the ones that genuinely need the rebuild.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible import jobenv


BASE = """--extra-index-url https://download.pytorch.org/whl/cu121
# a comment
torch==2.5.1
narrator @ git+https://github.com/x/y@%s#subdirectory=python
""" % ("a" * 40)


def _changes(before: str, after: str) -> list[str]:
    return jobenv.unverifiable_recipe_changes(before, after, "r.txt")


def test_a_comment_only_edit_is_not_a_reason_to_rebuild() -> None:
    after = BASE.replace("# a comment", "# a much longer comment\n# and another")
    assert _changes(BASE, after) == []


def test_a_moved_version_pin_is_left_to_the_package_check() -> None:
    # `env_status` compares every pin against `pip list`, so a pin that moved
    # is already answered exactly; saying so twice is how the two answers start
    # to disagree.
    assert _changes(BASE, BASE.replace("2.5.1", "2.6.0")) == []


def test_a_moved_direct_reference_is_left_to_the_package_check() -> None:
    assert _changes(BASE, BASE.replace("a" * 40, "b" * 40)) == []


def test_a_changed_index_url_is_refused_because_the_pin_would_still_match() -> None:
    """The case the whole refusal exists for.

    `torch==2.5.1` from PyPI and from the cu121 index are the same version
    string and different binaries - one of them with no CUDA in it at all - and
    nothing `env_status` reads afterwards can tell which one is installed.
    """
    after = BASE.replace("cu121", "cu124")
    problems = _changes(BASE, after)
    assert problems, "a changed index URL must never be re-stamped away"
    assert any("cu124" in p for p in problems)


def test_a_dropped_requirement_is_refused_even_though_every_check_passes() -> None:
    """`env_status` asks what is PRESENT, never what is extra."""
    after = BASE.replace("torch==2.5.1\n", "")
    assert _changes(BASE, after) == ["r.txt no longer requires 'torch', which is still installed"]


def test_an_added_requirement_is_fine_because_absence_would_have_failed() -> None:
    after = BASE + "soundfile==0.12.1\n"
    assert _changes(BASE, after) == []


# ----------------------------------------------------------- the install door


class _Spec:
    """The two attributes `_restamp` reads off a spec."""

    job_type = "tts"
    key = "tts-higgs-v3"


def _stamp(tmp_path: Path, **over: object) -> jobenv.EnvStatus:
    return jobenv.EnvStatus(
        installed=True,
        path=tmp_path,
        detail="",
        python_version="3.12.3",
        packages={},
        pack_sha256=None,
        **{"recipe_sha256": "0" * 64, "recipe_text": BASE, **over},  # type: ignore[arg-type]
    )


def test_a_stamp_without_the_recipe_text_is_refused_by_name(tmp_path: Path) -> None:
    """The bytes behind the recorded hash are gone, so nothing can be proven."""
    recipe = tmp_path / "r.txt"
    recipe.write_text(BASE, encoding="utf-8")
    with pytest.raises(jobenv.EnvError) as caught:
        jobenv._restamp(
            tmp_path, _Spec(), "cuda-linux",  # type: ignore[arg-type]
            existing=_stamp(tmp_path, recipe_text=None),
            recipe=recipe,
            here="1" * 64,
        )
    assert "predates recording the recipe's text" in str(caught.value)
    assert "--force" in str(caught.value)


def test_an_index_url_change_refuses_rather_than_stamping(tmp_path: Path) -> None:
    recipe = tmp_path / "r.txt"
    recipe.write_text(BASE.replace("cu121", "cu124"), encoding="utf-8")
    with pytest.raises(jobenv.EnvError) as caught:
        jobenv._restamp(
            tmp_path, _Spec(), "cuda-linux",  # type: ignore[arg-type]
            existing=_stamp(tmp_path), recipe=recipe, here="1" * 64,
        )
    assert "cu124" in str(caught.value)


def test_a_pin_only_change_rewrites_the_stamp_and_records_the_new_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    env = home / "envs" / "tts-higgs-v3"
    env.mkdir(parents=True)
    monkeypatch.setattr(jobenv, "stamp_path", lambda h, s: env / "crucible-env.json")

    recipe = tmp_path / "r.txt"
    after = BASE.replace("a" * 40, "b" * 40)
    recipe.write_text(after, encoding="utf-8")

    said: list[str] = []
    jobenv._restamp(
        home, _Spec(), "cuda-linux",  # type: ignore[arg-type]
        existing=_stamp(tmp_path), recipe=recipe,
        here=jobenv.recipe_sha256(recipe), on_line=said.append,
    )

    record = json.loads((env / "crucible-env.json").read_text(encoding="utf-8"))
    assert record["recipe_sha256"] == jobenv.recipe_sha256(recipe)
    # THE TEXT, not only the hash: without it the next drift is unprovable all
    # over again, which is the defect this whole change is about.
    assert record["recipe_text"] == after
    assert record["python_version"] == "3.12.3", "a re-stamp invents no facts"
    # It says which pin moved, because "two hashes differ" is what was useless.
    assert any("narrator" in line and "b" * 40 in line for line in said)


def test_the_normalisation_is_the_one_the_hash_uses(tmp_path: Path) -> None:
    """A stamp whose text says CRLF and whose hash was taken over LF disagrees
    with itself, and the disagreement only shows up on the other platform."""
    recipe = tmp_path / "r.txt"
    recipe.write_bytes(BASE.replace("\n", "\r\n").encode())
    assert jobenv.recipe_text(recipe) == BASE
    assert jobenv.recipe_sha256(recipe) == jobenv.recipe_sha256(
        _written(tmp_path / "lf.txt", BASE)
    )


def _written(path: Path, text: str) -> Path:
    path.write_bytes(text.encode())
    return path
