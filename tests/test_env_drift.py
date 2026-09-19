"""What an install DOES about a recipe that moved, and when it refuses instead.

PHASE20-CODE-NOT-ENVIRONMENTS.md section 4: an env is brought to its recipe by
pip, INTO the venv that is already there, and `--force` is the only thing left
that deletes one. These tests hold the line between the edits pip acting on
this recipe will make true and the two it will not — an `--index-url` that
moves which wheel a satisfied pin resolves to, and a requirement that vanished
and stays installed — because a stamp written after either of those would claim
bytes nobody installed.
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


def _env(tmp_path: Path, recipe_text: str | None, *, environment: str | None) -> tuple[Path, Path]:
    """A stamped env directory and its recipe, both on disk."""
    directory = tmp_path / "envs" / "tts-higgs-v3"
    directory.mkdir(parents=True)
    recipe = tmp_path / "r.txt"
    recipe.write_text(BASE, encoding="utf-8")
    record: dict[str, object] = {
        "backend": "cuda-linux",
        "recipe": recipe.name,
        "python_version": "3.12.3",
    }
    if recipe_text is not None:
        record["recipe_text"] = recipe_text
    if environment is not None:
        record["environment_sha256"] = environment
        record["direct_references"] = jobenv.recipe_direct_references(recipe)
    (directory / "crucible-env.json").write_text(json.dumps(record), encoding="utf-8")
    return directory, recipe


def _plan(directory: Path, recipe: Path, *, installed: bool = True) -> jobenv.EnvPlan:
    return jobenv.plan_env(
        directory=directory,
        stamp=directory / "crucible-env.json",
        recipe=recipe,
        backend_kind="cuda-linux",
        installed=installed,
        force=False,
        install_command="crucible install tts",
    )


def test_a_stamp_without_the_recipe_text_is_refused_by_name(tmp_path: Path) -> None:
    """The bytes behind the recorded stamp are gone, so nothing can be proven —
    and `pip install -r` is exactly the thing that would look like it had."""
    directory, recipe = _env(tmp_path, None, environment=None)
    with pytest.raises(jobenv.EnvError) as caught:
        _plan(directory, recipe)
    assert "stamped before the recipe's text was recorded" in str(caught.value)
    assert "--force" in str(caught.value)


def test_an_index_url_change_refuses_rather_than_pipping(tmp_path: Path) -> None:
    """THE CASE THE REFUSAL EXISTS FOR. `torch==2.5.1` from PyPI and from the
    cu121 index are the same version string and different binaries, and pip run
    over the new recipe finds the pin already satisfied and does nothing."""
    directory, recipe = _env(tmp_path, BASE, environment=jobenv.environment_sha256(
        _written(tmp_path / "before.txt", BASE)
    ))
    recipe.write_text(BASE.replace("cu121", "cu124"), encoding="utf-8")
    with pytest.raises(jobenv.EnvError) as caught:
        _plan(directory, recipe)
    assert "cu124" in str(caught.value)
    assert "--force" in str(caught.value)


def test_a_moved_pin_is_pip_into_the_venv_that_is_there(tmp_path: Path) -> None:
    """The ordinary drift, and the whole ruling: the answer is pip over the
    recipe, never a delete and rebuild."""
    before = _written(tmp_path / "before.txt", BASE)
    directory, recipe = _env(
        tmp_path, BASE, environment=jobenv.environment_sha256(before)
    )
    after = BASE.replace("2.5.1", "2.6.0")
    recipe.write_text(after, encoding="utf-8")
    plan = _plan(directory, recipe, installed=False)
    assert plan.action == jobenv.PLAN_RECIPE
    assert plan.lines == ()


def test_a_moved_direct_reference_is_that_one_line_and_nothing_else(
    tmp_path: Path,
) -> None:
    """The 1b case: the environment half hashes the same, so the 13 GB around
    narrator did not move and is not reinstalled."""
    before = _written(tmp_path / "before.txt", BASE)
    directory, recipe = _env(
        tmp_path, BASE, environment=jobenv.environment_sha256(before)
    )
    after = BASE.replace("a" * 40, "b" * 40)
    recipe.write_text(after, encoding="utf-8")
    plan = _plan(directory, recipe)
    assert plan.action == jobenv.PLAN_REFERENCES
    assert plan.lines == (
        "narrator @ git+https://github.com/x/y@%s#subdirectory=python" % ("b" * 40),
    )


def test_an_env_that_matches_its_recipe_is_left_alone(tmp_path: Path) -> None:
    before = _written(tmp_path / "before.txt", BASE)
    directory, recipe = _env(
        tmp_path, BASE, environment=jobenv.environment_sha256(before)
    )
    assert _plan(directory, recipe).action == jobenv.PLAN_NOTHING


def test_a_wrong_backend_is_the_one_drift_that_is_genuinely_a_rebuild(
    tmp_path: Path,
) -> None:
    """A venv full of one backend's wheels is not re-pointed at another's by
    pip, so this refuses by name instead of pretending it can be fixed."""
    directory, recipe = _env(tmp_path, BASE, environment="0" * 64)
    stamp = directory / "crucible-env.json"
    record = json.loads(stamp.read_text(encoding="utf-8"))
    record["backend"] = "mlx-darwin"
    stamp.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(jobenv.EnvError) as caught:
        _plan(directory, recipe)
    assert "installed for backend 'mlx-darwin'" in str(caught.value)
    assert "crucible install tts --force" in str(caught.value)


def test_a_half_finished_install_is_built_again(tmp_path: Path) -> None:
    """No stamp means nothing downstream has ever trusted this venv."""
    directory = tmp_path / "envs" / "tts-higgs-v3"
    directory.mkdir(parents=True)
    recipe = _written(tmp_path / "r.txt", BASE)
    plan = _plan(directory, recipe)
    assert plan.action == jobenv.PLAN_BUILD
    assert "did not finish" in plan.detail


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
