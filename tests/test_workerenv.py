"""`crucible.workerenv` — recipes, pins, and the one requirement that is a commit.

`asr`'s recipe is `name==version` all the way down, which is what this module was
written for. `rvc`'s cannot be: `generate convert-dir` is in Owen's fork and not
on PyPI, and both call themselves `ultimate-rvc 0.5.11` — so the version is not
an identity and the commit is. These are the tests for that difference.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible import workerenv

FORK = (
    "ultimate-rvc @ git+https://github.com/telltaleatheist/ultimate-rvc"
    "@05cc3f1ba921f070e16eaf3e1f188073c31c0101"
)
URL = FORK.split(" @ ", 1)[1]


def recipe(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cuda-linux.txt"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------- the shipped


def test_the_rvc_recipe_pins_a_commit_not_a_branch() -> None:
    path = workerenv.recipe_for("rvc", "cuda-linux")
    refs = workerenv.recipe_direct_refs(path)
    assert refs["ultimate-rvc"].endswith("@05cc3f1ba921f070e16eaf3e1f188073c31c0101")
    # And it is NOT in the version pins, because `pip list` cannot answer it.
    assert "ultimate-rvc" not in workerenv.recipe_pins(path)


def test_both_rvc_recipes_pin_the_same_commit() -> None:
    """An RVC checkpoint is not quantised per backend and neither is the engine."""
    linux = workerenv.recipe_direct_refs(workerenv.recipe_for("rvc", "cuda-linux"))
    mac = workerenv.recipe_direct_refs(workerenv.recipe_for("rvc", "mlx-darwin"))
    assert linux == mac


def test_the_align_recipe_is_all_version_pins() -> None:
    path = workerenv.recipe_for("align", "cuda-linux")
    assert workerenv.recipe_direct_refs(path) == {}
    pins = workerenv.recipe_pins(path)
    assert pins["qwen-asr"] == "0.0.6"
    assert pins["torch"] == "2.14.0"


def test_every_recipe_that_ships_has_a_headline_package() -> None:
    """`crucible doctor` reads it to describe the env at a glance.

    Per (job type, BACKEND), since `asr` gained a second engine: cuda-linux
    installs faster-whisper and mlx-darwin installs mlx-whisper. A recipe file
    with no entry here would be a KeyError in the doctor rather than a message,
    so the test walks the recipes that exist rather than the job types.
    """
    for job_type in workerenv.WORKER_JOB_TYPES:
        for path in workerenv.recipes_dir(job_type).glob("*.txt"):
            assert (job_type, path.stem) in workerenv.HEADLINE_PACKAGE, path


def test_every_headline_package_is_in_its_own_recipe() -> None:
    """The one that would have shipped a broken Mac `asr` env: before the table
    was keyed by backend, `envs/asr/mlx-darwin.txt` was checked for
    faster-whisper, which it will never contain."""
    for job_type in workerenv.WORKER_JOB_TYPES:
        for path in workerenv.recipes_dir(job_type).glob("*.txt"):
            headline = workerenv.headline_package(job_type, path.stem)
            named = set(workerenv.recipe_pins(path)) | set(
                workerenv.recipe_direct_refs(path)
            )
            assert headline in named, f"{path.name} does not install {headline}"


def test_a_pair_nobody_decided_is_refused_by_name() -> None:
    with pytest.raises(workerenv.WorkerEnvError) as caught:
        workerenv.headline_package("asr", "llama-windows")
    assert "no headline package" in str(caught.value)


# ------------------------------------------------------------------ refusals


def test_a_branch_is_not_a_pin(tmp_path: Path) -> None:
    path = recipe(
        tmp_path, "ultimate-rvc @ git+https://github.com/x/y@bookforge\n"
    )
    with pytest.raises(workerenv.WorkerEnvError) as caught:
        workerenv.recipe_pins(path)
    assert "a branch name is not a pin" in str(caught.value)


def test_an_unpinned_requirement_is_refused(tmp_path: Path) -> None:
    with pytest.raises(workerenv.WorkerEnvError) as caught:
        workerenv.recipe_pins(recipe(tmp_path, "torch>=2.7\n"))
    assert "pinned exactly" in str(caught.value)


def test_index_urls_and_comments_are_not_requirements(tmp_path: Path) -> None:
    path = recipe(
        tmp_path,
        "# a comment\n--extra-index-url https://example/whl\n\ntorch==2.7.0+cu128\n",
    )
    assert workerenv.recipe_pins(path) == {"torch": "2.7.0+cu128"}
    assert workerenv.recipe_direct_refs(path) == {}


# ------------------------------------------------------------------- status


def _stamped(home: Path, job_type: str, backend: str = "cuda-linux") -> Path:
    directory = workerenv.worker_env_dir(home, job_type)
    (directory / "bin").mkdir(parents=True)
    (directory / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (directory / "crucible-env.json").write_text(
        '{"job_type": "%s", "backend": "%s", "recipe": "%s.txt", '
        '"python_version": "3.11.16", "seconds": 1.0}' % (job_type, backend, backend),
        encoding="utf-8",
    )
    return directory


def test_a_matching_env_names_the_commit_and_not_the_version(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """0.5.11 is what pip reports for the fork AND for PyPI's release, so the
    doctor line that said it would be telling nobody anything."""
    _stamped(home, "rvc")
    path = workerenv.recipe_for("rvc", "cuda-linux")
    monkeypatch.setattr(
        workerenv,
        "installed_packages",
        lambda _h, _t: {**workerenv.recipe_pins(path), "ultimate-rvc": "0.5.11"},
    )
    monkeypatch.setattr(
        workerenv,
        "installed_direct_refs",
        lambda _h, _t: dict(workerenv.recipe_direct_refs(path)),
    )
    status = workerenv.env_status(home, "rvc", "cuda-linux")
    assert status.installed is True
    assert "ultimate-rvc @ 05cc3f1ba921" in status.detail
    assert "0.5.11" not in status.detail


def test_the_wrong_commit_is_not_ready_and_says_which(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure this exists to catch: the PyPI release installed instead of
    the fork, reporting the same version and lacking `convert-dir`."""
    _stamped(home, "rvc")
    path = workerenv.recipe_for("rvc", "cuda-linux")
    monkeypatch.setattr(
        workerenv, "installed_packages", lambda _h, _t: dict(workerenv.recipe_pins(path))
    )
    monkeypatch.setattr(workerenv, "installed_direct_refs", lambda _h, _t: {})
    status = workerenv.env_status(home, "rvc", "cuda-linux")
    assert status.installed is False
    assert "ultimate-rvc is absent" in status.detail
    assert URL in status.detail


def test_a_headline_that_is_not_installed_is_a_build_bug_said_out_loud(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stamped(home, "align")
    path = workerenv.recipe_for("align", "cuda-linux")
    pins = workerenv.recipe_pins(path)
    del pins["qwen-asr"]
    monkeypatch.setattr(workerenv, "installed_packages", lambda _h, _t: dict(pins))
    # Every remaining pin matches, so the only thing wrong is the headline —
    # which `env_status` must not answer with a KeyError.
    monkeypatch.setattr(
        workerenv, "recipe_pins", lambda _p: {k: v for k, v in pins.items()}
    )
    with pytest.raises(workerenv.WorkerEnvError) as caught:
        workerenv.env_status(home, "align", "cuda-linux")
    assert "the package the align env exists for" in str(caught.value)
