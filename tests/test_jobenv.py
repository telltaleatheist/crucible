"""A job type's env recipes and its status (PHASE2-LLM.md section 2).

None of these build a venv — that is minutes of pip. They test what the server
reads *about* an env, which is what decides whether a load is refused with
`env_missing`.

The `tts` envs are here too, because their naming rule is the interesting half:
`cuda-linux` has one venv per narrator engine and `mlx-darwin` has one for both
(PHASE3-TTS.md section 4), and a rule with a branch in it is a rule that needs a
test on each side.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible import jobenv
from crucible.jobenv import (
    EnvError,
    env_dir,
    env_status,
    llm_env,
    recipe_for,
    recipe_pins,
    recipes_dir,
    require_env,
    tts_env,
)


def stamp_env(home: Path, backend_kind: str) -> Path:
    """A venv directory shaped the way `crucible install llm` leaves one."""
    directory = env_dir(home, llm_env(backend_kind))
    (directory / "bin").mkdir(parents=True, exist_ok=True)
    (directory / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (directory / "crucible-env.json").write_text(
        json.dumps(
            {
                "backend": backend_kind,
                "recipe": f"{backend_kind}.txt",
                "python_version": "3.11.16",
                "seconds": 196.4,
            }
        ),
        encoding="utf-8",
    )
    return directory


# ------------------------------------------------------------------ recipes


def test_this_build_ships_a_recipe_for_each_backend() -> None:
    assert sorted(p.stem for p in recipes_dir("llm").glob("*.txt")) == [
        "cuda-linux",
        "mlx-darwin",
    ]


def test_a_backend_with_no_recipe_is_refused_by_name() -> None:
    with pytest.raises(EnvError) as caught:
        recipe_for(jobenv.EnvSpec("llm", "llm", "rocm-linux", "vllm"))
    assert "no llm env recipe for 'rocm-linux'" in str(caught.value)
    assert "['cuda-linux', 'mlx-darwin']" in str(caught.value)


def test_each_recipe_pins_its_engine_exactly() -> None:
    assert recipe_pins(recipe_for(llm_env("cuda-linux")))["vllm"] == "0.29.0"
    assert recipe_pins(recipe_for(llm_env("mlx-darwin")))["mlx-lm"] == "0.31.3"


def test_every_requirement_in_every_recipe_is_pinned() -> None:
    """A recipe with a floating requirement is not a recipe."""
    for backend in ("cuda-linux", "mlx-darwin"):
        pins = recipe_pins(recipe_for(llm_env(backend)))
        assert len(pins) > 1, f"{backend} pins only its engine"
        for name, version in pins.items():
            assert version, f"{backend}: {name} has no version"


def test_every_tts_recipe_pins_the_same_narrator_commit() -> None:
    """narrator is ONE package, and a recipe is a statement about bytes.

    The three `tts` recipes each carry their own direct reference because each
    names a different extra, so the sha is written three times — and three
    copies of one fact is the shape `docs/ARCHITECTURE.md` §1 names. A bump that
    lands on two of the three gives a host whose Higgs env speaks one wire and
    whose Orpheus env speaks another, and `crucible doctor` calls both installed
    because each matches the recipe that built it.
    """
    shas = {
        spec.recipe_name: jobenv.recipe_direct_references(recipe_for(spec))["narrator"]
        for spec in (
            tts_env("higgs-v3", "cuda-linux"),
            tts_env("orpheus", "cuda-linux"),
            tts_env("higgs-v3", "mlx-darwin"),
        )
    }
    assert len(set(shas.values())) == 1, shas


def test_an_unpinned_requirement_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "cuda-linux.txt"
    path.write_text("# a comment\nvllm==0.29.0\ntorch>=2.0\n", encoding="utf-8")
    with pytest.raises(EnvError) as caught:
        recipe_pins(path)
    assert "is not a `name==version` pin" in str(caught.value)


def test_recipe_names_are_normalised(tmp_path: Path) -> None:
    """`huggingface_hub` in a freeze is `huggingface-hub` to pip list."""
    path = tmp_path / "x.txt"
    path.write_text("huggingface_hub==1.31.0\n", encoding="utf-8")
    assert recipe_pins(path) == {"huggingface-hub": "1.31.0"}


# ------------------------------------------------------------------- status


def test_no_venv_is_not_installed(home: Path) -> None:
    status = env_status(home, llm_env("cuda-linux"), "cuda-linux")
    assert status.installed is False
    assert "crucible install llm" in status.detail


def test_a_venv_with_no_stamp_is_not_installed(home: Path) -> None:
    """An interrupted install must never be handed to an engine."""
    directory = env_dir(home, llm_env("cuda-linux"))
    (directory / "bin").mkdir(parents=True)
    (directory / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    status = env_status(home, llm_env("cuda-linux"), "cuda-linux")
    assert status.installed is False
    assert "did not finish" in status.detail


def test_an_env_built_for_another_backend_is_refused(home: Path) -> None:
    """The wrong-backend refusal: an mlx env on a CUDA host is not an llm env."""
    stamp_env(home, "mlx-darwin")
    status = env_status(home, llm_env("cuda-linux"), "cuda-linux")
    assert status.installed is False
    assert "installed for backend 'mlx-darwin'" in status.detail
    assert "this host is 'cuda-linux'" in status.detail
    assert "--force" in status.detail


def test_an_env_missing_a_pinned_package_is_not_installed(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamp_env(home, "cuda-linux")
    pins = recipe_pins(recipe_for(llm_env("cuda-linux")))
    short = {name: version for name, version in pins.items() if name != "torch"}
    monkeypatch.setattr(jobenv, "installed_packages", lambda _home, _spec: short)
    status = env_status(home, llm_env("cuda-linux"), "cuda-linux")
    assert status.installed is False
    assert "torch is absent" in status.detail


def test_an_env_at_the_wrong_version_is_not_installed(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drift into a different torch is a refusal, not a shrug."""
    stamp_env(home, "cuda-linux")
    pins = dict(recipe_pins(recipe_for(llm_env("cuda-linux"))))
    pins["torch"] = "2.5.1"
    monkeypatch.setattr(jobenv, "installed_packages", lambda _home, _spec: pins)
    status = env_status(home, llm_env("cuda-linux"), "cuda-linux")
    assert status.installed is False
    assert "torch is 2.5.1, recipe pins" in status.detail


def test_a_complete_env_is_installed(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamp_env(home, "cuda-linux")
    pins = recipe_pins(recipe_for(llm_env("cuda-linux")))
    monkeypatch.setattr(jobenv, "installed_packages", lambda _home, _spec: dict(pins))
    status = env_status(home, llm_env("cuda-linux"), "cuda-linux")
    assert status.installed is True
    assert "vllm 0.29.0" in status.detail
    assert "python 3.11.16" in status.detail
    assert (
        require_env(home, llm_env("cuda-linux"), "cuda-linux")
        == env_dir(home, llm_env("cuda-linux")) / "bin" / "python"
    )


def test_require_env_refuses_rather_than_guessing_an_interpreter(home: Path) -> None:
    with pytest.raises(EnvError) as caught:
        require_env(home, llm_env("cuda-linux"), "cuda-linux")
    assert "crucible install llm" in str(caught.value)


# ------------------------------------------------- pinning what is not on PyPI


def _recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> Path:
    root = tmp_path / "recipes"
    (root / "tts").mkdir(parents=True)
    path = root / "tts" / "higgs-v3-cuda-linux.txt"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("CRUCIBLE_RECIPES_DIR", str(root))
    return path


SHA = "4ebc529f30cfa205b820cf494e48fcb76ac12977"
REFERENCE = f"narrator[higgs-v3-server] @ git+https://example.invalid/x@{SHA}#subdirectory=python"


def test_a_direct_reference_is_a_pin_and_is_not_read_as_a_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """narrator is not on PyPI, so it cannot be `name==version`.

    `pip list` reports its declared version, which does not move when the commit
    does — so a version check here would call an env built from last month's
    commit a match. The two shapes are therefore read by two functions, and
    `recipe_pins` must not report the reference as a version pin.
    """
    path = _recipe(tmp_path, monkeypatch, f"{REFERENCE}\ntorch==2.13.0\n")
    assert recipe_pins(path) == {"torch": "2.13.0"}
    assert jobenv.recipe_direct_references(path) == {"narrator": SHA}


def test_a_direct_reference_without_a_commit_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A branch is a moving target; an env built from one cannot be said to
    match the recipe that built it."""
    path = _recipe(
        tmp_path, monkeypatch, "narrator @ git+https://example.invalid/x@main\n"
    )
    with pytest.raises(EnvError) as caught:
        jobenv.recipe_direct_references(path)
    assert "names no commit" in str(caught.value)
    assert "a branch name is not a pin" in str(caught.value)


def test_a_line_that_is_neither_shape_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _recipe(tmp_path, monkeypatch, "torch>=2.13\n")
    with pytest.raises(EnvError) as caught:
        recipe_pins(path)
    assert "every requirement in a recipe is pinned exactly" in str(caught.value)


def test_the_commit_an_env_was_built_from_is_read_off_pips_own_record(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PEP 610: pip writes `direct_url.json` beside a distribution installed from
    a URL, and for a VCS install it carries the commit pip actually resolved."""
    spec = tts_env("higgs-v3", "cuda-linux")
    packages = env_dir(home, spec) / "lib" / "python3.11" / "site-packages"
    dist = packages / "narrator-0.1.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "direct_url.json").write_text(
        json.dumps(
            {
                "url": "https://example.invalid/x",
                "vcs_info": {"vcs": "git", "commit_id": SHA},
            }
        ),
        encoding="utf-8",
    )
    # A package installed from an index has no such file and is not reported.
    (packages / "torch-2.13.0.dist-info").mkdir()
    assert jobenv.installed_direct_references(home, spec) == {"narrator": SHA}


def test_an_env_built_from_another_commit_is_not_ready(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = tts_env("higgs-v3", "cuda-linux")
    directory = env_dir(home, spec)
    (directory / "bin").mkdir(parents=True)
    (directory / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (directory / "crucible-env.json").write_text(
        json.dumps(
            {
                "backend": "cuda-linux",
                "recipe": "higgs-v3-cuda-linux.txt",
                "python_version": "3.11.16",
                "seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    _recipe(tmp_path, monkeypatch, f"{REFERENCE}\ntorch==2.13.0\n")
    monkeypatch.setattr(
        jobenv, "installed_packages", lambda _home, _spec: {"torch": "2.13.0"}
    )
    packages = directory / "lib" / "python3.11" / "site-packages"
    dist = packages / "narrator-0.1.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "direct_url.json").write_text(
        json.dumps({"vcs_info": {"vcs": "git", "commit_id": "0" * 40}}),
        encoding="utf-8",
    )
    status = env_status(home, spec, "cuda-linux")
    assert status.installed is False
    assert "narrator was installed from 0000000" in status.detail
    assert SHA in status.detail
