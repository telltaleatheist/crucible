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
