from __future__ import annotations

from pathlib import Path

from crucible import jobenv, ladder
from crucible.backend import CUDA_LINUX


def _fake_env(home: Path, spec: jobenv.EnvSpec) -> None:
    python = jobenv.env_python(home, spec)
    python.parent.mkdir(parents=True)
    python.write_text("")


def test_the_env_rung_smokes_only_the_envs_whose_recipe_ships_torch(tmp_path: Path) -> None:
    _fake_env(tmp_path, jobenv.worker_env("asr", CUDA_LINUX))
    _fake_env(tmp_path, jobenv.worker_env("align", CUDA_LINUX))
    _fake_env(tmp_path, jobenv.llm_env(CUDA_LINUX))

    assert sorted(ladder.installed_env_pythons(tmp_path, CUDA_LINUX)) == ["align", "asr", "llm"]
    assert sorted(ladder.installed_torch_env_pythons(tmp_path, CUDA_LINUX)) == ["align", "llm"]


def test_the_faster_whisper_asr_recipe_has_no_torch_to_smoke() -> None:
    pins = jobenv.recipe_pins(jobenv.recipe_for(jobenv.worker_env("asr", CUDA_LINUX)))
    assert "faster-whisper" in pins
    assert "torch" not in pins
