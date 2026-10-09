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



def test_the_env_rung_finds_the_audio_envs_an_install_just_built(tmp_path: Path) -> None:
    for spec in jobenv.audio_envs(CUDA_LINUX):
        _fake_env(tmp_path, spec)

    found = ladder.installed_torch_env_pythons(tmp_path, CUDA_LINUX)
    assert sorted(found) == ["audio-stable-audio-3", "audio-yue2"]


def test_every_env_names_each_venv_once_with_its_own_key() -> None:
    keys = [spec.key for spec in jobenv.every_env(CUDA_LINUX)]
    assert len(keys) == len(set(keys))
    assert {"llm", "asr", "tts-higgs-v3", "audio-yue2", "video-ltx"} <= set(keys)
    for spec in jobenv.every_env(CUDA_LINUX):
        assert jobenv.recipe_for(spec).is_file()


def test_the_llm_rungs_still_find_the_llm_env_by_its_key(tmp_path: Path) -> None:
    _fake_env(tmp_path, jobenv.llm_env(CUDA_LINUX))
    assert "llm" in ladder.installed_env_pythons(tmp_path, CUDA_LINUX)
