from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible import cli
from crucible.config import load_config

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND

ASR_MODELS = ["whisper-large-v3-turbo", "whisper-tiny"]


@pytest.fixture
def viable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)


@pytest.fixture
def mac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_MAC_BACKEND)


def test_init_records_the_asr_flag(home: Path, viable: None) -> None:
    assert cli.main(["init", "--enable-asr"]) == 0
    assert load_config(home).enable_asr is True


def test_asr_is_off_unless_asked_for(home: Path, viable: None) -> None:
    assert cli.main(["init"]) == 0
    assert load_config(home).enable_asr is False


def test_doctor_reports_the_missing_asr_env(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-asr"]) == 0
    capsys.readouterr()
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["config"]["enable_asr"] is True
    envs = {entry["job_type"]: entry for entry in report["worker_envs"]}
    assert envs["asr"]["installed"] is False
    assert "crucible install asr" in envs["asr"]["detail"]
    assert any(problem.startswith("asr_env:") for problem in report["problems"])


def test_doctor_says_nothing_about_asr_when_it_is_off(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-echo"]) == 0
    capsys.readouterr()
    assert cli.main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["worker_envs"] == []


def test_the_mac_asr_recipe_installs_mlx_whisper_and_not_faster_whisper() -> None:
    from crucible import jobenv

    assert jobenv.worker_env("asr", "mlx-darwin").headline == "mlx-whisper"
    recipe = jobenv.recipe_for(jobenv.worker_env("asr", "mlx-darwin"))
    pins = jobenv.recipe_pins(recipe)
    assert pins["mlx-whisper"] == "0.4.3"
    assert "faster-whisper" not in pins
    assert "ctranslate2" not in pins


def test_models_list_covers_both_manifest_directories(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["models", "list", "--json"]) == 0
    rows = {row["id"]: row for row in json.loads(capsys.readouterr().out)}
    assert "qwen3.5-9b" in rows
    for model_id in ASR_MODELS:
        assert rows[model_id]["backend_supported"] is True
        assert rows[model_id]["installed"] is False
        assert "crucible models pull" in rows[model_id]["detail"]
        assert rows[model_id]["context_default"] is None
    assert rows["qwen3.5-9b"]["context_default"] == 16384


def test_pulling_an_unknown_model_names_every_manifest(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["models", "pull", "whisper-enormous"]) == 1
    error = capsys.readouterr().err
    assert "whisper-tiny" in error
    assert "qwen3.5-9b" in error


def test_one_id_declared_twice_is_refused_rather_than_resolved(
    home: Path,
    viable: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clashing = tmp_path / "models-fixture"
    clashing.mkdir()
    (clashing / "whisper-tiny.toml").write_text(
        """
[model]
id = "whisper-tiny"
family = "qwen3.5"
params_b = 9
context_default = 4096
trained_context = 262144
modalities = ["text"]

[backends.cuda-linux]
engine = "vllm"
hf_repo = "Qwen/Qwen3.5-9B"
revision = "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
memory_bytes_estimate = 1755830268
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CRUCIBLE_MODELS_DIR", str(clashing))
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["models", "pull", "whisper-tiny"]) == 1
    assert "a model id names one model" in capsys.readouterr().err
