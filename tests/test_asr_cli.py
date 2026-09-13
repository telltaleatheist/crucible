"""What the command line gained with `asr`: a flag, an installer, and one
namespace of model ids across two manifest directories.

`crucible init` refuses on win32 by design, so like `tests/test_cli.py` these run
on Linux (in WSL on Owen's PC) and not on Windows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible import cli
from crucible.config import load_config

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND

ASR_MODELS = [
    "faster-whisper-base",
    "faster-whisper-distil-large-v3",
    "faster-whisper-large-v3",
    "faster-whisper-medium",
    "faster-whisper-small",
    "faster-whisper-tiny",
]


@pytest.fixture
def viable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "detect_backend", lambda: FAKE_BACKEND)


@pytest.fixture
def mac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "detect_backend", lambda: FAKE_MAC_BACKEND)


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


def test_installing_asr_on_the_mac_refuses_and_names_what_ships(
    home: Path, mac: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """There is no mlx-darwin recipe, and the reason is CTranslate2's."""
    assert cli.main(["init", "--enable-asr"]) == 0
    capsys.readouterr()
    assert cli.main(["install", "asr"]) == 1
    error = capsys.readouterr().err
    assert "no asr env recipe for backend 'mlx-darwin'" in error
    assert "['cuda-linux']" in error


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
        # Whisper's window is 30 seconds of audio and is not a knob, so an ASR
        # row has no context at all rather than a number nobody set.
        assert rows[model_id]["context_default"] is None
    assert rows["qwen3.5-9b"]["context_default"] == 12288


def test_pulling_an_unknown_model_names_every_manifest(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["models", "pull", "whisper-enormous"]) == 1
    error = capsys.readouterr().err
    assert "faster-whisper-tiny" in error
    assert "qwen3.5-9b" in error


def test_one_id_declared_twice_is_refused_rather_than_resolved(
    home: Path,
    viable: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`crucible models pull <id>` is one question; two answers is not an answer."""
    clashing = tmp_path / "asr-fixture"
    clashing.mkdir()
    (clashing / "qwen3.5-9b.toml").write_text(
        """
[model]
id = "qwen3.5-9b"
family = "faster-whisper"
parameters_m = 74

[backends.cuda-linux]
engine = "faster-whisper"
hf_repo = "Systran/faster-whisper-base"
revision = "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
memory_bytes_estimate = 1755830268
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CRUCIBLE_ASR_DIR", str(clashing))
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["models", "pull", "qwen3.5-9b"]) == 1
    assert "a model id names one model" in capsys.readouterr().err
