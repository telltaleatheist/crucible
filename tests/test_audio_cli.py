from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible import cli, jobenv
from crucible.config import load_config

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND


@pytest.fixture
def viable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)


def test_init_records_the_audio_flag(home: Path, viable: None, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["init", "--enable-audio"]) == 0
    assert "audio:    enabled" in capsys.readouterr().out
    assert load_config(home).enable_audio is True


def test_install_audio_builds_every_engine_env_of_this_backend(
    home: Path, viable: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-audio"]) == 0
    capsys.readouterr()
    built: list[str] = []

    def install_env(home_dir: Path, spec: jobenv.EnvSpec, backend_kind: str, **_: object):
        built.append(spec.key)
        return jobenv.EnvStatus(True, home_dir, f"{spec.key} ok", "3.11.16", {spec.headline: "0.1", "torch": "2"})

    stepped: list[tuple[str, ...]] = []
    monkeypatch.setattr(jobenv, "install_env", install_env)
    monkeypatch.setattr(cli.install, "_smoke_import", lambda *_: None)
    monkeypatch.setattr(cli.install, "_measure_step", lambda *_a, **_k: None)
    monkeypatch.setattr(cli.install, "_capability_step", lambda _c, _b, *types: stepped.append(types) or 0)
    assert cli.main(["install", "audio"]) == 0
    assert built == ["audio-stable-audio-3", "audio-yue2"]
    assert stepped == [("audio",)]
    out = capsys.readouterr().out
    assert "audio engine: stable-audio-3" in out and "audio engine: yue2" in out


def test_a_failed_audio_env_says_the_rerun_builds_only_what_is_missing(
    home: Path, viable: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-audio"]) == 0
    capsys.readouterr()

    def install_env(home_dir: Path, spec: jobenv.EnvSpec, backend_kind: str, **_: object):
        raise jobenv.EnvError("pip could not reach the index")

    monkeypatch.setattr(jobenv, "install_env", install_env)
    assert cli.main(["install", "audio"]) == 1
    said = capsys.readouterr().err
    assert "pip could not reach the index" in said
    assert "`crucible install audio` again builds only what is missing" in said


def test_install_audio_refuses_a_narrator_engine(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-audio"]) == 0
    capsys.readouterr()
    assert cli.main(["install", "audio", "--narrator-engine", "higgs-v3"]) == 1
    assert "means nothing for 'audio'" in capsys.readouterr().err


def test_doctor_reports_one_audio_env_per_engine(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_MAC_BACKEND)
    assert cli.main(["init", "--enable-audio"]) == 0
    capsys.readouterr()
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert sorted(report["audio_envs"]) == ["audio-stable-audio-3"]
    assert report["audio_envs"]["audio-stable-audio-3"]["installed"] is False
    assert any("crucible install audio" in problem for problem in report["problems"])


def test_models_list_shows_the_audio_models(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-audio"]) == 0
    capsys.readouterr()
    assert cli.main(["models", "list", "--json"]) == 0
    rows = {row["id"]: row for row in json.loads(capsys.readouterr().out)}
    assert rows["yue2-3b"]["hf_repo"] == "m-a-p/YuE2-3B"
    assert "crucible models pull stable-audio-3-medium" in rows["stable-audio-3-medium"]["detail"]
