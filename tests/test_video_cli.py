from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible import cli, hosttools, jobenv
from crucible.config import load_config

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND


@pytest.fixture
def viable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)


def test_init_records_the_video_flag(home: Path, viable: None, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["init", "--enable-video"]) == 0
    assert "video:    enabled" in capsys.readouterr().out
    assert load_config(home).enable_video is True


def test_install_video_builds_the_env_and_places_ffmpeg(
    home: Path, viable: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-video"]) == 0
    capsys.readouterr()
    built: list[str] = []

    def install_env(home_dir: Path, spec: jobenv.EnvSpec, backend_kind: str, **_: object):
        built.append(spec.key)
        return jobenv.EnvStatus(True, home_dir, f"{spec.key} ok", "3.11.16", {spec.headline: "0.1", "torch": "2"})

    placed: list[Path] = []
    stepped: list[tuple[str, ...]] = []
    monkeypatch.setattr(jobenv, "install_env", install_env)
    monkeypatch.setattr(cli.install, "_smoke_import", lambda *_: None)
    monkeypatch.setattr(hosttools, "ensure_ffmpeg", lambda home_dir, **_: placed.append(home_dir) or "ffmpeg: ok")
    monkeypatch.setattr(hosttools, "ensure_silero_vad", lambda home_dir: "speech detector: ok")
    monkeypatch.setattr(cli.install, "_measure_step", lambda *_a, **_k: None)
    monkeypatch.setattr(cli.install, "_capability_step", lambda _c, _b, *types: stepped.append(types) or 0)
    assert cli.main(["install", "video"]) == 0
    assert built == ["video-ltx"]
    assert placed == [home]
    assert stepped == [("video",)]
    assert "video engine: ltx" in capsys.readouterr().out


def test_install_video_on_the_mac_says_it_is_cuda_only(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_MAC_BACKEND)
    assert cli.main(["init", "--enable-video"]) == 0
    capsys.readouterr()
    assert cli.main(["install", "video"]) == 1
    said = capsys.readouterr().err
    assert "video has no engine on 'mlx-darwin'" in said and "cuda-linux" in said


def test_install_video_refuses_a_narrator_engine(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-video"]) == 0
    capsys.readouterr()
    assert cli.main(["install", "video", "--narrator-engine", "higgs-v3"]) == 1
    assert "means nothing for 'video'" in capsys.readouterr().err


def test_doctor_reports_the_video_env(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-video"]) == 0
    capsys.readouterr()
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert sorted(report["video_envs"]) == ["video-ltx"]
    assert report["video_envs"]["video-ltx"]["installed"] is False
    assert any("crucible install video" in problem for problem in report["problems"])


def test_models_list_shows_the_video_model_and_its_pull(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-video"]) == 0
    capsys.readouterr()
    assert cli.main(["models", "list", "--json"]) == 0
    rows = {row["id"]: row for row in json.loads(capsys.readouterr().out)}
    assert rows["ltx-2.5-distilled"]["hf_repo"] == "Lightricks/LTX-2.5-Diffusers"
    assert "crucible models pull ltx-2.5-distilled" in rows["ltx-2.5-distilled"]["detail"]


def test_models_pull_refuses_the_gated_repo_without_a_token(
    home: Path, viable: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert cli.main(["init", "--enable-video"]) == 0
    capsys.readouterr()
    assert cli.main(["models", "pull", "ltx-2.5-distilled"]) == 1
    said = capsys.readouterr()
    assert "Lightricks/LTX-2.5-Diffusers is gated" in said.out + said.err
