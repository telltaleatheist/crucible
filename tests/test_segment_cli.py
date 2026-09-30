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


def test_init_records_the_segment_flag(home: Path, viable: None, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["init", "--enable-segment"]) == 0
    assert "segment:  enabled" in capsys.readouterr().out
    assert load_config(home).enable_segment is True


def test_a_config_without_the_flag_reads_it_as_off(home: Path, viable: None) -> None:
    assert cli.main(["init"]) == 0
    assert load_config(home).enable_segment is False


def test_install_segment_builds_its_one_env_and_smoke_imports_the_model_code(
    home: Path, viable: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-segment"]) == 0
    capsys.readouterr()
    built: list[str] = []
    smoked: list[tuple[str, str]] = []

    def install_env(home_dir: Path, spec: jobenv.EnvSpec, backend_kind: str, **_: object):
        built.append(spec.key)
        return jobenv.EnvStatus(True, home_dir, f"{spec.key} ok", "3.11.16", {spec.headline: "5.17.0", "torch": "2.14.0"})

    stepped: list[tuple[str, ...]] = []
    monkeypatch.setattr(jobenv, "install_env", install_env)
    monkeypatch.setattr(cli.install, "_smoke_import", lambda _python, key, kind: smoked.append((key, kind)))
    monkeypatch.setattr(cli.install, "_measure_step", lambda *_a, **_k: None)
    monkeypatch.setattr(cli.install, "_capability_step", lambda _c, _b, *types: stepped.append(types) or 0)
    assert cli.main(["install", "segment"]) == 0
    assert built == ["segment"]
    assert smoked == [("segment", "cuda-linux")]
    assert stepped == [("segment",)]
    assert "segment/cuda-linux.txt" in capsys.readouterr().out.replace("\\", "/")


def test_doctor_reports_the_segment_env(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_MAC_BACKEND)
    assert cli.main(["init", "--enable-segment"]) == 0
    capsys.readouterr()
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    envs = {entry["job_type"]: entry for entry in report["worker_envs"]}
    assert envs["segment"]["installed"] is False
    assert report["config"]["enable_segment"] is True
    assert any("crucible install segment" in problem for problem in report["problems"])


def test_models_list_shows_the_segment_models(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-segment"]) == 0
    capsys.readouterr()
    assert cli.main(["models", "list", "--json"]) == 0
    rows = {row["id"]: row for row in json.loads(capsys.readouterr().out)}
    assert rows["birefnet"]["hf_repo"] == "ZhengPeng7/BiRefNet"
    assert rows["sam2.1-hiera-large"]["hf_repo"] == "facebook/sam2.1-hiera-large"
    assert "crucible models pull birefnet" in rows["birefnet"]["detail"]
