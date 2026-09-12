"""The command line: init, doctor, token — with the host probe monkeypatched."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible import cli
from crucible.config import config_path, load_config
from crucible.errors import NoViableBackend

from .conftest import FAKE_BACKEND


@pytest.fixture
def viable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "detect_backend", lambda: FAKE_BACKEND)


def test_init_writes_a_0600_config_with_a_token(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-echo"]) == 0
    path = config_path(home)
    assert path.exists()
    assert oct(path.stat().st_mode & 0o777) == "0o600"

    config = load_config(home)
    assert config.backend_kind == "cuda-linux"
    assert config.enable_echo is True
    assert len(config.token) >= 40
    assert config.token not in capsys.readouterr().out


def test_init_refuses_to_clobber_an_existing_config(home: Path, viable: None) -> None:
    assert cli.main(["init"]) == 0
    first = load_config(home).token
    assert cli.main(["init"]) == 1
    assert load_config(home).token == first


def test_init_force_mints_a_new_token(home: Path, viable: None) -> None:
    assert cli.main(["init"]) == 0
    first = load_config(home).token
    assert cli.main(["init", "--force"]) == 0
    assert load_config(home).token != first


def test_init_refuses_without_a_backend(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse() -> None:
        raise NoViableBackend("no nvidia-smi on this Linux host")

    monkeypatch.setattr(cli, "detect_backend", refuse)
    assert cli.main(["init"]) == 1
    assert not config_path(home).exists()
    assert "no nvidia-smi" in capsys.readouterr().err


def test_doctor_json_is_healthy_after_init(
    home: Path, viable: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("crucible.cli.detect_backend", lambda: FAKE_BACKEND)
    assert cli.main(["init", "--enable-echo"]) == 0
    capsys.readouterr()

    assert cli.main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["healthy"] is True
    assert report["problems"] == []
    assert report["backend"]["kind"] == "cuda-linux"
    assert report["config"]["mode"] == "0o600"
    echo = [entry for entry in report["job_types"] if entry["name"] == "echo"][0]
    assert echo == {
        "name": "echo",
        "enabled": True,
        "ready": True,
        "detail": "enabled; copies inputs to artifacts, uses no accelerator",
        "models": [],
    }


def test_doctor_is_unhealthy_without_a_config(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["healthy"] is False
    assert any("config" in problem for problem in report["problems"])


def test_doctor_is_unhealthy_when_the_backend_changed(
    home: Path, viable: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["init"]) == 0
    capsys.readouterr()

    from crucible.backend import Backend, Gpu

    elsewhere = Backend(
        kind="mlx-darwin",
        platform="darwin",
        arch="arm64",
        gpu=Gpu(vendor="apple", name="Apple M2 Ultra", vram_bytes=1),
        detail="test double",
    )
    monkeypatch.setattr(cli, "detect_backend", lambda: elsewhere)
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert any("backend_changed" in problem for problem in report["problems"])


def test_token_needs_show(home: Path, viable: None) -> None:
    assert cli.main(["init"]) == 0
    assert cli.main(["token"]) == 1


def test_token_show_prints_the_token(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["token", "--show"]) == 0
    assert capsys.readouterr().out.strip() == load_config(home).token


def test_token_without_a_config_is_refused(home: Path) -> None:
    assert cli.main(["token", "--show"]) == 1
