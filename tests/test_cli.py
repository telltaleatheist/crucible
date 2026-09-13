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


# ------------------------------------------------------------------- the tts
# job type: `crucible voices`, the doctor's env rows, and `crucible install tts`.


def test_doctor_reports_one_tts_env_per_narrator_engine(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """On cuda-linux the two engines cannot share a venv, so there are two rows —
    and neither is installed on a fresh host, which is a problem and says so."""
    assert cli.main(["init", "--enable-tts"]) == 0
    capsys.readouterr()

    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["config"]["enable_tts"] is True
    assert sorted(report["tts_envs"]) == ["higgs-v3", "orpheus"]
    for engine, entry in report["tts_envs"].items():
        assert entry["installed"] is False
        assert f"envs/tts-{engine}" in entry["detail"]
        assert "crucible install tts" in entry["detail"]
    assert any("tts_env[higgs-v3]" in problem for problem in report["problems"])


def test_doctor_says_nothing_about_tts_when_it_is_off(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-echo"]) == 0
    capsys.readouterr()
    assert cli.main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["tts_envs"] == {}


def test_the_two_tts_engines_share_one_env_on_the_mac(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """On mlx-darwin they genuinely do, so the two rows name the same directory."""
    from .conftest import FAKE_MAC_BACKEND

    monkeypatch.setattr(cli, "detect_backend", lambda: FAKE_MAC_BACKEND)
    assert cli.main(["init", "--enable-tts"]) == 0
    capsys.readouterr()
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    paths = {entry["path"] for entry in report["tts_envs"].values()}
    assert paths == {str(home / "envs" / "tts")}


def test_voices_list_names_the_pull_command(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-tts"]) == 0
    capsys.readouterr()
    assert cli.main(["voices", "list", "--json"]) == 0
    listed = {row["id"]: row for row in json.loads(capsys.readouterr().out)}
    assert listed["deathstalker"]["installed"] is False
    assert listed["deathstalker"]["max_chars"] == 800
    assert listed["deathstalker"]["estimate_basis"] == "declared"
    assert "crucible voices pull deathstalker" in listed["deathstalker"]["detail"]


def test_voices_pull_refuses_an_unknown_voice(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-tts"]) == 0
    capsys.readouterr()
    assert cli.main(["voices", "pull", "gandalf"]) == 1
    assert "no manifest for voice 'gandalf'" in capsys.readouterr().err


def test_installing_tts_needs_a_narrator_engine(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two envs on this backend, so the command may not pick one for you."""
    assert cli.main(["init", "--enable-tts"]) == 0
    capsys.readouterr()
    assert cli.main(["install", "tts"]) == 1
    assert "needs --narrator-engine" in capsys.readouterr().err


def test_installing_llm_refuses_a_narrator_engine(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-llm"]) == 0
    capsys.readouterr()
    assert cli.main(["install", "llm", "--narrator-engine", "higgs-v3"]) == 1
    assert "means nothing for 'llm'" in capsys.readouterr().err


def test_installing_tts_says_which_recipe_is_missing(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """This build ships no envs/tts recipes — narrator is pinned by git sha and
    that is the next builder's file. The refusal names the directory, which is
    better than pretending the job type has no installer."""
    assert cli.main(["init", "--enable-tts"]) == 0
    capsys.readouterr()
    assert cli.main(["install", "tts", "--narrator-engine", "higgs-v3"]) == 1
    assert "tts env recipes at" in capsys.readouterr().err
