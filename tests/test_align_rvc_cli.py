from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible import cli, jobenv
from crucible.config import load_config

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND, configure_box

RVC_MODELS = [
    "deathstalker-rvc-v1",
    "deathstalker-rvc-v3",
    "girlfriend",
    "mistborn-rvc-v1",
    "owen-morgan",
    "sigma",
    "us-female-1",
]


@pytest.fixture
def viable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)


@pytest.fixture
def mac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_MAC_BACKEND)


def test_init_records_both_flags(home: Path, viable: None) -> None:
    assert cli.main(["init", "--enable-align", "--enable-rvc"]) == 0
    config = load_config(home)
    assert config.enable_align is True
    assert config.enable_rvc is True


def test_both_are_off_unless_asked_for(home: Path, viable: None) -> None:
    assert cli.main(["init"]) == 0
    config = load_config(home)
    assert config.enable_align is False
    assert config.enable_rvc is False


def test_doctor_reports_each_missing_worker_env_once(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-align", "--enable-rvc", "--enable-asr"]) == 0
    capsys.readouterr()
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    envs = {entry["job_type"]: entry for entry in report["worker_envs"]}
    assert sorted(envs) == ["align", "asr", "rvc"]
    assert "crucible install align" in envs["align"]["detail"]
    assert "crucible install rvc" in envs["rvc"]["detail"]
    assert report["llm_env"] is None
    assert not [p for p in report["problems"] if p.startswith("llm_env")]


def test_doctor_says_nothing_about_them_when_they_are_off(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    cli.main(["doctor", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert [entry["job_type"] for entry in report["worker_envs"]] == []
    types = {entry["name"]: entry for entry in report["job_types"]}
    assert types["align"]["enabled"] is False
    assert types["rvc"]["enabled"] is False


def test_align_is_installable_and_rvc_is_installable(viable: None) -> None:
    assert "align" in cli.INSTALLABLE_JOB_TYPES
    assert "rvc" in cli.INSTALLABLE_JOB_TYPES
    assert set(jobenv.WORKER_JOB_TYPES) == {"align", "asr", "rvc"}


def test_every_worker_type_has_a_mac_recipe_now(mac: None) -> None:
    for job_type in jobenv.WORKER_JOB_TYPES:
        recipe = jobenv.recipe_for(jobenv.worker_env(job_type, FAKE_MAC_BACKEND.kind))
        assert recipe.is_file(), job_type
        assert jobenv.recipe_pins(recipe), job_type
    with pytest.raises(jobenv.EnvError) as caught:
        jobenv.recipe_for(jobenv.worker_env("align", "llama-windows"))
    assert "no 'align' env on 'llama-windows'" in str(caught.value)


def test_the_aligner_joins_the_one_namespace_of_model_ids(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["models", "list", "--json"]) == 0
    rows = {row["id"]: row for row in json.loads(capsys.readouterr().out)}
    assert "qwen3-aligner" in rows
    assert rows["qwen3-aligner"]["hf_repo"] == "Qwen/Qwen3-ForcedAligner-0.6B"
    assert rows["qwen3-aligner"]["context_default"] is None
    assert not any(row.startswith("deathstalker") for row in rows)


def test_rvc_list_shows_every_manifest_and_where_it_stands(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["rvc", "list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["id"] for row in rows] == RVC_MODELS
    row = next(r for r in rows if r["id"] == "sigma")
    assert row["installed"] is False
    assert row["model_name"] == "Sigma Male Narrator"
    assert row["archive"] == "rvc/sigma.tar.gz"
    assert "crucible rvc pull sigma" in row["detail"]


def test_pulling_an_unknown_rvc_model_names_what_ships(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["rvc", "pull", "deathstalker-rvc-v2"]) == 1
    error = capsys.readouterr().err
    assert "no RVC manifest for 'deathstalker-rvc-v2'" in error
    assert "deathstalker-rvc-v1" in error


def test_an_rvc_id_and_a_voice_id_may_be_the_same_word(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    from crucible.rvcmodels import load_rvc_manifest
    from crucible.voicecatalog import load_voice

    configure_box(home)
    assert load_rvc_manifest("sigma").weights_family == "rvc"
    assert load_voice("sigma").weights_family == "voices"
