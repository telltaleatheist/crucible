"""What the command line gained with `align` and `rvc`.

Two flags, two more worker envs, the aligner joining the one namespace of model
ids, and `crucible rvc` — a command of its own, because an RVC model's weights
are one archive fetched by name rather than a repo snapshot, so `crucible models
pull` could not serve one.

`crucible init` refuses on win32 by design, so like `tests/test_cli.py` these run
on Linux (in WSL on Owen's PC) and not on Windows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible import cli, workerenv
from crucible.config import load_config

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND

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
    monkeypatch.setattr(cli, "detect_backend", lambda: FAKE_BACKEND)


@pytest.fixture
def mac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "detect_backend", lambda: FAKE_MAC_BACKEND)


# --------------------------------------------------------------------- init


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


# ------------------------------------------------------------------- doctor


def test_doctor_reports_each_missing_worker_env_once(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """Three worker envs now, and each one gets exactly one row and one problem.

    The "once" is the point, and it is not hypothetical: `_doctor_report` had the
    llm env's report nested INSIDE this loop, where with one worker type it never
    ran at all and with three it would have written the same row three times and
    appended the same problem three times. `fix(doctor)` took it out for its own
    reason — it was also a TypeError — and this is the test that keeps it out as
    the number of worker types grows.
    """
    assert cli.main(["init", "--enable-align", "--enable-rvc", "--enable-asr"]) == 0
    capsys.readouterr()
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    envs = {entry["job_type"]: entry for entry in report["worker_envs"]}
    assert sorted(envs) == ["align", "asr", "rvc"]
    assert "crucible install align" in envs["align"]["detail"]
    assert "crucible install rvc" in envs["rvc"]["detail"]
    # The llm env is off, so it is neither reported nor complained about.
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


# ------------------------------------------------------------------ install


def test_align_is_installable_and_rvc_is_installable(viable: None) -> None:
    """Both are worker envs, which is a fact about the SHAPE of their work."""
    assert "align" in cli.INSTALLABLE_JOB_TYPES
    assert "rvc" in cli.INSTALLABLE_JOB_TYPES
    assert set(workerenv.WORKER_JOB_TYPES) == {"align", "asr", "rvc"}


def test_installing_align_on_the_mac_refuses_and_names_what_ships(
    home: Path, mac: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["install", "align"]) == 1
    error = capsys.readouterr().err
    assert "no align env recipe for backend 'mlx-darwin'" in error
    assert "['cuda-linux']" in error


def test_rvc_has_a_mac_recipe_unlike_align(mac: None) -> None:
    """The Mac is a real rvc backend; it is not a real align backend YET."""
    assert workerenv.recipe_for("rvc", FAKE_MAC_BACKEND.kind).is_file()
    with pytest.raises(workerenv.WorkerEnvError):
        workerenv.recipe_for("align", FAKE_MAC_BACKEND.kind)


# ------------------------------------------------------------------- models


def test_the_aligner_joins_the_one_namespace_of_model_ids(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["models", "list", "--json"]) == 0
    rows = {row["id"]: row for row in json.loads(capsys.readouterr().out)}
    assert "qwen3-aligner" in rows
    assert rows["qwen3-aligner"]["hf_repo"] == "Qwen/Qwen3-ForcedAligner-0.6B"
    # An aligner has no context. Null means "this model has no such knob".
    assert rows["qwen3-aligner"]["context_default"] is None
    # And an RVC model is NOT in that namespace: its weights are an archive.
    assert not any(row.startswith("deathstalker") for row in rows)


# ---------------------------------------------------------------------- rvc


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
    """`sigma` is both, and the two must not be able to collide on disk."""
    from crucible.rvcmodels import load_rvc_manifest
    from crucible.voices import load_voice

    assert load_rvc_manifest("sigma").weights_family == "rvc"
    assert load_voice("sigma").weights_family == "voices"
