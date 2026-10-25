"""`crucible env sync`, the installer's `env-sync` step: an update brings every env installed
here that no longer matches its recipe to it.

2026-10-25: narrator's pin moved to dc05e7e7 (2.0.24, the PC hiss fix). The deploy carried
the recipe to the Mac but rebuilt no env, so its tts env stayed on 94c1f49d and every Higgs
voice read `loadable: false` until `crucible install tts` was run by hand."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible import cli, jobenv
from crucible.cli import install as install_cli

from .conftest import FAKE_BACKEND


@pytest.fixture
def box(home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)
    assert cli.main(["init", "--enable-llm"]) == 0
    capsys.readouterr()
    return home


def _envs(monkeypatch: pytest.MonkeyPatch, home: Path, states: dict[str, str | None]) -> list[str]:
    """Each env key -> None (not installed here), "ok" or a drift sentence; returns the
    keys `_build_env` was asked to rebuild."""
    specs = [SimpleNamespace(key=key, job_type=key) for key in states]
    monkeypatch.setattr(jobenv, "every_env", lambda backend_kind: specs)

    def python(home_dir: Path, spec) -> Path:
        path = home / "fake-envs" / spec.key / "python"
        if states[spec.key] is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
        return path

    monkeypatch.setattr(jobenv, "env_python", python)
    monkeypatch.setattr(
        jobenv, "env_status",
        lambda home_dir, spec, kind: SimpleNamespace(
            installed=states[spec.key] == "ok", detail=states[spec.key]),
    )
    rebuilt: list[str] = []

    def build(config, backend, spec, args) -> None:
        assert args.force is False, "a sync installs only what moved; it never wipes an env"
        rebuilt.append(spec.key)
        return None

    monkeypatch.setattr(install_cli, "_build_env", build)
    return rebuilt


def test_an_env_whose_pin_moved_is_brought_to_its_recipe(box, monkeypatch, capsys) -> None:
    drift = ("/home/x/.crucible/envs/tts does not match mlx-darwin.txt: narrator was installed "
             "from 94c1f49d, recipe pins dc05e7e7")
    rebuilt = _envs(monkeypatch, box, {"tts-higgs-v3": drift, "llm": "ok"})
    assert cli.main(["env", "sync"]) == 0
    assert rebuilt == ["tts-higgs-v3"]
    assert "dc05e7e7" in capsys.readouterr().out


def test_an_env_not_installed_here_is_left_alone(box, monkeypatch, capsys) -> None:
    rebuilt = _envs(monkeypatch, box, {"video-ltx": None, "llm": "ok"})
    assert cli.main(["env", "sync"]) == 0
    assert rebuilt == []
    assert "matches its recipe" in capsys.readouterr().out


def test_an_env_that_cannot_be_brought_to_its_recipe_fails_the_step_by_name(
    box, monkeypatch, capsys
) -> None:
    _envs(monkeypatch, box, {"tts-higgs-v3": "drifted"})
    monkeypatch.setattr(install_cli, "_build_env", lambda *a: "pip could not reach the index")
    assert cli.main(["env", "sync"]) != 0
    assert "env_sync_failed" in capsys.readouterr().err
