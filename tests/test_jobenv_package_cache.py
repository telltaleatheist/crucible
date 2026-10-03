from __future__ import annotations

from pathlib import Path

import pytest

from crucible import jobenv


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "envs" / "tts"
    site = directory / "lib" / "python3.11" / "site-packages"
    site.mkdir(parents=True)
    python = directory / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    monkeypatch.setattr(jobenv, "env_dir", lambda _home, _spec: directory)
    monkeypatch.setattr(jobenv, "env_python", lambda _home, _spec: python)
    monkeypatch.setattr(jobenv, "_PACKAGES", {})
    return site


def test_pip_list_runs_once_until_the_env_changes(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every /v1/info read every voice's env status, and a `pip list` per read held the
    event loop 1-3 s on the PC (2026-10-02)."""
    runs: list[int] = []
    listed = {"torch": "2.5.1"}
    monkeypatch.setattr(jobenv, "_pip_list", lambda _home, _spec: runs.append(1) or dict(listed))
    assert jobenv.installed_packages(Path("home"), None) == listed
    assert jobenv.installed_packages(Path("home"), None) == listed
    assert len(runs) == 1
    (env / "narrator-1.0.dist-info").mkdir()
    import os
    os.utime(env, ns=(env.stat().st_atime_ns, env.stat().st_mtime_ns + 1_000_000))
    listed["narrator"] = "1.0"
    assert jobenv.installed_packages(Path("home"), None) == listed
    assert len(runs) == 2, "an install moves site-packages, and the list is read again"


def test_a_caller_cannot_change_what_the_cache_holds(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobenv, "_pip_list", lambda _home, _spec: {"torch": "2.5.1"})
    jobenv.installed_packages(Path("home"), None)["torch"] = "9"
    assert jobenv.installed_packages(Path("home"), None) == {"torch": "2.5.1"}
