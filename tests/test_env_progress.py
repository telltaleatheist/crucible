from __future__ import annotations

import threading
from pathlib import Path

import pytest

from crucible import cli, hosttools, jobenv
from crucible.cli.install import EnvProgress

from .conftest import FAKE_BACKEND


def test_the_progress_reads_pip_s_phases() -> None:
    said: list[str] = []
    progress = EnvProgress(said.append, clock=lambda: 0.0)
    assert progress.sentence() == "setting up the venv and pip"
    progress.feed("Collecting torch==2.5.1 (from -r recipe.txt (line 3))")
    progress.feed("  Downloading torch-2.5.1-cp311-none-linux_x86_64.whl (906.4 MB)")
    progress.feed("Collecting numpy==1.26.4")
    assert progress.sentence() == "2 packages collected so far"
    progress.feed("  Building wheel for flash-attn (setup.py): started")
    assert progress.sentence() == "building the wheel for flash-attn"
    progress.feed("  Building wheel for flash-attn (setup.py): finished with status 'done'")
    progress.feed("Installing collected packages: numpy, torch, flash-attn")
    assert said == ["  installing 3 packages"]
    assert progress.sentence() == "installing 3 packages"
    progress.feed("Successfully installed flash-attn-2.7 numpy-1.26.4 torch-2.5.1")
    assert progress.sentence() == "pip has finished; finishing the env"


def test_the_progress_speaks_on_its_own_clock_while_pip_is_silent() -> None:
    said: list[str] = []
    spoke = threading.Event()

    def say(line: str) -> None:
        said.append(line)
        spoke.set()

    with EnvProgress(say, every=0.01) as progress:
        progress.feed("Collecting torch==2.5.1")
        assert spoke.wait(5.0)
    assert any(line.startswith("  still installing, ") for line in said)


def test_install_prints_progress_without_verbose(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)
    assert cli.main(["init"]) == 0
    capsys.readouterr()

    def install_env(home_dir: Path, spec: jobenv.EnvSpec, backend_kind: str, **kwargs: object):
        on_line = kwargs["on_line"]
        assert callable(on_line), "the progress needs pip's lines even without --verbose"
        on_line("Collecting torch==2.5.1")
        on_line("Installing collected packages: torch")
        return jobenv.EnvStatus(True, home_dir, "ok", "3.11.16", {spec.headline: "0.1", "torch": "2"})

    monkeypatch.setattr(jobenv, "install_env", install_env)
    monkeypatch.setattr(cli.install, "_smoke_import", lambda *_: None)
    monkeypatch.setattr(cli.install, "_measure_step", lambda *_a, **_k: None)
    monkeypatch.setattr(cli.install, "_capability_step", lambda *_a: 0)
    monkeypatch.setattr(hosttools, "ensure_ffmpeg", lambda home_dir, **_: "ffmpeg: ok")
    monkeypatch.setattr(hosttools, "ensure_silero_vad", lambda home_dir: "speech detector: ok")
    assert cli.main(["install", "audio"]) == 0
    out = capsys.readouterr().out
    assert "  installing 1 packages" in out
    assert "Collecting torch" not in out, "pip's own lines stay behind --verbose"
