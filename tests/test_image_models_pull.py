from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible import cli, imageweights

from .conftest import FAKE_BACKEND


@pytest.mark.parametrize("model", ["dreamshaper-xl-lightning", "qwen-image-2.1"])
def test_models_pull_takes_an_image_model(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str
) -> None:
    """`crucible models pull <image model>` reaches the image weights' own pull. It died
    on `spec.form` (a chat model's alone) from 1.0.128 to 1.0.138: Victoria's laptop's
    install step for the small rung (2026-10-10)."""
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)
    assert cli.main(["init", "--enable-image"]) == 0
    capsys.readouterr()
    pulled: list[str] = []

    def pull(config, manifest, spec, *, force, on_line):
        pulled.append(manifest.id)
        return SimpleNamespace(bytes=7_100_000_000, path=home / "models" / manifest.id)

    monkeypatch.setattr(imageweights, "pull", pull)
    assert cli.main(["models", "pull", model]) == 0
    assert pulled == [model]
    assert f"{model}: 7.10 GB at" in capsys.readouterr().out
