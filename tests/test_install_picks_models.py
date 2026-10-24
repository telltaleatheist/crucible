"""Installing a package is one press that also downloads the models picked with it (Owen,
2026-10-24: "click install package, and itll activate it in wsl and download the models.
the user can pick which models they want to download"); on the command line a pull of a
model whose package is not installed says so (install the package, then pull).

The card is Victoria's laptop's: an 8 GiB RTX 3070 with a 1 GiB desktop allowance, where
qwen3-asr-0.6b fits six pieces at a time and qwen3-asr-1.7b does not fit even one."""

from __future__ import annotations

from types import SimpleNamespace

from crucible import catalog, installplan, jobflags, verdict
from crucible.cli import weights as cli_weights

GIB = 1 << 30


def _asr_plan() -> dict:
    decisions = verdict.decide_all(
        "cuda-linux", total_bytes=8 * GIB, desktop_allowance_bytes=GIB,
        gpu_vendor="nvidia", chosen={}, audio_low_vram=False,
    )
    return installplan.install_plan("asr", decisions, card=None, total_bytes=8 * GIB, pool="card")


def test_the_install_plan_lists_each_model_once_with_whether_it_runs_here() -> None:
    models = {row["id"]: row for row in _asr_plan()["models"]}
    assert len(models) == len(_asr_plan()["models"]), "a model is listed once"
    small, large = models["qwen3-asr-0.6b"], models["qwen3-asr-1.7b"]
    assert small["usable"] is True and small["recommended"] is True
    assert large["usable"] is False and large["recommended"] is False
    assert any("Cannot" in line for line in large["lines"])
    assert sum(row["recommended"] for row in models.values()) >= 1


def _pull_note(monkeypatch, *, built: bool | None, enabled: bool) -> str | None:
    subject = SimpleNamespace(job_type="asr")
    monkeypatch.setattr(catalog, "find", lambda config, backend, kind, model_id: subject)
    monkeypatch.setattr(jobflags, "env_built", lambda config, backend, job_type: built)
    config = SimpleNamespace(enable_asr=enabled)
    return cli_weights._package_still_needed(config, None, "qwen3-asr-0.6b")


def test_a_pull_whose_package_is_not_installed_says_to_install_it(monkeypatch) -> None:
    note = _pull_note(monkeypatch, built=False, enabled=False)
    assert note is not None
    assert "the asr package is not installed" in note and "`crucible install asr`" in note


def test_a_pull_whose_package_is_ready_says_nothing_more(monkeypatch) -> None:
    assert _pull_note(monkeypatch, built=True, enabled=True) is None


def test_a_built_package_that_is_off_is_still_named(monkeypatch) -> None:
    assert _pull_note(monkeypatch, built=True, enabled=False) is not None
