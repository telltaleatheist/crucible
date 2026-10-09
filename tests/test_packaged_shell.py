from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from crucible import controller_client, local
from crucible.cli import orchestrator
from crucible.platform import packaged
from crucible.platform.errors import LocalError

PACKAGE = "Claude_2.19675.0.0_x64__pzs8sxrjxfjjc"
ROOT = Path(__file__).resolve().parent.parent


def _install_ps1() -> str:
    return (ROOT / "sdk" / "bootstrap" / "scripts" / "install.ps1").read_text(encoding="utf-8")


def _inside_a_package(monkeypatch) -> None:
    monkeypatch.setattr(packaged.sys, "platform", "win32")
    monkeypatch.setattr(packaged, "_kernel32_package_name", lambda: PACKAGE)


def test_off_windows_nothing_is_asked() -> None:
    def never() -> str:
        raise AssertionError("the package probe is a Windows question")

    assert packaged.package_name("linux", never) is None
    packaged.refuse_packaged("anything", platform="darwin", probe=never)


def test_an_ordinary_process_passes() -> None:
    assert packaged.package_name("win32", lambda: None) is None
    packaged.refuse_packaged("`crucible local register`", platform="win32", probe=lambda: None)


def test_a_packaged_process_is_refused_by_name_with_the_package_and_the_way_out() -> None:
    with pytest.raises(LocalError) as caught:
        packaged.refuse_packaged("`crucible local register`", platform="win32", probe=lambda: PACKAGE)
    text = str(caught.value)
    assert text.startswith("packaged_shell: `crucible local register` is running inside")
    assert PACKAGE in text
    assert "Nothing has been written." in text
    assert text.endswith(packaged.ORDINARY_POWERSHELL)


@pytest.mark.parametrize("action", local.HOME_WRITERS)
def test_every_local_verb_that_writes_the_home_is_refused_before_it_writes(monkeypatch, action) -> None:
    _inside_a_package(monkeypatch)
    monkeypatch.setattr(local, "publish_installation", lambda *_: pytest.fail("wrote installation.json"))
    monkeypatch.setattr(local, "run_engine_verb", lambda *_: pytest.fail("started the engine"))
    with pytest.raises(LocalError, match="packaged_shell"):
        local._answer(action, lambda verb: pytest.fail(f"ran the tray verb {verb}"))


def test_a_reading_verb_is_not_refused(monkeypatch) -> None:
    _inside_a_package(monkeypatch)
    monkeypatch.setattr(local, "status", lambda: {"state": "running"})
    assert local._answer("status", lambda verb: None) == {"state": "running"}


def test_the_controller_is_never_spawned_from_inside_a_package(monkeypatch, tmp_path) -> None:
    _inside_a_package(monkeypatch)
    monkeypatch.setattr(controller_client.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned"))
    with pytest.raises(LocalError, match="packaged_shell"):
        controller_client.spawn(tmp_path)


def test_the_orchestrator_refuses_before_it_starts_a_tray(monkeypatch, capsys) -> None:
    _inside_a_package(monkeypatch)
    monkeypatch.setattr(orchestrator.sys, "platform", "win32")
    args = argparse.Namespace(try_again=False, install_startup=False, remove_startup=False, headless=True)
    assert orchestrator.cmd_orchestrator(args) != 0
    assert "packaged_shell" in capsys.readouterr().err


def test_install_ps1_says_the_same_sentence_as_the_host() -> None:
    assert packaged.refusal_sentence("$PackageName", "this installer") in _install_ps1()


def test_install_ps1_asks_before_anything_is_written_or_removed() -> None:
    script = _install_ps1()
    asked = script.index("[CruciblePackage]::Ask([ref]$PackageName)")
    for first_act in ("if ($Uninstall) {", "New-Item", "Remove-Item", "curl.exe", "Set-Content", "Start-Process"):
        assert asked < script.index(first_act), f"{first_act} comes before the package check"
    assert "if ($PackageCode -eq 0) { Die " in script
    assert "if ($PackageCode -ne 15700) { Die \"package_check_failed:" in script

