from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pytest

from crucible import wsl
from crucible.host import installer
from crucible.platform.errors import HostError
from crucible.platform.runner import RunResult
from crucible.platform.wsl_table import CRUCIBLE_DISTRO, WSL_CONF_TEXT

WHOAMI = " ".join(wsl.whoami_argv(CRUCIBLE_DISTRO))
TERMINATE = " ".join(wsl.terminate_argv(CRUCIBLE_DISTRO))
HELP = " ".join(wsl.help_argv())
SET_USER = " ".join(wsl.set_default_user_argv(CRUCIBLE_DISTRO, wsl.GUEST_USER))

# `wsl --help` as WSL 2.5.7 prints it (captured on Owen's PC, NULs stripped), and
# the same text with the --set-default-user option taken out, as an older WSL's.
WSL_HELP = (Path(__file__).parent / "data" / "wsl-help-2.5.7.txt").read_text(encoding="utf-8")
OLD_WSL_HELP = "\n".join(
    line for line in WSL_HELP.splitlines()
    if "--set-default-user" not in line and "Set the default user of the distribution" not in line
)


def _ok(stdout: str = "") -> RunResult:
    return RunResult(code=0, stdout=stdout, stderr="", failure=None)


class Distro:
    """A `crucible` distro that reads /etc/wsl.conf only when it starts, as WSL does."""

    def __init__(self, *, booted_with_conf: bool, conf_names_user: bool = True, present: bool = True,
                 terminate: RunResult | None = None, whoami: RunResult | None = None,
                 help_text: str = WSL_HELP, set_user: RunResult | None = None,
                 registry_names_user: bool = True) -> None:
        self.present = present
        self.help_text = help_text
        self.set_user = set_user or _ok()
        self.registry_names_user = registry_names_user
        self.registered = False
        self.booted_with_conf = booted_with_conf
        self.conf_names_user = conf_names_user
        self.terminate = terminate or _ok()
        self.whoami = whoami
        self.calls: list[str] = []

    def run(self, argv: Sequence[str], *, timeout_s: float, env: Mapping[str, str] | None = None) -> RunResult:
        line = " ".join(argv)
        self.calls.append(line)
        if line == TERMINATE:
            if self.terminate.ok:
                self.booted_with_conf = True
            return self.terminate
        if line == HELP:
            return _ok(self.help_text)
        if line == SET_USER:
            if self.set_user.ok and self.registry_names_user:
                self.registered = True
            return self.set_user
        if line == WHOAMI:
            if self.whoami is not None:
                return self.whoami
            named = self.registered or (self.booted_with_conf and self.conf_names_user)
            return _ok(("crucible" if named else "root") + "\n")
        if line.endswith("-l -v"):
            return _ok(f"  {CRUCIBLE_DISTRO}  Running  2\n" if self.present else "  Ubuntu  Running  2\n")
        if "cat /etc/wsl.conf" in line:
            return _ok(WSL_CONF_TEXT)
        return _ok()


def _walk(tmp_path: Path, runner: Distro, events: list) -> installer.EngineInstall:
    return installer.EngineInstall(
        runner, events.append, release="1.0.112", home=tmp_path,
        install_sh_url="https://example.invalid/install.sh",
    )


def test_the_import_restarts_the_distro_so_its_wsl_conf_is_the_one_running(tmp_path: Path, monkeypatch) -> None:
    runner = Distro(booted_with_conf=False, present=False)
    walk = _walk(tmp_path, runner, [])
    monkeypatch.setattr(walk, "_download_ubuntu_image", lambda downloads: downloads / "rootfs.tar.gz")
    monkeypatch.setattr(walk, "_unpack_ubuntu_image", lambda archive, destination: None)
    walk._import_distro()
    assert not any("cat /etc/wsl.conf" in call for call in runner.calls), "took the import path, not the keep path"
    assert runner.calls.count(TERMINATE) == 1
    assert runner.calls.index(WHOAMI) < runner.calls.index(TERMINATE)
    assert runner.calls.index(TERMINATE) < runner.calls.index(SET_USER), (
        "the boot is read from who it enters as BEFORE the registry is told, which would answer it"
    )
    assert runner.calls[-1] == WHOAMI, "the restart is verified, not assumed"
    assert runner.booted_with_conf


def test_an_existing_distro_still_on_its_first_boot_is_restarted(tmp_path: Path) -> None:
    events: list = []
    runner = Distro(booted_with_conf=False)
    _walk(tmp_path, runner, events)._import_distro()
    assert runner.calls.count(TERMINATE) == 1
    lines = [e.data["text"] for e in events if e.event == "line"]
    assert any("takes effect" in line and "crucible the user" in line for line in lines), lines


def test_a_distro_that_already_enters_as_crucible_is_left_running(tmp_path: Path) -> None:
    runner = Distro(booted_with_conf=True)
    _walk(tmp_path, runner, [])._import_distro()
    assert TERMINATE not in runner.calls
    assert WHOAMI in runner.calls
    assert SET_USER in runner.calls, "the registry is told even when wsl.conf already holds"


def test_the_registry_names_the_user_where_wsl_conf_did_not_take(tmp_path: Path) -> None:
    # The laptop's case: the conf said crucible and the distro still entered as root.
    events: list = []
    runner = Distro(booted_with_conf=False, conf_names_user=False)
    _walk(tmp_path, runner, events)._import_distro()
    assert runner.calls.count(TERMINATE) == 1
    assert runner.calls.count(SET_USER) == 1
    assert runner.calls[-1] == WHOAMI, "and it is verified, not assumed"
    lines = [e.data["text"] for e in events if e.event == "line"]
    assert any(SET_USER in line for line in lines), lines


def test_an_older_wsl_without_the_option_keeps_wsl_conf_and_says_so(tmp_path: Path) -> None:
    events: list = []
    runner = Distro(booted_with_conf=False, help_text=OLD_WSL_HELP)
    _walk(tmp_path, runner, events)._import_distro()
    assert HELP in runner.calls
    assert SET_USER not in runner.calls, "asked of wsl.exe, never tried blind"
    lines = [e.data["text"] for e in events if e.event == "line"]
    assert any(
        "has no `wsl --manage <distro> --set-default-user`" in line and "wsl --update" in line
        for line in lines
    ), lines


def test_an_older_wsl_whose_conf_did_not_take_is_refused_with_the_update_named(tmp_path: Path) -> None:
    runner = Distro(booted_with_conf=False, conf_names_user=False, help_text=OLD_WSL_HELP)
    with pytest.raises(HostError) as caught:
        _walk(tmp_path, runner, [])._import_distro()
    assert caught.value.code == "distro_default_user"
    assert '"root"' in caught.value.message
    assert "wsl --update" in caught.value.message
    assert f"wsl -d {CRUCIBLE_DISTRO} -u root --exec cat /etc/wsl.conf" in caught.value.message
    assert installer.TRY_AGAIN_HINT in caught.value.message
    assert runner.calls.count(TERMINATE) == 1, "one restart, then a refusal, not a loop"


def test_a_distro_that_enters_as_root_even_after_the_registry_is_refused_by_name(tmp_path: Path) -> None:
    runner = Distro(booted_with_conf=False, conf_names_user=False, registry_names_user=False)
    with pytest.raises(HostError) as caught:
        _walk(tmp_path, runner, [])._import_distro()
    assert caught.value.code == "distro_default_user"
    assert '"root"' in caught.value.message
    assert f"wsl --manage {CRUCIBLE_DISTRO} --set-default-user crucible" in caught.value.message
    assert runner.calls.count(TERMINATE) == 1
    assert runner.calls.count(SET_USER) == 1, "one of each, then a refusal, not a loop"


def test_a_set_default_user_that_fails_is_refused_with_what_wsl_said(tmp_path: Path) -> None:
    runner = Distro(
        booted_with_conf=True,
        set_user=RunResult(code=1, stdout="", stderr="The user was not found.", failure=None),
    )
    with pytest.raises(HostError) as caught:
        _walk(tmp_path, runner, [])._import_distro()
    assert caught.value.code == "distro_default_user"
    assert "The user was not found." in caught.value.message
    assert installer.TRY_AGAIN_HINT in caught.value.message


def test_a_help_that_is_not_usage_text_is_not_read_as_no(tmp_path: Path) -> None:
    runner = Distro(booted_with_conf=True, help_text="")
    with pytest.raises(HostError) as caught:
        _walk(tmp_path, runner, [])._import_distro()
    assert caught.value.code == "wsl_read_failed"
    assert SET_USER not in runner.calls


def test_the_option_is_read_from_usage_text_as_wsl_exe_writes_it() -> None:
    from crucible.host import wslstate

    utf16_read_as_text = chr(0).join(WSL_HELP)
    assert wslstate.sets_default_user(_ok(utf16_read_as_text)) is True
    assert wslstate.sets_default_user(_ok(OLD_WSL_HELP)) is False
    assert wslstate.sets_default_user(RunResult(code=None, stdout="", stderr="", failure="not found")) is None


def test_a_restart_that_fails_is_refused_with_what_wsl_said(tmp_path: Path) -> None:
    runner = Distro(booted_with_conf=False, terminate=RunResult(code=1, stdout="", stderr="Access is denied.", failure=None))
    with pytest.raises(HostError) as caught:
        _walk(tmp_path, runner, [])._import_distro()
    assert caught.value.code == "distro_restart_failed"
    assert "Access is denied." in caught.value.message


def test_a_distro_that_will_not_say_who_it_enters_as_is_not_guessed_about(tmp_path: Path) -> None:
    runner = Distro(booted_with_conf=True, whoami=RunResult(code=1, stdout="", stderr="boom", failure=None))
    with pytest.raises(HostError) as caught:
        _walk(tmp_path, runner, [])._import_distro()
    assert caught.value.code == "wsl_read_failed"
    assert TERMINATE not in runner.calls


def test_the_default_user_is_asked_without_naming_a_user() -> None:
    assert wsl.whoami_argv(CRUCIBLE_DISTRO) == ["wsl.exe", "-d", CRUCIBLE_DISTRO, "--exec", "id", "-un"]
    assert "[user]\ndefault=crucible\n" in WSL_CONF_TEXT
