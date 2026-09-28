from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from crucible import catalog, cli, service, uninstall
from crucible.backend import CUDA_LINUX, MLX_DARWIN


class Runner:

    def __init__(self, answers: dict[tuple[str, ...], service.Ran] | None = None):
        self.calls: list[tuple[str, ...]] = []
        self._answers = answers or {}

    def __call__(self, argv: Sequence[str]) -> service.Ran:
        argv = tuple(argv)
        self.calls.append(argv)
        for prefix, ran in self._answers.items():
            if argv[: len(prefix)] == prefix:
                return service.Ran(
                    argv=argv,
                    returncode=ran.returncode,
                    stdout=ran.stdout,
                    stderr=ran.stderr,
                )
        return service.Ran(argv=argv, returncode=0, stdout="", stderr="")

    def ran(self, *words: str) -> bool:
        return any(call[: len(words)] == words for call in self.calls)


def answer(*, code: int = 0, out: str = "", err: str = "") -> service.Ran:
    return service.Ran(argv=(), returncode=code, stdout=out, stderr=err)


WINDOWS_ENV = {
    "LOCALAPPDATA": r"C:\Users\test\AppData\Local",
    "APPDATA": r"C:\Users\test\AppData\Roaming",
    "USERNAME": "test",
}


@pytest.fixture
def installed_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    (home / "envs" / "tts-higgs-v3" / "bin").mkdir(parents=True)
    (home / "envs" / "tts-higgs-v3" / "bin" / "python").write_text("x", encoding="utf-8")
    (home / "logs").mkdir()
    (home / "logs" / "serve.log").write_text("hello", encoding="utf-8")
    (home / "jobs" / "j1").mkdir(parents=True)
    (home / "uploads").mkdir()
    (home / "downloads").mkdir()
    (home / "server" / "bin").mkdir(parents=True)
    (home / "server" / "bin" / "crucible").write_text("#!/bin/sh\n", encoding="utf-8")
    for dirname in uninstall.SUBJECT_DIRS.values():
        directory = home / dirname / "something"
        directory.mkdir(parents=True)
        (directory / "weights.bin").write_bytes(b"w" * 1000)
    (home / "config.toml").write_text(
        '[server]\nname = "crucible@test"\nhost = "127.0.0.1"\nport = 7100\n'
        '[auth]\ntoken = "t"\n[backend]\nkind = "cuda-linux"\n[jobs]\n'
        '[accelerator]\ndesktop_allowance_bytes = 0\n',
        encoding="utf-8",
    )
    (home / "pairing").write_text("crucible://x@127.0.0.1:7100/#t\n", encoding="utf-8")
    return home


@pytest.fixture
def unit_home(tmp_path: Path) -> Path:
    operator = tmp_path / "operator"
    unit = service.unit_path(operator)
    unit.parent.mkdir(parents=True)
    unit.write_text("[Unit]\n", encoding="utf-8")
    return operator


def make(
    home: Path,
    *,
    platform: str = "linux",
    runner: Runner | None = None,
    user_home: Path | None = None,
    **kwargs: object,
) -> uninstall.Plan:
    return uninstall.plan(
        home=home,
        platform=platform,
        env=WINDOWS_ENV,
        runner=runner if runner is not None else Runner(),
        user_home=user_home if user_home is not None else home / "no-operator-home",
        executable=str(home / "server" / "bin" / "python"),
        **kwargs,
    )


def step(plan: uninstall.Plan, name: str) -> uninstall.Step:
    for found in plan.steps:
        if found.name == name:
            return found
    raise AssertionError(f"no step called {name!r} in {[s.name for s in plan.steps]}")


def test_every_catalog_kind_has_a_directory_and_no_seventh_exists() -> None:
    assert tuple(uninstall.SUBJECT_DIRS) == catalog.KINDS


def test_the_platform_table_agrees_with_service_pys_backend_table() -> None:
    assert uninstall.UNINSTALL_MECHANISM["linux"] == service.SERVICE_MECHANISM[CUDA_LINUX]
    assert uninstall.UNINSTALL_MECHANISM["darwin"] == service.SERVICE_MECHANISM[MLX_DARWIN]
    assert uninstall.UNINSTALL_MECHANISM["win32"] == uninstall.STARTUP
    assert set(service.SERVICE_MECHANISM) == {CUDA_LINUX, MLX_DARWIN}


def test_a_platform_with_no_supervisor_is_refused_by_name() -> None:
    with pytest.raises(uninstall.UninstallError) as caught:
        uninstall.mechanism_for_platform("freebsd")
    assert "uninstall_no_mechanism" in str(caught.value)
    assert "freebsd" in str(caught.value)


def test_the_plan_is_the_install_list_read_upwards(installed_home: Path) -> None:
    plan = make(installed_home)
    names = [s.name for s in plan.steps]
    order = [
        "stop-engine",
        "remove-service",
        "remove-envs",
        "remove-pairing",
        "remove-config",
        "remove-logs",
        "weights:model",
        "remove-home",
    ]
    positions = [names.index(name) for name in order]
    assert positions == sorted(positions), names


def test_the_guest_is_stopped_and_uninstalled_before_the_tray_that_holds_it_open_is_ended(
    installed_home: Path,
) -> None:
    runner = Runner({("wsl.exe", "-l"): answer(out=f"Ubuntu\n{uninstall.CRUCIBLE_DISTRO}\n")})
    plan = make(installed_home, platform="win32", runner=runner, wsl_too=True)
    names = [s.name for s in plan.steps]
    assert names.index("stop-engine") < names.index("wsl-guest")
    assert names.index("wsl-guest") < names.index("stop-controller")
    assert names.index("stop-controller") < names.index("remove-service")


def test_a_dry_run_touches_nothing(installed_home: Path) -> None:
    before = sorted(path.name for path in installed_home.iterdir())
    plan = make(installed_home, purge_weights=True)
    assert plan.dry_run is True
    assert all(not s.done for s in plan.steps)
    assert sorted(path.name for path in installed_home.iterdir()) == before


def test_a_dry_run_names_every_step_the_real_run_performs(
    installed_home: Path, unit_home: Path
) -> None:
    dry = make(installed_home, user_home=unit_home)
    live = make(installed_home, user_home=unit_home, runner=Runner())
    assert [s.name for s in dry.steps] == [s.name for s in live.steps]
    assert [s.action for s in dry.steps] == [s.action for s in live.steps]


def test_a_real_run_removes_the_state_and_keeps_the_weights(
    installed_home: Path, unit_home: Path
) -> None:
    runner = Runner()
    plan = uninstall.run(
        make(installed_home, user_home=unit_home, runner=runner)
    )
    assert not plan.fatal
    assert not (installed_home / "config.toml").exists()
    assert not (installed_home / "pairing").exists()
    assert not (installed_home / "envs").exists()
    assert not (installed_home / "logs").exists()
    assert (installed_home / "jobs" / "j1").exists()
    assert (installed_home / "uploads").exists()
    for dirname in uninstall.SUBJECT_DIRS.values():
        assert (installed_home / dirname).is_dir(), f"{dirname} is KEPT by default"
    assert plan.kept()["weights_bytes"] == 6000, "six subject dirs, 1000 bytes each"


def test_purge_weights_removes_all_six_subject_directories(
    installed_home: Path, unit_home: Path
) -> None:
    plan = uninstall.run(
        make(installed_home, user_home=unit_home, purge_weights=True)
    )
    assert not plan.fatal
    for dirname in uninstall.SUBJECT_DIRS.values():
        assert not (installed_home / dirname).exists()
    assert plan.kept()["weights_bytes"] == 0
    assert plan.removed_bytes() >= 6000


def test_the_pack_it_is_running_from_is_kept_and_named(
    installed_home: Path, unit_home: Path
) -> None:
    plan = uninstall.run(
        make(installed_home, user_home=unit_home, purge_weights=True)
    )
    pack = step(plan, "pack:server")
    assert pack.action == uninstall.KEEP
    assert "install.sh --uninstall" in pack.what
    assert "running this very command" in pack.what
    assert (installed_home / "server" / "bin" / "crucible").is_file()


def test_the_home_survives_while_the_pack_is_in_it(
    installed_home: Path, unit_home: Path
) -> None:
    plan = uninstall.run(
        make(installed_home, user_home=unit_home, purge_weights=True)
    )
    home_step = step(plan, "remove-home")
    assert home_step.action == uninstall.KEEP
    assert home_step.refused is not None
    assert home_step.refused.code == "home_not_empty"
    assert home_step.refused.fatal is False
    assert "server" in home_step.refused.message
    assert installed_home.is_dir()


def test_the_home_goes_when_the_last_step_leaves_it_empty(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text("x", encoding="utf-8")
    (home / "pairing").write_text("x", encoding="utf-8")
    plan = uninstall.run(
        uninstall.plan(
            home=home,
            platform="linux",
            env={},
            runner=Runner(),
            user_home=tmp_path / "operator",
            executable=str(tmp_path / "elsewhere" / "python"),
        )
    )
    assert not plan.fatal
    assert not home.exists()


def test_an_entry_crucible_did_not_write_is_kept_and_reported(
    installed_home: Path, unit_home: Path
) -> None:
    stray = installed_home / "hf-token.txt"
    stray.write_text("hf_xxx", encoding="utf-8")
    plan = uninstall.run(
        make(installed_home, user_home=unit_home, purge_weights=True)
    )
    kept = step(plan, "keep-unknown:hf-token.txt")
    assert kept.action == uninstall.KEEP
    assert stray.is_file()
    assert str(stray) in plan.kept()["paths"]


def test_nothing_outside_crucible_home_is_ever_removed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "somebody-elses-file"
    outside.write_text("x", encoding="utf-8")
    with pytest.raises(uninstall.UninstallError) as caught:
        uninstall._remove_path(home, outside)
    assert "unsafe_target" in str(caught.value)
    assert outside.is_file()


def test_the_service_is_stopped_before_anything_is_deleted(
    installed_home: Path, unit_home: Path
) -> None:
    runner = Runner()
    uninstall.run(make(installed_home, user_home=unit_home, runner=runner))
    stop_at = runner.calls.index(("systemctl", "--user", "stop", service.UNIT_NAME))
    disable_at = runner.calls.index(
        ("systemctl", "--user", "disable", "--now", service.UNIT_NAME)
    )
    assert stop_at < disable_at
    assert runner.ran("systemctl", "--user", "daemon-reload")
    assert not service.unit_path(unit_home).exists()


def test_a_missing_service_is_refused_by_name_and_is_not_a_failure(
    installed_home: Path, tmp_path: Path
) -> None:
    runner = Runner()
    plan = uninstall.run(
        make(installed_home, user_home=tmp_path / "no-unit", runner=runner)
    )
    for name in ("stop-engine", "remove-service"):
        refused = step(plan, name).refused
        assert refused is not None
        assert refused.code == "service_not_installed"
        assert refused.fatal is False
        assert str(service.unit_path(tmp_path / "no-unit")) in refused.message
    assert not plan.fatal
    assert not runner.ran("systemctl")


def test_a_launchd_uninstall_boots_the_agent_out_and_deletes_the_plist(
    installed_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    operator = tmp_path / "operator"
    plist = service.plist_path(operator)
    plist.parent.mkdir(parents=True)
    plist.write_text("<plist/>", encoding="utf-8")
    monkeypatch.setattr(service.os, "getuid", lambda: 501, raising=False)
    runner = Runner({("launchctl", "list"): answer(out=f"123 0 {service.LAUNCHD_LABEL}\n")})
    plan = uninstall.run(
        make(installed_home, platform="darwin", user_home=operator, runner=runner)
    )
    assert plan.mechanism == service.LAUNCHD
    assert runner.ran("launchctl", "bootout")
    assert not plist.exists()


def test_a_service_that_will_not_stop_preserves_its_runtime_and_config(
    installed_home: Path, unit_home: Path
) -> None:
    runner = Runner(
        {("systemctl", "--user", "stop"): answer(code=1, err="Failed to stop")}
    )
    plan = uninstall.run(make(installed_home, user_home=unit_home, runner=runner))
    assert [s.name for s in plan.fatal] == ["stop-engine"]
    assert (installed_home / "config.toml").exists(), "a live service must retain its config"
    assert (installed_home / "envs").exists(), "a live service must retain its runtime"
    assert plan.to_dict()["ok"] is False


def test_on_windows_the_service_is_the_startup_shortcut(installed_home: Path) -> None:
    plan = make(installed_home, platform="win32")
    removal = step(plan, "remove-service")
    assert removal.target.endswith("Startup\\Crucible.lnk")
    assert "at login" in removal.what


def test_the_tray_is_asked_to_stop_its_engine_then_to_quit_and_is_never_force_killed(
    installed_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (installed_home / "host.pid").write_text("4242", encoding="utf-8")
    python = str(installed_home / "server" / "bin" / "python")
    runner = Runner({("wsl.exe", "-l"): answer(out=f"Ubuntu\n{uninstall.CRUCIBLE_DISTRO}\n")})
    monkeypatch.setattr(
        uninstall, "alive",
        lambda pid: pid == 4242 and not runner.ran(python, "-m", "crucible.cli", "local", "shutdown"),
    )
    plan = uninstall.run(make(installed_home, platform="win32", runner=runner, wsl_too=True))
    assert step(plan, "stop-engine").target == "pid 4242"
    assert step(plan, "stop-engine").done and step(plan, "stop-controller").done
    stop = runner.calls.index((python, "-m", "crucible.cli", "local", "stop"))
    guest = next(i for i, call in enumerate(runner.calls) if call[0] == "wsl.exe" and "uninstall" in call[-1])
    quit_ = runner.calls.index((python, "-m", "crucible.cli", "local", "shutdown"))
    assert stop < guest < quit_, runner.calls
    for call in runner.calls:
        assert "taskkill" not in call[0].lower(), call
        assert "/F" not in call and "/T" not in call, call


def test_a_controller_that_will_not_quit_is_named_with_its_log_and_nothing_is_removed(
    installed_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (installed_home / "host.pid").write_text("4242", encoding="utf-8")
    monkeypatch.setattr(uninstall, "alive", lambda pid: pid == 4242)
    plan = uninstall.run(make(installed_home, platform="win32"))
    refused = step(plan, "stop-controller").refused
    assert refused is not None and refused.fatal
    assert refused.code == "stop_failed"
    assert "pid 4242" in refused.message
    assert str(installed_home / "host.log") in refused.message
    assert "Task Manager" in refused.message
    assert "taskkill" not in refused.message
    assert (installed_home / "config.toml").exists()
    assert (installed_home / "envs").exists()


def test_a_controller_with_no_engine_has_nothing_to_stop_and_is_still_ended(
    installed_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (installed_home / "host.pid").write_text("4242", encoding="utf-8")
    (installed_home / "pairing").unlink()
    python = str(installed_home / "server" / "bin" / "python")
    runner = Runner()
    monkeypatch.setattr(
        uninstall, "alive",
        lambda pid: pid == 4242 and not runner.ran(python, "-m", "crucible.cli", "local", "shutdown"),
    )
    plan = uninstall.run(make(installed_home, platform="win32", runner=runner))
    assert step(plan, "stop-engine").done
    assert not runner.ran(python, "-m", "crucible.cli", "local", "stop")
    assert runner.ran(python, "-m", "crucible.cli", "local", "shutdown")


def test_a_stale_lock_is_not_a_running_tray(
    installed_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (installed_home / "host.pid").write_text("4242", encoding="utf-8")
    monkeypatch.setattr(uninstall, "alive", lambda pid: False)
    plan = make(installed_home, platform="win32")
    for name in ("stop-engine", "stop-controller"):
        refused = step(plan, name).refused
        assert refused is not None
        assert refused.code == "engine_not_running"
        assert refused.fatal is False


def test_wsl_too_refuses_by_name_when_the_crucible_distro_is_not_there(
    installed_home: Path,
) -> None:
    runner = Runner({("wsl.exe", "-l"): answer(out="Ubuntu\ndocker-desktop\n")})
    plan = make(installed_home, platform="win32", runner=runner, wsl_too=True)
    refused = step(plan, "wsl-guest").refused
    assert refused is not None
    assert refused.code == "wsl_distro_absent"
    assert refused.fatal is True
    assert "Ubuntu" in refused.message


def test_wsl_too_never_touches_a_distro_crucible_did_not_import(
    installed_home: Path,
) -> None:
    runner = Runner(
        {("wsl.exe", "-l"): answer(out=f"Ubuntu\n{uninstall.CRUCIBLE_DISTRO}\n")}
    )
    uninstall.run(
        make(installed_home, platform="win32", runner=runner, wsl_too=True)
    )
    wsl_calls = [call for call in runner.calls if call[0] == "wsl.exe"]
    assert all("Ubuntu" not in word for call in wsl_calls for word in call)
    assert any("--unregister" not in word for call in wsl_calls for word in call)
    assert not any("--unregister" in word for call in wsl_calls for word in call)


def test_the_guest_argv_lets_the_GUESTs_bash_expand_its_own_home() -> None:
    argv = uninstall.wsl_uninstall_argv(purge_weights=True, dry_run=False)
    assert argv[:8] == [
        "wsl.exe",
        "-d",
        uninstall.CRUCIBLE_DISTRO,
        "-u",
        "crucible",
        "--exec",
        "bash",
        "-lc",
    ]
    assert argv[8] == (
        '"${CRUCIBLE_HOME:-$HOME/.crucible}/server/bin/crucible" uninstall --json '
        "--purge-weights"
    )
    assert "--wsl-too" not in argv[8], "there is no distro inside the distro"


def test_wsl_too_off_windows_is_refused_by_name(installed_home: Path) -> None:
    plan = make(installed_home, platform="linux", wsl_too=True)
    refused = step(plan, "wsl-guest").refused
    assert refused is not None
    assert refused.code == "wsl_not_here"
    assert refused.fatal is True


def test_the_json_shape_is_the_one_the_apps_read(
    installed_home: Path, unit_home: Path
) -> None:
    dry = make(installed_home, user_home=unit_home).to_dict()
    assert set(dry) == {
        "dry_run",
        "home",
        "platform",
        "mechanism",
        "backend_kind",
        "purge_weights",
        "wsl_too",
        "steps",
        "kept",
        "removed_bytes",
        "ok",
    }
    assert dry["dry_run"] is True
    assert dry["backend_kind"] == "cuda-linux", "READ from config.toml, never detected"
    assert set(dry["kept"]) == {"weights_bytes", "paths"}
    for row in dry["steps"]:
        assert set(row) >= {"name", "what", "action", "target", "done"}
        assert row["action"] in {"remove", "stop", "keep"}
        assert row["done"] is False
        if "refused" in row:
            assert set(row["refused"]) == {"code", "message", "fatal"}
    assert json.loads(json.dumps(dry))["home"] == str(installed_home)

    live = uninstall.run(make(installed_home, user_home=unit_home)).to_dict()
    assert live["dry_run"] is False
    assert any(row["done"] for row in live["steps"])


def test_a_home_with_no_config_reports_the_backend_as_null(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / "logs").mkdir(parents=True)
    plan = make(home)
    assert plan.backend_kind is None
    assert plan.to_dict()["backend_kind"] is None


def test_the_cli_verb_exists_with_its_four_flags() -> None:
    parsed = cli.build_parser().parse_args(
        ["uninstall", "--dry-run", "--purge-weights", "--wsl-too", "--json"]
    )
    assert parsed.func is cli.cmd_uninstall
    assert (parsed.dry_run, parsed.purge_weights, parsed.wsl_too, parsed.json) == (
        True,
        True,
        True,
        True,
    )


def test_the_cli_dry_run_prints_the_plan_and_changes_nothing(
    installed_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("CRUCIBLE_HOME", str(installed_home))
    before = sorted(path.name for path in installed_home.iterdir())
    code = cli.main(["uninstall", "--dry-run", "--json"])
    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["dry_run"] is True
    assert sorted(path.name for path in installed_home.iterdir()) == before


def test_the_cli_exits_1_when_a_step_could_not_be_done(
    installed_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("CRUCIBLE_HOME", str(installed_home))
    real = uninstall.plan

    def one_fatal(**kwargs: object) -> uninstall.Plan:
        built = real(**kwargs)
        built.steps[0].refused = uninstall.Refusal(
            code="stop_failed", message="the unit would not stop", fatal=True
        )
        built.steps[0].act = None
        return built

    monkeypatch.setattr(uninstall, "plan", one_fatal)
    code = cli.main(["uninstall", "--dry-run"])
    assert code == 1
    assert "uninstall_incomplete" in capsys.readouterr().err


def test_the_cli_refuses_by_name_when_this_host_cannot_say_where_home_is(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CRUCIBLE_HOME", raising=False)
    monkeypatch.setattr(cli.uninstall_cmd.sys, "platform", "win32")
    monkeypatch.setattr(cli.uninstall_cmd.os, "environ", {}, raising=False)
    assert cli.main(["uninstall", "--dry-run"]) == 1
    assert "LOCALAPPDATA" in capsys.readouterr().err
