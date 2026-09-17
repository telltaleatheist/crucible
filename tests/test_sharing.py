from pathlib import Path
import json

import pytest

from crucible import sharing, launcher
from crucible.errors import CrucibleError
from crucible.host.runner import RunResult


class Runner:
    def __init__(self, entry=None):
        self.entry = entry
        self.calls = []
        self.connected = True
        self.reject = False

    def run(self, argv, *, timeout_s):
        self.calls.append(argv)
        if argv[1:] == ["status", "--json"]:
            body = {"BackendState": "Running" if self.connected else "Stopped", "Self": {"DNSName": "pc.tail.ts.net."}}
        elif argv[1:] == ["serve", "status", "--json"]:
            body = {"TCP": {} if self.entry is None else {"7100": self.entry}}
        else:
            if self.reject:
                return RunResult(1, "", "permission denied", None)
            self.entry = None if argv[-1] == "off" else {"TCPForward": "127.0.0.1:7100"}
            body = {}
        return RunResult(0, json.dumps(body), "", None)


class Engine:
    target = "127.0.0.1:7100"

    def __init__(self):
        self.addresses = []
        self.fail_publish = False

    def verify(self):
        pass

    def advertise(self, field, addresses):
        assert field == "tailscale_advertise", field
        if self.fail_publish:
            raise sharing.SharingError("publish interrupted")
        self.addresses = addresses

    def request(self, *args):
        return {"tailscale_advertise": self.addresses}


def test_enable_publish_verify_and_disable_only_owned_port(tmp_path):
    runner, engine = Runner(), Engine()
    result = sharing.enable(tmp_path, runner, engine)
    assert result["remote_reachability"] == "not_tested"
    assert engine.addresses == ["pc.tail.ts.net:7100"]
    assert sharing.status(tmp_path, runner, engine)["state"] == "configured"
    sharing.disable(tmp_path, runner, engine)
    assert runner.entry is None and engine.addresses == []
    assert not (tmp_path / sharing.RECORD).exists()
    assert not any("reset" in argv for argv in runner.calls)


def test_existing_forward_requires_explicit_adoption(tmp_path):
    runner, engine = Runner({"TCPForward": "127.0.0.1:7100"}), Engine()
    with pytest.raises(sharing.SharingError, match="sharing_unowned"):
        sharing.enable(tmp_path, runner, engine)
    assert not (tmp_path / sharing.RECORD).exists()
    assert sharing.enable(tmp_path, runner, engine, adopt=True)["state"] == "configured"


@pytest.mark.parametrize("entry", [{"TCPForward": "localhost:9000"}, {"HTTPS": True}, {"TCPForward": "127.0.0.1:7100", "TerminateTLS": "pc"}])
def test_conflicting_forward_is_never_overwritten(tmp_path, entry):
    runner = Runner(entry)
    with pytest.raises(sharing.SharingError, match="sharing_port_conflict"):
        sharing.enable(tmp_path, runner, Engine(), adopt=True)
    assert runner.entry == entry


def test_interrupted_publication_keeps_intent_and_reconciles(tmp_path):
    runner, engine = Runner(), Engine()
    engine.fail_publish = True
    with pytest.raises(sharing.SharingError):
        sharing.enable(tmp_path, runner, engine)
    assert sharing.read(tmp_path)["state"] == "pending"
    engine.fail_publish = False
    sharing.enable(tmp_path, runner, engine)
    assert sharing.status(tmp_path, runner, engine)["state"] == "configured"


def test_external_drift_is_reported_and_not_deleted(tmp_path):
    runner, engine = Runner(), Engine()
    sharing.enable(tmp_path, runner, engine)
    runner.entry = {"TCPForward": "127.0.0.1:9999"}
    assert sharing.status(tmp_path, runner, engine)["state"] == "degraded"
    with pytest.raises(sharing.SharingError, match="sharing_ownership_changed"):
        sharing.disable(tmp_path, runner, engine)
    assert runner.entry == {"TCPForward": "127.0.0.1:9999"}
    assert engine.addresses == []


def test_offline_tailscale_is_not_reported_as_reachable(tmp_path):
    runner, engine = Runner(), Engine()
    sharing.enable(tmp_path, runner, engine)
    runner.connected = False
    assert sharing.status(tmp_path, runner, engine)["state"] == "degraded"


def test_cli_launcher_upgrade_and_uninstall_preserve_unrelated_path_entries(tmp_path):
    saved = [r"C:\Tools"]
    def registry(value=None):
        current = saved[0]
        if value is not None:
            saved[0] = value
        return current
    home = tmp_path / "Crucible"
    record = launcher.install(home, "C:/Python/python.exe", "C:/runtime", platform="win32", path_store=registry)
    launcher.install(home, "C:/Python/python.exe", "C:/runtime-v2", platform="win32", path_store=registry)
    assert saved[0].count(str(home / "bin")) == 1
    assert "runtime-v2" in Path(record["path"]).read_text()
    launcher.remove(home, path_store=registry)
    assert saved == [r"C:\Tools"]
    assert not Path(record["path"]).exists()


def test_cli_launcher_refuses_unowned_or_modified_files(tmp_path):
    home = tmp_path / "crucible"
    user = tmp_path / "user"
    record = launcher.install(home, "/usr/bin/python", "/runtime", platform="linux", user_home=user)
    Path(record["path"]).write_text("user content")
    with pytest.raises(CrucibleError, match="cli_launcher_modified"):
        launcher.remove(home)
    assert Path(record["path"]).read_text() == "user content"
    with pytest.raises(CrucibleError, match="cli_launcher_conflict"):
        launcher.install(home, "/usr/bin/python", "/runtime", platform="linux", user_home=user)


def test_a_launcher_that_already_runs_this_crucible_is_adopted(tmp_path):
    """An upgrade replacing a shim written by something else is not a conflict.

    Measured 2026-09-16: BOTH of Owen's machines carried a hand-written shim —
    its own comments say it existed because `Scripts\crucible.exe --version`
    exited 1 — and the 0.6.3 install refused each one with "nothing changed"
    and no remedy. Replacing a launcher that already launches THIS Crucible is
    what an upgrade is.
    """
    home = tmp_path / "crucible"
    user = tmp_path / "user"
    record = launcher.install(home, "/usr/bin/python", "/runtime",
                              platform="linux", user_home=user)
    assert record["adopted"] is False

    # A launcher for this home, written by something that is not this installer.
    Path(record["path"]).write_text(
        "#!/bin/sh" + chr(10) +
        "export CRUCIBLE_HOME=" + str(home) + chr(10) +
        "exec /some/other/python -m crucible.cli \"$@\"" + chr(10)
    )
    again = launcher.install(home, "/usr/bin/python", "/runtime",
                             platform="linux", user_home=user)
    assert again["adopted"] is True
    assert "-m crucible.cli" in Path(again["path"]).read_text()


def test_a_stranger_named_crucible_is_still_refused_and_told_what_to_do(tmp_path):
    """The refusal protects somebody else's file, and must stay — with a remedy."""
    home = tmp_path / "crucible"
    user = tmp_path / "user"
    record = launcher.install(home, "/usr/bin/python", "/runtime",
                              platform="linux", user_home=user)
    Path(record["path"]).write_text("#!/bin/sh" + chr(10) + "echo not ours" + chr(10))
    with pytest.raises(CrucibleError, match="cli_launcher_conflict") as caught:
        launcher.install(home, "/usr/bin/python", "/runtime",
                         platform="linux", user_home=user)
    assert "Move it aside" in str(caught.value), "a refusal with no remedy is a dead end"
    assert Path(record["path"]).read_text().endswith("not ours" + chr(10))


def test_windows_interfaces_read_structured_addresses_and_filter(monkeypatch):
    import subprocess
    from crucible import interfaces
    def run(argv, **kwargs):
        assert "Get-NetIPAddress" in argv[-1]
        assert kwargs["timeout"] == 15
        return subprocess.CompletedProcess(argv, 0, '["127.0.0.1","169.254.1.1","10.0.0.3","100.64.0.1","10.0.0.3"]', '')
    monkeypatch.setattr(subprocess, "run", run)
    assert interfaces._windows_ipv4_addresses() == ["10.0.0.3", "100.64.0.1"]


def test_windows_custom_home_is_the_same_for_host_and_installer():
    from crucible.host.paths import crucible_root, host_pack_dir
    env = {"CRUCIBLE_HOME": r"D:\My models\Crucible"}
    assert str(crucible_root(env)) == r"D:\My models\Crucible"
    assert str(host_pack_dir(env)) == r"D:\My models\Crucible\host"


def test_guided_import_downloads_verifies_and_imports_only_owned_distro(tmp_path):
    from crucible.host.installer import EngineInstall
    class ImportRunner:
        def __init__(self):
            self.calls = []
        def run(self, argv, *, timeout_s):
            self.calls.append(argv)
            if argv[:3] == ["wsl.exe", "-l", "-v"]:
                output = "Ubuntu Running 2\n"
            elif argv[0] == "certutil" or argv[-1].endswith(".sha256"):
                output = "a" * 64
            elif argv[-1] == "/etc/wsl.conf":
                output = "# crucible-rootfs\n[boot]\nsystemd=true\n"
            else:
                output = ""
            return RunResult(0, output, "", None)
    runner = ImportRunner()
    walk = EngineInstall(runner, lambda event: None, release="0.6.0", home=tmp_path,
                         install_sh_url="https://example.invalid/install.sh")
    walk._import_distro()
    imported = [argv for argv in runner.calls if "--import" in argv]
    assert len(imported) == 1 and imported[0][2] == "crucible"
    assert imported[0][-2:] == ["--version", "2"]
    assert not any("--unregister" in argv or "--shutdown" in argv for argv in runner.calls)
    assert runner.calls.index(imported[0]) > next(i for i, argv in enumerate(runner.calls) if argv[0] == "certutil")


def test_guided_switch_executes_callbacks_instead_of_reporting_promises(tmp_path):
    from crucible.host.installer import EngineInstall
    from crucible.host.errors import HostError
    callbacks = []
    walk = EngineInstall(Runner(), lambda event: None, release="0.6.0", home=tmp_path,
                         install_sh_url="https://example.invalid/install.sh",
                         stop_windows_server=lambda: callbacks.append("stop"),
                         switch_pairing=lambda: callbacks.append("switch"))
    walk._stop_windows_server()
    walk._switch_pairing()
    assert callbacks == ["stop", "switch"]
    missing = EngineInstall(Runner(), lambda event: None, release="0.6.0", home=tmp_path,
                            install_sh_url="https://example.invalid/install.sh")
    with pytest.raises(HostError, match="host_switch_unavailable"):
        missing._switch_pairing()


def test_guest_catalog_carries_the_required_api_header():
    from crucible.host.catalog import GuestCatalog
    argv = GuestCatalog(Runner(), "crucible", "token", 7100, where="guest").curl_argv("GET", "/v1/catalog", None)
    assert "X-Crucible-Api: 1" in argv
