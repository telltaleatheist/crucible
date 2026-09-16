"""CPU-only lifecycle boundaries; no installed service or real UI is touched."""
import sys
import threading
import json
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from crucible import desktop, local, sharing
from crucible.processlock import ProcessLock


def test_installation_record_preserves_venv_interpreter_symlink(monkeypatch, tmp_path):
    from pathlib import Path
    base = tmp_path / "base-python"
    base.touch()
    interpreter = tmp_path / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    try:
        interpreter.symlink_to(base)
    except OSError:
        pytest.skip("creating symlinks is not available to this test user")
    monkeypatch.setattr(local.sys, "executable", str(interpreter))
    record = json.loads(local.publish_installation(tmp_path / "home").read_text())
    assert Path(record["control"]["command"]) == interpreter
    assert Path(record["control"]["command"]) != interpreter.resolve()


def test_close_tray_waits_for_process_after_pid_file_removed(monkeypatch, tmp_path):
    from crucible.host import app
    pid = tmp_path / "tray.pid"
    pid.write_text("12345")
    monkeypatch.setattr(desktop, "crucible_home", lambda: tmp_path)
    checks = []
    def alive(value):
        assert value == 12345
        checks.append(value)
        return len(checks) < 4
    monkeypatch.setattr(app, "_alive", alive)
    waits = []
    def wait(seconds):
        waits.append(seconds)
        pid.unlink(missing_ok=True)
    monkeypatch.setattr(desktop.time, "sleep", wait)
    desktop.close_tray()
    assert len(checks) == 4
    assert len(waits) == 2


def test_close_tray_refuses_swap_if_process_lingers_after_pid_removal(monkeypatch, tmp_path):
    from crucible.host import app
    pid = tmp_path / "tray.pid"
    pid.write_text("12345")
    monkeypatch.setattr(desktop, "crucible_home", lambda: tmp_path)
    monkeypatch.setattr(app, "_alive", lambda value: True)
    times = iter([0, 0, 16])
    monkeypatch.setattr(desktop.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(desktop.time, "sleep", lambda seconds: pid.unlink())
    with pytest.raises(local.LocalError, match="tray_close_failed"):
        desktop.close_tray()


def test_process_lock_excludes_competitor_and_releases(tmp_path):
    first = ProcessLock(tmp_path / "process.lock")
    second = ProcessLock(tmp_path / "process.lock")
    try:
        assert first.acquire()
        assert not second.acquire()
        first.close()
        assert second.acquire()
        assert not first.acquire()
    finally:
        first.close()
        second.close()
    assert first.acquire()
    first.close()


def test_console_open_uses_ui_token_parameter(monkeypatch, tmp_path):
    monkeypatch.setattr(local, "connection", lambda _: ("http://127.0.0.1:7100", "test", "a+b/c"))
    opened = []
    monkeypatch.setattr(local.webbrowser, "open", opened.append)
    local.act("open-console", tmp_path)
    assert opened == ["http://127.0.0.1:7100/#token=a%2Bb%2Fc"]


def test_start_reports_optional_sharing_failure_without_hiding_healthy_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(local.sys, "platform", "win32")
    (tmp_path / "pairing").write_text("present")
    monkeypatch.setattr(local, "connection", lambda _: ("http://127.0.0.1:7100", "test", "secret"))
    monkeypatch.setattr(local, "request", lambda *a, **kw: {"crucible": True, "role": "orchestrator"})
    monkeypatch.setattr(local, "status", lambda home: {"state": "running", "detail": "ready"})
    def offline(home):
        assert home == tmp_path
        raise sharing.SharingError("Tailscale is offline")
    monkeypatch.setattr(sharing, "reconcile", offline)
    result = local.act("start", tmp_path)
    assert result["state"] == "running"
    assert result["sharing"] == {"state": "degraded", "detail": "Tailscale is offline", "remote_reachability": "not_tested"}


@pytest.mark.parametrize("existing_pairing", [False, True])
def test_start_does_not_spawn_over_http_error(monkeypatch, tmp_path, existing_pairing):
    from urllib.error import HTTPError
    monkeypatch.setattr(local.sys, "platform", "win32")
    if existing_pairing:
        (tmp_path / "pairing").write_text("present")
    monkeypatch.setattr(local, "connection", lambda _: ("http://127.0.0.1:7100", "test", "secret"))
    def incompatible(*a, **kw):
        raise HTTPError("http://127.0.0.1:7101/v1/ping", 404, "missing", {}, None)
    monkeypatch.setattr(local, "request", incompatible)
    monkeypatch.setattr(local, "_spawn_controller", lambda home: pytest.fail("must not spawn at an occupied port"))
    with pytest.raises(local.LocalError, match="wrong_controller"):
        local.act("start", tmp_path)


@pytest.mark.parametrize("owner", ["child", "wsl-unit", "found"])
@pytest.mark.parametrize("release,contract", [("0.6.0", None), ("0.6.99", 1), ("0.6.99", 2)])
def test_upgrade_uses_authenticated_supported_contract(monkeypatch, tmp_path, owner, release, contract):
    from crucible.host import app
    monkeypatch.setattr(local.sys, "platform", "win32")
    monkeypatch.setattr(local, "crucible_home", lambda: tmp_path)
    monkeypatch.setattr(desktop, "close_tray", lambda: None)
    monkeypatch.setattr(local, "connection", lambda home: ("http://127.0.0.1:7100", "test", "secret"))
    (tmp_path / "host.pid").write_text("12345")
    stopped = []
    calls = []
    def request(url, **kw):
        calls.append(url)
        if url.endswith("/v1/info"):
            assert kw["token"] == "secret"
            return {"role": "orchestrator", "server": {"version": release, "api_version": 1},
                    "engine": {"owner": owner}, "local_lifecycle_version": contract}
        if url.endswith("/quit"):
            assert kw["token"] == "secret" and kw["method"] == "POST"
            stopped.append(True)
            return {"quit": True}
        if stopped:
            if ":7100/" in url and owner in ("wsl-unit", "found"):
                pytest.fail("host replacement must not touch the guest engine")
            raise URLError(ConnectionRefusedError())
        return {"crucible": True, "role": "orchestrator"}
    monkeypatch.setattr(local, "request", request)
    monkeypatch.setattr(app, "_alive", lambda pid: not stopped)
    local_calls = []
    monkeypatch.setattr(local, "act", lambda action: local_calls.append(action))
    if contract == 2:
        with pytest.raises(local.LocalError, match="controller_upgrade_unsupported"):
            local.shutdown()
        assert not stopped and not local_calls
        return
    local.shutdown()
    assert local_calls == (["stop"] if contract == 1 else [])
    assert stopped == [True]
    assert not any("/local/" in url for url in calls)


def test_tray_ignores_a_reused_pid_when_kernel_lock_is_free(monkeypatch, tmp_path):
    import os
    (tmp_path / "tray.pid").write_text(str(os.getpid()))
    monkeypatch.setattr(desktop, "crucible_home", lambda: tmp_path)
    called = []
    monkeypatch.setattr(desktop, "_run_tray", called.append)
    desktop.tray()
    assert called == [tmp_path]


def test_tray_does_not_spawn_over_http_error(monkeypatch, tmp_path):
    from urllib.error import HTTPError
    monkeypatch.setitem(sys.modules, "pystray", SimpleNamespace())
    monkeypatch.setattr(local.sys, "platform", "win32")
    def incompatible(*a, **kw):
        raise HTTPError("http://127.0.0.1:7101/v1/ping", 404, "missing", {}, None)
    monkeypatch.setattr(local, "request", incompatible)
    monkeypatch.setattr(local, "_spawn_controller", lambda home: pytest.fail("must not spawn at an occupied port"))
    with pytest.raises(local.LocalError, match="wrong_controller"):
        desktop._run_tray(tmp_path)


@pytest.mark.parametrize("failure,expected", [(ConnectionRefusedError(), None), (TimeoutError(), "shutdown_unknown")])
def test_shutdown_distinguishes_absence_from_uncertainty(monkeypatch, tmp_path, failure, expected):
    monkeypatch.setattr(local.sys, "platform", "win32")
    monkeypatch.setattr(local, "crucible_home", lambda: tmp_path)
    monkeypatch.setattr(desktop, "close_tray", lambda: None)
    def offline(*a, **kw):
        raise URLError(failure)
    monkeypatch.setattr(local, "request", offline)
    if expected:
        with pytest.raises(local.LocalError, match=expected):
            local.shutdown()
    else:
        local.shutdown()


def test_tray_start_failure_releases_pid_and_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(desktop, "crucible_home", lambda: tmp_path)
    def fail(home):
        (home / "tray.pid").write_text("123")
        raise local.LocalError("startup failed")
    monkeypatch.setattr(desktop, "_run_tray", fail)
    with pytest.raises(local.LocalError, match="startup failed"):
        desktop.tray()
    assert not (tmp_path / "tray.pid").exists()
    guard = ProcessLock(tmp_path / "tray.lock")
    assert guard.acquire()
    guard.close()


def test_sharing_menu_error_survives_health_refresh(monkeypatch, tmp_path):
    from crucible.host import tray
    monkeypatch.setattr(desktop.sys, "platform", "linux")
    monkeypatch.setattr(tray, "icon_image", lambda color: None)
    monkeypatch.setattr(sharing, "read", lambda home: None)
    monkeypatch.setattr(sharing, "Engine", lambda home: object())
    def fail(*a, **kw):
        raise sharing.SharingError("sharing_unowned: matching forward already exists")
    monkeypatch.setattr(sharing, "enable", fail)
    monkeypatch.setattr(local, "status", lambda home: {"state": "running", "detail": "healthy"})
    watchers = []
    class Thread:
        def __init__(self, target, **kw): self.target = target
        def start(self):
            if self.target.__name__ == "watch": watchers.append(self.target)
            else: self.target()
    real_event = threading.Event
    class Event(real_event):
        def wait(self, timeout=None): self.set(); return True
    monkeypatch.setattr(desktop.threading, "Thread", Thread)
    monkeypatch.setattr(desktop.threading, "Event", Event)
    class Item:
        def __init__(self, text, action, **kw): self.text, self.action = text, action
    class Icon:
        def __init__(self, *a): pass
        def update_menu(self): pass
        def stop(self): pass
        def run(self):
            next(item for item in self.menu if item.text == "Enable Tailscale sharing").action()
            assert any(item.text == "Use existing Tailscale sharing" for item in self.menu)
            message = self.menu[0].text
            assert "matching Tailscale forward" in message
            watchers[0]()
            assert self.menu[0].text == message
            assert self.title == "Crucible — running"
    monkeypatch.setitem(sys.modules, "pystray", SimpleNamespace(Icon=Icon, Menu=lambda *a: a, MenuItem=Item))
    desktop._run_tray(tmp_path)


def test_a_controller_without_quit_still_shuts_down(monkeypatch, tmp_path):
    """0.6.0 was assumed to serve /quit. The one on owens-pc answered 404.

    Measured 2026-09-16: the legacy path POSTs /quit on the strength of a
    comment saying that release has it, and the installed 0.6.0 did not — which
    is what stopped the upgrade, with an unhandled HTTP error naming nothing.
    close_tray has already asked the process to go; whether it went is decided
    by the wait, not by an assumption about an old version.
    """
    from urllib.error import HTTPError

    from crucible.host import app

    monkeypatch.setattr(local.sys, "platform", "win32")
    monkeypatch.setattr(local, "crucible_home", lambda: tmp_path)
    monkeypatch.setattr(desktop, "close_tray", lambda: None)
    monkeypatch.setattr(local, "connection", lambda home: ("http://127.0.0.1:7100", "test", "secret"))
    (tmp_path / "host.pid").write_text("12345")
    gone = []

    def request(url, **kw):
        if url.endswith("/v1/info"):
            return {"role": "orchestrator", "server": {"version": "0.6.0", "api_version": 1},
                    "engine": {"owner": "wsl-unit"}, "local_lifecycle_version": None}
        if url.endswith("/quit"):
            gone.append(True)   # the tray signal is what actually stops it
            raise HTTPError(url, 404, "Not Found", {}, None)
        if gone:
            raise URLError(ConnectionRefusedError())
        return {"crucible": True, "role": "orchestrator"}

    monkeypatch.setattr(local, "request", request)
    monkeypatch.setattr(app, "_alive", lambda pid: not gone)
    monkeypatch.setattr(local, "act", lambda action: None)

    local.shutdown()          # a 404 must not be the end of the upgrade
    assert gone == [True]


@pytest.mark.parametrize("backend,raises", [
    ("cuda-linux", False),      # a guest unit: outliving its controller is normal
    ("mlx-darwin", False),
    ("llama-windows", True),    # the controller's OWN child, still answering
    (None, True),               # could not be asked: keep the refusal
])
def test_an_engine_outliving_its_controller_is_only_a_fault_when_native(
    monkeypatch, tmp_path, backend, raises
):
    """A WSL unit is MEANT to survive the Windows swap; a native child is not.

    Measured 2026-09-16: this refusal stopped an upgrade on a machine whose
    engine was a healthy WSL guest, because the check asked whether an engine
    was answering and not whether it was one this controller should have
    stopped.
    """
    from crucible.host import app

    monkeypatch.setattr(local.sys, "platform", "win32")
    monkeypatch.setattr(local, "crucible_home", lambda: tmp_path)
    monkeypatch.setattr(desktop, "close_tray", lambda: None)
    monkeypatch.setattr(local, "connection", lambda home: ("http://127.0.0.1:7100", "test", "secret"))
    monkeypatch.setattr(app, "_alive", lambda pid: False)

    def request(url, **kw):
        if ":7101/" in url:
            raise URLError(ConnectionRefusedError())   # no controller at all
        if url.endswith("/v1/ping"):
            return {"crucible": True}
        if url.endswith("/v1/info"):
            if backend is None:
                raise URLError(ConnectionRefusedError())
            return {"host": {"backend": backend}}
        raise AssertionError(url)

    monkeypatch.setattr(local, "request", request)
    if raises:
        with pytest.raises(local.LocalError, match="engine_unmanaged"):
            local.shutdown()
    else:
        local.shutdown()


@pytest.mark.parametrize("gone_as", [
    ConnectionRefusedError(),                 # the port stopped accepting
    ConnectionResetError(),                   # the held socket was torn down
])
def test_a_controller_that_quit_is_gone_however_the_socket_ended(
    monkeypatch, tmp_path, gone_as
):
    """A refusal and a reset both mean the controller went. Only one was read.

    Measured 2026-09-16 on the 0.6.3 -> 0.6.5 upgrade: the orchestrator was
    asked to quit, DID quit, and tore down the socket it was holding as it
    exited. That is WinError 10054, which was not in the list, so a clean
    shutdown was reported as `controller_shutdown_unknown` and the upgrade
    stopped on a machine with nothing wrong with it.
    """
    from crucible.host import app

    monkeypatch.setattr(local.sys, "platform", "win32")
    monkeypatch.setattr(local, "crucible_home", lambda: tmp_path)
    monkeypatch.setattr(desktop, "close_tray", lambda: None)
    monkeypatch.setattr(local, "connection", lambda home: ("http://127.0.0.1:7100", "test", "secret"))
    (tmp_path / "host.pid").write_text("12345")
    stopped = []

    def request(url, **kw):
        if url.endswith("/v1/info"):
            return {"role": "orchestrator", "server": {"version": "0.6.99", "api_version": 1},
                    "engine": {"owner": "wsl-unit"}, "local_lifecycle_version": 1}
        if url.endswith("/quit"):
            stopped.append(True)
            return {"quit": True}
        if stopped:
            raise URLError(gone_as)
        return {"crucible": True, "role": "orchestrator"}

    monkeypatch.setattr(local, "request", request)
    monkeypatch.setattr(app, "_alive", lambda pid: not stopped)
    monkeypatch.setattr(local, "act", lambda action: None)

    local.shutdown()
    assert stopped == [True]
