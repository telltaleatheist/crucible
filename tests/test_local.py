"""Installed lifecycle: test observable failures, not just launch arguments."""
import json
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import URLError

import pytest

from crucible import local


def test_record_is_atomic_and_contains_no_credentials(tmp_path):
    path = local.publish_installation(tmp_path)
    value = json.loads(path.read_text())
    assert value["schema_version"] == 1
    assert Path(value["control"]["command"]).is_file()
    assert value["control"]["args"] == ["-m", "crucible.cli", "local"]
    assert "token" not in value
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("ping,state", [
    ({"hello": "world"}, "wrong_service"),
    ({"crucible": True, "name": "another"}, "wrong_service"),
    ({"crucible": True, "name": "expected"}, "running"),
])
def test_status_checks_engine_identity_and_authenticated_info(monkeypatch, tmp_path, ping, state):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = ping if self.path.endswith("ping") else {"server": {"name": "expected", "api_version": 1}}
            if self.path.endswith("info"):
                assert self.headers["Authorization"] == "Bearer test-token"
                assert self.headers["X-Crucible-Api"] == "1"
            raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(local, "connection", lambda _: (f"http://127.0.0.1:{server.server_port}", "expected", "test-token"))
    try:
        assert local.status(tmp_path)["state"] == state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_timeout_is_not_reported_as_stopped(monkeypatch, tmp_path):
    monkeypatch.setattr(local.sys, "platform", "win32")
    monkeypatch.setattr(local, "connection", lambda _: ("http://127.0.0.1:7100", "test", "token"))
    def unreachable(*args, **kwargs): raise URLError("timed out")
    monkeypatch.setattr(local, "request", unreachable)
    assert local.status(tmp_path)["state"] == "unreachable"


def test_windows_stopped_intent_survives_controller_restart(tmp_path, monkeypatch):
    from crucible.host.app import Host, HostContext, engine_token
    from crucible.host.log import HostLog
    from crucible.host.presence import Presence, UnitProbe, SCOPE_USER
    from crucible.host.menu import Distro, Engine, Owner
    from types import SimpleNamespace
    (tmp_path / "engine.stopped").write_text("stopped")
    (tmp_path / "pairing").write_text("crucible://engine@127.0.0.1:7100/#secret")
    context = HostContext(runner=SimpleNamespace(), log=HostLog(tmp_path / "log", tmp_path / "old"),
                          home=tmp_path, watcher=SimpleNamespace(watch_s=0.001),
                          presence=Presence(Distro.UNKNOWN, Engine.STOPPED, "stopped", Owner.NONE), release="test")
    host = Host(context)
    assert host.local_status()["state"] == "stopped"
    assert engine_token(context) == "secret"
    # The watch must not call any recovery method on this deliberately tiny watcher.
    thread = threading.Thread(target=host.watch)
    thread.start()
    host._stop.set()
    thread.join(timeout=1)
    assert not thread.is_alive()


def test_windows_failed_stop_does_not_publish_stopped(tmp_path):
    from crucible.host.app import Host, HostContext
    from crucible.host.log import HostLog
    from crucible.host.presence import Presence, UnitProbe, SCOPE_USER
    from crucible.host.menu import Distro, Engine, Owner
    from crucible.host.runner import RunResult
    from crucible.host.errors import HostError
    from types import SimpleNamespace
    runner = SimpleNamespace(run=lambda *args, **kwargs: RunResult(code=1, stdout="", stderr="denied", failure=None))
    context = HostContext(runner=runner, log=HostLog(tmp_path / "log", tmp_path / "old"), home=tmp_path,
        # A guest whose unit is a USER one, which is the path this test is
        # about: `probe_unit` is what the stop now asks FIRST to decide
        # which manager to speak to.
        watcher=SimpleNamespace(guest_uid=lambda: "1000", _distro="crucible",
            probe_unit=lambda: UnitProbe(True, "enabled", "user unit",
                SCOPE_USER)),
        presence=Presence(Distro.PRESENT, Engine.RUNNING, "running", Owner.WSL_UNIT), release="test")
    host = Host(context)
    with pytest.raises(HostError, match="denied"):
        host.local_stop()
    assert not (tmp_path / "engine.stopped").exists()
    assert host.local_status()["state"] == "running"


def test_controller_exit_waits_for_child_shutdown(tmp_path):
    from crucible.host.app import Host, HostContext
    from crucible.host.log import HostLog
    from crucible.host.presence import Presence, UnitProbe, SCOPE_USER
    from crucible.host.menu import Distro, Engine, Owner
    from types import SimpleNamespace
    entered, finish = threading.Event(), threading.Event()
    def stop_child():
        entered.set()
        assert finish.wait(3)
    context = HostContext(runner=SimpleNamespace(), log=HostLog(tmp_path / "log", tmp_path / "old"),
        home=tmp_path, watcher=SimpleNamespace(release=lambda: None, stop_child=stop_child),
        presence=Presence(Distro.ABSENT, Engine.RUNNING, "native", Owner.HOST_CHILD), release="test")
    host = Host(context)
    thread = threading.Thread(target=host.quit)
    thread.start()
    try:
        assert entered.wait(3)
        assert host._stop.is_set()
        assert not host._shutdown_complete.is_set()
    finally:
        finish.set()
        thread.join(3)
    assert not thread.is_alive()
    assert host._shutdown_complete.is_set()


def test_http_error_is_not_a_missing_process(monkeypatch, tmp_path):
    from urllib.error import HTTPError
    monkeypatch.setattr(local, "connection", lambda _: ("http://127.0.0.1:7100", "test", "token"))
    def wrong(*args, **kwargs): raise HTTPError("http://127.0.0.1:7100/v1/ping", 404, "missing", {}, None)
    monkeypatch.setattr(local, "request", wrong)
    assert local.status(tmp_path)["state"] == "wrong_service"


def test_an_empty_info_document_is_not_healthy(monkeypatch, tmp_path):
    monkeypatch.setattr(local, "connection", lambda _: ("http://127.0.0.1:7100", "test", "token"))
    monkeypatch.setattr(local, "request", lambda url, **kw: {"crucible": True, "name": "test"} if url.endswith("ping") else {})
    assert local.status(tmp_path)["state"] == "wrong_service"


def test_watch_failure_is_not_an_operator_stop(monkeypatch, tmp_path):
    monkeypatch.setattr(local.sys, "platform", "win32")
    monkeypatch.setattr(local, "connection", lambda _: ("http://127.0.0.1:7100", "test", "token"))
    def request(url, **kwargs):
        if url.endswith("/local/status"): return {"state": "stopped", "intentional": False}
        raise URLError("timed out")
    monkeypatch.setattr(local, "request", request)
    assert local.status(tmp_path)["state"] == "unreachable"


@pytest.mark.parametrize("fault", [None, "ping", "empty-info", "info-name", "api-version", "native-backend"])
def test_wsl_move_publishes_only_the_authenticated_guest(tmp_path, monkeypatch, fault):
    from types import SimpleNamespace
    from crucible.host import app
    from crucible.host.log import HostLog
    from crucible.host.presence import Presence, UnitProbe, SCOPE_USER
    from crucible.host.menu import Distro, Engine, Owner
    from crucible.host.errors import HostError
    line = "crucible://guest@127.0.0.1:7100/#guest-token"
    pairing = tmp_path / "pairing"
    pairing.write_text("old pairing")
    events = []
    old = SimpleNamespace(release=lambda: events.append("old hold released"))
    guest = SimpleNamespace(boot=lambda: Presence(Distro.PRESENT, Engine.RUNNING, "guest up", Owner.WSL_UNIT),
                            read_guest_pairing=lambda distro: line,
                            _distro="crucible", hold=lambda distro: events.append("guest held"))
    context = app.HostContext(runner=SimpleNamespace(), log=HostLog(tmp_path / "log", tmp_path / "old"),
        home=tmp_path, watcher=old,
        presence=Presence(Distro.ABSENT, Engine.RUNNING, "native", Owner.HOST_CHILD), release="test")
    host = app.Host(context)
    monkeypatch.setattr(app, "PresenceWatcher", lambda *a, **kw: guest)
    monkeypatch.setattr(app, "_write_pairing", lambda c: pairing.write_text(line))
    monkeypatch.setattr(host, "claim", lambda: events.append("guest claimed") or True)
    def request(url, **kwargs):
        if url.endswith("ping"):
            return {"crucible": True, "name": "wrong" if fault == "ping" else "guest"}
        assert kwargs["token"] == "guest-token"
        events.append("authenticated")
        if fault == "empty-info":
            return {}
        return {"server": {"name": "wrong" if fault == "info-name" else "guest",
                           "api_version": 2 if fault == "api-version" else 1},
                "host": {"backend": "llama-windows" if fault == "native-backend" else "cuda-linux"}}
    monkeypatch.setattr(local, "request", request)
    if fault is not None:
        with pytest.raises(HostError, match="not reaching"):
            host.finish_wsl_move()
        assert pairing.read_text() == "old pairing"
        assert context.watcher is old
        assert "guest claimed" not in events
    else:
        host.finish_wsl_move()
        assert context.presence.owner is Owner.WSL_UNIT
        assert pairing.read_text() == line
        assert events == ["old hold released", "authenticated", "guest held", "guest claimed"]


@pytest.mark.parametrize("already_emitted", [False, True])
def test_controller_failure_always_has_one_terminal_event(tmp_path, already_emitted):
    from types import SimpleNamespace
    from urllib.request import Request, urlopen
    from crucible.host.door import OrchestratorDoor, serve, INSTALL_PATH
    from crucible.host.log import HostLog
    from crucible.host.errors import HostError
    from crucible.host.installer import Event
    def failing(emit):
        if already_emitted:
            emit(Event("failed", {"code": "engine_move_failed", "message": "guest is wrong"}))
        raise HostError("engine_move_failed", "guest is wrong")
    door = OrchestratorDoor(HostLog(tmp_path / "log", tmp_path / "old"), failing,
                            token=lambda: "test", orchestrator=SimpleNamespace(name="test"))
    server = serve(door, port=0)
    try:
        request = Request(f"http://127.0.0.1:{server.server_port}{INSTALL_PATH}",
            data=b'{"target":"wsl"}', headers={"Authorization": "Bearer test"}, method="POST")
        with urlopen(request, timeout=3) as response:
            events = [json.loads(line) for line in response]
        assert len(events) == 1
        assert events[0]["event"] == "failed"
        assert events[0]["data"]["code"] == "engine_move_failed"
    finally:
        server.shutdown()
        server.server_close()


def _engine_serving(monkeypatch, tmp_path, version):
    """A fake paired engine that reports `version` from /v1/info."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.endswith("ping"):
                body = {"crucible": True, "name": "expected"}
            else:
                server_block = {"name": "expected", "api_version": 1}
                if version is not None:
                    server_block["version"] = version
                body = {"server": server_block}
            raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(
        local, "connection",
        lambda _: (f"http://127.0.0.1:{server.server_port}", "expected", "test-token"),
    )
    return server, thread


def test_the_observation_carries_the_engines_own_version(monkeypatch, tmp_path):
    server, thread = _engine_serving(monkeypatch, tmp_path, local.VERSION)
    try:
        observed = local.status(tmp_path)
        assert observed["state"] == "running"
        assert observed["version"] == local.VERSION
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_an_engine_of_another_version_is_not_a_successful_start(monkeypatch, tmp_path):
    """The stale-process defect, measured 2026-09-16.

    `systemctl enable --now` does nothing to an already-running unit, so an
    upgrade that rewrote ExecStart left the OLD executable serving while the
    install reported success end to end. Whatever the route to it, an engine
    answering with a version this installation did not install is not a start
    that worked.
    """
    from types import SimpleNamespace
    from crucible import service as service_module

    server, thread = _engine_serving(monkeypatch, tmp_path, "0.0.1-previous")
    # HERMETIC ON PURPOSE. `act` defaults `home` to the real `crucible_home()`
    # and then drives systemctl for real; an earlier draft of this test did
    # exactly that and passed by touching the live engine. The service call is
    # stubbed and `home` is the tmp dir, so what is under test is the version
    # check and nothing else.
    monkeypatch.setattr(local, "load_config",
                        lambda _h: SimpleNamespace(backend_kind="cuda-linux"))
    monkeypatch.setattr(service_module, "start", lambda *a, **k: None)
    try:
        with pytest.raises(local.LocalError) as caught:
            local.act("start", tmp_path)
        assert "engine_version_stale" in str(caught.value)
        assert "0.0.1-previous" in str(caught.value)
        assert local.VERSION in str(caught.value)
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_an_engine_too_old_to_report_a_version_is_not_called_stale(monkeypatch, tmp_path):
    """Absent is not mismatched; refusing it would invent a fault from a missing key."""
    server, thread = _engine_serving(monkeypatch, tmp_path, None)
    try:
        observed = local.status(tmp_path)
        assert observed["state"] == "running"
        assert observed["version"] is None
    finally:
        server.shutdown(); server.server_close(); thread.join()

def test_a_refusal_behind_an_http_error_still_names_itself():
    """`HTTP Error 409: Conflict` is not a diagnosis; the body was.

    Measured 2026-09-17: an upgrade refused with exactly that line, and the
    reason the door had written into the response - which stop had failed and
    why - was thrown away by the last `str(exc)` on the path.
    """
    import io
    import urllib.error

    body = json.dumps(
        {"error": {"code": "engine_stop_failed", "message": "Unit not loaded"}}
    ).encode("utf-8")
    exc = urllib.error.HTTPError(
        "http://127.0.0.1:7101/local/stop", 409, "Conflict", {}, io.BytesIO(body)
    )
    said = local.said(exc)
    assert "engine_stop_failed" in said
    assert "Unit not loaded" in said
    assert "409" in said, "the status code is still worth keeping"


def test_a_failure_with_no_structured_body_is_reported_as_it_came():
    """No invention. A refusal that said nothing parseable says what it said."""
    import io
    import urllib.error

    exc = urllib.error.HTTPError(
        "http://127.0.0.1:7101/local/stop", 502, "Bad Gateway", {},
        io.BytesIO(b"<html>nginx</html>"),
    )
    assert local.said(exc) == str(exc)
    assert local.said(ValueError("plain")) == "plain"

