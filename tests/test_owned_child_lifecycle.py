"""Real CPU subprocesses prove the private pipe runs ASGI/worker cleanup."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from crucible.host.runner import ControlledChild, ProcessRunner


def wait_file(path: Path, process: subprocess.Popen, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert process.poll() is None, "test engine exited before its expected lifecycle marker"
        if time.monotonic() >= deadline:
            raise AssertionError(f"missing lifecycle marker: {path.name}")
        time.sleep(0.02)


def fixture_script(tmp_path: Path) -> Path:
    script = tmp_path / "engine.py"
    script.write_text("""
import os, subprocess, sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
sys.path.insert(0, REPO_PATH)
from crucible import cli
root = Path(sys.argv[1])
@asynccontextmanager
async def lifespan(app):
    worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    (root / 'worker.pid').write_text(str(worker.pid))
    (root / 'ready').write_text('ready')
    try:
        yield
    finally:
        worker.terminate()
        worker.wait(timeout=5)
        (root / 'cleaned').write_text('worker exited')
app = FastAPI(lifespan=lifespan)
from types import SimpleNamespace
cli.load_config = lambda: SimpleNamespace(backend_kind='fixture', host='127.0.0.1', port=0, name='fixture')
cli.detect_backend = lambda: SimpleNamespace(kind='fixture', gpu=SimpleNamespace(name='CPU fixture'))
cli._sync_pairing_file = lambda config: None
import crucible.api
crucible.api.create_app = lambda config, backend: app
raise SystemExit(cli.cmd_serve(SimpleNamespace(host=None, port=None, log_level='error', controller_stdin=True)))
""".replace("REPO_PATH", repr(str(Path(__file__).resolve().parents[1]))), encoding="utf-8")
    return script


@pytest.mark.parametrize("before_ready", [False, True])
def test_owned_engine_stops_its_worker_before_exiting(tmp_path, before_ready):
    script = fixture_script(tmp_path)
    process = subprocess.Popen([sys.executable, str(script), str(tmp_path)], stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    child = ControlledChild(process)
    try:
        if not before_ready:
            wait_file(tmp_path / "ready", process)
        child.terminate()
        assert child.wait(10) == 0, process.stderr.read().decode()
        assert (tmp_path / "cleaned").read_text() == "worker exited"
    finally:
        child.terminate()
        if process.poll() is None:
            process.kill()
            process.wait(5)


def test_controller_death_closes_the_pipe_and_cleans_engine_worker(tmp_path):
    engine = fixture_script(tmp_path)
    controller_code = """
import subprocess,sys,time
from pathlib import Path
engine=subprocess.Popen([sys.executable,sys.argv[1],sys.argv[2]],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
Path(sys.argv[2],'engine.pid').write_text(str(engine.pid))
time.sleep(60)
"""
    controller = subprocess.Popen([sys.executable, "-c", controller_code, str(engine), str(tmp_path)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_file(tmp_path / "ready", controller)
        controller.terminate()
        controller.wait(5)
        deadline = time.monotonic() + 10
        while not (tmp_path / "cleaned").exists():
            if time.monotonic() > deadline:
                raise AssertionError("engine did not run worker cleanup after controller death")
            time.sleep(0.02)
    finally:
        if controller.poll() is None:
            controller.terminate()
            controller.wait(5)
        if not (tmp_path / "cleaned").exists():
            # Only these test-owned fixture pids, never a process-name sweep.
            for name in ("worker.pid", "engine.pid"):
                if (tmp_path / name).exists():
                    try:
                        import signal
                        os.kill(int((tmp_path / name).read_text()), signal.SIGTERM)
                    except OSError:
                        pass


def test_owned_pipe_is_opt_in_and_argv_runs_python_directly(monkeypatch):
    from crucible.host.app import server_argv
    seen = []
    class Pipe:
        closed = False
        def close(self): self.closed = True
    pipe = Pipe()
    process = SimpleNamespace(stdin=pipe, pid=12, poll=lambda: None, wait=lambda **kw: 0)
    def popen(argv, **kwargs):
        seen.append((argv, kwargs))
        return process
    monkeypatch.setattr(subprocess, "Popen", popen)
    argv = server_argv({"LOCALAPPDATA": r"C:\Users\fixture\AppData\Local"})
    assert argv[0].endswith(r"host\python.exe")
    assert argv[1:] == ["-m", "crucible.cli", "serve", "--controller-stdin"]
    runner = ProcessRunner(sys.platform, os.environ)
    child = runner.spawn(argv)
    assert isinstance(child, ControlledChild)
    assert seen[-1][1]["stdin"] == subprocess.PIPE
    child.terminate()
    assert pipe.closed
    runner.spawn([sys.executable, "-c", "print('unmanaged')"])
    assert seen[-1][1]["stdin"] == subprocess.DEVNULL


def test_control_port_bind_failure_cleans_owned_child_and_fails(tmp_path, monkeypatch):
    from crucible.host import app
    from crucible.host.errors import HostError
    from crucible.host.menu import Distro, Engine, Owner
    from crucible.host.presence import Presence
    events = []
    monkeypatch.setenv("CRUCIBLE_HOME", str(tmp_path))
    monkeypatch.setattr(app, "log_path", lambda env: tmp_path / "host.log")
    monkeypatch.setattr(app, "previous_log_path", lambda env: tmp_path / "host.old.log")
    monkeypatch.setattr(app, "acquire", lambda home: None)
    monkeypatch.setattr(app.startup, "install", lambda runner: SimpleNamespace(detail="fixture"))
    monkeypatch.setattr(app, "consented_distro", lambda home: None)
    watcher = SimpleNamespace(release=lambda: events.append("release"), stop_child=lambda: events.append("child stopped"))
    monkeypatch.setattr(app, "PresenceWatcher", lambda *args, **kwargs: watcher)
    def start(host):
        host._c.presence = Presence(Distro.ABSENT, Engine.RUNNING, "fixture", Owner.HOST_CHILD)
    monkeypatch.setattr(app.Host, "start", start)
    monkeypatch.setattr(app.Host, "claim", lambda self: False)
    monkeypatch.setattr(app, "_write_pairing", lambda context: None)
    def occupied(*args, **kwargs): raise OSError("port occupied")
    monkeypatch.setattr(app, "serve", occupied)
    with pytest.raises(HostError, match="host_door_unavailable"):
        app.run(headless=True)
    assert events == ["release", "child stopped"]


def test_stop_timeout_preserves_the_child_handle_for_retry(tmp_path):
    from crucible.host.presence import PresenceWatcher
    from crucible.host.log import HostLog
    class StubbornChild:
        pid = 123
        attempts = 0
        def poll(self): return None
        def terminate(self): self.attempts += 1
        def wait(self, timeout): raise subprocess.TimeoutExpired("owned fixture", timeout)
    watcher = PresenceWatcher(SimpleNamespace(), HostLog(tmp_path / "host.log", tmp_path / "old.log"))
    child = StubbornChild()
    watcher.child = child
    with pytest.raises(subprocess.TimeoutExpired):
        watcher.stop_child(timeout_s=0.01)
    assert watcher.child is child
    assert child.attempts == 1
