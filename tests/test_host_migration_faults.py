"""CPU-only migration fault injection: temporary weights and loopback peers."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from crucible.host import app, installer, presence
from crucible.host.catalog import HttpCatalog, StoppedWindowsCatalog
from crucible.host.errors import HostError
from crucible.host.menu import Distro, Engine, Owner
from tests.test_host import FakeCatalog, Scripted, _context, migration


@contextmanager
def guest_http(*, reject_info=False):
    """A real authenticated transport, with no subprocesses or inference."""
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            calls.append((self.command, self.path, self.headers.get('Authorization')))
            status = 200
            if self.path == '/v1/ping':
                body = {'crucible': True, 'name': 'guest'}
            elif self.headers.get('Authorization') != 'Bearer token' or reject_info:
                status, body = 401, {'error': {'code': 'unauthorized', 'message': 'fixture rejection'}}
            elif self.path == '/v1/info':
                body = {'server': {'name': 'guest', 'api_version': 1}, 'host': {'backend': 'cuda-linux'}}
            elif self.path == '/v1/catalog':
                body = {'subjects': [{'kind': 'model', 'id': 'a', 'installed': True}]}
            else:
                status, body = 404, {}
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_DELETE(self):
            calls.append((self.command, self.path, self.headers.get('Authorization')))
            self.send_error(500, 'The guest must never receive native deletion')

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def move_host(tmp_path, monkeypatch, base):
    context = _context(tmp_path, Scripted())
    context.presence = presence.Presence(Distro.ABSENT, Engine.STOPPED, 'native stopped for move', Owner.HOST_CHILD)
    old = context.watcher
    guest = SimpleNamespace(
        boot=lambda: presence.Presence(Distro.PRESENT, Engine.RUNNING, 'guest ready', Owner.WSL_UNIT),
        read_guest_pairing=lambda distro: 'crucible://guest@127.0.0.1:7100/#token',
        _distro='crucible', hold=lambda distro: None, release=lambda: None,
    )
    monkeypatch.setattr(app, 'PresenceWatcher', lambda *args, **kwargs: guest)
    monkeypatch.setattr(app, 'engine_url', lambda path='': base + path)
    (tmp_path / 'pairing').write_text('original pairing', encoding='utf-8')
    installer.record_cleanup(tmp_path, {('model', 'a')})
    host = app.Host(context)
    monkeypatch.setattr(host, 'claim', lambda: True)
    return host, old


def test_guest_http_auth_failure_cannot_publish_owner_or_enable_cleanup(tmp_path, monkeypatch):
    with guest_http(reject_info=True) as (base, calls):
        host, old = move_host(tmp_path, monkeypatch, base)
        with pytest.raises(HTTPError) as caught:
            host.finish_wsl_move()
        assert caught.value.code == 401
        assert host._c.watcher is old
        assert host._c.presence.owner is Owner.HOST_CHILD
        assert (tmp_path / 'pairing').read_text() == 'original pairing'
        assert installer.cleanup_subjects(tmp_path) == {('model', 'a')}
        with pytest.raises(HostError, match='Windows models are kept'):
            host.stopped_windows_catalog()
        assert not any(method == 'DELETE' for method, _, _ in calls)


@pytest.mark.parametrize('fault', ['pairing', 'claim', 'hold'])
def test_activation_failure_does_not_commit_wsl_ownership(tmp_path, monkeypatch, fault):
    with guest_http() as (base, calls):
        host, old = move_host(tmp_path, monkeypatch, base)
        if fault == 'pairing':
            monkeypatch.setattr(app, '_write_pairing', lambda context: None)
        else:
            monkeypatch.setattr(app, '_write_pairing', lambda context: (tmp_path / 'pairing').write_text('crucible://guest@127.0.0.1:7100/#token'))
        if fault == 'claim':
            monkeypatch.setattr(host, 'claim', lambda: False)
        if fault == 'hold':
            def failed_hold():
                raise OSError('fixture hold could not start')
            monkeypatch.setattr(host, '_hold', failed_hold)
        with pytest.raises((HostError, OSError)):
            host.finish_wsl_move()
        assert host._c.watcher is old
        assert host._c.presence.owner is Owner.HOST_CHILD
        assert installer.cleanup_subjects(tmp_path) == {('model', 'a')}
        with pytest.raises(HostError, match='Windows models are kept'):
            host.stopped_windows_catalog()
        assert not any(method == 'DELETE' for method, _, _ in calls)


def test_native_stop_timeout_keeps_owned_child_and_prevents_switch(tmp_path, monkeypatch):
    context = _context(tmp_path, Scripted())
    context.presence = presence.Presence(Distro.ABSENT, Engine.RUNNING, 'native', Owner.HOST_CHILD)
    child = SimpleNamespace(poll=lambda: None, terminate=lambda: None)
    def timed_out(timeout):
        raise subprocess.TimeoutExpired('fixture native engine', timeout)
    child.wait = timed_out
    context.watcher.child = child
    host = app.Host(context)
    windows = FakeCatalog('native', [('model', 'a'), ('model', 'b')])
    guest = FakeCatalog('guest', [('model', 'a'), ('model', 'b')])
    walk = migration(windows, guest, [], tmp_path)
    for method in ('_wsl_state', '_import_distro', '_guest_ready', '_guest_install', '_migrate_config', '_install_job_types', '_lan_door'):
        monkeypatch.setattr(walk, method, lambda: None)
    walk._stop_windows_callback = host.stop_windows_for_move
    walk._switch_pairing_callback = lambda: pytest.fail('A live native child prohibits the switch')
    walk._windows_after_switch = lambda: pytest.fail('No cleanup before activation')
    with pytest.raises(subprocess.TimeoutExpired):
        walk.run()
    assert context.watcher.child is child
    assert context.presence.owner is Owner.HOST_CHILD
    assert windows._keys() == {('model', 'a'), ('model', 'b')}
    assert installer.cleanup_subjects(tmp_path) == windows._keys()


def test_restart_journal_retires_missing_stamp_via_owner_not_guest_http(tmp_path, monkeypatch):
    from crucible import catalog, weights
    root = tmp_path / 'models' / 'a' / 'llama-windows'
    root.mkdir(parents=True)
    residue = root / 'remaining.gguf'
    residue.write_bytes(b'fixture native weights after stamp was deleted')
    config = SimpleNamespace(backend_kind='llama-windows', home=tmp_path)
    backend = SimpleNamespace(kind='llama-windows')
    manifest = SimpleNamespace(weights_family='models', id='a')
    spec = SimpleNamespace(backend='llama-windows')
    row = SimpleNamespace(kind='model', id='a', name='a', installed=lambda: None,
                          remove=lambda: weights.remove(config, manifest, spec))
    monkeypatch.setattr(catalog, 'subjects', lambda *_: [row])
    installer.record_cleanup(tmp_path, {('model', 'a')})
    with guest_http() as (base, calls):
        stopped = StoppedWindowsCatalog(config, backend, installer.cleanup_subjects(tmp_path))
        walk = migration(stopped, HttpCatalog(base, 'token', where='guest fixture'), [], tmp_path)
        walk._migrate_weights(allow_pull=False)
        assert not residue.exists()
        # Simulate restart before the controller removed the journal. The named
        # owner operation is idempotent even with the whole native tree gone.
        restarted = StoppedWindowsCatalog(config, backend, installer.cleanup_subjects(tmp_path))
        migration(restarted, HttpCatalog(base, 'token', where='guest fixture'), [], tmp_path)._migrate_weights(allow_pull=False)
        assert all(method == 'GET' for method, _, _ in calls)
        assert all(auth == 'Bearer token' for _, _, auth in calls)
