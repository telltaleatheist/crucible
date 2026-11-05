from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

from crucible import hosttools
from tests.fake_engine import _LoopbackServer

BODY = b"the tool's bytes"


class _Host(BaseHTTPRequestHandler):
    answers: list[int] = []

    def do_GET(self) -> None:
        code = self.answers.pop(0)
        self.send_response(code)
        body = BODY if code == 200 else b""
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture
def host() -> Iterator[str]:
    server = _LoopbackServer(("127.0.0.1", 0), _Host)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/tool"
    server.shutdown()
    server.server_close()
    thread.join(5)


def test_a_504_from_the_release_host_is_waited_on(
    host: str, tmp_path: Path, real_tool_download: Callable[..., str]
) -> None:
    _Host.answers = [504, 503, 200]
    target = tmp_path / "tool"
    assert real_tool_download(host, target, pause_s=0) == hashlib.sha256(BODY).hexdigest()
    assert target.read_bytes() == BODY
    assert _Host.answers == []


def test_a_404_is_refused_at_once_as_the_tools_own_error(
    host: str, tmp_path: Path, real_tool_download: Callable[..., str]
) -> None:
    _Host.answers = [404, 200]
    with pytest.raises(hosttools.HostToolError) as refused:
        real_tool_download(host, tmp_path / "tool", pause_s=0)
    assert refused.value.code == "tool_download_failed"
    assert "404" in refused.value.message
    assert _Host.answers == [200], "a 404 is not weather: it is not asked twice"
