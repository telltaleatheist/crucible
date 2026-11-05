from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from crucible import interpreter
from crucible.backend import CUDA_LINUX, LLAMA_WINDOWS, MLX_DARWIN
from crucible.interpreter import InterpreterError


def test_every_pin_names_a_version_a_url_and_a_digest() -> None:
    assert interpreter.INTERPRETERS, "a build that pins no interpreter installs nothing"
    for (backend_kind, minor), pin in interpreter.INTERPRETERS.items():
        assert pin.python_version.startswith(minor + "."), (backend_kind, minor)
        assert len(pin.sha256) == 64
        assert pin.release in pin.url and pin.asset in pin.url
        assert "install_only" in pin.asset
        assert pin.url.startswith(
            "https://github.com/astral-sh/python-build-standalone/releases/download/"
        ), pin.url


def test_the_server_runs_311_on_every_backend() -> None:
    assert interpreter.SERVER_PYTHON == "3.11"
    releases = set()
    for backend_kind in (CUDA_LINUX, MLX_DARWIN, LLAMA_WINDOWS):
        pin = interpreter.pin_for(backend_kind, interpreter.SERVER_PYTHON)
        assert pin.python_version.startswith("3.11.")
        releases.add(pin.release)
    assert len(releases) == 1, releases
    assert "x86_64-unknown-linux-gnu" in interpreter.pin_for(CUDA_LINUX, "3.11").asset
    assert "aarch64-apple-darwin" in interpreter.pin_for(MLX_DARWIN, "3.11").asset
    assert "x86_64-pc-windows-msvc" in interpreter.pin_for(LLAMA_WINDOWS, "3.11").asset


def test_the_windows_pin_is_the_asset_that_exists_and_carries_no_shared_infix() -> None:
    pin = interpreter.pin_for(LLAMA_WINDOWS, "3.11")
    assert pin.asset == (
        "cpython-3.11.16+20260901-x86_64-pc-windows-msvc-install_only.tar.gz"
    )
    assert pin.sha256 == (
        "6be524fa6752af802146a4adc7d098565425b0b1c166e19a5a7a4c8cccb86bf6"
    )
    assert "-shared" not in pin.asset


def test_the_higgs_recipes_312_is_pinned_and_digested() -> None:
    pin = interpreter.pin_for(CUDA_LINUX, "3.12")
    assert pin.python_version == "3.12.14"
    assert pin.asset == (
        "cpython-3.12.14+20260901-x86_64-unknown-linux-gnu-install_only.tar.gz"
    )
    assert pin.sha256 == (
        "936c246dfdbbfa7cb22dd01814a21f582a892689fae96b06071a5e433baffa22"
    )


def test_a_version_nobody_pinned_is_refused_by_name() -> None:
    with pytest.raises(InterpreterError) as caught:
        interpreter.pin_for(CUDA_LINUX, "3.99")
    assert caught.value.code == "interpreter_not_pinned"
    assert "3.99" in str(caught.value)
    with pytest.raises(InterpreterError) as caught:
        interpreter.pin_for("rocm-linux", "3.11")
    assert caught.value.code == "interpreter_not_pinned"


def test_the_interpreter_layout_is_asked_for_and_never_spelled() -> None:
    root = Path("/p")
    assert interpreter.interpreter_python(root, CUDA_LINUX) == root / "bin" / "python"
    assert interpreter.interpreter_python(root, MLX_DARWIN) == root / "bin" / "python"
    assert interpreter.interpreter_python(root, LLAMA_WINDOWS) == root / "python.exe"
    with pytest.raises(InterpreterError) as caught:
        interpreter.interpreter_python(root, "rocm-linux")
    assert caught.value.code == "interpreter_not_pinned"


def _install_only_tarball(destination: Path, *, marker: str = "print('hi')\n") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        data = marker.encode()
        info = tarfile.TarInfo("python/bin/python")
        info.size = len(data)
        info.mode = 0o755
        archive.addfile(info, io.BytesIO(data))
    blob = buffer.getvalue()
    destination.write_bytes(blob)
    return blob


def _pin(sha: str) -> interpreter.StandalonePython:
    return interpreter.StandalonePython(
        python_version="3.12.14",
        release="20260901",
        asset="cpython-3.12.14+20260901-x86_64-unknown-linux-gnu-install_only.tar.gz",
        sha256=sha,
    )


def test_a_download_verifies_unpacks_and_stamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "src.tar.gz"
    blob = _install_only_tarball(source)
    pin = _pin(hashlib.sha256(blob).hexdigest())
    monkeypatch.setattr(
        interpreter, "fetch", lambda url, path, **kw: path.write_bytes(blob)
    )
    home = tmp_path / "home"
    python = interpreter.ensure_interpreter(home, CUDA_LINUX, "3.12", pin=pin)
    assert python == home / "interpreters" / "3.12.14" / "bin" / "python"
    assert python.is_file()
    stamp = json.loads(
        (home / "interpreters" / "3.12.14" / interpreter.STAMP_NAME).read_text()
    )
    assert stamp["python_version"] == "3.12.14"
    assert stamp["sha256"] == pin.sha256
    assert stamp["asset"] == pin.asset


def test_a_second_call_with_a_matching_stamp_downloads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "src.tar.gz"
    blob = _install_only_tarball(source)
    pin = _pin(hashlib.sha256(blob).hexdigest())
    calls: list[str] = []

    def fetch(url: str, path: Path, **kw: object) -> None:
        calls.append(url)
        path.write_bytes(blob)

    monkeypatch.setattr(interpreter, "fetch", fetch)
    home = tmp_path / "home"
    interpreter.ensure_interpreter(home, CUDA_LINUX, "3.12", pin=pin)
    interpreter.ensure_interpreter(home, CUDA_LINUX, "3.12", pin=pin)
    assert calls == [pin.url]


def test_a_digest_that_does_not_match_is_refused_by_name_and_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "src.tar.gz"
    blob = _install_only_tarball(source)
    pin = _pin("0" * 64)
    monkeypatch.setattr(
        interpreter, "fetch", lambda url, path, **kw: path.write_bytes(blob)
    )
    home = tmp_path / "home"
    with pytest.raises(InterpreterError) as caught:
        interpreter.ensure_interpreter(home, CUDA_LINUX, "3.12", pin=pin)
    assert caught.value.code == "interpreter_sha_mismatch"
    assert pin.asset in str(caught.value)
    assert not (home / "interpreters" / "3.12.14").exists()


def test_the_progress_line_round_trips() -> None:
    line = interpreter.progress_line(17, 100, "cpython.tar.gz")
    assert interpreter.parse_progress_line(line) == {
        "bytes_done": 17,
        "bytes_total": 100,
        "file": "cpython.tar.gz",
    }
    assert interpreter.parse_progress_line("Collecting torch==2.8.0") is None
    assert interpreter.parse_progress_line(interpreter.PROGRESS_PREFIX + "{oops") is None
    assert (
        interpreter.parse_progress_line(interpreter.PROGRESS_PREFIX + '{"a": 1}') is None
    )


def test_an_unknown_total_stays_null_rather_than_zero() -> None:
    parsed = interpreter.parse_progress_line(interpreter.progress_line(5, None, "f"))
    assert parsed == {"bytes_done": 5, "bytes_total": None, "file": "f"}


# ── fetch: a dropped download is continued, not restarted ─────────────────────

import http.server
import threading
from contextlib import contextmanager
from typing import Iterator


class _Blob(http.server.BaseHTTPRequestHandler):
    """Serves `blob`, honouring `Range: bytes=N-`. Each request's behaviour is the next entry
    of `plan`: ("cut", n) cuts the connection after n body bytes, "whole" ignores the Range
    and sends everything, ("status", code) answers with that code, None sends what was asked."""

    blob = b""
    plan: list = []
    seen: list = []

    def do_GET(self) -> None:  # noqa: N802
        start = 0
        ranged = self.headers.get("Range")
        if ranged is not None:
            start = int(ranged.removeprefix("bytes=").removesuffix("-"))
        self.seen.append(ranged)
        step = self.plan.pop(0) if self.plan else None
        if isinstance(step, tuple) and step[0] == "status":
            self.send_error(step[1])
            return
        if step == "whole":
            start, ranged = 0, None
        body = self.blob[start:]
        self.send_response(206 if ranged is not None else 200)
        self.send_header("Content-Length", str(len(body)))
        if ranged is not None:
            self.send_header("Content-Range", f"bytes {start}-{len(self.blob) - 1}/{len(self.blob)}")
        self.end_headers()
        if isinstance(step, tuple) and step[0] == "cut":
            self.wfile.write(body[: step[1]])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@contextmanager
def _serving(blob: bytes, plan: list) -> Iterator[tuple[str, list]]:
    handler = type("Handler", (_Blob,), {"blob": blob, "plan": list(plan), "seen": []})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/blob", handler.seen
    finally:
        server.shutdown()
        server.server_close()


BLOB = bytes(range(256)) * 4096  # 1 MiB, every offset distinguishable


def test_a_dropped_download_continues_from_the_bytes_it_has(tmp_path: Path) -> None:
    target = tmp_path / "image.tar.gz"
    pauses: list[float] = []
    with _serving(BLOB, [("cut", 300_000), ("cut", 200_000), None]) as (url, seen):
        interpreter.fetch(url, target, chunk=65536, sleep=pauses.append)
    assert target.read_bytes() == BLOB
    assert seen == [None, "bytes=300000-", "bytes=500000-"], "each try asked only for the rest"
    assert pauses == [interpreter.FETCH_PAUSE_S] * 2, "every drop was waited on, not hammered"
    assert not interpreter.partial_path(target).exists()


def test_a_killed_fetch_is_continued_by_the_next_one(tmp_path: Path) -> None:
    target = tmp_path / "wheel.whl"
    interpreter.partial_path(target).write_bytes(BLOB[:123_456])
    with _serving(BLOB, [None]) as (url, seen):
        interpreter.fetch(url, target, sleep=lambda s: None)
    assert target.read_bytes() == BLOB
    assert seen == ["bytes=123456-"]


def test_a_server_that_ignores_the_range_starts_the_file_over(tmp_path: Path) -> None:
    target = tmp_path / "x.bin"
    interpreter.partial_path(target).write_bytes(b"stale bytes that are not a prefix")
    with _serving(BLOB, ["whole"]) as (url, _seen):
        interpreter.fetch(url, target, sleep=lambda s: None)
    assert target.read_bytes() == BLOB


def test_kept_bytes_the_server_cannot_continue_are_dropped_and_fetched_again(tmp_path: Path) -> None:
    target = tmp_path / "x.bin"
    interpreter.partial_path(target).write_bytes(BLOB)  # already whole: the Range is unsatisfiable
    with _serving(BLOB, [("status", 416), None]) as (url, seen):
        interpreter.fetch(url, target, sleep=lambda s: None)
    assert target.read_bytes() == BLOB
    assert seen == [f"bytes={len(BLOB)}-", None]


def test_a_404_ends_the_fetch_at_once_and_is_not_waited_on(tmp_path: Path) -> None:
    pauses: list[float] = []
    with _serving(BLOB, [("status", 404)]) as (url, seen):
        with pytest.raises(InterpreterError) as caught:
            interpreter.fetch(url, tmp_path / "x.bin", sleep=pauses.append)
    assert caught.value.code == "interpreter_download_failed"
    assert "HTTP 404" in caught.value.message
    assert pauses == [] and len(seen) == 1


def test_a_budget_spent_keeps_the_bytes_and_says_where(tmp_path: Path) -> None:
    target = tmp_path / "x.bin"
    with _serving(BLOB, [("cut", 100_000)] * 3) as (url, _seen):
        with pytest.raises(InterpreterError) as caught:
            interpreter.fetch(url, target, attempts=3, chunk=65536, sleep=lambda s: None)
    kept = interpreter.partial_path(target)
    assert kept.stat().st_size == 300_000
    assert str(kept) in caught.value.message and "continues from them" in caught.value.message
    assert not target.exists()
