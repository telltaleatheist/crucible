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
