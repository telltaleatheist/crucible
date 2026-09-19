"""The pinned CPython — one owner, one publisher, verified by digest.

PHASE20-CODE-NOT-ENVIRONMENTS.md section 2: an interpreter is downloaded ONCE,
at install, from wherever its bytes are published, and is never rebuilt or
re-hosted because our code changed. These are the assertions that used to live
in `tests/test_envpack.py` around the same table — they guard something that is
still true, so they moved rather than died.
"""

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


# --------------------------------------------------------------- the table


def test_every_pin_names_a_version_a_url_and_a_digest() -> None:
    """Version, url and sha256 on every row, and the url is the publisher's.

    A row that named no digest would be a download nobody can check, and a row
    whose url was not python-build-standalone's would be an interpreter from a
    second publisher — which is the two-owners shape PHASE20 exists to remove.
    """
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
    """`requires-python = ">=3.11"` is the floor the wheel declares, and the
    recipes were resolved by pip against 3.11. One minor for the server, three
    backends, one release."""
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
    """A REGRESSION PIN ON PHASE15 4.4's OWN CORRECTION.

    The doc first wrote the Windows interpreter as
    `x86_64-pc-windows-msvc-SHARED-install_only` and then corrected itself from
    the release's `SHA256SUMS`: there is no such asset on 20260901, the
    `-shared` infix is retired, and the Windows `install_only` build IS the
    shared one. A pin nobody can download fails on the machine that runs
    `install.ps1` and nowhere else, so the spelling is asserted here where it
    costs a second.
    """
    pin = interpreter.pin_for(LLAMA_WINDOWS, "3.11")
    assert pin.asset == (
        "cpython-3.11.16+20260901-x86_64-pc-windows-msvc-install_only.tar.gz"
    )
    assert pin.sha256 == (
        "6be524fa6752af802146a4adc7d098565425b0b1c166e19a5a7a4c8cccb86bf6"
    )
    assert "-shared" not in pin.asset


def test_the_higgs_recipes_312_is_pinned_and_digested() -> None:
    """The one recipe that names a CPython the server does not run.

    sglang-omni 0.1.4 pulls torch 2.13.0+cu130 and flashinfer against 3.12, so
    `jobenv.interpreter_for` downloads this pin rather than searching PATH.
    Digest read from the upstream 20260901 SHA256SUMS on 2026-09-16.
    """
    pin = interpreter.pin_for(CUDA_LINUX, "3.12")
    assert pin.python_version == "3.12.14"
    assert pin.asset == (
        "cpython-3.12.14+20260901-x86_64-unknown-linux-gnu-install_only.tar.gz"
    )
    assert pin.sha256 == (
        "936c246dfdbbfa7cb22dd01814a21f582a892689fae96b06071a5e433baffa22"
    )


def test_a_version_nobody_pinned_is_refused_by_name() -> None:
    """Refused BEFORE anything downloads, naming what IS pinned. A guess here
    would be an interpreter from a url nobody verified."""
    with pytest.raises(InterpreterError) as caught:
        interpreter.pin_for(CUDA_LINUX, "3.99")
    assert caught.value.code == "interpreter_not_pinned"
    assert "3.99" in str(caught.value)
    with pytest.raises(InterpreterError) as caught:
        interpreter.pin_for("rocm-linux", "3.11")
    assert caught.value.code == "interpreter_not_pinned"


def test_the_interpreter_layout_is_asked_for_and_never_spelled() -> None:
    """python-build-standalone's Windows `install_only` tree is `python.exe`,
    `pythonw.exe`, `Scripts\\`, `Lib\\`, `DLLs\\` — there is no `bin/` at all.
    Three call sites spelling `bin/python` inline would be three places that
    have to learn this and two that will not."""
    root = Path("/p")
    assert interpreter.interpreter_python(root, CUDA_LINUX) == root / "bin" / "python"
    assert interpreter.interpreter_python(root, MLX_DARWIN) == root / "bin" / "python"
    assert interpreter.interpreter_python(root, LLAMA_WINDOWS) == root / "python.exe"
    with pytest.raises(InterpreterError) as caught:
        interpreter.interpreter_python(root, "rocm-linux")
    assert caught.value.code == "interpreter_not_pinned"


# ------------------------------------------------------------ the download


def _install_only_tarball(destination: Path, *, marker: str = "print('hi')\n") -> bytes:
    """A python-build-standalone-shaped `install_only` archive, in memory.

    One top-level `python/` directory, `python/bin/python` inside it — which is
    the contract `download_interpreter` unpacks against and the reason it moves
    `python/` rather than the archive's root.
    """
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
        interpreter, "_fetch", lambda url, path, **kw: path.write_bytes(blob)
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
    """ONCE. An interpreter is a publisher's bytes pinned by digest, so a tree
    whose stamp names that digest is those bytes — there is nothing a second
    download could correct."""
    source = tmp_path / "src.tar.gz"
    blob = _install_only_tarball(source)
    pin = _pin(hashlib.sha256(blob).hexdigest())
    calls: list[str] = []

    def fetch(url: str, path: Path, **kw: object) -> None:
        calls.append(url)
        path.write_bytes(blob)

    monkeypatch.setattr(interpreter, "_fetch", fetch)
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
        interpreter, "_fetch", lambda url, path, **kw: path.write_bytes(blob)
    )
    home = tmp_path / "home"
    with pytest.raises(InterpreterError) as caught:
        interpreter.ensure_interpreter(home, CUDA_LINUX, "3.12", pin=pin)
    assert caught.value.code == "interpreter_sha_mismatch"
    assert pin.asset in str(caught.value)
    assert not (home / "interpreters" / "3.12.14").exists()


# ------------------------------------------------------------ the byte wire


def test_the_progress_line_round_trips() -> None:
    """The one channel between `crucible install` and the install TASK: the
    child's stdout. A declared wire carrying JSON, not a log scrape (R4)."""
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
    """`weights.ProgressHook` says total may be None, and zero is a lie."""
    parsed = interpreter.parse_progress_line(interpreter.progress_line(5, None, "f"))
    assert parsed == {"bytes_done": 5, "bytes_total": None, "file": "f"}
