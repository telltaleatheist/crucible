"""`crucible env tools`, the installer's `env-tools` step: an update leaves every pinned host
tool an installed job type uses at this release's pin.

Victoria's laptop, 2026-10-11: the llama-server pin moved in 1.0.132, the laptop updated
1.0.124 → 2.0.0, and no update replaced ~/.crucible/tools/bin/llama-server; only
`crucible install llm` did. Doctor said llama.cpp was not placed and the installer said ready."""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from crucible import cli, hosttools, jobenv

from .conftest import FAKE_BACKEND

TOOLS_RELEASE = "https://github.com/telltaleatheist/crucible/releases/download/tools/"
ROOT = "llama-server-test-linux-x86_64"


def _archive(body: bytes) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:xz") as bundle:
        info = tarfile.TarInfo(f"{ROOT}/bin/llama-server")
        info.size = len(body)
        info.mode = 0o755
        bundle.addfile(info, io.BytesIO(body))
    return raw.getvalue()


def _pin(monkeypatch: pytest.MonkeyPatch, archive: bytes, fetched: list[str]) -> hosttools.ToolBuild:
    build = hosttools.ToolBuild(
        version="test",
        url=TOOLS_RELEASE + f"{ROOT}.tar.xz",
        sha256=hashlib.sha256(archive).hexdigest(),
        bytes=len(archive),
        root=ROOT,
        provenance="a test archive",
    )
    monkeypatch.setattr(hosttools, "llama_server_build", lambda platform_key=None: build)

    def download(url: str, destination: Path) -> str:
        fetched.append(url)
        destination.write_bytes(archive)
        return hashlib.sha256(archive).hexdigest()

    monkeypatch.setattr(hosttools, "_download", download)
    return build


@pytest.fixture
def box(home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> list[str]:
    """A cuda-linux home with no env yet, whose ffmpeg and speech detector say what they
    did (their own placement is tests/test_hosttools*.py's)."""
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)
    assert cli.main(["init", "--enable-llm"]) == 0
    said: list[str] = []
    monkeypatch.setattr(
        hosttools, "ensure_ffmpeg", lambda home_dir, **_: said.append("ffmpeg") or "ffmpeg: ok"
    )
    monkeypatch.setattr(
        hosttools, "ensure_silero_vad", lambda home_dir: said.append("vad") or "vad: ok"
    )
    capsys.readouterr()
    return said


def _installed(home: Path, spec: jobenv.EnvSpec) -> None:
    python = jobenv.env_python(home, spec)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("", encoding="utf-8")


def test_an_update_replaces_a_llama_server_an_older_pin_placed(
    home: Path, box: list[str], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _installed(home, jobenv.llm_env(FAKE_BACKEND.kind))
    fetched: list[str] = []
    old = _pin(monkeypatch, _archive(b"the 1.0.124 server"), fetched)
    hosttools.ensure_llama_server(home)
    new = _pin(monkeypatch, _archive(b"the 2.0.0 server"), fetched)
    assert not hosttools.llama_server_placed(home, new)

    assert cli.main(["env", "tools"]) == 0
    out = capsys.readouterr().out
    assert "llama-server: test placed at" in out
    assert hosttools.llama_server_path(home).read_bytes() == b"the 2.0.0 server"
    assert hosttools.llama_server_placed(home, new)
    assert not hosttools.llama_server_placed(home, old)
    stamp = json.loads(hosttools.llama_server_stamp(home).read_text(encoding="utf-8"))
    assert stamp["sha256"] == new.sha256
    assert box == ["ffmpeg", "vad"]

    fetched.clear()
    assert cli.main(["env", "tools"]) == 0
    assert "llama-server: test already at" in capsys.readouterr().out
    assert fetched == [], "a stamp that names the pin is kept, not fetched again"


def test_a_host_without_the_llm_env_gets_no_llama_server(
    home: Path, box: list[str], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _installed(home, jobenv.worker_env("asr", FAKE_BACKEND.kind))
    fetched: list[str] = []
    _pin(monkeypatch, _archive(b"the server"), fetched)
    assert cli.main(["env", "tools"]) == 0
    out = capsys.readouterr().out
    assert "'asr'" in out and "llama-server" not in out
    assert fetched == [] and not hosttools.llama_server_path(home).exists()
    assert box == ["ffmpeg", "vad"], "ffmpeg belongs to every installed job type"


def test_a_host_with_no_env_places_nothing_and_says_so(
    home: Path, box: list[str], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fetched: list[str] = []
    _pin(monkeypatch, _archive(b"the server"), fetched)
    assert cli.main(["env", "tools"]) == 0
    assert "there is none to place" in capsys.readouterr().out
    assert fetched == [] and box == []


def test_a_llama_server_that_cannot_be_placed_fails_the_step_by_name(
    home: Path, box: list[str], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _installed(home, jobenv.llm_env(FAKE_BACKEND.kind))
    _pin(monkeypatch, _archive(b"the server"), [])

    def offline(url: str, destination: Path) -> str:
        raise hosttools.HostToolError("tool_download_failed", f"{url} could not be fetched")

    monkeypatch.setattr(hosttools, "_download", offline)
    assert cli.main(["env", "tools"]) == 1
    err = capsys.readouterr().err
    assert "llama-server is not" in err and "tool_download_failed" in err


def test_a_failed_ffmpeg_fails_the_step_by_name(
    home: Path, box: list[str], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _installed(home, jobenv.worker_env("asr", FAKE_BACKEND.kind))

    def refuse(home_dir: Path, **_: object) -> str:
        raise hosttools.HostToolError("tool_sha_mismatch", "not the pinned bytes")

    monkeypatch.setattr(hosttools, "ensure_ffmpeg", refuse)
    assert cli.main(["env", "tools"]) == 1
    err = capsys.readouterr().err
    assert "ffmpeg is not" in err and "tool_sha_mismatch" in err
