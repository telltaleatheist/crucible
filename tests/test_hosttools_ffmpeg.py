from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

import pytest

from crucible import hosttools

TOOLS_RELEASE = "https://github.com/telltaleatheist/crucible/releases/download/tools/"


@pytest.mark.parametrize("platform_key", sorted(hosttools.FFMPEG_BUILDS))
def test_every_pinned_build_is_on_our_tools_release_with_a_full_digest(platform_key: str) -> None:
    build = hosttools.FFMPEG_BUILDS[platform_key]
    assert build.url.startswith(TOOLS_RELEASE)
    assert build.url.endswith(".tar.xz")
    assert re.fullmatch(r"[0-9a-f]{64}", build.sha256)
    assert build.bytes > 10_000_000
    assert build.root == build.url.rsplit("/", 1)[-1].removesuffix(".tar.xz")
    assert build.version in build.root
    assert build.provenance


def _archive_with(root: str, programs: tuple[str, ...]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:xz") as bundle:
        for name in programs:
            body = f"#!/bin/sh\necho fake {name}\n".encode()
            info = tarfile.TarInfo(f"{root}/bin/{name}")
            info.size = len(body)
            info.mode = 0o755
            bundle.addfile(info, io.BytesIO(body))
    return raw.getvalue()


def _pinned(monkeypatch: pytest.MonkeyPatch, archive: bytes, *, sha256: str | None = None) -> hosttools.ToolBuild:
    build = hosttools.ToolBuild(
        version="n8.1.3",
        url=TOOLS_RELEASE + "ffmpeg-n8.1.3-test-lgpl.tar.xz",
        sha256=sha256 or hashlib.sha256(archive).hexdigest(),
        bytes=len(archive),
        root="ffmpeg-n8.1.3-test-lgpl",
        provenance="a test archive",
    )
    monkeypatch.setattr(hosttools, "ffmpeg_build", lambda platform_key=None: build)
    return build


def _fetch_from(archive: bytes):
    def fetch(url: str, destination: Path) -> str:
        destination.write_bytes(archive)
        return hashlib.sha256(archive).hexdigest()

    return fetch


def test_the_pinned_archive_is_placed_and_stamped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _archive_with("ffmpeg-n8.1.3-test-lgpl", hosttools.FFMPEG_PROGRAMS)
    build = _pinned(monkeypatch, archive)
    said = hosttools.ensure_ffmpeg(tmp_path, fetch=_fetch_from(archive))
    assert "n8.1.3" in said
    for name in hosttools.FFMPEG_PROGRAMS:
        placed = tmp_path / hosttools.TOOLS_DIR_NAME / "bin" / name
        assert placed.read_text().startswith("#!/bin/sh")
    stamp = json.loads(hosttools.ffmpeg_stamp(tmp_path).read_text())
    assert stamp["sha256"] == build.sha256
    assert hosttools.ffmpeg_placed(tmp_path, build)
    assert "already in" in hosttools.ensure_ffmpeg(tmp_path, fetch=_fetch_from(archive))


def test_bytes_that_do_not_match_the_pin_place_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _archive_with("ffmpeg-n8.1.3-test-lgpl", hosttools.FFMPEG_PROGRAMS)
    _pinned(monkeypatch, archive, sha256="0" * 64)
    with pytest.raises(hosttools.HostToolError) as refused:
        hosttools.ensure_ffmpeg(tmp_path, fetch=_fetch_from(archive))
    assert refused.value.code == "tool_sha_mismatch"
    assert not (tmp_path / hosttools.TOOLS_DIR_NAME / "bin" / "ffmpeg").exists()
    assert not hosttools.ffmpeg_stamp(tmp_path).exists()


def test_an_archive_missing_ffprobe_is_refused_by_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _archive_with("ffmpeg-n8.1.3-test-lgpl", ("ffmpeg",))
    _pinned(monkeypatch, archive)
    with pytest.raises(hosttools.HostToolError) as refused:
        hosttools.ensure_ffmpeg(tmp_path, fetch=_fetch_from(archive))
    assert refused.value.code == "tool_unpack_failed"
    assert "ffprobe" in refused.value.message
