from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from crucible import llamacpp, weights
from crucible.config import Config

GIB = 1024 ** 3


def _config(home: Path) -> Config:
    home.mkdir(parents=True, exist_ok=True)
    return Config(
        path=home / "config.toml",
        home=home,
        name="crucible@staged",
        host="127.0.0.1",
        port=7101,
        token="t",
        backend_kind="llama-windows",
        enable_echo=True,
        enable_llm=True,
        enable_asr=False,
        enable_tts=False,
        enable_align=False,
        enable_rvc=False,
        enable_denoise=False,
        desktop_allowance_bytes=3 * GIB,
        desktop_allowance_basis="stated",
        capability=None,
    )


def _zip_bytes(names: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, payload in names.items():
            bundle.writestr(name, payload)
    return buffer.getvalue()


class Release:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.asked: list[str] = []

    def fetch(self, url: str, destination: Path, on_progress) -> None:
        self.asked.append(url)
        destination.write_bytes(self.payloads[url.rsplit("/", 1)[-1]])


def _pin(monkeypatch, tmp_path: Path, payloads: dict[str, bytes]) -> None:
    assets = []
    for name, payload in payloads.items():
        sample = tmp_path / "hash" / name
        sample.parent.mkdir(parents=True, exist_ok=True)
        sample.write_bytes(payload)
        assets.append(llamacpp.Asset(name=name, bytes=len(payload), sha256=weights.sha256_of(sample)))
    monkeypatch.setattr(llamacpp, "CUDA_ASSETS", tuple(assets))
    monkeypatch.setattr(llamacpp, "CPU_ASSETS", tuple(assets))


OLD = {"server.zip": _zip_bytes({"llama-server.exe": b"MZ the old server"})}
NEW = {"server.zip": _zip_bytes({"llama-server.exe": b"MZ the new server"})}


def _installed_old(tmp_path: Path, monkeypatch) -> Config:
    config = _config(tmp_path / "home")
    _pin(monkeypatch, tmp_path, OLD)
    llamacpp.pull(config, llamacpp.CUDA_BUILD, fetch=Release(OLD).fetch)
    return config


def _only_the_engine_is_left(config: Config) -> None:
    assert sorted(p.name for p in llamacpp.engine_dir(config).parent.iterdir()) == [
        llamacpp.LLAMA_CPP_ID
    ]


def test_a_forced_reinstall_whose_download_fails_its_digest_leaves_the_old_engine(
    tmp_path: Path, monkeypatch
) -> None:
    config = _installed_old(tmp_path, monkeypatch)
    stamp = llamacpp.stamp_path(config).read_text(encoding="utf-8")
    _pin(monkeypatch, tmp_path / "new", NEW)
    tampered = {"server.zip": _zip_bytes({"llama-server.exe": b"not what was published"})}

    with pytest.raises(llamacpp.EngineSubjectError) as caught:
        llamacpp.pull(config, llamacpp.CUDA_BUILD, force=True, fetch=Release(tampered).fetch)

    assert caught.value.code == "engine_sha_mismatch"
    assert "pull again" in caught.value.message
    assert llamacpp.server_path(config).read_bytes() == b"MZ the old server"
    assert llamacpp.stamp_path(config).read_text(encoding="utf-8") == stamp
    assert llamacpp.installed(config, llamacpp.CUDA_BUILD) is not None
    _only_the_engine_is_left(config)


def test_a_forced_reinstall_that_cannot_find_its_server_leaves_the_old_engine(
    tmp_path: Path, monkeypatch
) -> None:
    config = _installed_old(tmp_path, monkeypatch)
    empty = {"server.zip": _zip_bytes({"readme.txt": b"nothing"})}
    _pin(monkeypatch, tmp_path / "empty", empty)

    with pytest.raises(llamacpp.EngineSubjectError):
        llamacpp.pull(config, llamacpp.CUDA_BUILD, force=True, fetch=Release(empty).fetch)

    assert llamacpp.server_path(config).read_bytes() == b"MZ the old server"
    _only_the_engine_is_left(config)


def test_a_forced_reinstall_swaps_the_verified_engine_in_whole(
    tmp_path: Path, monkeypatch
) -> None:
    config = _installed_old(tmp_path, monkeypatch)
    _pin(monkeypatch, tmp_path / "new", NEW)

    found = llamacpp.pull(config, llamacpp.CUDA_BUILD, force=True, fetch=Release(NEW).fetch)

    assert found.path == llamacpp.engine_dir(config)
    assert llamacpp.server_path(config).read_bytes() == b"MZ the new server"
    record = json.loads(llamacpp.stamp_path(config).read_text(encoding="utf-8"))
    assert record["build"] == llamacpp.CUDA_BUILD
    _only_the_engine_is_left(config)


def test_an_engine_that_cannot_be_moved_aside_is_refused_and_kept(
    tmp_path: Path, monkeypatch
) -> None:
    config = _installed_old(tmp_path, monkeypatch)
    _pin(monkeypatch, tmp_path / "new", NEW)
    target = llamacpp.engine_dir(config)
    rename = Path.rename

    def held_open(self: Path, destination):
        if self == target:
            raise PermissionError("the process cannot access the file")
        return rename(self, destination)

    monkeypatch.setattr(Path, "rename", held_open)
    with pytest.raises(llamacpp.EngineSubjectError) as caught:
        llamacpp.pull(config, llamacpp.CUDA_BUILD, force=True, fetch=Release(NEW).fetch)

    assert caught.value.code == "engine_replace_failed"
    assert "unload-model" in caught.value.message
    assert llamacpp.server_path(config).read_bytes() == b"MZ the old server"
    _only_the_engine_is_left(config)


def test_a_first_pull_that_fails_leaves_nothing_behind(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path / "home")
    _pin(monkeypatch, tmp_path, OLD)
    tampered = {"server.zip": b"not a zip"}

    with pytest.raises(llamacpp.EngineSubjectError):
        llamacpp.pull(config, llamacpp.CUDA_BUILD, fetch=Release(tampered).fetch)

    assert not llamacpp.engine_dir(config).exists()
    assert list(llamacpp.engine_dir(config).parent.iterdir()) == []
