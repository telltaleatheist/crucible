from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import capability, capabilitystore, cli
from crucible.config import config_path, load_config

from .conftest import FAKE_BACKEND, TOKEN


def _record_with(decisions: tuple[Any, ...]) -> Any:
    return capability.record(
        FAKE_BACKEND.kind,
        total_bytes=FAKE_BACKEND.gpu.vram_bytes,
        desktop_allowance_bytes=3 * 1024**3,
        decisions=decisions,
        routes={},
    )


def _decide(name: str) -> Any:
    entry = capability.BY_NAME[name]
    return capability.decide(
        entry,
        FAKE_BACKEND.kind,
        total_bytes=FAKE_BACKEND.gpu.vram_bytes,
        desktop_allowance_bytes=3 * 1024**3,
        gpu_vendor=FAKE_BACKEND.gpu.vendor,
        chosen=None,
    )


def test_a_record_written_by_another_process_is_served_without_a_restart(
    make_client: Callable[..., TestClient], auth: dict[str, str], home: Path
) -> None:
    with make_client(capability=None) as client:
        undecided = client.get("/v1/capability", headers=auth)
        assert undecided.status_code == 503
        assert undecided.json()["error"]["code"] == "capability_undecided"

        capabilitystore.write_capability(load_config(home), FAKE_BACKEND, (_decide("echo"),), {})

        decided = client.get("/v1/capability", headers=auth)
        assert decided.status_code == 200, decided.text
        rows = {row["capability"]: row for row in decided.json()["classes"]}
        assert rows["echo"]["enabled"] is True
        assert decided.json()["total_bytes"] == FAKE_BACKEND.gpu.vram_bytes


def test_a_flag_turned_off_in_the_file_is_honoured_by_the_door(
    make_client: Callable[..., TestClient], auth: dict[str, str], home: Path
) -> None:
    with make_client(enable_echo=True) as client:
        accepted = client.post(
            "/v1/jobs",
            headers=auth,
            json={"type": "echo", "params": {}, "inputs": {"x.bin": {"inline_base64": "YQ=="}}},
        )
        assert accepted.status_code == 202, accepted.text

        current = load_config(home)
        capabilitystore.write_capability(current, FAKE_BACKEND, (), {"enable_echo": False})

        refused = client.post(
            "/v1/jobs",
            headers=auth,
            json={"type": "echo", "params": {}, "inputs": {"x.bin": {"inline_base64": "YQ=="}}},
        )
        assert refused.status_code != 202, refused.text
        assert "echo" in refused.json()["error"]["message"]


def test_an_unchanged_file_is_not_re_read(
    make_client: Callable[..., TestClient], auth: dict[str, str], home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import crucible.config as config_module

    with make_client() as client:
        calls: list[Path | None] = []
        real = config_module.load_config

        def counting(home_arg: Path | None = None) -> Any:
            calls.append(home_arg)
            return real(home_arg)

        monkeypatch.setattr(config_module, "load_config", counting)
        for _ in range(5):
            assert client.get("/v1/ping", headers=auth).status_code == 200
        assert calls == []


def test_a_file_that_will_not_read_keeps_the_last_good_document(
    make_client: Callable[..., TestClient], auth: dict[str, str], home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with make_client(capability=_record_with((_decide("echo"),))) as client:
        assert client.get("/v1/capability", headers=auth).status_code == 200
        path = config_path(home)
        good = path.read_bytes()
        path.write_bytes(b"[server\nthis is not toml")
        for _ in range(3):
            assert client.get("/v1/ping", headers=auth).status_code == 200
            still = client.get("/v1/capability", headers=auth)
            assert still.status_code == 200
            assert {row["capability"] for row in still.json()["classes"]} == {"echo"}
        err = capsys.readouterr().err
        assert err.count("could not be re-read") == 1, err
        path.write_bytes(good)
        assert client.get("/v1/capability", headers=auth).status_code == 200


def test_the_stamp_is_taken_before_the_read(home: Path, make_client: Callable[..., TestClient]) -> None:
    with make_client():
        loaded = load_config(home)
        assert loaded.stamp is not None
        path = config_path(home)
        current = path.stat()
        assert loaded.stamp == (current.st_mtime_ns, current.st_size)
        assert loaded.follow_file() is False
        capabilitystore.write_capability(loaded, FAKE_BACKEND, (), {})
        assert loaded.follow_file() is True
        assert loaded.follow_file() is False



def _disagreeing_record(**overrides: Any) -> Any:
    import dataclasses

    return dataclasses.replace(_record_with((_decide("echo"),)), **overrides)


def _written_with(home: Path, record: Any, desktop_allowance_bytes: int) -> None:
    from crucible.config import write_config

    write_config(
        home,
        name="crucible@test",
        host="127.0.0.1",
        port=7100,
        token=TOKEN,
        backend_kind=FAKE_BACKEND.kind,
        enable_echo=True,
        enable_llm=False,
        enable_asr=False,
        enable_tts=False,
        enable_align=False,
        enable_rvc=False,
        enable_denoise=False,
        desktop_allowance_bytes=desktop_allowance_bytes,
        retention_days=7,
        desktop_allowance_basis="stated",
        desktop_allowance_note="",
        capability=record,
    )


def test_a_record_decided_for_another_backend_does_not_load(home: Path) -> None:
    from crucible.errors import ConfigError

    _written_with(home, _disagreeing_record(backend_kind="mlx-darwin"), 3 * 1024**3)
    with pytest.raises(ConfigError) as caught:
        load_config(home)
    message = str(caught.value)
    assert "'mlx-darwin'" in message and "'cuda-linux'" in message
    assert "`crucible capability --write`" in message
    assert load_config(home, tolerate_stale_record=True).capability is not None


def test_a_record_decided_with_another_reserve_does_not_load(home: Path) -> None:
    from crucible.errors import ConfigError

    _written_with(home, _disagreeing_record(), 5 * 1024**3)
    with pytest.raises(ConfigError) as caught:
        load_config(home)
    message = str(caught.value)
    assert str(3 * 1024**3) in message and str(5 * 1024**3) in message
    assert "[capability] desktop_allowance_bytes" in message
    assert "[accelerator] desktop_allowance_bytes" in message
    assert "`crucible capability --write`" in message
    tolerated = load_config(home, tolerate_stale_record=True)
    assert tolerated.desktop_allowance_bytes == 5 * 1024**3


def test_capability_write_repairs_a_record_that_disagrees(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _written_with(home, _disagreeing_record(), 5 * 1024**3)
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)
    assert cli.main(["capability", "--write"]) == 0
    capsys.readouterr()
    repaired = load_config(home)
    assert repaired.capability is not None
    assert repaired.capability.desktop_allowance_bytes == 5 * 1024**3
