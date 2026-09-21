"""A running server follows its own config.toml.

THE INCIDENT, Mac Studio, 2026-09-21. Crucible 1.0.17 deployed with dots-ocr's
new `[backends.mlx-darwin]` block; the service restarted at 13:31; `crucible
install llm --force` rebuilt the env and wrote `pages: yes` into config.toml
at 13:33. `GET /v1/capability` went on answering the 13:31 document — *"ships
none with a mlx-darwin block"* — and BookForge, obeying the wire, refused to
route a single page to a machine whose own CLI said yes. It took a restart.

That was the second time this shipped. In 1.0.16 `crucible install` restarted
the service BEFORE writing the record, and the server was one write behind
for the same reason from the other side. The fix is not an order, it is an
owner: the FILE is the authority, and `Config.follow_file()` re-reads it
whenever its stamp moves, from a middleware that runs before every request.
`adopt()` replaces the one Config object's fields in place, so every route,
the residency and the store — all of which close over that object — see the
new document without anybody being rebound.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import capability, cli
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
    """THE INCIDENT. A server that decided nothing; then the CLI's own writer
    records a decision in the file; then the same server answers with it."""
    with make_client(capability=None) as client:
        undecided = client.get("/v1/capability", headers=auth)
        assert undecided.status_code == 503
        assert undecided.json()["error"]["code"] == "capability_undecided"

        # `cli._write_capability` is what `crucible install` and `crucible
        # capability --write` call — the real writer, in its own process
        # here in spirit: nothing tells the app.
        cli._write_capability(load_config(home), FAKE_BACKEND, (_decide("echo"),), {})

        decided = client.get("/v1/capability", headers=auth)
        assert decided.status_code == 200, decided.text
        rows = {row["capability"]: row for row in decided.json()["classes"]}
        assert rows["echo"]["enabled"] is True
        assert decided.json()["total_bytes"] == FAKE_BACKEND.gpu.vram_bytes


def test_a_flag_turned_off_in_the_file_is_honoured_by_the_door(
    make_client: Callable[..., TestClient], auth: dict[str, str], home: Path
) -> None:
    """`[jobs] enable_*` stays the one owner of what a server offers, and the
    doors read it at request time — so turning it off on disk turns the door
    off, with no restart between."""
    with make_client(enable_echo=True) as client:
        accepted = client.post(
            "/v1/jobs",
            headers=auth,
            json={"type": "echo", "params": {}, "inputs": {"x.bin": {"inline_base64": "YQ=="}}},
        )
        assert accepted.status_code == 202, accepted.text

        current = load_config(home)
        cli._write_capability(current, FAKE_BACKEND, (), {"enable_echo": False})

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
    """One stat per request, a parse only when the stamp moves — a server
    answering /v1/ping a thousand times must not parse TOML a thousand times."""
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
    """Reported, not served. The writer stages and `os.replace`s so a torn file
    is not expected — but a broken one must not take `/v1/ping` down with it,
    and it must be said once, not once per request."""
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
        # And once the file is repaired, it is followed again.
        path.write_bytes(good)
        assert client.get("/v1/capability", headers=auth).status_code == 200


def test_the_stamp_is_taken_before_the_read(home: Path, make_client: Callable[..., TestClient]) -> None:
    """A write that lands between the stat and the parse must not be missed:
    the stamp is older than the content, so the next follow re-reads once."""
    with make_client():
        loaded = load_config(home)
        assert loaded.stamp is not None
        path = config_path(home)
        current = path.stat()
        assert loaded.stamp == (current.st_mtime_ns, current.st_size)
        assert loaded.follow_file() is False
        # Rewrite the same document: the stamp moves even if the bytes do not.
        cli._write_capability(loaded, FAKE_BACKEND, (), {})
        assert loaded.follow_file() is True
        assert loaded.follow_file() is False
