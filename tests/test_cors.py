"""`[server] cors_origins`: web pages in a listed origin may call this server.

B-Side's iPhone app (capacitor://localhost) pairs by address and follows jobs with
fetch-streamed SSE; until 1.0.101 every such call died at its preflight (405).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible.config import load_config
from crucible.errors import ConfigError

from .conftest import configure_box

PHONE = "capacitor://localhost"


def preflight(client: TestClient, path: str, origin: str) -> Any:
    return client.options(path, headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization, x-crucible-api, content-type",
    })


def test_a_listed_origin_s_preflight_is_answered_with_what_it_may_send(
    make_client: Callable[..., TestClient],
) -> None:
    with make_client(cors_origins=(PHONE,)) as client:
        answer = preflight(client, "/v1/pairing/start", PHONE)
    assert answer.status_code == 204
    assert answer.headers["access-control-allow-origin"] == PHONE
    allowed = answer.headers["access-control-allow-headers"].lower()
    for header in ("authorization", "x-crucible-api", "last-event-id", "content-type", "range"):
        assert header in allowed
    assert "POST" in answer.headers["access-control-allow-methods"]


def test_an_unlisted_origin_is_answered_exactly_as_before(make_client: Callable[..., TestClient]) -> None:
    with make_client(cors_origins=(PHONE,)) as client:
        answer = preflight(client, "/v1/pairing/start", "https://evil.example")
    assert answer.status_code == 405
    assert "access-control-allow-origin" not in answer.headers


def test_an_empty_list_changes_nothing(make_client: Callable[..., TestClient], auth: dict[str, str]) -> None:
    with make_client() as client:
        assert preflight(client, "/v1/pairing/start", PHONE).status_code == 405
        answer = client.get("/v1/info", headers={**auth, "Origin": PHONE})
    assert answer.status_code == 200
    assert "access-control-allow-origin" not in answer.headers


def test_real_responses_carry_the_origin_refusals_and_streams_included(
    make_client: Callable[..., TestClient], auth: dict[str, str],
) -> None:
    with make_client(cors_origins=(PHONE,)) as client:
        info = client.get("/v1/info", headers={**auth, "Origin": PHONE})
        assert info.headers["access-control-allow-origin"] == PHONE
        assert "content-range" in info.headers["access-control-expose-headers"].lower()
        # A refusal must be readable too, or the page sees a network error, not the reason.
        refused = client.get("/v1/info", headers={"Origin": PHONE, "X-Crucible-Api": auth["X-Crucible-Api"]})
        assert refused.status_code == 401
        assert refused.headers["access-control-allow-origin"] == PHONE
        # A job's event stream (SSE) is labelled at its start like any response.
        job = client.post("/v1/jobs", headers={**auth, "Origin": PHONE},
                          json={"type": "echo", "params": {"text": "hi"}, "inputs": {}})
        assert job.status_code == 202, job.text
        with client.stream("GET", f"/v1/jobs/{job.json()['job_id']}/events",
                           headers={**auth, "Origin": PHONE}) as stream:
            assert stream.headers["access-control-allow-origin"] == PHONE
            assert stream.headers["content-type"].startswith("text/event-stream")


def test_a_config_edit_takes_effect_without_a_restart(
    make_client: Callable[..., TestClient], home: Path,
) -> None:
    with make_client() as client:
        assert preflight(client, "/v1/pairing/start", PHONE).status_code == 405
        config = load_config(home)
        configure_box(home, cors_origins=(PHONE,))
        assert load_config(home).cors_origins == (PHONE,)
        assert config.cors_origins == ()
        assert preflight(client, "/v1/pairing/start", PHONE).status_code == 204


def test_a_rewrite_keeps_the_origins_a_person_listed(home: Path) -> None:
    configure_box(home, cors_origins=(PHONE,))
    # Every `crucible install` rewrites config.toml; [server] is the writer's table.
    from crucible.config import write_config  # noqa: PLC0415 (the call below mirrors an install's)
    config = load_config(home)
    write_config(
        home, name=config.name, host=config.host, port=config.port, token=config.token,
        backend_kind=config.backend_kind, enable_echo=True, enable_llm=False, enable_asr=False,
        enable_tts=False, enable_align=False, enable_rvc=False, enable_denoise=False,
        desktop_allowance_bytes=config.desktop_allowance_bytes, retention_days=7,
        desktop_allowance_basis="stated", desktop_allowance_note="",
    )
    assert load_config(home).cors_origins == (PHONE,)


@pytest.mark.parametrize("bad", [["*"], ["capacitor://localhost/"], ["localhost:7300"], "capacitor://localhost"])
def test_what_is_not_an_origin_is_refused_by_name(home: Path, bad: Any) -> None:
    configure_box(home)
    path = home / "config.toml"
    text = path.read_text(encoding="utf-8")
    value = f'"{bad}"' if isinstance(bad, str) else "[" + ", ".join(f'"{entry}"' for entry in bad) + "]"
    path.write_text(text.replace("[server]\n", f"[server]\ncors_origins = {value}\n", 1), encoding="utf-8")
    with pytest.raises(ConfigError, match="cors_origins"):
        load_config(home)
