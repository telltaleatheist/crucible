"""Shared fixtures.

Every test runs against a throwaway `CRUCIBLE_HOME` under pytest's tmp_path, so no
test can see or touch a real `~/.crucible`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import API_VERSION
from crucible.api import create_app
from crucible.backend import Backend, Gpu
from crucible.config import load_config, mint_token, write_config

# A stand-in for a real host probe. Backend detection itself is tested separately
# against monkeypatched probes; the API tests must not depend on the machine.
FAKE_BACKEND = Backend(
    kind="cuda-linux",
    platform="linux",
    arch="x86_64",
    gpu=Gpu(vendor="nvidia", name="NVIDIA GeForce RTX 3090 Ti", vram_bytes=25_757_220_864),
    detail="test double",
)

FAKE_MAC_BACKEND = Backend(
    kind="mlx-darwin",
    platform="darwin",
    arch="arm64",
    gpu=Gpu(vendor="apple", name="Apple M2 Ultra", vram_bytes=68_719_476_736),
    detail="test double",
)

TOKEN = "test-token-not-minted"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "crucible-home"
    monkeypatch.setenv("CRUCIBLE_HOME", str(root))
    return root


@pytest.fixture
def make_client(home: Path) -> Callable[..., TestClient]:
    def factory(
        *,
        enable_echo: bool = True,
        enable_llm: bool = False,
        enable_asr: bool = False,
        token: str = TOKEN,
        backend: Backend = FAKE_BACKEND,
        desktop_allowance_bytes: int = 3 * 1024 ** 3,
    ) -> TestClient:
        write_config(
            home,
            name="crucible@test",
            host="127.0.0.1",
            port=7100,
            token=token,
            backend_kind=backend.kind,
            enable_echo=enable_echo,
            enable_llm=enable_llm,
            enable_asr=enable_asr,
            desktop_allowance_bytes=desktop_allowance_bytes,
        )
        config = load_config(home)
        return TestClient(create_app(config, backend))

    return factory


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> Iterator[TestClient]:
    with make_client() as instance:
        yield instance


@pytest.fixture
def auth() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {TOKEN}",
        "X-Crucible-Api": str(API_VERSION),
    }


def parse_sse(lines: Iterator[str]) -> list[dict[str, Any]]:
    """Turn an SSE byte stream's lines into [{id, event, data}] in arrival order."""
    events: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for line in lines:
        if line == "":
            if current:
                events.append(current)
                current = {}
            continue
        if line.startswith(":"):  # keepalive comment
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "id":
            current["id"] = int(value)
        elif field == "event":
            current["event"] = value
        elif field == "data":
            current["data"] = json.loads(value)
    if current:
        events.append(current)
    return events


__all__ = [
    "FAKE_BACKEND",
    "FAKE_MAC_BACKEND",
    "TOKEN",
    "parse_sse",
    "mint_token",
]
