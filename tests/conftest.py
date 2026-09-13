"""Shared fixtures.

Every test runs against a throwaway `CRUCIBLE_HOME` under pytest's tmp_path, so no
test can see or touch a real `~/.crucible`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from crucible import API_VERSION, accelerator, llmenv
from crucible.accelerator import GIB
from crucible.api import create_app
from crucible.backend import Backend, Gpu
from crucible.config import load_config, mint_token, write_config
from crucible.jobs.llm import residency as residency_module
from crucible.manifests import load_manifest

from .fake_engine import FakeEngine

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
def make_app(home: Path) -> Callable[..., FastAPI]:
    """One configured server instance, as an ASGI app.

    Almost every test wants it wrapped in a `TestClient` (see `make_client`); the
    disconnect tests want the bare app, because they run it under a real uvicorn
    on a real socket (tests/live_server.py).
    """

    def factory(
        *,
        enable_echo: bool = True,
        enable_llm: bool = False,
        enable_asr: bool = False,
        token: str = TOKEN,
        backend: Backend = FAKE_BACKEND,
        desktop_allowance_bytes: int = 3 * 1024 ** 3,
    ) -> FastAPI:
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
        return create_app(load_config(home), backend)

    return factory


@pytest.fixture
def make_client(make_app: Callable[..., FastAPI]) -> Callable[..., TestClient]:
    def factory(**options: Any) -> TestClient:
        return TestClient(make_app(**options))

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


# ------------------------------------------------------------- the llm fixtures
#
# The env, the weights and the engine, stood up exactly as the real code paths
# read them: a stamped venv directory, a stamped weights directory, and an
# `Engine` that serves a trivial OpenAI surface on a real loopback port
# (tests/fake_engine.py). No GPU and no 19 GB of weights. Nothing about the
# server's own logic is faked — the preflight refusals, the exclusive lane, the
# event stream and the proxy are exactly what runs on the PC.


@pytest.fixture
def fake_env(home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stamped `~/.crucible/envs/llm` that `env_status` accepts."""
    directory = llmenv.llm_env_dir(home)
    (directory / "bin").mkdir(parents=True)
    (directory / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (directory / "crucible-env.json").write_text(
        json.dumps(
            {
                "backend": FAKE_BACKEND.kind,
                "recipe": f"{FAKE_BACKEND.kind}.txt",
                "python_version": "3.11.16",
                "seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    pins = llmenv.recipe_pins(llmenv.recipe_for(FAKE_BACKEND.kind))
    monkeypatch.setattr(llmenv, "installed_packages", lambda _home: dict(pins))
    return directory


@pytest.fixture
def fake_weights(home: Path) -> Callable[[str], Path]:
    """Stamp a model as pulled at exactly the revision its manifest pins."""

    def stamp(model_id: str) -> Path:
        spec = load_manifest(model_id).spec(FAKE_BACKEND.kind)
        directory = home / "models" / model_id / FAKE_BACKEND.kind
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "crucible-pull.json").write_text(
            json.dumps(
                {
                    "model": model_id,
                    "backend": FAKE_BACKEND.kind,
                    "hf_repo": spec.hf_repo,
                    "revision": spec.revision,
                    "bytes": 19_306_310_880,
                    "seconds": 300.0,
                    "pulled": "2026-09-12T19:00:00+0000",
                }
            ),
            encoding="utf-8",
        )
        return directory

    return stamp


@pytest.fixture
def idle_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (22 * GIB, 24 * GIB))


@pytest.fixture
def engine_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., list[FakeEngine]]:
    """`engines`, for a test that needs the engine configured.

    Returns the same list of built engines; the keyword arguments go to every
    `FakeEngine` the residency builds, so a test can say what the engine stops
    for or what it refuses without writing its own `build_engine` patch.
    """

    def install(**options: Any) -> list[FakeEngine]:
        built: list[FakeEngine] = []

        def build(engine_name: str, python: Path, log_path: Path) -> FakeEngine:
            engine = FakeEngine(python, log_path, **options)
            built.append(engine)
            return engine

        monkeypatch.setattr(residency_module, "build_engine", build)
        monkeypatch.setattr(
            residency_module,
            "engine_model_name",
            lambda engine_name, model_dir, model_id: model_id,
        )
        return built

    return install


# ------------------------------------------------------------------ helpers


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
