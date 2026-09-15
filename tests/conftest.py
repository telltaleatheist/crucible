"""Shared fixtures.

Every test runs against a throwaway `CRUCIBLE_HOME` under pytest's tmp_path, so no
test can see or touch a real `~/.crucible`.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from crucible import API_VERSION, accelerator, jobenv
from crucible.accelerator import GIB
from crucible.api import create_app
from crucible.backend import Backend, Gpu
from crucible.config import load_config, mint_token, write_config
from crucible import residency as residency_module
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
        enable_tts: bool = False,
        enable_align: bool = False,
        enable_rvc: bool = False,
        enable_denoise: bool = False,
        token: str = TOKEN,
        backend: Backend = FAKE_BACKEND,
        desktop_allowance_bytes: int = 3 * 1024 ** 3,
        # None means this host has DECIDED NOTHING, which is the honest default
        # for a fixture: a config written before anything probed the card. Pass a
        # record to test a server that has decided.
        capability: Any = None,
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
            enable_tts=enable_tts,
            enable_align=enable_align,
            enable_rvc=enable_rvc,
            enable_denoise=enable_denoise,
            desktop_allowance_bytes=desktop_allowance_bytes,
            capability=capability,
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
    spec = jobenv.llm_env(FAKE_BACKEND.kind)
    directory = jobenv.env_dir(home, spec)
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
    pins = jobenv.recipe_pins(jobenv.recipe_for(spec))
    monkeypatch.setattr(jobenv, "installed_packages", lambda _home, _spec: dict(pins))
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


@contextmanager
def holding_the_card(client: TestClient, act: str = "clean") -> Iterator[None]:
    """Keep what a job left resident on the card across the job's end.

    OWEN'S RULING, 2026-09-14 (`crucible/settle.py`): the card is cleared the
    moment the last of four holders lets go. A test that wants to LOOK at what a
    job left resident is, by definition, looking after the last holder let go —
    so it has to be one.

    A chat completion in flight is the holder it takes, because it is the only
    one of the four that can be held BEFORE the job runs. A lease is the honest
    holder for a run of renders or aligns and since 2026-09-14 it can name a
    voice or an aligner (`crucible/leases.py`) — but a lease never loads, so it
    cannot be taken until the first job has made the thing resident, which is
    exactly the window a test that wants to LOOK at what one job left for the
    next is standing in. `test_tts_render.py` and `test_align_api.py` each carry
    the real-lease version of the same measurement beside the tests that use
    this.

    Nothing is faked: `Settlement.holder` reads this record through exactly the
    code path `/v1/activity` reports it from.
    """
    with client.app.state.inflight.tracked(
        act=act, model="a test holding the card", client=None
    ):
        yield


def a_clearance_to_hold(engine: FakeEngine) -> tuple[threading.Event, threading.Event]:
    """Make `engine.stop()` block, so a clearance can be caught mid-flight.

    Returns `(reached, release)`: `reached` is set once the settlement is inside
    the engine's stop — the exact state PHASE15-HOST.md section 8's T6 saw an
    `unload-model` arrive in, 2026-09-15 — and `release` lets it finish. Two
    events and no sleep, because what is being tested is a state and not a
    duration.
    """
    reached, release = threading.Event(), threading.Event()
    stop = engine.stop

    def held_stop() -> None:
        reached.set()
        assert release.wait(timeout=30), "the test never released the clearance"
        stop()

    engine.stop = held_stop  # type: ignore[method-assign]
    return reached, release


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


def wav_bytes(seconds: float, rate: int = 24000) -> bytes:
    """A real, minimal, silent mono PCM16 WAV of `seconds`.

    Real because `crucible/voicereference.py` reads the duration out of the
    RIFF header rather than taking a client's word for it, so a test that
    handed it a made-up blob would be testing the refusal path and nothing
    else. `wave` writes it, `wave` reads it: one standard-library container,
    no fixture file in the repo, and the duration is arithmetic a reader can
    check.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


def wav_base64(seconds: float, rate: int = 24000) -> str:
    """`wav_bytes`, encoded the way `params.reference.data` carries it."""
    return base64.b64encode(wav_bytes(seconds, rate)).decode("ascii")


__all__ = [
    "FAKE_BACKEND",
    "FAKE_MAC_BACKEND",
    "TOKEN",
    "holding_the_card",
    "parse_sse",
    "mint_token",
    "wav_base64",
    "wav_bytes",
]
