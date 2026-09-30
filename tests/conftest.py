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
from crucible.memorybudget import GIB
from crucible.api import create_app
from crucible.backend import Backend, Gpu
from crucible.config import DEFAULT_OPEN_PAIRING, load_config, mint_token, write_config
from crucible.narratorengines import declared_tts_footprints
from crucible import residency as residency_module
from crucible import engines as engines_module
from crucible.manifests import load_manifest

from .fake_engine import FakeEngine

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


def configure_box(
    home: Path,
    *,
    enable_echo: bool = True,
    enable_llm: bool = False,
    enable_asr: bool = False,
    enable_tts: bool = False,
    enable_align: bool = False,
    enable_rvc: bool = False,
    enable_denoise: bool = False,
    enable_image: bool = False,
    enable_audio: bool = False,
    enable_segment: bool = False,
    enable_video: bool = False,
    token: str = TOKEN,
    backend: Backend = FAKE_BACKEND,
    desktop_allowance_bytes: int | None = None,
    capability: Any = None,
    open_pairing: bool = DEFAULT_OPEN_PAIRING,
    tts_engines: Any = None,
) -> None:
    if desktop_allowance_bytes is None:
        desktop_allowance_bytes = (
            3 * 1024 ** 3
            if capability is None
            else capability.desktop_allowance_bytes
        )
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
        enable_image=enable_image,
        enable_audio=enable_audio,
        enable_segment=enable_segment,
        enable_video=enable_video,
        desktop_allowance_bytes=desktop_allowance_bytes,
        retention_days=7,
        desktop_allowance_basis="stated",
        desktop_allowance_note="",
        capability=capability,
        open_pairing=open_pairing,
        tts_engines=(
            declared_tts_footprints(backend.kind)
            if tts_engines is None
            else tts_engines
        ),
    )


@pytest.fixture
def make_app(home: Path) -> Callable[..., FastAPI]:

    def factory(*, backend: Backend = FAKE_BACKEND, **options: Any) -> FastAPI:
        configure_box(home, backend=backend, **options)
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


def write_env_stamp(
    home: Path,
    spec: jobenv.EnvSpec,
    backend_kind: str,
    *,
    python_version: str = "3.11.16",
    seconds: float = 1.0,
) -> Path:
    recipe = jobenv.recipe_for(spec)
    stamp = jobenv.stamp_path(home, spec)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(
        json.dumps(
            {
                "backend": backend_kind,
                "recipe": recipe.name,
                "environment_sha256": jobenv.environment_sha256(recipe),
                "direct_references": jobenv.recipe_direct_references(recipe),
                "recipe_text": jobenv.recipe_text(recipe),
                "python_version": python_version,
                "seconds": seconds,
            }
        ),
        encoding="utf-8",
    )
    return stamp


def installed_as_the_recipe_says(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        jobenv,
        "installed_packages",
        lambda _home, spec: jobenv.recipe_pins(jobenv.recipe_for(spec)),
    )
    monkeypatch.setattr(
        jobenv,
        "installed_direct_references",
        lambda _home, spec: jobenv.recipe_direct_references(jobenv.recipe_for(spec)),
    )


def stamp_env(
    home: Path,
    spec: jobenv.EnvSpec,
    backend_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    python: Path | None = None,
    python_version: str = "3.11.16",
    seconds: float = 1.0,
) -> Path:
    interpreter = jobenv.env_python(home, spec)
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    if python is None:
        interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    else:
        interpreter.symlink_to(python)
    write_env_stamp(
        home, spec, backend_kind, python_version=python_version, seconds=seconds
    )
    installed_as_the_recipe_says(monkeypatch)
    return jobenv.env_dir(home, spec)


@pytest.fixture
def fake_env(home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return stamp_env(
        home, jobenv.llm_env(FAKE_BACKEND.kind), FAKE_BACKEND.kind, monkeypatch
    )


@pytest.fixture
def fake_weights(home: Path) -> Callable[[str], Path]:

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

    def install(**options: Any) -> list[FakeEngine]:
        built: list[FakeEngine] = []

        def build(engine_name: str, python: Path, log_path: Path) -> FakeEngine:
            engine = FakeEngine(python, log_path, **options)
            built.append(engine)
            return engine

        monkeypatch.setattr(engines_module, "build_engine", build)
        monkeypatch.setattr(
            engines_module,
            "engine_model_name",
            lambda engine_name, model_dir, model_id: model_id,
        )
        return built

    return install


@contextmanager
def holding_the_card(client: TestClient, act: str = "clean") -> Iterator[None]:
    with client.app.state.inflight.tracked(
        act=act, model="a test holding the card", client=None
    ):
        yield


def a_clearance_to_hold(engine: FakeEngine) -> tuple[threading.Event, threading.Event]:
    reached, release = threading.Event(), threading.Event()
    stop = engine.stop

    def held_stop() -> None:
        reached.set()
        assert release.wait(timeout=30), "the test never released the clearance"
        stop()

    engine.stop = held_stop
    return reached, release


def parse_sse(lines: Iterator[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for line in lines:
        if line == "":
            if current:
                events.append(current)
                current = {}
            continue
        if line.startswith(":"):
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
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


def wav_base64(seconds: float, rate: int = 24000) -> str:
    return base64.b64encode(wav_bytes(seconds, rate)).decode("ascii")


__all__ = [
    "FAKE_BACKEND",
    "FAKE_MAC_BACKEND",
    "TOKEN",
    "configure_box",
    "end_process_tree",
    "holding_the_card",
    "installed_as_the_recipe_says",
    "parse_sse",
    "mint_token",
    "stamp_env",
    "wav_base64",
    "wav_bytes",
    "write_env_stamp",
]


@pytest.fixture(autouse=True)
def _never_the_real_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRUCIBLE_HOME", str(tmp_path / "unconfigured-home"))
    monkeypatch.delenv("HF_TOKEN", raising=False)


@pytest.fixture(autouse=True)
def _state_the_systemd_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from crucible import service

    monkeypatch.setattr(service, "in_wsl", lambda: False)
    monkeypatch.setattr(service, "SYSTEM_UNIT_DIR", tmp_path / "etc-systemd-system")


_CHILD_GRACE_SECONDS = 5.0


def end_process_tree(pid: int) -> None:
    import os
    import signal
    import subprocess
    import sys

    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
        return
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


@pytest.fixture(autouse=True)
def _no_child_outlives_its_test(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    import subprocess
    import time

    spawned: list[subprocess.Popen[Any]] = []
    real = subprocess.Popen

    class _Recorded(real):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            spawned.append(self)

    monkeypatch.setattr(subprocess, "Popen", _Recorded)
    yield
    monkeypatch.setattr(subprocess, "Popen", real)

    deadline = time.monotonic() + _CHILD_GRACE_SECONDS
    survivors: list[subprocess.Popen[Any]] = []
    for process in spawned:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            survivors.append(process)
    if not survivors:
        return
    named = []
    for process in survivors:
        end_process_tree(process.pid)
        argv = process.args if isinstance(process.args, (list, tuple)) else [process.args]
        named.append(f"pid {process.pid}: {' '.join(str(part) for part in argv)[:300]}")
    pytest.fail(
        f"{len(survivors)} child process(es) outlived this test and were ended by "
        "the reaper. Whatever started them owns stopping them:\n  "
        + "\n  ".join(named),
        pytrace=False,
    )


class HubUnreachableInTests(RuntimeError):
    pass


PINNED_VOICE_MANIFESTS = Path(__file__).resolve().parent / "fixtures" / "voice-manifests"


def _hub_is_offline(*args: Any, **kwargs: Any) -> Any:
    raise HubUnreachableInTests(
        "the test suite never reaches huggingface.co; fake the fetch with "
        "tests/fake_hub.py or monkeypatch huggingface_hub in this test"
    )


def _pinned_manifest_or_offline(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    local_dir: str | None = None,
    **kwargs: Any,
) -> str:
    source = (
        PINNED_VOICE_MANIFESTS / repo_id.replace("/", "--") / str(revision) / filename
    )
    if local_dir is None or not source.is_file():
        raise HubUnreachableInTests(
            f"the test suite never reaches huggingface.co; {repo_id}@{revision} "
            f"{filename} is not in {PINNED_VOICE_MANIFESTS}. A pin moved: copy that "
            "revision's manifest there, or fake the fetch in this test"
        )
    target = Path(local_dir) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    return str(target)


PINNED_VOICE_TAG = "crucible"


def _fixture_voice_refs() -> dict[tuple[str, str], str]:
    refs: dict[tuple[str, str], str] = {}
    for repo_dir in sorted(PINNED_VOICE_MANIFESTS.iterdir()):
        shas = sorted(p.name for p in repo_dir.iterdir() if p.is_dir())
        if len(shas) == 1:
            refs[(repo_dir.name.replace("--", "/", 1), PINNED_VOICE_TAG)] = shas[0]
    return refs


FIXTURE_VOICE_REFS = _fixture_voice_refs()

HUB_REFS: dict[tuple[str, str], str] = {}


def _ref_or_offline(self: Any, repo_id: str, *args: Any, revision: str | None = None, **kwargs: Any) -> Any:
    from types import SimpleNamespace

    sha = HUB_REFS.get((repo_id, str(revision)))
    if sha is None:
        return _hub_is_offline()
    return SimpleNamespace(id=repo_id, sha=sha)


def cached_as_of_the_fixtures(home: Path, hf_repo: str, ref: str) -> Any:
    from crucible import voicerefs

    found = voicerefs.read_checks(home).get(f"{hf_repo}@{ref}")
    if found is not None:
        return found
    sha = FIXTURE_VOICE_REFS.get((hf_repo, ref))
    if sha is None:
        return None
    return voicerefs.RefCheck(
        hf_repo=hf_repo, ref=ref, revision=sha, checked_at="fixture", error=None
    )


@pytest.fixture(autouse=True)
def _no_test_reaches_the_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    from crucible import voicerefs

    HUB_REFS.clear()
    HUB_REFS.update(FIXTURE_VOICE_REFS)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _pinned_manifest_or_offline)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _hub_is_offline)
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", _ref_or_offline)
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", _hub_is_offline)
    monkeypatch.setattr(voicerefs, "cached", cached_as_of_the_fixtures)
