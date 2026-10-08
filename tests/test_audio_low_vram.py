"""`[audio] low_vram`: a host whose card cannot hold YuE2 whole holds one half at a time.

Owen, 2026-10-08: "this would be a configuration for systems with low ram, not for high
ram systems like this pc. only for victoria's laptop". Off unless a host's config says
so; only a model whose manifest declares a low-VRAM figure honours it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from crucible.api import create_app
from crucible.audiomodels import AudioManifestError, load_audio_manifest, parse_audio_manifest
from crucible.config import ConfigError, load_config
from crucible.memorybudget import GIB

from .conftest import FAKE_BACKEND, configure_box
from .test_audio_api import LYRICS, SONG, TAGS, _envs, _weights, loads, run_job, transcript

__all__ = ["transcript"]

SPEC = load_audio_manifest(SONG).spec(FAKE_BACKEND.kind)


def _client(home: Path, low_vram: bool | None) -> TestClient:
    configure_box(home, enable_audio=True)
    if low_vram is not None:
        path = home / "config.toml"
        path.write_bytes(path.read_bytes() + f"\n[audio]\nlow_vram = {str(low_vram).lower()}\n".encode())
    return TestClient(create_app(load_config(home), FAKE_BACKEND))


def _song(home: Path, monkeypatch: pytest.MonkeyPatch, transcript: Path, low_vram: bool | None,
          auth: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    _envs(home, FAKE_BACKEND.kind, monkeypatch)
    _weights(home, SONG, FAKE_BACKEND.kind)
    with _client(home, low_vram) as client:
        _, events = run_job(client, auth, model=SONG, params={"tags": TAGS, "lyrics": LYRICS, "seed": 3})
    assert events[-1]["event"] == "done", events[-1]
    return loads(transcript)[0], events[-1]["data"]["audio"]


def test_yue2_declares_a_low_vram_need_below_its_whole_one() -> None:
    assert SPEC.low_vram_memory_bytes_estimate is not None
    assert SPEC.low_vram_memory_bytes_estimate < 8 * GIB < SPEC.memory_bytes_estimate
    assert "measured" in (SPEC.low_vram_memory_note or "")
    mac = load_audio_manifest(SONG).spec("mlx-darwin")
    assert mac.low_vram_memory_bytes_estimate is None


def test_low_vram_is_off_unless_the_host_says_so(
    home: Path, monkeypatch: pytest.MonkeyPatch, idle_card: None, transcript: Path, auth: dict[str, str]
) -> None:
    load, audio = _song(home, monkeypatch, transcript, None, auth)
    assert (load["low_vram"], load["memory_budget_bytes"]) == (False, SPEC.memory_bytes_estimate)
    assert (audio["low_vram"], audio["memory_bytes_estimate"]) == (False, SPEC.memory_bytes_estimate)


def test_a_low_vram_host_loads_yue2_against_its_halved_need(
    home: Path, monkeypatch: pytest.MonkeyPatch, idle_card: None, transcript: Path, auth: dict[str, str]
) -> None:
    load, audio = _song(home, monkeypatch, transcript, True, auth)
    need = SPEC.low_vram_memory_bytes_estimate
    assert (load["low_vram"], load["memory_budget_bytes"], load["memory_cap_bytes"]) == (True, need, need)
    assert (audio["low_vram"], audio["memory_bytes_estimate"]) == (True, need)


def test_a_model_with_no_low_vram_figure_ignores_the_setting(
    home: Path, monkeypatch: pytest.MonkeyPatch, idle_card: None, transcript: Path, auth: dict[str, str]
) -> None:
    from .test_audio_api import PROMPT, SFX

    _envs(home, FAKE_BACKEND.kind, monkeypatch)
    _weights(home, SFX, FAKE_BACKEND.kind)
    with _client(home, True) as client:
        _, events = run_job(client, auth, model=SFX,
                            params={"prompt": PROMPT, "duration_s": 3, "steps": 4})
    assert events[-1]["event"] == "done", events[-1]
    assert loads(transcript)[0]["low_vram"] is False


def test_the_setting_must_be_a_bool_and_survives_a_rewrite(home: Path) -> None:
    configure_box(home)
    path = home / "config.toml"
    original = path.read_bytes()
    assert load_config(home).audio_low_vram is False
    path.write_bytes(original + b"\n[audio]\nlow_vram = \"yes\"\n")
    with pytest.raises(ConfigError, match="low_vram"):
        load_config(home)
    path.write_bytes(original + b"\n[audio]\nlow_vram = true\n")
    configure_box(home)  # what `crucible install` does to the file
    assert load_config(home).audio_low_vram is True


def _note_line(text: str, replacement: str | None) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("low_vram_memory_note"):
            if replacement is not None:
                lines.append(replacement)
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda t: _note_line(t, None), "go together"),
        (lambda t: t.replace("low_vram_memory_bytes_estimate = 7_300_000_000",
                             "low_vram_memory_bytes_estimate = 16_000_000_000"), "below"),
        (lambda t: _note_line(t, 'low_vram_memory_note = "  "'), "empty"),
    ],
)
def test_a_bad_low_vram_figure_is_refused(change: Any, message: str) -> None:
    path = load_audio_manifest(SONG).path
    text = path.read_text(encoding="utf-8")
    changed = change(text)
    assert changed != text
    with pytest.raises(AudioManifestError, match=message):
        parse_audio_manifest(changed, path, SONG)
