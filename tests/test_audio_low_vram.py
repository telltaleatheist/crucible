"""`[audio] low_vram`: a host whose card cannot hold YuE2 whole holds one half at a time.

Owen, 2026-10-08: "this would be a configuration for systems with low ram, not for high
ram systems like this pc. only for victoria's laptop". Off unless a host's config says
so; only a model whose manifest declares a low-VRAM figure honours it.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from crucible import capabilitystore, installplan, settings, verdict
from crucible.api import create_app
from crucible.audiomodels import (
    AudioManifestError,
    HeldNeed,
    load_audio_manifest,
    parse_audio_manifest,
)
from crucible.backend import Backend, Gpu
from crucible.capabilityclasses import BY_NAME, CLASSES
from crucible.cli import doctor
from crucible.config import Config, ConfigError, load_config
from crucible.errors import ApiError
from crucible.jobs import audio as audio_job
from crucible.memorybudget import GIB
from crucible.residency import Residency

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


# --- The capability verdict weighs the need the audio job admits against -------------------
#
# Victoria's laptop, 2026-10-08: an 8 GiB RTX 3070 with low_vram on made songs (peak 5.25
# GiB) while `crucible install audio` recorded "capability song: NO ... yue2-3b at 14.9
# GiB", and the done event said 7.3 GB. One rule now names the need for both.

LAPTOP = Backend(
    kind="cuda-linux",
    platform="linux",
    arch="x86_64",
    gpu=Gpu(vendor="nvidia", name="NVIDIA GeForce RTX 3070 Laptop GPU", vram_bytes=8 * GIB),
    detail="test double",
)


def _verdict(name: str, low_vram: bool, chosen: str | None = None) -> verdict.Decision:
    return verdict.decide(
        BY_NAME[name], "cuda-linux", total_bytes=8 * GIB, desktop_allowance_bytes=GIB,
        gpu_vendor="nvidia", chosen=chosen, audio_low_vram=low_vram,
    )


def test_one_rule_names_the_need_for_the_job_and_the_verdict() -> None:
    assert SPEC.need_on(True) == HeldNeed(SPEC.low_vram_memory_bytes_estimate, True)
    assert SPEC.need_on(False) == HeldNeed(SPEC.memory_bytes_estimate, False)
    sfx = load_audio_manifest("stable-audio-3-small-sfx").spec(FAKE_BACKEND.kind)
    assert sfx.need_on(True) == HeldNeed(sfx.memory_bytes_estimate, False)
    (song,) = _verdict("song", True).candidates
    assert song.memory_bytes_estimate == SPEC.need_on(True).bytes
    assert song.whole_bytes == SPEC.memory_bytes_estimate


def test_an_eight_gig_card_with_low_vram_makes_songs() -> None:
    song = _verdict("song", True)
    assert song.enabled is True, song.reason
    assert song.selected == SONG
    assert "[audio] low_vram" in song.reason
    assert "6.8 GiB with [audio] low_vram" in song.reason
    assert "(14.9 GiB whole)" in song.reason
    assert "one part of it on the card at a time" in song.summary
    assert "short by" not in song.reason


def test_an_eight_gig_card_without_low_vram_is_told_to_turn_it_on() -> None:
    song = _verdict("song", False)
    assert song.enabled is False
    assert song.shortfall_bytes == SPEC.memory_bytes_estimate - 7 * GIB
    assert "short by 7.9 GiB" in song.reason
    assert "[audio] low_vram is off on this host" in song.reason
    assert "`[audio] low_vram = true`" in song.reason
    assert "crucible capability --write" in song.reason
    assert "[audio] low_vram" in song.summary
    assert "more memory than this machine has free" not in song.summary


def test_a_chosen_song_model_names_the_setting_too() -> None:
    chosen = _verdict("song", False, chosen=SONG)
    assert chosen.enabled is False
    assert "[audio] low_vram is off on this host" in chosen.reason
    assert "This choice fit the machine it was made on" not in chosen.reason
    assert _verdict("song", True, chosen=SONG).enabled is True


def test_low_vram_that_still_does_not_fit_says_it_is_already_on() -> None:
    song = verdict.decide(
        BY_NAME["song"], "cuda-linux", total_bytes=6 * GIB, desktop_allowance_bytes=GIB,
        gpu_vendor="nvidia", chosen=None, audio_low_vram=True,
    )
    assert song.enabled is False
    assert "6.8 GiB with [audio] low_vram" in song.reason
    assert "is off on this host" not in song.reason
    assert song.shortfall_bytes == SPEC.low_vram_memory_bytes_estimate - 5 * GIB


def test_music_on_the_same_card_is_weighed_whole_with_or_without_the_setting() -> None:
    # stable-audio-3-medium declares no low-VRAM figure, so the setting changes nothing for
    # it: measured 6.8 GB (6.3 GiB) at 380 s fits 7.0 GiB whole either way.
    medium = load_audio_manifest("stable-audio-3-medium").spec(FAKE_BACKEND.kind)
    assert medium.low_vram_memory_bytes_estimate is None
    assert medium.memory_bytes_estimate < 7 * GIB
    for low_vram in (False, True):
        music = _verdict("music", low_vram)
        assert music.enabled is True
        assert "low_vram" not in music.reason + music.summary


def test_the_install_plan_says_songs_will_be_made(home: Path) -> None:
    decisions = verdict.decide_all(
        "cuda-linux", total_bytes=8 * GIB, desktop_allowance_bytes=GIB,
        gpu_vendor="nvidia", chosen={}, audio_low_vram=True,
    )
    plan = installplan.install_plan(
        "audio", decisions, card=None, total_bytes=8 * GIB, pool="card"
    )
    rows = {row["capability"]: row for row in plan["classes"]}
    assert rows["song"]["enabled"] is True
    assert rows["song"]["line"].startswith(f"Will make songs with vocals with {SONG}")
    pulled = installplan.subject_plan(SONG, decisions, card=None, total_bytes=8 * GIB, pool="card")
    assert pulled["usable"] is True


def _set_low_vram(home: Path, low_vram: bool) -> None:
    """`crucible install` carries [audio] through a rewrite, so set it in place."""
    path = home / "config.toml"
    text = path.read_text(encoding="utf-8")
    line = f"low_vram = {str(low_vram).lower()}"
    if "audio" in tomllib.loads(text):
        lines = text.splitlines()
        lines = [line if row.startswith("low_vram = ") else row for row in lines]
        text = "\n".join(lines) + "\n"
    else:
        text += f"\n[audio]\n{line}\n"
    path.write_text(text, encoding="utf-8")


def _laptop_config(home: Path, low_vram: bool) -> Config:
    configure_box(home, enable_audio=True, backend=LAPTOP, desktop_allowance_bytes=GIB)
    _set_low_vram(home, low_vram)
    return load_config(home)


def test_the_recorded_capability_follows_the_host_setting(home: Path) -> None:
    for low_vram, enabled in ((True, True), (False, False)):
        config = _laptop_config(home, low_vram)
        decided = {d.capability: d for d in capabilitystore.decide_for(config, LAPTOP)}
        assert decided["song"].enabled is enabled, decided["song"].reason
        assert "[audio] low_vram" in decided["song"].reason


def test_the_catalog_and_the_settings_choices_name_the_held_need(home: Path) -> None:
    config = _laptop_config(home, True)
    registry_type = audio_job.AudioJobType(config, LAPTOP, Residency(config))
    assert registry_type.vram_estimate(SONG) == SPEC.low_vram_memory_bytes_estimate
    rows = {row.id: row for row in registry_type.describe_models()}
    assert rows[SONG].vram_bytes == SPEC.low_vram_memory_bytes_estimate
    off = _laptop_config(home, False)
    assert audio_job.AudioJobType(off, LAPTOP, Residency(off)).vram_estimate(SONG) == (
        SPEC.memory_bytes_estimate
    )


def _with_record(home: Path, low_vram_decided: bool, low_vram_now: bool) -> Config:
    decisions = capabilitystore.decide_on(
        "cuda-linux", total_bytes=8 * GIB, desktop_allowance_bytes=GIB, gpu_vendor="nvidia",
        card=None, chosen={}, audio_low_vram=low_vram_decided,
    )
    record = capabilitystore.record_of(
        "cuda-linux", total_bytes=8 * GIB, desktop_allowance_bytes=GIB,
        decisions=decisions, routes={},
    )
    configure_box(home, enable_audio=True, backend=LAPTOP, capability=record)
    _set_low_vram(home, low_vram_now)
    return load_config(home)


def test_choosing_yue2_on_a_small_card_without_low_vram_names_the_setting(home: Path) -> None:
    config = _with_record(home, False, False)
    with pytest.raises(ApiError) as refused:
        settings.resolve(config, {"local_models": {"song": SONG}})
    assert refused.value.code == "local_model_does_not_fit"
    assert "[audio] low_vram is off on this host" in refused.value.message
    on = _with_record(home, True, True)
    settings.resolve(on, {"local_models": {"song": SONG}})
    choices = settings.document(on, installed={c: False for c in _every_candidate()})
    (song,) = choices["local_model_choices"]["song"]
    assert (song["memory_bytes_estimate"], song["fits"], song["low_vram"]) == (
        SPEC.low_vram_memory_bytes_estimate, True, True
    )


def _every_candidate() -> set[str]:
    return {
        c.id
        for entry in CLASSES
        if entry.candidates is not None
        for c in entry.candidates("cuda-linux")
    }


def test_doctor_says_a_record_decided_without_low_vram_is_stale(home: Path) -> None:
    config = _with_record(home, False, True)
    (finding,) = doctor._low_vram_findings(config, LAPTOP)
    assert finding.code == "capability_stale"
    assert "the record says song: NO" in finding.message
    assert "with [audio] low_vram on this host decides yes" in finding.message
    assert finding.fix == "crucible capability --write"
    assert doctor._low_vram_findings(_with_record(home, True, True), LAPTOP) == []
