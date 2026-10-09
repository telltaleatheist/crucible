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
    # Crucible would have turned it on itself, so this is a host where a person turned
    # it off: the fix is the setter, not a hand edit and a restart.
    assert "`crucible audio low-vram on`" in song.reason
    assert "`crucible audio low-vram auto`" in song.reason
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
    # stable-audio-3-medium declares no low-VRAM figure: 8.0 GB (7.45 GiB) against 7.0 GiB.
    medium = load_audio_manifest("stable-audio-3-medium").spec(FAKE_BACKEND.kind)
    for low_vram in (False, True):
        music = _verdict("music", low_vram)
        assert music.enabled is False
        assert music.shortfall_bytes == medium.memory_bytes_estimate - 7 * GIB
        assert "short by 0.5 GiB" in music.reason
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


# --- Crucible turns it on where the card needs it, and only there ---------------------------
#
# A friend's 8 GiB RTX 3070 laptop, 2026-10-08: low_vram had to be turned on by hand inside a
# WSL distro its owner did not know existed. Owen's ruling of the same day stands: it stays
# off on a card that holds YuE2 whole (his 3090 Ti, the Mac), and a person's own setting is
# never changed.

BIG_CARD = Backend(
    kind="cuda-linux",
    platform="linux",
    arch="x86_64",
    gpu=Gpu(vendor="nvidia", name="NVIDIA GeForce RTX 3090 Ti", vram_bytes=24 * GIB),
    detail="test double",
)


def _audio_table(home: Path) -> dict[str, Any] | None:
    return tomllib.loads((home / "config.toml").read_text(encoding="utf-8")).get("audio")


def _decided_box(home: Path, backend: Backend, low_vram: bool | None) -> Config:
    """A box with a record decided the way 1.0.114 decided it (the setting as the file
    said), and `[audio] low_vram` as a person left it (None: never touched)."""
    decisions = capabilitystore.decide_on(
        "cuda-linux", total_bytes=backend.gpu.vram_bytes, desktop_allowance_bytes=GIB,
        gpu_vendor="nvidia", card=None, chosen={}, audio_low_vram=bool(low_vram),
    )
    record = capabilitystore.record_of(
        "cuda-linux", total_bytes=backend.gpu.vram_bytes, desktop_allowance_bytes=GIB,
        decisions=decisions, routes={},
    )
    configure_box(home, enable_audio=True, backend=backend, capability=record,
                  desktop_allowance_bytes=GIB)
    if low_vram is not None:
        _set_low_vram(home, low_vram)
    return load_config(home)


def _step(home: Path, backend: Backend, capsys: pytest.CaptureFixture[str]) -> str:
    from crucible.cli import capability as capability_cli

    capability_cli._capability_step(load_config(home), backend, "audio")
    return capsys.readouterr().out


def test_an_eight_gig_card_gets_low_vram_turned_on_and_is_told_so(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _decided_box(home, LAPTOP, None)
    out = _step(home, LAPTOP, capsys)
    assert (
        "Crucible turned [audio] low_vram on: yue2-3b needs 14.9 GiB whole and this card "
        "gives a job 7.0 GiB, so it now holds only the part each stage uses (6.8 GiB)"
    ) in out
    assert "`crucible audio low-vram off` turns it off" in out
    assert _audio_table(home) == {"low_vram": True, "low_vram_auto": True}
    config = load_config(home)
    assert (config.audio_low_vram, config.audio_low_vram_auto) == (True, True)
    song = config.capability.row("song")
    assert song.enabled is True and "with [audio] low_vram" in song.reason
    assert config.enable_audio is True
    # Decided again with nothing changed, it is said once, not every time.
    assert "Crucible turned" not in _step(home, LAPTOP, capsys)


def test_a_card_that_holds_yue2_whole_stays_off_and_its_file_untouched(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _decided_box(home, BIG_CARD, None)
    out = _step(home, BIG_CARD, capsys)
    assert "Crucible turned" not in out
    assert _audio_table(home) is None
    config = load_config(home)
    assert (config.audio_low_vram, config.audio_low_vram_auto) == (False, True)
    assert config.capability.row("song").enabled is True
    assert "low_vram" not in config.capability.row("song").reason


def test_a_persons_off_is_never_turned_on(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _decided_box(home, LAPTOP, False)
    out = _step(home, LAPTOP, capsys)
    assert "Crucible turned" not in out
    assert _audio_table(home) == {"low_vram": False}
    config = load_config(home)
    assert (config.audio_low_vram, config.audio_low_vram_auto) == (False, False)
    song = config.capability.row("song")
    assert song.enabled is False
    assert "`crucible audio low-vram on`" in song.reason


def test_a_persons_on_stays_on_a_card_that_does_not_need_it(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _decided_box(home, BIG_CARD, True)
    assert "Crucible turned" not in _step(home, BIG_CARD, capsys)
    assert _audio_table(home) == {"low_vram": True}
    words = capabilitystore.low_vram_for(load_config(home), BIG_CARD).words
    assert "set by hand" in words and "`crucible audio low-vram auto`" in words


def test_crucibles_own_on_comes_off_on_a_card_that_holds_it_whole(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _decided_box(home, LAPTOP, None)
    _step(home, LAPTOP, capsys)
    assert _audio_table(home) == {"low_vram": True, "low_vram_auto": True}
    # The same config on a 24 GiB card (the drive moved into a desktop): decided again.
    path = home / "config.toml"
    text = path.read_text(encoding="utf-8").replace(
        f"total_bytes = {LAPTOP.gpu.vram_bytes}", f"total_bytes = {BIG_CARD.gpu.vram_bytes}"
    )
    path.write_text(text, encoding="utf-8")
    out = _step(home, BIG_CARD, capsys)
    assert "Crucible turned [audio] low_vram off, which it had turned on itself" in out
    assert "yue2-3b fits whole (14.9 GiB of 23.0 GiB)" in out
    assert _audio_table(home) is None
    assert load_config(home).audio_low_vram is False


def test_the_other_keys_of_audio_survive_crucibles_writes(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _decided_box(home, LAPTOP, None)
    path = home / "config.toml"
    path.write_text(path.read_text(encoding="utf-8") + '\n[audio]\nnote = "mine"\n',
                    encoding="utf-8")
    _step(home, LAPTOP, capsys)
    assert _audio_table(home) == {"note": "mine", "low_vram": True, "low_vram_auto": True}


def test_low_vram_auto_without_the_setting_is_refused_by_name(home: Path) -> None:
    configure_box(home)
    path = home / "config.toml"
    path.write_bytes(path.read_bytes() + b"\n[audio]\nlow_vram_auto = true\n")
    with pytest.raises(ConfigError, match="crucible audio low-vram auto"):
        load_config(home)


def test_capability_write_turns_it_on_and_the_dry_run_says_it_would(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from crucible import cli

    _decided_box(home, LAPTOP, None)
    monkeypatch.setattr(cli.common, "detect_backend", lambda: LAPTOP)
    assert cli.main(["capability"]) == 0
    dry = capsys.readouterr().out
    assert "[audio] low_vram is recorded off, and Crucible would turn it on" in dry
    assert _audio_table(home) is None
    assert cli.main(["capability", "--write"]) == 0
    assert "Crucible turned [audio] low_vram on" in capsys.readouterr().out
    assert _audio_table(home) == {"low_vram": True, "low_vram_auto": True}


def test_the_server_deciding_a_card_with_no_record_turns_it_on(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from crucible.api.app import _decider

    configure_box(home, enable_audio=True, backend=LAPTOP, desktop_allowance_bytes=GIB)
    config = load_config(home)
    _decider(config, LAPTOP)("audio")
    assert "Crucible turned [audio] low_vram on" in capsys.readouterr().err
    assert config.audio_low_vram is True
    assert config.capability is not None and config.capability.row("song").enabled is True


def test_a_settings_allowance_that_squeezes_yue2_turns_it_on_with_the_record(
    home: Path,
) -> None:
    config = _decided_box(home, BIG_CARD, None)
    allowance = BIG_CARD.gpu.vram_bytes - 8 * GIB  # leaves 8 GiB: YuE2 whole does not fit
    resolved = settings.resolve(config, {"desktop_allowance_bytes": allowance})
    assert any(line.startswith("Crucible turned [audio] low_vram on") for line in resolved.changed)
    settings.apply(config, resolved, gpu_vendor="nvidia", card=None)
    assert _audio_table(home) == {"low_vram": True, "low_vram_auto": True}
    assert config.audio_low_vram is True
    assert config.capability.row("song").enabled is True


# --- What doctor and Settings say in each state ----------------------------------------------


def _doctor(home: Path) -> dict[str, Any]:
    host = doctor.survey(home)
    return doctor.assemble(home, [doctor.check_capability(doctor.Host(
        home, LAPTOP, None, host.config, host.config_refusal, None
    ))])


def test_doctor_names_the_setting_in_each_state(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _decided_box(home, LAPTOP, None)
    (problem,) = _doctor(home)["problems"]
    assert problem.startswith(
        "capability_stale: [audio] low_vram is recorded off, and Crucible would turn it on"
    )
    assert "`crucible capability --write` does that" in problem

    _step(home, LAPTOP, capsys)
    auto_on = _doctor(home)
    assert auto_on["problems"] == []
    entry = auto_on["audio_low_vram"]
    assert (entry["on"], entry["set_by"], entry["card"]["verdict"]) == (True, "crucible", "needed")
    (said,) = [line for line in doctor.lines_capability(auto_on) if line.startswith("audio:")]
    assert "[audio] low_vram is on, because this card needs it" in said
    assert "Crucible turned it on" in said and "`crucible audio low-vram off`" in said

    # A person's on: the hand edit that drops low_vram_auto makes it theirs.
    path = home / "config.toml"
    path.write_text(path.read_text(encoding="utf-8").replace("low_vram_auto = true\n", ""),
                    encoding="utf-8")
    by_hand = _doctor(home)
    assert by_hand["problems"] == []
    assert (by_hand["audio_low_vram"]["on"], by_hand["audio_low_vram"]["set_by"]) == (True, "person")
    assert "is on, set by hand" in by_hand["audio_low_vram"]["words"]

    _set_low_vram(home, False)
    off = _doctor(home)["audio_low_vram"]
    assert (off["on"], off["set_by"]) == (False, "person")
    assert "`crucible audio low-vram on` or `crucible audio low-vram auto` turns it back on" in (
        off["words"]
    )


def test_settings_show_the_state_and_what_the_card_makes_of_it(home: Path) -> None:
    big = _decided_box(home, BIG_CARD, None)
    installed = {c: False for c in _every_candidate()}
    entry = settings.document(big, installed=installed)["audio_low_vram"]
    assert (entry["on"], entry["state"], entry["set_by"], entry["pending"]) == (
        False, "auto", "crucible", False
    )
    assert entry["card"]["verdict"] == "not_needed"
    assert entry["card"]["models"] == [{
        "id": SONG, "whole_bytes": SPEC.memory_bytes_estimate,
        "low_bytes": SPEC.low_vram_memory_bytes_estimate,
    }]
    small = settings.document(_decided_box(home, LAPTOP, False), installed=installed)[
        "audio_low_vram"
    ]
    assert (small["on"], small["state"], small["set_by"]) == (False, "off", "person")
    assert small["card"]["verdict"] == "needed" and small["card"]["needing"] == [SONG]


def test_settings_offer_nothing_where_no_model_can_be_split(home: Path) -> None:
    from .conftest import FAKE_MAC_BACKEND

    total = FAKE_MAC_BACKEND.gpu.vram_bytes
    decisions = capabilitystore.decide_on(
        "mlx-darwin", total_bytes=total, desktop_allowance_bytes=GIB,
        gpu_vendor="apple", card=None, chosen={}, audio_low_vram=False,
    )
    record = capabilitystore.record_of(
        "mlx-darwin", total_bytes=total, desktop_allowance_bytes=GIB,
        decisions=decisions, routes={},
    )
    configure_box(home, backend=FAKE_MAC_BACKEND, capability=record, desktop_allowance_bytes=GIB)
    assert settings.low_vram_entry(load_config(home)) is None


# --- The Settings route: the same door as `crucible audio low-vram` --------------------------


def _laptop_client(home: Path, low_vram: bool | None) -> TestClient:
    return TestClient(create_app(_decided_box(home, LAPTOP, low_vram), LAPTOP))


def test_the_route_sets_it_the_way_the_cli_does(home: Path, auth: dict[str, str]) -> None:
    door = "/v1/settings/audio/low-vram"
    with _laptop_client(home, None) as client:
        before = client.get("/v1/settings", headers=auth).json()["audio_low_vram"]
        assert (before["on"], before["pending"]) == (False, True)

        off = client.put(door, headers=auth, json={"state": "off"})
        assert off.status_code == 200, off.text
        entry = off.json()["audio_low_vram"]
        assert (entry["on"], entry["state"], entry["set_by"]) == (False, "off", "person")
        assert _audio_table(home) == {"low_vram": False}
        assert load_config(home).capability.row("song").enabled is False

        auto = client.put(door, headers=auth, json={"state": "auto"})
        entry = auto.json()["audio_low_vram"]
        assert (entry["on"], entry["state"], entry["set_by"]) == (True, "auto", "crucible")
        assert _audio_table(home) == {"low_vram": True, "low_vram_auto": True}
        assert load_config(home).capability.row("song").enabled is True

        on = client.put(door, headers=auth, json={"state": "on"})
        assert on.json()["audio_low_vram"]["state"] == "on"
        assert _audio_table(home) == {"low_vram": True}
        # The server follows the file on the next request.
        assert client.get("/v1/settings", headers=auth).json()["audio_low_vram"]["state"] == "on"
        rows = client.get("/v1/activity", headers=auth).json()["settings"]["writes"]
        assert rows[0]["changed"] == ["[audio] low_vram = on"]
        assert any(
            line.startswith("Crucible turned [audio] low_vram on")
            for row in rows for line in row["changed"]
        )


@pytest.mark.parametrize("body", [{"state": "yes"}, {"state": "on", "also": 1}, {}, ["on"], "on"])
def test_the_route_refuses_a_body_it_cannot_read_by_name(
    home: Path, auth: dict[str, str], body: Any
) -> None:
    with _laptop_client(home, None) as client:
        refused = client.put("/v1/settings/audio/low-vram", headers=auth, json=body)
    assert refused.status_code == 400
    assert refused.json()["error"]["code"] == "invalid_request"
    assert "'auto'" in refused.json()["error"]["message"]
    assert _audio_table(home) is None


def test_the_route_refuses_on_where_no_model_can_be_split(
    home: Path, auth: dict[str, str]
) -> None:
    from .conftest import FAKE_MAC_BACKEND

    configure_box(home, backend=FAKE_MAC_BACKEND)
    with TestClient(create_app(load_config(home), FAKE_MAC_BACKEND)) as client:
        refused = client.put("/v1/settings/audio/low-vram", headers=auth, json={"state": "on"})
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "low_vram_not_offered"
    assert _audio_table(home) is None


def test_the_page_draws_the_switch_from_the_settings_document() -> None:
    from crucible.api import UI_DIR

    script = (UI_DIR / "app.js").read_text(encoding="utf-8")
    assert "'/v1/settings/audio/low-vram'" in script
    for key in ("entry.set_by", "entry.card.verdict", "entry.words", "entry.state"):
        assert key in script, key
    for verdict_name in ("needed", "not_needed", "too_small"):
        assert f"{verdict_name}: [" in script, verdict_name
