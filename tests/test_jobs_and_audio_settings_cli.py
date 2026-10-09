from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from crucible import cli
from crucible.backend import Backend, Gpu
from crucible.cli import jobs_cmd
from crucible.config import config_path, load_config

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND

SIX_GIG = 6 * 1024**3


@pytest.fixture
def viable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)


@pytest.fixture
def tiny_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli.common,
        "detect_backend",
        lambda: Backend(
            kind="cuda-linux",
            platform="linux",
            arch="x86_64",
            gpu=Gpu(vendor="nvidia", name="NVIDIA T1000", vram_bytes=SIX_GIG),
            detail="test double",
        ),
    )


def _document(home: Path) -> dict:
    return tomllib.loads(config_path(home).read_text(encoding="utf-8"))


def _with_extra_tables(home: Path) -> None:
    path = config_path(home)
    path.write_text(
        path.read_text(encoding="utf-8")
        + '\n[hf]\ntoken = "hf_kept"\n\n[audio]\nnote = "a person wrote this"\n',
        encoding="utf-8",
    )


def _decided(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["capability", "--write"]) == 0
    capsys.readouterr()


@pytest.fixture
def envs_built(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs_cmd, "env_built", lambda _config, _backend, _type: True)


def test_jobs_enable_turns_one_flag_on_and_keeps_the_token_and_every_other_table(
    home: Path, viable: None, envs_built: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init"]) == 0
    _decided(capsys)
    _with_extra_tables(home)
    before = load_config(home)

    assert cli.main(["jobs", "enable", "audio"]) == 0
    out = capsys.readouterr().out
    assert "[jobs] enable_audio = true" in out
    assert "takes this up by itself" in out

    after = load_config(home)
    assert after.enable_audio is True
    assert after.token == before.token
    assert after.capability == before.capability
    assert after.enable_tts == before.enable_tts
    document = _document(home)
    assert document["hf"] == {"token": "hf_kept"}
    assert document["audio"] == {"note": "a person wrote this"}


def test_jobs_enable_refuses_a_type_whose_env_is_not_built_and_names_the_install(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init"]) == 0
    _decided(capsys)
    before = config_path(home).read_bytes()
    assert cli.main(["jobs", "enable", "audio"]) == 1
    said = capsys.readouterr().err
    assert "env_not_built" in said
    assert "`crucible install audio` builds the env and turns" in said
    assert config_path(home).read_bytes() == before


def test_jobs_disable_turns_it_off_and_says_how_to_turn_it_back_on(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-audio"]) == 0
    capsys.readouterr()
    token = load_config(home).token
    assert cli.main(["jobs", "disable", "audio"]) == 0
    assert "`crucible jobs enable audio` turns it back on" in capsys.readouterr().out
    assert load_config(home).enable_audio is False
    assert load_config(home).token == token
    assert cli.main(["jobs", "disable", "audio"]) == 0
    assert "already false" in capsys.readouterr().out


def test_jobs_enable_refuses_by_name_when_nothing_has_decided_the_card(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["jobs", "enable", "audio"]) == 1
    said = capsys.readouterr().err
    assert "job_type_undecided" in said
    assert "crucible capability --write" in said
    assert load_config(home).enable_audio is False


def test_jobs_enable_refuses_a_type_this_card_cannot_hold_and_writes_nothing(
    home: Path, tiny_card: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init"]) == 0
    _decided(capsys)
    before = config_path(home).read_bytes()
    assert cli.main(["jobs", "enable", "video"]) == 1
    said = capsys.readouterr().err
    assert "job_type_cannot_hold" in said
    assert "Nothing was written" in said
    assert config_path(home).read_bytes() == before


def test_jobs_list_says_each_type_s_flag_fit_and_env(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-echo"]) == 0
    _decided(capsys)
    assert cli.main(["jobs", "list", "--json"]) == 0
    rows = {row["job_type"]: row for row in json.loads(capsys.readouterr().out)["job_types"]}
    assert rows["echo"]["enabled"] is True and rows["echo"]["env_built"] is None
    assert rows["audio"]["enabled"] is False
    assert rows["audio"]["fits"] is True
    assert rows["audio"]["env_built"] is False


def test_audio_low_vram_on_sets_the_one_key_and_keeps_the_token_and_the_rest_of_audio(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-audio"]) == 0
    _with_extra_tables(home)
    capsys.readouterr()
    token = load_config(home).token

    assert cli.main(["audio", "low-vram", "on"]) == 0
    out = capsys.readouterr().out
    assert "[audio] low_vram = true" in out
    assert "yue2-3b" in out
    # The audio verdict is decided again by the setter, not left stale for a later
    # `crucible capability --write` (Victoria's laptop, 2026-10-08).
    assert "capability:" in out and "recorded in" in out
    assert "low_vram" in out.split("capability:", 1)[1]
    config = load_config(home)
    assert config.audio_low_vram is True
    assert config.token == token
    assert config.enable_audio is True
    document = _document(home)
    assert document["audio"] == {"note": "a person wrote this", "low_vram": True}
    assert document["hf"] == {"token": "hf_kept"}

    assert cli.main(["audio", "low-vram"]) == 0
    assert "[audio] low_vram is on" in capsys.readouterr().out
    assert cli.main(["audio", "low-vram", "off"]) == 0
    capsys.readouterr()
    assert load_config(home).audio_low_vram is False


def test_a_later_install_keeps_what_the_setter_wrote(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init"]) == 0
    assert cli.main(["audio", "low-vram", "on"]) == 0
    capsys.readouterr()
    config = load_config(home)
    assert cli.capability._capability_step(config, FAKE_BACKEND, "audio") == 0
    after = load_config(home)
    assert after.audio_low_vram is True
    assert after.enable_audio is True, "an install that fits turns its type on"


def test_audio_low_vram_is_refused_where_no_model_can_be_split(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_MAC_BACKEND)
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["audio", "low-vram", "on"]) == 1
    assert "low_vram_not_offered" in capsys.readouterr().err
    assert load_config(home).audio_low_vram is False


EIGHT_GIG = Backend(
    kind="cuda-linux",
    platform="linux",
    arch="x86_64",
    gpu=Gpu(vendor="nvidia", name="NVIDIA GeForce RTX 3070 Laptop GPU", vram_bytes=8 * 1024**3),
    detail="test double",
)


@pytest.fixture
def laptop(home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Victoria's laptop as `crucible install` left it: a measured 1 GiB desktop, the
    card decided, and nobody has touched [audio]."""
    from .conftest import configure_box

    monkeypatch.setattr(cli.common, "detect_backend", lambda: EIGHT_GIG)
    configure_box(home, enable_audio=True, backend=EIGHT_GIG, desktop_allowance_bytes=1024**3)
    _decided(capsys)


def test_the_card_decision_turned_it_on_and_the_setter_says_who_did(
    home: Path, laptop: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _document(home)["audio"] == {"low_vram": True, "low_vram_auto": True}
    assert cli.main(["audio", "low-vram"]) == 0
    out = capsys.readouterr().out
    assert "[audio] low_vram is on in" in out
    assert "because this card needs it" in out and "Crucible turned it on" in out


def test_off_by_hand_on_a_card_that_needs_it_stays_off(
    home: Path, laptop: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["audio", "low-vram", "off"]) == 0
    out = capsys.readouterr().out
    assert "[audio] low_vram = false" in out
    assert "set by a person" in out
    assert _document(home)["audio"] == {"low_vram": False}
    _decided(capsys)  # the card decided again does not undo a person's off
    assert _document(home)["audio"] == {"low_vram": False}
    assert load_config(home).capability.row("song").enabled is False

    assert cli.main(["audio", "low-vram", "auto"]) == 0
    out = capsys.readouterr().out
    assert "[audio] low_vram = true" in out
    assert "Crucible turned [audio] low_vram on" in out
    assert _document(home)["audio"] == {"low_vram": True, "low_vram_auto": True}
    assert load_config(home).capability.row("song").enabled is True


def test_on_by_hand_takes_crucibles_on_as_the_persons(
    home: Path, laptop: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["audio", "low-vram", "on"]) == 0
    capsys.readouterr()
    assert _document(home)["audio"] == {"low_vram": True}
    assert cli.main(["audio", "low-vram", "on"]) == 0
    assert "already on" in capsys.readouterr().out


def test_auto_on_a_big_card_is_off_and_writes_no_audio_table(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init"]) == 0
    _decided(capsys)
    assert cli.main(["audio", "low-vram", "on"]) == 0
    assert cli.main(["audio", "low-vram", "auto"]) == 0
    out = capsys.readouterr().out
    assert "[audio] low_vram = false" in out
    assert "this card does not need it" in out
    assert "audio" not in _document(home)
