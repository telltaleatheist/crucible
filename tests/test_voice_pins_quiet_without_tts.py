from __future__ import annotations

import logging
from pathlib import Path

import pytest

from crucible import voicerepo
from crucible.voices import VoiceError


def _pins_refused(home: Path, monkeypatch: pytest.MonkeyPatch, jobs: str) -> list[str]:
    monkeypatch.setenv("CRUCIBLE_HOME", str(home))
    (home / "voices").mkdir(parents=True)
    (home / "voices" / "pins.toml").write_text(
        '[nightingale]\nhf_repo = "someone/nightingale"\nref = "crucible"\n',
        encoding="utf-8",
    )
    (home / "config.toml").write_text(jobs, encoding="utf-8")

    def unresolved(_pin: object) -> None:
        raise VoiceError("voice_ref_unresolved: nothing looked the tag up yet")

    monkeypatch.setattr(voicerepo, "voice_for_pin", unresolved)
    voicerepo._REFUSALS_SAID.clear()
    return list(voicerepo.load_pinned()[1])


def test_a_host_with_tts_off_is_not_warned_about_its_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="crucible.voicerepo"):
        refused = _pins_refused(tmp_path, monkeypatch, "[jobs]\nenable_tts = false\n")
    assert "nightingale" in refused, "the refusal is still there for whoever asks"
    assert not [r for r in caplog.records if "is pinned to" in r.getMessage()]


def test_a_host_that_serves_tts_is_still_warned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="crucible.voicerepo"):
        _pins_refused(tmp_path, monkeypatch, "[jobs]\nenable_tts = true\n")
    assert [r for r in caplog.records if "nightingale" in r.getMessage()]


def test_a_config_that_cannot_say_does_not_silence_the_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="crucible.voicerepo"):
        _pins_refused(tmp_path, monkeypatch, "this is not toml [")
    assert [r for r in caplog.records if "nightingale" in r.getMessage()]
