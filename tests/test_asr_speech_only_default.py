from __future__ import annotations

import pytest
from pydantic import ValidationError

from crucible.jobs.asr import AsrParams


def params(**overrides: object) -> AsrParams:
    body: dict[str, object] = {"language": "en", "vad_filter": False, "word_timestamps": True}
    body.update(overrides)
    return AsrParams.model_validate(body)


def test_a_job_that_names_neither_detector_gets_speech_only() -> None:
    assert params().speech_only is True


def test_vad_filter_false_still_resolves_speech_only_on() -> None:
    assert params(vad_filter=False, context="names").speech_only is True


def test_vad_filter_true_alone_turns_speech_only_off() -> None:
    assert params(vad_filter=True).speech_only is False


def test_an_explicit_false_transcribes_everything() -> None:
    assert params(speech_only=False).speech_only is False


def test_both_detectors_asked_for_is_still_refused() -> None:
    with pytest.raises(ValidationError, match="two speech detectors"):
        params(vad_filter=True, speech_only=True)


def test_a_speech_knob_with_the_default_is_accepted() -> None:
    assert params(speech_threshold=0.6).speech_only is True


def test_a_speech_knob_with_an_explicit_false_is_refused() -> None:
    with pytest.raises(ValidationError, match="speech_only is false"):
        params(speech_only=False, speech_threshold=0.6)
