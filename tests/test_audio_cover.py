from __future__ import annotations

import base64
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from crucible.audiomodels import load_all_audio_manifests
from crucible.errors import ApiError, JobError
from crucible.jobs.audio import cover
from crucible.jobs.audio.params import AudioParams, refuse_what_the_model_cannot_take, settle

from .conftest import close_queue_session, open_queue_session
from .test_audio_api import (  # noqa: F401  (fixtures)
    SONG,
    idle_card,
    loads,
    ready,
    refusal,
    run_job,
    submit,
    transcript,
)

HERE = Path(__file__).resolve().parent
WORKER = HERE.parent / "crucible" / "jobs" / "audio" / "yue2_worker.py"
AUDIOCORE = WORKER.parent / "audiocore.py"
# A score YuE2 planned, with a sung Vocal melody and chords on it.
BALLAD = (HERE / "fixtures" / "yue2-score-ballad.abc").read_text(encoding="utf-8")
# A score YuE2 rendered as an instrumental: the melody already on the instrument.
HARBOR = (HERE / "fixtures" / "yue2-scores" / "harbor-1001.abc").read_text(encoding="utf-8")
# A score YuE2 wrote that the skill's strict parser refuses (a bar longer than its meter).
LANTERN = (HERE / "fixtures" / "yue2-scores" / "lantern-1000-refused.abc").read_text(encoding="utf-8")
TAGS = "Instrumental, slow somber piano ballad, strings, no vocals, 66 BPM"
SUNG_TAGS = "English, warm piano ballad, expressive female voice, 66 BPM"
WORDS = "[Verse]\nThe kettle sings the morning in\nThe window holds the rain\n\n[Chorus]\nStay a while\n"


def _tools() -> Any:
    sys.path.insert(0, str(WORKER.parent / "yue2music"))
    try:
        import abc_tools
    finally:
        sys.path.pop(0)
    return abc_tools


def _check(model: str, **params: Any) -> AudioParams:
    manifest = load_all_audio_manifests()[model]
    parsed = AudioParams(**params)
    refuse_what_the_model_cannot_take(parsed, manifest, manifest.spec("cuda-linux"))
    return parsed


# --- the request: what a cover takes and refuses before anything runs ---------------------


@pytest.mark.parametrize(
    ("params", "code", "param", "words"),
    [
        ({"instrumental": True, "planning_set": "harbor"}, "audio_param_conflict", "planning_set", "means nothing beside `score`"),
        ({"instrumental": True, "planning_lyrics": "[Verse]\nla la\n"}, "audio_param_conflict", "planning_lyrics", "nothing is planned"),
        ({"instrumental": True, "min_duration_s": 60}, "audio_param_conflict", "min_duration_s", "bars and tempo"),
        ({"lyrics": WORDS, "max_duration_s": 200}, "audio_param_conflict", "max_duration_s", "nothing is sized"),
        ({"instrumental": True, "lyrics": "[Verse]\n\n[Chorus]\n"}, "audio_param_conflict", "lyrics", "come from the score itself"),
        ({}, "audio_param_missing", None, "lyrics` written to the score's phrases"),
        ({"lyrics": "[Verse]\n\n[Chorus]\n"}, "audio_param_missing", None, "have only section tags"),
    ],
)
def test_a_cover_refuses_what_means_nothing_beside_a_score(
    params: dict, code: str, param: str | None, words: str
) -> None:
    with pytest.raises(ApiError) as caught:
        _check(SONG, tags=TAGS, score="score.abc", **params)
    assert caught.value.code == code, caught.value.message
    assert words in caught.value.message
    if param is not None:
        assert caught.value.details["param"] == param


def test_chords_without_a_score_and_a_score_on_stable_audio_are_refused() -> None:
    with pytest.raises(ApiError) as caught:
        _check(SONG, tags=TAGS, lyrics=WORDS, chords="strip")
    assert (caught.value.code, caught.value.details["param"]) == ("audio_param_conflict", "chords")
    with pytest.raises(ApiError) as caught:
        _check("stable-audio-3-medium", prompt="a sad piano piece", score="score.abc")
    assert caught.value.code == "audio_param_unsupported" and "'score'" in caught.value.message
    with pytest.raises(Exception, match="must not start with a dot"):
        AudioParams(tags=TAGS, score=".score.abc")


def test_an_instrumental_cover_is_not_given_a_pool_set() -> None:
    parsed = _check(SONG, tags=TAGS, instrumental=True, score="score.abc", chords="keep")
    settled = settle(parsed, load_all_audio_manifests()[SONG].spec("cuda-linux"), 7)
    assert settled.planning_lyrics is None and settled.length_range is None


# --- the score: checked and made ready on the server ------------------------------------


def test_an_instrumental_cover_moves_the_given_melody_to_the_instrument() -> None:
    made = cover.prepare(BALLAD, "score.abc", instrumental=True, lyrics=None, chords=None,
                         longest_s=360)
    tools = _tools()
    before, after = tools.parse_abc(BALLAD), tools.parse_abc(made.abc)
    assert after.voices["Vocal"].notes == [], "nothing left for a voice to sing"
    assert all(note in after.voices["Ins"].notes for note in before.voices["Vocal"].notes)
    assert (made.kind, made.cot, made.chords, made.chords_chosen_by) == ("instrumental", "full", "keep", "default")
    assert made.lyrics == "[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Outro]\n", "only the score's section tags"
    assert made.transfer["vocal_notes_before"] == len(before.voices["Vocal"].notes) > 0
    assert made.seconds == pytest.approx(made.transfer["nominal_seconds"])
    stripped = cover.prepare(BALLAD, "score.abc", instrumental=True, lyrics=None, chords="strip",
                             longest_s=360)
    assert (stripped.cot, stripped.chords_chosen_by) == ("melody", "request")
    assert tools.parse_abc(stripped.abc).voices["Vocal"].chords == []


def test_a_sung_cover_sings_the_given_melody_melody_only_by_default() -> None:
    made = cover.prepare(BALLAD, "score.abc", instrumental=False, lyrics=WORDS, chords=None,
                         longest_s=360)
    tools = _tools()
    assert (made.kind, made.cot, made.chords, made.lyrics) == ("sung", "melody", "strip", WORDS)
    assert tools.parse_abc(made.abc).voices["Vocal"].chords == []
    assert tools.parse_abc(made.abc).voices["Vocal"].notes == tools.parse_abc(BALLAD).voices["Vocal"].notes
    kept = cover.prepare(BALLAD, "score.abc", instrumental=False, lyrics=WORDS, chords="keep",
                         longest_s=360)
    assert (kept.abc, kept.cot) == (BALLAD, "full"), "the score exactly as given"


@pytest.mark.parametrize(
    ("text", "instrumental", "longest", "code", "words"),
    [
        (LANTERN, True, 360, "score_invalid", "note/rest exceeds meter duration"),
        (LANTERN, False, 360, "score_invalid", "note/rest exceeds meter duration"),
        ("X:1\nT:\nM:4/4\nK:C\nCDEF|\n", True, 360, "score_invalid", "Incomplete native two-voice ABC"),
        (BALLAD.replace("V: Ins\n", "w: la la la\nV: Ins\n", 1), False, 360, "score_invalid", "w: la la la"),
        (HARBOR, False, 360, "score_invalid", "no notes in its Vocal voice"),
        (HARBOR, True, 100, "score_too_long", "lasts 113.7 s"),
    ],
)
def test_a_score_the_skill_refuses_or_too_long_is_refused_by_name(
    text: str, instrumental: bool, longest: float, code: str, words: str
) -> None:
    with pytest.raises(JobError) as caught:
        cover.prepare(text, "score.abc", instrumental=instrumental, lyrics=WORDS, chords=None,
                      longest_s=longest)
    assert caught.value.code == code and words in caught.value.message, caught.value.message


# --- the worker: the given score is the plan, in one pass ---------------------------------


def _load_worker(monkeypatch: pytest.MonkeyPatch) -> Any:
    from crucible.jobs.audio import lorafold, planning, scorelength

    workerio = types.ModuleType("workerio")
    workerio.claim_stdout = lambda: None
    workerio.send = lambda *args, **kwargs: None
    # audiocore imports the shared encoder by name (crucible/jobs/workerio.py); these
    # tests stop before a song is saved, so it is never called.
    workerio.write_audio = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "workerio", workerio)
    spec = importlib.util.spec_from_file_location("audiocore_under_test", AUDIOCORE)
    audiocore = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audiocore)
    siblings = {"planning": planning, "scorelength": scorelength, "audiocore": audiocore,
                "lorafold": lorafold}
    workerio.load_sibling = lambda name, _file: siblings[name]
    spec = importlib.util.spec_from_file_location("yue2_worker_under_test", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    def __init__(self, tokens: int | None = None) -> None:
        self.tokens = tokens

    def encode(self, text: str) -> list[int]:
        return [1] * (len(text) // 3 if self.tokens is None else self.tokens)


class FakePipe:
    """YuE2Pipeline.plan given a score: the score back as the plan, nothing decoded."""

    def __init__(self, tokens: int | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.tokenizer = FakeTokenizer(tokens)
        self.generation_config = SimpleNamespace(abc=SimpleNamespace(max_tokens=4096))

    def plan(self, tags, lyrics, **kwargs):
        self.calls.append({"tags": tags, "lyrics": lyrics, **kwargs})
        assert "abc" in kwargs, "a cover never asks YuE2 to plan a score"
        return SimpleNamespace(abc=kwargs["abc"], truncated=False)


class Steps:
    def __init__(self) -> None:
        self.entered: list[str] = []

    def enter(self, stage: str, steps=None) -> None:
        self.entered.append(stage)


def _engine(module: Any, pipe: FakePipe) -> Any:
    engine = module.YuE2Engine.__new__(module.YuE2Engine)
    engine._pipe = pipe
    return engine


def test_the_worker_renders_an_instrumental_cover_unsung_from_the_given_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_worker(monkeypatch)
    made = cover.prepare(BALLAD, "score.abc", instrumental=True, lyrics=None, chords=None,
                         longest_s=360)
    engine = _engine(module, FakePipe())
    job = SimpleNamespace(tags=TAGS, seed=7, cfg=1.0, instrumental=True, cover=made.request())
    steps = Steps()
    plan = engine._given_plan(job, steps)

    (call,) = engine._pipe.calls
    assert (call["abc"], call["cot"], call["seed"], call["cfg_scale"]) == (made.abc, "full", 7, 1.0)
    assert call["lyrics"] == "[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Outro]\n", "only section tags"
    assert plan.abc == made.abc and steps.entered == ["scoring"]
    assert engine.score_tokens == len(made.abc) // 3
    assert engine.length["score_seconds"] == round(made.seconds, 2)
    assert (engine.length["sized"], engine.length["min_duration_s"]) == (False, None)


def test_the_worker_sings_a_sung_cover_with_the_lyrics_given(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_worker(monkeypatch)
    made = cover.prepare(BALLAD, "score.abc", instrumental=False, lyrics=WORDS, chords=None,
                         longest_s=360)
    engine = _engine(module, FakePipe())
    job = SimpleNamespace(tags=SUNG_TAGS, seed=11, cfg=1.0, instrumental=False, cover=made.request())
    engine._given_plan(job, Steps())
    (call,) = engine._pipe.calls
    assert (call["lyrics"], call["cot"], call["abc"]) == (WORDS, "melody", made.abc)
    assert call["tags"] == SUNG_TAGS


def test_a_score_past_the_token_budget_is_refused_before_anything_renders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_worker(monkeypatch)
    made = cover.prepare(BALLAD, "score.abc", instrumental=False, lyrics=WORDS, chords=None,
                         longest_s=360)
    engine = _engine(module, FakePipe(tokens=4097))
    job = SimpleNamespace(tags=SUNG_TAGS, seed=11, cfg=1.0, instrumental=False, cover=made.request())
    with pytest.raises(Exception) as raised:
        engine._given_plan(job, Steps())
    assert type(raised.value).__name__ == "Refused"
    assert raised.value.code == "score_too_many_tokens" and "4097 tokens" in raised.value.message
    assert engine._pipe.calls == [], "nothing was planned"


# --- through the API, with the fake worker --------------------------------------------


def _inline(text: str) -> dict:
    return {"inline_base64": base64.b64encode(text.encode("utf-8")).decode("ascii")}


def _artifact(client: TestClient, auth: dict[str, str], job_id: str, name: str) -> str:
    return client.get(f"/v1/jobs/{job_id}/artifacts/{name}", headers=auth).text


def _generates(transcript: Path) -> list[dict]:
    rows = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row.get("op") == "generate"]


def test_an_uploaded_score_is_covered_unsung_and_the_record_says_how(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    job_id, events = run_job(
        ready, auth, model=SONG,
        params={"tags": TAGS, "instrumental": True, "score": "score.abc", "seed": 5},
        inputs={"score.abc": _inline(BALLAD)},
    )
    assert events[-1]["event"] == "done", events[-1]
    done = events[-1]["data"]
    assert sorted(done["artifacts"]) == ["audio.flac", "given-score.abc", "score.abc"]
    audio = done["audio"]
    made = cover.prepare(BALLAD, "score.abc", instrumental=True, lyrics=None, chords=None,
                         longest_s=360)
    assert audio["cover"] == {
        "kind": "instrumental", "source": "upload", "input": "score.abc", "reference": None,
        "given": "given-score.abc", "cot": "full", "chords": "kept", "chords_chosen_by": "default",
        "score_tokens": 900, "score_seconds": round(made.seconds, 2), "score": made.reading,
    }
    assert audio["planning_lyrics"] is None and audio["lyrics"] is None
    assert audio["notes"]["instrumental_transfer"] == made.transfer
    assert audio["notes"]["score_before_transfer"] == BALLAD
    assert "scoring" not in audio["decode_stages"], "a given score decodes nothing"
    assert _artifact(ready, auth, job_id, "given-score.abc") == BALLAD
    assert _artifact(ready, auth, job_id, "score.abc") == made.abc
    (generate,) = _generates(transcript)
    assert generate["cover"] == made.request() and generate["planning_lyrics"] is None


def test_a_sung_cover_by_reference_to_a_held_score(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    """The chain B-Sides runs: one job's held score, sung with new words by the next."""
    first, events = run_job(
        ready, auth, model=SONG, hold=True,
        params={"tags": TAGS, "instrumental": True, "score": "score.abc", "chords": "strip"},
        inputs={"score.abc": _inline(BALLAD)},
    )
    assert events[-1]["event"] == "done", events[-1]
    second, events = run_job(
        ready, auth, model=SONG,
        params={"tags": SUNG_TAGS, "lyrics": WORDS, "score": "melody.abc"},
        inputs={"melody.abc": {"artifact": {"job_id": first, "name": "given-score.abc"}}},
    )
    assert events[-1]["event"] == "done", events[-1]
    audio = events[-1]["data"]["audio"]
    assert audio["cover"]["source"] == "reference"
    assert audio["cover"]["reference"] == {"job_id": first, "name": "given-score.abc"}
    assert (audio["cover"]["kind"], audio["cover"]["cot"], audio["cover"]["chords"]) == ("sung", "melody", "stripped")
    assert audio["lyrics"] == WORDS and audio["notes"] is None
    sung = _generates(transcript)[-1]["cover"]
    assert (sung["lyrics"], sung["cot"], sung["kind"]) == (WORDS, "melody", "sung")
    assert _artifact(ready, auth, second, "given-score.abc") == BALLAD


def test_a_score_refused_by_the_skill_fails_the_job_without_loading_yue2(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    _, events = run_job(
        ready, auth, model=SONG,
        params={"tags": TAGS, "instrumental": True, "score": "score.abc"},
        inputs={"score.abc": _inline(LANTERN)},
    )
    assert events[-1]["event"] == "failed"
    error = events[-1]["data"]["error"]
    assert error["code"] == "score_invalid" and "exceeds meter duration" in error["message"]
    assert loads(transcript) == []


def test_a_score_past_the_token_budget_fails_by_name_and_the_worker_stays(
    ready: TestClient, auth: dict[str, str], transcript: Path, monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The worker answers the refusal and stays loaded: in a queue session, which holds the
    model between songs, the next song is made without loading YuE2 again."""
    session_id = open_queue_session(ready, auth, act="song")
    counted = tmp_path / "score-tokens"
    counted.write_text("5000", encoding="utf-8")
    monkeypatch.setenv("CRUCIBLE_FAKE_AUDIO_SCORE_TOKENS", str(counted))
    body = dict(model=SONG, params={"tags": TAGS, "instrumental": True, "score": "score.abc"},
                inputs={"score.abc": _inline(BALLAD)})
    _, events = run_job(ready, auth, **body)
    assert events[-1]["event"] == "failed"
    error = events[-1]["data"]["error"]
    assert error["code"] == "score_too_many_tokens" and "5000 tokens" in error["message"]
    counted.write_text("1200", encoding="utf-8")
    _, events = run_job(ready, auth, **{**body, "inputs": {"score.abc": _inline(BALLAD)}})
    assert events[-1]["event"] == "done", events[-1]
    assert events[-1]["data"]["audio"]["cover"]["score_tokens"] == 1200
    assert len(loads(transcript)) == 1, "the refusal left YuE2 loaded"
    close_queue_session(ready, auth, session_id)


@pytest.mark.parametrize(
    ("params", "inputs", "code"),
    [
        ({"score": "score.abc"}, {}, "invalid_inputs"),
        ({"score": "score.abc"}, {"other.abc": None}, "invalid_inputs"),
        ({"score": "score.abc"}, {"score.abc": None, "extra.abc": None}, "invalid_inputs"),
    ],
)
def test_a_cover_carries_exactly_the_input_its_score_names(
    ready: TestClient, auth: dict[str, str], params: dict, inputs: dict, code: str
) -> None:
    sent = {name: _inline(BALLAD) for name in inputs}
    _, events = run_job(ready, auth, model=SONG,
                        params={"tags": TAGS, "instrumental": True, **params}, inputs=sent)
    assert events[-1]["event"] == "failed"
    assert events[-1]["data"]["error"]["code"] == code


def test_planning_params_beside_a_score_are_refused_at_the_door(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    error = refusal(submit(
        ready, auth, model=SONG,
        params={"tags": TAGS, "instrumental": True, "score": "score.abc", "planning_set": "harbor"},
        inputs={"score.abc": _inline(BALLAD)},
    ))
    assert (error["code"], error["details"]["param"]) == ("audio_param_conflict", "planning_set")
    assert loads(transcript) == []
